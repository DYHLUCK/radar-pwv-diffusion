"""Finite sampler-aware training study. All model classes/inference stay unchanged."""
from pathlib import Path
import os
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import argparse
import fcntl
import gc
import random
import time
import numpy as np
import torch
from common import ROOT, DATA, RUNS, read, write, sha, now, csvwrite
from data import Store, batches
from models import EMA
from metrics import select
from normalization import get as normalization
from objectives import generative_loss
from optimize import (configure, load_old, old_run, get_baseline, evaluate,
                      selection, curve_rows, disk_check, save, read_stat, save_stat)
from sampler_objective import sampled_forecast_loss

TUNE = ROOT / 'sampler_tuning'
SEEDS = [20261019, 20261020, 20261021]
METHODS = ['ddpm', 'resdiff', 'flow']


def source_fingerprint():
    return {'training_source_freeze_sha256': sha(ROOT / 'SOURCE_FREEZE.json'),
            'extension_sources': {name: sha(TUNE / name) for name in
                    ['sampler_objective.py', 'train_sampler.py', 'check_sampler.py', 'check_training.py']}}


def rng_state(generator):
    return {'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all(),
            'python_rng': random.getstate(), 'numpy_rng': np.random.get_state(),
            'sampling_rng': generator.get_state()}


def restore_rng(state, generator):
    torch.set_rng_state(state['torch_rng']); torch.cuda.set_rng_state_all(state['cuda_rng'])
    random.setstate(state['python_rng']); np.random.set_state(state['numpy_rng'])
    generator.set_state(state['sampling_rng'])


def validation_labels(store, rnd):
    times = store.times[store.starts(rnd, 'valid') + 9]
    blocks = ((times - times.min().astype('datetime64[D]')).astype('timedelta64[h]').astype(int) // 72)
    return np.unique(blocks, return_inverse=True)[1]


def aggregate_blocks(values, labels):
    """Exact pooled sufficient statistics; no full per-issue output stored."""
    assert all(a.shape[0] == len(labels) for a in values.values())
    return {k: np.stack([a[labels == i].sum(0) for i in range(labels.max() + 1)])
            for k, a in values.items()}


def step(model, base, optimizer, ema, x, y, method, recipe, scale, variant, steps_done,
         generator, coefficient=.5, frequency=4, sample_batch=16):
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        with torch.no_grad():
            bp = base(x).float() if base is not None else None
        native = generative_loss(model, x, y, method, recipe, bp)
        auxiliary = None
        if variant == 'sampled_skill' and steps_done % frequency == 0:
            count = min(sample_batch, len(x))
            auxiliary = sampled_forecast_loss(model, x[:count], y[:count], method, scale,
                        bp[:count] if bp is not None else None, 4, generator)
        total = native if auxiliary is None else native + coefficient * auxiliary
    assert torch.isfinite(total)
    total.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
    assert torch.isfinite(norm)
    optimizer.step(); ema.update(model)
    return {'total': total.detach().item(), 'native': native.detach().item(),
            'sampled_loss': auxiliary.detach().item() if auxiliary is not None else None,
            'gradient_norm': norm.item(), 'sampled_examples': min(sample_batch, len(x)) if auxiliary is not None else 0,
            'additional_sampling_forward_examples': 16 * min(sample_batch, len(x)) if auxiliary is not None else 0}


def run(store, rnd, method, variant, seed):
    assert variant in ['control', 'sampled_skill']
    rd = TUNE / 'runs' / rnd / method / variant / str(seed)
    rd.mkdir(parents=True, exist_ok=True)
    if (rd / 'summary.json').exists():
        rec = read(rd / 'summary.json')
        assert sha(rd / 'selected.pt') == rec['checkpoint_sha256']
        return rec
    disk_check(); configure(seed)
    model, previous = load_old(rnd, method, seed)
    base = None
    if method == 'resdiff':
        base, _ = load_old(rnd, 'deterministic', seed)
        base.eval().requires_grad_(False)
        assert sha(old_run(rnd, 'deterministic', seed) / 'selected.pt') == previous['base_checkpoint_sha256']
    baseline = get_baseline(store, rnd, method, seed, model, base)
    labels = validation_labels(store, rnd)
    config = {'round': rnd, 'method': method, 'variant': variant, 'seed': seed, 'epochs': 16,
              'batch_size': 128, 'learning_rate': 5e-5, 'weight_decay': .0001,
              'auxiliary_coefficient': .5, 'sampling_frequency_batches': 4,
              'sampling_examples_per_selected_batch': 16, 'gradient_steps': 4,
              'sampling_NFE': 16, 'old_recipe': previous['recipe'],
              'initialization': str(old_run(rnd, method, seed) / 'selected.pt'),
              'initialization_sha256': previous['checkpoint_sha256'],
              'dataset_id': store.manifest['dataset_id'], 'source_fingerprint': source_fingerprint(),
              'protocol_sha256': sha(TUNE / 'protocol.json'),
              'training_index_sha256': sha(DATA / f'{rnd}_train.npy'),
              'validation_index_sha256': sha(DATA / f'{rnd}_valid.npy'),
              'validation_windows': len(labels), 'validation_blocks_72h': int(labels.max() + 1),
              'saved_statistics': 'per-72h-block sums; pooled metrics preserved; no full fields',
              'architecture_and_inference_unchanged': True, 'test_used': False}
    if (rd / 'config.json').exists():
        assert read(rd / 'config.json') == config
    else:
        write(rd / 'config.json', config)
    ema = EMA(model)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=.0001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=16, eta_min=1e-6)
    generator = torch.Generator(device='cuda').manual_seed(seed + 1729)
    first = 1; history = []; candidates = []; steps_done = 0; sample_examples = 0
    if (rd / 'last.pt').exists():
        ck = torch.load(rd / 'last.pt', map_location='cpu', weights_only=False)
        assert ck['config'] == config
        model.load_state_dict(ck['model']); ema.model.load_state_dict(ck['ema'])
        ema.updates = ck['ema_updates']; opt.load_state_dict(ck['optimizer'])
        scheduler.load_state_dict(ck['scheduler']); restore_rng(ck, generator)
        first = ck['epoch'] + 1; history = ck['history']; candidates = ck['candidates']
        steps_done = ck['steps_done']; sample_examples = ck['sample_examples']
    scale = normalization(rnd)['std_normalized']
    for epoch in range(first, 17):
        disk_check(); model.train(); start = time.perf_counter()
        total = 0.; native_total = 0.; seen = 0; epoch_aux = []
        for bi, ss in enumerate(batches(store.starts(rnd, 'train'), 128, True, seed + epoch * 100003)):
            x, y = store.batch(ss)
            result = step(model, base, opt, ema, x, y, method, previous['recipe'], scale,
                          variant, steps_done, generator)
            steps_done += 1; seen += len(ss)
            total += result['total'] * len(ss); native_total += result['native'] * len(ss)
            sample_examples += result['sampled_examples']
            if result['sampled_loss'] is not None: epoch_aux.append(result['sampled_loss'])
            if bi % 50 == 0:
                write(TUNE / 'state.json', {'phase': 'training', 'pid': os.getpid(), 'updated_at': now(),
                      'run': str(rd), 'round': rnd, 'method': method, 'variant': variant,
                      'seed': seed, 'epoch': epoch, 'batch': bi + 1, 'samples_seen': seen})
        assert seen == len(store.starts(rnd, 'train'))
        scheduler.step()
        rec = {'epoch': epoch, 'loss': total / seen, 'native_loss': native_total / seen,
               'mean_sampled_loss_on_aux_batches': float(np.mean(epoch_aux)) if epoch_aux else None,
               'samples': seen, 'steps_done': steps_done, 'sampled_examples_cumulative': sample_examples,
               'seconds': time.perf_counter() - start}
        if epoch % 4 == 0:
            rec.update(selection(evaluate(ema.model, store, store.starts(rnd, 'screen'), method, base)[0]))
            save(rd / f'candidate_{epoch:02}.pt', {'model': ema.model.state_dict(), 'epoch': epoch, 'config': config})
            candidates.append({'epoch': epoch, **{k: rec[k] for k in
                    ['selection_CSI', 'selection_POD', 'selection_FAR', 'selection_FSS3', 'RMSE_dbz']}})
            remaining = candidates.copy(); keep = []
            while remaining and len(keep) < 2:
                chosen = select(remaining); keep.append(chosen); remaining.remove(chosen)
            candidates = keep
        history.append(rec)
        save(rd / 'last.pt', {'config': config, 'model': model.state_dict(), 'ema': ema.model.state_dict(),
             'ema_updates': ema.updates, 'optimizer': opt.state_dict(), 'scheduler': scheduler.state_dict(),
             'epoch': epoch, 'history': history, 'candidates': candidates, 'steps_done': steps_done,
             'sample_examples': sample_examples, **rng_state(generator)})
        for fp in rd.glob('candidate_*.pt'):
            if int(fp.stem.split('_')[-1]) not in [c['epoch'] for c in candidates]: fp.unlink()
        write(rd / 'history.json', history)
        print('SAMPLER_EPOCH', rnd, method, variant, seed, rec, flush=True)
    full = [{'epoch': 0, **baseline}]
    for c in candidates:
        epoch = c['epoch']; fp = rd / f'full_validation_{epoch:02}.npz'
        model.load_state_dict(torch.load(rd / f'candidate_{epoch:02}.pt', map_location='cpu', weights_only=False)['model'])
        if not fp.exists():
            write(TUNE / 'state.json', {'phase': 'full_validation', 'run': str(rd), 'epoch': epoch,
                                      'pid': os.getpid(), 'updated_at': now()})
            save_stat(fp, aggregate_blocks(evaluate(model, store, store.starts(rnd, 'valid'), method, base)[0], labels))
        full.append({'epoch': epoch, **selection(read_stat(fp))})
    chosen = select(full)
    if chosen['epoch'] == 0:
        ck = torch.load(old_run(rnd, method, seed) / 'selected.pt', map_location='cpu', weights_only=False)
        v = aggregate_blocks(read_stat(ROOT / 'baseline_validation' / rnd / method / str(seed) / 'stats.npz'), labels)
    else:
        ck = torch.load(rd / f"candidate_{chosen['epoch']:02}.pt", map_location='cpu', weights_only=False)
        v = read_stat(rd / f"full_validation_{chosen['epoch']:02}.npz")
    save(rd / 'selected.pt', {'model': ck['model'], 'epoch': chosen['epoch'], 'config': config})
    save_stat(rd / 'validation_stats.npz', v)
    csvwrite(rd / 'validation_curves.csv', curve_rows(v, round=rnd, method=method, variant=variant, seed=seed))
    delta = {k: chosen[k] - baseline[k] for k in baseline}
    eligible = (delta['selection_CSI'] > .003 and delta['selection_POD'] >= -.03
                and delta['selection_FAR'] <= .03 and chosen['RMSE_dbz'] <= baseline['RMSE_dbz'] * 1.1)
    rec = {**config, 'run_dir': str(rd), 'status': 'complete', 'completed_at': now(),
           'selection': chosen, 'baseline': baseline, 'delta_vs_original': delta,
           'eligible_for_confirmation': eligible, 'selected_epoch': chosen['epoch'],
           'full_validation_candidates': full, 'checkpoint_sha256': sha(rd / 'selected.pt'),
           'sampling_forward_examples': sample_examples * 16,
           'native_forward_examples': len(store.starts(rnd, 'train')) * 16,
           'parameters': sum(p.numel() for p in model.parameters())}
    write(rd / 'summary.json', rec)
    print('SAMPLER_RUN_COMPLETE', rnd, method, variant, seed, chosen, 'eligible', eligible, flush=True)
    del model, base, ema, opt, scheduler
    gc.collect(); torch.cuda.empty_cache()
    return rec


def control(store, rnd, method, seed):
    rd = RUNS / rnd / method / 'continue' / str(seed)
    if not (rd / 'summary.json').exists():
        return run(store, rnd, method, 'control', seed)
    r = read(rd / 'summary.json')
    assert r['epochs'] == 16 and r['batch_size'] == 128 and r['learning_rate'] == 5e-5
    assert r['source_hashes'] == read(ROOT / 'SOURCE_FREEZE.json')
    assert r['initialization_sha256'] == sha(old_run(rnd, method, seed) / 'selected.pt')
    assert sha(rd / 'selected.pt') == r['checkpoint_sha256']
    rec = {**r, 'run_dir': str(rd), 'variant': 'control', 'reused_phase1_continue': True,
           'reuse_basis': 'same initialization, native loss, 16epochs, batch/order, LR/scheduler/EMA and inference'}
    write(TUNE / 'control_references' / rnd / method / f'{seed}.json', rec)
    return rec


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['pilot', 'all'], default='all')
    args = parser.parse_args()
    completed = read(ROOT / 'analysis_complete.json')
    assert completed['mean_curves_sha256'] == sha(ROOT / 'validation_mean_curves.csv')
    assert read(ROOT / 'checks/results_audit.json')['passed']
    assert all(sha(ROOT / 'src' / name) == h for name, h in read(ROOT / 'SOURCE_FREEZE.json').items())
    assert read(TUNE / 'checks.json')['passed']
    assert read(TUNE / 'checks.json')['source_sha256'] == sha(TUNE / 'sampler_objective.py')
    assert read(TUNE / 'training_checks.json')['passed']
    assert read(TUNE / 'training_checks.json')['fingerprint'] == source_fingerprint()
    freeze = TUNE / 'SOURCE_FREEZE.json'
    if freeze.exists(): assert read(freeze) == source_fingerprint()
    else: write(freeze, source_fingerprint())
    lock = (TUNE / 'study.lock').open('w'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    configure(SEEDS[0]); disk_check()
    write(TUNE / 'process.json', {'pid': os.getpid(), 'started_at': now(), 'source_fingerprint': source_fingerprint()})
    store = Store(); pilot = []
    for rnd in ['R1', 'R2']:
        for method in METHODS:
            c = control(store, rnd, method, SEEDS[0])
            candidate = run(store, rnd, method, 'sampled_skill', SEEDS[0])
            pilot.append({'round': rnd, 'method': method, 'control': c, 'candidate': candidate,
                'delta_CSI_vs_continue': candidate['selection']['selection_CSI'] - c['selection']['selection_CSI']})
    write(TUNE / 'pilot_results.json', pilot)
    if args.stage == 'all':
        for r in pilot:
            if r['candidate']['eligible_for_confirmation']:
                for seed in SEEDS[1:]:
                    control(store, r['round'], r['method'], seed)
                    run(store, r['round'], r['method'], 'sampled_skill', seed)
    write(TUNE / 'state.json', {'phase': 'training_complete', 'updated_at': now(), 'test_used': False,
          'next': 'multi-member validation, figures and comparison with phase1 best/control/original'})


if __name__ == '__main__':
    main()
