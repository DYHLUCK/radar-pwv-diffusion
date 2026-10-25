"""Restartable validation-only optimization; no test split is loaded here."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import copy
import fcntl
import gc
import random
import shutil
import time
from pathlib import Path
import numpy as np
import torch
from common import ROOT, DATA, RUNS, read, write, sha, now, csvwrite
from data import Store, batches, fixed_noise
from models import build, sample, EMA
from metrics import stats, concatenate, selection_values, select, rows, pooled
from normalization import get as normalization
from optimization_losses import loss

OLD = ROOT.parent / 'radar_pwv_generative_20261019_v2_normalized'
METHODS = ['deterministic', 'ddpm', 'resdiff', 'flow']
VARIANTS = ['continue', 'reconstruction', 'balanced']
SEEDS = [20261019, 20261020, 20261021]


def save(path, obj):
    path = Path(path)
    tmp = path.with_suffix('.tmp')
    torch.save(obj, tmp)
    os.replace(tmp, path)


def disk_check():
    free = shutil.disk_usage(ROOT).free
    used = sum(p.stat().st_size for p in ROOT.rglob('*')
               if p.is_file() and not p.is_symlink())
    # rglob does not traverse the shared data-directory symlink.
    rec = {'time': now(), 'free_gib': free / 2**30, 'new_experiment_gib': used / 2**30,
           'minimum_free_gib': 20, 'budget_gib': 4}
    write(ROOT / 'disk_status.json', rec)
    if free < 20 * 2**30 or used > 4 * 2**30:
        write(ROOT / 'state.json', {'phase': 'disk_budget_wait', **rec, 'pid': os.getpid()})
        raise RuntimeError('Disk guard: no old results deleted; checkpoint retained')


def configure(seed):
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def old_run(rnd, method, seed):
    candidates = list((OLD / 'runs' / rnd / 'formal' / method).glob(f'*/{seed}'))
    assert len(candidates) == 1, candidates
    return candidates[0]


def load_old(rnd, method, seed):
    rd = old_run(rnd, method, seed)
    summary = read(rd / 'summary.json')
    assert sha(rd / 'selected.pt') == summary['checkpoint_sha256']
    ck = torch.load(rd / 'selected.pt', map_location='cpu', weights_only=False)
    model = build(method).cuda()
    model.load_state_dict(ck['model'])
    return model, summary


def selection(v):
    result = selection_values(v)
    result['RMSE_dbz'] = float(np.sqrt(v['error'][..., 0].sum() / v['error'][..., 2].sum()))
    return result


@torch.no_grad()
def evaluate(model, store, starts, method, base=None, members=1):
    model.eval()
    all_members = []
    for member in range(members):
        parts = []
        for ss in batches(starts, 128):
            x, y = store.batch(ss)
            noise = fixed_noise(ss, member) if method != 'deterministic' else None
            with torch.autocast('cuda', dtype=torch.bfloat16):
                bp = base(x).float() if base is not None else None
                prediction = sample(model, x, noise, method, bp, 16)
            p = (prediction * 50).half().float()
            v = stats(p, y * 50)
            d = p - y * 50
            v['error'] = torch.stack([d.square().sum((2, 3)), d.abs().sum((2, 3)),
                                      torch.full_like(d[:, :, 0, 0], 66 * 70)], -1).cpu().numpy().astype(np.float64)
            parts.append(v)
        all_members.append(concatenate(parts))
    return all_members


def curve_rows(v, **meta):
    result = rows(v, **meta)
    e = v['error'].sum(0)
    for r in result:
        li = r['lead_minutes'] // 6 - 1
        r['RMSE_dbz'] = float(np.sqrt(e[li, 0] / e[li, 2]))
        r['MAE_dbz'] = float(e[li, 1] / e[li, 2])
    return result


def save_stat(path, v):
    np.savez_compressed(path, **v)


def read_stat(path):
    with np.load(path) as f:
        return {k: a for k, a in f.items()}


def get_baseline(store, rnd, method, seed, model, base):
    dest = ROOT / 'baseline_validation' / rnd / method / str(seed)
    dest.mkdir(parents=True, exist_ok=True)
    if not (dest / 'receipt.json').exists():
        v = evaluate(model, store, store.starts(rnd, 'valid'), method, base)[0]
        save_stat(dest / 'stats.npz', v)
        csvwrite(dest / 'curves.csv', curve_rows(v, round=rnd, method=method, seed=seed, variant='original'))
        write(dest / 'receipt.json', {'created_at': now(), 'selection': selection(v),
              'original_checkpoint_sha256': sha(old_run(rnd, method, seed) / 'selected.pt'),
              'stats_sha256': sha(dest / 'stats.npz'), 'split': 'valid'})
    rec = read(dest / 'receipt.json')
    assert sha(dest / 'stats.npz') == rec['stats_sha256']
    return rec['selection']


def run(store, rnd, method, variant, seed, epochs=16):
    rd = RUNS / rnd / method / variant / str(seed)
    rd.mkdir(parents=True, exist_ok=True)
    if (rd / 'summary.json').exists():
        rec = read(rd / 'summary.json')
        assert sha(rd / 'selected.pt') == rec['checkpoint_sha256']
        return rec
    disk_check()
    configure(seed)
    model, old_summary = load_old(rnd, method, seed)
    base = None
    if method == 'resdiff':
        base, _ = load_old(rnd, 'deterministic', seed)
        base.eval().requires_grad_(False)
        assert sha(old_run(rnd, 'deterministic', seed) / 'selected.pt') == old_summary['base_checkpoint_sha256']
    baseline = get_baseline(store, rnd, method, seed, model, base)
    config = {'round': rnd, 'method': method, 'variant': variant, 'seed': seed, 'epochs': epochs,
              'batch_size': 128, 'learning_rate': 1e-4 if method == 'deterministic' else 5e-5,
              'old_recipe': old_summary['recipe'], 'initialization': str(old_run(rnd, method, seed) / 'selected.pt'),
              'initialization_sha256': old_summary['checkpoint_sha256'],
              'dataset_id': store.manifest['dataset_id'], 'protocol_sha256': sha(ROOT / 'protocol.json'),
              'source_hashes': {p.name: sha(p) for p in sorted((ROOT / 'src').glob('*.py'))},
              'training_index_sha256': sha(DATA / f'{rnd}_train.npy'),
              'validation_index_sha256': sha(DATA / f'{rnd}_valid.npy'), 'validation_member': 0,
              'test_used': False, 'architecture_unchanged': True}
    if (rd / 'config.json').exists():
        assert read(rd / 'config.json') == config
    else:
        write(rd / 'config.json', config)
    ema = EMA(model)
    opt = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-6)
    history = []
    candidates = []
    first = 1
    if (rd / 'last.pt').exists():
        ck = torch.load(rd / 'last.pt', map_location='cpu', weights_only=False)
        assert ck['config'] == config
        model.load_state_dict(ck['model'])
        ema.model.load_state_dict(ck['ema'])
        ema.updates = ck['ema_updates']
        opt.load_state_dict(ck['optimizer'])
        sched.load_state_dict(ck['scheduler'])
        history = ck['history']; candidates = ck['candidates']; first = ck['epoch'] + 1
        torch.set_rng_state(ck['torch_rng']); torch.cuda.set_rng_state_all(ck['cuda_rng'])
        random.setstate(ck['python_rng']); np.random.set_state(ck['numpy_rng'])
    screen = store.starts(rnd, 'screen')
    scale = normalization(rnd)['std_normalized']
    for epoch in range(first, epochs + 1):
        disk_check()
        begin = time.perf_counter()
        model.train()
        total = 0.; count = 0
        for b, ss in enumerate(batches(store.starts(rnd, 'train'), 128, True, seed + epoch * 100003)):
            x, y = store.batch(ss)
            opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                with torch.no_grad():
                    bp = base(x).float() if base is not None else None
                value = loss(model, x, y, method, variant, config['old_recipe'], scale, bp)
            assert torch.isfinite(value), (rnd, method, variant, epoch, b)
            value.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            assert torch.isfinite(norm)
            opt.step(); ema.update(model)
            total += value.detach().item() * len(ss); count += len(ss)
            if b % 50 == 0:
                write(ROOT / 'state.json', {'phase': 'training', 'pid': os.getpid(), 'updated_at': now(),
                      'round': rnd, 'method': method, 'variant': variant, 'seed': seed,
                      'epoch': epoch, 'max_epochs': epochs, 'batch': b + 1,
                      'samples': count, 'run': str(rd)})
        assert count == len(store.starts(rnd, 'train'))
        sched.step()
        rec = {'epoch': epoch, 'train_loss': total / count, 'seconds': time.perf_counter() - begin}
        if epoch % 4 == 0:
            v = evaluate(ema.model, store, screen, method, base)[0]
            rec.update(selection(v))
            path = rd / f'candidate_{epoch:02}.pt'
            save(path, {'model': ema.model.state_dict(), 'epoch': epoch, 'config': config})
            candidates.append({**selection(v), 'epoch': epoch})
            keep = []
            remaining = candidates.copy()
            while remaining and len(keep) < 2:
                chosen = select(remaining); keep.append(chosen)
                remaining.remove(chosen)
            candidates = keep
        history.append(rec)
        save(rd / 'last.pt', {'model': model.state_dict(), 'ema': ema.model.state_dict(),
             'ema_updates': ema.updates, 'optimizer': opt.state_dict(), 'scheduler': sched.state_dict(),
             'config': config, 'history': history, 'candidates': candidates, 'epoch': epoch,
             'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all(),
             'python_rng': random.getstate(), 'numpy_rng': np.random.get_state()})
        # Commit the resumable candidate list BEFORE removing discarded files.
        # A crash at any point leaves the last committed checkpoint recoverable.
        for p in rd.glob('candidate_*.pt'):
            if int(p.stem.split('_')[-1]) not in [c['epoch'] for c in candidates]:
                p.unlink()
        write(rd / 'history.json', history)
        print('EPOCH', rnd, method, variant, seed, rec, flush=True)
    # Original weights are an explicit epoch-0 fallback, not hidden training success.
    full = [{'epoch': 0, **baseline}]
    for c in candidates:
        epoch = c['epoch']
        fp = rd / f'full_validation_{epoch:02}.npz'
        model.load_state_dict(torch.load(rd / f'candidate_{epoch:02}.pt', map_location='cpu', weights_only=False)['model'])
        if not fp.exists():
            write(ROOT / 'state.json', {'phase': 'full_validation', 'updated_at': now(),
                  'pid': os.getpid(), 'run': str(rd), 'epoch': epoch})
            save_stat(fp, evaluate(model, store, store.starts(rnd, 'valid'), method, base)[0])
        full.append({'epoch': epoch, **selection(read_stat(fp))})
    chosen = select(full)
    if chosen['epoch'] == 0:
        ck = torch.load(old_run(rnd, method, seed) / 'selected.pt', map_location='cpu', weights_only=False)
        v = read_stat(ROOT / 'baseline_validation' / rnd / method / str(seed) / 'stats.npz')
    else:
        ck = torch.load(rd / f"candidate_{chosen['epoch']:02}.pt", map_location='cpu', weights_only=False)
        v = read_stat(rd / f"full_validation_{chosen['epoch']:02}.npz")
    save(rd / 'selected.pt', {'model': ck['model'], 'config': config, 'epoch': chosen['epoch']})
    save_stat(rd / 'validation_stats.npz', v)
    csvwrite(rd / 'validation_curves.csv', curve_rows(v, round=rnd, method=method, variant=variant, seed=seed))
    delta = {k: chosen[k] - baseline[k] for k in baseline}
    eligible = (delta['selection_CSI'] > .003 and delta['selection_POD'] >= -.03
                and delta['selection_FAR'] <= .03 and chosen['RMSE_dbz'] <= 1.1 * baseline['RMSE_dbz'])
    rec = {**config, 'completed_at': now(), 'status': 'complete', 'selection': chosen,
           'baseline': baseline, 'delta_vs_original': delta, 'eligible_for_confirmation': eligible,
           'full_validation_candidates': full, 'checkpoint_sha256': sha(rd / 'selected.pt'),
           'selected_epoch': chosen['epoch'], 'parameters': sum(p.numel() for p in model.parameters())}
    write(rd / 'summary.json', rec)
    print('RUN_COMPLETE', rnd, method, variant, seed, chosen, 'eligible', eligible, flush=True)
    del model, base, ema, opt, sched
    gc.collect(); torch.cuda.empty_cache()
    return rec


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['pilot', 'all'], default='all')
    args = parser.parse_args()
    lock = (ROOT / 'study.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    freeze = ROOT / 'SOURCE_FREEZE.json'
    source = {p.name: sha(p) for p in sorted((ROOT / 'src').glob('*.py'))}
    if freeze.exists():
        assert read(freeze) == source
    else:
        write(freeze, source)
    configure(SEEDS[0])
    write(ROOT / 'process.json', {'pid': os.getpid(), 'started_at': now(), 'source_hashes': source})
    disk_check()
    store = Store()
    pilot = []
    for rnd in ['R1', 'R2']:
        for method in METHODS:
            for variant in VARIANTS:
                pilot.append(run(store, rnd, method, variant, SEEDS[0]))
    write(ROOT / 'pilot_results.json', pilot)
    decisions = []
    for rnd in ['R1', 'R2']:
        for method in METHODS:
            rr = [r for r in pilot if r['round'] == rnd and r['method'] == method]
            control = next(r for r in rr if r['variant'] == 'continue')
            choices = [r for r in rr if r['variant'] != 'continue' and r['eligible_for_confirmation']]
            best = max(choices, key=lambda r: r['selection']['selection_CSI']) if choices else None
            decisions.append({'round': rnd, 'method': method, 'variant': best['variant'] if best else None,
                'pilot_delta_vs_continue': best['selection']['selection_CSI'] - control['selection']['selection_CSI'] if best else None,
                'basis': 'full validation only; no test read', 'needs_confirmation': bool(best)})
    write(ROOT / 'confirmation_plan.json', decisions)
    if args.stage == 'all':
        for d in decisions:
            if d['needs_confirmation']:
                for seed in SEEDS[1:]:
                    for variant in ['continue', d['variant']]:
                        run(store, d['round'], d['method'], variant, seed)
    write(ROOT / 'state.json', {'phase': 'validation_training_complete', 'updated_at': now(),
          'pid': os.getpid(), 'test_used': False, 'next': 'multi-member validation, figures, interpretation and audit'})
    print('VALIDATION_TRAINING_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
