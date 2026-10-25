"""Portable per-run training/evaluation adapters around the unchanged experiment core.

The exact historical multi-run orchestration is archived in reference/.
This adapter never starts work on import and never selects checkpoints on test.
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
from contextlib import nullcontext
from pathlib import Path
import random
import shutil
import time
import numpy as np
import torch
from common import read, write, csvwrite, sha
from dataset import Store, batches, fixed_noise
from models import build, sample, EMA
from objectives import deterministic_loss, generative_loss
from optimization_losses import loss as optimization_loss
from sampler_objective import sampled_forecast_loss
from metrics import stats, rows, select


def configure(seed, device):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.set_num_threads(4)
    if torch.device(device).type == 'cuda':
        if not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable; pass --device cpu for inspection')
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False; torch.use_deterministic_algorithms(True)


def precision(device, enabled=True):
    return torch.autocast('cuda', dtype=torch.bfloat16) if torch.device(device).type == 'cuda' and enabled else nullcontext()


def load_model(path, method, device):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    config = checkpoint.get('config', {})
    declared = checkpoint.get('method', config.get('method'))
    if declared is not None and declared != method:
        raise ValueError(f'Checkpoint method {declared} != requested {method}')
    state = checkpoint.get('model', checkpoint.get('state_dict', checkpoint))
    model = build(method).to(device); model.load_state_dict(state, strict=True)
    model.eval()
    return model, checkpoint


def base_for(method, checkpoint, base_path, device, generator_path=None):
    if method != 'resdiff':
        if base_path: raise ValueError('--base is only used by residual diffusion')
        return None
    if not base_path: raise ValueError('Residual diffusion requires --base pointing to its matching deterministic checkpoint')
    expected = checkpoint.get('base_checkpoint_sha256', checkpoint.get('config', {}).get('base_checkpoint_sha256'))
    # Historical optimization checkpoints did not all serialize this field; map their exact hashes.
    reference = Path(__file__).resolve().parents[1] / 'provenance/historical_checkpoints.json'
    if expected is None and generator_path and reference.exists():
        identity = sha(generator_path)
        for entry in read(reference)['checkpoints']:
            if entry['checkpoint_sha256'] == identity: expected = entry.get('base_checkpoint_sha256')
    if expected is not None and sha(base_path) != expected:
        raise ValueError('Wrong residual base checkpoint; expected the base paired with these generator weights')
    base, _ = load_model(base_path, 'deterministic', device)
    return base.eval().requires_grad_(False)


def atomic_checkpoint(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp'); torch.save(value, tmp); os.replace(tmp, path)


def disk_guard(output, minimum_free_gib, budget_gib):
    p = Path(output); p.mkdir(parents=True, exist_ok=True)
    used = sum(x.stat().st_size for x in p.rglob('*') if x.is_file() and not x.is_symlink())
    if shutil.disk_usage(p).free < minimum_free_gib * 2**30 or used > budget_gib * 2**30:
        raise RuntimeError('Output disk guard reached; existing files are preserved')


@torch.no_grad()
def evaluate_model(model, base, store, indices, method, members=1, batch_size=128, steps=16, bf16=True):
    if len(indices) == 0 or members < 1: raise ValueError('Non-empty indices and positive member count required')
    model.eval(); records = []
    count = 1 if method == 'deterministic' else members
    for member in range(count):
        sums = None
        for ss in batches(indices, batch_size):
            x, y = store.batch(ss)
            n = fixed_noise(ss, member, store.device) if method != 'deterministic' else None
            with precision(store.device, bf16):
                bp = base(x).float() if base is not None else None
                p = (sample(model, x, n, method, bp, steps) * 50).half().float()
            v = stats(p, y * 50)
            error = p - y * 50
            v['error'] = torch.stack([error.square().sum((2, 3)), error.abs().sum((2, 3)),
                           torch.full_like(error[:, :, 0, 0], 4620)], -1).cpu().numpy().astype(np.float64)
            # Pool counts/squared fraction errors across samples, never average per-image CSI.
            reduced = {k: a.sum(0, keepdims=True) for k, a in v.items()}
            if sums is None: sums = reduced
            else:
                for k in sums: sums[k] += reduced[k]
        member_rows = rows(sums, member=member)
        e = sums['error'][0]
        for r in member_rows:
            i = r['lead_minutes'] // 6 - 1
            r['RMSE_dbz'] = float(np.sqrt(e[i, 0] / e[i, 2])); r['MAE_dbz'] = float(e[i, 1] / e[i, 2])
        records.extend(member_rows)
    # Score each member first, then average its skill. This is NOT skill of an averaged field.
    means = []
    keys = ['CSI', 'POD', 'FAR', 'FSS1', 'FSS3', 'FSS5', 'RMSE_dbz', 'MAE_dbz']
    for lead in range(6, 121, 6):
        for threshold in [20., 30., 40.]:
            group = [r for r in records if r['lead_minutes'] == lead and r['threshold_dbz'] == threshold]
            item = {'lead_minutes': lead, 'threshold_dbz': threshold, 'members': count}
            for k in keys:
                values = np.asarray([r[k] for r in group]); valid = values[np.isfinite(values)]
                item[k] = float(valid.mean()) if len(valid) else float('nan')
            means.append(item)
    return records, means


def selection(means):
    focus = [r for r in means if r['lead_minutes'] in [30, 60, 90, 120]]
    out = {f'selection_{k}': float(np.nanmean([r[k] for r in focus])) for k in ['CSI', 'POD', 'FSS3']}
    out['selection_FAR'] = float(np.mean([r['FAR'] if np.isfinite(r['FAR']) else 1. for r in focus]))
    # Metrics are threshold independent; average squared per-lead RMSE to recover pooled RMSE.
    out['RMSE_dbz'] = float(np.sqrt(np.mean([r['RMSE_dbz'] ** 2 for r in means if r['threshold_dbz'] == 30.])))
    return out


def protected(candidate, original):
    return (candidate['selection_POD'] >= original['selection_POD'] - .03
            and candidate['selection_FAR'] <= original['selection_FAR'] + .03
            and candidate['RMSE_dbz'] <= 1.1 * original['RMSE_dbz'])


def train(config, data_root, output, device, init=None, base_path=None, resume=False):
    cfg = dict(config); method = cfg['method']; stage = cfg['stage']; seed = cfg['seed']
    if stage not in ['native', 'continue', 'reconstruction', 'balanced', 'sampled_skill']:
        raise ValueError('Unknown training stage')
    if stage != 'native' and not init: raise ValueError('Fine-tuning requires --init pointing to the original selected checkpoint')
    if method == 'deterministic' and stage == 'sampled_skill': raise ValueError('Sampling loss requires a generator')
    output = Path(output)
    if output.is_symlink(): raise ValueError('Output directory must not be a symlink')
    if output.exists() and any(output.iterdir()) and not resume:
        raise FileExistsError('Output already contains files; choose a new path or pass --resume')
    if resume and not (output / 'last.pt').is_file(): raise FileNotFoundError('No last.pt in output')
    configure(seed, device); disk_guard(output, cfg['minimum_free_gib'], cfg['output_budget_gib'])
    store = Store(data_root, device); training = store.starts(cfg['round'], 'train')
    screening = store.starts(cfg['round'], 'screen'); valid = store.starts(cfg['round'], 'valid')
    if not np.isin(screening, valid).all(): raise ValueError('Screen indices must be a subset of validation')
    normal = store.normalization(cfg['round'])
    if init:
        model, checkpoint = load_model(init, method, device)
        if method != 'deterministic':
            if not np.isclose(model.target_mean.item(), normal['mean_normalized'], atol=1e-7) or not np.isclose(model.target_std.item(), normal['std_normalized'], atol=1e-7):
                raise ValueError('Warm-start target normalization does not match this training cache')
    else:
        model = build(method).to(device); checkpoint = {}
        if method != 'deterministic': model.set_normalization(normal['mean_normalized'], normal['std_normalized'])
    base = base_for(method, checkpoint, base_path, device, init)
    identity = {'configuration': cfg, 'dataset_id': store.manifest['dataset_id'],
                'device': str(torch.device(device)), 'torch_version': str(torch.__version__),
                'data_file_sha256': {n: sha(store.root / n) for n in ['radar_u8.npy', 'pwv_u8.npy', 'times.npy']},
                'train_index_sha256': sha(store.root / f"{cfg['round']}_train.npy"),
                'valid_index_sha256': sha(store.root / f"{cfg['round']}_valid.npy"),
                'screen_index_sha256': sha(store.root / f"{cfg['round']}_screen.npy"),
                'initial_checkpoint_sha256': sha(init) if init else None,
                'base_checkpoint_sha256': sha(base_path) if base_path else None,
                'source_sha256': {p.name: sha(p) for p in sorted(Path(__file__).parent.glob('*.py'))}}
    if resume:
        if read(output / 'run.json') != identity: raise ValueError('Resume configuration/data/source differs')
    else:
        write(output / 'run.json', identity); write(output / 'normalization.json', normal)
    ema = EMA(model); optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['epochs'], eta_min=cfg['eta_min'])
    generator = torch.Generator(device=device).manual_seed(seed + 1729)
    history = []; candidates = []; first = 1; steps_done = 0; baseline = None
    if resume:
        saved = torch.load(output / 'last.pt', map_location='cpu', weights_only=True)
        model.load_state_dict(saved['model']); ema.model.load_state_dict(saved['ema']); ema.updates = saved['ema_updates']
        optimizer.load_state_dict(saved['optimizer']); scheduler.load_state_dict(saved['scheduler'])
        history = saved['history']; candidates = saved['candidates']; first = saved['epoch'] + 1
        steps_done = saved['steps_done']; baseline = saved['baseline']
        torch.set_rng_state(saved['torch_rng']); generator.set_state(saved['sampling_rng'])
        if torch.device(device).type == 'cuda': torch.cuda.set_rng_state_all(saved['cuda_rng'])
    elif init:
        _, curves = evaluate_model(model, base, store, valid, method, 1, cfg['batch_size'], bf16=cfg['bf16'])
        baseline = {'epoch': 0, **selection(curves)}
    for epoch in range(first, cfg['epochs'] + 1):
        disk_guard(output, cfg['minimum_free_gib'], cfg['output_budget_gib'])
        start = time.perf_counter(); model.train(); total = 0.; seen = 0; auxiliary_samples = 0
        for ss in batches(training, cfg['batch_size'], True, seed + epoch * 100003):
            x, y = store.batch(ss); optimizer.zero_grad(set_to_none=True)
            with precision(device, cfg['bf16']):
                with torch.no_grad(): bp = base(x).float() if base is not None else None
                if stage in ['reconstruction', 'balanced']:
                    value = optimization_loss(model, x, y, method, stage, cfg['recipe'], normal['std_normalized'], bp)
                else:
                    value = deterministic_loss(model(x), y, cfg['recipe']) if method == 'deterministic' else generative_loss(model, x, y, method, cfg['recipe'], bp)
                    if stage == 'sampled_skill' and steps_done % 4 == 0:
                        n = min(16, len(x)); auxiliary_samples += n
                        value = value + .5 * sampled_forecast_loss(model, x[:n], y[:n], method,
                            normal['std_normalized'], bp[:n] if bp is not None else None, 4, generator)
            if not torch.isfinite(value): raise FloatingPointError('Non-finite training loss')
            value.backward(); norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            if not torch.isfinite(norm): raise FloatingPointError('Non-finite gradients')
            optimizer.step(); ema.update(model); total += value.detach().item() * len(ss); seen += len(ss); steps_done += 1
        scheduler.step()
        entry = {'epoch': epoch, 'loss': total / seen, 'samples': seen, 'auxiliary_samples': auxiliary_samples,
                 'seconds': time.perf_counter() - start, 'learning_rate_next': scheduler.get_last_lr()[0]}
        if epoch % cfg['validation_every'] == 0 or epoch == cfg['epochs']:
            _, curves = evaluate_model(ema.model, base, store, screening, method, 1, cfg['batch_size'], bf16=cfg['bf16'])
            record = {'epoch': epoch, **selection(curves)}; entry.update(record)
            atomic_checkpoint(output / f'candidate_{epoch:03d}.pt', {'model': ema.model.state_dict(), 'method': method,
                'epoch': epoch, 'config': cfg, 'base_checkpoint_sha256': identity['base_checkpoint_sha256']})
            pool = candidates + [record]; candidates = []
            while pool and len(candidates) < cfg['keep_candidates']:
                choice = select(pool); candidates.append(choice); pool.remove(choice)
        history.append(entry)
        atomic_checkpoint(output / 'last.pt', {'model': model.state_dict(), 'ema': ema.model.state_dict(), 'ema_updates': ema.updates,
            'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(), 'epoch': epoch, 'history': history,
            'candidates': candidates, 'baseline': baseline, 'steps_done': steps_done, 'torch_rng': torch.get_rng_state(),
            'cuda_rng': torch.cuda.get_rng_state_all() if torch.device(device).type == 'cuda' else [],
            'sampling_rng': generator.get_state()})
        # Prune only this run's intermediate candidates, after committing recovery state.
        keep = {c['epoch'] for c in candidates} | {epoch}
        for path in output.glob('candidate_*.pt'):
            if int(path.stem.split('_')[-1]) not in keep: path.unlink()
        write(output / 'history.json', history); print(entry, flush=True)
        focus = [h for h in history if 'selection_CSI' in h]; best = -float('inf'); improved = 0
        for h in focus:
            if h['selection_CSI'] > best + .001: best = h['selection_CSI']; improved = h['epoch']
        if cfg['patience'] and epoch >= cfg['minimum_epochs'] and epoch - improved >= cfg['patience']: break
    # Full validation re-ranking, with original weights as an explicit epoch-0 fallback.
    if not history: raise RuntimeError('No training history')
    final_epoch = history[-1]['epoch']; shortlist = list(candidates)
    if cfg['include_final'] and all(c['epoch'] != final_epoch for c in shortlist): shortlist.append({'epoch': final_epoch})
    full = [baseline] if baseline is not None else []
    for c in shortlist:
        epoch = c['epoch']; current, _ = load_model(output / f'candidate_{epoch:03d}.pt', method, device)
        _, curves = evaluate_model(current, base, store, valid, method, 1, cfg['batch_size'], bf16=cfg['bf16'])
        full.append({'epoch': epoch, **selection(curves)})
    eligible = [c for c in full if baseline is None or c['epoch'] == 0 or protected(c, baseline)]
    chosen = select(eligible); epoch = chosen['epoch']
    selected_path = Path(init) if epoch == 0 else output / f'candidate_{epoch:03d}.pt'
    selected, _ = load_model(selected_path, method, device)
    atomic_checkpoint(output / 'selected.pt', {'model': selected.state_dict(), 'method': method, 'epoch': epoch,
        'config': cfg, 'base_checkpoint_sha256': identity['base_checkpoint_sha256'], 'normalization': normal})
    _, curves = evaluate_model(selected, base, store, valid, method, 1, cfg['batch_size'], bf16=cfg['bf16'])
    csvwrite(output / 'validation_curves.csv', curves)
    result = {'method': method, 'trained_epochs': final_epoch, 'selected_epoch': epoch, 'selection': chosen,
              'full_validation_candidates': full, 'test_used_for_selection': False,
              'checkpoint_sha256': sha(output / 'selected.pt'), 'base_checkpoint_sha256': identity['base_checkpoint_sha256']}
    write(output / 'summary.json', result); return result
