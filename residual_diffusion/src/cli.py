"""Command-line interface for one standalone model repository."""
from pathlib import Path
import argparse
import numpy as np
import torch
from common import read, write, csvwrite, sha
from dataset import Store, prepare, fixed_noise
from models import sample
from runtime import configure, load_model, base_for, precision, train, evaluate_model


def cli(method):
    parser = argparse.ArgumentParser(description=f'RADAR+PWV -> future RADAR ({method})')
    commands = parser.add_subparsers(dest='command', required=True)
    p = commands.add_parser('prepare', help='Build cache from a paired PNG manifest')
    p.add_argument('--frames-csv', required=True); p.add_argument('--output', required=True)
    for name in ['train', 'train-base'] if method == 'resdiff' else ['train']:
        p = commands.add_parser(name, help='Explicit training command; never starts on import')
        p.add_argument('--config', required=True); p.add_argument('--data-root', required=True); p.add_argument('--output', required=True)
        p.add_argument('--init'); p.add_argument('--base'); p.add_argument('--resume', action='store_true'); p.add_argument('--device', default='cuda')
    p = commands.add_parser('evaluate', help='All20 leads; no checkpoint selection here')
    p.add_argument('--checkpoint', required=True); p.add_argument('--data-root', required=True); p.add_argument('--round', choices=['R1', 'R2'], required=True)
    p.add_argument('--split', choices=['valid', 'test'], default='valid'); p.add_argument('--output', required=True)
    p.add_argument('--members', type=int, default=4); p.add_argument('--batch-size', type=int, default=128); p.add_argument('--device', default='cuda'); p.add_argument('--base')
    p = commands.add_parser('predict', help='Predict from one physical-unit radar/PWV NPZ')
    p.add_argument('--checkpoint', required=True); p.add_argument('--input', required=True); p.add_argument('--output', required=True)
    p.add_argument('--members', type=int, default=1); p.add_argument('--issue-id', type=int, default=0)
    p.add_argument('--device', default='cuda'); p.add_argument('--base')
    p = commands.add_parser('plot', help='Truth/forecast panels at30/60/90/120min')
    p.add_argument('--prediction', required=True); p.add_argument('--output', required=True)
    p.add_argument('--truth', help='Optional NPZ containing truth_dbz with shape(20,66,70)'); p.add_argument('--member', type=int, default=0)
    args = parser.parse_args()
    if args.command == 'prepare':
        print(prepare(args.frames_csv, args.output)); return
    if args.command in ['train', 'train-base']:
        config = read(args.config); expected = 'deterministic' if args.command == 'train-base' else method
        if config['method'] != expected: parser.error('Configuration does not belong to this model/command')
        print(train(config, args.data_root, args.output, args.device, args.init, args.base, args.resume)); return
    if args.command == 'plot':
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        with np.load(args.prediction, allow_pickle=False) as z: pred = z['prediction_dbz'][args.member]
        truth = None
        if args.truth:
            with np.load(args.truth, allow_pickle=False) as z: truth = z['truth_dbz']
            if truth.shape != (20, 66, 70): raise ValueError('truth_dbz must have shape(20,66,70)')
        arrays = [truth, pred] if truth is not None else [pred]
        labels = ['Truth', 'Forecast'] if truth is not None else ['Forecast']
        fig, axes = plt.subplots(len(arrays), 4, figsize=(10, 2.8 * len(arrays)), squeeze=False, constrained_layout=True)
        for i, arr in enumerate(arrays):
            for j, lead in enumerate([30, 60, 90, 120]):
                ax = axes[i, j]; im = ax.imshow(arr[lead // 6 - 1], vmin=0, vmax=50, cmap='turbo', interpolation='nearest')
                ax.set_title(f'T+{lead} min'); ax.set_xticks([]); ax.set_yticks([])
                if j == 0: ax.set_ylabel(labels[i])
        fig.colorbar(im, ax=axes.ravel().tolist(), label='dBZ', shrink=.8)
        out = Path(args.output)
        if out.exists(): raise FileExistsError(out)
        out.parent.mkdir(parents=True, exist_ok=True); fig.savefig(out, dpi=160); plt.close(fig); return
    configure(20261019, args.device)
    model, ck = load_model(args.checkpoint, method, args.device)
    base = base_for(method, ck, args.base, args.device, args.checkpoint)
    if args.command == 'evaluate':
        output = Path(args.output)
        if output.exists() and any(output.iterdir()): raise FileExistsError('Evaluation output must be new or empty')
        output.mkdir(parents=True, exist_ok=True)
        store = Store(args.data_root, args.device); ids = store.starts(args.round, args.split)
        identity = {'method': method, 'round': args.round, 'split': args.split, 'checkpoint_sha256': sha(args.checkpoint),
            'base_checkpoint_sha256': sha(args.base) if args.base else None, 'members': args.members, 'sampler_steps': 16,
            'index_sha256': sha(store.root / f'{args.round}_{args.split}.npy'), 'dataset_id': store.manifest['dataset_id'],
            'checkpoint_selection_performed': False, 'historical_test_may_have_been_seen': True}
        # A fixed input/weight receipt is written BEFORE any test predictions are scored.
        write(output / 'evaluation_lock.json', identity)
        records, means = evaluate_model(model, base, store, ids, method, args.members, args.batch_size)
        csvwrite(output / 'member_curves.csv', records); csvwrite(output / 'mean_member_skill.csv', means)
        write(output / 'complete.json', {**identity, 'windows': len(ids), 'files':
            {n: sha(output / n) for n in ['member_curves.csv', 'mean_member_skill.csv']}})
        print(f'Saved all20 lead curves for {len(ids)} windows in {output}'); return
    if args.command == 'predict':
        if args.members < 1 or args.issue_id < 0: raise ValueError('members positive; issue-id nonnegative')
        output = Path(args.output)
        if output.exists(): raise FileExistsError(output)
        with np.load(args.input, allow_pickle=False) as z:
            radar = z['radar_dbz'].astype(np.float32); pwv = z['pwv_mm'].astype(np.float32)
        if radar.shape != (10, 66, 70) or pwv.shape != radar.shape: raise ValueError('Input arrays must have shape(10,66,70)')
        if not np.isfinite(radar).all() or not np.isfinite(pwv).all() or radar.min() < 0 or radar.max() > 50 or pwv.min() < 0 or pwv.max() > 80:
            raise ValueError('Expected physical ranges0–50dBZ and0–80mm; no silent clipping')
        x = torch.from_numpy(np.concatenate([radar / 50, pwv / 80])[None]).to(args.device)
        outputs = []
        with torch.no_grad():
            for member in range(args.members):
                with precision(args.device):
                    bp = base(x).float() if base is not None else None
                    p = sample(model, x, fixed_noise([args.issue_id], member, args.device), method, bp, 16)
                outputs.append((p[0] * 50).half().cpu().numpy())
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open('wb') as stream: np.savez_compressed(stream, prediction_dbz=np.stack(outputs), lead_minutes=np.arange(6, 121, 6))
        write(output.with_suffix('.json'), {'method': method, 'checkpoint_sha256': sha(args.checkpoint),
              'base_checkpoint_sha256': sha(args.base) if args.base else None, 'issue_id': args.issue_id,
              'members': args.members, 'shape': list(np.stack(outputs).shape), 'units': 'dBZ', 'sampler_steps': 16,
              'sampling_seed': 819173, 'member_selection': 'all requested members retained'})
        print(output)
