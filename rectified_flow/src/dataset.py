"""Memory-mapped decoded uint8 caches and causal, split-contained windows."""
from pathlib import Path
import csv
import hashlib
import numpy as np
import torch
from PIL import Image
from common import read, write, sha

PERIODS = {
    'R1': {'train': ('2023-05-01', '2023-09-01'), 'valid': ('2025-05-01', '2025-07-01'), 'test': ('2025-07-01', '2025-09-01')},
    'R2': {'train': ('2025-05-01', '2025-09-01'), 'valid': ('2023-05-01', '2023-07-01'), 'test': ('2023-07-01', '2023-09-01')},
}

def fixed_noise(starts, member, device='cpu', sampling_seed=819173):
    # Identical sample-index/member noise identity to the audited experiment.
    noise = np.stack([np.random.default_rng(np.random.SeedSequence([sampling_seed, int(i), int(member)]))
                      .standard_normal((20, 66, 70), dtype=np.float32) for i in starts])
    return torch.from_numpy(noise).to(device)

def batches(starts, batch_size, shuffle=False, seed=0):
    if batch_size < 1: raise ValueError('batch_size must be positive')
    order = np.random.default_rng(seed).permutation(len(starts)) if shuffle else np.arange(len(starts))
    for i in range(0, len(order), batch_size): yield np.asarray(starts)[order[i:i + batch_size]]

class Store:
    def __init__(self, root, device='cpu'):
        self.root = Path(root); self.device = torch.device(device)
        self.manifest = read(self.root / 'manifest.json')
        self.radar = np.load(self.root / 'radar_u8.npy', mmap_mode='r', allow_pickle=False)
        self.pwv = np.load(self.root / 'pwv_u8.npy', mmap_mode='r', allow_pickle=False)
        self.times = np.load(self.root / 'times.npy', allow_pickle=False)
        if self.radar.shape != self.pwv.shape or self.radar.shape[1:] != (66, 70):
            raise ValueError('Expected radar/PWV shapes (N,66,70)')
        if self.radar.dtype != np.uint8 or self.pwv.dtype != np.uint8:
            raise ValueError('Caches must contain decoded uint8: zero=0, 255=physical upper bound')
        if self.times.shape != (len(self.radar),) or not np.issubdtype(self.times.dtype, np.datetime64):
            raise ValueError('times.npy must contain N datetime64 timestamps')
        if len(self.times) > 1 and not np.all(np.diff(self.times) > np.timedelta64(0, 'm')):
            raise ValueError('Timestamps must be unique and increasing')

    def starts(self, round_id, split):
        if round_id not in PERIODS or split not in ['train', 'screen', 'valid', 'test']:
            raise ValueError('Unknown round/split')
        ids = np.load(self.root / f'{round_id}_{split}.npy', allow_pickle=False)
        if ids.ndim != 1 or not np.issubdtype(ids.dtype, np.integer) or not len(ids):
            raise ValueError(f'{round_id}_{split}: non-empty integer start indices required')
        if len(np.unique(ids)) != len(ids) or np.any(ids < 0) or np.any(ids + 29 >= len(self.times)):
            raise ValueError('Invalid or duplicated start indices')
        lo, hi = map(np.datetime64, PERIODS[round_id]['valid' if split == 'screen' else split])
        for block in batches(ids, 2048):
            t = self.times[block[:, None] + np.arange(30)]
            if not np.all(np.diff(t, axis=1) == np.timedelta64(6, 'm')):
                raise ValueError('Every window must have 30 consecutive six-minute frames')
            if not np.all((t >= lo) & (t < hi)):
                raise ValueError('Window crosses the configured split boundary')
        return ids.astype(np.int64, copy=False)

    def batch(self, starts):
        indices = np.asarray(starts, dtype=np.int64)[:, None]
        p = indices + np.arange(10); t = indices + np.arange(10, 30)
        radar = torch.from_numpy(self.radar[p].astype(np.float32) / 255.).to(self.device)
        pwv = torch.from_numpy(self.pwv[p].astype(np.float32) / 255.).to(self.device)
        target = torch.from_numpy(self.radar[t].astype(np.float32) / 255.).to(self.device)
        return torch.cat([radar, pwv], dim=1), target

    def normalization(self, round_id):
        starts = self.starts(round_id, 'train')
        indices = np.unique((starts[:, None] + np.arange(10, 30)).ravel())
        total = square = 0.; pixels = 0
        for block in batches(indices, 256):
            x = self.radar[block].astype(np.float64) / 255.
            total += x.sum(); square += np.square(x).sum(); pixels += x.size
        mean = float(total / pixels); std = float(np.sqrt(max(0., square / pixels - mean * mean)))
        if std <= .001: raise ValueError('Training target variance too small')
        return {'mean_normalized': mean, 'std_normalized': std, 'mean_dbz': mean * 50,
                'std_dbz': std * 50, 'unique_target_frames': len(indices), 'fit_split': 'train',
                'training_index_sha256': sha(self.root / f'{round_id}_train.npy'),
                'dataset_id': self.manifest['dataset_id']}

def prepare(frames_csv, destination):
    """Decode supplied PNG pairs; do not interpolate or repair missing frames."""
    source = Path(frames_csv).resolve(); dest = Path(destination)
    if dest.exists() and any(dest.iterdir()): raise FileExistsError('Output cache must be new or empty')
    with source.open(encoding='utf-8-sig', newline='') as stream:
        records = list(csv.DictReader(stream))
    if not records or not {'time', 'radar_path', 'pwv_path'} <= set(records[0]):
        raise ValueError('CSV columns required: time,radar_path,pwv_path')
    parsed = []
    for r in records:
        t = np.datetime64(r['time'], 's')
        if np.isnat(t): raise ValueError('Invalid timestamp')
        if t.astype('datetime64[m]').astype('datetime64[s]') != t: raise ValueError('Timestamps must be whole minutes')
        rp = source.parent / r['radar_path']; pp = source.parent / r['pwv_path']
        if not rp.is_file() or not pp.is_file(): raise FileNotFoundError('Missing input PNG; remove incomplete CSV rows explicitly')
        parsed.append((t.astype('datetime64[m]'), rp, pp))
    parsed.sort(key=lambda x: x[0]); times = np.array([r[0] for r in parsed])
    if len(np.unique(times)) != len(times): raise ValueError('Duplicate timestamps')
    if len(times) < 30: raise ValueError('At least 30 paired frames required')
    dest.mkdir(parents=True, exist_ok=True)
    radar = np.lib.format.open_memmap(dest / 'radar_u8.npy', mode='w+', dtype=np.uint8, shape=(len(times), 66, 70))
    pwv = np.lib.format.open_memmap(dest / 'pwv_u8.npy', mode='w+', dtype=np.uint8, shape=radar.shape)
    for i, (_, rp, pp) in enumerate(parsed):
        for path, target in [(rp, radar), (pp, pwv)]:
            with Image.open(path) as image:
                if image.mode != 'L' or image.size != (70, 66):
                    raise ValueError(f'Expected original 70x66 grayscale L image: {path.name}')
                target[i] = 255 - np.asarray(image, dtype=np.uint8)
    radar.flush(); pwv.flush(); del radar, pwv
    np.save(dest / 'times.npy', times, allow_pickle=False)
    valid = np.array([i for i in range(len(times) - 29)
                      if np.all(np.diff(times[i:i + 30]) == np.timedelta64(6, 'm'))], dtype=np.int64)
    counts = {}
    for rnd, periods in PERIODS.items():
        for split, (lo, hi) in periods.items():
            ids = valid[(times[valid] >= np.datetime64(lo)) & (times[valid + 29] < np.datetime64(hi))]
            np.save(dest / f'{rnd}_{split}.npy', ids, allow_pickle=False); counts[f'{rnd}_{split}'] = len(ids)
            if split == 'valid':
                # For newly prepared data only; use the original frozen screen indices for historical replay.
                screen = ids[np.unique(np.linspace(0, len(ids) - 1, min(768, len(ids)), dtype=np.int64))] if len(ids) else ids
                np.save(dest / f'{rnd}_screen.npy', screen, allow_pickle=False)
    files = {p.name: sha(p) for p in sorted(dest.glob('*.npy'))}
    dataset_id = hashlib.sha256(str(sorted(files.items())).encode()).hexdigest()
    write(dest / 'manifest.json', {'dataset_id': dataset_id, 'files': files, 'counts': counts,
          'encoding': 'decoded uint8;0=0,255=50dBZ or80mm; PNG inversion performed once',
          'screen_policy': 'evenly spaced up to768; not the historical frozen screen when preparing new data'})
    return counts
