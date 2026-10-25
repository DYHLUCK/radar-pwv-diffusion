"""Portable constants and atomic JSON/CSV I/O. No machine-specific paths."""
from pathlib import Path
import csv
import hashlib
import json
import math
import os
import numpy as np

SEEDS = [20261019, 20261020, 20261021]
THRESHOLDS = [20., 30., 40.]
WINDOWS = [1, 3, 5]
LEADS = list(range(6, 121, 6))

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()

def clean(value):
    if isinstance(value, dict): return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)): return [clean(v) for v in value]
    if isinstance(value, np.ndarray): return clean(value.tolist())
    if isinstance(value, (bool, np.bool_)): return bool(value)
    if isinstance(value, (int, np.integer)): return int(value)
    if isinstance(value, (float, np.floating)): return float(value) if math.isfinite(value) else None
    if isinstance(value, Path): return str(value)
    return value

def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))

def write(path, value):
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(clean(value), ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    os.replace(tmp, p)

def csvwrite(path, records):
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(k for r in records for k in r))
    with p.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=keys); writer.writeheader(); writer.writerows(clean(records))
