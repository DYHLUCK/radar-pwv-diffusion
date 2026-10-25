"""Pooled contingency counts per lead/threshold; FSS pools squared fraction errors.

No-signal denominators yield NaN (serialized as null), never a perfect score.
Border neighbourhoods are truncated and normalized by the number of real cells.
"""
import numpy as np
import torch
import torch.nn.functional as F
from common import THRESHOLDS, WINDOWS, LEADS

@torch.no_grad()
def stats(prediction_dbz, truth_dbz):
    p, y = prediction_dbz.float(), truth_dbz.float()
    assert p.shape == y.shape and p.ndim == 4 and p.shape[1:] == (20,66,70)
    assert torch.isfinite(p).all() and torch.isfinite(y).all()
    conf, fraction = [], []
    for threshold in THRESHOLDS:
        a, b = p >= threshold, y >= threshold
        conf.append(torch.stack([(a&b).sum((2,3)), (a&~b).sum((2,3)), (~a&b).sum((2,3))], dim=-1))
        per_window = []
        for w in WINDOWS:
            af = F.avg_pool2d(a.float(), w, stride=1, padding=w//2, count_include_pad=False) if w>1 else a.float()
            bf = F.avg_pool2d(b.float(), w, stride=1, padding=w//2, count_include_pad=False) if w>1 else b.float()
            per_window.append(torch.stack([(af-bf).square().sum((2,3)), (af.square()+bf.square()).sum((2,3))], dim=-1))
        fraction.append(torch.stack(per_window, dim=-2))
    return {'conf': torch.stack(conf, dim=-2).cpu().numpy(),
            'fss': torch.stack(fraction, dim=-3).cpu().numpy().astype(np.float64)}

def divide(a, b):
    a, b = np.broadcast_arrays(a, b)
    return np.divide(a, b, out=np.full(a.shape, np.nan, dtype=float), where=b>0)

def pooled(statistics):
    c = statistics['conf'].sum(0)
    f = statistics['fss'].sum(0)
    h, fa, m = c[...,0], c[...,1], c[...,2]
    return {'CSI': divide(h, h+fa+m), 'POD': divide(h, h+m), 'FAR': divide(fa, h+fa),
            'FSS': 1-divide(f[...,0], f[...,1]), 'counts': c}

def concatenate(parts):
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}

def rows(statistics, **metadata):
    values = pooled(statistics)
    output = []
    for lead, minutes in enumerate(LEADS):
        for j, threshold in enumerate(THRESHOLDS):
            h, fa, miss = values['counts'][lead,j]
            output.append({**metadata, 'lead_minutes': minutes, 'threshold_dbz': threshold,
                           'CSI': values['CSI'][lead,j], 'POD': values['POD'][lead,j], 'FAR': values['FAR'][lead,j],
                           **{f'FSS{w}': values['FSS'][lead,j,k] for k,w in enumerate(WINDOWS)},
                           'hits': int(h), 'false_alarms': int(fa), 'misses': int(miss)})
    return output

def selection_values(statistics):
    v = pooled(statistics)
    focus = np.array([4,9,14,19])  # T+30,60,90,120; final report still contains all 20 lead times.
    return {'selection_CSI': float(np.nanmean(v['CSI'][focus])),
            'selection_POD': float(np.nanmean(v['POD'][focus])),
            'selection_FAR': float(np.nan_to_num(v['FAR'][focus], nan=1.).mean()),
            'selection_FSS3': float(np.nanmean(v['FSS'][focus,:,1]))}

def select(candidates):
    # Primary CSI, then control FAR/POD within a small predefined CSI tie band.
    assert candidates
    best_csi = max(c['selection_CSI'] for c in candidates)
    near = [c for c in candidates if c['selection_CSI'] >= best_csi-.003]
    # Near-tie score explicitly penalizes false alarms and rewards POD/FSS.
    return max(near, key=lambda c: (.5*c['selection_POD']-.5*c['selection_FAR']+.25*c['selection_FSS3'], c['selection_CSI'], -c.get('epoch',0)))
