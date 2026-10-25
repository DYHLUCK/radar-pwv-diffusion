"""Native generative objectives and controlled reflectivity/skill auxiliary losses."""
import torch
import torch.nn.functional as F
from models import diffusion_coefficients

def components(prediction, truth):
    p, y = prediction.float(), truth.float()
    weight = 1 + 2*(y >= .4) + 3*(y >= .6) + 4*(y >= .8)
    # Normalize weights per example; scale remains comparable across dry/heavy batches.
    weighted = ((p-y).square()*weight).mean((1,2,3)) / weight.float().mean((1,2,3))
    soft_scores, fss_scores = [], []
    for threshold in [.4, .6, .8]:
        # 0.02 in normalized units equals 1 dBZ of threshold softness.
        prob = ((p-threshold)/.02).sigmoid()
        event = (y >= threshold).float()
        hit = (prob*event).sum((2,3))
        union = (prob+event-prob*event).sum((2,3))
        # Ignore truth-empty frames in CSI surrogate; pixel loss penalizes their false alarms.
        supported = event.sum((2,3)) > 0
        csi_loss = torch.where(supported, 1-hit/union.clamp_min(1e-6), 0.)
        soft_scores.append(csi_loss.sum(1) / supported.sum(1).clamp_min(1))
        pp = F.avg_pool2d(prob, 3, stride=1, padding=1, count_include_pad=False)
        yy = F.avg_pool2d(event, 3, stride=1, padding=1, count_include_pad=False)
        num = (pp-yy).square().sum((2,3))
        den = (pp.square()+yy.square()).sum((2,3))
        fss_scores.append((num/den.clamp_min(1e-6)).mean(1))
    soft_csi = torch.stack(soft_scores).mean(0)
    fss_loss = torch.stack(fss_scores).mean(0)
    temporal = ((p[:,1:]-p[:,:-1]) - (y[:,1:]-y[:,:-1])).abs().mean((1,2,3))
    return weighted, soft_csi, fss_loss, temporal

def deterministic_loss(prediction, truth, recipe):
    if recipe == 'standard':
        return F.mse_loss(prediction.float(), truth.float())
    weighted, soft_csi, fss_loss, temporal = components(prediction, truth)
    if recipe == 'weighted':
        return weighted.mean()
    assert recipe == 'skill'
    return (weighted + .03*soft_csi + .05*fss_loss + .05*temporal).mean()

def generative_loss(model, x, y, method, recipe, base=None):
    y = y.float()
    z0 = model.normalize_target(y,base.float() if base is not None else None)
    noise = torch.randn_like(z0)
    t = torch.rand(len(x), device=x.device)
    condition = model.encode(x, base)
    if method == 'flow':
        tt = t[:,None,None,None]
        z = (1-tt)*noise + tt*z0
        target = z0-noise
        v = model(z, t, condition).float()
        reconstructed = model.restore_target(z+(1-tt)*v)
        confidence = t.square()
    else:
        a, s = diffusion_coefficients(t)
        z = a*z0 + s*noise
        target = a*noise - s*z0
        v = model(z, t, condition).float()
        reconstructed_z0 = a*z-s*v
        reconstructed = model.restore_target(reconstructed_z0,base.float() if base is not None else None)
        confidence = a[:,0,0,0].square()
    native = (v-target).square().mean((1,2,3))
    if recipe == 'standard':
        return native.mean()
    weighted, soft_csi, fss_loss, temporal = components(reconstructed, y)
    auxiliary = .5*weighted
    if recipe == 'skill':
        auxiliary = auxiliary + .03*soft_csi + .05*fss_loss + .05*temporal
    else:
        assert recipe == 'weighted'
    return (native + confidence*auxiliary).mean()
