"""Same 16-NFE forward sampler, with gradients through only the final K steps.

No new model parameters, altered path, changed noise prior or inference filter.
The training hypothesis is to supervise actual sampled forecasts in addition to
the original denoising/flow objective, rather than only teacher-forced x0.
"""
from contextlib import nullcontext
import torch
from models import diffusion_coefficients
from optimization_losses import clean_objective


def differentiable_sample(model, x, noise, method, base=None, steps=16, last_k=4):
    assert method in ['ddpm', 'resdiff', 'flow']
    assert 1 <= last_k <= steps
    condition = model.encode(x, base)
    detached = tuple(h.detach() for h in condition)
    z = noise.clone()
    for i in range(steps):
        tracked = i >= steps - last_k
        if i == steps - last_k:
            z = z.detach()
            # An outer autocast context may have cached detached BF16 weights
            # during the no-grad prefix. Recast for the tracked suffix so the
            # denoiser parameters, not only condition features, receive gradients.
            torch.clear_autocast_cache()
        with nullcontext() if tracked else torch.no_grad():
            cond = condition if tracked else detached
            if method == 'flow':
                t = torch.full((len(x),), i / steps, device=x.device)
                z = z + model(z, t, cond).float() / steps
            else:
                t = torch.full((len(x),), 1 - i / steps, device=x.device)
                a, s = diffusion_coefficients(t)
                v = model(z, t, cond).float()
                z0 = model.clip_target(a * z - s * v)
                epsilon = s * z + a * v
                if i == steps - 1:
                    z = z0
                else:
                    na, ns = diffusion_coefficients(t - 1 / steps)
                    z = na * z0 + ns * epsilon
    return model.restore_target(z, base).clamp(0, 1)


def sampled_forecast_loss(model, x, truth, method, scale, base=None, last_k=4, generator=None):
    noise = torch.randn((len(x), 20, 66, 70), device=x.device, dtype=torch.float32, generator=generator)
    p = differentiable_sample(model, x, noise, method, base, 16, last_k)
    return clean_objective(p, truth, 'balanced', scale).mean()
