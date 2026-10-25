"""Controlled fine-tuning hypotheses; unchanged forecasting/generation mechanisms."""
import torch
from models import diffusion_coefficients
from objectives import components, deterministic_loss, generative_loss


def clean_objective(prediction, truth, variant, scale):
    weighted, csi, fss, temporal = components(prediction, truth)
    # Put continuous errors into training-target standard-deviation units.
    # No test/validation values enter this scale.
    reconstruction = weighted / (scale * scale)
    if variant == 'reconstruction':
        return .25 * reconstruction
    assert variant == 'balanced'
    # Penalize background echo as well as supported event overlap. Unlike the old
    # CSI term this explicitly covers truth-empty frames. 5 dBZ softness avoids
    # calling a nearly zero prediction a foreground event.
    excess = (prediction.float() - .1).relu().square()
    background = (excess * (truth < .1)).mean((1, 2, 3)) / (scale * scale)
    return (.25 * reconstruction + .08 * csi + .08 * fss
            + .02 * temporal / scale + .05 * background)


def loss(model, x, truth, method, variant, old_recipe, scale, base=None):
    if variant == 'continue':
        return (deterministic_loss(model(x), truth, old_recipe)
                if method == 'deterministic' else
                generative_loss(model, x, truth, method, old_recipe, base))
    if method == 'deterministic':
        return clean_objective(model(x), truth, variant, scale).mean()
    target_clean = model.normalize_target(truth.float(), base)
    noise = torch.randn_like(target_clean)
    t = torch.rand(len(x), device=x.device)
    condition = model.encode(x, base)
    if method == 'flow':
        tt = t[:, None, None, None]
        noisy = (1 - tt) * noise + tt * target_clean
        target = target_clean - noise
        velocity = model(noisy, t, condition).float()
        reconstructed = model.restore_target(noisy + (1 - tt) * velocity)
        confidence = t.square()
    else:
        a, s = diffusion_coefficients(t)
        noisy = a * target_clean + s * noise
        target = a * noise - s * target_clean
        velocity = model(noisy, t, condition).float()
        reconstructed = model.restore_target(a * noisy - s * velocity, base)
        confidence = a[:, 0, 0, 0].square()
    native = (velocity - target).square().mean((1, 2, 3))
    # Keep a nonzero clean-field constraint near the sampling start, where the
    # former confidence weighting almost eliminated the auxiliary gradient.
    auxiliary = clean_objective(reconstructed, truth, variant, scale)
    return (native + (.25 + .75 * confidence) * auxiliary).mean()
