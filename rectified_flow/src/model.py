"""Model-specific interface; audited shared network remains in models.py."""
import torch
from models import build, sample
METHOD = 'flow'

def create_model():
    return build(METHOD)

@torch.no_grad()
def forecast(model, inputs, noise, base=None):
    """inputs:(B,20,66,70), normalized radar10+PWV10; output:(B,20,66,70)."""
    return sample(model, inputs, noise, METHOD, base, steps=16)
