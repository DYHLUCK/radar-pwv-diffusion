"""Small common conditional networks; method-inspired experiments, NOT full-paper reproductions.

All 20 future frames are generated jointly as channels. Spatial convs are denoiser
components; the candidate forecasting paradigms are diffusion and flow matching.
"""
import copy
import math
import torch
from torch import nn
import torch.nn.functional as F

class TimeEmbedding(nn.Module):
    def __init__(self, width=128):
        super().__init__()
        self.width = width
        self.mlp = nn.Sequential(nn.Linear(width, width*2), nn.SiLU(), nn.Linear(width*2, width))
    def forward(self, t):
        frequencies = torch.exp(-math.log(10000) * torch.arange(self.width//2, device=t.device).float() / (self.width//2-1))
        z = t[:, None].float() * frequencies[None] * 1000.
        return self.mlp(torch.cat([z.sin(), z.cos()], dim=-1))

class Block(nn.Module):
    def __init__(self, cin, cout, tdim=0):
        super().__init__()
        self.n1 = nn.GroupNorm(4, cin)
        self.c1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.n2 = nn.GroupNorm(4, cout)
        self.c2 = nn.Conv2d(cout, cout, 3, padding=1)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()
        self.time = nn.Linear(tdim, cout*2) if tdim else None
    def forward(self, x, time=None):
        h = self.c1(F.silu(self.n1(x)))
        h = self.n2(h)
        if self.time is not None:
            scale, bias = self.time(F.silu(time)).chunk(2, dim=1)
            h = h * (1 + scale[:, :, None, None]) + bias[:, :, None, None]
        return (self.skip(x) + self.c2(F.silu(h))) / math.sqrt(2)

class Condition(nn.Module):
    def __init__(self, width=24, with_base=False):
        super().__init__()
        self.with_base = with_base
        self.radar = nn.Conv2d(10, width//2, 3, padding=1)
        self.pwv = nn.Conv2d(10, width//2, 3, padding=1)
        self.merge = nn.Conv2d(width + (20 if with_base else 0), width, 3, padding=1)
        self.b0 = Block(width, width)
        self.down1 = nn.Conv2d(width, width*2, 3, stride=2, padding=1)
        self.b1 = Block(width*2, width*2)
        self.down2 = nn.Conv2d(width*2, width*4, 3, stride=2, padding=1)
        self.b2 = Block(width*4, width*4)
    def forward(self, x, base=None):
        # Reflect only the two bottom/right border cells; crop back to the original grid.
        x = F.pad(x, (0, 2, 0, 2), mode='reflect')
        h = [self.radar(2*x[:, :10]-1), self.pwv(2*x[:, 10:]-1)]
        if self.with_base:
            assert base is not None
            h.append(F.pad(2*base-1, (0, 2, 0, 2), mode='reflect'))
        a = self.b0(self.merge(torch.cat(h, dim=1)))
        b = self.b1(self.down1(a))
        c = self.b2(self.down2(b))
        return a, b, c

class Predictor(nn.Module):
    def __init__(self, width=24):
        super().__init__()
        self.condition = Condition(width)
        self.up1 = Block(width*6, width*2)
        self.up0 = Block(width*3, width)
        self.out = nn.Conv2d(width, 20, 3, padding=1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
    def forward(self, x):
        a, b, c = self.condition(x)
        h = self.up1(torch.cat([F.interpolate(c, size=b.shape[-2:], mode='bilinear', align_corners=False), b], dim=1))
        h = self.up0(torch.cat([F.interpolate(h, size=a.shape[-2:], mode='bilinear', align_corners=False), a], dim=1))
        last = (.02 + .96*x[:, 9:10]).clamp(.001, .999)
        return (self.out(h)[:, :, :66, :70] + torch.logit(last)).sigmoid()

class Generator(nn.Module):
    def __init__(self, width=24, residual=False):
        super().__init__()
        self.residual = residual
        # Scalars are serialized with each checkpoint; inverse scaling cannot be lost on export.
        # Default values reproduce V1 exactly for controlled equivalence checks.
        self.register_buffer('target_mean',torch.tensor(.5,dtype=torch.float32))
        self.register_buffer('target_std',torch.tensor(.5,dtype=torch.float32))
        self.condition = Condition(width, with_base=residual)
        self.time = TimeEmbedding(128)
        self.inp = nn.Conv2d(20, width, 3, padding=1)
        self.b0 = Block(width*2, width, 128)
        self.down1 = nn.Conv2d(width, width*2, 3, stride=2, padding=1)
        self.b1 = Block(width*4, width*2, 128)
        self.down2 = nn.Conv2d(width*2, width*4, 3, stride=2, padding=1)
        self.b2 = Block(width*8, width*4, 128)
        self.mid = Block(width*4, width*4, 128)
        self.up1 = Block(width*8, width*2, 128)
        self.up0 = Block(width*4, width, 128)
        self.out = nn.Conv2d(width, 20, 3, padding=1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
    def set_normalization(self,mean,std):
        assert std>0
        self.target_mean.fill_(mean)
        self.target_std.fill_(std)
    def normalize_target(self,y,base=None):
        return (y-base)/self.target_std if self.residual else (y-self.target_mean)/self.target_std
    def restore_target(self,z,base=None):
        return base+z*self.target_std if self.residual else z*self.target_std+self.target_mean
    def clip_target(self,z):
        lo=-1/self.target_std if self.residual else -self.target_mean/self.target_std
        hi=1/self.target_std if self.residual else (1-self.target_mean)/self.target_std
        return z.clamp(lo,hi)
    def encode(self, x, base=None):
        return self.condition(x, base)
    def forward(self, z, t, condition):
        a, b, c = condition
        time = self.time(t)
        z = F.pad(z, (0, 2, 0, 2), mode='reflect')
        h0 = self.b0(torch.cat([self.inp(z), a], dim=1), time)
        h1 = self.b1(torch.cat([self.down1(h0), b], dim=1), time)
        h2 = self.b2(torch.cat([self.down2(h1), c], dim=1), time)
        h = self.mid(h2, time)
        h = self.up1(torch.cat([F.interpolate(h, size=h1.shape[-2:], mode='bilinear', align_corners=False), h1, b], dim=1), time)
        h = self.up0(torch.cat([F.interpolate(h, size=h0.shape[-2:], mode='bilinear', align_corners=False), h0, a], dim=1), time)
        return self.out(F.silu(h))[:, :, :66, :70]

class EMA:
    def __init__(self, model, decay=.999):
        self.model = copy.deepcopy(model).eval().requires_grad_(False)
        self.decay = decay
        self.updates = 0
    @torch.no_grad()
    def update(self, model):
        self.updates += 1
        decay = min(self.decay, (1+self.updates)/(10+self.updates))
        for target, value in zip(self.model.parameters(), model.parameters()):
            target.lerp_(value.detach(), 1-decay)
        for target, value in zip(self.model.buffers(), model.buffers()):
            target.copy_(value)

def build(method, width=24):
    if method == 'deterministic':
        return Predictor(width)
    assert method in ['ddpm', 'resdiff', 'flow']
    return Generator(width, residual=method == 'resdiff')

def diffusion_coefficients(t):
    angle = ((t + .008) / 1.008) * math.pi / 2
    alpha = angle.cos().clamp(min=1e-5)
    sigma = angle.sin().clamp(min=1e-5)
    return alpha[:, None, None, None], sigma[:, None, None, None]

@torch.no_grad()
def sample(model, x, noise=None, method='deterministic', base=None, steps=16):
    if method == 'deterministic':
        return model(x).float().clamp(0, 1)
    condition = model.encode(x, base)
    z = noise.clone()
    if method == 'flow':
        for i in range(steps):
            t = torch.full((len(x),), i/steps, device=x.device)
            z = z + model(z, t, condition).float() / steps
        return model.restore_target(z).clamp(0, 1)
    for i in range(steps):
        t = torch.full((len(x),), 1 - i/steps, device=x.device)
        a, s = diffusion_coefficients(t)
        v = model(z, t, condition).float()
        z0 = model.clip_target(a*z - s*v)
        epsilon = s*z + a*v
        if i == steps-1:
            z = z0
        else:
            na, ns = diffusion_coefficients(t - 1/steps)
            z = na*z0 + ns*epsilon
    return model.restore_target(z,base).clamp(0, 1)
