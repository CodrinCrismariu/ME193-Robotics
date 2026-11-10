"""Augmentation for the letter classifier, including domain randomization of the background.

Everything works on whole batches of torch tensors, so it runs as part of training:
  geometry      random zoom, shift, rotation (+-15 deg) and mirror flip of the crop around the hand
  background    (domain randomization) the hand is cut out with its mask and pasted onto a random background:
                a patch of a random photo from anywhere in the dataset, a flat color, a gradient or smooth noise
  lighting      random brightness, contrast, saturation, color tint, grayscale, blur and noise
"""
import math

import torch
import torch.nn.functional as F

OUT = 96                  # classifier input size
EVAL_SCALE = 0.78         # how much of the context crop the classifier sees at test time (hand ~ 3/4 of the frame)


def _affine(n, train, device):
    if not train:
        s = torch.full((n,), EVAL_SCALE, device=device)
        return torch.stack([torch.stack([s, 0 * s, 0 * s], 1), torch.stack([0 * s, s, 0 * s], 1)], 1)
    s = EVAL_SCALE * (0.88 + 0.28 * torch.rand(n, device=device))           # zoom
    a = math.radians(15) * (2 * torch.rand(n, device=device) - 1)            # rotate
    f = torch.where(torch.rand(n, device=device) < 0.5, -1.0, 1.0)           # mirror
    t = 0.08 * (2 * torch.rand(n, 2, device=device) - 1)                     # shift
    cos, sin = torch.cos(a) * s, torch.sin(a) * s
    return torch.stack([torch.stack([cos * f, -sin, t[:, 0]], 1), torch.stack([sin * f, cos, t[:, 1]], 1)], 1)


def geometry(x, m, train):
    """Crop OUT x OUT views of the context crops x (N,3,S,S) and masks m (N,1,S,S), both float in [0,1]."""
    theta = _affine(x.shape[0], train, x.device)
    grid = F.affine_grid(theta, (x.shape[0], 1, OUT, OUT), align_corners=False)
    return (F.grid_sample(x, grid, padding_mode="border", align_corners=False),
            F.grid_sample(m, grid, padding_mode="zeros", align_corners=False))


def synthetic_backgrounds(n, device):
    """Flat colors, gradients and smooth noise, one kind per image."""
    kind = torch.randint(0, 3, (n,), device=device)
    flat = torch.rand(n, 3, 1, 1, device=device).expand(n, 3, OUT, OUT)
    c0, c1 = torch.rand(n, 3, 1, 1, device=device), torch.rand(n, 3, 1, 1, device=device)
    ang = 2 * math.pi * torch.rand(n, 1, 1, 1, device=device)
    yy, xx = torch.meshgrid(torch.linspace(-1, 1, OUT, device=device), torch.linspace(-1, 1, OUT, device=device), indexing="ij")
    ramp = ((xx * torch.cos(ang) + yy * torch.sin(ang)) + 1.5) / 3
    grad = c0 + (c1 - c0) * ramp
    k = int(torch.randint(3, 10, (1,)))
    noise = F.interpolate(torch.rand(n, 3, k, k, device=device), size=(OUT, OUT), mode="bicubic", align_corners=False).clamp(0, 1)
    out = torch.where(kind.view(n, 1, 1, 1) == 0, flat, torch.where(kind.view(n, 1, 1, 1) == 1, grad, noise))
    return out.contiguous()


def random_backgrounds(n, pool, device):
    """60% real photo patches from the pool (N,3,S,S uint8), 40% synthetic."""
    idx = torch.randint(0, pool.shape[0], (n,))
    real = pool[idx].to(device).float() / 255
    real = F.interpolate(real, size=(OUT, OUT), mode="bilinear", align_corners=False)
    real = torch.where(torch.rand(n, 1, 1, 1, device=device) < 0.5, real.flip(3), real)
    synth = synthetic_backgrounds(n, device)
    return torch.where(torch.rand(n, 1, 1, 1, device=device) < 0.6, real, synth)


def paste(x, m, bg):
    """Hand pixels of x (mask m, softened at the edge) over the background bg."""
    return m * x + (1 - m) * bg


def jitter_mask(m):
    """Grow or shrink the mask edge by a pixel at random, then soften it, so mask mistakes are not a clue."""
    grow = F.max_pool2d(m, 3, 1, 1)
    shrink = -F.max_pool2d(-m, 3, 1, 1)
    r = torch.rand(m.shape[0], 1, 1, 1, device=m.device)
    m = torch.where(r < 0.33, grow, torch.where(r < 0.66, shrink, m))
    return F.avg_pool2d(m, 3, 1, 1)


def lighting(x):
    n, dev = x.shape[0], x.device
    x = x * (0.6 + 0.8 * torch.rand(n, 1, 1, 1, device=dev))                                 # brightness
    mean = x.mean((1, 2, 3), keepdim=True)
    x = mean + (x - mean) * (0.7 + 0.6 * torch.rand(n, 1, 1, 1, device=dev))                  # contrast
    gray = x.mean(1, keepdim=True)
    x = gray + (x - gray) * (0.6 + 0.8 * torch.rand(n, 1, 1, 1, device=dev))                  # saturation
    x = x * (0.9 + 0.2 * torch.rand(n, 3, 1, 1, device=dev))                                  # color tint
    x = torch.where(torch.rand(n, 1, 1, 1, device=dev) < 0.15, x.mean(1, keepdim=True).expand_as(x), x)   # grayscale
    x = torch.where(torch.rand(n, 1, 1, 1, device=dev) < 0.2, F.avg_pool2d(x, 3, 1, 1), x)    # blur
    x = x + 0.04 * torch.rand(n, 1, 1, 1, device=dev) * torch.randn_like(x)                   # noise
    return x.clamp(0, 1)


def normalize(x):
    return (x - 0.5) / 0.25


def train_batch(x_u8, m_u8, pool, randomize_bg, p_bg=0.9):
    """A training batch: uint8 context crops -> augmented, normalized OUT x OUT inputs."""
    x, m = geometry(x_u8.float() / 255, m_u8.float() / 255, train=True)
    if randomize_bg:
        m = jitter_mask(m)
        bg = random_backgrounds(x.shape[0], pool, x.device)
        use = (torch.rand(x.shape[0], 1, 1, 1, device=x.device) < p_bg).float()
        x = use * paste(x, m, bg) + (1 - use) * x
    return normalize(lighting(x))


def eval_batch(x_u8, m_u8=None, backgrounds=None):
    """A test batch: the fixed center view; optionally the hand pasted onto given backgrounds (OUT x OUT, float)."""
    x, m = geometry(x_u8.float() / 255, (m_u8 if m_u8 is not None else x_u8[:, :1]).float() / 255, train=False)
    if backgrounds is not None:
        x = paste(x, F.avg_pool2d(m, 3, 1, 1), backgrounds)
    return normalize(x)
