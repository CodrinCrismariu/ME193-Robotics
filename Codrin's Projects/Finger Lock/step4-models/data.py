"""Loading the dataset for training: checked photos, a split by whole bursts, and hand crops with their masks."""
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "step3-dataset" / "dataset"
LABELS = ["C", "O", "D", "R", "I", "N", "none"]
CTX = 128             # side of the stored context crop around each hand (pixels)
CTX_SCALE = 1.7       # the context crop is this many times the box's longer side


def load_rows(data=DATA, checked_only=True):
    rows = [json.loads(l) for l in (data / "samples.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    return [r for r in rows if r.get("checked") or not checked_only]


def split_by_burst(rows, test=0.2, val=0.12, seed=0):
    """Assign whole bursts to train / val / test, separately for every label, so near-identical photos from one
    burst never end up on both sides. For each label, bursts are picked whose sizes add up closest to the wanted
    share (at least one burst each), always leaving at least two bursts for training.
    Returns {row id: 'train' | 'val' | 'test'}."""
    rng = random.Random(seed)
    groups = defaultdict(list)
    for r in rows:
        groups[(r["label"], r["session"], r["burst"])].append(r["id"])
    by_label = defaultdict(list)
    for key, ids in groups.items():
        by_label[key[0]].append(ids)
    part = {}
    for label, bursts in by_label.items():
        rng.shuffle(bursts)
        n = sum(len(b) for b in bursts)
        pool = [b for b in bursts if len(b) >= 5]            # stray single shots always stay in training
        for name, share in (("test", test), ("val", val)):
            target, have = share * n, 0
            while pool and len(pool) > 2 and (have == 0 or have < 0.8 * target):
                pick = min(pool, key=lambda b: abs(have + len(b) - target))
                if have and have + len(pick) > 1.3 * target:
                    break
                pool.remove(pick)
                have += len(pick)
                for sid in pick:
                    part[sid] = name
        for b in bursts:
            for sid in b:
                part.setdefault(sid, "train")
    return part


def context_crop(img, box, side=CTX, scale=CTX_SCALE):
    """Square crop centered on the box, scale x its longer side, resized to side x side (edges padded)."""
    x0, y0, x1, y1 = box
    cx, cy, s = (x0 + x1) / 2, (y0 + y1) / 2, max(x1 - x0, y1 - y0) * scale
    l, t = int(round(cx - s / 2)), int(round(cy - s / 2))
    r, b = l + int(round(s)), t + int(round(s))
    w, h = img.size
    pad = max(0, -l, -t, r - w, b - h)
    if pad:
        arr = np.asarray(img)
        arr = np.pad(arr, ((pad, pad), (pad, pad)) + ((0, 0),) * (arr.ndim - 2), mode="edge")
        img = Image.fromarray(arr)
        l, t, r, b = l + pad, t + pad, r + pad, b + pad
    return img.crop((l, t, r, b)).resize((side, side), Image.BILINEAR)


def load_crops(rows, data=DATA):
    """Context crops of every row with a box: images (N,3,CTX,CTX) uint8, masks (N,1,CTX,CTX) uint8, labels (N,)."""
    ims, ms, ys, kept = [], [], [], []
    for r in rows:
        if not r.get("box"):
            continue
        img = Image.open(data / "images" / f"{r['id']}.jpg").convert("RGB")
        mpath = data / "masks" / f"{r['id']}.png"
        mask = Image.open(mpath).convert("L") if mpath.exists() else Image.new("L", img.size, 0)
        ims.append(np.asarray(context_crop(img, r["box"])))
        ms.append(np.asarray(context_crop(mask, r["box"])))
        ys.append(LABELS.index(r["label"]))
        kept.append(r)
    x = torch.from_numpy(np.stack(ims)).permute(0, 3, 1, 2).contiguous()
    m = torch.from_numpy(np.stack(ms))[:, None].contiguous()
    return x, m, torch.tensor(ys), kept


def background_patches(rows, n, side=CTX, seed=0, data=DATA):
    """Random square patches of real photos (any label, any place) to paste hands onto."""
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        r = rng.choice(rows)
        img = Image.open(data / "images" / f"{r['id']}.jpg").convert("RGB")
        w, h = img.size
        s = rng.randint(48, 200)
        x, y = rng.randint(0, w - min(s, w)), rng.randint(0, h - min(s, h))
        out.append(np.asarray(img.crop((x, y, x + min(s, w), y + min(s, h))).resize((side, side), Image.BILINEAR)))
    return torch.from_numpy(np.stack(out)).permute(0, 3, 1, 2).contiguous()
