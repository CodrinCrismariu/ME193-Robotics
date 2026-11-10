"""Train the hand detector on whole camera frames and measure how well its boxes match the labeled ones.

Run from this folder:   ..\\.venv\\Scripts\\python train_detector.py            (runs/detector/)

Uses the same split by whole bursts as the classifier, so its test bursts are untouched by both networks.
"No hand" examples: the few photos with no hand, plus random squares of letter photos that stay clear of the hand
box (25% extra per batch), scaled up to the detector's input. Augmentation: random zoom, shift and mirror flip
(the box moves with the image) and random lighting.

Scores on the test bursts:
  hand / no hand    how often it correctly says whether there is a hand
  IoU               overlap of the found box with the labeled box: area of overlap / area of union (1 = perfect)
  empty crops       how often it imagines a hand in squares of the test photos that have no hand
"""
import argparse
import json
import random
import time
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.patches import Rectangle
from PIL import Image

import augment
from data import DATA, load_rows, split_by_burst
from detector import IN_H, IN_W, HandDetector, box_iou, decode, encode

HERE = Path(__file__).resolve().parent
SRC_W, SRC_H = 320, 240


def load_frames(rows, data=DATA):
    x = torch.from_numpy(np.stack([np.asarray(Image.open(data / "images" / f"{r['id']}.jpg").convert("RGB")) for r in rows]))
    boxes = torch.tensor([r["box"] if r.get("box") else [0, 0, 0, 0] for r in rows], dtype=torch.float32)
    has = torch.tensor([bool(r.get("box")) for r in rows])
    return x.permute(0, 3, 1, 2).contiguous(), boxes, has


def view(x_u8, boxes, train):
    """Frames (N,3,240,320) -> detector inputs (N,3,IN_H,IN_W) and boxes in input pixels.
    Training: random zoom (0.85-1.2x), shift (+-12%) and mirror flip; the boxes move with the image."""
    n, dev = x_u8.shape[0], x_u8.device
    if train:
        a = 1 / (0.85 + 0.35 * torch.rand(n))                      # half-size of the view in normalized input coords
        f = torch.where(torch.rand(n) < 0.5, -1.0, 1.0)
        t = 0.12 * (2 * torch.rand(n, 2) - 1)
    else:
        a, f, t = torch.ones(n), torch.ones(n), torch.zeros(n, 2)
    z = torch.zeros(n)
    theta = torch.stack([torch.stack([a * f, z, t[:, 0]], 1), torch.stack([z, a, t[:, 1]], 1)], 1)
    grid = F.affine_grid(theta, (n, 3, IN_H, IN_W), align_corners=False)
    img = F.grid_sample(x_u8.float() / 255, grid, padding_mode="border", align_corners=False)
    # input pixel -> normalized input -> normalized output (x_out = (x_in - t) / (a f)) -> output pixel
    xs = (boxes[:, [0, 2]] / SRC_W * 2 - 1 - t[:, :1]) / (a * f)[:, None]
    ys = (boxes[:, [1, 3]] / SRC_H * 2 - 1 - t[:, 1:]) / a[:, None]
    xs, ys = (xs + 1) / 2 * IN_W, (ys + 1) / 2 * IN_H
    out = torch.stack([xs.min(1).values, ys.min(1).values, xs.max(1).values, ys.max(1).values], 1)
    out[:, [0, 2]] = out[:, [0, 2]].clamp(0, IN_W)
    out[:, [1, 3]] = out[:, [1, 3]].clamp(0, IN_H)
    return img, out


def empty_windows(boxes, idx, n, gen, margin=6):
    """n random 4:3 windows (30-60% of the frame) from frames idx that stay clear of each frame's hand box.
    Returns (frame index, window x0, y0, x1, y1) for each; these crops have no hand in them."""
    out = []
    tries = 0
    while len(out) < n and tries < 50 * n:
        tries += 1
        k = int(idx[torch.randint(len(idx), (1,), generator=gen)])
        s = 0.3 + 0.3 * float(torch.rand(1, generator=gen))
        w, h = s * SRC_W, s * SRC_H
        x0 = float(torch.rand(1, generator=gen)) * (SRC_W - w)
        y0 = float(torch.rand(1, generator=gen)) * (SRC_H - h)
        bx0, by0, bx1, by1 = (boxes[k] + torch.tensor([-margin, -margin, margin, margin])).tolist()
        if x0 + w <= bx0 or x0 >= bx1 or y0 + h <= by0 or y0 >= by1:          # no overlap with the hand box
            out.append((k, x0, y0, x0 + w, y0 + h))
    return out


def empty_view(x_u8, windows, flip=None):
    """Detector inputs made from the given windows (scaled up to IN_W x IN_H): frames with no hand."""
    n = len(windows)
    k = torch.tensor([w[0] for w in windows])
    win = torch.tensor([w[1:] for w in windows])
    f = flip if flip is not None else torch.ones(n)
    ax, ay = (win[:, 2] - win[:, 0]) / SRC_W, (win[:, 3] - win[:, 1]) / SRC_H
    cx, cy = (win[:, 0] + win[:, 2]) / SRC_W - 1, (win[:, 1] + win[:, 3]) / SRC_H - 1
    z = torch.zeros(n)
    theta = torch.stack([torch.stack([ax * f, z, cx], 1), torch.stack([z, ay, cy], 1)], 1)
    grid = F.affine_grid(theta, (n, 3, IN_H, IN_W), align_corners=False)
    return F.grid_sample(x_u8[k].float() / 255, grid, padding_mode="border", align_corners=False)


def loss_fn(out, boxes, has):
    obj, reg, cell = encode(boxes, has)
    score = out[:, 0]
    l_obj = F.binary_cross_entropy_with_logits(score, obj, pos_weight=torch.tensor(20.0))
    if has.any():
        n = out.shape[0]
        pick = out.reshape(n, 5, -1)[torch.arange(n), :, cell][has]          # predictions at the true cell
        pred = torch.stack([torch.sigmoid(pick[:, 1]), torch.sigmoid(pick[:, 2]), pick[:, 3], pick[:, 4]], 1)
        l_box = F.smooth_l1_loss(pred, reg[has], beta=0.1)
    else:
        l_box = torch.zeros(())
    return l_obj + 5 * l_box


@torch.no_grad()
def evaluate(net, x_u8, boxes, has, empty=None, batch=128):
    """Scores on whole frames, plus (if given) the false-alarm rate on empty crops (N,3,IN_H,IN_W float)."""
    net.eval()
    preds, probs = [], []
    for i in range(0, len(x_u8), batch):
        img, _ = view(x_u8[i:i + batch], boxes[i:i + batch], train=False)
        b, p = decode(net(augment.normalize(img)))
        preds.append(b)
        probs.append(p)
    pred, prob = torch.cat(preds), torch.cat(probs)
    pred = pred * torch.tensor([SRC_W / IN_W, SRC_H / IN_H] * 2)            # back to 320 x 240 pixels
    found = prob > 0.5
    iou = box_iou(pred, boxes)
    res = {
        "hand_or_not_right": (found == has).float().mean().item(),
        "missed_hands": (~found & has).sum().item(), "false_hands": (found & ~has).sum().item(),
        "mean_iou": iou[has].mean().item() if has.any() else 0.0,
        "iou_over_0.5": (iou[has] > 0.5).float().mean().item() if has.any() else 0.0,
        "iou_over_0.7": (iou[has] > 0.7).float().mean().item() if has.any() else 0.0,
    }
    if empty is not None and len(empty):
        _, pe = decode(net(augment.normalize(empty)))
        res["empty_crops"] = len(empty)
        res["false_hands_on_empty_crops"] = (pe > 0.5).float().mean().item()
    return res, pred, prob


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--name", default="detector")
    p.add_argument("--empty-frac", type=float, default=0.25, help="extra no-hand crops per training batch (fraction)")
    a = p.parse_args()
    torch.manual_seed(a.seed)
    random.seed(a.seed)
    out_dir = HERE / "runs" / a.name
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    rows = load_rows()
    part = split_by_burst(rows)
    x, boxes, has = load_frames(rows)
    split = np.array([part[r["id"]] for r in rows])
    idx = {s: torch.from_numpy(np.flatnonzero(split == s)) for s in ("train", "val", "test")}
    print(f"{len(rows)} frames ({int(has.sum())} with a hand): train {len(idx['train'])}, val {len(idx['val'])}, "
          f"test {len(idx['test'])} (whole bursts); loaded in {time.time() - t0:.0f} s")
    # "no hand" examples: random squares of letter photos that stay clear of the hand box
    letter = torch.tensor([bool(r.get("box")) and r["label"] != "none" for r in rows])
    src = {s: idx[s][letter[idx[s]]] for s in idx}
    g = torch.Generator().manual_seed(123)
    val_empty = empty_view(x, empty_windows(boxes, src["val"], 300, g))
    test_empty = empty_view(x, empty_windows(boxes, src["test"], 400, g))
    train_gen = torch.Generator().manual_seed(a.seed)
    print(f"empty crops: {a.empty_frac:.0%} extra per training batch; {len(val_empty)} for validation, {len(test_empty)} for test")

    net = HandDetector()
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=5e-4)
    steps = a.epochs * ((len(idx["train"]) + a.batch - 1) // a.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps, pct_start=0.15)
    hist, best = [], (-1.0, None, 0)
    vi = idx["val"]
    for epoch in range(1, a.epochs + 1):
        net.train()
        order = idx["train"][torch.randperm(len(idx["train"]))]
        total, seen = 0.0, 0
        for i in range(0, len(order), a.batch):
            b = order[i:i + a.batch]
            img, bx = view(x[b], boxes[b], train=True)
            hb = has[b]
            wins = empty_windows(boxes, src["train"], max(1, round(a.empty_frac * len(b))), train_gen)
            if wins:                                                      # add empty crops: no hand, no box
                flip = torch.where(torch.rand(len(wins)) < 0.5, -1.0, 1.0)
                img = torch.cat([img, empty_view(x, wins, flip)])
                bx = torch.cat([bx, torch.zeros(len(wins), 4)])
                hb = torch.cat([hb, torch.zeros(len(wins), dtype=torch.bool)])
            loss = loss_fn(net(augment.normalize(augment.lighting(img))), bx, hb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            total += loss.item() * len(b)
            seen += len(b)
        m, _, _ = evaluate(net, x[vi], boxes[vi], has[vi], val_empty)
        hist.append({"epoch": epoch, "loss": total / seen, **{f"val_{k}": v for k, v in m.items()}})
        score = m["hand_or_not_right"] * (1 - m["false_hands_on_empty_crops"]) * m["mean_iou"]
        if score > best[0]:
            best = (score, {k: v.clone() for k, v in net.state_dict().items()}, epoch)
        if True:                                                        # every epoch, for watch_training.py
            print(f"epoch {epoch:3d}  loss {total / seen:.3f}  val: hand/no hand {m['hand_or_not_right']:.2f}  "
                  f"false hand on empty {m['false_hands_on_empty_crops']:.2f}  mean IoU {m['mean_iou']:.2f}  "
                  f"IoU>0.5 {m['iou_over_0.5']:.2f}   ({time.time() - t0:.0f} s)")
    net.load_state_dict(best[1])

    ti = idx["test"]
    res, pred, prob = evaluate(net, x[ti], boxes[ti], has[ti], test_empty)
    res.update({"best_epoch": best[2], "n_test": len(ti), "n_test_with_hand": int(has[ti].sum()), "history": hist})
    (out_dir / "metrics.json").write_text(json.dumps(res, indent=1))
    torch.save({"state_dict": net.state_dict(), "input": [IN_W, IN_H]}, out_dir / "model.pt")
    print(f"\nbest epoch {best[2]}.  TEST on {len(ti)} frames from held-out bursts ({res['n_test_with_hand']} with a hand):")
    print(f"  hand / no hand right:  {res['hand_or_not_right']:.1%}   (missed {res['missed_hands']} hands, "
          f"{res['false_hands']} false hands)")
    print(f"  box overlap (IoU):     mean {res['mean_iou']:.2f};  {res['iou_over_0.5']:.0%} over 0.5,  {res['iou_over_0.7']:.0%} over 0.7")
    print(f"  empty crops:           sees a hand in {res['false_hands_on_empty_crops']:.1%} of {res['empty_crops']} crops with no hand")

    # a look at some test frames: labeled box green, found box orange
    matplotlib.rcParams.update({"font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 8})
    rng = random.Random(0)
    pick = rng.sample(range(len(ti)), min(12, len(ti)))
    fig, axes = plt.subplots(3, 4, figsize=(12, 7))
    for ax, k in zip(axes.flat, pick):
        r = rows[int(ti[k])]
        ax.imshow(x[ti[k]].permute(1, 2, 0).numpy())
        if has[ti[k]]:
            b = boxes[ti[k]].tolist()
            ax.add_patch(Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], fill=False, ec="#2e9e5b", lw=2))
        if prob[k] > 0.5:
            b = pred[k].tolist()
            ax.add_patch(Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], fill=False, ec="#e8743b", lw=2, ls="--"))
        ax.set_title(f"{r['label']}  p(hand) {prob[k]:.2f}", fontsize=8)
        ax.axis("off")
    fig.suptitle("Test frames: labeled box (green) and detector box (orange, dashed)", x=0.01, ha="left", fontweight="bold")
    fig.savefig(out_dir / "test_predictions.png", dpi=100, bbox_inches="tight")
    print(f"saved {out_dir.relative_to(HERE)}  ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
