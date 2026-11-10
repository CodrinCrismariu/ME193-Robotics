"""Train the letter classifier on hand crops, with or without background randomization, and test whether it
reads the hand or the background.

Run from this folder:
  ..\\.venv\\Scripts\\python train_classifier.py                      domain randomization on  (runs/randomized/)
  ..\\.venv\\Scripts\\python train_classifier.py --no-bg-random       only the usual augmentation (runs/baseline/)

Three tests on held-out bursts:
  real photos         the crops as the camera saw them
  swapped background  the same hands pasted onto scenes from photos of OTHER letters:
                      a model that reads the hand keeps its accuracy; one that reads the scene drops
  hand removed        the hand painted out of the crop: a model that reads the hand can't name the letter any more
"""
import argparse
import json
import random
import time
from pathlib import Path

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

import augment
from data import LABELS, background_patches, load_crops, load_rows, split_by_burst
from models import LetterNet

HERE = Path(__file__).resolve().parent


def swapped_backgrounds(rows, y, n_per_label=300, seed=1):
    """For every sample, a scene patch from photos of a different letter (OUT x OUT, float)."""
    pools = {l: background_patches([r for r in rows if r["label"] == l], n_per_label, seed=seed + i)
             for i, l in enumerate(LABELS)}
    rng = random.Random(seed)
    out = []
    for yy in y.tolist():
        other = rng.choice([l for l in LABELS if l != LABELS[yy] and len(pools[l])])
        out.append(pools[other][rng.randrange(len(pools[other]))])
    bg = torch.stack(out).float() / 255
    return F.interpolate(bg, size=(augment.OUT, augment.OUT), mode="bilinear", align_corners=False)


def hand_removed(x_u8, m_u8):
    """Paint the hand out of each context crop (OpenCV inpainting over a slightly grown mask)."""
    out = []
    for img, m in zip(x_u8.permute(0, 2, 3, 1).numpy(), m_u8[:, 0].numpy()):
        hole = cv2.dilate((m > 64).astype(np.uint8), np.ones((7, 7), np.uint8))
        out.append(cv2.inpaint(np.ascontiguousarray(img), hole, 5, cv2.INPAINT_TELEA))
    return torch.from_numpy(np.stack(out)).permute(0, 3, 1, 2).contiguous()


@torch.no_grad()
def predict(net, inputs, batch=256):
    net.eval()
    return torch.cat([net(inputs[i:i + batch]) for i in range(0, len(inputs), batch)]).argmax(1)


def confusion_plot(y, pred, title, path):
    k = len(LABELS)
    cm = np.zeros((k, k), int)
    for t, p in zip(y.tolist(), pred.tolist()):
        cm[t, p] += 1
    fig, ax = plt.subplots(figsize=(5, 4.6))
    ax.imshow(cm, cmap="Blues")
    for i in range(k):
        for j in range(k):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=9)
    ax.set_xticks(range(k), LABELS)
    ax.set_yticks(range(k), LABELS)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title, loc="left", fontsize=10)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--no-bg-random", action="store_true", help="turn background randomization off (baseline)")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--name", default=None)
    a = p.parse_args()
    randomize = not a.no_bg_random
    name = a.name or ("randomized" if randomize else "baseline")
    out = HERE / "runs" / name
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(a.seed)
    random.seed(a.seed)
    matplotlib.rcParams.update({"font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 9})

    t0 = time.time()
    rows = load_rows()
    part = split_by_burst(rows)
    x, m, y, kept = load_crops(rows)
    split = np.array([part[r["id"]] for r in kept])
    idx = {s: torch.from_numpy(np.flatnonzero(split == s)) for s in ("train", "val", "test")}
    train_rows = [r for r in rows if part[r["id"]] == "train"]
    pool = background_patches(train_rows, 2000, seed=a.seed)
    val_swap = swapped_backgrounds(rows, y[idx["val"]], seed=11)
    test_swap = swapped_backgrounds(rows, y[idx["test"]], seed=22)
    test_removed = hand_removed(x[idx["test"]], m[idx["test"]])
    print(f"{len(kept)} hand crops: train {len(idx['train'])}, val {len(idx['val'])}, test {len(idx['test'])} "
          f"(whole bursts); background randomization {'ON' if randomize else 'OFF'}; loaded in {time.time() - t0:.0f} s")

    # what the network sees during training
    with torch.no_grad():
        ex = augment.train_batch(x[idx["train"][:16]], m[idx["train"][:16]], pool, randomize)
    fig, axes = plt.subplots(2, 8, figsize=(12, 3.3))
    for ax, im, yy in zip(axes.flat, ex, y[idx["train"][:16]]):
        ax.imshow((im * 0.25 + 0.5).clamp(0, 1).permute(1, 2, 0).numpy())
        ax.set_title(LABELS[yy], fontsize=9)
        ax.axis("off")
    fig.suptitle(f"Training inputs ({'with' if randomize else 'without'} background randomization)", x=0.01, ha="left", fontweight="bold")
    fig.savefig(out / "training_inputs.png", dpi=110, bbox_inches="tight")
    plt.close(fig)

    net = LetterNet()
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=5e-4)
    steps = a.epochs * ((len(idx["train"]) + a.batch - 1) // a.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps, pct_start=0.15)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.05)
    val_real_in = augment.eval_batch(x[idx["val"]])
    val_swap_in = augment.eval_batch(x[idx["val"]], m[idx["val"]], val_swap)
    hist, best = [], (-1, None, 0)
    for epoch in range(1, a.epochs + 1):
        net.train()
        order = idx["train"][torch.randperm(len(idx["train"]))]
        total, correct, seen = 0.0, 0, 0
        for i in range(0, len(order), a.batch):
            b = order[i:i + a.batch]
            inp = augment.train_batch(x[b], m[b], pool, randomize)
            logits = net(inp)
            loss = loss_fn(logits, y[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            total += loss.item() * len(b)
            correct += (logits.argmax(1) == y[b]).sum().item()
            seen += len(b)
        vr = (predict(net, val_real_in) == y[idx["val"]]).float().mean().item()
        vs = (predict(net, val_swap_in) == y[idx["val"]]).float().mean().item()
        hist.append({"epoch": epoch, "train_loss": total / seen, "train_acc": correct / seen, "val_real": vr, "val_swap": vs})
        score = (vr + vs) / 2
        if score > best[0]:
            best = (score, {k: v.clone() for k, v in net.state_dict().items()}, epoch)
        if True:                                                        # every epoch, for watch_training.py
            print(f"epoch {epoch:3d}  loss {total / seen:.3f}  train {correct / seen:.2f}  val real {vr:.2f}  val swapped bg {vs:.2f}"
                  f"   ({time.time() - t0:.0f} s)")
    net.load_state_dict(best[1])

    yt = y[idx["test"]]
    p_real = predict(net, augment.eval_batch(x[idx["test"]]))
    p_swap = predict(net, augment.eval_batch(x[idx["test"]], m[idx["test"]], test_swap))
    p_gone = predict(net, augment.eval_batch(test_removed))
    res = {
        "name": name, "background_randomization": randomize, "best_epoch": best[2],
        "test_real": (p_real == yt).float().mean().item(),
        "test_swapped_background": (p_swap == yt).float().mean().item(),
        "hand_removed_still_right": (p_gone == yt).float().mean().item(),
        "hand_removed_says_none": (p_gone == LABELS.index("none")).float().mean().item(),
        "n_train": len(idx["train"]), "n_val": len(idx["val"]), "n_test": len(idx["test"]), "history": hist,
    }
    (out / "metrics.json").write_text(json.dumps(res, indent=1))
    torch.save({"state_dict": net.state_dict(), "labels": LABELS, "input": augment.OUT, "eval_scale": augment.EVAL_SCALE}, out / "model.pt")
    confusion_plot(yt, p_real, f"{name}: test photos ({res['test_real']:.0%} right)", out / "confusion_test.png")
    confusion_plot(yt, p_swap, f"{name}: swapped backgrounds ({res['test_swapped_background']:.0%} right)", out / "confusion_swapped.png")
    print(f"\nbest epoch {best[2]}.  TEST on {len(yt)} photos from held-out bursts:")
    print(f"  real photos:             {res['test_real']:.1%} right")
    print(f"  swapped backgrounds:     {res['test_swapped_background']:.1%} right")
    print(f"  hand removed:            still names the letter {res['hand_removed_still_right']:.1%} of the time "
          f"(says 'none' {res['hand_removed_says_none']:.1%})")
    print(f"saved {out.relative_to(HERE)}  ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
