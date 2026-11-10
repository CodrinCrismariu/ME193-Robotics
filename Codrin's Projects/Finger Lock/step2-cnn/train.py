"""Train the small CNN in model.py to count fingers (1-5) on the prepared Sign Language Digits images.

Run:  python prepare_data.py     (once)
      python train.py            (about a minute or two on a laptop CPU)

Saves finger_cnn.pt (the trained weights plus what is needed to use them) and three figures:
figures/training_curves.png, figures/confusion_matrix.png, figures/test_mistakes.png
"""
import argparse
import json
import math
import time
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.colors import LinearSegmentedColormap
from torch import nn

from model import FingerCNN

HERE = Path(__file__).resolve().parent
DATA = HERE / "data" / "fingers_64.npz"
FIGS = HERE / "figures"
MODEL_OUT = HERE / "finger_cnn.pt"
BLUE, ORANGE, INK, INK_2, MUTED, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"


# ── data ──────────────────────────────────────────────────────────────────────

def load():
    d = np.load(DATA)
    to_tensor = lambda a: torch.tensor(a, dtype=torch.float32).unsqueeze(1) / 255.0   # (N, 1, 64, 64) in [0, 1]
    x = {k: to_tensor(d[f"x_{k}"]) for k in ("train", "val", "test")}
    y = {k: torch.tensor(d[f"y_{k}"]) for k in ("train", "val", "test")}
    mean, std = x["train"].mean().item(), x["train"].std().item()      # from the training set only
    for k in x:
        x[k] = (x[k] - mean) / std                                       # same scaling idea as step 1
    return x, y, d["classes"].tolist(), mean, std


def augment(x, max_rot=15, max_shift=0.10, max_zoom=0.10, max_light=0.2):
    """Random small changes so the network sees a 'new' version of every image each epoch.

    One affine transform per image (rotate +-15 deg, shift +-10%, zoom +-10%, 50% mirror flip),
    then a random brightness/contrast change. A mirrored hand still shows the same number of
    fingers, so the flip also teaches it left hands.
    """
    n = x.shape[0]
    angle = (torch.rand(n) * 2 - 1) * math.radians(max_rot)
    zoom = 1 + (torch.rand(n) * 2 - 1) * max_zoom
    flip = torch.where(torch.rand(n) < 0.5, -1.0, 1.0)
    shift = (torch.rand(n, 2) * 2 - 1) * max_shift * 2                 # grid coords span -1..1
    cos, sin = torch.cos(angle) / zoom, torch.sin(angle) / zoom
    theta = torch.stack([torch.stack([cos * flip, -sin, shift[:, 0]], 1),
                         torch.stack([sin * flip, cos, shift[:, 1]], 1)], 1)   # (n, 2, 3)
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    x = F.grid_sample(x, grid, padding_mode="border", align_corners=False)
    contrast = 1 + (torch.rand(n, 1, 1, 1) * 2 - 1) * max_light
    brightness = (torch.rand(n, 1, 1, 1) * 2 - 1) * max_light
    return x * contrast + brightness


# ── training ──────────────────────────────────────────────────────────────────

def evaluate(net, x, y, batch=256):
    """Loss, accuracy and predictions on a whole set, with dropout off and no gradients."""
    net.eval()
    with torch.no_grad():
        logits = torch.cat([net(x[i:i + batch]) for i in range(0, len(x), batch)])
    loss = F.cross_entropy(logits, y).item()
    pred = logits.argmax(1)
    return loss, (pred == y).float().mean().item(), pred, logits.softmax(1)


def train(net, x, y, epochs, lr, batch):
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()           # softmax + "how surprised were we by the right answer"
    hist = {k: [] for k in ("train_loss", "train_acc", "val_loss", "val_acc")}
    best_key, best_state, best_epoch = None, None, 0
    n = len(x["train"])
    for epoch in range(1, epochs + 1):
        net.train()                            # dropout on
        order = torch.randperm(n)
        total_loss, correct = 0.0, 0
        for i in range(0, n, batch):           # mini-batches of `batch` images
            idx = order[i:i + batch]
            xb, yb = augment(x["train"][idx]), y["train"][idx]
            logits = net(xb)                   # forward pass
            loss = loss_fn(logits, yb)
            opt.zero_grad()
            loss.backward()                    # backpropagation (autograd)
            opt.step()                         # gradient step (Adam)
            total_loss += loss.item() * len(idx)
            correct += (logits.argmax(1) == yb).sum().item()
        val_loss, val_acc, _, _ = evaluate(net, x["val"], y["val"])
        hist["train_loss"].append(total_loss / n)
        hist["train_acc"].append(correct / n)
        hist["val_loss"].append(val_loss)
        hist["val_acc"].append(val_acc)
        key = (val_acc, -val_loss)             # best accuracy on unseen images; ties -> lower loss
        if best_key is None or key > best_key:
            best_key, best_epoch = key, epoch
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            print(f"epoch {epoch:3d}   train loss {hist['train_loss'][-1]:.3f}  acc {hist['train_acc'][-1]:.3f}"
                  f"   val loss {val_loss:.3f}  acc {val_acc:.3f}")
    net.load_state_dict(best_state)
    return hist, best_epoch, best_key[0]


# ── plots ─────────────────────────────────────────────────────────────────────

def style():
    matplotlib.rcParams.update({
        "font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 10, "axes.facecolor": "#fcfcfb",
        "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK_2, "axes.titlesize": 11,
        "axes.titleweight": "bold", "axes.grid": True, "grid.color": GRID, "axes.spines.top": False,
        "axes.spines.right": False, "xtick.color": MUTED, "ytick.color": MUTED, "legend.frameon": False,
        "savefig.dpi": 200, "savefig.bbox": "tight",
    })


def save(fig, name):
    FIGS.mkdir(exist_ok=True)
    fig.savefig(FIGS / name)
    plt.close(fig)
    print(f"saved figures/{name}")


def plot_curves(hist, best_epoch):
    ep = np.arange(1, len(hist["train_loss"]) + 1)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.8))
    for ax, key, title in ((a1, "loss", "Cross-entropy loss"), (a2, "acc", "Accuracy")):
        ax.plot(ep, hist[f"train_{key}"], color=BLUE, lw=2, label="training (augmented)")
        ax.plot(ep, hist[f"val_{key}"], color=ORANGE, lw=2, label="validation (unseen)")
        ax.axvline(best_epoch, color=MUTED, lw=1, ls=(0, (3, 3)))
        ax.set_title(title, loc="left")
        ax.set_xlabel("epoch")
    a2.text(best_epoch, 0.05, f" best epoch {best_epoch}", color=INK_2, fontsize=9,
            transform=a2.get_xaxis_transform())
    a2.set_ylim(0, 1.02)
    a1.legend()
    save(fig, "training_curves.png")


def plot_confusion(y_true, y_pred, classes):
    k = len(classes)
    cm = np.zeros((k, k), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    cmap = LinearSegmentedColormap.from_list("blues", ["#f4f8fd", "#86b6ef", "#2a78d6", "#104281"])
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    ax.imshow(cm, cmap=cmap, vmin=0)
    ax.grid(False)
    for t in range(k):
        for p in range(k):
            ax.text(p, t, cm[t, p], ha="center", va="center", fontsize=10,
                    color="white" if cm[t, p] > cm.max() * 0.5 else INK)
    ax.set_xticks(range(k), classes)
    ax.set_yticks(range(k), classes)
    ax.set_xlabel("predicted fingers")
    ax.set_ylabel("true fingers")
    ax.set_title(f"Test set: {np.trace(cm)}/{cm.sum()} correct", loc="left")
    save(fig, "confusion_matrix.png")


def plot_mistakes(x_raw, y_true, y_pred, probs, classes, limit=16):
    wrong = np.flatnonzero(y_true != y_pred)[:limit]
    if len(wrong) == 0:
        print("no test mistakes to plot")
        return
    cols = min(8, len(wrong))
    rows = math.ceil(len(wrong) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(max(4.0, 1.6 * cols), 2.0 * rows + 0.4), squeeze=False,
                             layout="constrained")
    for ax in axes.flat:
        ax.axis("off")
    for ax, i in zip(axes.flat, wrong):
        ax.imshow(x_raw[i], cmap="gray", vmin=0, vmax=255)
        ax.set_title(f"true {classes[y_true[i]]}, said {classes[y_pred[i]]}\n({probs[i, y_pred[i]]:.0%} sure)",
                     fontsize=8, fontweight="normal", color=INK_2)
    fig.suptitle("Test images the CNN got wrong", x=0.02, ha="left", fontweight="bold")
    save(fig, "test_mistakes.png")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    torch.manual_seed(a.seed)

    x, y, classes, mean, std = load()
    print(f"{len(x['train'])} training, {len(x['val'])} validation, {len(x['test'])} test images; "
          f"classes {classes}")
    net = FingerCNN(n_classes=len(classes))
    print(f"{sum(p.numel() for p in net.parameters()):,} parameters")

    t0 = time.time()
    hist, best_epoch, best_val = train(net, x, y, a.epochs, a.lr, a.batch)
    print(f"trained in {time.time() - t0:.0f} s; best validation accuracy {best_val:.3f} at epoch {best_epoch}")

    test_loss, test_acc, pred, probs = evaluate(net, x["test"], y["test"])
    print(f"TEST accuracy {test_acc:.3f}  ({(pred == y['test']).sum().item()}/{len(pred)})")

    torch.save({"state_dict": net.state_dict(), "classes": classes, "img_size": 64,
                "mean": mean, "std": std}, MODEL_OUT)
    print(f"saved {MODEL_OUT.name}")
    (HERE / "history.json").write_text(json.dumps(
        {"best_epoch": best_epoch, "test_acc": test_acc, **{k: [round(v, 4) for v in vals] for k, vals in hist.items()}}))

    style()
    plot_curves(hist, best_epoch)
    plot_confusion(y["test"].numpy(), pred.numpy(), classes)
    plot_mistakes(np.load(DATA)["x_test"], y["test"].numpy(), pred.numpy(), probs.numpy(), classes)


if __name__ == "__main__":
    main()
