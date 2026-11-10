"""Can the letter be guessed WITHOUT looking at the hand? A check for background shortcuts.

If the photos of one letter were taken in their own stretch of time, anything that drifted meanwhile (camera
framing, where you sit, the light, the webcam's auto exposure) becomes a clue for that letter. A network can
learn the scene instead of the hand: it scores well on photos like the training ones and fails at the lock.

This script grays out the hand box in every photo, plus one box borrowed from each other letter (so the gray
shapes themselves say nothing about the letter), shrinks what is left to a 32x24 thumbnail, and trains a linear
classifier on it, testing on whole bursts it never saw. It also reports how much the box's position and shape
alone reveal, and the average brightness per letter. Writes figures/average_per_letter.png.

  ~14% correct (1 in 7)  -> the background carries no letter information: good
  well above that        -> the background alone gives the letter away: collect letters in shared setups

Run:  ..\\.venv\\Scripts\\python check_background_bias.py          (uses photos that have a box)
"""
import argparse
import json
import random
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

HERE = Path(__file__).resolve().parent
LABELS = ["C", "O", "D", "R", "I", "N", "none"]


def thumbnail(path, boxes):
    a = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255
    a = a.copy()
    for x0, y0, x1, y1 in boxes:                                   # gray out these boxes
        a[y0:y1, x0:x1] = 0.5
    small = Image.fromarray((a * 255).astype(np.uint8)).resize((32, 24), Image.BILINEAR)
    return np.asarray(small, dtype=np.float32).reshape(-1) / 255


def held_out_accuracy(X, y, groups, folds):
    """Linear classifier, k-fold over whole bursts (every label has bursts in every test fold where possible)."""
    accs = []
    for f in range(folds):
        test_groups = set()
        for l in range(len(LABELS)):
            gl = sorted({g for g, yy in zip(groups, y.tolist()) if yy == l})
            test_groups |= {gl[i] for i in range(len(gl)) if i % folds == f}
        te = torch.tensor([g in test_groups for g in groups])
        tr = ~te
        if te.sum() == 0 or tr.sum() == 0:
            continue
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
        torch.manual_seed(f)
        m = torch.nn.Linear(X.shape[1], len(LABELS))
        opt = torch.optim.Adam(m.parameters(), lr=1e-2, weight_decay=1e-3)
        for _ in range(300):
            opt.zero_grad()
            torch.nn.functional.cross_entropy(m((X[tr] - mu) / sd), y[tr]).backward()
            opt.step()
        accs.append((m((X[te] - mu) / sd).argmax(1) == y[te]).float().mean().item())
    return accs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=str(HERE / "dataset"))
    p.add_argument("--folds", type=int, default=5)
    a = p.parse_args()
    data = Path(a.data)
    rows = [json.loads(l) for l in (data / "samples.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    rows = [r for r in rows if r["box"]]                               # photos with a hand box
    y = torch.tensor([LABELS.index(r["label"]) for r in rows])
    groups = [(r["session"], r["burst"], r["label"]) for r in rows]
    print(f"{len(rows)} photos with a hand box, {len(set(groups))} bursts, {a.folds}-fold test on held-out bursts; "
          f"chance = {1 / len(LABELS):.0%}\n")
    rng = random.Random(0)
    by_label = {l: [r["box"] for r in rows if r["label"] == l] for l in LABELS}
    B = np.array([r["box"] for r in rows], dtype=float)
    w, h = B[:, 2] - B[:, 0], B[:, 3] - B[:, 1]
    geometry = torch.tensor(np.stack([B[:, 0] / 320, B[:, 1] / 240, w / 320, h / 240, w / h, w * h / 76800], 1), dtype=torch.float32)
    tests = {
        "box position and shape only": geometry,
        "background only (hand hidden)": torch.tensor(np.stack([thumbnail(
            data / "images" / f"{r['id']}.jpg", [r["box"]] + [rng.choice(by_label[l]) for l in LABELS if l != r["label"] and by_label[l]])
            for r in rows])),
        "whole photo": torch.tensor(np.stack([thumbnail(data / "images" / f"{r['id']}.jpg", []) for r in rows])),
    }
    for what, X in tests.items():
        accs = held_out_accuracy(X, y, groups, a.folds)
        print(f"{what:30s} {np.mean(accs):5.0%} correct   (folds {min(accs):.0%} to {max(accs):.0%})")

    print("\naverage brightness per label (0-255), in recording order:")
    order = sorted(LABELS, key=lambda l: min((r["time"] for r in rows if r["label"] == l), default="~"))
    means = {}
    for l in order:
        ims = [np.asarray(Image.open(data / "images" / f"{r['id']}.jpg").convert("RGB"), np.float32)
               for r in rows if r["label"] == l]
        if ims:
            means[l] = np.mean(ims, axis=0)
            print(f"  {l:5s} {means[l].mean():6.1f}")
    fig, axes = plt.subplots(1, len(means), figsize=(2.4 * len(means), 2.3))
    for ax, (l, m) in zip(np.atleast_1d(axes), means.items()):
        ax.imshow(m.astype(np.uint8))
        ax.set_title(f"{l}", fontsize=11)
        ax.axis("off")
    fig.suptitle("Average photo of each label, in recording order", x=0.01, ha="left", fontweight="bold")
    out = HERE / "figures" / "average_per_letter.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=110, bbox_inches="tight")
    print(f"\nsaved {out.relative_to(HERE)}")


if __name__ == "__main__":
    main()
