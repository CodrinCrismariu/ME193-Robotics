"""Check the collected dataset: counts, suspicious boxes, and a contact sheet of photos with their hand boxes.

Run:  python check_dataset.py               (reads dataset/, writes figures/dataset_overview.png)
      python check_dataset.py --per-label 10
"""
import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from PIL import Image

HERE = Path(__file__).resolve().parent
LABELS = ["C", "O", "D", "R", "I", "N", "none"]
IMG_W, IMG_H = 320, 240
ORANGE = "#e8743b"


def box_frac(b):
    return 0.0 if b is None else (b[2] - b[0]) * (b[3] - b[1]) / (IMG_W * IMG_H)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=str(HERE / "dataset"))
    p.add_argument("--per-label", type=int, default=8)
    a = p.parse_args()
    data = Path(a.data)
    index = data / "samples.jsonl"
    if not index.exists():
        print(f"No samples yet in {data}. Run collect_server.py and capture some first.")
        return
    rows = [json.loads(l) for l in index.read_text(encoding="utf-8").splitlines() if l.strip()]
    print(f"{len(rows)} samples in {data}\n")

    by_label = defaultdict(list)
    for r in rows:
        by_label[r["label"]].append(r)
    print(f"{'label':6s} {'count':>5s} {'checked':>7s} {'no hand':>7s} {'sessions':>8s} {'bursts':>6s} {'median box':>10s}")
    for l in LABELS:
        rs = by_label.get(l, [])
        if not rs:
            print(f"{l:6s} {0:5d}")
            continue
        sizes = [box_frac(r["box"]) for r in rs if r["box"]]
        med = f"{np.median(sizes) * 100:9.1f}%" if sizes else f"{'-':>10s}"
        print(f"{l:6s} {len(rs):5d} {sum(bool(r.get('checked')) for r in rs):7d} {sum(r['box'] is None for r in rs):7d} "
              f"{len({r['session'] for r in rs}):8d} {len({(r['session'], r['burst']) for r in rs}):6d} {med}")

    problems = []
    for r in rows:
        if not (data / "images" / f"{r['id']}.jpg").exists():
            problems.append((r["id"], "image file missing"))
        elif not r.get("checked"):
            problems.append((r["id"], "box not checked yet (open label.html)"))
        elif r["label"] != "none" and r["box"] is None:
            problems.append((r["id"], "a letter with no hand box"))
        elif r["box"] and box_frac(r["box"]) > 0.7:
            problems.append((r["id"], f"box covers {box_frac(r['box']):.0%} of the frame: the background probably changed"))
        elif r["box"] and box_frac(r["box"]) < 0.01:
            problems.append((r["id"], "tiny box: probably noise, not a hand"))
    print(f"\n{len(problems)} samples to look at" + (":" if problems else "."))
    for sid, why in problems[:30]:
        print(f"  {sid}: {why}")
    if len(problems) > 30:
        print(f"  ... and {len(problems) - 30} more")
    print(f"\nsessions: {dict(Counter(r['session'] for r in rows))}")

    # contact sheet
    matplotlib.rcParams.update({"font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 9})
    rng = random.Random(0)
    k = a.per_label
    fig, axes = plt.subplots(len(LABELS), k, figsize=(1.7 * k, 1.45 * len(LABELS)), squeeze=False)
    for i, l in enumerate(LABELS):
        rs = by_label.get(l, [])
        pick = rng.sample(rs, min(k, len(rs)))
        for j in range(k):
            ax = axes[i, j]
            ax.axis("off")
            if j < len(pick):
                r = pick[j]
                try:
                    ax.imshow(Image.open(data / "images" / f"{r['id']}.jpg").convert("RGB"))
                    if r["box"]:
                        x0, y0, x1, y1 = r["box"]
                        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, lw=1.6, ec=ORANGE))
                except FileNotFoundError:
                    pass
            if j == 0:
                ax.text(-0.08, 0.5, f"{l}\n{len(rs)}", transform=ax.transAxes, ha="right", va="center", fontsize=11, fontweight="bold")
    fig.suptitle("Dataset overview: random photos per label, hand box in orange", x=0.02, ha="left", fontweight="bold")
    out = HERE / "figures" / "dataset_overview.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nsaved {out.relative_to(HERE)}")


if __name__ == "__main__":
    main()
