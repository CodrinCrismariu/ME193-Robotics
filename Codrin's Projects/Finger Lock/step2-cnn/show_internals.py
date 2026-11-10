"""Look inside the trained CNN: run one test image through it and save what every layer computes.

Outputs
  figures/feature_maps.png         one-page overview: image -> conv maps -> pooled maps -> deeper maps -> probabilities
  ../movie/assets/*.png            the same pictures, one file each, for the explainer movie
  ../movie/assets/cnn_movie_data.json   the actual numbers (kernels, one convolution worked out, one max pool,
                                        dense-layer outputs, probabilities, training history, Part 1's V data)

Run:  python show_internals.py            (default: the most confident correct "3 fingers" test image)
      python show_internals.py --index 7  (any test image by position)
"""
import argparse
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from model import FingerCNN
from train import augment, load

HERE = Path(__file__).resolve().parent
MOVIE = HERE.parent / "movie" / "assets"
STEP1_DATA = HERE.parent / "step1-v-fit" / "data" / "annacode-v.tsv"

PAPER = np.array([246, 241, 231]) / 255      # zero
WARM = np.array([232, 116, 59]) / 255        # positive response
DEEP = np.array([110, 40, 14]) / 255         # strongest response
COOL = np.array([46, 107, 91]) / 255         # negative (only before ReLU): the movie's mat green


def warm_rgb(m, vmax):
    """Map values >= 0 to paper -> orange -> deep brown (one hue family, light to dark)."""
    v = np.clip(m / (vmax + 1e-9), 0, 1)[..., None]
    low = PAPER + (WARM - PAPER) * np.clip(v * 2, 0, 1)
    high = WARM + (DEEP - WARM) * np.clip(v * 2 - 1, 0, 1)
    return np.where(v < 0.5, low, high)


def diverging_rgb(m, vmax):
    """Negative -> blue, zero -> paper, positive -> orange (same scale both ways)."""
    v = np.clip(m / (vmax + 1e-9), -1, 1)[..., None]
    pos = PAPER + (WARM - PAPER) * np.clip(v, 0, 1)
    neg = PAPER + (COOL - PAPER) * np.clip(-v, 0, 1)
    return np.where(v >= 0, pos, neg)


def save_png(rgb_or_gray, path, size):
    a = np.asarray(rgb_or_gray)
    if a.dtype != np.uint8:
        a = (np.clip(a, 0, 1) * 255).round().astype(np.uint8)
    img = Image.fromarray(a)
    img = img.resize((size, size) if img.width == img.height else
                     (size, round(size * img.height / img.width)), Image.Resampling.NEAREST)
    img.save(path, optimize=True)


def mosaic(maps, cols, tile, gap):
    """Grid of feature maps (each scaled to its own max), transparent gaps between tiles."""
    n, h, w = maps.shape
    rows = int(np.ceil(n / cols))
    scale = tile // h
    out = np.zeros((rows * tile + (rows - 1) * gap, cols * tile + (cols - 1) * gap, 4), dtype=np.uint8)
    for i, m in enumerate(maps):
        r, c = divmod(i, cols)
        rgb = (warm_rgb(m, m.max()) * 255).round().astype(np.uint8)
        rgb = np.repeat(np.repeat(rgb, scale, 0), scale, 1)
        y, x = r * (tile + gap), c * (tile + gap)
        out[y:y + tile, x:x + tile, :3] = rgb
        out[y:y + tile, x:x + tile, 3] = 255
    return out


def run_layers(net, x):
    """Forward pass that keeps every intermediate result."""
    conv1, relu1, pool1 = net.features[0]
    out = {"c1_pre": conv1(x)}
    out["c1_relu"] = relu1(out["c1_pre"])
    out["c1_pool"] = pool1(out["c1_relu"])
    h = out["c1_pool"]
    for i in (1, 2, 3):
        h = net.features[i](h)
        out[f"b{i + 1}_pool"] = h
    flat, dense, relu, _drop, last = net.classifier
    out["flat"] = flat(h)
    out["dense"] = relu(dense(out["flat"]))
    out["logits"] = last(out["dense"])
    out["probs"] = out["logits"].softmax(1)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--index", type=int, default=None, help="test image position (default: best '3')")
    p.add_argument("--kernel", type=int, default=None, help="conv-1 kernel for the worked example (default: most edge-like)")
    a = p.parse_args()

    ckpt = torch.load(HERE / "finger_cnn.pt", weights_only=False)
    net = FingerCNN(n_classes=len(ckpt["classes"]))
    net.load_state_dict(ckpt["state_dict"])
    net.eval()
    x, y, classes, mean, std = load()
    raw = np.load(HERE / "data" / "fingers_64.npz")["x_test"]

    with torch.no_grad():
        probs_all = net(x["test"]).softmax(1)
        if a.index is None:
            target = classes.index(3)
            ok = (probs_all.argmax(1) == y["test"]) & (y["test"] == target)
            idx = int(torch.where(ok, probs_all[:, target], torch.tensor(-1.0)).argmax())
        else:
            idx = a.index
        xi = x["test"][idx:idx + 1]
        L = {k: v[0].numpy() for k, v in run_layers(net, xi).items()}
    label = classes[int(y["test"][idx])]
    pred = classes[int(L["probs"].argmax())]
    print(f"test image {idx}: {label} fingers, predicted {pred} "
          f"({', '.join(f'{c}: {q:.1%}' for c, q in zip(classes, L['probs']))})")

    W = net.features[0][0].weight.detach().numpy()[:, 0]      # (8, 3, 3)
    B = net.features[0][0].bias.detach().numpy()

    # one convolution step worked out, for the most edge-like kernel (weights that nearly cancel,
    # so flat areas give ~0 and only changes in brightness respond), at its strongest response
    edginess = np.abs(W.sum(axis=(1, 2))) / np.abs(W).sum(axis=(1, 2))
    k = int(edginess.argmin()) if a.kernel is None else a.kernel
    print("kernel weight sums / total |w|:", np.round(edginess, 2).tolist(), "-> using kernel", k)
    r, c = np.unravel_index(L["c1_relu"][k][1:-1, 1:-1].argmax(), (62, 62))
    r, c = int(r) + 1, int(c) + 1                               # keep the 3x3 patch inside the image
    patch = xi[0, 0, r - 1:r + 2, c - 1:c + 2].numpy()
    products = patch * W[k]
    conv_sum = float(products.sum() + B[k])
    assert abs(conv_sum - L["c1_pre"][k, r, c]) < 1e-4

    # one max pool worked out: the 4x4 block (2x2 pool windows) around that spot
    r0, c0 = (r // 2) * 2 - 2, (c // 2) * 2 - 2
    r0, c0 = int(np.clip(r0, 0, 60)), int(np.clip(c0, 0, 60))
    block = L["c1_relu"][k, r0:r0 + 4, c0:c0 + 4]
    pooled = block.reshape(2, 2, 2, 2).max(axis=(1, 3))
    assert np.allclose(pooled, L["c1_pool"][k, r0 // 2:r0 // 2 + 2, c0 // 2:c0 // 2 + 2])

    # an 8x8 crop of raw pixel values around the same spot
    pr0, pc0 = int(np.clip(r - 4, 0, 56)), int(np.clip(c - 4, 0, 56))
    crop = raw[idx, pr0:pr0 + 8, pc0:pc0 + 8]

    # ── movie assets ─────────────────────────────────────────────────────────
    MOVIE.mkdir(parents=True, exist_ok=True)
    save_png(raw[idx], MOVIE / "input.png", 512)
    torch.manual_seed(3)
    aug = augment(xi.repeat(6, 1, 1, 1))[:, 0].numpy() * std + mean
    for i, im in enumerate(aug):
        save_png((np.clip(im, 0, 1) * 255).round().astype(np.uint8), MOVIE / f"aug_{i}.png", 256)
    vmax1 = float(np.abs(L["c1_pre"][k]).max())
    save_png(diverging_rgb(L["c1_pre"][k], vmax1), MOVIE / "c1_pre_k.png", 512)
    save_png(diverging_rgb(L["c1_relu"][k], vmax1), MOVIE / "c1_relu_k.png", 512)
    for i in range(8):
        save_png(warm_rgb(L["c1_relu"][i], L["c1_relu"][i].max()), MOVIE / f"c1_relu_{i}.png", 256)
        save_png(warm_rgb(L["c1_pool"][i], L["c1_pool"][i].max()), MOVIE / f"c1_pool_{i}.png", 256)
    Image.fromarray(mosaic(L["b2_pool"], 4, 64, 6)).save(MOVIE / "b2_mosaic.png", optimize=True)
    Image.fromarray(mosaic(L["b3_pool"], 8, 64, 6)).save(MOVIE / "b3_mosaic.png", optimize=True)
    Image.fromarray(mosaic(L["b4_pool"], 8, 64, 6)).save(MOVIE / "b4_mosaic.png", optimize=True)

    hist = json.loads((HERE / "history.json").read_text())
    v = np.loadtxt(STEP1_DATA, skiprows=1, delimiter="\t")
    rd = lambda arr, n=3: np.round(np.asarray(arr, dtype=float), n).tolist()
    data = {
        "test_index": idx, "label": label, "pred": pred, "classes": classes,
        "probs": rd(L["probs"], 4), "logits": rd(L["logits"]),
        "pixels": {"r0": pr0, "c0": pc0, "values": crop.astype(int).tolist()},
        "kernels": rd(W), "kernel_bias": rd(B),
        "conv": {"k": k, "r": r, "c": c, "patch": rd(patch), "weights": rd(W[k]), "bias": round(float(B[k]), 3),
                 "products": rd(products), "out_pre": round(conv_sum, 3), "out_relu": round(max(conv_sum, 0), 3)},
        "pool": {"k": k, "r0": r0, "c0": c0, "block": rd(block), "out": rd(pooled)},
        "dense": rd(L["dense"]),
        "shapes": {"input": [1, 64, 64], "b1": [8, 32, 32], "b2": [16, 16, 16], "b3": [32, 8, 8], "b4": [32, 4, 4],
                   "flat": 512, "dense": 64, "out": 5},
        "params": int(sum(q.numel() for q in net.parameters())),
        "history": {"train_acc": hist["train_acc"], "val_acc": hist["val_acc"], "best_epoch": hist["best_epoch"],
                    "test_acc": round(hist["test_acc"], 4)},
        "v": {"x": rd(v[:, 1], 2), "y": rd(v[:, 2], 2), "line": [0.1771, 27.8102],
              "net": {"c": 76.617, "hinges": [[22.08, -2.017], [60.03, 4.020]]}},
    }
    (MOVIE / "cnn_movie_data.json").write_text(json.dumps(data))
    print(f"kernel {k} strongest at row {r}, col {c}: sum = {conv_sum:.3f} -> ReLU {max(conv_sum, 0):.3f}")
    print(f"max pool block at ({r0},{c0}): {np.round(block, 2).tolist()} -> {np.round(pooled, 2).tolist()}")
    print(f"saved {len(list(MOVIE.glob('*.png')))} images + cnn_movie_data.json to {MOVIE}")

    # ── one-page overview for Notion ────────────────────────────────────────
    matplotlib.rcParams.update({"font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 9,
                                "savefig.dpi": 200, "savefig.bbox": "tight"})
    fig = plt.figure(figsize=(13, 5.6))
    gs = fig.add_gridspec(4, 11, wspace=0.25, hspace=0.35)
    ax = fig.add_subplot(gs[:, 0:2])
    ax.imshow(raw[idx], cmap="gray")
    ax.set_title(f"input 64×64\n(true: {label} fingers)", fontsize=9)
    ax.axis("off")
    for i in range(8):
        a1 = fig.add_subplot(gs[i // 2, 2 + i % 2])
        a1.imshow(warm_rgb(L["c1_relu"][i], L["c1_relu"][i].max()))
        a1.axis("off")
        a2 = fig.add_subplot(gs[i // 2, 4 + i % 2])
        a2.imshow(warm_rgb(L["c1_pool"][i], L["c1_pool"][i].max()))
        a2.axis("off")
        if i == 0:
            a1.set_title("conv 1 + ReLU\n8 maps, 64×64", fontsize=9, loc="left")
            a2.set_title("max pool\n8 maps, 32×32", fontsize=9, loc="left")
    for col, key, cols, title in ((6, "b2_pool", 4, "block 2\n16 × 16×16"), (7, "b3_pool", 8, "block 3\n32 × 8×8"),
                                  (8, "b4_pool", 8, "block 4\n32 × 4×4")):
        axm = fig.add_subplot(gs[:, col])
        axm.imshow(mosaic(L[key], cols if key == "b2_pool" else 4, 64, 6))
        axm.set_title(title, fontsize=9)
        axm.axis("off")
    axp = fig.add_subplot(gs[1:3, 9:11])
    bars = axp.barh([str(cn) for cn in classes], L["probs"], color=["#c3c2b7"] * 5)
    bars[int(L["probs"].argmax())].set_color("#e8743b")
    axp.set_xlim(0, 1)
    axp.set_title("softmax: probability of\n1-5 fingers", fontsize=9, loc="left")
    for s in ("top", "right"):
        axp.spines[s].set_visible(False)
    for b_, q in zip(bars, L["probs"]):
        axp.text(min(q + 0.02, 0.8), b_.get_y() + b_.get_height() / 2, f"{q:.1%}", va="center", fontsize=8)
    fig.suptitle("Inside the finger CNN: what each layer computes for one test image "
                 "(each map shaded from 0 = paper to its own max = dark)", x=0.12, ha="left", fontsize=11,
                 fontweight="bold")
    (HERE / "figures").mkdir(exist_ok=True)
    fig.savefig(HERE / "figures" / "feature_maps.png")
    print("saved figures/feature_maps.png")


if __name__ == "__main__":
    main()
