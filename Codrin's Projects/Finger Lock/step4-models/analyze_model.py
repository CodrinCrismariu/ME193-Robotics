"""Look inside the trained networks: what the convolutions learned, where the classifier looks, which neurons
respond to which hand shapes, how the features change with training, and what one training step does.

Run from this folder (after training):   ..\\.venv\\Scripts\\python analyze_model.py
Writes ../movie2/assets/ (pictures + analysis.json for the explainer film) and figures/inside_*.png.

What it measures, all on the held-out test bursts:
  kernels      the 24 first-layer 3x3 color filters, and whether each prefers hand or background pixels
  feature maps what every block outputs for one test crop
  Grad-CAM     where each classifier looks when it names the letter, and what share of that falls on the hand
               (hand outline from step3-dataset/make_hand_masks.py), for the baseline and the randomized model
  channels     the last conv layer's 160 channels: which ones fire mostly for one letter (top-20 crops)
  embedding    the 160 numbers before the last layer, shrunk to 2-D with PCA: untrained vs trained network,
               and how well letters separate (nearest-centroid accuracy)
  first step   one real training step from random weights: loss, gradient size per layer, loss after the step
  detector     its 12 x 9 grid of hand scores on a hand, on a "none" sign, and on an empty crop with a face
"""
import json
import random
from collections import Counter
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image

import augment
from data import LABELS, background_patches, context_crop, load_crops, load_rows, split_by_burst
from detector import GRID_H, GRID_W, HandDetector, decode
from models import LetterNet
from train_detector import SRC_H, SRC_W, empty_view, empty_windows, load_frames, view

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "movie2" / "assets"
FIG = HERE / "figures"
TEAL = LinearSegmentedColormap.from_list("teal", ["#10141B", "#1d5f5a", "#34D1BF", "#F2EEE6"])
CORAL = np.array([255, 106, 61]) / 255


def load_clf(name):
    net = LetterNet()
    net.load_state_dict(torch.load(HERE / "runs" / name / "model.pt", weights_only=False)["state_dict"])
    return net.eval()


def blocks(net, inp):
    outs, h = [], inp
    for b in net.features:
        h = b(h)
        outs.append(h)
    return outs


def to_img(t):
    """normalized (3,H,W) tensor -> HxWx3 float in [0,1]"""
    return (t * 0.25 + 0.5).clamp(0, 1).permute(1, 2, 0).numpy()


def save_rgb(arr, path, scale=2):
    im = Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8))
    if scale != 1:
        im = im.resize((im.width * scale, im.height * scale), Image.BILINEAR)
    im.save(path, optimize=True)


def mosaic(maps, cols, tile, gap=6):
    """(K,h,w) maps -> RGBA mosaic, each map shaded dark -> teal -> white by its own max"""
    k = len(maps)
    rows = int(np.ceil(k / cols))
    out = np.zeros((rows * tile + (rows - 1) * gap, cols * tile + (cols - 1) * gap, 4))
    for i, mp in enumerate(maps):
        r, c = divmod(i, cols)
        v = mp / (mp.max() + 1e-6)
        rgb = TEAL(np.asarray(Image.fromarray((v * 255).astype(np.uint8)).resize((tile, tile), Image.NEAREST)) / 255)[..., :3]
        out[r * (tile + gap):r * (tile + gap) + tile, c * (tile + gap):c * (tile + gap) + tile, :3] = rgb
        out[r * (tile + gap):r * (tile + gap) + tile, c * (tile + gap):c * (tile + gap) + tile, 3] = 1
    return out


def grad_cam(net, inp, cls):
    """Grad-CAM on the last conv block (160 x 6 x 6), upsampled to the input size. Returns (N,96,96) in [0,1]."""
    feats = blocks(net, inp)[-1].detach().requires_grad_(True)
    logits = net.head(feats)
    logits.gather(1, cls[:, None]).sum().backward()
    w = feats.grad.mean((2, 3), keepdim=True)
    cam = F.relu((w * feats).sum(1, keepdim=True)).detach()
    cam = F.interpolate(cam, size=inp.shape[2:], mode="bilinear", align_corners=False)[:, 0]
    return cam / (cam.flatten(1).max(1).values[:, None, None] + 1e-9)


def heat_overlay(img, cam):
    a = (cam ** 0.8 * 0.75)[..., None]
    return img * 0.65 * (1 - a) + CORAL * a


def gap_features(net, inp):
    return F.adaptive_avg_pool2d(blocks(net, inp)[-1], 1).flatten(1)


def pca2(feats):
    z = feats - feats.mean(0)
    _, _, v = torch.linalg.svd(z, full_matrices=False)
    p = z @ v[:2].T
    return p / p.abs().max(0).values


def nearest_centroid_acc(f_train, y_train, f_test, y_test):
    cent = torch.stack([f_train[y_train == k].mean(0) for k in range(len(LABELS))])
    return (torch.cdist(f_test, cent).argmin(1) == y_test).float().mean().item()


@torch.no_grad()
def main():
    torch.manual_seed(0)
    random.seed(0)
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(exist_ok=True)
    rows = load_rows()
    part = split_by_burst(rows)
    x, m, y, kept = load_crops(rows)
    split = np.array([part[r["id"]] for r in kept])
    te = torch.from_numpy(np.flatnonzero(split == "test"))
    va = torch.from_numpy(np.flatnonzero(split == "val"))
    tr = torch.from_numpy(np.flatnonzero(split == "train"))
    clf, base = load_clf("randomized_200"), load_clf("baseline")
    inp_te = augment.eval_batch(x[te])
    _, mask_te = augment.geometry(x[te].float() / 255, m[te].float() / 255, train=False)
    mask_te = (mask_te[:, 0] > 0.5).float()
    yt = y[te]
    res = {"labels": LABELS, "n_test": len(te)}

    # 1. sample crops, one confident correct test crop per label
    prob = clf(inp_te).softmax(1)
    pick = {}
    for k, l in enumerate(LABELS):
        ok = torch.nonzero((yt == k) & (prob.argmax(1) == k)).flatten()
        j = int(ok[prob[ok, k].argmax()]) if len(ok) else int(torch.nonzero(yt == k)[0])
        pick[l] = j
        save_rgb(to_img(inp_te[j]), OUT / f"crop_{l}.png")
    res["crop_probs"] = {l: round(float(prob[j, LABELS.index(l)]), 3) for l, j in pick.items()}

    # 2. first-layer kernels
    W1 = clf.features[0][0].weight.detach()                       # (24,3,3,3)
    tiles = []
    for k in range(W1.shape[0]):
        w = W1[k].permute(1, 2, 0).numpy()
        tiles.append((w - w.min()) / (w.max() - w.min() + 1e-9))
    cell, gap = 30, 8
    grid = np.zeros((4 * 3 * cell + 3 * gap, 6 * 3 * cell + 5 * gap, 4))
    for k, t in enumerate(tiles):
        r, c = divmod(k, 6)
        big = np.repeat(np.repeat(t, cell, 0), cell, 1)
        y0, x0 = r * (3 * cell + gap), c * (3 * cell + gap)
        grid[y0:y0 + 3 * cell, x0:x0 + 3 * cell, :3] = big
        grid[y0:y0 + 3 * cell, x0:x0 + 3 * cell, 3] = 1
    Image.fromarray((grid * 255).astype(np.uint8)).save(OUT / "kernels.png", optimize=True)
    b1 = blocks(clf, inp_te)[0]                                    # (N,24,48,48) after ReLU + pool
    m48 = F.interpolate(mask_te[:, None], size=b1.shape[2:], mode="area")[:, 0]
    hand_mean = (b1 * m48[:, None]).sum((0, 2, 3)) / (m48.sum() + 1e-9)
    bg_mean = (b1 * (1 - m48[:, None])).sum((0, 2, 3)) / ((1 - m48).sum() + 1e-9)
    spatial = W1.mean(1).flatten(1).var(1)                          # pattern across the 3x3 window
    colour = W1.mean((2, 3)).var(1)                                 # contrast between R, G and B
    res["kernels"] = {"count": int(W1.shape[0]), "prefer_hand": int((hand_mean > bg_mean).sum()),
                      "color_kernels": int((colour > spatial).sum()), "pattern_kernels": int((colour <= spatial).sum()),
                      "hand_vs_bg_ratio": [round(float(h / (b + 1e-9)), 2) for h, b in zip(hand_mean, bg_mean)]}
    # the most hand-loving and the most background-loving filter, on the I crop: weights + what each one outputs
    ratio = hand_mean / (bg_mean + 1e-9)
    j = pick["I"]
    conv1 = clf.features[0][0](inp_te[j:j + 1])                    # (1,24,96,96) raw filter output before BatchNorm
    for tag, k in (("hand", int(ratio.argmax())), ("bg", int(ratio.argmin()))):
        mp = F.relu(clf.features[0][1](conv1))[0, k].numpy()       # after BatchNorm + ReLU: what the next layer receives
        Image.fromarray((TEAL(mp / (mp.max() + 1e-9))[..., :3] * 255).astype(np.uint8)).resize((192, 192), Image.NEAREST).save(OUT / f"k1_{tag}_map.png")
        res["kernels"][f"{tag}_kernel"] = {"index": k, "ratio": round(float(ratio[k]), 2),
                                           "weights": [[[round(float(W1[k, c, r, q]), 3) for q in range(3)] for r in range(3)] for c in range(3)]}
        t = tiles[k]
        Image.fromarray((np.repeat(np.repeat(t, 64, 0), 64, 1) * 255).astype(np.uint8)).save(OUT / f"k1_{tag}.png")
    save_rgb(to_img(inp_te[j]), OUT / "k1_input.png")

    # 3. feature maps of one crop (the I) through every block
    j = pick["I"]
    outs = blocks(clf, inp_te[j:j + 1])
    shapes = []
    for bi, (o, n_show, cols) in enumerate(zip(outs, (6, 6, 8, 8, 8), (3, 3, 4, 4, 4))):
        o = o[0]
        top = o.mean((1, 2)).argsort(descending=True)[:n_show]
        Image.fromarray((mosaic(o[top].numpy(), cols, 96) * 255).astype(np.uint8)).save(OUT / f"fmap_b{bi + 1}.png", optimize=True)
        shapes.append(list(o.shape))
    res["block_shapes"] = shapes

    # 4. Grad-CAM: where each model looks, and what share of it falls on the hand
    with torch.enable_grad():
        cam_r = torch.cat([grad_cam(clf, inp_te[i:i + 64], yt[i:i + 64]) for i in range(0, len(te), 64)])
        cam_b = torch.cat([grad_cam(base, inp_te[i:i + 64], yt[i:i + 64]) for i in range(0, len(te), 64)])
    on_hand = lambda cam: ((cam * mask_te).flatten(1).sum(1) / (cam.flatten(1).sum(1) + 1e-9))
    hr, hb = on_hand(cam_r), on_hand(cam_b)
    res["gradcam"] = {"randomized_on_hand": round(float(hr.mean()), 3), "baseline_on_hand": round(float(hb.mean()), 3),
                      "hand_area": round(float(mask_te.mean()), 3),
                      "randomized_more_on_hand_share": round(float((hr > hb).float().mean()), 3)}
    show = []
    for l in ("C", "O", "R", "N"):
        k = LABELS.index(l)
        idx = torch.nonzero(yt == k).flatten()
        med = hr[idx].median()
        j = int(idx[(hr[idx] - med).abs().argmin()])
        img = to_img(inp_te[j])
        save_rgb(heat_overlay(img, cam_r[j].numpy()), OUT / f"cam_rand_{l}.png")
        save_rgb(heat_overlay(img, cam_b[j].numpy()), OUT / f"cam_base_{l}.png")
        show.append({"label": l, "rand_on_hand": round(float(hr[j]), 2), "base_on_hand": round(float(hb[j]), 2)})
    res["gradcam"]["examples"] = show

    # 5. channels of the last conv layer that respond to one letter
    nt = torch.cat([va, te])
    inp_nt = augment.eval_batch(x[nt])
    acts = gap_features(clf, inp_nt)                                 # (N,160)
    ynt = y[nt]
    purity, dom = [], []
    for c in range(acts.shape[1]):
        top = acts[:, c].argsort(descending=True)[:20]
        lab, cnt = Counter(ynt[top].tolist()).most_common(1)[0]
        purity.append(cnt / 20)
        dom.append(lab)
    purity = np.array(purity)
    chosen, used = [], set()
    for c in np.argsort(-purity):
        if LABELS[dom[c]] != "none" and dom[c] not in used:
            chosen.append(int(c))
            used.add(dom[c])
        if len(chosen) == 3:
            break
    res["channels"] = {"total": int(acts.shape[1]), "letter_specific_80": int((purity >= 0.8).sum()),
                       "letter_specific_100": int((purity >= 1.0).sum()), "shown": []}
    last = blocks(clf, inp_nt)[-1]
    for c in chosen:
        top = acts[:, c].argsort(descending=True)[:5]
        strip = []
        for j in top.tolist():
            img = to_img(inp_nt[j])
            mp = F.interpolate(last[j:j + 1, c:c + 1], size=img.shape[:2], mode="bilinear", align_corners=False)[0, 0]
            a = (mp / (mp.max() + 1e-9)).clamp(0, 1).numpy()[..., None] * 0.6
            strip.append(img * (1 - a) * 0.8 + np.array([52, 209, 191]) / 255 * a)
        gapc = np.zeros((96, 8, 3))
        row = np.concatenate(sum([[s, gapc] for s in strip], [])[:-1], 1)
        save_rgb(row, OUT / f"channel_{c}.png")
        res["channels"]["shown"].append({"channel": c, "letter": LABELS[dom[c]], "purity": float(purity[c])})

    # 6. the feature space: untrained vs trained, PCA to 2-D
    torch.manual_seed(0)
    fresh = LetterNet().eval()
    inp_va = augment.eval_batch(x[va])
    emb = {}
    for name, net in (("untrained", fresh), ("trained", clf)):
        ft, fv = gap_features(net, inp_te), gap_features(net, inp_va)
        emb[name] = {"acc": round(nearest_centroid_acc(fv, y[va], ft, yt), 3), "xy": pca2(ft)}
    rng = random.Random(1)
    sel = sum([rng.sample(torch.nonzero(yt == k).flatten().tolist(), min(30, int((yt == k).sum()))) for k in range(len(LABELS))], [])
    res["embedding"] = {"untrained_acc": emb["untrained"]["acc"], "trained_acc": emb["trained"]["acc"],
                        "points": [{"l": LABELS[int(yt[j])], "a": [round(float(v), 3) for v in emb["untrained"]["xy"][j]],
                                    "b": [round(float(v), 3) for v in emb["trained"]["xy"][j]]} for j in sel]}

    # 7. one real training step from random weights, then 40 more
    with torch.enable_grad():
        torch.manual_seed(0)
        net = LetterNet()
        net.train()
        pool = background_patches([r for r in rows if part[r["id"]] == "train"], 400, seed=0)
        opt = torch.optim.Adam(net.parameters(), lr=2e-3)
        g = torch.Generator().manual_seed(0)
        fixed = tr[torch.randperm(len(tr), generator=g)[:64]]
        fx = augment.train_batch(x[fixed], m[fixed], pool, True)
        loss_fn = torch.nn.CrossEntropyLoss()
        out = net(fx)
        loss0 = loss_fn(out, y[fixed])
        p0 = out.softmax(1).gather(1, y[fixed][:, None]).mean().item()
        opt.zero_grad()
        loss0.backward()
        convs = [mod for mod in net.modules() if isinstance(mod, (torch.nn.Conv2d, torch.nn.Linear))]
        gnorm = [round(float(mod.weight.grad.norm()), 4) for mod in convs]
        wnorm = [round(float(mod.weight.detach().norm()), 3) for mod in convs]
        before = [mod.weight.detach().clone() for mod in convs]
        opt.step()
        change = [round(float((mod.weight.detach() - b).norm() / b.norm()), 5) for mod, b in zip(convs, before)]
        with torch.no_grad():
            out1 = net(fx)
            loss1 = loss_fn(out1, y[fixed]).item()
        steps = [{"step": 0, "loss": round(loss0.item(), 4), "p_correct": round(p0, 4)},
                 {"step": 1, "loss": round(loss1, 4), "p_correct": round(out1.softmax(1).gather(1, y[fixed][:, None]).mean().item(), 4)}]
        for s in range(2, 41):
            b = tr[torch.randperm(len(tr), generator=g)[:64]]
            l = loss_fn(net(augment.train_batch(x[b], m[b], pool, True)), y[b])
            opt.zero_grad()
            l.backward()
            opt.step()
            with torch.no_grad():
                o = net(fx)
                steps.append({"step": s, "loss": round(loss_fn(o, y[fixed]).item(), 4),
                              "p_correct": round(o.softmax(1).gather(1, y[fixed][:, None]).mean().item(), 4)})
    res["first_step"] = {"params": int(sum(p.numel() for p in net.parameters())), "layers": ["conv1", "conv2", "conv3", "conv4", "conv5", "dense"],
                         "grad_norm": gnorm, "weight_norm": wnorm, "relative_change": change, "steps": steps}

    # 8. training curves
    hist = {}
    for name in ("baseline", "randomized_200"):
        h = json.loads((HERE / "runs" / name / "metrics.json").read_text())
        hist[name] = {k: [round(e[k], 4) for e in h["history"]] for k in ("train_loss", "train_acc", "val_real", "val_swap")}
        hist[name].update({k: h[k] for k in ("test_real", "test_swapped_background", "hand_removed_still_right", "best_epoch")})
    res["training"] = hist

    # 9. domain randomization examples and the background shortcut (average photo per label)
    with torch.enable_grad():
        pool = background_patches([r for r in rows if part[r["id"]] == "train"], 600, seed=3)
    ex_idx = [int(torch.nonzero(y[tr] == k).flatten()[3]) for k in range(len(LABELS))]
    torch.manual_seed(5)
    ex = augment.train_batch(x[tr[ex_idx]], m[tr[ex_idx]], pool, True)
    for k, l in enumerate(LABELS):
        save_rgb(to_img(ex[k]), OUT / f"aug_{l}.png")
    frames, boxes, has = load_frames(rows)
    for k, l in enumerate(LABELS):
        sel_l = [i for i, r in enumerate(rows) if r["label"] == l]
        avg = frames[sel_l].float().mean(0).permute(1, 2, 0).numpy() / 255
        save_rgb(avg, OUT / f"avg_{l}.png", scale=1)
    res["dataset"] = {"photos": len(rows), "per_label": {l: sum(r["label"] == l for r in rows) for l in LABELS},
                      "bursts": len({(r["session"], r["burst"]) for r in rows}),
                      "split": {s: int((split == s).sum()) for s in ("train", "val", "test")}}

    # 10. the detector's grid
    det = HandDetector()
    det.load_state_dict(torch.load(HERE / "runs" / "detector" / "model.pt", weights_only=False)["state_dict"])
    det.eval()
    allpart = np.array([part[r["id"]] for r in rows])
    t_idx = np.flatnonzero(allpart == "test")
    hand_i = next(i for i in t_idx if rows[i]["label"] == "I" and rows[i]["box"])
    none_i = next(i for i in t_idx if rows[i]["label"] == "none" and rows[i]["box"])
    g2 = torch.Generator().manual_seed(7)
    wins = empty_windows(boxes, torch.tensor([i for i in t_idx if rows[i]["label"] != "none" and rows[i]["box"]]), 40, g2)
    face_win = max(wins, key=lambda w: (w[3] - w[1]))                      # a big empty crop (likely the face)
    def full_pipeline(u8):
        """the lock server's predict(): detector on the frame, then the classifier on a crop around its box"""
        inp, _ = view(u8, torch.zeros(1, 4), train=False)
        o = det(augment.normalize(inp))
        bx, p = decode(o)
        b = (bx[0] * torch.tensor([SRC_W / inp.shape[3], SRC_H / inp.shape[2]] * 2)).tolist()
        b = [max(0.0, b[0]), max(0.0, b[1]), min(float(SRC_W), b[2]), min(float(SRC_H), b[3])]
        crop = torch.from_numpy(np.asarray(context_crop(Image.fromarray(u8[0].permute(1, 2, 0).numpy()), b)))[None].permute(0, 3, 1, 2).contiguous()
        pr = clf(augment.eval_batch(crop)).softmax(1)[0]
        save_rgb(to_img(augment.eval_batch(crop)[0]), OUT / f"det_{name}_crop.png")
        return inp, o, bx, p, pr

    det_out = []
    empty = empty_view(frames, [face_win])
    empty_u8 = (F.interpolate(empty, size=(SRC_H, SRC_W), mode="bilinear", align_corners=False) * 255).round().clamp(0, 255).to(torch.uint8)
    for name, u8 in (("hand", frames[hand_i:hand_i + 1]), ("none", frames[none_i:none_i + 1]), ("empty", empty_u8)):
        img, o, bx, p, pr = full_pipeline(u8)
        save_rgb(img[0].permute(1, 2, 0).numpy(), OUT / f"det_{name}.png")
        det_out.append({"name": name, "label": rows[hand_i]["label"] if name == "hand" else ("none" if name == "none" else "empty"),
                        "grid": [[round(float(v), 3) for v in row] for row in torch.sigmoid(o[0, 0])],
                        "p_hand": round(float(p[0]), 3), "box": [round(float(v), 1) for v in bx[0]],
                        "classifier": {l: round(float(v), 3) for l, v in zip(LABELS, pr)}})
    res["detector"] = {"grid_w": GRID_W, "grid_h": GRID_H, "frames": det_out,
                       "metrics": {k: v for k, v in json.loads((HERE / "runs" / "detector" / "metrics.json").read_text()).items() if k != "history"}}
    res["pipeline"] = json.loads((HERE / "runs" / "pipeline_randomized_200.json").read_text())

    (OUT / "analysis.json").write_text(json.dumps(res))

    # Notion figures
    matplotlib.rcParams.update({"font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 9})
    fig, axes = plt.subplots(2, 4, figsize=(11, 5.6))
    for c, e in enumerate(show):
        for r, (name, key) in enumerate((("baseline", "base"), ("randomized", "rand"))):
            axes[r, c].imshow(Image.open(OUT / f"cam_{key}_{e['label']}.png"))
            axes[r, c].set_title(f"{name}: {e['label']}  ({e[key + '_on_hand']:.0%} on the hand)", fontsize=9)
            axes[r, c].axis("off")
    fig.suptitle(f"Grad-CAM: where the classifier looks (orange). Over all {len(te)} test crops: baseline "
                 f"{res['gradcam']['baseline_on_hand']:.0%} on the hand, randomized {res['gradcam']['randomized_on_hand']:.0%} "
                 f"(the hand covers {res['gradcam']['hand_area']:.0%} of the crop)", x=0.01, ha="left", fontsize=10, fontweight="bold")
    fig.savefig(FIG / "inside_gradcam.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, key in zip(axes, ("a", "b")):
        for k, l in enumerate(LABELS):
            pts = np.array([p[key] for p in res["embedding"]["points"] if p["l"] == l])
            ax.scatter(pts[:, 0], pts[:, 1], s=14, label=l, color=plt.cm.tab10(k))
        name = "untrained" if key == "a" else "trained"
        ax.set_title(f"{name}: nearest-centroid accuracy {res['embedding'][name + '_acc']:.0%}", loc="left")
        ax.set_xticks([])
        ax.set_yticks([])
    axes[1].legend(frameon=False, fontsize=8, loc="best")
    fig.suptitle("The 160 features before the last layer, shrunk to 2-D (PCA), for the test crops", x=0.01, ha="left", fontweight="bold")
    fig.savefig(FIG / "inside_embedding.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    print(json.dumps({k: res[k] for k in ("crop_probs", "kernels", "gradcam", "channels")}, indent=1)[:3000])
    print("embedding acc: untrained", res["embedding"]["untrained_acc"], "trained", res["embedding"]["trained_acc"])
    print("first step: loss", steps[0]["loss"], "->", steps[1]["loss"], "-> step 40", steps[-1]["loss"],
          "| p(correct)", steps[0]["p_correct"], "->", steps[-1]["p_correct"])
    print("grad norms", gnorm, "\nrelative change", change)
    print("detector:", [(d["name"], d["p_hand"]) for d in det_out])
    print(f"saved {len(list(OUT.glob('*.png')))} pictures + analysis.json to {OUT}")


if __name__ == "__main__":
    main()
