# Part 4: The two networks (hand detector + letter classifier)

Both networks are built and trained from scratch in PyTorch on my own dataset (`step3-dataset/`). The **hand detector** finds my hand in the whole camera frame. The **letter classifier** reads a crop of just the hand as C, O, D, R, I, N or none.

## Results on held-out test bursts

| | result |
|---|---:|
| Classifier on real test crops | 89.5% |
| Classifier, same hands on other letters' backgrounds | 85.5% (first model without randomization: 45.0%) |
| Classifier with the hand erased (chance is 14%) | 14.5% (first model: 32.8%, so it was reading the background) |
| Detector: hand vs. no hand (370 test frames) | 99.7% (0 missed, 1 false alarm) |
| Detector: mean IoU with my boxes / share with IoU > 0.5 | 0.78 / 96% |
| Detector: false "hand" on 400 empty crops | 5% |
| **End to end** (detector box → classifier), 370 test frames | **89.5%** (90.0% with my labeled boxes) |

## The networks

**Letter classifier** (`models.py`, 349,439 weights): a 96×96 color crop around the hand → 5 blocks of 3×3 convolution → batch norm → ReLU (→ 2×2 max pool in blocks 1–4), with 24, 48, 96, 128 and 160 kernels → global average pooling → dropout 0.3 → linear 160 → 7. Trained with cross-entropy (label smoothing 0.05), AdamW, a one-cycle learning rate up to 0.002, batches of 64, for 200 epochs. The epoch with the best validation accuracy is kept.

**Hand detector** (`detector.py`, 338,389 weights): a small YOLO-style grid detector. The whole frame, shrunk to 192×144 → 4 conv/pool blocks (16, 32, 64, 96 kernels) → a 12 × 9 grid → 2 more blocks of 128 (the second dilated) → a 1×1 head with 5 numbers per cell: hand score, dx, dy, tw, th (box size 48 px × e^tw). Trained with weighted binary cross-entropy on the scores plus smooth-L1 on the box at the true cell, for 60 epochs. Extra empty crops, cut from the parts of photos away from the hand, teach it what isn't a hand.

**Domain randomization** (`augment.py`): the hand is cut out with its mask (`step3-dataset/make_hand_masks.py`). 9 times in 10 it is pasted onto a random background (a patch of another training photo or a synthetic texture), with new light, colour and crop every time.

## Files

| file | what it does |
|---|---|
| `data.py` | loads the dataset, splits it by burst (train / validation / test), makes the crops around the hand |
| `augment.py` | crops, light and colour changes, and the random backgrounds |
| `models.py` | the letter classifier |
| `detector.py` | the hand detector, and how boxes become grid targets and back |
| `train_classifier.py` | trains the classifier and runs the three shortcut tests (`--no-bg-random` for the baseline, `--epochs`, `--name`) |
| `train_detector.py` | trains the detector |
| `compare_runs.py` | baseline vs. randomized on the shortcut tests → `figures/shortcut_test.png` |
| `pipeline_eval.py` | end-to-end test, the way the lock runs → `figures/pipeline_confusion.png` |
| `analyze_model.py` | looks inside the trained networks (kernels, feature maps, Grad-CAM, letter channels, PCA, one training step) → `figures/inside_*.png` and the numbers used in the stop-motion movie |
| `watch_training.py` | live training plots from the `runs_*.log` files |
| `runs/<name>/` | each trained model (`model.pt`) with its `metrics.json` and confusion matrices |

## Run it

From this folder, after building the dataset in `step3-dataset/`:

```
..\.venv\Scripts\python train_classifier.py --epochs 200 --name randomized_200
..\.venv\Scripts\python train_classifier.py --no-bg-random --name baseline
..\.venv\Scripts\python train_detector.py
..\.venv\Scripts\python compare_runs.py
..\.venv\Scripts\python pipeline_eval.py
..\.venv\Scripts\python analyze_model.py
```

The trained models used by the lock (`runs/randomized_200/` and `runs/detector/`) are included, so `step5-lock/` runs without retraining. The dataset itself (photos of me) is not in the repository.
