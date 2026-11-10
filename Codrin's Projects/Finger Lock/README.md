# Sign Lock

A lock that opens when I sign my name, **C-O-D-R-I-N**, letter by letter in American Sign Language at a webcam. Two small neural networks, both built and trained from scratch in PyTorch, read every camera frame. A **hand detector** finds my hand and draws a box around it, and a **letter classifier** reads the boxed hand as one of the six letters or "none". A lock policy then decides whether to accept each letter and whether the letters open the lock. I took and labeled all 1,540 training photos myself. On test frames the networks never saw, the system reads the right label **89.5%** of the time.

ME 193 (AI and Robotics), Tufts, Fall 2026, Midterm 2.

> **Note:** I'm at the CoRL robotics conference this week, so I couldn't build the physical lock (button, LEDs and servo in a box) or move the models onto the Arduino UNO Q. Everything here runs on my laptop: the lock is a web page with an on-screen button, LEDs and letter slots, driven by the two networks.

## Status

| part | what | status |
|---|---|---|
| [`step1-v-fit/`](step1-v-fit/) | Gradient descent by hand: one neuron, one neuron + ReLU, and two ReLU neurons fitting the V data from Anna's Code (math, Excel, numpy, PyTorch check) | done |
| [`step2-cnn/`](step2-cnn/) | First version: a CNN that counts fingers (1–5), trained on a public stand-in dataset | done (97–99% on the stand-in data) |
| [`step3-dataset/`](step3-dataset/) | My own dataset: a web app to collect webcam photos of C, O, D, R, I, N and none, a box labeler, and a test for background shortcuts | done (1,540 photos) |
| [`step4-models/`](step4-models/) | The hand detector and the letter classifier, trained from scratch, with background randomization and a look inside the trained networks | done (89.5% end to end) |
| [`step5-lock/`](step5-lock/) | The lock: a web page that runs both networks on the webcam, with the lock policy | done |
| [`movie2/`](movie2/) | Stop-motion movie explaining how the networks work, with my voice-over; the script and the measured numbers it uses | done |
| physical box and UNO Q | button, LEDs and servo; running the models on the Arduino UNO Q | not done (I'm at CoRL) |

## How it works

1. **Hand detector** (338,389 weights): the whole frame shrunk to 192×144 → a 12 × 9 grid of cells, each giving a hand score and a box. The top-scoring cell draws the box.
2. **Letter classifier** (349,439 weights): a 96×96 crop around the box → 5 blocks of 3×3 convolution, batch norm, ReLU and 2×2 max pooling → global average pooling → 7 probabilities.
3. **Lock policy**: probabilities averaged over 4 frames; a letter counts when it stays at 70% or more for 0.8 s; 20 s timeout; 3 wrong codes → 30 s lockout. Details in [`step5-lock/README.md`](step5-lock/README.md).

My first classifier partly read the letter from the background: I had shot one letter at a time, and between letters the framing and the light shifted. Pasting my hand onto random backgrounds during training fixed it. See [`step3-dataset/README.md`](step3-dataset/README.md) and [`step4-models/README.md`](step4-models/README.md).

## Folder layout

```
step1-v-fit/     Part 1: fitting the V (README = the write-up), Excel workbook, figures
step2-cnn/       first version: the finger-counting CNN on stand-in data
step3-dataset/   my dataset: collector and labeler web app, box pre-labeling, hand masks, shortcut test
step4-models/    the two networks: training, tests, analysis, trained models in runs/
step5-lock/      the lock: lock_server.py + web page (lock policy in its README)
movie/           the first stop-motion movie (finger-counting version)
movie2/          the final stop-motion movie: voice-over script and the measured numbers (analysis.json)
```

## Run the lock

Python 3.12, from this folder:

```
python -m venv .venv
.venv\Scripts\activate
pip install numpy matplotlib pillow openpyxl
pip install torch --index-url https://download.pytorch.org/whl/cpu

cd step5-lock
..\.venv\Scripts\python lock_server.py
```

Chrome or Edge opens `http://localhost:8001`. Allow camera access, press the button (or Space), and sign C-O-D-R-I-N. The trained models are included in `step4-models/runs/`. My photos are not: they are pictures of me, so they stay off GitHub. To collect your own dataset, see `step3-dataset/`.

## Credits

- Anna's Code and the V data: from the ME 193 course repository.
- Stand-in data for the first version: [Sign Language Digits Dataset](https://github.com/ardamavi/Sign-Language-Digits-Dataset) by Arda Mavi (Apache-2.0).
- Hand-box pre-labeling and training masks: MediaPipe Hands (Google). The lock's networks don't use it.
- Built with help from Claude (Anthropic).
