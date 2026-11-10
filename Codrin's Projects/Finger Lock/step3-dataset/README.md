# Part 3: Building my own sign-language dataset

The lock now reads **letters in American Sign Language**: the six letters of my name, **C, O, D, R, I, N**, plus a **none** class for everything else. Instead of downloading someone else's photos, I collect my own with the USB webcam that will sit over the lock. The finished system uses two networks, both built and trained from scratch in PyTorch:

1. **Hand detector:** camera frame → a bounding box around my hand (or "no hand").
2. **Letter classifier:** the boxed hand, cut out of the frame → C, O, D, R, I, N or none.

So every photo needs two labels: the letter (chosen when the photo is taken) and the hand's bounding box (checked in the labeler).

## Run it

```
cd step3-dataset
..\.venv\Scripts\python collect_server.py
```

Your browser opens the **collector** at `http://localhost:8000`; the **box labeler** is at `http://localhost:8000/label.html` (there is a link in the collector's header). Use Chrome or Edge, and allow camera access. The terminal prints a line for every photo and box saved; press Ctrl+C there to stop. Everything is saved in `step3-dataset/dataset/`.

## 1. Collect photos (collector)

1. **Camera:** pick the USB webcam. Try **Lock exposure**: if your camera allows it, the brightness stops jumping around when your hand comes into view.
2. **Label:** press **C O D R I N** for a letter or **X** for none. The hint under the video describes the hand shape.
3. **Capture:** **Space** takes one photo. **B** starts a burst (by default 3-2-1, then 20 photos 4 per second): move your hand a little during it, shifting, tilting and changing the distance. **Z** undoes the last photo.

The gallery shows the latest photos; click one to open it in the box labeler.

### What makes a good dataset

- **Aim for about 200 photos per letter and 300 for none.** The progress bars track this.
- **Use several sessions.** Press **New session** when you change something: the time of day, the room light, a sleeve, the surface, or the camera height.
- **Vary the hand.** Move it around the whole frame, rotate it a little, change the distance, and use both the left and right hand if you'll use both. The detector has to learn to find the hand anywhere.
- **Make none rich.** Include no hand at all, a fist, an open hand, other letters (A, B, E, L, V…), a hand halfway between two letters, and your arm moving in and out of view.

## 2. Pre-fill the boxes

```
..\.venv\Scripts\python prelabel_boxes.py
```

A pretrained hand tracker (MediaPipe Hands) draws a first box on every photo whose box hasn't been checked yet; boxes you have already checked are never touched. Run it while you are not capturing photos. The tracker's model file (7.8 MB) downloads on the first run. Photos where it finds no hand are left without a box, for you to draw.

## 3. Check every box (labeler)

The labeler shows one photo at a time with its box. Draw the box **from the fingertips to the wrist, leaving out the forearm**.

| key | action |
|---|---|
| drag on the photo | draw a new box |
| **Enter** | confirm the box and go to the next photo |
| **H** | "no hand" in this photo, then next |
| **U** | undo the box you just drew |
| **A / D** or **← / →** | previous / next photo, without saving |
| **Delete** | delete the photo |

The **Which photos** menu shows *Not checked yet* by default, so working through it once checks every box. A confirmed box turns green; an unchecked one is orange.

**How the boxes were made:** each box was pre-filled by a pretrained hand tracker (MediaPipe Hands). 141 photos, including every photo where the tracker found no hand, were reviewed and corrected one by one in the labeler (listed in `dataset/reviewed_in_labeler.txt`); the other 1,399 pre-filled boxes were accepted as they were, after a contact-sheet spot check showed them tight around the hand. Pre-labeling with a pretrained model is standard practice; both networks of the lock are still built and trained from scratch.

`python check_dataset.py` prints counts per letter, lists photos whose box is not checked yet or looks wrong, and makes `figures/dataset_overview.png`, a contact sheet of random photos with their boxes.

## Checking for background shortcuts

```
..\.venv\Scripts\python check_background_bias.py
```

A network learns whatever separates the classes most easily. If each letter is recorded in its own stretch of time, anything that drifts meanwhile (the camera framing, where I sit, the light, the webcam's automatic exposure) becomes a clue for that letter, even in the same room. The network can learn the scene instead of the hand, then fail at the lock, where the scene never changes.

The script grays out the hand box in every photo, plus one box borrowed from each other letter, so the gray shapes themselves say nothing about the letter. It then trains a simple linear classifier on what is left and tests it on whole bursts it never saw. Chance is 1 in 7 (14%).

**First session** (1,540 photos, all of one letter, then all of the next):

| what the classifier sees | correct on unseen bursts |
|---|---|
| only the box's position and shape | 40% |
| the photo with the hand hidden | **86%** |
| the whole photo | 87% |

The average photo of each letter (`figures/average_per_letter.png`) shows why: the framing shifted between letters (the C photos show mostly curtains, while by N the view has moved right to the picture on the wall), and the average brightness rose steadily in recording order (C 102 → O 105 → D 106 → R 109 → I 112 → N 116 → none 119). The fix is to take the next photos with **all seven labels in each setup**: stay in one position and lighting, cycle C → O → D → R → I → N → none with a short burst each, then change position or lighting and cycle again.

## What is saved

```
dataset/
  images/<id>.jpg     320×240 color photo, exactly as the camera saw it (not mirrored)
  samples.jsonl       one line per photo: id, label, box, checked, session, burst, time
```

`box` is `[x0, y0, x1, y1]` in pixels of the 320×240 photo (x1 and y1 exclusive), or `null` when there is no hand. `checked` becomes true when the box has been confirmed (one by one in the labeler, or accepted in bulk); only checked photos are used for training.

When training, split by **burst and session**, not by photo. Photos from one burst are nearly identical, so splitting them across training and test would inflate the score.

## Privacy

These are photos of you. `step3-dataset/dataset/` is in `.gitignore`, so it won't be pushed to GitHub unless you choose to publish it.

## Next

- `train_detector.py`: a small CNN, from scratch, that outputs "is there a hand?" and the four box coordinates.
- `train_classifier.py`: a small CNN on the boxed hand crop, with 7 classes.
- `live.py`: webcam → detector → crop → classifier → letter, then the lock policy.
