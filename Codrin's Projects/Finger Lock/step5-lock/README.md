# Part 5: The sign lock (laptop version)

A web page that opens a lock when you sign the secret code at the webcam: my name, **C-O-D-R-I-N**, one letter at a time. Every frame goes through the two networks from Part 4: the **hand detector** finds the hand, and the **letter classifier** reads the boxed hand as C, O, D, R, I, N or none. The page then decides, frame by frame, whether to accept a letter, and whether the signed letters open the lock.

## Run it

```
cd step5-lock
..\.venv\Scripts\python lock_server.py
```

Chrome or Edge opens `http://localhost:8001`; allow camera access and pick the USB webcam. Options: `--classifier <run>` and `--detector <run>` pick trained models from `step4-models/runs/`, `--min-hand 0.5` sets how sure the detector must be that there is a hand.

The page shows:
- the live camera with the detector's box and the current letter;
- bars with the classifier's probabilities for all seven labels, averaged over the last few frames;
- the lock: a padlock, one LED and one slot per letter of the code, and a ring that fills while you hold a letter.

The secret code (3 to 8 letters from C O D R I N, `CODRIN` by default), the hold time, the confidence threshold and the smoothing are under **Settings**.

## The lock policy: how it decides

| state | what happens |
|---|---|
| **idle** | LEDs off. Signing does nothing. Pressing the button (or **Space**) wakes the lock. |
| **listening** | LEDs glow. For every frame, the classifier's probabilities are averaged over the last 4 frames, and the top label counts as a *candidate* if it is a letter (not none) and at least 70% sure. A letter is **accepted** when the same candidate lasts for **0.8 s**: its LED lights up, a tick plays, and it fills the next slot. |
| | The same letter can't count twice in a row: after a letter is accepted, the hand has to show a different letter or leave for 0.3 s first. One long "C" is one C, not C-C-C. |
| | If no new letter is accepted for **20 s**, the lock goes back to idle. |
| **open** | Once as many letters as the code has are in (six for CODRIN), the right code opens the lock (green LEDs, the shackle lifts, a chord plays). It locks again by itself after **6 s**. |
| **wrong** | A wrong code flashes red and buzzes. **Three wrong codes in a row lock the button for 30 s.** |

Why these rules:
- **Averaging over frames, the 70% threshold and the 0.8 s hold:** one confident frame is not enough. The letter has to be read consistently for most of a second, so a hand passing through a C on the way to another letter doesn't count.
- **"No hand" counts as 100% none:** lowering the hand clears the candidate right away.
- **The wake button:** nothing is read until someone deliberately starts an attempt.
- **The lockout:** with 6 letters to choose from there are 6⁶ = 46,656 possible six-letter codes, and a wrong attempt takes about 10 seconds to sign, so three tries per 30 s makes guessing hopeless. A three-letter code would have only 6³ = 216.

These rules were checked with a simulation that feeds the page fake predictions on a fake clock: signing before the button press does nothing; a letter held 0.5 s isn't accepted but 1.1 s is; holding C for 3 s more doesn't add a second C; going straight from one letter to the next works; three of six letters keep it listening; C-O-D-R-I-N opens the lock and it relocks after 6 s; three wrong codes cause a lockout that ends after 30 s; 20 s without a letter times out; and a hand flickering on and off every 0.2 s never counts.

## Files

```
lock_server.py   serves the page; /api/predict runs the detector and the classifier on one frame
web/index.html   the page
web/lock.js      camera, sending frames, smoothing, the lock policy, sounds
web/style.css    looks
```

The server reuses the preprocessing from training (`step4-models`), so the networks see webcam frames exactly as they saw the training photos: 320×240, not mirrored, the detector on the whole frame shrunk to 192×144, the classifier on a crop around the box.
