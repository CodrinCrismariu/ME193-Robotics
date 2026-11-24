# 🧭 Campus Navigator

A GPS-free, Wi-Fi-free campus navigator built on the step counter. The MPU6050
on the ankle gives steps and distance (the step counter's neural network) and
heading (a gyro attitude filter). Labelled walks between waypoints become the
behaviour-cloning data. Open `http://<board-ip>:7000`.

Waypoints: **In front of class, Up stairs, Down stairs, In front of Nolop, In Nolop**.
Directions: **Forward, Turn left, Turn right** (a U-turn is two turns the same way).

## Status

- [x] **Part 1: labelled walks.** Record walks, detect 90° turns, measure the distance between turns and waypoints, and review and fix them.
- [ ] Part 2: the stairs classifier (trained on strides labelled between *Down stairs* and *Up stairs*).
- [x] **Part 3a: navigation.** A behaviour-cloning policy built from every recorded leg, from both walkers, drives a turn-by-turn navigator.
- [ ] Part 3b: DAgger corrections while navigating.

## Recording a walk

1. Enter your name and height, then save. Partners use their own names, so their walks can be told apart.
2. Stand still until the page says the gyro drift is measured (a few seconds).
3. Tap the waypoint you're at (this only picks it), press **Start recording**, and walk. **Reset** zeroes steps, distance and turns; nothing is saved until you start recording.
4. At each waypoint, stand still for about 3 s (this re-measures the drift), then tap **Reached** on it.
5. **Stop and save.** Each stretch between two waypoints becomes a leg, like
   `In front of class → Down stairs: ↑ Forward 23 m · ↰ Turn left · ↑ Forward 8 m`.
6. Under **Saved walks**, check the directions. Tap a turn to fix it (it was really left or right, a U-turn, or not a turn), and fix any waypoint label you tapped by mistake.

## Navigating

Pick **Where are you?** and **Where to?**; the destination list only offers places your recorded walks have reached from there. Then press **Start navigating**. The page shows one big instruction at a time, like "Go forward 25 m, then turn left", then "Turn left", then "You've arrived at In Nolop".

- **Policy (`nav.py`):** every quarter metre of every recorded leg is one example: (leg, turns made so far, metres since the waypoint) → (next direction, metres until it). The policy copies the majority action of the 5 nearest examples and averages their distance to it.
- **Landmarks:** making the expected turn snaps the distance to where the recorded walks turned. A turn the other way gets a "Wrong way?" warning.
- **Re-planning:** **I'm at …** puts you at a waypoint at any time and plans again from there.

**Leave-one-out test on the two recorded walks:** fitted on one, navigating the other's raw sensor data through the board's code. Every turn was called at the right place, and arrival at In Nolop came within ~1 s.

## How it works

    MPU6050 --> sketch (100 Hz) --> Python, every 0.25 s:
      stream.py   step counter (copy of step-counter's): steps, distance, foot contacts
      heading.py  attitude filter -> heading -> 90-degree turns; per-stride stairs features
      walks.py    labelled walks -> legs (forward m / turn left / turn right)
      nav.py      behaviour-cloning policy (k-NN on the legs) + turn-by-turn navigator

- **Heading:** the gyro is integrated as a full 3-D orientation, with the tilt pulled towards gravity. Heading is the compass direction of a sensor axis across the shin, so the leg's swing and twist don't count as turning.
- **Turns:** the heading, averaged over about a stride, moves more than 55° away from the current straight and settles. The turn is snapped to 90° or 180°.
- **Distance between events:** the step counter's distance at each turn and waypoint.
- **Stairs features:** recorded for every stride (height change, stride time, shin tilt, rotation and impact peaks).

**Gyro drift:** this board's gyro is off by about 3.7 °/s around the shin. Uncorrected, that would look like turning 200° a minute, so the app measures it whenever you stand still for 2 s.

**Tested:**
- On simulated walks with known turns: 26 of 33 turns correct, and ~20° of drift per minute. The misses were turns the simulator placed a second apart.
- On the real recordings: correct only once the drift is measured.

Saved data lives in `data/walks/` on the board (deploying never touches it). **Download all (.zip)** gets every walk: a JSON file with labels, counts, turns and strides, plus the raw IMU as CSV.

`stream.py`, `gait.py`, `nn.py` and `model.npz` are copies from `../step-counter/python`. Keep them in step.
