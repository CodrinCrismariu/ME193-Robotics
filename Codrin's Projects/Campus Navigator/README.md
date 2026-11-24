# Campus Navigator

A walking navigator that works without GPS or Wi-Fi positioning. An MPU6050 on my ankle feeds an Arduino UNO Q, which runs my step counter (a neural network that gives steps and distance) plus a gyro heading tracker that detects 90° and 180° turns. Walks we recorded between labelled waypoints are the demonstrations for a **behaviour-cloning** policy (k-nearest neighbours). While navigating, the board serves a phone page with one instruction at a time: "Go forward 23 m to Up stairs", "Turn left", "You've arrived at In Nolop".

ME 193 (AI and Robotics), Tufts, Fall 2026.

**Route:** In front of class → Up stairs → Down stairs → In front of Nolop → In Nolop (4 legs, 4 turns, about 65 m).
**Result:** it navigated the whole route in two live tests and announced arrival both times. Turns and distances are in [`notion/Campus-Navigator-Notion.md`](notion/Campus-Navigator-Notion.md).

## Videos

- [`notion/Campus-Navigator-side-by-side.mp4`](notion/Campus-Navigator-side-by-side.mp4): a filmed run, with the phone screen lined up next to it.
- [`notion/Campus-Navigator-demo.mp4`](notion/Campus-Navigator-demo.mp4): a screen recording of the phone during the first run (a 33 s connection drop is sped up and labelled).

## Folder layout

| path | what |
|---|---|
| [`campus-nav/`](campus-nav/) | The Arduino App Lab app that runs on the UNO Q. Its [README](campus-nav/README.md) covers recording walks, navigating, and how each part works |
| [`campus-nav/python/stream.py`, `nn.py`, `gait.py`, `model.npz`](campus-nav/python/) | The step counter: the neural network (numpy port), steps, distance, foot contacts |
| [`campus-nav/python/heading.py`](campus-nav/python/heading.py) | 3-D orientation from the gyro, heading, 90° turn detection, per-stride stairs features |
| [`campus-nav/python/walks.py`](campus-nav/python/walks.py) | Recording labelled walks, edits, and splitting walks into legs |
| [`campus-nav/python/nav.py`](campus-nav/python/nav.py) | The behaviour-cloning policy and the turn-by-turn navigator |
| [`campus-nav/python/main.py`](campus-nav/python/main.py) | Ties it together and serves the phone page |
| [`campus-nav/assets/`](campus-nav/assets/) | The phone page: navigate, record, review and fix walks |
| [`campus-nav/sketch/`](campus-nav/sketch/) | Reads the MPU6050 at 100 Hz on the board's microcontroller |
| [`notion/`](notion/) | The write-up (Markdown for Notion), its figures and the videos |

## Running it

1. Wire an MPU6050 to the UNO Q's I²C pins (SDA/SCL) and strap it to the outside of the ankle.
2. Copy `campus-nav/` to `/home/arduino/ArduinoApps/campus-nav` on the board, then start it from Arduino App Lab. Or use the VS Code kit from the class repo ([`Public stuff/UNOQ code`](https://github.com/chrisbuerginrogers/ME193-Robotics/tree/main/Public%20stuff/UNOQ%20code)), whose deploy tool does both: `python tools/deploy.py campus-nav`.
3. Open `http://<board-ip>:7000` on a phone. Stand still for a few seconds so the gyro drift gets measured, then record walks or navigate.

Recorded walks are saved on the board in `campus-nav/data/walks/`. They're not in this repo. Use **Download all (.zip)** on the page to get them.
