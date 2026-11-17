# Replay hand-disturbance timing

Outputs: `real-vs-simulation.mp4` (synchronized side-by-side) and `simulation-timed.mp4` (standalone simulation). Both are 20 seconds at normal playback speed. The comparison uses the first 20 seconds of `policy on real.MOV`.

Four hand-contact windows were visually estimated: 8.48–9.32 s, 11.90–12.48 s, 14.60–15.64 s, and 18.96–19.68 s. Policy startup is estimated at 3.8 s. Force and torque magnitudes/directions are illustrative estimates; this is a timing replay, not a measured force reconstruction or quantitative validation of the simulator.

The policy checkpoint is the one referenced by the Windows deployment project: `train-20261116-193502/model_299.pt`. The environment uses the original CAD arm surfaces and the current hardware command saturation/rounding. Only the scheduled hand-contact wrenches act; unrelated random pushes and automatic resets are disabled. The policy controls the robot freely between contacts; poses are not scripted.

`hand_schedule.json` contains all timing and wrench assumptions. `manifest.json` records input hashes. Full simulation telemetry and the rendering script snapshot are in `../replay_hand_disturbances-20261116-221327/`. Validation checked 1000 finite telemetry rows and zero external wrenches outside the scheduled contact windows.

Recompose locally:

```bash
.venv/bin/python 09_onshape_robot/mjlab_task/compose_hand_comparison.py \
  'policy on real.MOV' runs/replay_hand_disturbances-20261116-221327/simulation.mp4 \
  runs/hand-disturbance-replay/hand_schedule.json runs/hand-disturbance-replay
```
