# Orange 0999 deployment experiment, 2026-11-16

Connected on Windows using legoeducation 1.2.0. The old 1.1.1 SDK could discover
the device but rejected its RPC firmware version. The new project virtual
environment uses 1.2.0; the original AI in Mobile Robotics environment was unchanged.

Exported the actual `train-20261116-193502` / `model_299.pt` actor with mjlab on
the HPC cluster (job 7271377). Validated its [1,160] -> [1,2] ONNX interface and
ran inference on live LEGO sensor readings.

Measurements:

- Flat baseline saved after the user confirmed the reference pose.
- User's 90 degree left turn: yaw change 906 raw units, integrated Z gyro 89.97.
  This supports yaw in tenths of degrees, gyro in degrees/s.
- Flat acceleration magnitude about 1029 raw units per g.
- Independent timed probes: 200 ms at 20%, then 250 ms at 30%, in both directions.
  Clockwise increased each encoder. Loaded speeds were asymmetric.
- Compared 16 axis/direction candidates against the exact trained MuJoCo robot.
  Both probe sets ranked body axes `[-2,1,3]`, motor signs `[1,-1]` first, with
  approximately 7% and 13% fit-error margins over the next candidate. This is
  suggestive, not conclusive, and remains marked `automatic_estimate`.
- Speed scaling extrapolates 20% encoder measurements; it was not measured at
  100%. Full-rate/loaded calibration and tumbling heading accuracy remain unverified.

Completed a three-second powered policy experiment, capped at 20%, using the
explicit estimated-calibration option. Live policy actions changed with sensor
feedback. This was a restricted experiment, not an evaluation of the full
trained gait: output limits substantially reduced its requested speeds.

The reported yaw changed from about 18.3 to 46.3 degrees, approximately 28 degrees
of heading drift; straight travel was not demonstrated. After the run, a separate
connection confirmed both reported motor speeds were 0 and both gyro and motor
power readings were 0. See `after_policy.json` and `policy_trial.log`.

The SDK emitted unhandled pending-command Future warnings at intentional
disconnect after overlapping nonblocking timed commands. The run itself
completed with fresh telemetry. This SDK behavior needs resolution before
longer operation. Motion commands remain finite-duration, and cleanup now waits
for the explicit stop response before disconnecting.

Nine local tests passed, including rejection of automatic estimates unless
explicitly enabled. These files were originally local; they were collected
for GitHub publication with the project presentation on November 17, 2026.
