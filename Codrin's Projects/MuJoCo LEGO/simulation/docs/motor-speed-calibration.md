# Motor command limits — November 16, 2026

LEGO's Double Motor 45651 specification lists 322–346 rpm unloaded,
192–220 rpm at 50 mN·m, and 148.1–173.1 mN·m stall torque, measured at 4.2 V.
Source: https://static.prod01.ue1.p.pcomm.net/legoedu/content/classroomandteacherresources/Science_Tech%20Spec%20Sheet_Digital_1HY25_ENUS.pdf

The previous simulation used 21 rad/s per action unit (about 200 rpm), within
the specified loaded-speed range. However, its Gaussian policy outputs were
not clipped, so this scale did not actually bound requested motor speeds.

The physical adapter in `MuJoCo LEGO/lego_robot.py` converts a policy output
to a speed using the same 21 rad/s scale, converts to a whole percentage,
and saturates at hardware.json's `max_percent` (currently 100). Its current
`rad_s_at_100_percent` estimates are:

| Output | Command ceiling, rad/s | Command ceiling, rpm |
|---|---:|---:|
| Left | 16.494 | 157.5 |
| Right | 14.076 | 134.4 |

`beam_task.py` now uses the same per-output saturation and percentage rounding.
The policy's previous-command observation contains the applied command after
saturation and rounding, matching the physical adapter. The 21 rad/s policy
scale stays unchanged so the existing ONNX deployment contract remains valid.
This changes training behavior; older checkpoints were trained without these limits.

These command ceilings are **automatic estimates**, extrapolated from short
20% motor pulses. They are not measurements of either motor's true maximum
speed. The probes mix acceleration and contact loads; the 30% probes also show
direction dependence. Do not replace the manufacturer's no-load maximum with
these values or treat them as a fitted torque-speed curve. A full-speed,
steady-state measurement is needed to verify the percentage-to-speed mapping.

Source data: `C:\Users\Useru\Documents\ME193-Robotics\Codrin's Projects\MuJoCo LEGO`,
including `hardware.json`, `calibration/motor_probes_20percent.json`,
`calibration/motor_probes.json`, and `calibration/RESULTS.md`.
The recorded policy test was capped at 20%; the current hardware configuration
allows 100%. A 20% cap would reduce the above limits by a factor of five and
would need to be represented in training too.

The existing torque cap, motor gain, gearbox inertia, sensor noise, and delays
have not been fitted by these short speed probes. Those assumptions remain
separate from this correction to command speed and saturation.
