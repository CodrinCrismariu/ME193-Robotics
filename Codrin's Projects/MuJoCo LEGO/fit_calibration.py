"""Compare discrete axis/direction candidates against the actual MuJoCo model.

Run with WSL's existing MuJoCo environment. This produces a fit report, not a
verified calibration: contact conditions may differ from the simulation.
"""
import itertools
import json
from pathlib import Path

import mujoco
import numpy as np


def main():
    root = Path(__file__).resolve().parent
    trials = json.loads((root / "calibration/motor_probes.json").read_text())["trials"]
    baseline = json.loads((root / "calibration/baseline.json").read_text())[-1]
    spec = mujoco.MjSpec.from_file(str(root / 'simulation/onshape/beam_robot.xml'))
    spec.body('chassis').add_freejoint()
    spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, 0.1])
    model = spec.compile()
    model.opt.timestep = .002
    body = model.body('chassis').id
    # Shaft along raw X or Y; either physical direction is possible.
    axes_candidates = ([1, 2, 3], [-1, -2, 3], [-2, 1, 3], [2, -1, 3])
    results = []
    for axes, signs in itertools.product(axes_candidates, itertools.product((-1, 1), repeat=2)):
        rotation = np.zeros((3, 3))
        for i,a in enumerate(axes):
            rotation[i, abs(a)-1] = np.sign(a)
        error, count = 0., 0
        for trial in trials:
            data = mujoco.MjData(model)
            data.qpos[2] = .017
            data.qpos[3] = 1
            for i,k in enumerate(('left','right')):
                data.qpos[7+i] = np.deg2rad(trial['before'][k]['position'] - baseline[k]['position']) * signs[i]
            mujoco.mj_forward(model, data)
            # Settle with the measured beam configuration held close to its start.
            target = data.qpos[7:].copy()
            for _ in range(150):
                data.qfrc_applied[6:] = np.clip(2*(target-data.qpos[7:]) - .05*data.qvel[6:], -.16,.16)
                mujoco.mj_step(model, data)
            data.time = 0
            key = trial['motor']
            motor_i = 0 if key == 'left' else 1
            direction = 1 if trial['direction'] == 'clockwise' else -1
            seq = trial['samples']
            t0 = seq[0]['received_at']['imu']
            for s in seq:
                t = s['received_at']['imu'] - t0
                while data.time < t:
                    target_speed = np.zeros(2)
                    if data.time < trial.get('duration_ms', 200)/1000:
                        target_speed[motor_i] = direction*signs[motor_i]*16*trial['percent']/100
                    data.qfrc_applied[6:] = np.clip(.05*(target_speed-data.qvel[6:]),-.16,.16)
                    mujoco.mj_step(model,data)
                # Object velocity in body-local coordinates, angular then linear.
                velocity = np.zeros(6)
                mujoco.mj_objectVelocity(model,data,mujoco.mjtObj.mjOBJ_BODY,body,velocity,1)
                observed = rotation @ np.deg2rad([s['imu']['gyroscope'+a] for a in 'XYZ'])
                error += np.sum((velocity[:3]-observed)**2)
                count += 3
        results.append({'body_axes':axes,'motor_sign':list(signs),'gyro_rmse_rad_s':float(np.sqrt(error/count))})
    results.sort(key=lambda x:x['gyro_rmse_rad_s'])
    (root/'calibration/model_fit.json').write_text(json.dumps(results,indent=2)+'\n')
    print(json.dumps(results[:6],indent=2))


if __name__ == '__main__':
    main()
