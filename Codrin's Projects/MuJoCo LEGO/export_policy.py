"""Run in the SAME mjlab environment as training, after a checkpoint finishes.

python export_policy.py model_299.pt --task-dir /home/useru/mujoco-hpc/09_onshape_robot/mjlab_task
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoint", type=Path)
    p.add_argument("--task-dir", type=Path, default=Path(__file__).resolve().parent)
    p.add_argument("--output", type=Path, default=Path("exported"))
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()
    sys.path.insert(0, str(args.task_dir.resolve()))
    import beam_task
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
    from mjlab.utils.torch import configure_torch_backends
    from lego_robot import ACTION_SCALE, CONTROL_DT, HISTORY, TERMS

    if beam_task.SENSOR_HISTORY != HISTORY or beam_task.MAX_BEAM_SPEED != ACTION_SCALE:
        raise ValueError("Training constants changed; update deployment contract")
    configure_torch_backends()
    cfg = load_env_cfg("Beam-Drive-Straight", play=True)
    cfg.scene.num_envs = 1
    if cfg.sim.mujoco.timestep * cfg.decimation != CONTROL_DT:
        raise ValueError("Training control period changed")
    if list(cfg.observations["actor"].terms) != list(TERMS):
        raise ValueError("Training observation terms changed")
    agent = load_rl_cfg("Beam-Drive-Straight")
    env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=cfg, device=args.device), clip_actions=agent.clip_actions)
    try:
        runner = (load_runner_cls("Beam-Drive-Straight") or MjlabOnPolicyRunner)(env, asdict(agent), device=args.device)
        runner.load(str(args.checkpoint.resolve()), load_cfg={"actor": True}, strict=True, map_location=args.device)
        args.output.mkdir(parents=True, exist_ok=True)
        # Official exporter includes the trained actor's observation normalizer.
        runner.export_policy_to_onnx(str(args.output.resolve()), filename="policy.onnx")
        import onnx
        model = onnx.load(str(args.output / "policy.onnx"))
        onnx.checker.check_model(model)
        shape = lambda v: [d.dim_value for d in v.type.tensor_type.shape.dim]
        if len(model.graph.input) != 1 or shape(model.graph.input[0]) != [1, 160]:
            raise ValueError("Exported actor must have one [1,160] input")
        if len(model.graph.output) != 1 or shape(model.graph.output[0]) != [1, 2]:
            raise ValueError("Exported actor must have one [1,2] output")
        manifest = {
            "task": "Beam-Drive-Straight", "history": HISTORY, "observation_size": 160,
            "action_scale": ACTION_SCALE, "control_dt": CONTROL_DT,
            "terms": list(TERMS), "history_order": "term-major-oldest-first",
            "checkpoint": args.checkpoint.name,
            "beam_task_sha256": hashlib.sha256(Path(beam_task.__file__).read_bytes()).hexdigest(),
        }
        (args.output / "policy.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"Exported policy.onnx and policy.json to {args.output.resolve()}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
