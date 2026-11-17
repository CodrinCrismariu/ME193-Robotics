"""Train our tasks. Same as mjlab's own `train`, but it registers our tasks first.

    python train.py Drive-Straight --env.scene.num-envs 4096          # the wheeled robot
    python train.py Beam-Drive-Straight --env.scene.num-envs 4096     # the double motor with two beams
"""
import importlib.util
import sys

if importlib.util.find_spec("drive_task"):
  import drive_task  # noqa: F401  (registers "Drive-Straight", the wheeled robot)
if importlib.util.find_spec("beam_task"):
  import beam_task  # noqa: F401  (registers "Beam-Drive-Straight", the double motor with two beams)
from mjlab.scripts.train import main

if __name__ == "__main__":
  sys.exit(main())
