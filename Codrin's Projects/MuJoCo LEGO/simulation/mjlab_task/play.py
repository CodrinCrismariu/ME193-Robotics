"""Watch a trained policy (or a dummy one). Same as mjlab's `play`, with our tasks registered.
Works the same for Beam-Drive-Straight (the double motor with two beams).

    python play.py Drive-Straight --agent zero          # wheels do nothing: checks the robot loads
    python play.py Drive-Straight --agent random        # random wheel speeds
    python play.py Drive-Straight --checkpoint-file logs/rsl_rl/drive_straight/<run>/model_499.pt
"""
import importlib.util
import sys

if importlib.util.find_spec("drive_task"):
  import drive_task  # noqa: F401  (registers "Drive-Straight", the wheeled robot)
if importlib.util.find_spec("beam_task"):
  import beam_task  # noqa: F401  (registers "Beam-Drive-Straight", the double motor with two beams)
from mjlab.scripts.play import main

if __name__ == "__main__":
  sys.exit(main())
