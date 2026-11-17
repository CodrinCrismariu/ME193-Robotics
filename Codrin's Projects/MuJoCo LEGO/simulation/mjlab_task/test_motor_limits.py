"""Check that training commands match the real LEGO adapter's speed conversion.

Run on the cluster with beam_task.py and onshape/ staged alongside this file.
"""
import unittest

import torch

from beam_task import MAX_BEAM_SPEED, MOTOR_COMMAND_LIMITS, beam_drive_env_cfg
from mjlab.envs import ManagerBasedRlEnv


class MotorLimitsTest(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    cfg = beam_drive_env_cfg()
    cfg.scene.num_envs = 1
    cls.env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
    cls.action = cls.env.action_manager.get_term("beams")

  @classmethod
  def tearDownClass(cls):
    cls.env.close()

  def check_command(self, requested):
    self.action.process_actions(torch.tensor([requested], device="cuda:0"))
    limits = [MOTOR_COMMAND_LIMITS[n] for n in self.action.target_names]
    expected = [round(max(-100, min(100, 100 * a * 21.0 / limit))) * limit / 100
                for a, limit in zip(requested, limits)]
    actual = self.action._processed_actions[0].tolist()
    for command, reference, limit in zip(actual, expected, limits):
      self.assertAlmostEqual(command, reference, places=5)
      self.assertLessEqual(abs(command), limit + 1e-6)
    torch.testing.assert_close(self.action.raw_action * MAX_BEAM_SPEED,
                               self.action._processed_actions)

  def test_large_policy_outputs_saturate_in_both_directions(self):
    self.check_command([10.0, -10.0])
    self.check_command([-10.0, 10.0])

  def test_submax_commands_match_whole_percent_hardware_commands(self):
    self.check_command([0.123, -0.234])
    self.check_command([-0.41, 0.32])

  def test_zero_and_reset_clear_previous_command(self):
    self.check_command([0.0, 0.0])
    self.check_command([1.0, 1.0])
    self.action.reset()
    self.assertEqual(self.action.raw_action.abs().sum().item(), 0)


if __name__ == "__main__":
  unittest.main()
