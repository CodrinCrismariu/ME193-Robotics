"""Regression checks for the hooked-arm CAD export. Run after build_beam_robot.py.

    ../.venv/bin/python -m unittest test_beam_geometry.py
"""
from pathlib import Path
import unittest

import mujoco
import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parent / "onshape"


class BeamGeometryTest(unittest.TestCase):
  def test_hooks_are_not_filled_by_collision_geometry(self):
    model = mujoco.MjModel.from_xml_path(str(ROOT / "beam_robot.xml"))
    for side in ("left", "right"):
      with self.subTest(side=side):
        surface = trimesh.load(ROOT / "assets" / f"{side}_beam_visual.stl", force="mesh")
        hulls = [trimesh.load(p, force="mesh")
                 for p in (ROOT / "assets").glob(f"{side}_beam_collision_*.stl")]
        self.assertTrue(surface.is_watertight)
        self.assertGreater(len(hulls), 1)
        # The current hooked arms occupy less than half their enclosing boxes.
        self.assertGreater(np.prod(surface.extents) / surface.volume, 2)
        self.assertAlmostEqual(sum(h.volume for h in hulls) / surface.volume, 1, delta=.03)
        bounds = np.array([h.bounds for h in hulls])
        np.testing.assert_allclose(bounds[:, 0].min(0), surface.bounds[0], atol=1e-5)
        np.testing.assert_allclose(bounds[:, 1].max(0), surface.bounds[1], atol=1e-5)
        # Mass belongs to the original CAD volume, not the enlarged collision hulls.
        self.assertAlmostEqual(model.body(f"{side}_beam").mass[0], surface.volume * 1050, delta=1e-8)
        geom = model.geom(f"{side}_beam_visual")
        self.assertEqual(geom.contype[0], 0)
        self.assertEqual(geom.conaffinity[0], 0)

  def test_corrected_scene_remains_finite(self):
    model = mujoco.MjModel.from_xml_path(str(ROOT / "beam_scene.xml"))
    data = mujoco.MjData(model)
    for _ in range(round(10 / model.opt.timestep)):
      data.ctrl[:] = 6
      mujoco.mj_step(model, data)
    self.assertTrue(np.isfinite(data.qpos).all())
    self.assertTrue(np.isfinite(data.qvel).all())
    self.assertGreater(data.qpos[2], 0)


if __name__ == "__main__":
  unittest.main()
