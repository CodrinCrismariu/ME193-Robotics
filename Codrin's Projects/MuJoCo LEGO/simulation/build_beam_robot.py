"""Turn the Onshape export of the double-motor robot into a model MuJoCo and mjlab can drive.

    python3 build_beam_robot.py        # reads onshape/robot.xml (the export), writes onshape/beam_robot.xml and beam_scene.xml

Why this is needed: in the Onshape assembly the two beams are placed against the motor but not mated, so the export has
three loose parts (0 degrees of freedom), no motors, and no masses (the parts have no material). This script measures the
export and builds the robot properly:

  * "chassis": the double motor. Its mesh is drawn as exported; it collides as a simple box (much faster for training).
  * "left_beam", "right_beam": each beam turns on a hinge joint about the motor's shaft axis, at the hub where it sits.
    Its original CAD surface is rendered, with separate convex collision pieces preserving concave hooked ends.
  * The robot is turned so that the shaft axis is the world y axis and a positive beam speed drives toward +x (forward),
    the same convention as the wheeled robot. Seen from behind, the beam on Onshape's +x hub is then on the left.

Run it again after every new export. If you later mate the beams in Onshape (revolute mates named dof_left_beam and
dof_right_beam) and give the parts materials, the export itself has joints and masses, and this script is not needed.

Build dependencies: pip install -r requirements-build.txt (only needed locally, not on the training cluster).
"""
import math
from pathlib import Path

import mujoco
import numpy as np
import coacd
import trimesh

HERE = Path(__file__).resolve().parent
EXPORT = HERE / "onshape" / "robot.xml"
OUT = HERE / "onshape" / "beam_robot.xml"
SCENE = HERE / "onshape" / "beam_scene.xml"

# ---- what the export does not say (no materials in Onshape): change to match your parts ----------------------------
MOTOR_MASS = 0.15            # kg, the double motor with its battery
BEAM_DENSITY = 1050          # kg/m^3 (ABS plastic), applied to the actual exported arm volume
FLOOR_FRICTION = {"chassis": 0.4, "beam": 0.8}   # sliding friction of the motor case and of the beam ends on the floor.
# (Their geoms get priority 1, so in a contact with the floor these values are used. Without it MuJoCo takes the larger
# of the two frictions, and the floor in mjlab is 1.0.)
GEARBOX_INERTIA = 2e-4       # kg*m^2 the motor's spinning rotor adds at the output shaft (rotor inertia x gear ratio^2).
                             # Real geared motors have far more of this than the light beams themselves.
# ---------------------------------------------------------------------------------------------------------------------


def world_vertices(m, d, geom):
  mid = m.geom_dataid[geom]
  v = m.mesh_vert[m.mesh_vertadr[mid]: m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
  return v @ d.geom_xmat[geom].reshape(3, 3).T + d.geom_xpos[geom]


def main():
  m = mujoco.MjModel.from_xml_path(str(EXPORT))
  d = mujoco.MjData(m)
  mujoco.mj_forward(m, d)
  meshes = {}   # mesh name -> (vertices in the Onshape frame, geom pos, geom quat, body name)
  for g in range(m.ngeom):
    if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH and m.geom_group[g] == 3:
      meshes[m.mesh(m.geom_dataid[g]).name] = (world_vertices(m, d, g), m.geom_pos[g].copy(), m.geom_quat[g].copy())
  # The motor is the detailed mesh; the two arms may be concave (e.g. hooked ends).
  motor_name = max(meshes, key=lambda k: len(meshes[k][0]))
  motor = meshes.pop(motor_name)[0]
  if len(meshes) != 2:
    raise SystemExit(f"expected the motor and two beams in {EXPORT}, found {[motor_name, *meshes]}")
  lo, hi = motor.min(0), motor.max(0)

  # Shaft axis: the round hubs are the parts of the motor that stick out furthest along x (within 3 mm of each end).
  hub = motor[motor[:, 0] > hi[0] - 0.003]
  axis_y, axis_z = (hub[:, 1].min() + hub[:, 1].max()) / 2, (hub[:, 2].min() + hub[:, 2].max()) / 2
  origin = np.array([(lo[0] + hi[0]) / 2, axis_y, axis_z])   # chassis origin: on the shaft axis, midway between hubs

  def to_robot(p):
    """Onshape frame -> robot frame: move the origin to the shaft centre, turn +90 deg about z (Onshape -y = forward)."""
    x, y, z = np.asarray(p) - origin
    return np.array([-y, x, z])

  quat_turn = np.array([math.cos(math.pi / 4), 0, 0, math.sin(math.pi / 4)])
  # The motor mesh geom pose as written in the export (its body sits at the Onshape origin). Read from the spec, not
  # the compiled model: compiling moves a mesh's origin to its centre of mass.
  spec = mujoco.MjSpec.from_file(str(EXPORT))
  spec_geom = next(g for g in spec.geoms
                  if g.type == mujoco.mjtGeom.mjGEOM_MESH and g.meshname == motor_name)
  q = np.zeros(4)
  mujoco.mju_mulQuat(q, quat_turn, np.asarray(spec_geom.quat, dtype=float))
  motor_geom_pos, motor_geom_quat = to_robot(spec_geom.pos), q
  box_center = to_robot((lo + hi) / 2)
  size = hi - lo
  box_half = np.array([size[1], size[0], size[2]]) / 2   # x and y swap with the turn

  beams = {}
  coacd.set_log_level("warn")
  for name, (v, _, _) in meshes.items():
    blo, bhi = v.min(0), v.max(0)
    side = "left" if blo[0] >= hi[0] - 0.001 else "right" if bhi[0] <= lo[0] + 0.001 else None
    if side is None:
      raise SystemExit(f"beam {name} is not against either end of the motor (x {blo[0]:.4f}..{bhi[0]:.4f})")
    hinge = to_robot([(blo[0] + bhi[0]) / 2, axis_y, axis_z])          # where the beam meets the shaft axis
    center = to_robot((blo + bhi) / 2) - hinge                          # beam centre relative to its hinge
    bsize = bhi - blo
    # Preserve the exported surface for rendering. MuJoCo collides a mesh as its convex hull,
    # so a single mesh would fill the space inside a hook. Split the collision surface into
    # convex pieces instead, retaining the hook and the shaft hole to the decomposition tolerance.
    source_geom = next(g for g in spec.geoms
                       if g.type == mujoco.mjtGeom.mjGEOM_MESH and g.meshname == name)
    source_mesh = next(mesh for mesh in spec.meshes if mesh.name == name)
    surface = trimesh.load(EXPORT.parent / "assets" / source_mesh.file, force="mesh")
    if not surface.is_watertight or surface.volume <= 0:
      raise ValueError(f"arm mesh {name} must be a closed surface with positive volume")
    rotation = np.zeros(9)
    mujoco.mju_quat2Mat(rotation, np.asarray(source_geom.quat, dtype=float))
    turn = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    transform = np.eye(4)
    transform[:3, :3] = turn @ rotation.reshape(3, 3)
    transform[:3, 3] = to_robot(source_geom.pos) - hinge
    surface.apply_transform(transform)
    visual_file = f"{side}_beam_visual.stl"
    surface.export(EXPORT.parent / "assets" / visual_file)
    parts = coacd.run_coacd(coacd.Mesh(surface.vertices, surface.faces), threshold=0.02,
                           preprocess_mode="off", resolution=3000, mcts_iterations=100, seed=0)
    collision_files = []
    for i, (vertices, faces) in enumerate(parts):
      filename = f"{side}_beam_collision_{i}.stl"
      trimesh.Trimesh(vertices, faces).export(EXPORT.parent / "assets" / filename)
      collision_files.append(filename)
    if not collision_files:
      raise ValueError(f"no collision pieces generated for {name}")
    beams[side] = dict(hinge=hinge, center=center, half=np.array([bsize[1], bsize[0], bsize[2]]) / 2,
                       length=max(bsize[1], bsize[2]), visual=visual_file, collisions=collision_files,
                       mass=surface.volume * BEAM_DENSITY, com=surface.center_mass,
                       inertia=surface.moment_inertia * BEAM_DENSITY)
  if set(beams) != {"left", "right"}:
    raise SystemExit(f"need one beam at each end of the motor, found {sorted(beams)}")

  f = lambda a: " ".join(f"{x:.6g}" for x in a)
  beam_bodies = ""
  beam_assets = ""
  for side, b in sorted(beams.items()):
    beam_assets += f'\n    <mesh name="{side}_beam_visual" file="{b["visual"]}"/>'
    collision_geoms = ""
    for i, filename in enumerate(b["collisions"]):
      mesh_name = f"{side}_beam_collision_{i}"
      geom_name = f"{side}_beam" if i == 0 else mesh_name
      beam_assets += f'\n    <mesh name="{mesh_name}" file="{filename}"/>'
      collision_geoms += f"""
        <geom name="{geom_name}" type="mesh" mesh="{mesh_name}" density="0" group="3"
              friction="{FLOOR_FRICTION['beam']} 0.005 0.0001" priority="1" rgba=".62 .81 .93 1"/>"""
    inertia = b["inertia"]
    beam_bodies += f"""
      <body name="{side}_beam" pos="{f(b['hinge'])}">
        <joint name="{side}_beam" type="hinge" axis="0 -1 0" armature="{GEARBOX_INERTIA}" damping="0.001"/>
        <inertial pos="{f(b['com'])}" mass="{b['mass']:.9g}"
                  fullinertia="{f([inertia[0,0], inertia[1,1], inertia[2,2], inertia[0,1], inertia[0,2], inertia[1,2]])}"/>
        <geom name="{side}_beam_visual" type="mesh" mesh="{side}_beam_visual"
              contype="0" conaffinity="0" density="0" group="2" rgba=".62 .81 .93 1"/>{collision_geoms}
      </body>"""
  def robot_xml(free: bool) -> str:
    """The robot's XML. free=True: it can move (a free joint), starting on the floor; for the hand-driving scene."""
    chassis_open = '<body name="chassis" pos="0 0 0.017">\n      <freejoint/>' if free else '<body name="chassis">'
    return f"""
  <compiler angle="radian" meshdir="assets" autolimits="true"/>
  <asset>
    <mesh name="motor" file="{motor_name}.stl"/>{beam_assets}
  </asset>
  <worldbody>
    {chassis_open}
      <geom name="motor_visual" type="mesh" mesh="motor" pos="{f(motor_geom_pos)}" quat="{f(motor_geom_quat)}"
            contype="0" conaffinity="0" density="0" group="2" rgba=".97 .53 0 1"/>
      <geom name="motor_case" type="box" pos="{f(box_center)}" size="{f(box_half)}" mass="{MOTOR_MASS}"
            friction="{FLOOR_FRICTION['chassis']} 0.005 0.0001" priority="1" group="3"/>{beam_bodies}
    </body>"""

  header = "<!-- Written by build_beam_robot.py from robot.xml (the Onshape export). Do not edit: change the script. -->"
  # The robot for mjlab: no free joint and no motors (the task adds both).
  OUT.write_text(f"""{header}
<mujoco model="beam_robot">{robot_xml(free=False)}
  </worldbody>
</mujoco>
""")
  # A scene to try the robot by hand: python -m mujoco.viewer --mjcf=onshape/beam_scene.xml (Control panel sliders =
  # beam speeds in rad/s). The same robot, free to move, on a floor, with speed motors like the ones training uses.
  SCENE.write_text(f"""{header}
<mujoco model="beam_scene">{robot_xml(free=True)}
    <light pos="0 0 3" dir="0 0 -1" directional="true"/>
    <geom name="floor" type="plane" size="0 0 0.05" rgba=".2 .3 .4 1"/>
  </worldbody>
  <option integrator="implicitfast"/>
  <actuator>
    <velocity name="left_beam" joint="left_beam" kv="0.05" forcerange="-0.1 0.1" ctrlrange="-10 10"/>
    <velocity name="right_beam" joint="right_beam" kv="0.05" forcerange="-0.1 0.1" ctrlrange="-10 10"/>
  </actuator>
</mujoco>
""")
  print(f"motor {size[0]*1000:.0f} x {size[1]*1000:.0f} x {size[2]*1000:.0f} mm, shaft {-box_center[2]*1000+box_half[2]*1000:.1f} mm "
        f"above the bottom of the case")
  for side, b in sorted(beams.items()):
    reach = np.abs(b["center"][[0, 2]]).max() + b["half"][[0, 2]].max()
    print(f"{side} beam: {b['length']*1000:.0f} mm long, reaches {reach*1000:.0f} mm from the shaft")
    print(f"  original CAD surface; {len(b['collisions'])} collision pieces; mass {b['mass']*1000:.2f} g")
  print(f"wrote {OUT.relative_to(HERE)} and {SCENE.relative_to(HERE)}")


if __name__ == "__main__":
  main()
