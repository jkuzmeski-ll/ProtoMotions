# SPDX-License-Identifier: MIT
"""Headless Newton drop test: confirm the foot-collision geoms stop the character.

Loads a biomech foot-collision MJCF variant exactly the way
``protomotions.simulator.newton.simulator.NewtonSimulator`` does (``add_mjcf`` with
``enable_self_collisions=False``, a ground plane, ``SolverMuJoCo``), drops the skeleton
from a small height, and steps ~1.5 s under gravity. If the foot geoms collide with the
ground the root settles at a plausible standing height and the lowest foot-collision shape
rests near z=0; if ``self_collisions=False`` had stripped the foot colliders (the concern
this checks), the body would fall through to large negative z.

Run::

    .venv/Scripts/python.exe projects/biomech/tools/check_sim_foot_contact.py boxes
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
_ASSETS = _REPO / "protomotions" / "data" / "assets"


def main(scheme: str = "boxes") -> int:
    import newton
    import numpy as np
    import warp as wp

    asset = _ASSETS / "mjcf" / (
        "biomech_rajagopal.xml" if scheme == "none"
        else f"biomech_rajagopal_{scheme}.xml"
    )
    if not asset.exists():
        print(f"asset not found: {asset}")
        return 1

    # --- build robot exactly like NewtonSimulator._create_envs ---
    robot = newton.ModelBuilder(up_axis=newton.Axis.Z)
    newton.solvers.SolverMuJoCo.register_custom_attributes(robot)
    robot.default_joint_cfg = newton.ModelBuilder.JointDofConfig(
        armature=0.1, target_ke=3000.0, target_kd=100.0
    )
    robot.default_shape_cfg.mu = 1.0
    robot.add_mjcf(
        str(asset),
        ignore_names=["floor", "ground"],
        ignore_classes=["wrap"],
        collapse_fixed_joints=False,
        floating=True,
        enable_self_collisions=False,
    )
    robot.approximate_meshes("convex_hull")

    builder = newton.ModelBuilder()
    builder.replicate(robot, 1)
    builder.add_ground_plane()
    model = builder.finalize()
    model.set_gravity((0.0, 0.0, -9.81))

    # --- drop from a small height, zero joint angles ---
    joint_q = wp.to_torch(model.joint_q).clone()
    # free-joint layout: [x, y, z, qx, qy, qz, qw]; start slightly above standing.
    joint_q[0:3] = 0.0
    joint_q[2] = 1.40  # feet start ~0.5 m above the floor for a clear drop
    joint_q[3:7] = joint_q.new_tensor([0.0, 0.0, 0.0, 1.0])
    joint_q[7:] = 0.0
    model.joint_q = wp.from_torch(joint_q, dtype=wp.float32)
    joint_qd = wp.to_torch(model.joint_qd).clone()
    joint_qd[:] = 0.0
    model.joint_qd = wp.from_torch(joint_qd, dtype=wp.float32)

    solver = newton.solvers.SolverMuJoCo(model, njmax=1000, nconmax=800, iterations=50)
    state_0 = model.state()
    state_1 = model.state()
    control = model.control()
    contacts = model.contacts()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)

    # foot-collision shape indices + their bodies (to read world z)
    lbl = model.shape_label
    shp_body = wp.to_torch(model.shape_body).cpu().numpy()
    shp_xform = wp.to_torch(model.shape_transform).cpu().numpy()  # local, per shape
    foot_shapes = [i for i in range(len(shp_body)) if "col_" in (lbl[i] or "")]

    def lowest_foot_z(state) -> float:
        bq = wp.to_torch(state.body_q).cpu().numpy()  # (nbody, 7) world transforms
        zmin = np.inf
        for i in foot_shapes:
            b = shp_body[i]
            # world pos of the shape origin = body_pos + R(body_quat) @ shape_local_pos
            bp, bq_ = bq[b, 0:3], bq[b, 3:7]
            lp = shp_xform[i, 0:3]
            # quat (x,y,z,w) rotate lp
            x, y, z, w = bq_
            # rotate via quaternion
            t = 2.0 * np.cross([x, y, z], lp)
            wp_ = bp + lp + w * t + np.cross([x, y, z], t)
            zmin = min(zmin, float(wp_[2]))
        return zmin

    dt = 1.0 / 200.0
    steps = 300  # 1.5 s
    root_z0 = float(wp.to_torch(state_0.body_q).cpu().numpy()[0, 2])
    min_foot_z = lowest_foot_z(state_0)
    for _ in range(steps):
        state_0.clear_forces()
        solver.step(state_0, state_1, control, contacts, dt)
        state_0, state_1 = state_1, state_0
        min_foot_z = min(min_foot_z, lowest_foot_z(state_0))

    root_z = float(wp.to_torch(state_0.body_q).cpu().numpy()[0, 2])
    foot_z = lowest_foot_z(state_0)
    print(f"scheme={scheme}  start root z={root_z0:.3f}")
    print(f"after {steps} steps ({steps*dt:.2f}s): root z={root_z:.3f}, "
          f"final lowest foot-collision z={foot_z*1e3:+.1f}mm, "
          f"min over drop={min_foot_z*1e3:+.1f}mm")
    # The decisive collision test: the feet must be STOPPED by the floor, not fall
    # through it. Without foot colliders the feet would free-fall ~1.5 m to deep
    # negative z; with them, the lowest foot geom bottoms out near z=0. (The passive
    # humanoid topples afterwards -- there is no balance controller -- which is why we
    # test the minimum foot depth over the drop rather than a final standing pose.)
    stopped = np.isfinite(min_foot_z) and min_foot_z > -0.03
    print("RESULT:", "feet contacted the ground (collision active)" if stopped
          else "feet fell through the floor (collision missing)")
    return 0 if stopped else 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else "boxes"))
