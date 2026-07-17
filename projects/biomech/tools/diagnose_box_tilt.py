# SPDX-License-Identifier: MIT
"""Pin down the calcn_r box tilt at one loaded frame: geom_quat? motion rot vs recompute?"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

_BIOMECH = Path(__file__).resolve().parents[1]
_REPO = _BIOMECH.parents[1]
_ASSET_ROOT = _REPO / "protomotions" / "data" / "assets"
_ASSET = _ASSET_ROOT / "mjcf" / "biomech_rajagopal_boxes.xml"
_MOTION = _BIOMECH / "data" / "motions" / "biomech_s001_walk.motion"
_CACHE = _BIOMECH / "docs" / "figures" / "_s001_ik_cache.npz"


def main() -> int:
    import mujoco
    import torch
    from biomech.contact.kinematics import foot_trajectory_from_motion
    from biomech.export.mjcf import dart_q_to_mjcf_qpos
    from biomech.export.protomotions_robot import build_biomech_robot_config

    FR = 44
    model = mujoco.MjModel.from_xml_path(str(_ASSET))
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "col_calcn_r_box")
    print(f"box geom_pos={model.geom_pos[gid]}")
    print(f"box geom_quat={model.geom_quat[gid]}  (wxyz; identity=[1,0,0,0])")
    print(f"box geom_size={model.geom_size[gid]}")

    data = torch.load(str(_MOTION), weights_only=False)
    cfg = build_biomech_robot_config(
        asset_file_name="mjcf/biomech_rajagopal_boxes.xml", asset_root=str(_ASSET_ROOT))
    clip = SimpleNamespace(body_names=cfg.kinematic_info.body_names, data=data)
    pos, quat, _, _ = foot_trajectory_from_motion(clip, "calcn_r")

    def q2m(q):  # xyzw
        x, y, z, w = q
        return np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])

    Rm = q2m(quat[FR])
    calcn_y_world = Rm @ np.array([0.0, 1.0, 0.0])
    print(f"\n.motion calcn_r frame {FR}: calcn+y (local[0,1,0]) in world = {calcn_y_world}")
    print(f"  pitch atan2(x,z) = {np.degrees(np.arctan2(calcn_y_world[0], calcn_y_world[2])):.2f} deg")
    print(f"  (flat if +y ~ world +z, i.e. [~0,~0,~1])")

    # recompute rot from cache poses via dart_q_to_mjcf_qpos + Y-up->Z-up
    from biomech.export.motion import R_OS2PM
    cache = np.load(_CACHE, allow_pickle=True)
    spec = cache["spec_pickle"].item()
    poses = np.asarray(cache["poses"], dtype=np.float64)
    scales = np.asarray(cache["scales"], dtype=np.float64)
    m2 = mujoco.MjModel.from_xml_string(
        __import__("biomech.export.mjcf", fromlist=["export_mjcf"]).export_mjcf(
            spec, group_scales=scales, coupled_knee="coupled").xml)
    d2 = mujoco.MjData(m2)
    cid = mujoco.mj_name2id(m2, mujoco.mjtObj.mjOBJ_BODY, "calcn_r")
    qp = dart_q_to_mjcf_qpos(spec, poses[FR], scales, "coupled")
    x, y, z, w = qp[3:7]; qp[3:7] = (w, x, y, z)
    d2.qpos[:] = qp
    mujoco.mj_kinematics(m2, d2)
    R_yup = d2.xmat[cid].reshape(3, 3)
    R_zup = R_OS2PM @ R_yup
    cy = R_zup @ np.array([0.0, 1.0, 0.0])
    print(f"\nrecompute calcn_r frame {FR}: calcn+y in Z-up world = {cy}")
    print(f"  pitch atan2(x,z) = {np.degrees(np.arctan2(cy[0], cy[2])):.2f} deg")

    # box bottom corner world heights from the .motion
    signs = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1)
                      for sz in (-1, 1)], float)
    corners = model.geom_pos[gid] + signs * model.geom_size[gid]
    wz = (pos[FR] + corners @ Rm.T)[:, 2]
    print(f"\nbox bottom corner world z (mm): {np.sort(wz)[:4]*1e3}")
    print(f"  spread = {(wz.max()-wz.min())*1e3:.1f} mm")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
