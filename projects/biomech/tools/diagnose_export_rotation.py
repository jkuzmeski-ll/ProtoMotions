# SPDX-License-Identifier: MIT
"""Isolate a foot-orientation discrepancy between WarpSkeleton FK and the MJCF export.

The cached fit poses give a flat foot during stance under WarpSkeleton FK, but the
exported .motion foot reads ~10 deg toe-down. Rotations are untouched by TM2OG/ground
registration (translations only), so any rotation difference comes from the MJCF FK path
(dart_q_to_mjcf_qpos + MuJoCo) vs the Warp FK. This compares calcn_r +y pitch from both
paths on the SAME cached poses, frame by frame.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

_BIOMECH = Path(__file__).resolve().parents[1]
_REPO = _BIOMECH.parents[1]
_CACHE = _BIOMECH / "docs" / "figures" / "_s001_ik_cache.npz"
_ASSET = _REPO / "protomotions" / "data" / "assets" / "mjcf" / "biomech_rajagopal.xml"
DEG = 180.0 / np.pi


def main() -> int:
    import mujoco
    from biomech.export.mjcf import dart_q_to_mjcf_qpos
    from biomech.export.motion import R_OS2PM
    from biomech.skeleton.skeleton import WarpSkeleton

    cache = np.load(_CACHE, allow_pickle=True)
    spec = cache["spec_pickle"].item()
    poses = np.asarray(cache["poses"], dtype=np.float64)
    scales = np.asarray(cache["scales"], dtype=np.float64)
    F = poses.shape[0]

    # --- WarpSkeleton FK (OpenSim Y-up) ---
    skel = WarpSkeleton(spec, device="cpu")
    world, _ = skel.forward(poses, scales)
    bidx = {b.name: i for i, b in enumerate(spec.bodies)}
    Rw = np.asarray(world)[:, bidx["calcn_r"], :3, :3]

    # --- MJCF/MuJoCo FK (as build_simbody_motion does) ---
    model = mujoco.MjModel.from_xml_path(str(_ASSET))
    data = mujoco.MjData(model)
    cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "calcn_r")

    def sag_yup(n):  # OpenSim Y-up sagittal pitch (toe-down +)
        return float(np.degrees(np.arctan2(n[0], n[1])))

    def sag_zup(n):  # Z-up: up is +z, fwd is +x
        return float(np.degrees(np.arctan2(n[0], n[2])))

    warp_p = np.array([sag_yup(Rw[f] @ [0, 1, 0]) for f in range(F)])
    mj_p = np.empty(F)
    mj_zup_p = np.empty(F)
    for f in range(F):
        qp = dart_q_to_mjcf_qpos(spec, poses[f], scales, "coupled")
        x, y, z, w = qp[3:7]; qp[3:7] = (w, x, y, z)
        data.qpos[:] = qp
        mujoco.mj_kinematics(model, data)
        Rm = data.xmat[cid].reshape(3, 3)  # Y-up (OpenSim)
        mj_p[f] = sag_yup(Rm @ [0, 1, 0])
        # same rotation after Y-up -> Z-up bake the exporter applies
        Rz = R_OS2PM @ Rm
        mj_zup_p[f] = sag_zup(Rz @ (R_OS2PM @ [0, 1, 0]))

    diff = mj_p - warp_p
    print(" frame   warp_pitch(Yup)   mjcf_pitch(Yup)   diff")
    for f in range(20, 84, 4):
        print(f" {f:4d}     {warp_p[f]:8.2f}         {mj_p[f]:8.2f}      {diff[f]:6.2f}")
    print(f"\ncalcn_r +y pitch: WarpSkeleton vs MuJoCo(dart_q_to_mjcf_qpos)")
    print(f"  mean diff = {diff.mean():.2f} deg (min {diff.min():.2f}, max {diff.max():.2f})")
    print(f"  std diff  = {diff.std():.2f} deg")
    if abs(diff.mean()) > 1.0:
        print("  => EXPORT PATH ROTATES THE FOOT: the MJCF FK disagrees with the Warp FK.")
    else:
        print("  => FK paths agree; the toe-down is elsewhere (geom/registration).")

    # --- cross-check against the stored .motion rotations directly ---
    import torch
    from biomech.export.protomotions_robot import build_biomech_robot_config
    motion = _BIOMECH / "data" / "motions" / "biomech_s001_walk.motion"
    clip = torch.load(str(motion), weights_only=False)
    rot = np.asarray(clip["rigid_body_rot"])  # (F,B,4) xyzw, Z-up
    cfg = build_biomech_robot_config(
        asset_root=str(_REPO / "protomotions" / "data" / "assets"))
    ci = cfg.kinematic_info.body_names.index("calcn_r")

    def q2m(q):
        x, y, z, w = q
        return np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])

    print("\n .motion stored calcn_r pitch (Z-up: +y bone axis -> world; toe-down +)")
    up_yup = R_OS2PM @ np.array([0.0, 1.0, 0.0])  # calcn +y expressed after Z-up bake
    for f in range(20, 84, 8):
        n = q2m(rot[f, ci]) @ up_yup
        pit = np.degrees(np.arctan2(n[0], n[2]))  # Z-up sagittal
        print(f" {f:4d}   motion_pitch={pit:6.2f}   warp_pitch={warp_p[f]:6.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
