# SPDX-License-Identifier: MIT

"""Check the exported foot-ground collision variants against the registered motion clip.

For each collision asset (spheres / boxes), loads the MJCF, drives each foot body with
the exported clip's world pose, and reports the lowest world-z reached by that body's
*colliding* geoms during stance vs. swing. A physically-correct reference should have the
collision surface reach ~0 (floor) while the foot is planted and clear well above it in
swing. Also validates that each variant still builds a matching ProtoMotions RobotConfig.

Run from the repo root::

    .venv/Scripts/python.exe projects/biomech/tools/check_foot_collision.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # projects/

import mujoco  # noqa: E402
import torch  # noqa: E402

from biomech.contact.elastic_foundation import _quat_rotate_np  # noqa: E402
from biomech.contact.kinematics import foot_trajectory_from_motion  # noqa: E402
from biomech.export.protomotions_robot import build_biomech_robot_config  # noqa: E402

_ASSET_ROOT = Path("protomotions/data/assets")
_MJCF = _ASSET_ROOT / "mjcf"
_MOTION = Path("projects/biomech/data/motions/biomech_s001_walk.motion")
# Measured right-foot stance windows (RHEE height), not the mislabeled GRF pick.
_R_STANCE = list(range(10, 26)) + list(range(115, 131))


def _lowest_world_z(m, bid, pos, quat):
    """Per-frame lowest world z over a body's colliding geoms (exact for spheres)."""
    F = pos.shape[0]
    minz = np.full(F, np.inf)
    for g in range(m.ngeom):
        if m.geom_bodyid[g] != bid or m.geom_contype[g] == 0:
            continue
        gp = np.asarray(m.geom_pos[g], float)
        gs = np.asarray(m.geom_size[g], float)
        if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE:
            pts, r = np.array([gp]), float(gs[0])  # sphere low point = center_z - r
        else:
            hx, hy, hz = gs[:3]
            pts = np.array([gp + [sx * hx, sy * hy, sz * hz]
                            for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
            r = 0.0
        pl = np.broadcast_to(pts, (F, pts.shape[0], 3))
        wz = (pos[:, None, :] + _quat_rotate_np(quat[:, None, :], pl))[:, :, 2]
        minz = np.minimum(minz, wz.min(axis=1) - r)
    return minz


def main() -> int:
    data = torch.load(str(_MOTION), weights_only=False)
    cfg = build_biomech_robot_config(
        asset_file_name="mjcf/biomech_rajagopal.xml",
        asset_root=str(_ASSET_ROOT.resolve()),
    )
    clip = SimpleNamespace(body_names=cfg.kinematic_info.body_names, data=data)

    for asset in ("biomech_rajagopal_spheres.xml", "biomech_rajagopal_boxes.xml"):
        path = _MJCF / asset
        if not path.exists():
            print(f"MISSING: {path} (run tools/export_s001_subject.py)")
            continue
        m = mujoco.MjModel.from_xml_path(str(path))
        n_col = int(np.count_nonzero(m.geom_contype))
        print(f"\n== {asset}  (ngeom={m.ngeom}, colliding={n_col})")
        for body in ("calcn_r", "toes_r"):
            bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, body)
            pos, quat, _, _ = foot_trajectory_from_motion(clip, body)
            minz = _lowest_world_z(m, bid, pos, quat)
            stance = minz[_R_STANCE]
            print(f"  {body:9s}  stance min={stance.min()*1000:6.1f} mm  "
                  f"overall min={minz.min()*1000:6.1f} mm  "
                  f"swing max={minz.max()*1000:6.1f} mm")
        # config still valid for the variant asset (same body order / action count)
        vcfg = build_biomech_robot_config(
            asset_file_name=f"mjcf/{asset}", asset_root=str(_ASSET_ROOT.resolve())
        )
        ok = vcfg.kinematic_info.body_names == cfg.kinematic_info.body_names
        print(f"  RobotConfig OK: body order match={ok}, actions={vcfg.number_of_actions}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
