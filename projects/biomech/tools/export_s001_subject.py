# SPDX-License-Identifier: MIT

"""Export the fitted S001 subject as a runnable ProtoMotions Newton imitation subject (M8).

Takes the cached S001 gold-standard fit (`docs/figures/_s001_ik_cache.npz`, produced by
`tools/make_s001_ik_figures.py`) and emits, self-consistently (same group scales for the
skeleton geometry and the motion clip):

  1. the fitted-skeleton MJCF asset at
     `protomotions/data/assets/mjcf/biomech_rajagopal.xml` (the file the registered
     `biomech` robot config points at), and
  2. a sim-body-aligned `.motion` clip at
     `projects/biomech/data/motions/biomech_s001_walk.motion` whose `rigid_body_*`
     arrays line up 1:1 with the robot's `kinematic_info` (dummy bodies included).

Run from the repo root::

    .venv/Scripts/python.exe projects/biomech/tools/export_s001_subject.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # projects/

import torch  # noqa: E402

from biomech.export.protomotions_robot import (  # noqa: E402
    build_simbody_motion,
    write_biomech_asset,
)
from biomech.osim import parse_osim  # noqa: E402
from biomech.session import load_session  # noqa: E402
from biomech.tests import (  # noqa: E402
    LEFT_BELT,
    RIGHT_BELT,
    SPEEDCHANGE,
    TRIAL_C3D,
)

_BIOMECH = Path(__file__).resolve().parents[1]
_REPO = _BIOMECH.parents[1]
_OSIM = _BIOMECH / "models" / "rajagopal_data" / "Rajagopal2015.osim"
_CACHE = _BIOMECH / "docs" / "figures" / "_s001_ik_cache.npz"
_MOTION_OUT = _BIOMECH / "data" / "motions" / "biomech_s001_walk.motion"
_ASSET_NAME = "mjcf/biomech_rajagopal.xml"


def _load_belt_speed(window: tuple[int, int]) -> np.ndarray | None:
    """Net belt speed (m/s) over the clip's point-frame window, or None.

    Ground truth for TM2OG. Returns the mean of the two belt traces (split-belt),
    sliced to ``window`` and length-matched to the clip. Returns None if the belt
    logs are unavailable so the exporter still produces an (in-place) clip.
    """
    lo, hi = window
    if not (Path(TRIAL_C3D).exists() and Path(LEFT_BELT).exists()):
        print("NOTE: belt logs not found; exporting in-place (no TM2OG).")
        return None
    session = load_session(
        str(TRIAL_C3D),
        left_belt_path=str(LEFT_BELT),
        right_belt_path=str(RIGHT_BELT),
        speedchange_path=str(SPEEDCHANGE),
    )
    sides = [session.belt_speed_point[s][lo:hi] for s in ("left", "right")
             if s in session.belt_speed_point]
    belt = np.nanmean(np.stack(sides, axis=0), axis=0)
    print(f"belt speed over window {window}: "
          f"min/mean/max = {belt.min():.3f}/{belt.mean():.3f}/{belt.max():.3f} m/s")
    return belt


def main() -> int:
    if not _CACHE.exists():
        print(f"ERROR: S001 fit cache missing: {_CACHE}")
        print("Run: .venv/Scripts/python.exe projects/biomech/tools/make_s001_ik_figures.py --fresh")
        return 1

    spec = parse_osim(str(_OSIM))
    cache = np.load(_CACHE, allow_pickle=True)
    poses = np.asarray(cache["poses"], dtype=np.float64)  # (F, 37) DART q
    scales = np.asarray(cache["scales"], dtype=np.float64)  # (60,) group scales
    fps = float(cache["fps"])
    window = tuple(int(v) for v in cache["window"])  # (lo, hi) point frames
    print(f"S001 fit: {poses.shape[0]} frames @ {fps} fps, "
          f"{poses.shape[1]} DOFs, {scales.shape[0]} group-scale components")

    # Belt speed (ground truth for treadmill-to-overground mapping) over the exact
    # window the clip covers. Split-belt: use the mean of the two belts as the
    # net whole-body forward belt speed (both belts read 1.5 m/s for S001 walk).
    belt_speed = _load_belt_speed(window)

    # 1. asset (regenerated with the S001 subject's group scales)
    asset_path = write_biomech_asset(
        spec,
        asset_file_name=_ASSET_NAME,
        group_scales=scales,
        coupled_knee="coupled",
        repo_root=_REPO,
    )
    print(f"wrote asset -> {asset_path}")

    # 2. sim-body-aligned motion clip (same scales -> geometry matches the asset),
    #    mapped treadmill -> overground using the measured belt speed (TM2OG).
    clip = build_simbody_motion(
        spec, poses, fps=fps, group_scales=scales, coupled_knee="coupled",
        belt_speed=belt_speed,
    )
    _MOTION_OUT.parent.mkdir(parents=True, exist_ok=True)
    torch.save(clip.data, str(_MOTION_OUT))
    pelvis = clip.data["rigid_body_pos"][:, 0, :]
    pelvis_z = pelvis[:, 2]
    travel = pelvis[-1] - pelvis[0]
    horiz_travel = float((travel[:2] ** 2).sum() ** 0.5)
    expected = float(belt_speed[:-1].mean() * (poses.shape[0] - 1) / fps) if belt_speed is not None else 0.0
    print(f"wrote motion -> {_MOTION_OUT}")
    print(f"  bodies={len(clip.body_names)}  frames={clip.data['rigid_body_pos'].shape[0]}  "
          f"pelvis z-up range=[{float(pelvis_z.min()):.3f}, {float(pelvis_z.max()):.3f}] m")
    if belt_speed is not None:
        print(f"  TM2OG: pelvis horiz travel={horiz_travel:.3f} m "
              f"(belt integral v dt ~= {expected:.3f} m)")

    # 3. sanity: clip body order matches a RobotConfig built from the same asset
    try:
        from biomech.export.protomotions_robot import build_biomech_robot_config

        cfg = build_biomech_robot_config(asset_file_name=_ASSET_NAME, asset_root=str(_REPO / "protomotions" / "data" / "assets"))
        assert clip.body_names == cfg.kinematic_info.body_names, "body order mismatch"
        print(f"validated: clip body order == robot kinematic_info "
              f"({cfg.number_of_actions} actions, anchor '{cfg.anchor_body_name}')")
    except Exception as exc:  # noqa: BLE001
        print(f"NOTE: robot-config validation skipped/failed: {exc}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
