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
    foot_contacts_from_clip,
    register_clip_to_ground,
    write_biomech_asset,
)
from biomech.osim import parse_osim  # noqa: E402
from biomech.session import load_session  # noqa: E402
from biomech.tests import (  # noqa: E402
    CAL_C3D,
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
_ASSET_NAME_SPHERES = "mjcf/biomech_rajagopal_spheres.xml"
_ASSET_NAME_BOXES = "mjcf/biomech_rajagopal_boxes.xml"


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

    cache = np.load(_CACHE, allow_pickle=True)
    # Use the *enriched* spec that produced the cached poses (unlocked MTP + baked
    # ankle-neutral + added foot markers). Falling back to the stock model would weld
    # the MTP and drop the ankle-neutral bake, desyncing the asset from the poses.
    if "spec_pickle" in cache:
        spec = cache["spec_pickle"].item()
    else:
        print("NOTE: cache has no enriched spec; parsing stock model "
              "(regenerate the cache with make_s001_ik_figures.py --fresh).")
        spec = parse_osim(str(_OSIM))
    poses = np.asarray(cache["poses"], dtype=np.float64)  # (F, 37) DART q
    scales = np.asarray(cache["scales"], dtype=np.float64)  # (60,) group scales
    fps = float(cache["fps"])
    window = tuple(int(v) for v in cache["window"])  # (lo, hi) point frames
    print(f"S001 fit: {poses.shape[0]} frames @ {fps} fps, "
          f"{poses.shape[1]} DOFs, {scales.shape[0]} group-scale components")

    # Foot-flat correction is already applied to the cached poses (make_s001_ik_figures
    # rotates each foot flat about its ankle DOF after the fit), so the plantar sole is
    # planted in stance -- no export-time edit needed.

    # Belt speed (ground truth for treadmill-to-overground mapping) over the exact
    # window the clip covers. Split-belt: use the mean of the two belts as the
    # net whole-body forward belt speed (both belts read 1.5 m/s for S001 walk).
    belt_speed = _load_belt_speed(window)

    # 1. asset (regenerated with the S001 subject's group scales). Render each body with
    #    its actual OpenSim bone mesh(es) (visual-only) instead of capsule placeholders;
    #    bodies without a mesh fall back to capsules.
    asset_path = write_biomech_asset(
        spec,
        asset_file_name=_ASSET_NAME,
        group_scales=scales,
        coupled_knee="coupled",
        repo_root=_REPO,
        bone_meshes=True,
    )
    print(f"wrote asset -> {asset_path}")

    # 2. sim-body-aligned motion clip (same scales -> geometry matches the asset),
    #    mapped treadmill -> overground using the measured belt speed (TM2OG).
    clip = build_simbody_motion(
        spec, poses, fps=fps, group_scales=scales, coupled_knee="coupled",
        belt_speed=belt_speed,
    )

    # 2b. Ground registration (physiological/physical correctness for mimic training):
    #     the raw fit keeps the shod-treadmill capture height, so the subject floats/
    #     penetrates relative to the sim floor. Drop the clip onto the same
    #     flat-foot contact plane the validated hydroelastic GRF calibration used, so
    #     during stance the subject's plantar sole rests on z=0 (no float, no
    #     penetration). Uses the static trial (sole geometry) + measured belt GRF.
    if Path(CAL_C3D).exists() and Path(TRIAL_C3D).exists() and Path(LEFT_BELT).exists():
        static_session = load_session(str(CAL_C3D), filter_cutoff_hz=None)
        trial_session = load_session(
            str(TRIAL_C3D), left_belt_path=str(LEFT_BELT),
            right_belt_path=str(RIGHT_BELT), speedchange_path=str(SPEEDCHANGE),
        )
        ground = register_clip_to_ground(
            clip, spec, static_session, trial_session, window,
            group_scales=scales,
        )
        print(f"  ground registration: dropped clip by {ground:.4f} m "
              f"(stance sole now rests on z=0)")
    else:
        static_session = None
        print("NOTE: static/belt trials unavailable; skipping ground registration.")

    # 2c. Foot-ground collision variants. The visual bone meshes are non-colliding, so a
    #     physically simulated mimic character needs explicit foot collision geometry.
    #     Emit two comparable assets (same skeleton/FK/inertia + the shared motion clip):
    #     an OpenSim-style multi-sphere foot and a ProtoMotions-style single-box-per-body
    #     foot, both sized to the subject's real plantar sole so they touch z=0 in stance.
    if static_session is not None:
        from biomech.export.foot_collision import foot_collision_geoms

        for scheme, asset_name in (
            ("spheres", _ASSET_NAME_SPHERES),
            ("boxes", _ASSET_NAME_BOXES),
        ):
            geoms = foot_collision_geoms(spec, scales, scheme, static_session)
            path = write_biomech_asset(
                spec,
                asset_file_name=asset_name,
                group_scales=scales,
                coupled_knee="coupled",
                repo_root=_REPO,
                bone_meshes=True,
                collision_geoms=geoms,
            )
            n_box = sum(1 for g in geoms if g.kind == "box")
            n_sph = sum(1 for g in geoms if g.kind == "sphere")
            print(f"wrote {scheme} collision asset -> {path} "
                  f"({n_sph} spheres, {n_box} boxes)")
    else:
        print("NOTE: static trial unavailable; skipping foot collision variants.")

    # 2d. Per-body foot-contact labels for the mimic contact_match reward. Derived from
    #     the measured per-foot GRF (contact iff the foot is loaded) gated by each foot
    #     body's collision-geom height above the registered floor (real heel/toe timing).
    #     Uses the boxes collision asset (plantar surface identical to spheres, so contact
    #     timing matches either training variant). Without these the reference has no
    #     contact labels and contact_match crashes / must be disabled.
    if static_session is not None and "grf_R" in cache.files and "grf_L" in cache.files:
        grf_by_side = {"R": np.asarray(cache["grf_R"]), "L": np.asarray(cache["grf_L"])}
        boxes_path = _REPO / "protomotions" / "data" / "assets" / _ASSET_NAME_BOXES
        clip.data["rigid_body_contacts"] = foot_contacts_from_clip(
            clip, str(boxes_path), grf_by_side
        )
        rc = clip.data["rigid_body_contacts"]
        n_on = int(rc.sum())
        per_body = {clip.body_names[i]: int(rc[:, i].sum())
                    for i in range(rc.shape[1]) if int(rc[:, i].sum()) > 0}
        print(f"  foot contacts: {n_on} body-frames flagged across {rc.shape[0]} frames "
              f"-> {per_body}")
    else:
        print("NOTE: GRF / static trial unavailable; motion has no contact labels.")

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
