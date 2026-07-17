# SPDX-License-Identifier: MIT
#
# Milestone M8 (Newton imitation env) bridge: turn a fitted biomech subject into a
# runnable ProtoMotions setup on the Newton simulator. This is the M3->train handoff the
# motion exporter flagged ("wiring this clip to a concrete ProtoMotions robot ... needs
# that robot's config; this module produces the clip").
#
# Three artifacts, all Windows-native and testable without launching training:
#   1. the fitted skeleton MJCF written into ProtoMotions' asset tree,
#   2. a ProtoMotions ``.motion`` clip whose ``rigid_body_*`` arrays align 1:1 with the
#      simulator's body set, and
#   3. a validated ``RobotConfig`` for the fitted skeleton.
#
# The body-set crux (learned here): the M3 MJCF splits every multi-DOF joint into a chain
# of massless dummy bodies (``<body>__q0`` ...), so ProtoMotions' ``extract_kinematic_info``
# reports MORE bodies (e.g. 38) than the 20 anatomical bodies ``build_motion`` emits. A
# mimic env needs the clip's body order to match the sim body order exactly. We therefore
# build the clip's body transforms with **MuJoCo forward kinematics on the exact exported
# MJCF** (``mj_kinematics`` -> ``xpos``/``xmat`` for every sim body incl. dummies), which
# is bit-exact vs the Warp skeleton (validated in M3) and guaranteed to align with the
# robot's ``kinematic_info``. Frames are converted OpenSim Y-up -> ProtoMotions Z-up.

"""Biomech fit -> runnable ProtoMotions Newton imitation setup (M8 bridge)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from biomech.export.motion import (
    MotionExportResult,
    R_OS2PM,
    _angular_velocity,
    _finite_diff_lin,
    _matrix_to_quat_xyzw,
)

# Default ProtoMotions asset root (matches RobotAssetConfig.asset_root).
PM_ASSET_ROOT = "protomotions/data/assets"
_ASSET_SUBDIR = "mjcf"

# Semantic body maps for the Rajagopal2015 skeleton (grounded in its body names).
# The model has no dedicated head body, so the topmost trunk body ('torso') stands in for
# both head and torso semantics (documented, not invented anatomy).
RAJAGOPAL_BODY_MAP = {
    "all_left_foot_bodies": ["calcn_l", "toes_l", "talus_l"],
    "all_right_foot_bodies": ["calcn_r", "toes_r", "talus_r"],
    "all_left_hand_bodies": ["hand_l"],
    "all_right_hand_bodies": ["hand_r"],
    "head_body_name": ["torso"],
    "torso_body_name": ["torso"],
}


@dataclass
class ProtoMotionsBundle:
    """Paths + metadata for a ProtoMotions-ready biomech subject."""

    asset_path: Path  # written MJCF (absolute)
    asset_file_name: str  # relative to the ProtoMotions asset root
    motion_path: Optional[Path]  # written .motion clip (if requested)
    body_names: list[str]  # sim body order of the motion clip
    dof_names: list[str]
    fps: float


# ---------------------------------------------------------------------------
# 1. asset
# ---------------------------------------------------------------------------
def write_biomech_asset(
    spec,
    asset_file_name: str = "mjcf/biomech_rajagopal.xml",
    group_scales: Optional[np.ndarray] = None,
    coupled_knee: str = "coupled",
    asset_root: str = PM_ASSET_ROOT,
    repo_root: Optional[str | Path] = None,
    visual_geoms: bool = True,
    subject_mass: Optional[float] = None,
    bone_meshes: bool = False,
    collision_geoms: Optional[list] = None,
) -> Path:
    """Export the fitted skeleton MJCF into the ProtoMotions asset tree.

    Returns the absolute path written. ``asset_file_name`` is the path relative to
    ``asset_root`` used by :class:`RobotAssetConfig`. ``visual_geoms`` (default True) adds
    non-colliding capsule/sphere bones so the robot is visible in the renderer.
    ``bone_meshes`` (default False), if True, renders each body with its actual OpenSim
    bone mesh(es) instead of capsule placeholders (requires the converted STL meshes under
    ``<asset_root>/mesh/biomech_rajagopal/`` -- see ``tools/convert_bone_meshes.py``).
    ``subject_mass`` (kg), if given, rescales body masses/inertias so the robot's whole-body
    mass matches the subject (anthropometric mass on top of ``group_scales`` geometry).
    ``collision_geoms`` (optional), if given, adds colliding foot-ground geoms (see
    :mod:`biomech.export.foot_collision`) so a physically simulated character makes contact.
    """
    from biomech.export.mjcf import export_mjcf

    mesh_map = None
    if bone_meshes:
        from biomech.export.bone_geometry import (
            bone_mesh_dir,
            bone_meshes_available,
            default_bone_geometry,
        )

        root = Path(repo_root) if repo_root is not None else Path.cwd()
        resolved_asset_root = root / asset_root
        if bone_meshes_available(resolved_asset_root):
            mesh_map = default_bone_geometry()
        else:
            print(
                "NOTE: bone_meshes requested but converted STL meshes were not found at "
                f"{bone_mesh_dir(resolved_asset_root)}; falling back to capsule bones. "
                "Run: python projects/biomech/tools/convert_bone_meshes.py"
            )

    res = export_mjcf(
        spec,
        group_scales=group_scales,
        coupled_knee=coupled_knee,
        visual_geoms=visual_geoms,
        subject_mass=subject_mass,
        bone_meshes=mesh_map,
        collision_geoms=collision_geoms,
    )
    root = Path(repo_root) if repo_root is not None else Path.cwd()
    out = root / asset_root / asset_file_name
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(res.xml)
    return out


# ---------------------------------------------------------------------------
# 2. sim-body-aligned motion clip (MuJoCo FK over the full body set)
# ---------------------------------------------------------------------------
def build_simbody_motion(
    spec,
    q_t: np.ndarray,
    fps: float,
    group_scales: Optional[np.ndarray] = None,
    coupled_knee: str = "coupled",
    mjcf_xml: Optional[str] = None,
    belt_speed: Optional[np.ndarray] = None,
) -> MotionExportResult:
    """Build a ProtoMotions motion clip aligned to the **simulator** body set.

    Unlike :func:`biomech.export.motion.build_motion` (which emits the 20 anatomical
    bodies), this emits every body MuJoCo sees in the exported MJCF -- including the
    massless dummy bodies the exporter inserts for multi-DOF joints -- so the clip's
    ``rigid_body_*`` arrays line up 1:1 with a ``RobotConfig.kinematic_info`` built from
    the same MJCF. Body world transforms come from MuJoCo forward kinematics on that MJCF
    (bit-exact vs the Warp skeleton), then Y-up -> Z-up.

    If ``belt_speed`` (a per-frame ``(F,)`` belt-speed trace in m/s) is given, the clip
    is mapped from treadmill to overground via :func:`biomech.export.tm2og.tm2og_motion`
    (virtual-origin method): the forward displacement ``∫ v_belt dt`` is added to every
    body position and ``v_belt`` to every body linear velocity, along the forward
    direction inferred from the stance feet. Rotations/DOFs are untouched.
    """
    import mujoco

    from biomech.export.mjcf import dart_q_to_mjcf_qpos, export_mjcf

    q_t = np.asarray(q_t, dtype=np.float64)
    if q_t.ndim == 1:
        q_t = q_t[None, :]
    F = q_t.shape[0]
    dt = 1.0 / float(fps)

    if mjcf_xml is None:
        mjcf_xml = export_mjcf(spec, group_scales=group_scales, coupled_knee=coupled_knee).xml
    model = mujoco.MjModel.from_xml_string(mjcf_xml)
    data = mujoco.MjData(model)

    # sim body order excluding MuJoCo's implicit 'world' body (id 0)
    body_ids = list(range(1, model.nbody))
    body_names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) for b in body_ids
    ]
    nb = len(body_ids)

    pos = np.zeros((F, nb, 3), dtype=np.float64)
    rot = np.zeros((F, nb, 3, 3), dtype=np.float64)
    for f in range(F):
        qp = dart_q_to_mjcf_qpos(spec, q_t[f], group_scales, coupled_knee)
        # dart_q_to_mjcf_qpos emits the free-root quat as xyzw (Newton); MuJoCo's free
        # joint expects wxyz -- swap it or the whole body renders mis-oriented.
        x, y, z, w = qp[3:7]
        qp[3:7] = (w, x, y, z)
        data.qpos[:] = qp
        mujoco.mj_kinematics(model, data)
        for i, b in enumerate(body_ids):
            pos[f, i] = data.xpos[b]
            rot[f, i] = data.xmat[b].reshape(3, 3)

    # OpenSim Y-up -> ProtoMotions/Newton Z-up
    pos = np.einsum("ij,fnj->fni", R_OS2PM, pos)
    rot = np.einsum("ij,fnjk->fnik", R_OS2PM, rot)

    quat = _matrix_to_quat_xyzw(rot)
    lin_vel = _finite_diff_lin(pos, dt)
    ang_vel = _angular_velocity(rot, dt)

    qpos = np.stack(
        [dart_q_to_mjcf_qpos(spec, q_t[f], group_scales, coupled_knee) for f in range(F)]
    )
    dof_pos = qpos[:, 7:]
    dof_vel = _finite_diff_lin(dof_pos, dt)

    import torch

    def t32(a):
        return torch.as_tensor(np.asarray(a, dtype=np.float32))

    data_dict = {
        "rigid_body_pos": t32(pos),
        "rigid_body_rot": t32(quat),
        "rigid_body_vel": t32(lin_vel),
        "rigid_body_ang_vel": t32(ang_vel),
        "dof_pos": t32(dof_pos),
        "dof_vel": t32(dof_vel),
        "fps": float(fps),
    }
    if belt_speed is not None:
        from biomech.export.tm2og import tm2og_motion

        tm2og_motion(data_dict, np.asarray(belt_speed), float(fps), body_names)

    return MotionExportResult(
        data=data_dict, body_names=body_names, dof_names=[], fps=float(fps)
    )


def register_clip_to_ground(
    clip: MotionExportResult,
    spec,
    static_session,
    trial_session,
    window: tuple,
    group_scales: Optional[np.ndarray] = None,
    fz_threshold: float = 50.0,
    right_plate_x_sign: Optional[int] = None,
    penetration: float = 0.0,
) -> float:
    """Ground-register a Z-up sim-body clip onto the physically-validated contact plane.

    Reuses the calibrated contact machinery (the same registration that drove the
    hydroelastic GRF fit to ~1% on this walk): build the subject's plantar sole from the
    static trial, extract each foot's world trajectory from the clip, and register the
    ground from genuinely *flat-foot* frames (planted, near-horizontal, low vertical
    speed, high measured load) via the median lowest sole point. Both feet are registered
    independently and the clip is dropped onto their mean plane (a split-belt treadmill's
    two belts are at the same physical height), so during stance the subject's sole rests
    on the sim floor (z=0) with neither float nor penetration -- the correct vertical
    datum for a physically-plausible mimic reference. Returns the applied z shift (m).

    ``right_plate_x_sign`` selects which force plate is the right foot; when ``None`` it is
    auto-detected (the assignment under which each foot's sole is lower while loaded than
    unloaded), since the default lab convention is flipped for some captures (e.g. S001).
    A constant z-shift leaves rotations and velocities untouched.
    """
    from biomech.contact.foot_geometry import subject_sole_from_session
    from biomech.contact.kinematics import foot_trajectory_from_motion
    from biomech.contact.pipeline import (
        detect_right_plate_x_sign,
        measured_belt_grf,
    )
    from biomech.contact.stance import (
        flat_foot_mask,
        register_ground_flatfoot,
    )

    lo, hi = window
    # Pre-compute each foot's sole + trajectory once for the registration below.
    soles, trajs = {}, {}
    for side, body in (("R", "calcn_r"), ("L", "calcn_l")):
        if body not in clip.body_names:
            continue
        soles[side] = subject_sole_from_session(
            static_session, spec, side, group_scales=group_scales
        )
        pos, quat, linvel, _ = foot_trajectory_from_motion(clip, body)
        trajs[side] = (pos, quat, linvel)

    # Auto-detect the belt->foot assignment (the default lab convention is flipped for
    # some captures, e.g. S001) so registration is gated on the correct foot.
    if right_plate_x_sign is None:
        right_plate_x_sign = detect_right_plate_x_sign(
            trial_session, static_session, spec, clip, window,
            group_scales=group_scales, fz_threshold=fz_threshold,
        )
    belt = measured_belt_grf(trial_session, right_plate_x_sign)

    planes = []
    for side in soles:
        sole = soles[side]
        pos, quat, linvel = trajs[side]
        fz = stance = None
        if side in belt:
            fz = np.asarray(belt[side][0])[lo:hi, 2]
            stance = fz > fz_threshold
        flat = flat_foot_mask(
            sole, pos, quat, linvel=linvel, fz=fz, fz_threshold=fz_threshold
        )
        planes.append(
            register_ground_flatfoot(
                sole, pos, quat, flat, penetration=penetration, fallback=stance
            )
        )
    if not planes:
        return 0.0
    ground = float(np.mean(planes))
    clip.data["rigid_body_pos"][..., 2] -= ground
    return ground


def _body_collision_points(model, body_id) -> np.ndarray:
    """Colliding geom surface points for a body, in its MJCF (Y-up) local frame.

    Boxes contribute their 8 corners; spheres their single bottom point (local -y). Only
    geoms that actually collide (``contype``/``conaffinity`` != 0) are included, so this
    matches what the simulator's foot contact sensor can register. Returns ``(P, 3)`` (may
    be empty for a body with no colliders, e.g. ``talus``).
    """
    import mujoco

    pts: list[np.ndarray] = []
    signs = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1)
                      for sz in (-1, 1)], dtype=np.float64)
    for g in range(model.ngeom):
        if model.geom_bodyid[g] != body_id:
            continue
        if model.geom_contype[g] == 0 and model.geom_conaffinity[g] == 0:
            continue  # visual only
        gp = model.geom_pos[g].astype(np.float64)
        gt = model.geom_type[g]
        if gt == mujoco.mjtGeom.mjGEOM_BOX:
            pts.append(gp + signs * model.geom_size[g].astype(np.float64))
        elif gt == mujoco.mjtGeom.mjGEOM_SPHERE:
            r = float(model.geom_size[g][0])
            pts.append(gp[None, :] + np.array([[0.0, -r, 0.0]]))  # plantar (-y) point
        else:  # capsule/other: use the center as a coarse fallback
            pts.append(gp[None, :])
    if not pts:
        return np.zeros((0, 3), dtype=np.float64)
    return np.concatenate(pts, axis=0)


def foot_contacts_from_clip(
    clip: MotionExportResult,
    mjcf_path: str,
    grf_by_side: dict,
    *,
    sides: tuple[str, ...] = ("R", "L"),
    height_thresh: float = 0.02,
    load_frac: float = 0.25,
    load_floor: float = 50.0,
):
    """Per-body foot-contact labels ``[F, num_bodies]`` (bool) aligned to ``clip.body_names``.

    A foot body registers contact on a frame when BOTH:

    * its lowest colliding-geom point is within ``height_thresh`` (m) of the sim floor
      (z = 0, the ground-registration datum), giving real per-body heel/toe timing, AND
    * the measured GRF for that foot exceeds ``max(load_floor, load_frac * peak)`` N, which
      removes swing-phase near-ground false positives and light scuffs.

    Geom points live in the MJCF (Y-up) body frame; the clip stores each body's Z-up world
    quaternion (Y-up->Z-up folded in), so rotating the local points by that quaternion and
    adding the world position yields the Z-up world point directly. Bodies without
    colliders (e.g. ``talus``) never contact, matching the simulator's foot sensors.

    ``grf_by_side`` maps side ("R"/"L") -> per-frame vertical-or-3-vector GRF ``(F,)``/``(F,3)``.
    Returns a ``torch.bool`` tensor.
    """
    import mujoco
    import torch

    from biomech.contact.elastic_foundation import _quat_rotate_np
    from biomech.contact.kinematics import foot_trajectory_from_motion

    model = mujoco.MjModel.from_xml_path(str(mjcf_path))
    body_names = list(clip.body_names)
    n_frames = int(np.asarray(clip.data["rigid_body_pos"]).shape[0])
    contacts = np.zeros((n_frames, len(body_names)), dtype=bool)

    for side in sides:
        grf = np.asarray(grf_by_side[side], dtype=np.float64)
        fz = grf[:, 2] if grf.ndim == 2 else grf
        peak = float(np.nanmax(fz)) if fz.size else 0.0
        loaded = fz > max(load_floor, load_frac * peak)
        for body in (f"calcn_{side.lower()}", f"toes_{side.lower()}"):
            if body not in body_names:
                continue
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body)
            local = _body_collision_points(model, bid)
            if local.shape[0] == 0:
                continue
            pos, quat, _, _ = foot_trajectory_from_motion(clip, body)
            world = pos[:, None, :] + _quat_rotate_np(quat[:, None, :],
                                                      local[None, :, :])
            min_z = world[:, :, 2].min(axis=1)
            contacts[:, body_names.index(body)] = (min_z < height_thresh) & loaded

    return torch.from_numpy(contacts)


# ---------------------------------------------------------------------------
# 3. RobotConfig
# ---------------------------------------------------------------------------
def build_biomech_robot_config(
    asset_file_name: str = "mjcf/biomech_rajagopal.xml",
    *,
    asset_root: str = PM_ASSET_ROOT,
    body_map: Optional[dict] = None,
    anchor_body_name: str = "torso",
    default_root_height: float = 0.94,
    control_type: str = "built_in_pd",
):
    """Construct a validated ProtoMotions ``RobotConfig`` for the fitted skeleton.

    Instantiating the returned config runs ProtoMotions' ``extract_kinematic_info`` /
    ``extract_control_info`` on the MJCF at ``asset_root/asset_file_name`` (so the asset
    must already be written, e.g. by :func:`write_biomech_asset`). Requires
    ``protomotions`` to be importable.
    """
    from protomotions.robot_configs.base import (
        ControlConfig,
        ControlType,
        RobotAssetConfig,
        RobotConfig,
    )

    return RobotConfig(
        asset=RobotAssetConfig(
            asset_root=asset_root,
            asset_file_name=asset_file_name,
            self_collisions=False,
        ),
        common_naming_to_robot_body_names=dict(body_map or RAJAGOPAL_BODY_MAP),
        anchor_body_name=anchor_body_name,
        default_root_height=default_root_height,
        control=ControlConfig(control_type=ControlType.from_str(control_type)),
    )


def export_protomotions_bundle(
    spec,
    q_t: np.ndarray,
    fps: float,
    *,
    group_scales: Optional[np.ndarray] = None,
    coupled_knee: str = "coupled",
    asset_file_name: str = "mjcf/biomech_rajagopal.xml",
    asset_root: str = PM_ASSET_ROOT,
    repo_root: Optional[str | Path] = None,
    motion_path: Optional[str | Path] = None,
    subject_mass: Optional[float] = None,
) -> ProtoMotionsBundle:
    """Write the MJCF asset (+ optional sim-body motion clip) and return the bundle."""
    asset_path = write_biomech_asset(
        spec,
        asset_file_name=asset_file_name,
        group_scales=group_scales,
        coupled_knee=coupled_knee,
        asset_root=asset_root,
        repo_root=repo_root,
        subject_mass=subject_mass,
    )
    clip = build_simbody_motion(
        spec, q_t, fps, group_scales=group_scales, coupled_knee=coupled_knee
    )
    out_motion = None
    if motion_path is not None:
        import torch

        out_motion = Path(motion_path)
        out_motion.parent.mkdir(parents=True, exist_ok=True)
        torch.save(clip.data, str(out_motion))
    return ProtoMotionsBundle(
        asset_path=asset_path,
        asset_file_name=asset_file_name,
        motion_path=out_motion,
        body_names=clip.body_names,
        dof_names=clip.dof_names,
        fps=float(fps),
    )
