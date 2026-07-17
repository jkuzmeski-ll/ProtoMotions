# SPDX-License-Identifier: MIT
#
# OpenSim-style *marker placement* for the foot, from a static (calibration) trial.
#
# The stock Rajagopal2015 marker set is sparse on the foot: only ``RCAL`` (heel),
# ``RMT5`` (5th-met head) and ``RTOE`` (placed at the toe tip) on ``calcn``, and *nothing*
# on ``toes``. But the S001 Plug-in-Gait capture carries a much richer foot set (verified
# in the C3D, 100% present in both the static and dynamic trials):
#
#   * ``HEE``/``HEE2``/``HEE3`` — a triangular calcaneus cluster (fully constrains the
#     hindfoot orientation, which the single ``RCAL`` cannot),
#   * ``MTH1``/``MTH5`` — 1st/5th metatarsal heads,
#   * ``TOE`` — on the **met-2 head** (i.e. on ``calcn``, at the MTP line — NOT the toe tip
#     the stock model places it at, a ~2.7 cm forward mismatch that the fit was absorbing
#     by pitching the foot into a spurious plantarflexion offset),
#   * ``HLX`` — on the **hallux**, ~5 cm distal to the MTP line, i.e. on the ``toes``
#     segment. This is the only marker distal to the MTP joint, so mapping it is what makes
#     ``mtp_angle`` observable at all.
#
# This module runs the OpenSim "marker placement" step: fit the *static* trial with the
# existing (sparse) markers to get segment scales + a foot-flat pose, then express each rich
# foot marker's measured position in its owning body frame. Those become the model marker
# offsets. Re-seating ``RCAL``/``RTOE``/``RMT5`` and adding the cluster/met-1/hallux markers
# both (a) over-constrains the calcaneus so the ankle stops absorbing foot placement error,
# and (b) puts a marker on ``toes`` so the MTP angle is recovered.
#
# Frames: markers come in the lab (Z-up) frame; the skeleton FK is OpenSim (Y-up). We reuse
# ``marker_map.R_PM2OS`` for the rotation, exactly as ``build_observations`` does.

"""OpenSim-style foot marker placement from a static trial (enriches the model marker set)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from biomech.fitting.marker_map import R_PM2OS
from biomech.osim.spec import MarkerSpec, SkeletonSpec


# New model markers to add: model name -> (owning body, capture label), per side.
# The body assignment is the anatomically-correct segment (calcn for hindfoot/forefoot,
# toes distal to the MTP joint).
_FOOT_ADDITIONS = {
    "R": {
        "RCAL2": ("calcn_r", "RHEE2"),
        "RCAL3": ("calcn_r", "RHEE3"),
        "RMT1": ("calcn_r", "RMTH1"),
        "RTOE_TIP": ("toes_r", "RHLX"),
    },
    "L": {
        "LCAL2": ("calcn_l", "LHEE2"),
        "LCAL3": ("calcn_l", "LHEE3"),
        "LMT1": ("calcn_l", "LMTH1"),
        "LTOE_TIP": ("toes_l", "LHLX"),
    },
}

# Existing foot markers whose model offset we re-seat from the static trial (marker
# placement). ``RTOE`` is the important one: the stock model puts it at the toe tip but the
# capture marker is on the met-2 head.
_FOOT_RESEAT = {
    "RCAL": ("calcn_r", "RHEE"),
    "RTOE": ("calcn_r", "RTOE"),
    "RMT5": ("calcn_r", "RMTH5"),
    "LCAL": ("calcn_l", "LHEE"),
    "LTOE": ("calcn_l", "LTOE"),
    "LMT5": ("calcn_l", "LMTH5"),
}


@dataclass
class MarkerPlacement:
    """Result of the static foot marker placement."""

    added: List[str]                 # new model marker names added to the spec
    reseated: List[str]              # existing model markers whose offset changed
    unlocked: List[str]              # coordinate names unlocked (e.g. mtp_angle_{r,l})
    offsets: Dict[str, np.ndarray]   # model name -> body-frame offset (unscaled, meters)
    residual_mm: Dict[str, float]    # model name -> static placement residual (mm)
    group_scales: np.ndarray         # scales used for placement (from the static fit)
    poses: np.ndarray                # (Fw, ndof) fitted static poses used for placement
    window: Tuple[int, int]          # frame window used for the static fit
    ankle_neutral: Optional[Dict[str, float]] = None  # coord -> baked static offset (radians)
    foot_flat: Optional[Dict[str, float]] = None      # ankle coord -> foot-flat correction (radians)


def unlock_mtp(spec: SkeletonSpec, sides: Sequence[str] = ("r", "l")) -> List[str]:
    """Unlock the ``mtp_angle_{r,l}`` coordinates so the MTP DOF can be fit.

    The stock Rajagopal2015 model ships the metatarsophalangeal joint ``locked`` (frozen
    at 0), which is the standard choice when there is no marker distal to the MTP. Once a
    hallux/toes marker is added (see :func:`place_foot_markers`) the MTP becomes
    observable, so we free it. Returns the list of coordinate names unlocked.
    """
    unlocked: List[str] = []
    for side in sides:
        try:
            joint = spec.joint(f"mtp_{side}")
        except KeyError:
            continue
        for coord in joint.coordinates:
            if coord.locked:
                coord.locked = False
                unlocked.append(coord.name)
    return unlocked


def _rot_z4(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    T = np.eye(4, dtype=np.float64)
    T[0, 0] = c
    T[0, 1] = -s
    T[1, 0] = s
    T[1, 1] = c
    return T


def register_ankle_neutral(
    spec: SkeletonSpec,
    static_poses: np.ndarray,
    coords: Sequence[str] = ("ankle_angle_r", "ankle_angle_l"),
) -> Dict[str, float]:
    """Re-zero pin-joint coordinates at the static (standing) neutral pose.

    The stock Rajagopal ankle neutral does not coincide with a given subject's flat-foot
    standing pose, so a fitted static trial reads a nonzero "plantarflexion" (~ -10 deg for
    S001, matching Plug-in-Gait's ``RStaticPlantFlex``). That constant offset then
    contaminates the whole dynamic ankle-angle trace. This is the OpenSim/PiG static-offset
    correction: measure the mean coordinate value over the static trial and re-zero it.

    The re-zero is done *physically* by baking ``Rz(-off)`` into the joint's child frame,
    so with ``q' = q - off`` every body's world pose is provably unchanged (the contact /
    export pipeline is untouched) while the coordinate now reads 0 at standing. Only
    single-axis pin joints (revolute about Z, e.g. the ankle) are supported.

    Returns ``{coordinate_name: offset_radians}`` for the coordinates re-zeroed.
    """
    static_poses = np.asarray(static_poses, dtype=np.float64)
    dof = spec.dof_index_map()
    offsets: Dict[str, float] = {}
    for coord in coords:
        if coord not in dof:
            continue
        joint = next((j for j in spec.joints if coord in j.dof_names), None)
        if joint is None:
            continue
        if joint.joint_class != "PinJoint":
            raise NotImplementedError(
                f"neutral registration only supports PinJoint coords, got "
                f"{joint.joint_class} for {coord!r}"
            )
        off = float(np.nanmean(static_poses[:, dof[coord]]))
        # bake Rz(-off) into the child frame: T_child' = T_child @ Rz(-off)
        joint.T_child = joint.T_child @ _rot_z4(-off)
        # shift the coordinate limits so the same physical range stays reachable
        c = joint.coordinates[0]
        c.range_lo -= off
        c.range_hi -= off
        c.default_value -= off
        offsets[coord] = off
    return offsets


def compute_foot_flat_offset(
    spec: SkeletonSpec,
    static_poses: np.ndarray,
    group_scales: np.ndarray,
    coords: Sequence[str] = ("ankle_angle_r", "ankle_angle_l"),
    device: str = "cpu",
) -> Dict[str, float]:
    """Ankle correction (radians) that makes each foot *flat* at the static-flat pose.

    The static (calibration) trial is a foot-flat standing trial, so the plantar sole is
    known to be horizontal. The marker fit, however, reconstructs the ``calcn`` body a
    constant ~13-14 deg toe-down even here: the heel marker sits high on the calcaneus and
    the offset prior pulls the foot offsets toward the model's low plantar-heel defaults, so
    the bone rotates heel-up/toe-down to reconcile. That constant plantarflexion is then
    frozen into the re-seated foot offsets and inherited by every dynamic frame.

    This measures, per ankle, the sagittal rotation ``dq`` about the ankle DOF that drives
    the ``calcn`` forward axis (+x) to horizontal (plantar sole flat) at the static pose.
    Because the ankle is the leaf-side joint of the shank->foot chain, adding ``dq`` to the
    ankle coordinate rotates *only* the foot (and its toes child) about the ankle -- the
    pelvis/hip/knee/shank (``q`` upstream of the ankle) are untouched. Applying the same
    constant across a dynamic clip removes the artifact while preserving the real gait
    plantar/dorsiflexion (the frame-to-frame ankle motion) exactly.

    The returned value is a coordinate *delta*, which is invariant to the ankle-neutral
    zero-shift baked by :func:`register_ankle_neutral`, so it is valid to add to poses
    expressed in either the pre- or post-neutral-bake frame.

    Returns ``{coordinate_name: dq_radians}`` for each ankle coordinate resolved.
    """
    from biomech.skeleton.skeleton import WarpSkeleton

    static_poses = np.asarray(static_poses, dtype=np.float64)
    group_scales = np.asarray(group_scales, dtype=np.float64)
    dof = spec.dof_index_map()
    body_index = {b.name: i for i, b in enumerate(spec.bodies)}
    skel = WarpSkeleton(spec, device=device)

    def _toe_down(poses: np.ndarray, bi: int) -> float:
        # mean toe-down (rad) of the calcn +x axis (OpenSim vertical = +y) over frames
        world, _ = skel.forward(poses, group_scales)
        Rc = np.asarray(world)[:, bi, :3, :3]
        fx_y = Rc @ np.array([1.0, 0.0, 0.0])  # (F,3); take vertical component [:,1]
        return float(np.mean(-np.arcsin(np.clip(fx_y[:, 1], -1.0, 1.0))))

    out: Dict[str, float] = {}
    for coord in coords:
        if coord not in dof:
            continue
        side = coord.rsplit("_", 1)[-1]  # "r" / "l"
        body = f"calcn_{side}"
        if body not in body_index:
            continue
        j = dof[coord]
        bi = body_index[body]
        # Newton on the (near-linear) toe-down(dq) with FK-measured local sensitivity.
        dq = 0.0
        for _ in range(4):
            p = static_poses.copy()
            p[:, j] += dq
            f0 = _toe_down(p, bi)
            if abs(f0) < 1e-4:  # < ~0.006 deg
                break
            eps = 1e-3
            p[:, j] += eps
            slope = (_toe_down(p, bi) - f0) / eps
            if abs(slope) < 1e-6:
                break
            dq -= f0 / slope
        out[coord] = dq
    return out


def _body_group_map(spec: SkeletonSpec) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for gi, group in enumerate(spec.scale_groups):
        for name in group:
            out[name] = gi
    return out


def _capture_positions_opensim(
    session, label: str, frame_range: Optional[Tuple[int, int]] = None
) -> np.ndarray:
    """Per-frame world positions of a capture label, rotated lab Z-up -> OpenSim Y-up.

    Returns ``(F, 3)`` with NaN where the marker is missing.
    """
    labels = list(session.marker_labels)
    if label not in labels:
        raise KeyError(f"capture label {label!r} not in session")
    pts = np.asarray(session.markers, dtype=np.float64)[:, labels.index(label), :]
    if frame_range is not None:
        lo, hi = frame_range
        pts = pts[lo:hi]
    return np.einsum("ij,fj->fi", R_PM2OS, pts)


def place_foot_markers(
    spec: SkeletonSpec,
    static_session,
    *,
    mapping=None,
    marker_config=None,
    device: str = "cpu",
    frame_range: Optional[Tuple[int, int]] = None,
    sides: Sequence[str] = ("R", "L"),
    reseat: bool = True,
    unlock_mtp_joint: bool = True,
    register_neutral: bool = True,
) -> MarkerPlacement:
    """Add the rich foot markers to ``spec`` (in place) via static marker placement.

    Fits the static trial with the current (sparse) marker set to obtain segment scales +
    a foot-flat pose, then expresses each rich foot marker's measured position in its owning
    body frame. New markers are appended to ``spec.markers``; existing foot markers are
    re-seated (unless ``reseat=False``). Returns a :class:`MarkerPlacement` describing the
    change.

    The offset stored on a :class:`MarkerSpec` is *unscaled*: FK applies
    ``world = T_body @ (group_scale ⊙ offset)``, so we divide the measured body-frame
    position by the fitted group scale before storing.
    """
    # Local import to avoid a heavy import at module load (mirrors pipeline.reconstruct).
    from biomech.contact.pipeline import reconstruct_window
    from biomech.skeleton.skeleton import WarpSkeleton

    F = np.asarray(static_session.markers).shape[0]
    window = frame_range if frame_range is not None else (0, F)

    # 1) Fit the static trial with the *current* (pre-augmentation) marker set.
    result, _obs, _anat = reconstruct_window(
        static_session, spec, window, mapping=mapping,
        marker_config=marker_config, device=device,
    )
    group_scales = np.asarray(result.group_scales, dtype=np.float64)
    poses = np.asarray(result.poses, dtype=np.float64)  # (Fw, ndof)

    # 2) FK the fitted static poses -> per-frame body transforms (OpenSim frame).
    skel = WarpSkeleton(spec, device=device)
    world, _ = skel.forward(poses, group_scales)  # (Fw, B, 4, 4)
    body_index = {b.name: i for i, b in enumerate(spec.bodies)}
    body_group = _body_group_map(spec)
    scales3 = group_scales.reshape(-1, 3)

    lo, hi = window

    def _place(model_name: str, body: str, label: str) -> Tuple[np.ndarray, float]:
        # measured positions over the fit window, OpenSim frame
        p = _capture_positions_opensim(static_session, label)[lo:hi]  # (Fw, 3)
        T = world[:, body_index[body], :, :]  # (Fw, 4, 4)
        R = T[:, :3, :3]
        t = T[:, :3, 3]
        # local = R^T (p - t): position in the (scaled) body frame
        local = np.einsum("fji,fj->fi", R, p - t)  # R^T @ (p - t)
        mask = np.isfinite(local).all(axis=1)
        if not mask.any():
            raise ValueError(f"no finite frames to place marker {model_name!r} ({label})")
        local_mean = local[mask].mean(axis=0)
        s = scales3[body_group[body]]
        offset = local_mean / s  # unscaled offset stored on the MarkerSpec
        # placement residual: spread of the per-frame local positions (rigid-body noise)
        resid_mm = float(np.linalg.norm(local[mask] - local_mean, axis=1).mean() * 1e3)
        return offset, resid_mm

    offsets: Dict[str, np.ndarray] = {}
    residual_mm: Dict[str, float] = {}
    added: List[str] = []
    reseated: List[str] = []

    # 3) Re-seat existing foot markers (marker placement on the stock set).
    if reseat:
        for model_name, (body, label) in _FOOT_RESEAT.items():
            side = model_name[0]
            if side not in sides:
                continue
            try:
                m = spec.marker(model_name)
            except KeyError:
                continue
            offset, resid = _place(model_name, body, label)
            m.offset = offset
            offsets[model_name] = offset
            residual_mm[model_name] = resid
            reseated.append(model_name)

    # 4) Add the new rich foot markers.
    existing = {m.name for m in spec.markers}
    for side in sides:
        for model_name, (body, label) in _FOOT_ADDITIONS[side].items():
            if model_name in existing:
                continue
            offset, resid = _place(model_name, body, label)
            spec.markers.append(
                MarkerSpec(name=model_name, body=body, offset=offset, fixed=True)
            )
            offsets[model_name] = offset
            residual_mm[model_name] = resid
            added.append(model_name)

    # 5) Unlock the MTP joint now that a toes marker constrains it.
    unlocked: List[str] = []
    if unlock_mtp_joint:
        unlocked = unlock_mtp(spec, sides=tuple(s.lower() for s in sides))

    # 5b) Foot-flat correction: measure the constant ankle rotation that flattens each foot
    #     at the (known foot-flat) static pose. Computed *before* the ankle-neutral bake so
    #     poses/spec are in one consistent frame; the result is a zero-shift-invariant delta
    #     applied to the ankle DOF at export to plant the plantar sole in stance.
    foot_flat: Dict[str, float] = {}
    flat_coords = tuple(f"ankle_angle_{s.lower()}" for s in sides)
    foot_flat = compute_foot_flat_offset(
        spec, poses, group_scales, coords=flat_coords, device=device
    )

    # 6) Re-zero the ankle at the static (standing) neutral (PiG static-offset correction).
    ankle_neutral: Dict[str, float] = {}
    if register_neutral:
        coords = tuple(
            f"ankle_angle_{s.lower()}" for s in sides
        )
        ankle_neutral = register_ankle_neutral(spec, poses, coords=coords)

    return MarkerPlacement(
        added=added,
        reseated=reseated,
        unlocked=unlocked,
        offsets=offsets,
        residual_mm=residual_mm,
        group_scales=group_scales,
        poses=poses,
        window=(lo, hi),
        ankle_neutral=ankle_neutral,
        foot_flat=foot_flat,
    )
