# SPDX-License-Identifier: MIT
#
# Foot-ground collision geometry for the exported biomech MJCF. The visual bone meshes
# are non-colliding (contype/conaffinity=0), so a physically-simulated mimic character
# needs explicit collision geoms on the foot bodies or it falls through the floor.
#
# Two schemes are provided, both sized to the *subject's real plantar sole* (the tapered
# footprint built from the static-trial plantar markers by
# :func:`biomech.contact.foot_geometry.subject_sole_from_session`, expressed in the
# ``calcn`` body frame and scaled by the subject's group scales). Using the sole -- rather
# than the bone mesh -- means the collision surface coincides with the fat-pad contact
# plane the ground registration drops onto z=0, so the geoms actually touch the floor
# during stance instead of floating ~15-19 mm up on the bare calcaneus bone.
#
#   * ``"spheres"`` -- OpenSim-style: several discrete contact spheres per foot body
#     (heel medial/lateral + ball medial/lateral on ``calcn``; two on ``toes``), each
#     tangent to the local plantar surface. Mirrors how OpenSim/Falisse contact models
#     distribute compliant spheres over the sole.
#   * ``"boxes"`` -- ProtoMotions-style: a single box per foot body (the AABB of that
#     body's share of the sole footprint), a flat-bottomed foot block.
#
# The forefoot/toe portion of the sole is split off onto the articulating ``toes`` body
# (at the MTP joint) so toe collision follows metatarsophalangeal flexion at push-off,
# instead of being welded to the rearfoot.
#
# Frame note: the sole lives in the raw OpenSim ``calcn`` body frame (x forward, y up,
# z lateral; plantar normal -y). The MJCF body frames are the same Y-up OpenSim frames,
# so geoms placed here in that frame rotate correctly into Z-up world via the body pose.

"""Subject-sized foot-ground collision geometry (spheres or box per foot body)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class CollisionGeom:
    """One colliding MJCF geom attached to a body, in that body's local frame."""

    body: str          # MJCF body name it attaches to (e.g. "calcn_r", "toes_r")
    kind: str          # "box" or "sphere"
    pos: np.ndarray    # (3,) center in the body frame (meters)
    size: np.ndarray   # box: (3,) half-extents; sphere: (1,) radius (meters)
    name: str          # unique geom name


def _toes_from_calcn(spec, group_scales, body: str) -> np.ndarray:
    """4x4 transform mapping a point in the ``toes`` frame to its parent ``calcn`` frame.

    Uses the same scaled zero-config joint placement the MJCF exporter emits, so the
    split is consistent with the written asset.
    """
    from biomech.export.mjcf import _ScaleMap, _joint_transform, _scaled_frames
    from biomech.skeleton import spatial as S

    sm = _ScaleMap(spec, group_scales)
    joint_of_body = {j.child_body: j for j in spec.joints}
    j = joint_of_body[body]
    Tp, Tc = _scaled_frames(j, sm)
    return Tp @ _joint_transform(j, np.zeros(j.num_dofs)) @ S.se3_inverse_np(Tc)


def _sole_points(spec, group_scales, static_session, side: str) -> np.ndarray:
    """Subject plantar sole points (N,3) in the ``calcn_{side}`` body frame."""
    from biomech.contact.foot_geometry import subject_sole_from_session

    sole = subject_sole_from_session(
        static_session, spec, side, group_scales=group_scales
    )
    return np.asarray(sole.points, dtype=np.float64)


def _bbox(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return points.min(axis=0), points.max(axis=0)


def _box_geom(points: np.ndarray, body: str, name: str, min_half: float = 0.006) -> CollisionGeom:
    """A single box = AABB of ``points`` (flat-bottomed foot block for that body)."""
    lo, hi = _bbox(points)
    center = 0.5 * (lo + hi)
    half = np.maximum(0.5 * (hi - lo), min_half)
    return CollisionGeom(body=body, kind="box", pos=center, size=half, name=name)


def _sphere_xz(
    x: float, z: float, y_plane: float, body: str, name: str, radius: float
) -> CollisionGeom:
    """A sphere at footprint (x, z) resting on the sole contact plane ``y_plane``."""
    pos = np.array([x, y_plane + radius, z], dtype=np.float64)
    return CollisionGeom(
        body=body, kind="sphere", pos=pos,
        size=np.array([radius], dtype=np.float64), name=name,
    )


def _radius_for(points: np.ndarray, frac: float, lo: float, hi: float) -> float:
    """Sphere radius from a region's mediolateral width, clamped to [lo, hi]."""
    width = float(points[:, 2].max() - points[:, 2].min())
    return float(np.clip(frac * width, lo, hi))


def _quadrant_spheres(
    points: np.ndarray, body: str, prefix: str, radius: float
) -> list[CollisionGeom]:
    """Four spheres near the plantar footprint corners (heel/ball x medial/lateral).

    All rest on the region's deepest plantar level (a common contact plane, as OpenSim
    contact spheres are laid out), and sit at the footprint edges (inset by the radius)
    so a sphere is under the posterior heel and the metatarsal heads -- making heel-strike
    and push-off contact the floor together.
    """
    y_plane = float(points[:, 1].min())
    x0, x1 = float(points[:, 0].min()), float(points[:, 0].max())
    z0, z1 = float(points[:, 2].min()), float(points[:, 2].max())
    out: list[CollisionGeom] = []
    for xtag, xv in (("heel", x0 + radius), ("ball", x1 - radius)):
        for ztag, zv in (("lat", z0 + radius), ("med", z1 - radius)):
            out.append(
                _sphere_xz(xv, zv, y_plane, body, f"{prefix}_{xtag}_{ztag}", radius)
            )
    return out


def _pair_spheres(
    points: np.ndarray, body: str, prefix: str, radius: float
) -> list[CollisionGeom]:
    """Two spheres at the distal toe footprint, split medial/lateral, on one plane."""
    y_plane = float(points[:, 1].min())
    x1 = float(points[:, 0].max())
    z0, z1 = float(points[:, 2].min()), float(points[:, 2].max())
    out: list[CollisionGeom] = []
    for ztag, zv in (("lat", z0 + radius), ("med", z1 - radius)):
        out.append(_sphere_xz(x1 - radius, zv, y_plane, body, f"{prefix}_{ztag}", radius))
    return out


def _side_geoms(
    spec, group_scales, static_session, side: str, scheme: str
) -> list[CollisionGeom]:
    calcn = f"calcn_{side.lower()}"
    toes = f"toes_{side.lower()}"
    pts = _sole_points(spec, group_scales, static_session, side)

    # Split the sole at the MTP joint: points ahead of the toes-body origin (expressed in
    # the calcn frame) become toe collision, transformed into the toes body frame so they
    # articulate with metatarsophalangeal flexion.
    M = _toes_from_calcn(spec, group_scales, toes)  # toes -> calcn
    R, t = M[:3, :3], M[:3, 3]
    mtp_x = float(t[0])
    fore = pts[:, 0] >= mtp_x
    rear_pts = pts[~fore]
    toe_pts_calcn = pts[fore]
    # calcn-frame -> toes-frame: p_toes = R^T (p_calcn - t)
    toe_pts = (toe_pts_calcn - t) @ R if toe_pts_calcn.shape[0] else toe_pts_calcn

    geoms: list[CollisionGeom] = []
    if scheme == "boxes":
        if rear_pts.shape[0]:
            geoms.append(_box_geom(rear_pts, calcn, f"col_{calcn}_box"))
        if toe_pts.shape[0]:
            geoms.append(_box_geom(toe_pts, toes, f"col_{toes}_box"))
    elif scheme == "spheres":
        if rear_pts.shape[0]:
            r = _radius_for(rear_pts, frac=0.28, lo=0.012, hi=0.022)
            geoms += _quadrant_spheres(rear_pts, calcn, f"col_{calcn}", r)
        if toe_pts.shape[0]:
            r = _radius_for(toe_pts, frac=0.28, lo=0.010, hi=0.018)
            geoms += _pair_spheres(toe_pts, toes, f"col_{toes}", r)
    else:  # pragma: no cover - guarded by caller
        raise ValueError(f"unknown foot collision scheme: {scheme!r}")
    return geoms


def foot_collision_geoms(
    spec,
    group_scales: Optional[np.ndarray],
    scheme: str,
    static_session,
    sides: tuple[str, ...] = ("R", "L"),
) -> list[CollisionGeom]:
    """Build foot-ground collision geoms for both feet under one of the two schemes.

    ``scheme`` is ``"spheres"`` (OpenSim-style discrete plantar spheres, multiple per
    foot body) or ``"boxes"`` (a single AABB box per foot body). Geometry is sized to the
    subject's real plantar sole (``static_session`` provides the static-trial foot
    markers), so the collision surface coincides with the ground-registration datum.
    """
    if scheme not in ("spheres", "boxes"):
        raise ValueError(f"foot collision scheme must be 'spheres' or 'boxes', got {scheme!r}")
    geoms: list[CollisionGeom] = []
    for side in sides:
        if f"calcn_{side.lower()}" not in {b.name for b in spec.bodies}:
            continue
        geoms += _side_geoms(spec, group_scales, static_session, side, scheme)
    return geoms
