# SPDX-License-Identifier: MIT
"""Side-view render of the two foot-collision variants against the sim floor.

Drives the ground-registered motion clip (floor at z=0) and, for each variant
(OpenSim-style spheres / ProtoMotions-style single box per foot body), overlays that
variant's colliding foot geoms on the right-foot bone mesh at heel-strike and flat-foot,
so we can see whether the collision surface actually reaches the floor and how the two
styles differ. Uses the exported clip's body world transforms directly (the Y-up->Z-up
bake is already in the body quaternions), so no separate registration datum is needed.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import PolyCollection  # noqa: E402
from matplotlib.patches import Circle  # noqa: E402

_BIOMECH = Path(__file__).resolve().parents[1]
_REPO = _BIOMECH.parents[1]
_CACHE = _BIOMECH / "docs" / "figures" / "_s001_ik_cache.npz"
_MOTION = _BIOMECH / "data" / "motions" / "biomech_s001_walk.motion"
_MESH = _REPO / "protomotions" / "data" / "assets" / "mesh" / "biomech_rajagopal"
_OUT = _BIOMECH / "docs" / "figures" / "foot_collision_check.png"

_FOOT = (("calcn_r", "r_foot"), ("toes_r", "r_bofoot"))


def _sagittal(pos, quat, pts_body, h=None):
    """Project body-frame points to the foot's sagittal plane (u=forward, v=up=world z).

    ``pts_body`` are in the body frame; ``pos``/``quat`` are the body's world pose for one
    frame. If ``h`` (a world-plane forward unit vector) is given it is used for the u axis;
    otherwise it is derived from this body's own forward direction. Returns (u, v, h).
    """
    from biomech.contact.elastic_foundation import _quat_rotate_np

    q = quat[None, :]
    world = pos + _quat_rotate_np(np.broadcast_to(q, (pts_body.shape[0], 4)), pts_body)
    if h is None:
        fwd = _quat_rotate_np(q, np.array([[1.0, 0.0, 0.0]]))[0]
        h = fwd[:2] / (np.linalg.norm(fwd[:2]) + 1e-12)
    u = world[:, :2] @ h
    v = world[:, 2]
    return u, v, h


def main() -> int:
    import trimesh
    import torch

    from biomech.contact.kinematics import foot_trajectory_from_motion
    from biomech.export.foot_collision import foot_collision_geoms
    from biomech.export.mjcf import _ScaleMap
    from biomech.session import load_session
    from biomech.tests import CAL_C3D

    cache = np.load(_CACHE, allow_pickle=True)
    spec = cache["spec_pickle"].item()
    scales = np.asarray(cache["scales"], dtype=np.float64)
    sm = _ScaleMap(spec, scales)

    data = torch.load(str(_MOTION), weights_only=False)

    # body order for the clip (not stored in the clip dict) comes from a RobotConfig
    from biomech.export.protomotions_robot import build_biomech_robot_config
    from types import SimpleNamespace
    cfg = build_biomech_robot_config(
        asset_file_name="mjcf/biomech_rajagopal.xml",
        asset_root=str((_REPO / "protomotions" / "data" / "assets").resolve()),
    )
    clip = SimpleNamespace(body_names=cfg.kinematic_info.body_names, data=data)

    # bone-mesh triangles (raw STL, scaled by per-body group scale) in the body frame
    bone = {}
    for body, stem in _FOOT:
        m = trimesh.load(_MESH / f"{stem}.stl")
        cs = sm.of(body)
        bone[body] = (np.asarray(m.vertices) * cs, np.asarray(m.faces))

    # per-body world trajectories
    traj = {body: foot_trajectory_from_motion(clip, body)[:2] for body, _ in _FOOT}

    # collision geoms for both schemes (need the static trial for the subject sole)
    static = load_session(str(CAL_C3D), filter_cutoff_hz=None)
    schemes = {
        "OpenSim-style spheres": foot_collision_geoms(spec, scales, "spheres", static),
        "ProtoMotions-style boxes": foot_collision_geoms(spec, scales, "boxes", static),
    }
    _R_BODIES = {body for body, _ in _FOOT}
    schemes = {k: [g for g in v if g.body in _R_BODIES] for k, v in schemes.items()}

    from biomech.contact.elastic_foundation import _quat_rotate_np
    from biomech.export.foot_collision import _sole_points, _toes_from_calcn

    def lowest_collision_z(geoms, fr, bodies=None):
        """World-z of the deepest point of any collision geom at frame ``fr``.

        If ``bodies`` is given, only geoms on those bodies are considered.
        """
        zmin = np.inf
        for g in geoms:
            if bodies is not None and g.body not in bodies:
                continue
            pos, quat = traj[g.body][0][fr], traj[g.body][1][fr]
            q = quat[None, :]
            if g.kind == "sphere":
                # deepest point is orientation-independent: center_z - radius
                cz = pos[2] + _quat_rotate_np(q, g.pos[None, :])[0][2]
                zmin = min(zmin, cz - float(np.asarray(g.size).ravel()[0]))
            else:
                hx, hy, hz = g.size
                corners = np.array([g.pos + [sx*hx, sy*hy, sz*hz]
                                    for sx in (-1, 1) for sy in (-1, 1)
                                    for sz in (-1, 1)])
                cw = pos + _quat_rotate_np(np.broadcast_to(q, (8, 4)), corners)
                zmin = min(zmin, float(cw[:, 2].min()))
        return zmin

    nframes = traj["calcn_r"][0].shape[0]

    # --- scheme-independent gait-event detection from the plantar sole kinematics ---
    # Split the subject sole at the MTP into rear (calcn frame) and toe (toes frame), and
    # track, per frame, the world-z of a posterior-heel point, a distal-toe point, and the
    # whole sole. The events are properties of the *motion*, so both collision schemes are
    # rendered at the same frames (directly comparable).
    sole = _sole_points(spec, scales, static, "R")
    M = _toes_from_calcn(spec, scales, "toes_r")
    R, t = M[:3, :3], M[:3, 3]
    fore = sole[:, 0] >= float(t[0])
    rear = sole[~fore]
    toe = (sole[fore] - t) @ R
    heel_pt = rear[int(np.argmin(rear[:, 0]))][None, :]          # posterior-most (calcn)
    toe_pt = toe[int(np.argmax(toe[:, 0]))][None, :]             # distal-most (toes)

    def _wz(body, pts_body, fr):
        pos, quat = traj[body][0][fr], traj[body][1][fr]
        q = np.broadcast_to(quat[None, :], (pts_body.shape[0], 4))
        return pos[2] + _quat_rotate_np(q, pts_body)[:, 2]

    heel_z = np.array([_wz("calcn_r", heel_pt, f)[0] for f in range(nframes)])
    toe_z = np.array([_wz("toes_r", toe_pt, f)[0] for f in range(nframes)])
    sole_z = np.array([min(_wz("calcn_r", rear, f).min(), _wz("toes_r", toe, f).min())
                       for f in range(nframes)])

    # Primary stance = the contiguous low-sole block around the deepest-contact frame.
    thresh = float(sole_z.min()) + 0.02
    i0 = int(np.argmin(sole_z))
    s = i0
    while s > 0 and sole_z[s - 1] < thresh:
        s -= 1
    e = i0
    while e < nframes - 1 and sole_z[e + 1] < thresh:
        e += 1
    mid = (s + e) // 2
    # Characteristic postures: heel strike = heel down / toe up (dorsiflexion); toe off =
    # heel up / toe down (plantarflexion); midstance = both lowest (flat); swing = clearance.
    hs = s + int(np.argmax((toe_z - heel_z)[s:mid + 1])) if mid > s else s
    to = mid + int(np.argmax((heel_z - toe_z)[mid:e + 1])) if e > mid else e
    ms = s + int(np.argmin((heel_z + toe_z)[s:e + 1]))
    sw = int(np.argmax(sole_z))
    events = [
        (f"heel strike (frame {hs})", hs),
        (f"midstance (frame {ms})", ms),
        (f"toe off (frame {to})", to),
        (f"mid swing (frame {sw})", sw),
    ]

    nrows, ncols = len(schemes), len(events)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.4 * ncols, 4.6 * nrows),
                             squeeze=False)
    for r, (sname, geoms) in enumerate(schemes.items()):
        by_body: dict = {}
        for g in geoms:
            by_body.setdefault(g.body, []).append(g)
        for c, (ename, fr) in enumerate(events):
            gz = lowest_collision_z(geoms, fr)
            ftitle = (f"{ename}\nlowest geom z = {gz*1e3:+.1f} mm  |  "
                      f"heel z = {heel_z[fr]*1e3:+.1f} mm")
            ax = axes[r][c]
            # forward axis from the rearfoot for a consistent u origin
            cpos, cquat = traj["calcn_r"][0][fr], traj["calcn_r"][1][fr]
            _, _, h = _sagittal(cpos, cquat, np.zeros((1, 3)))
            u0 = cpos[:2] @ h
            polys = []
            for body, _ in _FOOT:
                pos, quat = traj[body][0][fr], traj[body][1][fr]
                v_body, faces = bone[body]
                u, v, _ = _sagittal(pos, quat, v_body, h)
                u = u - u0
                for tri in faces:
                    polys.append(np.column_stack([u[tri], v[tri]]))
            ax.add_collection(PolyCollection(
                polys, facecolors=(0.85, 0.82, 0.70, 0.85),
                edgecolors=(0.4, 0.38, 0.3, 0.30), linewidths=0.15))
            # collision geoms
            for body, _ in _FOOT:
                pos, quat = traj[body][0][fr], traj[body][1][fr]
                for g in by_body.get(body, []):
                    if g.kind == "sphere":
                        cu, cv, _ = _sagittal(pos, quat, g.pos[None, :], h)
                        ax.add_patch(Circle((cu[0] - u0, cv[0]),
                                            float(np.asarray(g.size).ravel()[0]),
                                            facecolor=(0.20, 0.80, 0.35, 0.45),
                                            edgecolor=(0.05, 0.45, 0.15, 0.9), lw=1.2))
                    else:
                        hx, hy, hz = g.size
                        corners = np.array([g.pos + [sx*hx, sy*hy, sz*hz]
                                            for sx in (-1, 1) for sy in (-1, 1)
                                            for sz in (-1, 1)])
                        cu, cv, _ = _sagittal(pos, quat, corners, h)
                        pts = np.column_stack([cu - u0, cv])
                        hull = pts[_convex_hull(pts)]
                        ax.add_patch(plt.Polygon(hull, closed=True,
                                                 facecolor=(0.20, 0.80, 0.35, 0.35),
                                                 edgecolor=(0.05, 0.45, 0.15, 0.9), lw=1.2))
            ax.axhline(0.0, color="saddlebrown", lw=2, label="sim floor (z=0)")
            ax.set_title(f"{sname}\n{ftitle}", fontsize=9)
            ax.set_xlabel("forward (m)"); ax.set_ylabel("up z (m)")
            ax.set_aspect("equal"); ax.autoscale_view()
            ax.set_ylim(-0.05, 0.28)
            ax.legend(loc="upper right", fontsize=8)
    fig.suptitle(
        "S001 right foot: collision geometry vs sim floor (sole-registered clip); "
        "green = colliding foot geoms.  Sole uses the anatomical plantar normal "
        "(calcn +y).  After the foot-flat ankle correction the stance foot is "
        "plantigrade: heel and forefoot both rest on z=0 through midstance, then the "
        "foot rolls into a real plantarflexed push-off and lifts in swing.", fontsize=11)
    fig.tight_layout()
    fig.savefig(str(_OUT), dpi=120)
    print("wrote", _OUT)
    return 0


def _convex_hull(pts: np.ndarray) -> np.ndarray:
    """Indices of the 2D convex hull (monotone chain), CCW."""
    order = np.lexsort((pts[:, 1], pts[:, 0]))
    p = pts[order]

    def cross(o, a, b):
        return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])

    lower = []
    for i in range(len(p)):
        while len(lower) >= 2 and cross(p[lower[-2]], p[lower[-1]], p[i]) <= 0:
            lower.pop()
        lower.append(i)
    upper = []
    for i in range(len(p) - 1, -1, -1):
        while len(upper) >= 2 and cross(p[upper[-2]], p[upper[-1]], p[i]) <= 0:
            upper.pop()
        upper.append(i)
    return order[np.array(lower[:-1] + upper[:-1])]


if __name__ == "__main__":
    raise SystemExit(main())
