# SPDX-License-Identifier: MIT
#
# Milestone M5 (rung 1) — convenience bridge from gold-standard kinematics to the
# distributed elastic-foundation foot contact model.
#
# The contact law in ``biomech.contact.elastic_foundation`` consumes a per-frame foot
# body pose + spatial velocity in the world frame (Z-up, xyzw quaternion). Those are
# exactly the per-body fields the M3 motion exporter produces from the float64 Warp/DART
# FK (``biomech.export.motion.build_motion``). This module slices a chosen foot body out
# of a ``MotionExportResult`` (or straight out of ``WarpSkeleton`` FK) and feeds it to
# ``evaluate_contact`` — so a fitted, gold-standard ``q(t)`` drives the distributed
# contact prediction directly, with no dynamics integration ("prescribed kinematics").
#
# The sole geometry (``FootSole``) is expressed in the foot **body frame**; for rung 1
# the analytic soles are approximate, and M4 will replace them with subject plantar
# geometry (SDF). World is Z-up, ground is the plane ``z = ground_z`` (+Z normal).

"""Drive the elastic-foundation foot contact from gold-standard kinematics (M5)."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from biomech.contact.elastic_foundation import (
    ContactPrediction,
    ElasticFoundationParams,
    FootSole,
    evaluate_contact,
)

# Default GRF/foot body names for the Rajagopal2015 lower body.
RIGHT_FOOT_BODY = "calcn_r"
LEFT_FOOT_BODY = "calcn_l"


def _as_np(a) -> np.ndarray:
    """Accept torch tensors or numpy arrays; return a float64 numpy array."""
    if hasattr(a, "detach"):
        a = a.detach().cpu().numpy()
    return np.asarray(a, dtype=np.float64)


def foot_trajectory_from_motion(
    result,
    body_name: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Slice a foot body's world pose + spatial velocity out of a ``MotionExportResult``.

    Args:
        result: a :class:`biomech.export.motion.MotionExportResult` (its ``data`` holds
            ``rigid_body_{pos,rot,vel,ang_vel}`` in Z-up world, xyzw quaternions).
        body_name: anatomical body to extract (e.g. ``"calcn_r"``).

    Returns:
        ``(pos, quat, linvel, angvel)`` each ``(F, ...)`` float64, world frame.
    """
    if body_name not in result.body_names:
        raise KeyError(
            f"body {body_name!r} not in motion body set {result.body_names}"
        )
    bi = result.body_names.index(body_name)
    data = result.data
    pos = _as_np(data["rigid_body_pos"])[:, bi, :]
    quat = _as_np(data["rigid_body_rot"])[:, bi, :]
    linvel = _as_np(data["rigid_body_vel"])[:, bi, :]
    angvel = _as_np(data["rigid_body_ang_vel"])[:, bi, :]
    return pos, quat, linvel, angvel


def evaluate_foot_contact_from_motion(
    result,
    body_name: str,
    sole: FootSole,
    params: Optional[ElasticFoundationParams] = None,
    ground_z: float = 0.0,
    backend: str = "numpy",
    device: str = "cuda",
    keep_points: bool = False,
) -> ContactPrediction:
    """Predict GRF/COP for a foot body over a gold-standard motion clip.

    A thin wrapper that pulls the foot body's trajectory from ``result`` and runs
    :func:`biomech.contact.elastic_foundation.evaluate_contact` under prescribed
    kinematics. The returned :class:`ContactPrediction` uses the same GRF/COP fields as
    :class:`biomech.io.force_plate.ForcePlate`, so it can be compared to measured data.
    """
    params = params or ElasticFoundationParams()
    pos, quat, linvel, angvel = foot_trajectory_from_motion(result, body_name)
    return evaluate_contact(
        sole,
        params,
        pos,
        quat,
        linvel,
        angvel,
        ground_z=ground_z,
        backend=backend,
        device=device,
        keep_points=keep_points,
    )


def evaluate_both_feet_from_motion(
    result,
    sole_right: FootSole,
    sole_left: FootSole,
    params: Optional[ElasticFoundationParams] = None,
    ground_z: float = 0.0,
    right_body: str = RIGHT_FOOT_BODY,
    left_body: str = LEFT_FOOT_BODY,
    backend: str = "numpy",
    device: str = "cuda",
) -> Tuple[ContactPrediction, ContactPrediction]:
    """Predict GRF/COP for both feet (right belt == right foot; matches the split-belt
    treadmill convention used throughout ``biomech``)."""
    right = evaluate_foot_contact_from_motion(
        result, right_body, sole_right, params, ground_z, backend, device
    )
    left = evaluate_foot_contact_from_motion(
        result, left_body, sole_left, params, ground_z, backend, device
    )
    return right, left
