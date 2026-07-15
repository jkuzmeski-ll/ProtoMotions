# SPDX-License-Identifier: MIT

"""Treadmill-to-overground (TM2OG) mapping of a motion clip.

Port of the *virtual-origin* method of Jung & Lee, "Treadmill-to-Overground
Mapping of Marker Trajectory for Treadmill-Based Continuous Gait Analysis",
*Sensors* 21(3):786, 2021 (DOI 10.3390/s21030786), adapted to drive a
ProtoMotions/Newton clip.

Paper concept (Eqs. 2, 6, 7)
----------------------------
As the belt moves backward, a frame attached to the belt -- the *virtual origin*
-- moves backward with it. The position vector from that backward-moving virtual
origin to a marker inside the limited treadmill volume equals the position vector
from a *fixed* origin to the equivalent forward-moving overground marker::

    p_vo(t)      = p_vo(t-1) + Δp_vo(t)        # virtual origin (moves backward)
    p_hat_bn(t)  = p_bn(t)   - p_vo(t)         # overground-mapped body position

The virtual-origin increment ``Δp_vo`` is the belt displacement over one step,
projected onto the belt travel axis. The paper recovers it from a marker chain on
the belt; **we have no belt markers**, so we use the instrumented-treadmill belt
speed log directly as ground truth: ``Δp_vo = v_belt·dt`` along the belt travel
direction. The belt moves *backward* under the feet, so ``p_vo`` accumulates in
the backward direction and ``-p_vo`` is a growing *forward* displacement::

    x_overground(t) = x_treadmill(t) + ∫₀ᵗ v_belt(τ) dτ   (along +forward)

For a physics sim we also apply the corresponding **Galilean velocity shift**
(the paper only remaps positions): every body's linear velocity gains ``v_belt``
along +forward, so a foot that was planted on the belt (moving backward at the
belt speed) becomes ~stationary overground, as it should be.

This is a pure fore-aft translation + velocity offset: rotations, angular
velocities and joint DOFs are untouched, so relative body geometry (and hence
foot-ground contact) is preserved exactly.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

# calcn contacts the belt for the largest fraction of stance, so it is the most
# reliable body for inferring the belt travel axis from a reconstructed clip.
_DEFAULT_FOOT_BODIES = ("calcn_r", "calcn_l")


def cumulative_belt_displacement(belt_speed: np.ndarray, fps: float) -> np.ndarray:
    """Virtual-origin travel distance ``∫ v_belt dt`` per frame (starts at 0).

    Trapezoidal cumulative integral of a per-frame belt-speed trace (m/s) sampled
    at ``fps`` Hz. Returns a ``(F,)`` array of non-negative distances (m) when the
    belt speed is non-negative; frame 0 is 0.
    """

    belt_speed = np.asarray(belt_speed, dtype=np.float64)
    if belt_speed.ndim != 1:
        raise ValueError(f"belt_speed must be 1-D, got shape {belt_speed.shape}")
    dt = 1.0 / float(fps)
    disp = np.zeros_like(belt_speed)
    if belt_speed.size > 1:
        disp[1:] = np.cumsum(0.5 * (belt_speed[1:] + belt_speed[:-1]) * dt)
    return disp


def infer_travel_direction(
    rigid_body_pos: np.ndarray,
    fps: float,
    foot_indices: Sequence[int],
    *,
    up_axis: int = 2,
    stance_percentile: float = 30.0,
) -> np.ndarray:
    """Infer the overground **forward** unit vector (horizontal) from stance feet.

    Physical anchor (independent of any assumed lab axis): a foot planted on the
    belt moves *backward* with the belt, so the mean horizontal velocity of the
    feet during their low-height (stance) frames points along the belt travel
    direction. Overground forward is the opposite of that::

        travel_dir = -normalize(mean stance-foot horizontal velocity)

    The returned vector lies in the horizontal plane (zero on ``up_axis``).
    """

    pos = np.asarray(rigid_body_pos, dtype=np.float64)
    if pos.ndim != 3:
        raise ValueError(f"rigid_body_pos must be (F,B,3), got {pos.shape}")
    vel = np.gradient(pos, axis=0) * float(fps)  # (F,B,3)
    horiz = [a for a in range(3) if a != up_axis]

    drag = np.zeros(2, dtype=np.float64)
    n = 0
    for i in foot_indices:
        z = pos[:, i, up_axis]
        thr = np.percentile(z, stance_percentile)
        stance = z <= thr
        if not np.any(stance):
            continue
        drag += vel[stance, i][:, horiz].mean(axis=0)
        n += 1
    if n == 0 or not np.linalg.norm(drag):
        raise ValueError("could not infer belt travel direction from stance feet")
    drag /= np.linalg.norm(drag)

    travel = np.zeros(3, dtype=np.float64)
    travel[horiz[0]] = -drag[0]
    travel[horiz[1]] = -drag[1]
    return travel


def apply_tm2og(
    rigid_body_pos: np.ndarray,
    rigid_body_vel: np.ndarray,
    belt_speed: np.ndarray,
    fps: float,
    travel_dir: Sequence[float],
) -> tuple[np.ndarray, np.ndarray]:
    """Map a treadmill clip to overground in place-safe (returns new arrays).

    Adds the virtual-origin forward displacement ``∫ v_belt dt`` to every body's
    position and the belt speed ``v_belt`` to every body's linear velocity, both
    along ``travel_dir`` (the overground forward unit vector). Frame 0 keeps its
    original position (displacement starts at 0) but its velocity is already
    shifted, so the whole clip is a consistent overground trajectory.

    Parameters
    ----------
    rigid_body_pos, rigid_body_vel
        ``(F, B, 3)`` world position / linear velocity arrays.
    belt_speed
        ``(F,)`` per-frame belt speed (m/s), ground truth from the belt log.
    fps
        Clip frame rate (Hz).
    travel_dir
        Overground forward unit vector (3,); e.g. from
        :func:`infer_travel_direction`.
    """

    pos = np.array(rigid_body_pos, dtype=np.float64, copy=True)
    vel = np.array(rigid_body_vel, dtype=np.float64, copy=True)
    belt_speed = np.asarray(belt_speed, dtype=np.float64)
    tdir = np.asarray(travel_dir, dtype=np.float64)

    F = pos.shape[0]
    if belt_speed.shape[0] != F:
        raise ValueError(
            f"belt_speed has {belt_speed.shape[0]} frames but clip has {F}"
        )

    disp = cumulative_belt_displacement(belt_speed, fps)  # (F,)
    pos += disp[:, None, None] * tdir[None, None, :]
    vel += belt_speed[:, None, None] * tdir[None, None, :]
    return pos, vel


def tm2og_motion(
    data: dict,
    belt_speed: np.ndarray,
    fps: float,
    body_names: Sequence[str],
    *,
    travel_dir: Optional[Sequence[float]] = None,
    foot_bodies: Sequence[str] = _DEFAULT_FOOT_BODIES,
    up_axis: int = 2,
) -> np.ndarray:
    """Apply TM2OG to a motion ``data`` dict in place; return the travel_dir used.

    Shifts ``data['rigid_body_pos']`` and ``data['rigid_body_vel']`` only (dtype
    preserved); rotations, angular velocities and DOFs are left untouched. If
    ``travel_dir`` is None it is inferred from the stance feet named in
    ``foot_bodies``.
    """

    import torch

    pos_t = data["rigid_body_pos"]
    vel_t = data["rigid_body_vel"]
    is_torch = isinstance(pos_t, torch.Tensor)
    pos = pos_t.detach().cpu().numpy() if is_torch else np.asarray(pos_t)
    vel = vel_t.detach().cpu().numpy() if is_torch else np.asarray(vel_t)

    if travel_dir is None:
        name_to_idx = {n: i for i, n in enumerate(body_names)}
        foot_indices = [name_to_idx[n] for n in foot_bodies if n in name_to_idx]
        if not foot_indices:
            raise ValueError(
                f"none of foot_bodies={list(foot_bodies)} in body_names"
            )
        travel_dir = infer_travel_direction(
            pos, fps, foot_indices, up_axis=up_axis
        )

    new_pos, new_vel = apply_tm2og(pos, vel, belt_speed, fps, travel_dir)

    if is_torch:
        data["rigid_body_pos"] = torch.as_tensor(new_pos, dtype=pos_t.dtype)
        data["rigid_body_vel"] = torch.as_tensor(new_vel, dtype=vel_t.dtype)
    else:
        data["rigid_body_pos"] = new_pos.astype(pos.dtype, copy=False)
        data["rigid_body_vel"] = new_vel.astype(vel.dtype, copy=False)
    return np.asarray(travel_dir, dtype=np.float64)
