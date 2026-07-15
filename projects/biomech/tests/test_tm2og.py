# SPDX-License-Identifier: MIT

"""Treadmill-to-overground (TM2OG) mapping tests.

Validates the virtual-origin port (Jung & Lee, Sensors 21(3):786, 2021): the
overground forward displacement equals the belt travel distance, progress is
monotonic under a constant belt, rotations/DOFs are untouched, and the stance
foot is (near) stationary overground.
"""

import numpy as np

from biomech.export.tm2og import (
    apply_tm2og,
    cumulative_belt_displacement,
    infer_travel_direction,
    tm2og_motion,
)


def _synthetic_treadmill_clip(fps=100.0, F=150, belt=1.5):
    """A body walking-in-place: one foot planted (moves +Y with belt) at a time.

    Returns (data_dict, body_names, belt_speed). Bodies: root(0), calcn_r(1),
    calcn_l(2). Feet alternate stance/swing every half of a 1 s cycle; during
    stance a foot sits low and moves at the belt speed along +Y (the belt drag
    axis), during swing it lifts and swings forward (-Y).
    """
    import torch

    t = np.arange(F) / fps
    body_names = ["root", "calcn_r", "calcn_l"]
    pos = np.zeros((F, 3, 3), dtype=np.float64)
    # root: bobs vertically, no net horizontal motion (treadmill).
    pos[:, 0, 2] = 1.0 + 0.02 * np.sin(2 * np.pi * t)
    phase = (t % 1.0) < 0.5  # True -> right foot stance
    for f in range(F):
        # right foot
        if phase[f]:
            pos[f, 1, 2] = 0.02  # planted low
        else:
            pos[f, 1, 2] = 0.10  # lifted
        # left foot (opposite)
        if not phase[f]:
            pos[f, 2, 2] = 0.02
        else:
            pos[f, 2, 2] = 0.10
    # planted feet ride the belt in +Y at belt speed; give them that Y position.
    yr = np.zeros(F)
    yl = np.zeros(F)
    for f in range(1, F):
        yr[f] = yr[f - 1] + (belt / fps if phase[f] else -belt / fps)
        yl[f] = yl[f - 1] + (belt / fps if not phase[f] else -belt / fps)
    pos[:, 1, 1] = yr
    pos[:, 2, 1] = yl

    vel = np.gradient(pos, axis=0) * fps
    rot = np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (F, 3, 1))  # identity xyzw
    ang = np.zeros((F, 3, 3))
    dof = np.zeros((F, 5))

    def t32(a):
        return torch.as_tensor(np.asarray(a, dtype=np.float32))

    data = {
        "rigid_body_pos": t32(pos),
        "rigid_body_rot": t32(rot),
        "rigid_body_vel": t32(vel),
        "rigid_body_ang_vel": t32(ang),
        "dof_pos": t32(dof),
        "dof_vel": t32(dof),
        "fps": float(fps),
    }
    belt_speed = np.full(F, belt, dtype=np.float64)
    return data, body_names, belt_speed


def test_cumulative_displacement_matches_integral():
    fps = 100.0
    F = 150
    belt = 1.5
    disp = cumulative_belt_displacement(np.full(F, belt), fps)
    assert disp[0] == 0.0
    # constant belt -> Δx(t) = v * t; total over F-1 steps
    expected = belt * (F - 1) / fps
    assert abs(disp[-1] - expected) < 1e-9
    # monotonic non-decreasing
    assert np.all(np.diff(disp) >= -1e-12)


def test_cumulative_displacement_variable_speed():
    fps = 100.0
    v = np.linspace(1.0, 2.0, 200)  # ramping belt
    disp = cumulative_belt_displacement(v, fps)
    # trapezoidal reference
    ref = np.concatenate([[0.0], np.cumsum(0.5 * (v[1:] + v[:-1]) / fps)])
    assert np.allclose(disp, ref, atol=1e-12)


def test_infer_travel_direction_is_minus_y():
    data, names, _ = _synthetic_treadmill_clip()
    pos = data["rigid_body_pos"].numpy().astype(np.float64)
    tdir = infer_travel_direction(pos, 100.0, foot_indices=[1, 2])
    # belt drags feet +Y; overground forward is -Y.
    assert np.allclose(tdir, [0.0, -1.0, 0.0], atol=1e-6)


def test_total_travel_equals_belt_distance():
    """Paper's key check: TTD(overground) == ∫ v_belt dt (~0.3% in the paper)."""
    data, names, belt = _synthetic_treadmill_clip(belt=1.5)
    tdir = tm2og_motion(data, belt, 100.0, names)
    pos = data["rigid_body_pos"].numpy().astype(np.float64)
    root_travel = pos[-1, 0] - pos[0, 0]
    dist = float(np.linalg.norm(root_travel))
    expected = 1.5 * (pos.shape[0] - 1) / 100.0
    assert abs(dist - expected) / expected < 3e-3
    # forward direction is -Y
    assert root_travel[1] < 0


def test_stance_foot_becomes_stationary_overground():
    data, names, belt = _synthetic_treadmill_clip(belt=1.5)
    fps = 100.0
    # stance-foot horizontal speed BEFORE mapping (belt drag ~1.5 m/s).
    pos0 = data["rigid_body_pos"].numpy().astype(np.float64)
    phase = (np.arange(pos0.shape[0]) / fps % 1.0) < 0.5
    v0 = np.gradient(pos0, axis=0)[:, 1, 1] * fps  # right foot Vy
    before = np.abs(v0[phase]).mean()
    tm2og_motion(data, belt, fps, names)
    pos1 = data["rigid_body_pos"].numpy().astype(np.float64)
    v1 = np.gradient(pos1, axis=0)[:, 1, 1] * fps
    after = np.abs(v1[phase]).mean()
    assert before > 1.0  # foot really was riding the belt
    assert after < 0.15  # ~stationary overground


def test_rotations_and_dofs_untouched():
    data, names, belt = _synthetic_treadmill_clip()
    rot0 = data["rigid_body_rot"].clone()
    ang0 = data["rigid_body_ang_vel"].clone()
    dof0 = data["dof_pos"].clone()
    tm2og_motion(data, belt, 100.0, names)
    assert np.array_equal(data["rigid_body_rot"].numpy(), rot0.numpy())
    assert np.array_equal(data["rigid_body_ang_vel"].numpy(), ang0.numpy())
    assert np.array_equal(data["dof_pos"].numpy(), dof0.numpy())


def test_velocity_shift_consistent_with_position():
    """v gains v_belt along travel_dir; d(pos)/dt should track the new v."""
    data, names, belt = _synthetic_treadmill_clip(belt=1.2)
    fps = 100.0
    vel_before = data["rigid_body_vel"].numpy().astype(np.float64).copy()
    tdir = tm2og_motion(data, belt, fps, names)
    vel_after = data["rigid_body_vel"].numpy().astype(np.float64)
    delta = vel_after - vel_before
    # every body's velocity increment == belt_speed * travel_dir
    expected = belt[:, None, None] * np.asarray(tdir)[None, None, :]
    assert np.allclose(delta, expected, atol=1e-4)


def test_first_frame_position_unchanged():
    data, names, belt = _synthetic_treadmill_clip()
    pos0 = data["rigid_body_pos"].numpy().astype(np.float64).copy()
    tm2og_motion(data, belt, 100.0, names)
    pos1 = data["rigid_body_pos"].numpy().astype(np.float64)
    # displacement starts at 0 -> frame 0 positions identical
    assert np.allclose(pos1[0], pos0[0], atol=1e-6)


def test_apply_tm2og_numpy_arrays_directly():
    F = 50
    pos = np.zeros((F, 2, 3))
    vel = np.zeros((F, 2, 3))
    belt = np.full(F, 2.0)
    tdir = [0.0, -1.0, 0.0]
    p, v = apply_tm2og(pos, vel, belt, 100.0, tdir)
    assert p[0, 0, 1] == 0.0
    assert p[-1, 0, 1] < 0  # moved in -Y
    assert np.allclose(v[:, :, 1], -2.0)
    # input arrays not mutated
    assert np.all(pos == 0.0)
