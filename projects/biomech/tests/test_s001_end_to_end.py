# SPDX-License-Identifier: MIT

"""End-to-end smoke test on the real S001 capture (skipped if data/torch absent).

Exercises the full real-data path that the marker bridge unblocks:

    load_session -> observations_from_session -> IKInitializer.run
      -> build_motion(fitted q) -> foot pose trajectory -> elastic-foundation contact

This is a *smoke* test: it verifies the pieces connect and produce finite, sane outputs
on real (noisy, gappy) markers, not that the biomechanical fit is publication-grade
(that needs the deferred anthropometric prior + marker polishing). It runs on a small
frame window for speed.

No pytest dependency: run ``python projects/biomech/run_tests.py``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from biomech.fitting.marker_map import (
    anatomical_mask,
    observations_from_session,
    s001_marker_map,
)
from biomech.tests import (
    LEFT_BELT,
    RIGHT_BELT,
    SkipTest,
    TRIAL_C3D,
    require,
)

_ROOT = Path(__file__).resolve().parents[1]
_OSIM = _ROOT / "models" / "rajagopal_data" / "Rajagopal2015.osim"


def _require_torch():
    try:
        import torch  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        raise SkipTest(f"torch not available: {exc}")


def _pick_window(obs, present, n=6):
    """Pick a contiguous window where most mapped markers are visible."""
    vis = np.isfinite(obs[:, present, :]).all(axis=2).mean(axis=1)  # (F,)
    F = obs.shape[0]
    best_start, best_score = 0, -1.0
    step = max(1, F // 200)
    for s in range(0, F - n, step):
        score = vis[s:s + n].mean()
        if score > best_score:
            best_score, best_start = score, s
    return best_start


def test_s001_end_to_end_smoke():
    require(TRIAL_C3D)
    _require_torch()

    from biomech.contact.elastic_foundation import sample_flat_sole
    from biomech.contact.kinematics import evaluate_foot_contact_from_motion
    from biomech.export.motion import build_motion
    from biomech.fitting.ik import MarkerIKConfig
    from biomech.fitting.ik_initializer import IKInitializer
    from biomech.osim import parse_osim
    from biomech.session import load_session
    from biomech.skeleton.skeleton import WarpSkeleton

    session = load_session(str(TRIAL_C3D))
    spec = parse_osim(str(_OSIM))
    skel = WarpSkeleton(spec)
    model_names = skel.marker_names()

    mm = s001_marker_map()
    obs, present = observations_from_session(session, model_names, mm)
    anat = anatomical_mask(model_names, mm)
    assert present.sum() >= 30
    assert anat.sum() >= 10

    # small window with good visibility
    n = 6
    s = _pick_window(obs, present, n=n)
    obs_win = obs[s:s + n]

    init = IKInitializer(skel, obs_win, anatomical=anat)
    result = init.run(MarkerIKConfig(max_iters=40))

    # scales and poses are finite and plausible
    assert result.group_scales.shape[0] == 3 * len(spec.scale_groups)
    assert np.all(np.isfinite(result.group_scales))
    assert np.all(result.group_scales > 0.3) and np.all(result.group_scales < 3.0)
    assert result.poses.shape == (n, spec.num_dofs)
    assert np.all(np.isfinite(result.poses))
    # marker RMS is finite and within a loose real-data bound (PiG vs model offsets)
    assert np.all(np.isfinite(result.marker_rms))
    assert np.nanmedian(result.marker_rms) < 0.10  # < 10 cm median

    # feed the fitted pose + scale through the gold-standard motion exporter
    fps = session.point_rate
    motion = build_motion(spec, result.poses, fps=fps, group_scales=result.group_scales)
    assert "calcn_r" in motion.body_names
    posr, quatr, _, _ = (
        motion.data["rigid_body_pos"],
        motion.data["rigid_body_rot"],
        None,
        None,
    )
    # drive the distributed contact model with the real reconstructed foot motion:
    # set the ground just above the lowest right-foot sample so it makes contact.
    from biomech.contact.kinematics import foot_trajectory_from_motion

    fp, _, _, _ = foot_trajectory_from_motion(motion, "calcn_r")
    ground_z = float(np.min(fp[:, 2])) + 0.005
    sole = sample_flat_sole(0.22, 0.09, 8, 4)
    pred = evaluate_foot_contact_from_motion(
        motion, "calcn_r", sole, ground_z=ground_z, backend="numpy"
    )
    assert pred.grf.shape == (n, 3)
    assert np.all(np.isfinite(pred.grf))
    assert np.any(pred.total_normal > 0.0)  # some frame is in contact
