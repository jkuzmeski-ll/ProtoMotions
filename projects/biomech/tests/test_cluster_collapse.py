# SPDX-License-Identifier: MIT

"""Tests for thigh/shank cluster-to-centroid collapse (biomech.fitting.cluster_collapse).

Validates that :func:`collapse_clusters` (a) adds one centroid marker per cluster on the
owning body with offset = mean of the member offsets, (b) remaps the centroid to the *set*
of member capture labels while dropping the individual members, and (c) that
``build_observations`` then averages those member labels per frame (NaN-aware).

No pytest dependency: run ``python projects/biomech/run_tests.py``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from biomech.fitting.cluster_collapse import LOWER_BODY_CLUSTERS, collapse_clusters
from biomech.fitting.marker_map import build_observations, s001_marker_map

_ROOT = Path(__file__).resolve().parents[1]
_OSIM = _ROOT / "models" / "rajagopal_data" / "Rajagopal2015.osim"


def _spec():
    from biomech.osim import parse_osim

    return parse_osim(str(_OSIM))


def test_collapse_adds_centroids_and_drops_members():
    spec = _spec()
    mm = s001_marker_map()
    by_name = {m.name: m for m in spec.markers}

    new_map, added = collapse_clusters(spec, mm)
    assert set(added) == set(LOWER_BODY_CLUSTERS)

    by_name_after = {m.name: m for m in spec.markers}
    for centroid, (body, members) in LOWER_BODY_CLUSTERS.items():
        # centroid exists on the right body, offset = mean of member offsets
        m = by_name_after[centroid]
        assert m.body == body
        expect = np.mean([by_name[x].offset for x in members], axis=0)
        assert np.allclose(m.offset, expect)
        # centroid maps to the tuple of member capture labels; members are unmapped
        cap = new_map.model_to_capture[centroid]
        assert isinstance(cap, tuple)
        assert cap == tuple(mm.model_to_capture[x] for x in members)
        for x in members:
            assert x not in new_map.model_to_capture


def test_build_observations_averages_centroid():
    spec = _spec()
    mm = s001_marker_map()
    new_map, _ = collapse_clusters(spec, mm)

    from biomech.skeleton.skeleton import WarpSkeleton

    model_names = WarpSkeleton(spec).marker_names()
    # synth capture: the three right-thigh labels at known positions
    labels = ["RTHI", "RTH2", "RTH3"]
    pts = np.zeros((2, 3, 3))
    pts[:, 0, :] = [0.0, 0.0, 0.0]
    pts[:, 1, :] = [3.0, 6.0, 9.0]
    pts[:, 2, :] = [6.0, 12.0, 18.0]  # mean -> [3, 6, 9]
    obs, present = build_observations(labels, pts, model_names, new_map, to_opensim=False)
    ci = model_names.index("RTH_C")
    assert present[ci]
    assert np.allclose(obs[:, ci, :], [3.0, 6.0, 9.0])


def test_build_observations_centroid_nan_aware():
    """A missing member is ignored; the centroid is the mean of the present members."""
    spec = _spec()
    mm = s001_marker_map()
    new_map, _ = collapse_clusters(spec, mm)

    from biomech.skeleton.skeleton import WarpSkeleton

    model_names = WarpSkeleton(spec).marker_names()
    labels = ["RTHI", "RTH2", "RTH3"]
    pts = np.zeros((1, 3, 3))
    pts[:, 0, :] = [2.0, 4.0, 6.0]
    pts[:, 1, :] = [4.0, 8.0, 12.0]
    pts[:, 2, :] = np.nan  # RTH3 missing this frame
    obs, _ = build_observations(labels, pts, model_names, new_map, to_opensim=False)
    ci = model_names.index("RTH_C")
    assert np.allclose(obs[0, ci, :], [3.0, 6.0, 9.0])  # mean of the two present
