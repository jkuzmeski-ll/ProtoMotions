# SPDX-License-Identifier: MIT
#
# Collapse soft-tissue marker clusters (thigh / shank tracking plates) to one centroid.
#
# The Rajagopal thigh and tibia tracking clusters (``RTH1/2/3``, ``RTB1/2/3`` + left) sit
# on soft tissue that slides relative to the bone during gait. Fitting all three members
# individually feeds the pose IK three mutually-inconsistent constraints per segment: they
# carry the largest per-marker reprojection residual (see fig 10) and pull the segment's
# long-axis rotation around frame-to-frame. Averaging each cluster to a single centroid
# keeps the well-averaged segment *position* constraint while removing the conflicting
# soft-tissue pull. The tradeoff is losing the cluster's long-axis-rotation information
# (which surface plates resolve poorly anyway).
#
# Applied *before* reconstruction: adds one centroid marker per cluster to the model
# (offset = mean of the member offsets, then refined by the fitter like any tracking
# marker) and returns a ``MarkerMap`` whose centroid model marker maps to the *set* of
# member capture labels. ``build_observations`` averages that set per frame (NaN-aware).
# The individual member markers are left unmapped so they no longer constrain the fit.

"""Collapse thigh/shank soft-tissue marker clusters to a single centroid marker."""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from biomech.fitting.marker_map import MarkerMap
from biomech.osim.spec import MarkerSpec, SkeletonSpec

# centroid model name -> (owning body, member model marker names)
LOWER_BODY_CLUSTERS: Dict[str, Tuple[str, List[str]]] = {
    "RTH_C": ("femur_r", ["RTH1", "RTH2", "RTH3"]),
    "LTH_C": ("femur_l", ["LTH1", "LTH2", "LTH3"]),
    "RTB_C": ("tibia_r", ["RTB1", "RTB2", "RTB3"]),
    "LTB_C": ("tibia_l", ["LTB1", "LTB2", "LTB3"]),
}


def collapse_clusters(
    spec: SkeletonSpec,
    mapping: MarkerMap,
    clusters: Dict[str, Tuple[str, List[str]]] = LOWER_BODY_CLUSTERS,
) -> Tuple[MarkerMap, List[str]]:
    """Add centroid markers to ``spec`` (in place) and return a collapsed ``MarkerMap``.

    For each cluster a single centroid marker is added on the owning body with offset =
    mean of the member offsets (the fitter refines it). The returned map points the
    centroid model marker at the *tuple* of member capture labels (averaged per frame by
    :func:`build_observations`) and drops the individual member markers.

    Returns ``(new_map, added)`` where ``added`` is the list of centroid names created.
    Clusters with fewer than 2 present members are skipped unchanged.
    """
    m2c: Dict[str, object] = dict(mapping.model_to_capture)
    anat = set(mapping.anatomical)
    by_name = {m.name: m for m in spec.markers}
    existing = set(by_name)
    added: List[str] = []

    for centroid, (body, members) in clusters.items():
        offs = [by_name[m].offset for m in members if m in by_name]
        caps = [m2c[m] for m in members if isinstance(m2c.get(m), str)]
        if len(offs) < 2 or len(caps) < 2:
            continue  # not enough of this cluster present; leave it alone
        centroid_offset = np.mean(np.stack(offs, axis=0), axis=0)
        if centroid not in existing:
            spec.markers.append(
                MarkerSpec(name=centroid, body=body, offset=centroid_offset, fixed=False)
            )
            added.append(centroid)
        m2c[centroid] = tuple(caps)
        for m in members:
            m2c.pop(m, None)
            anat.discard(m)

    return MarkerMap(model_to_capture=m2c, anatomical=anat), added
