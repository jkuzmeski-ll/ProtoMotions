# SPDX-License-Identifier: MIT
#
# Windows-native port of Nimble's fit priors
# (``dart/biomechanics/MarkerOffsetPrior.*`` and ``Anthropometrics.*``). These are the
# regularizers that resolve the scale/offset/pose gauge ambiguity in the bilevel marker
# fit: without them the fitter can trade a body scale against a marker offset against a
# pose and reach many equally-good marker reprojections, most of them anatomically wrong.
#
#   * MarkerOffsetPrior : a quadratic pull of each marker's local offset toward its model
#     (.osim) value, stronger for anatomical landmarks (which sit on bone and must not
#     drift) than for soft-tissue cluster markers. This mirrors the per-marker prior
#     weights ``MarkerFitter`` already applies inline; exposing it as an object lets the
#     pipeline build/inspect the weights and keeps the port's structure faithful.
#   * Anthropometrics : a prior on the group *scales* from subject segment lengths (the
#     Plug-in-Gait ``.mp`` metadata), so weakly-observed anisotropic scale axes stay
#     anatomically plausible instead of being dragged to the bounds by soft-tissue
#     cluster motion. The heavy lifting lives in ``anthropometry.py``; this module gives
#     it the ``priors``-namespace home the source map expects and a small dataclass API.

"""Marker-offset + anthropometric priors for the marker fit (port of Nimble priors)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from biomech.fitting.anthropometry import anthropometric_scale_prior, read_mp
from biomech.osim.spec import SkeletonSpec
from biomech.skeleton.skeleton import WarpSkeleton

__all__ = [
    "MarkerOffsetPrior",
    "Anthropometrics",
    "read_mp",
    "anthropometric_scale_prior",
]


@dataclass
class MarkerOffsetPrior:
    """Quadratic prior on marker local offsets toward the model values.

    Adds ``0.5 * sum_m w_m * ||offset_m - offset0_m||^2`` to the marker-fit objective.
    Anatomical landmarks get ``base_weight * anatomical_factor``; everything else gets
    ``base_weight``. Produces the per-marker weight vector ``MarkerFitter`` consumes as
    its ``offset_prior_weight`` schedule.
    """

    base_weight: float = 1.0
    anatomical_factor: float = 25.0

    def per_marker_weights(self, anatomical: np.ndarray) -> np.ndarray:
        """Return the ``(M,)`` prior weight for each marker given its anatomical flag."""
        anatomical = np.asarray(anatomical, dtype=bool)
        return np.where(
            anatomical,
            self.base_weight * self.anatomical_factor,
            self.base_weight,
        ).astype(np.float64)

    @staticmethod
    def anatomical_flags(spec: SkeletonSpec) -> np.ndarray:
        """Anatomical-landmark flags from the model's marker definitions."""
        return np.array([m.anatomical for m in spec.markers], dtype=bool)


@dataclass
class Anthropometrics:
    """Subject anthropometric scale prior from a Plug-in-Gait ``.mp`` file.

    Wraps :func:`biomech.fitting.anthropometry.anthropometric_scale_prior` in a small,
    reusable object. Call :meth:`scale_prior` to get the ``(target, weights)`` vectors to
    hand to ``MarkerFitConfig.scale_prior_target`` / ``scale_prior_weights``.
    """

    mp_path: str | Path
    length_weight: float = 20.0
    width_weight: float = 3.0
    include_upper_body: bool = True

    def measurements(self) -> dict[str, float]:
        """Raw parsed ``.mp`` key -> value (millimeters/degrees as stored)."""
        return read_mp(self.mp_path)

    def scale_prior(
        self, spec: SkeletonSpec
    ) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
        """``(target, weights, diagnostics)`` over ``3 * num_scale_groups`` axes."""
        return anthropometric_scale_prior(
            spec,
            self.mp_path,
            length_weight=self.length_weight,
            width_weight=self.width_weight,
            include_upper_body=self.include_upper_body,
        )

    def apply_to_config(self, spec: SkeletonSpec, config) -> dict[str, float]:
        """Set ``config.scale_prior_target/weights`` in place; return diagnostics."""
        target, weights, diag = self.scale_prior(spec)
        config.scale_prior_target = target
        config.scale_prior_weights = weights
        return diag
