# SPDX-License-Identifier: MIT
#
# Port of Nimble's ``dart/math/SimmSpline.{hpp,cpp}`` to Windows-native NumPy.
#
# The algorithm is adapted (via Nimble, MIT) from OpenSim's ``SimmSpline`` by
# Peter Loan, originally Apache-2.0:
#   Copyright (c) 2005-2017 Stanford University and the Authors.
# The natural-cubic fit with SIMM's specific end conditions and the clamped /
# end-segment extrapolation behavior are reproduced exactly for bit-comparable
# parity with Nimble (which is our gold-standard reference).

"""OpenSim/SIMM cubic spline used for OpenSim ``CustomJoint`` coordinate coupling.

This is the coupling function behind the gold-standard coupled knee in the
Rajagopal model (``walker_knee_*`` map ``knee_angle`` to Euler + translation via
these splines). The class builds the piecewise-cubic coefficients ``b, c, d`` from
knots ``(x, y)`` and evaluates the value and derivatives; the Warp batched-FK path
(later) reuses the same coefficient arrays inside a device function.

Reference: ``reference/nimble/dart/math/SimmSpline.cpp`` and constants in
``dart/math/MathTypes.hpp`` (``TINY_NUMBER``, ``ROUNDOFF_ERROR``).
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

# Lifted verbatim from Nimble's dart/math/MathTypes.hpp so the fit matches.
TINY_NUMBER = 1.0e-7
ROUNDOFF_ERROR = 2.0e-13


def _equal_within_error(a: float, b: float) -> bool:
    return abs(a - b) <= ROUNDOFF_ERROR


class SimmSpline:
    """SIMM natural cubic spline (exact port of Nimble/OpenSim ``SimmSpline``).

    Parameters
    ----------
    x, y:
        Knot abscissae (must be monotonically increasing) and ordinates. At least
        two knots are required.

    On each interval ``[x[k], x[k+1]]`` the spline is
    ``y[k] + dx*(b[k] + dx*(c[k] + dx*d[k]))`` with ``dx = X - x[k]``. Out-of-range
    inputs reuse the nearest end segment's cubic (matching Nimble's active code
    path; the slope-extrapolation branch is commented out in the C++).
    """

    def __init__(self, x: Sequence[float], y: Sequence[float]) -> None:
        self._x = [float(v) for v in x]
        self._y = [float(v) for v in y]
        if len(self._x) != len(self._y):
            raise ValueError("SimmSpline: x and y must have the same length")
        if len(self._x) < 2:
            raise ValueError("SimmSpline: need at least 2 knots")
        self._b: list[float] = []
        self._c: list[float] = []
        self._d: list[float] = []
        self._calc_coefficients()

    # -- construction -------------------------------------------------------
    def _calc_coefficients(self) -> None:
        x, y = self._x, self._y
        n = len(x)
        b = [0.0] * n
        c = [0.0] * n
        d = [0.0] * n

        if n == 2:
            t = max(TINY_NUMBER, x[1] - x[0])
            b[0] = b[1] = (y[1] - y[0]) / t
            self._b, self._c, self._d = b, c, d
            return

        nm1 = n - 1
        nm2 = n - 2

        # Set up tridiagonal system: b=diagonal, d=offdiagonal, c=rhs.
        d[0] = max(TINY_NUMBER, x[1] - x[0])
        c[1] = (y[1] - y[0]) / d[0]
        for i in range(1, nm1):
            d[i] = max(TINY_NUMBER, x[i + 1] - x[i])
            b[i] = 2.0 * (d[i - 1] + d[i])
            c[i + 1] = (y[i + 1] - y[i]) / d[i]
            c[i] = c[i + 1] - c[i]

        # End conditions: 3rd derivatives at the ends from divided differences.
        b[0] = -d[0]
        b[nm1] = -d[nm2]
        c[0] = 0.0
        c[nm1] = 0.0

        if n > 3:
            d31 = max(TINY_NUMBER, x[3] - x[1])
            d20 = max(TINY_NUMBER, x[2] - x[0])
            d1 = max(TINY_NUMBER, x[nm1] - x[n - 3])
            d2 = max(TINY_NUMBER, x[nm2] - x[n - 4])
            d30 = max(TINY_NUMBER, x[3] - x[0])
            d3 = max(TINY_NUMBER, x[nm1] - x[n - 4])
            c[0] = c[2] / d31 - c[1] / d20
            c[nm1] = c[nm2] / d1 - c[n - 3] / d2
            c[0] = c[0] * d[0] * d[0] / d30
            c[nm1] = -c[nm1] * d[nm2] * d[nm2] / d3

        # Forward elimination.
        for i in range(1, n):
            t = d[i - 1] / b[i - 1]
            b[i] -= t * d[i - 1]
            c[i] -= t * c[i - 1]

        # Back substitution.
        c[nm1] /= b[nm1]
        for j in range(0, nm1):
            i = nm2 - j
            c[i] = (c[i] - d[i] * c[i + 1]) / b[i]

        # Polynomial coefficients.
        b[nm1] = (y[nm1] - y[nm2]) / d[nm2] + d[nm2] * (c[nm2] + 2.0 * c[nm1])
        for i in range(0, nm1):
            b[i] = (y[i + 1] - y[i]) / d[i] - d[i] * (c[i + 1] + 2.0 * c[i])
            d[i] = (c[i + 1] - c[i]) / d[i]
            c[i] *= 3.0
        c[nm1] *= 3.0
        d[nm1] = d[nm2]

        self._b, self._c, self._d = b, c, d

    # -- interval lookup ----------------------------------------------------
    def _find_interval(self, ax: float) -> int:
        x = self._x
        n = len(x)
        if n < 3:
            return 0
        if _equal_within_error(ax, x[0]) or ax < x[0]:
            return 0
        if _equal_within_error(ax, x[n - 1]) or ax > x[n - 1]:
            return n - 1
        i, j = 0, n
        while True:
            k = (i + j) // 2
            if ax < x[k]:
                j = k
            elif ax > x[k + 1]:
                i = k
            else:
                return k

    # -- evaluation ---------------------------------------------------------
    def calc_value(self, x: float) -> float:
        k = self._find_interval(float(x))
        dx = float(x) - self._x[k]
        b, c, d = self._b, self._c, self._d
        return self._y[k] + dx * (b[k] + dx * (c[k] + dx * d[k]))

    def calc_derivative(self, order: int, x: float) -> float:
        if order > 3:
            return 0.0
        k = self._find_interval(float(x))
        dx = float(x) - self._x[k]
        b, c, d = self._b, self._c, self._d
        if order == 1:
            return b[k] + dx * (2.0 * c[k] + 3.0 * dx * d[k])
        if order == 2:
            return 2.0 * c[k] + 6.0 * dx * d[k]
        if order == 3:
            return 6.0 * d[k]
        return 0.0

    # -- vectorized convenience --------------------------------------------
    def calc_value_array(self, xs) -> np.ndarray:
        xs = np.asarray(xs, dtype=np.float64).ravel()
        return np.array([self.calc_value(float(v)) for v in xs], dtype=np.float64)

    def calc_derivative_array(self, order: int, xs) -> np.ndarray:
        xs = np.asarray(xs, dtype=np.float64).ravel()
        return np.array([self.calc_derivative(order, float(v)) for v in xs], dtype=np.float64)

    # -- accessors / ops ----------------------------------------------------
    @property
    def x(self) -> np.ndarray:
        return np.array(self._x, dtype=np.float64)

    @property
    def y(self) -> np.ndarray:
        return np.array(self._y, dtype=np.float64)

    @property
    def coefficients(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Piecewise-cubic ``(b, c, d)`` (for the Warp device-function path)."""
        return (
            np.array(self._b, dtype=np.float64),
            np.array(self._c, dtype=np.float64),
            np.array(self._d, dtype=np.float64),
        )

    def offset_by(self, offset: float) -> "SimmSpline":
        return SimmSpline(self._x, [v + float(offset) for v in self._y])

    def __len__(self) -> int:
        return len(self._x)
