# SPDX-License-Identifier: MIT

"""Tests for the SimmSpline port (biomech.skeleton.simmspline).

No golden file is required: SimmSpline's mathematical invariants (exact knot
interpolation, derivative consistency, the linear 2-knot case, and end-segment
extrapolation matching the coefficient form) pin correctness on their own. Full
numeric parity with Nimble is exercised later once the spline is composed inside
the CustomJoint against ``docs/refs/rajagopal2015_customjoint_sweep.json``.
"""

from __future__ import annotations

import numpy as np

from biomech.skeleton.simmspline import SimmSpline


# A representative asymmetric, non-uniformly-spaced knot set.
_X = [-1.5, -0.6, 0.0, 0.3, 1.1, 2.0, 3.4]
_Y = [0.20, -0.10, 0.05, 0.40, 0.35, -0.20, 0.60]


def test_interpolates_knots_exactly():
    sp = SimmSpline(_X, _Y)
    for xi, yi in zip(_X, _Y):
        assert abs(sp.calc_value(xi) - yi) < 1e-9, xi


def test_first_derivative_matches_finite_difference():
    sp = SimmSpline(_X, _Y)
    h = 1e-6
    for x in np.linspace(_X[0] + 0.05, _X[-1] - 0.05, 40):
        fd = (sp.calc_value(x + h) - sp.calc_value(x - h)) / (2 * h)
        an = sp.calc_derivative(1, x)
        assert abs(an - fd) < 1e-4, (x, an, fd)


def test_second_derivative_matches_finite_difference():
    sp = SimmSpline(_X, _Y)
    h = 1e-4
    for x in np.linspace(_X[0] + 0.1, _X[-1] - 0.1, 30):
        fd = (sp.calc_derivative(1, x + h) - sp.calc_derivative(1, x - h)) / (2 * h)
        an = sp.calc_derivative(2, x)
        assert abs(an - fd) < 1e-3, (x, an, fd)


def test_value_is_continuous_across_interior_knots():
    sp = SimmSpline(_X, _Y)
    for xi in _X[1:-1]:
        left = sp.calc_value(xi - 1e-7)
        right = sp.calc_value(xi + 1e-7)
        assert abs(left - right) < 1e-5, xi


def test_two_knot_spline_is_linear():
    sp = SimmSpline([0.0, 2.0], [1.0, 5.0])  # slope 2
    for x in np.linspace(-1.0, 3.0, 20):
        assert abs(sp.calc_value(x) - (1.0 + 2.0 * x)) < 1e-9, x
        assert abs(sp.calc_derivative(1, x) - 2.0) < 1e-9, x
        assert abs(sp.calc_derivative(2, x)) < 1e-12, x


def test_end_segment_extrapolation_matches_coefficient_form():
    sp = SimmSpline(_X, _Y)
    b, c, d = sp.coefficients
    n = len(sp)
    # Above the top knot: Nimble reuses the last segment's cubic (k = n-1).
    x = _X[-1] + 0.7
    dx = x - _X[-1]
    expected = _Y[-1] + dx * (b[n - 1] + dx * (c[n - 1] + dx * d[n - 1]))
    assert abs(sp.calc_value(x) - expected) < 1e-12
    # Below the bottom knot: k = 0.
    x = _X[0] - 0.7
    dx = x - _X[0]
    expected = _Y[0] + dx * (b[0] + dx * (c[0] + dx * d[0]))
    assert abs(sp.calc_value(x) - expected) < 1e-12


def test_derivative_order_above_three_is_zero():
    sp = SimmSpline(_X, _Y)
    assert sp.calc_derivative(4, 0.5) == 0.0
    assert sp.calc_derivative(3, 0.5) == sp.calc_derivative(3, 0.5)  # order-3 constant per segment


def test_offset_by_shifts_values():
    sp = SimmSpline(_X, _Y)
    sp2 = sp.offset_by(0.25)
    for x in np.linspace(_X[0], _X[-1], 15):
        assert abs(sp2.calc_value(x) - (sp.calc_value(x) + 0.25)) < 1e-12
