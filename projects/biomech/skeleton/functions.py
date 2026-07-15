# SPDX-License-Identifier: MIT
#
# Coupling-function evaluation for the biomech skeleton FK. Ports the value path of
# Nimble's ``math::CustomFunction`` subclasses used by the Rajagopal model's joints:
# ``ConstantFunction``, ``LinearFunction`` and (the gold-standard part) ``SimmSpline``
# (see ``skeleton/simmspline.py``). The functions are flattened into flat arrays so a
# single Warp device function can evaluate any of them inside the batched FK kernel,
# which keeps the whole FK differentiable w.r.t. ``q`` via Warp autodiff.

"""Flattened coupling-function table + Warp/NumPy evaluation (M2b)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import warp as wp

from biomech.osim.spec import (
    ConstantFunctionSpec,
    CouplingFunction,
    LinearFunctionSpec,
    MultiplierFunctionSpec,
    SimmSplineSpec,
)

FN_CONSTANT = 0
FN_LINEAR = 1
FN_SIMMSPLINE = 2


@dataclass
class FunctionTable:
    """Flat, Warp-ready arrays describing a set of coupling functions.

    Each function ``fid`` has a type in ``fn_type`` and:
      - ``FN_CONSTANT``: value in ``fn_p0``.
      - ``FN_LINEAR``:   slope in ``fn_p0``, intercept in ``fn_p1``.
      - ``FN_SIMMSPLINE``: knots/coeffs in ``fn_x/fn_y/fn_b/fn_c/fn_d`` over the
        half-open block ``[fn_kstart, fn_kstart + fn_kcount)``.
    """

    fn_type: np.ndarray
    fn_p0: np.ndarray
    fn_p1: np.ndarray
    fn_kstart: np.ndarray
    fn_kcount: np.ndarray
    fn_x: np.ndarray
    fn_y: np.ndarray
    fn_b: np.ndarray
    fn_c: np.ndarray
    fn_d: np.ndarray

    def eval_np(self, fid: int, x: float) -> float:
        t = int(self.fn_type[fid])
        if t == FN_CONSTANT:
            return float(self.fn_p0[fid])
        if t == FN_LINEAR:
            return float(self.fn_p0[fid] * x + self.fn_p1[fid])
        # SimmSpline
        k0 = int(self.fn_kstart[fid])
        n = int(self.fn_kcount[fid])
        xs = self.fn_x[k0 : k0 + n]
        k = _simm_interval_np(xs, float(x))
        dx = float(x) - float(xs[k])
        b = self.fn_b[k0 + k]
        c = self.fn_c[k0 + k]
        d = self.fn_d[k0 + k]
        return float(self.fn_y[k0 + k] + dx * (b + dx * (c + dx * d)))


def _simm_interval_np(xs: np.ndarray, ax: float) -> int:
    n = len(xs)
    if n < 3:
        return 0
    if ax <= xs[0]:
        return 0
    if ax >= xs[n - 1]:
        return n - 1
    i, j = 0, n
    while True:
        k = (i + j) // 2
        if ax < xs[k]:
            j = k
        elif ax > xs[k + 1]:
            i = k
        else:
            return k


@dataclass
class FunctionTableBuilder:
    """Accumulates coupling functions and assigns each a stable ``fid``."""

    _type: list = field(default_factory=list)
    _p0: list = field(default_factory=list)
    _p1: list = field(default_factory=list)
    _kstart: list = field(default_factory=list)
    _kcount: list = field(default_factory=list)
    _x: list = field(default_factory=list)
    _y: list = field(default_factory=list)
    _b: list = field(default_factory=list)
    _c: list = field(default_factory=list)
    _d: list = field(default_factory=list)

    def add(self, fn: CouplingFunction) -> int:
        # The parser already folds MultiplierFunction into the leaf, but recurse
        # defensively in case a spec is built by hand.
        if isinstance(fn, MultiplierFunctionSpec):
            raise ValueError(
                "MultiplierFunctionSpec must be folded into its leaf before "
                "building a FunctionTable (parser does this)."
            )
        fid = len(self._type)
        if isinstance(fn, ConstantFunctionSpec):
            self._type.append(FN_CONSTANT)
            self._p0.append(float(fn.constant))
            self._p1.append(0.0)
            self._kstart.append(0)
            self._kcount.append(0)
        elif isinstance(fn, LinearFunctionSpec):
            self._type.append(FN_LINEAR)
            self._p0.append(float(fn.slope))
            self._p1.append(float(fn.intercept))
            self._kstart.append(0)
            self._kcount.append(0)
        elif isinstance(fn, SimmSplineSpec):
            spline = fn.spline()
            b, c, d = spline.coefficients
            k0 = len(self._x)
            n = len(spline)
            self._x.extend(spline.x.tolist())
            self._y.extend(spline.y.tolist())
            self._b.extend(b.tolist())
            self._c.extend(c.tolist())
            self._d.extend(d.tolist())
            self._type.append(FN_SIMMSPLINE)
            self._p0.append(0.0)
            self._p1.append(0.0)
            self._kstart.append(k0)
            self._kcount.append(n)
        else:
            raise ValueError(
                f"Unsupported coupling function for the Warp FK kernel: "
                f"{type(fn).__name__} (only Constant/Linear/SimmSpline are used "
                f"by the target models)."
            )
        return fid

    def build(self) -> FunctionTable:
        # Ensure knot arrays are never zero-length (Warp needs a valid array).
        x = self._x or [0.0]
        return FunctionTable(
            fn_type=np.array(self._type, dtype=np.int32),
            fn_p0=np.array(self._p0, dtype=np.float64),
            fn_p1=np.array(self._p1, dtype=np.float64),
            fn_kstart=np.array(self._kstart, dtype=np.int32),
            fn_kcount=np.array(self._kcount, dtype=np.int32),
            fn_x=np.array(x, dtype=np.float64),
            fn_y=np.array(self._y or [0.0], dtype=np.float64),
            fn_b=np.array(self._b or [0.0], dtype=np.float64),
            fn_c=np.array(self._c or [0.0], dtype=np.float64),
            fn_d=np.array(self._d or [0.0], dtype=np.float64),
        )


# ---------------------------------------------------------------------------
# Warp device evaluation
# ---------------------------------------------------------------------------


@wp.func
def simm_interval(
    fn_x: wp.array(dtype=wp.float64), k0: wp.int32, n: wp.int32, ax: wp.float64
) -> wp.int32:
    if n < 3:
        return 0
    if ax <= fn_x[k0]:
        return 0
    if ax >= fn_x[k0 + n - 1]:
        return n - 1
    i = wp.int32(0)
    j = n
    k = wp.int32(0)
    # bounded binary search (n is small; cap iterations for safety)
    for _ in range(64):
        k = (i + j) / 2
        if ax < fn_x[k0 + k]:
            j = k
        elif ax > fn_x[k0 + k + 1]:
            i = k
        else:
            return k
    return k


@wp.func
def eval_function(
    fid: wp.int32,
    x: wp.float64,
    fn_type: wp.array(dtype=wp.int32),
    fn_p0: wp.array(dtype=wp.float64),
    fn_p1: wp.array(dtype=wp.float64),
    fn_kstart: wp.array(dtype=wp.int32),
    fn_kcount: wp.array(dtype=wp.int32),
    fn_x: wp.array(dtype=wp.float64),
    fn_y: wp.array(dtype=wp.float64),
    fn_b: wp.array(dtype=wp.float64),
    fn_c: wp.array(dtype=wp.float64),
    fn_d: wp.array(dtype=wp.float64),
) -> wp.float64:
    t = fn_type[fid]
    if t == 0:  # constant
        return fn_p0[fid]
    if t == 1:  # linear
        return fn_p0[fid] * x + fn_p1[fid]
    # SimmSpline
    k0 = fn_kstart[fid]
    n = fn_kcount[fid]
    k = simm_interval(fn_x, k0, n, x)
    dx = x - fn_x[k0 + k]
    b = fn_b[k0 + k]
    c = fn_c[k0 + k]
    d = fn_d[k0 + k]
    return fn_y[k0 + k] + dx * (b + dx * (c + dx * d))
