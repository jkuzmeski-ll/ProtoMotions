"""biomech.osim — OpenSim ``.osim`` model parser (port of Nimble ``OpenSimParser``).

Milestone M2a. Parses an OpenSim ``.osim`` XML into a backend-agnostic
``SkeletonSpec`` (bodies, joints, marker offsets, group-scale groups) that the
``biomech.skeleton`` Warp kinematics consume.

First deliverable: an *inventory* of which joint types and coupling-function types
(SimmSpline / linear / polynomial / constant) the target model actually uses, to bound
the ``biomech.skeleton`` port. See ``docs/20_nimble_port_plan.md`` (M2a) and
``docs/21_nimble_source_map.md``.

Reference C++: ``reference/nimble/dart/biomechanics/OpenSimParser.{hpp,cpp}``.
"""

from biomech.osim.parser import euler_xyz_to_matrix, parse_osim
from biomech.osim.spec import (
    BodySpec,
    ConstantFunctionSpec,
    CoordinateSpec,
    JointSpec,
    LinearFunctionSpec,
    MarkerSpec,
    MultiplierFunctionSpec,
    PiecewiseLinearFunctionSpec,
    PolynomialFunctionSpec,
    SimmSplineSpec,
    SkeletonSpec,
    TransformAxisSpec,
)

__all__ = [
    "parse_osim",
    "euler_xyz_to_matrix",
    "SkeletonSpec",
    "BodySpec",
    "JointSpec",
    "CoordinateSpec",
    "MarkerSpec",
    "TransformAxisSpec",
    "LinearFunctionSpec",
    "ConstantFunctionSpec",
    "SimmSplineSpec",
    "PolynomialFunctionSpec",
    "PiecewiseLinearFunctionSpec",
    "MultiplierFunctionSpec",
]
