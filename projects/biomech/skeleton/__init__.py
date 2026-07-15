"""biomech.skeleton — Warp-native differentiable OpenSim skeleton kinematics.

Milestone M2b. A small, exact, differentiable articulated-kinematics core in Warp that
reproduces the *gold-standard* OpenSim joint behavior the MuJoCo solver cannot express:

- ``simmspline.py``  : OpenSim SIMM cubic spline (value + derivatives).
                       (Adapted from OpenSim, Apache-2.0; keep attribution.)
- ``functions.py``   : coupling-function set (SimmSpline / linear / polynomial / const).
- ``spatial.py``     : SO3/SE3 exp/log, Euler orders, adjoints (Warp device fns).
- ``joints.py``      : CustomJoint (SimmSpline-coupled), EulerFreeJoint, EulerJoint,
                       Revolute/Universal/Weld/Translational — only the types the
                       target model uses.
- ``skeleton.py``    : tree FK, group (anisotropic segment) scaling, marker sites,
                       batched-over-frames FK + marker Jacobians (Warp autodiff first,
                       analytic later where profiling demands).

Scope: implement ONLY the joint/function types the target ``.osim`` uses (from M2a).
No contact and no inverse dynamics live here — those go through the MuJoCo solver.

Reference C++: ``reference/nimble/dart/dynamics/{Skeleton,CustomJoint,EulerFreeJoint,
EulerJoint,...}.{hpp,cpp}`` and ``dart/math/{SimmSpline,Geometry}.{hpp,cpp}``.

Note: import ``WarpSkeleton``/``fk_numpy`` from ``biomech.skeleton.skeleton`` (not
this package ``__init__``) — ``biomech.osim.spec`` imports ``skeleton.simmspline``,
so importing the Warp FK here would create a circular import.
"""
