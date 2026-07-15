# 20 — Nimble → Newton/Warp port plan (Windows-native)

This is the plan for porting Nimble's biomechanics math/methods to a **Windows-native,
Warp-accelerated** implementation, using the **Newton MuJoCo solver** for dynamics and
contact. Ground truth for the algorithms is the vendored source at
`projects/biomech/reference/nimble/` (read-only). The file-by-file map with status is
`21_nimble_source_map.md`.

## What Nimble actually does (so we port the right things)

Nimble's "AddBiomechanics" pipeline fits a **biomechanical OpenSim skeleton** to marker
+ force-plate data, in these stages:

1. **OpenSim model → skeleton** (`OpenSimParser`): parse a `.osim` into a DART
   `Skeleton` — bodies, joints, marker offsets, and the **group scale** vector
   (anisotropic per-segment scaling). The gold-standard fidelity lives here: OpenSim
   `CustomJoint`s whose DOFs drive an Euler+translation joint through **SimmSpline**
   coupling functions (e.g. the knee's translation is a spline of flexion angle).

2. **Marker de-noising** (`MarkerFixer`): RANSAC-style gap fill / outlier rejection and
   relabeling on raw marker traces.

3. **Closed-form initialization** (`IKInitializer`): estimate joint centers in world
   space per frame **in closed form** (no gradient descent), then back out body scales
   and per-frame poses. Methods: MDS triangulation, Chang–Pollard 2006 center-of-
   rotation, Gamage–Lasenby 2002 axis fit, least-squares concentric sphere fit. This is
   the robust seed for the nonlinear fit.

4. **Bilevel marker fit** (`MarkerFitter`): nonlinear optimization over
   **{group scales, marker offsets, per-frame poses}** minimizing marker reprojection
   error + anatomical/marker-offset priors + joint-limit terms, using **analytic
   Jacobians**. Outputs the scaled skeleton, marker offsets, and `q(t)`.

5. **Dynamics fit** (`DynamicsFitter`): using measured GRF/COP, refine mass / COM /
   inertia and lightly adjust kinematics so the trajectory is **dynamically consistent**
   (root residual "hand of god" forces driven toward zero), via the equations of motion
   `M(q) q̈ + C(q,q̇) + G(q) = τ + Σ Jᶜᵀ F_grf`. Analogous to OpenSim RRA + mass tuning.

6. **Output** (`SubjectOnDisk`, `.b3d`): serialized trials with `q, q̇, q̈`, GRF/COP,
   marker data, and per-frame quality metrics.

The parts that are **irreducible to port** (Newton/MuJoCo has no equivalent):
- SimmSpline `CustomJoint` FK + Jacobians (gold-standard joint kinematics).
- Group (anisotropic segment) scaling with markers attached to segments.
- The IKInitializer closed-form joint-center math.
- The bilevel MarkerFitter objective/optimizer.

The parts we **reuse** instead of porting:
- Articulated dynamics (mass matrix, bias forces, forward/inverse dynamics) → **MuJoCo
  solver** (`mujoco_warp`), not a hand-ported Featherstone/RNEA.
- Batched IK/FK/Jacobian scaffolding for the MJCF model → `newton.eval_fk`,
  `newton.eval_jacobian`, `newton.eval_mass_matrix`, `newton.ik.*` (only for the
  MuJoCo-side standard-joint model, not the SimmSpline fit).
- C3D + force plates → **already done** in `biomech.io` (M1); we do not port
  `C3DLoader`/`C3DForcePlatforms`, we only cross-check conventions.

## Target architecture

Two engines, deliberately separated:

```
biomech/
  io/         (DONE, M1) C3D, force plates, treadmill, session
  osim/       (M2a) .osim parser -> SkeletonSpec (bodies, joints, markers, scales)
  skeleton/   (M2b) Warp-native differentiable OpenSim skeleton:
              joints (Euler/EulerFree/Revolute/Universal/Weld/Custom+SimmSpline),
              group scaling, marker sites; batched FK + marker Jacobians
  fitting/    (M2c-e) IKInitializer, MarkerFitter, DynamicsFitter ports
  export/     (M3) SkeletonSpec + q(t) -> MJCF + ProtoMotions MotionLib
  contact/    (M5+) custom elastic-foundation / hydroelastic contact for MuJoCo solver
  reference/  (gitignored) vendored Nimble C++ (read-only)
```

- **Fit engine (`osim` + `skeleton` + `fitting`)**: pure kinematics, full OpenSim
  fidelity, Warp-accelerated, no contact. Produces gold-standard `q(t)`, scales,
  marker offsets, marker RMS.
- **Physics engine (`export` + Newton `SolverMuJoCo` + `contact`)**: dynamics fit,
  tracking rollouts, and contact research on the exported MJCF + fitted motion.

The gold-standard reference `q(t)` is produced by the fit engine and **never** routed
through the 18-keypoint/PyRoki path.

## Why our own Warp skeleton (not just Newton's model) for the fit

Newton/MuJoCo joint types are {free, ball, hinge/revolute, slide/prismatic, D6}. OpenSim
gold-standard models need **SimmSpline-coupled** DOFs and **anisotropic group scaling of
segments with attached markers**. Those cannot be expressed in MuJoCo FK without
approximation. Since the kinematic fit needs only FK + marker Jacobians (no contact, no
inverse dynamics), we implement a small, exact, differentiable kinematics core in Warp.
This is the same design decision Nimble made (it forked DART to add `CustomJoint`).

Scope control: we implement **only** the joint types the Rajagopal-2015 / gait models
actually use (confirm from the `.osim` during M2a), not all of DART.

## The math to port (grounded in the vendored source)

### SimmSpline (`dart/math/SimmSpline.{hpp,cpp}`)
Natural cubic spline with SIMM's specific end conditions. Store knots `x[]`, values
`y[]`, and per-interval coefficients `b,c,d` from `calcCoefficients()`. Need
`calcValue(x)` and `calcDerivative(order, x)` (orders 1–2 for velocity/accel and
Jacobians). Port to: a Warp device function evaluating the piecewise cubic + a NumPy
builder that computes `b,c,d`. Keep OpenSim/Apache attribution.

### CustomJoint (`dart/dynamics/CustomJoint.{hpp,cpp}`)
Wraps a 6-DOF Euler(3)+translation(3) joint. Each of the 6 outputs is a
`CustomFunction` (usually SimmSpline or linear) **driven by one input DOF**
(`mFunctionDrivenByDof`), with an Euler `AxisOrder` and a `FlipAxisMap` (±1 per axis).
Port the key methods:
- `getCustomFunctionPositions/Velocities/Accelerations(x, dx, ddx)` → the 6 Euler+trans
  values and their time derivatives.
- `updateRelativeTransform()` → parent→child transform from the 6 values.
- `getRelativeJacobianStatic(pos)` and `getRelativeJacobianDerivWrtPositionStatic` →
  the joint Jacobian mapping input-DOF velocities to spatial velocity, and its position
  derivative. (Time-deriv and 2nd-deriv variants only if DynamicsFitter needs them; for
  the kinematic fit, position + first Jacobian suffice.)
- `zeroTranslationInCustomFunctions()` — moves constant offsets baked into the custom
  functions into the parent transform (needed to import cleanly).

### The rest of the joint set (subset of `dart/dynamics/*Joint.*`)
`EulerFreeJoint` (6-DOF root used by OpenSim), `EulerJoint` (axis order + flip),
`RevoluteJoint`, `UniversalJoint`, `WeldJoint`, `TranslationalJoint`. Only implement
`updateRelativeTransform` + `getRelativeJacobianStatic` for each. Confirm the actual set
from the model in M2a before writing them.

### Skeleton FK + marker Jacobian (`dart/dynamics/Skeleton.cpp`, subset)
Compose per-joint relative transforms along the tree to get world body transforms; a
marker is a body + local offset (scaled). Marker world position = `T_body * (scale ⊙
offset)`. The marker Jacobian w.r.t. `q` chains the joint Jacobians; w.r.t. group scales
and w.r.t. marker offsets are additional analytic blocks. **Batch over frames** in a
Warp kernel (one thread per (frame, marker) or per frame). Use Warp autodiff for
Jacobians first (fast to write, correct), then swap in analytic Jacobians where profiling
demands it.

### IKInitializer (`dart/biomechanics/IKInitializer.{hpp,cpp}`)
Port the closed-form pipeline (`runFullPipeline`):
`prescaleBasedOnAnatomicalMarkers` → `closedFormMDSJointCenterSolver` →
`closedFormPivotFindingJointCenterSolver` → `recenterAxisJointsBasedOnBoneAngles` →
`estimateGroupScalesClosedForm` → `estimatePosesClosedForm`. The reusable numerical
kernels (all static, all portable, all good Warp/NumPy targets):
- `getPointCloudFromDistanceMatrix` (classical MDS),
- `leastSquaresConcentricSphereFit`,
- `getChangPollard2006JointCenterMultiMarker`,
- `gamageLasenby2002AxisFit`,
- `getLocalScale`, `centerPointOnAxis`, `findCubicRealRoots`.
Also port the **StackedBody/StackedJoint** simplification (merge stacked low-DOF joints
and collapse welds) — it defines the topology the closed-form solvers run on.

### MarkerFitter (`dart/biomechanics/MarkerFitter.{hpp,cpp}`)
The bilevel fit. Nimble uses IPOPT; we replace the optimizer (no IPOPT on Windows) with
either (a) a Warp-batched **CMA-ES / population** outer loop over {scales, marker
offsets} with per-frame IK inner loops, or (b) Gauss–Newton / Levenberg–Marquardt on the
stacked residual using our Warp Jacobians. Port the **objective and priors**, not the
solver: marker reprojection error, marker-offset prior (`MarkerOffsetPrior`), anatomical
marker regularization (`Anthropometrics`), joint-limit penalties, and the marker
weighting. Reuse `newton.ik` LM/L-BFGS optimizers where they fit; otherwise a small
Warp LM.

### DynamicsFitter (`dart/biomechanics/DynamicsFitter.{hpp,cpp}`)
Port the **problem formulation** (residual reduction + linear mass/COM/inertia solve +
GRF consistency), but compute the dynamics terms with the **MuJoCo solver**
(`mj_inverse` via `mujoco_warp`) on the exported MJCF, rather than porting DART's
Featherstone. Map measured GRF/COP (already in `biomech.io`) to MuJoCo external forces at
the foot contact frames. This is the largest single `.cpp` (670 KB) — port incrementally
and only the pieces the research needs (feet-focused, lower-body).

## Milestones (execution order)

Numbering continues from M1 (io/session, DONE).

- **M1.5 — finish ingestion loose ends.** Integrate `Speedchange*` (protocol events +
  variable-rate belt time-base anchoring) incl. per-belt `Speedchangeleft/right101`.
  Small, already scoped; do before/along M2. (See `21_...` and M1 status.)
- **M2a — `osim` parser.** Parse the target `.osim` into a `SkeletonSpec`
  (bodies/joints/markers/group-scale groups). First: **inventory which joint/function
  types the model uses** to bound the skeleton port. Validate against Nimble's parse of
  the same file (transforms, marker offsets).
- **M2b — `skeleton` Warp kinematics.** SimmSpline + the needed joints + group scaling
  + marker sites; batched FK + marker positions. Validate FK against Nimble at random
  `q`/scales (mm-level). Add Jacobian (autodiff first).
- **M2c — `IKInitializer` port.** Closed-form joint centers + scales + poses. Validate
  joint-center estimates and marker RMS vs. Nimble on S001.
- **M2d — `MarkerFitter` port.** Bilevel fit (Warp CMA-ES/LM). Target marker RMS
  parity with Nimble; report per-marker error + offsets.
- **M2e — `DynamicsFitter` port on MuJoCo.** Residual reduction + mass/inertia from GRF
  using MuJoCo inverse dynamics. Report residuals + GRF match.
- **M3 — export.** `SkeletonSpec` + `q(t)` → MJCF (MuJoCo-solver ready) + ProtoMotions
  MotionLib (`gts/grs/gvs/gavs/dps/dvs`). Coupled knee approximated in MJCF.
- **M4 — subject foot geometry.** Plantar SDF from static C3D landmarks (reuse
  `data/scripts/calibrate_lower_body_elipsoid_from_static_c3d.py`).
- **M5 — contact rung 1.** Elastic-foundation Warp contact under prescribed kinematics
  in the MuJoCo solver (first research result), then M6 batched calibration, M7
  hydroelastic, M8 Newton imitation env.

## Validation strategy (parity, not vibes)

Because Nimble runs on Linux, keep a **golden-values workflow**: run Nimble once in a
throwaway Linux env (WSL/Docker/Colab) on S001 to dump reference outputs
(joint centers, group scales, `q(t)`, marker RMS, residuals) to JSON, commit those
goldens under `docs/refs/`, then unit-test each ported stage against them on Windows.
This is a one-time reference generation, not a runtime dependency — the pipeline itself
stays Windows-native.

## Risks / notes

- SimmSpline `CustomJoint` Jacobians are the trickiest math; start with Warp autodiff and
  finite-difference checks before any analytic version.
- MuJoCo cannot represent SimmSpline coupling exactly; the **fit** keeps full fidelity,
  the **MJCF export** approximates — quantify the knee-coupling error at export time.
- DART uses `s_t` (double) throughout; the fit needs float64 for parity. Warp supports
  float64 kernels — use it for the fit; the sim side can be float32.
- Don't port dead weight: `Cortex*`, `Streaming*`, `MarkerLabeller`, `LilypadSolver`,
  `BatchGaitInverseDynamics`, `SkeletonConverter` are out of scope unless a need appears.
