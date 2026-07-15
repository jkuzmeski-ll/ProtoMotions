# 21 — Nimble source → port map (status tracker)

Ground-truth C++ is vendored (read-only, gitignored) at
`projects/biomech/reference/nimble/`. This table maps each source file to its target
module in `biomech/` and a port status. Update the **Status** column as work lands.

Status legend: `TODO` · `WIP` · `DONE` · `REUSE` (Newton/MuJoCo covers it) ·
`SKIP` (out of scope).

## Math core — `dart/math/`

| Nimble file | Purpose | Target | Status |
|---|---|---|---|
| `SimmSpline.{hpp,cpp}` | OpenSim cubic spline for coupled DOFs | `skeleton/simmspline.py` | **DONE** (8 tests) |
| `CustomFunction.{hpp,cpp}` | fn base (value/deriv) | `skeleton/functions.py` | **DONE** (M2b; Constant/Linear/SimmSpline flattened for Warp) |
| `LinearFunction`, `PolynomialFunction`, `ConstantFunction`, `PiecewiseLinearFunction` | other coupling fns | `skeleton/functions.py` | **DONE** (Constant/Linear used by model; poly/piecewise in spec only) |
| `Geometry.{hpp,cpp}` | SO3/SE3 exp/log, Euler, adjoints | `skeleton/spatial.py` (Warp) | **DONE** (M2b; 4 Euler orders + Rodrigues + SE3, NumPy+Warp) |
| `MathTypes.hpp` | `s_t`, Eigen typedefs | (float64 Warp convention) | N/A |
| `IKSolver.{hpp,cpp}` | generic damped IK | `newton.ik.*` / `fitting/ik.py` | **DONE** (M2c; batched Warp LM marker IK, `fitting/ik.py`, 6 tests) |
| `AssignmentMatcher`, `PolynomialFitter`, `MultivariateGaussian`, `Random`, `FiniteDifference`, `RelativeFilter`, `GraphFlowDiscretizer` | misc utils | as-needed | SKIP unless needed |

## Dynamics / skeleton — `dart/dynamics/`

| Nimble file | Purpose | Target | Status |
|---|---|---|---|
| `Skeleton.{hpp,cpp}` | tree FK, Jacobians, group scale, mass matrix | `skeleton/skeleton.py` (FK+Jac, Warp) | **DONE** (M2b/d; batched FK + marker pos + group scaling + autodiff marker Jac wrt q + marker-loss grads wrt scales & offsets + `set_marker_offsets`) |
| `Joint.{hpp,cpp}`, `GenericJoint.hpp` | joint base | `skeleton/skeleton.py` | **DONE** (M2b; `T_parent·T_joint·T_child⁻¹` + parent/child scale) |
| `CustomJoint.{hpp,cpp}` | SimmSpline-coupled Euler+trans (gold standard) | `skeleton/skeleton.py` | **DONE** (M2b; 6-slot euler+trans, trans reordered by axis) |
| `EulerFreeJoint.{hpp,cpp}` | 6-DOF OpenSim root | `skeleton/skeleton.py` | **DONE** (M2b; via custom-family path) |
| `EulerJoint.{hpp,cpp}` | Euler w/ axis order + flip | `skeleton/skeleton.py` | **DONE** (M2b; via custom-family path) |
| `RevoluteJoint`, `UniversalJoint`, `WeldJoint`, `TranslationalJoint` | standard joints | `skeleton/skeleton.py` | **DONE** (M2b; Revolute=Z, Universal=X,Y, Weld=I) |
| `BodyNode.{hpp,cpp}` | body frame + inertia | `skeleton/skeleton.py` | **DONE** (FK/scale; inertia deferred to MuJoCo) |
| `Marker.{hpp,cpp}` | marker = body + local offset | `skeleton/skeleton.py` | **DONE** (M2b; `T_body·(scale⊙offset)`) |
| `Inertia.{hpp,cpp}` | spatial inertia | dynamics via MuJoCo | REUSE |
| `SimpleFeatherstone.{hpp,cpp}`, `*Joint` dynamics | forward/inverse dynamics | MuJoCo solver | REUSE |
| `ScapulathoracicJoint`, `ConstantCurve*Joint`, `EllipsoidJoint`, shapes | not in target models | — | SKIP unless needed |

## Biomechanics — `dart/biomechanics/`

| Nimble file | Purpose | Target | Status |
|---|---|---|---|
| `OpenSimParser.{hpp,cpp}` (243 KB) | `.osim` → skeleton + markers + scales | `osim/parser.py` | **DONE** (M2a; parse-half of `readOsim30`, 9 parity tests) |
| `IKInitializer.{hpp,cpp}` (157 KB) | closed-form joint centers/scales/poses | `fitting/ik_initializer.py` + `fitting/closed_form.py` | **WIP** (M2c core DONE: closed-form kernels; MDS joint centers; group scales; pose IK; **prescale** (`estimate_prescale`: isotropic marker-span scale seeds weakly-observed axes/empty groups instead of 1.0) DONE; 10 tests. Deferred: pivot-finding / axis-recenter polishing — need the vendored Nimble source (currently absent on disk)) |
| `MarkerFitter.{hpp,cpp}` (407 KB) | bilevel scale+offset+pose fit | `fitting/marker_fitter.py` | **WIP** (M2d core DONE: bilevel block-coordinate descent — Warp IK poses + closed-form offset LS + Gauss–Newton scales, offset prior; 3 tests. Deferred: anthropometric prior, full per-marker weighting schedule) |
| `MarkerFixer.{hpp,cpp}` | RANSAC gap-fill/denoise markers | `fitting/marker_fixer.py` | **DONE** (rigid-body pairwise-distance outlier rejection + robust velocity-spike gate + short-gap linear fill; NumPy, 6 tests) |
| `MarkerOffsetPrior.{hpp,cpp}` | prior on marker offsets | `fitting/priors.py` | **DONE** (`MarkerOffsetPrior`: quadratic pull to model offsets, anatomical anchoring; consolidated from `marker_fitter.py` inline; 2 tests) |
| `Anthropometrics.{hpp,cpp}` | anthropometric priors | `fitting/priors.py` | **DONE** (`Anthropometrics` scale prior from Plug-in-Gait `.mp` segment lengths → `MarkerFitConfig.scale_prior_target/weights`; `anthropometry.py` core; 2 tests. Note: naive scale-fixation hurts dynamic gait — see benchmark; use as a weak prior) |
| `DynamicsFitter.{hpp,cpp}` (670 KB) | residual reduction + mass/inertia from GRF | `fitting/dynamics_fitter.py` (on MuJoCo) | **WIP** (M2e DONE: `ResidualForceHelper` port on `mj_inverse`+`mj_applyFT`; root residual; GRF/COP→wrench adapter; free-joint kinematics via `mj_differentiatePos`; linear per-segment **mass** ID; **full 10-param-per-body inertial ID** (mass+COM+inertia, exact FD regressor through the MuJoCo principal-axis reparam, relative Tikhonov to the anthropometric prior + physical-cone projection); **GPU-batched** residual+mass ID on `mujoco_warp`; **kinematic RRA** (`rra_kinematics`/`DynamicsFitter.rra`: banded Gauss–Newton on the root translation to null the residual vs fixed measured GRF, track+smoothness reg, line search; 2 tests); 19 tests, validated to ~1e-13 + synthetic recovery. Deferred: cross-world regressor batching (blocked by `mjw.Model` param sharing)) |
| `IKErrorReport.{hpp,cpp}` | marker RMS/max reporting | `fitting/report.py` | **DONE** (`marker_errors` → overall/per-frame/per-marker RMS/max/mean over visible pairs via Warp FK; `worst_markers`/`format`; 4 tests) |
| `MarkerLabeller` (label matching) | measured-label → model-marker map | `fitting/marker_map.py` (real-data bridge) | **DONE** (S001 Plug-in-Gait→Rajagopal map, 44 markers, lab Z-up→OpenSim Y-up; `anatomical_mask`; end-to-end smoke on real S001; 6 tests) |
| `enums.hpp`, `macros.hpp` | small shared defs | inline | TODO |
| `C3DLoader.*`, `C3DForcePlatforms.*`, `ForcePlate.*` | C3D + force plates | `biomech.io` (M1) | DONE (cross-check only) |
| `SubjectOnDisk.*` (`.b3d`) | serialized output format | `export/subject.py` + MotionLib | **DONE** (`FittedSubject` bundle: group scales + marker offsets/names + q(t) + fps + marker RMS + inertial params + MJCF XML + metadata → single compressed `.npz` w/ JSON header; `save_subject`/`load_subject` round-trip; `to_mjcf`/`to_motion`/`from_marker_fit`; 4 tests) |
| — (MJCF for the sim) | fitted skeleton -> Newton MuJoCo model | `export/mjcf.py` | **DONE** (M3; coupled/hinge modes, FK exact vs MuJoCo, Newton `eval_fk` + `SolverMuJoCo` validated; 6 tests) |
| — (gold-standard clip) | fitted `q(t)` -> ProtoMotions `.motion` | `export/motion.py` | **DONE** (M3; Warp-FK global transforms, Y-up→Z-up, `gts/grs/gvs/gavs/dps/dvs`; 7 tests) |
| — (M8 Newton imitation env) | fit -> runnable ProtoMotions Newton mimic setup | `export/protomotions_robot.py` + `protomotions/robot_configs/biomech.py` + `projects/biomech/experiments/mimic_newton.py` | **DONE** (`write_biomech_asset`/`build_simbody_motion` [MuJoCo FK over the full 38-body sim set incl. dummy bodies, aligns 1:1 with `extract_kinematic_info`]/`build_biomech_robot_config`/`export_protomotions_bundle`; `"biomech"` robot registered in `factory.py` [31 DOFs, anchor `torso`, uniform PD]; `tools/export_s001_subject.py` regenerates the committed asset with S001 scales + writes `data/motions/biomech_s001_walk.motion`; 7 bridge tests. **Confirmed runnable on Newton** via `train_agent.py` [sim built, robot+clip loaded, foot contact sensors, CUDA graph, PPO rollouts+save]. Windows fix: gloo DDP backend in `train_agent.py` [NCCL absent on Windows]. Remaining: full train-to-convergence GPU run) |
| — (treadmill→overground / TM2OG) | make the treadmill clip translate over ground for a physics sim | `export/tm2og.py` (wired into `build_simbody_motion` + `tools/export_s001_subject.py`) | **DONE** (port of Jung & Lee, *Sensors* 21(3):786, 2021 virtual-origin method: `x_og = x_tm + ∫v_belt dt` along +forward, plus Galilean `v += v_belt`; belt-speed log is ground truth [no belt markers]; walking axis inferred empirically = **−Y** via `infer_travel_direction` [negate stance-foot horizontal vel]; rotations/DOFs untouched → contact geometry preserved; S001 clip now travels −2.28 m [belt ∫v dt ≈ 2.235 m], stance foot ~stationary; 9 tests) |
| `MarkerLabeller`, `Cortex*`, `Streaming*`, `LilypadSolver`, `BatchGaitInverseDynamics`, `SkeletonConverter`, `LinkBeamSearch`, `Marker*BeamSearch` | labeling/streaming/other | — | SKIP |

## Newton/MuJoCo reuse (not ported)

| Need | Newton/MuJoCo API (pin: newton 1.0.0 / warp 1.14.0 / mujoco 3.5.0) |
|---|---|
| FK for MJCF model | `newton.eval_fk` |
| IK scaffolding (MJCF model) | `newton.ik.IKSolver`, `IKObjectivePosition/Rotation/JointLimit`, `IKOptimizerLM/LBFGS` |
| Jacobians (MJCF model) | `newton.eval_jacobian` |
| Mass matrix / dynamics | `newton.eval_mass_matrix`, `SolverMuJoCo` (`mujoco_warp`, `mj_inverse`) |
| Simulation + contact | `SolverMuJoCo` + `Model/State/Control/Contacts` |
| Model build / MJCF import | `newton.ModelBuilder` |

**Newton gotcha (learned in M3):** `ModelBuilder.add_mjcf` **merges** consecutive
joints on one body into a single compound joint whose `eval_fk` rotation does NOT equal
MuJoCo's sequential-hinge composition, and it drops the sub-joint names that
`<equality>` constraints reference (so `SolverMuJoCo` fails to compile). The MJCF
exporter therefore splits every multi-DOF joint into a chain of **massless dummy bodies**
(one single-DOF joint each). Real MuJoCo FK of the exporter's single-body-multi-joint
form is bit-exact vs the Warp skeleton (1e-15); the dummy-chain form matches under both
Newton `eval_fk` (float32, ~2e-6) and `SolverMuJoCo`.

**M2e dynamics notes (learned):** the residual engine is `mujoco.mj_inverse` (NOT
`mj_rne` alone — `mj_inverse` correctly accounts for the coupled-knee `<equality>`
constraint forces). Verified: ID round-trip (apply `qfrc_inverse` in forward dynamics
recovers `qacc`) is exact (~4e-14). Contact GRF→generalized force uses `mj_applyFT`
(force + free-moment at the COP world point on the foot body); free-joint translational
root rows equal the world force. Root residual = `qfrc_inverse[:6] - Σ Fs[:6]`. The root
residual is **exactly linear** in each `m.body_mass` (mutating `body_mass` is reflected
immediately by `mj_inverse`, no recompile; `∂²r/∂m²` ~1e-13), so an FD mass regressor is
exact — this is Nimble's linear inertial identification directly on the MuJoCo model.
Free-joint velocities/accelerations must come from `mj_differentiatePos` (quaternion on
the manifold), not component-wise diff of `qpos`. Watch synthetic trajectories: few
frames at high fps ⇒ huge finite-diff accelerations ⇒ ill-posed mass ID; use quasi-static
or slow motion for sanity tests.

## Contact models — novel (not in Nimble)

Nimble uses point/sphere contact only; the distributed-surface contact work is the
project's own research payload ("between FEA and point contact"), built directly on
Warp / the Newton MuJoCo solver rather than ported.

| Rung | Purpose | Target | Status |
|---|---|---|---|
| M5 rung 1 | distributed elastic-foundation (Winkler) foot contact under prescribed gold-standard kinematics (Warp kernel + NumPy reference); `FootSole` + flat/ellipsoid soles; `reduce_wrench`→net GRF/COP/free-moment matching `ForcePlate` | `contact/elastic_foundation.py`, `contact/kinematics.py` | **DONE** (12 tests; Warp/NumPy parity ~1e-3; gold-standard `q(t)` bridge from `MotionExportResult`) |
| M6 rung 2 | calibrate contact params vs measured GRF (COP = diagnostic) via log-space Levenberg–Marquardt over the Warp-batched forward | `contact/calibration.py` | **DONE** (8 tests; `calibrate_elastic_foundation` + generic-core `calibrate_hydroelastic` with selectable free params; self-consistent recovery CPU + GPU; note: `k`/`hc_alpha` need a varying penetration rate to separate) |
| M7 rung 3 | hydroelastic / pressure-field foot contact (Warp kernel + NumPy reference): spatially-varying compliance (`FootSole.modulus`), hyperelastic stiffening, energetically-consistent Hunt–Crossley dissipation, Stribeck friction; reduces exactly to the Winkler bed in the linear limit | `contact/hydroelastic.py` | **DONE (law)** (7 tests; Winkler-limit parity, Warp/NumPy ~2e-3; still uses analytic soles until M4) |
| M4 | subject plantar geometry from static C3D to replace analytic soles; feeds `FootSole` points/areas/modulus | `contact/foot_geometry.py` + reuse `data/scripts/calibrate_lower_body_elipsoid_from_static_c3d.py` | **DONE** (tapered subject sole in the `calcn` frame: static-C3D dimensions + scaled marker anchors + heel-pad/arch compliance map; validated on real S001 static; 6 tests) |
| pipeline | stitch the whole chain on a real capture: load → observations → `IKInitializer` seed → `MarkerFitter` (full) → `build_motion` → subject sole → robust per-stance ground registration → measured split-belt GRF → `calibrate_hydroelastic` | `contact/pipeline.py` | **DONE** (3 tests incl. real S001: full fit ~1.4 cm marker RMS; vertical GRF calibrated to ~1% on real belt data; `mu` excluded by design — unobservable in planted stance) |
| robust dynamic calibration | stance segmentation + flat-foot ground registration (robust to the foot rolling) + aggregate (per-stance sub-bin-mean) calibration objective for walk/run windows | `contact/stance.py` + calibration `objective="aggregate"` + pipeline `registration="flatfoot"` | **DONE** (8 tests; synthetic: aggregate ~0.5% k-error vs per-frame ~4% under 0.8 mm jitter; real S001 walk recovers physical stiffness vs per-frame degeneration; residual now attributed to reconstruction quality) |
| contact-in-sim | run the distributed contact model **inside** MuJoCo forward dynamics (the engine Newton `SolverMuJoCo` integrates): each `mj_step` applies the Warp-computed contact wrench as `xfrc_applied` from the body's current world state | `contact/forward_sim.py` | **DONE** (5 tests; rigid-body drop settles to the analytic elastic-foundation equilibrium GRF==weight / z==-weight/(k*A) on CPU + Warp; hydroelastic linear-limit matches; bounded sliding friction. Next: full-skeleton MJCF + GPU `mujoco_warp` stepping + forward/RRA tracking) |
