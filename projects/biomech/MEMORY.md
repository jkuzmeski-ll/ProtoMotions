# biomech — memory / home base

This folder is the **single home base** for the gold-standard biomechanics → Newton
contact-model project. Everything (code + memory + reference source) lives here, on
purpose, separate from the agent skills. **Start here** at the top of any session.

## Read order

1. `docs/00_overview.md` — goal, locked decisions, verified local environment,
   **solver decision (MuJoCo)**, licensing, guardrails.
2. `docs/20_nimble_port_plan.md` — the Windows-native Nimble→Newton/Warp **port plan**.
3. `docs/21_nimble_source_map.md` — Nimble file → our module map + **live status**.
4. `docs/01_milestone1_status.md` — Milestone 1 (ingestion) status.
5. `docs/10_kinodynamic_retargeting_newton.md`,
   `docs/11_foot_contact_modeling_newton.md` — long-form design background.
6. `docs/22_ik_process.md` — the **IK / marker-fitting** process end to end
   (engine → per-frame IK → closed-form seed → bilevel fit) with source refs.
7. `README.md` — M1 usage (C3D/session loader).

## One-paragraph state

**M1 (local C3D + treadmill + force-plate → `CaptureSession`) is DONE** (`io/`,
`session.py`, `frames.py`, 17 tests pass on S001). The project has **pivoted**: instead
of running Nimble on Linux (no Windows wheels), we are **porting Nimble's biomechanics
math to Windows on Warp**, using the **Newton MuJoCo solver** for dynamics/contact.
Nimble C++ is vendored read-only at `reference/nimble/`.

**Golden references generated** (one-time real-Nimble run in WSL2; see
`tools/nimble_golden/`): parser/FK/scaling/CustomJoint goldens for **Rajagopal2015** are
in `docs/refs/`. The M2a joint-type inventory is done (5 joint types; coupled knees are
the only CustomJoints). **Port progress:** `skeleton/simmspline.py` is ported and tested,
**M2a `osim/parser.py` is DONE**, **M2b `skeleton/` FK is DONE** — a batched,
differentiable **Warp** FK (`skeleton/spatial.py`, `functions.py`, `skeleton.py`) that
reproduces Nimble's tree FK to ~machine precision on every FK/CustomJoint-sweep/scaling
golden (body transforms + markers), on **CPU and CUDA**, with a NumPy reference
(`fk_numpy`) and an **autodiff marker Jacobian** (`WarpSkeleton.marker_jacobian_wrt_q`,
validated vs finite differences). **M2c core is DONE:** `fitting/closed_form.py`
**M2c core is DONE:** `fitting/closed_form.py`
(MDS / Chang–Pollard / Gamage–Lasenby / sphere-fit / Kabsch / cubic-roots / coplanar
helpers), `fitting/ik.py` (batched Warp-driven **Levenberg–Marquardt marker IK**, port
of Nimble `fitMarkersToWorldPositions`/`solveIK`; validated to machine precision by
round-trip), and `fitting/ik_initializer.py` (closed-form **MDS joint centers → group
scales → pose IK**, port of `IKInitializer::runFullPipeline` minus the deferred
prescale/pivot/axis-recenter polishing; validated by synthetic round-trip).
**M2d core is DONE:** `fitting/marker_fitter.py` (bilevel **block-coordinate descent** —
Warp IK poses + **closed-form per-marker offset LS** + **Gauss–Newton scales** with an
offset prior + weak scale→neutral Tikhonov stand-in for the deferred anthropometric
prior; validated by synthetic round-trip: sub-mm/low-mm marker RMS, offsets anchored,
offset-fit beats scales-only). The skeleton now also exposes `marker_loss_grads`
(Warp autodiff of the weighted marker loss wrt group scales **and** marker offsets,
validated vs FD) and `set_marker_offsets`.
**M3 export is DONE** (`export/mjcf.py` + `export/motion.py`): fitted `SkeletonSpec` (+
group scales) → **MJCF** for the Newton MuJoCo solver, and fitted `q(t)` → ProtoMotions
**`.motion`** clip. The MJCF FK is **bit-exact vs the Warp/DART skeleton under real
MuJoCo** (`coupled` mode 1e-15; `hinge` mode reports the bounded ~3 cm / 0.26 rad
dropped-knee-coupling error), matches Newton `eval_fk` (float32) and compiles under
`SolverMuJoCo`. The SimmSpline walker knee is preserved via coupled DOFs + `<equality>`
quartic `polycoef` (fit residual <1e-4). MotionLib export uses the gold-standard Warp FK
(Y-up→Z-up `R_OS2PM`), computes `gts/grs/gvs/gavs/dps/dvs`, and loads as a `RobotState`.
**M2e DynamicsFitter core is DONE** (`fitting/dynamics_fitter.py`): a Windows-native port
of Nimble's `ResidualForceHelper` on the **Newton MuJoCo solver** — root residual
`qfrc_inverse[:6] - Σ Jᶜᵀ F` via `mj_inverse` + `mj_applyFT` on the M3 MJCF (validated to
~1e-13 vs the inverse-dynamics round-trip), a GRF/COP→contact-wrench adapter (split-belt:
right belt=right foot), free-joint-aware kinematics (`mj_differentiatePos`), and a
**linear per-segment mass identification** (exact FD regressor + Tikhonov + backtracking
line-search for positivity; validated by synthetic mass-recovery round-trip). The **full
10-param-per-body inertial identification** (mass + COM + full inertia) is also DONE
(`identify_inertial_params`): the root residual is exactly linear in the 10 inertial
params per body (`[m, m·c, I_origin]`, validated ~1e-13), so an FD regressor through
MuJoCo's principal-axis reparam is exact; solved by relative Tikhonov toward the
anthropometric prior + a **physical-cone projection** line search that keeps every body
a valid rigid body while driving the residual down (synthetic recovery: force residual
→ 0, torque strongly reduced, all bodies physical). A **GPU-batched** path
(`BatchedResidualHelper` / `identify_masses_batched`) runs the whole residual over frames
in one `mujoco_warp.inverse` launch (contacts via `xfrc_accumulate`), matching the CPU
engine to ~1e-4 (Newton MuJoCo GPU).
**94 tests pass total.** Next: cross-world regressor batching, or the real-data bridge
(S001 marker map) to run the fit end-to-end on real capture.
**M5 (contact rung 1) + M6 (contact rung 2) DONE** (`contact/`): a distributed
**elastic-foundation (Winkler) foot contact as a Warp kernel** under prescribed
gold-standard kinematics (`contact/elastic_foundation.py`: `FootSole` + flat/ellipsoid
samplers, per-patch Winkler `k*d` + damping + regularized Coulomb friction, NumPy
float64 reference + Warp float32 kernel validated to ~1e-3 parity, `reduce_wrench` →
`ContactPrediction` with net GRF/COP/free-moment matching the `ForcePlate` fields); a
bridge that slices any foot body's world pose+spatial-velocity out of a
`MotionExportResult` and drives the contact model (`contact/kinematics.py`); and a
**log-space Levenberg–Marquardt calibration** of `(k_bed, c_bed, mu)` against measured
GRF (COP is a kinematics/geometry diagnostic, not a fit target) with a 3-column FD
Jacobian over the Warp-batched forward (`contact/calibration.py`, validated by
self-consistent synthetic recovery on CPU and GPU). **113 tests pass total.**
**M7 (contact rung 3) DONE** (`contact/hydroelastic.py`): a **hydroelastic
(pressure-field) foot contact** Warp kernel generalizing the Winkler bed —
spatially-varying plantar compliance (`FootSole.modulus`, e.g. soft heel pad vs stiff
forefoot), hyperelastic stiffening (`p = k·d·(1+b·d)`), **energetically-consistent
Hunt–Crossley dissipation** (`p ← p·(1+α·ṅ)`, non-adhesive at lift-off), and
pressure-dependent **Stribeck friction**; it reduces **exactly** to
`elastic_foundation` in the linear limit (verified) and the Warp kernel matches the
NumPy reference to ~2e-3. The law runs on the same analytic soles today and will drop
in subject plantar geometry once **M4** is built. **120 tests pass total.**
**Real-data bridge DONE** (`fitting/marker_map.py`): the S001 Plug-in-Gait →
Rajagopal2015 marker map (44 model markers mapped incl. all lower-body landmarks;
virtual joint centres / unsupported markers left NaN), `build_observations`
(lab Z-up → OpenSim Y-up, model-marker order) + `anatomical_mask` +
`observations_from_session` + `mapping_coverage`. An **end-to-end smoke test** now runs
on the real S001 capture: `load_session` → observations → `IKInitializer.run` →
`build_motion` → `evaluate_foot_contact_from_motion` (finite scales/poses, marker RMS
< 10 cm median on raw PiG markers, foot makes contact). **126 tests pass total.**
**M4 (subject foot geometry) DONE** (`contact/foot_geometry.py`): builds a subject-specific
tapered plantar **`FootSole` in the `calcn` body frame** from a static C3D — subject foot
*dimensions* (heel/forefoot width, length, ball offset, toe length; port of the
`foot_calibration` in `data/scripts/calibrate_lower_body_elipsoid_from_static_c3d.py`)
combined with anatomical *anchors* (scaled `RCAL`/`RMT5`/`RTOE` offsets, all on `calcn`)
for fit-consistent placement/scale. The sole carries a per-patch compliance `modulus`
map (soft heel fat-pad, relieved medial arch, stiffer forefoot) and drops straight into
`hydroelastic.evaluate_contact`. Validated on the **real S001 static capture** (both
feet: ~0.285 m length, forefoot wider than heel, symmetric). **132 tests pass total.**
**M6 extended to M7:** `contact.calibration.calibrate_hydroelastic` calibrates a
selectable subset of the hydroelastic pressure-field params against measured GRF using a
generic log-space LM core (note: `k_bed` and `hc_alpha` only separate when the
penetration rate varies across frames — constant `vn` makes them collinear).
**133 tests pass total.**
**Real-data pipeline DONE** (`contact/pipeline.py`): `run_subject_pipeline` stitches the
whole chain on a real capture — `load_session` → Plug-in-Gait→Rajagopal observations →
`IKInitializer` seed → **MarkerFitter** (full bilevel fit; drops marker RMS to ~1.4 cm on
S001, vs ~10 cm initializer-only) → `build_motion` → subject `FootSole` (M4) → **robust
per-stance ground registration** (80th-percentile of per-frame min sole z + penetration;
global-min was outlier-driven) → measured split-belt GRF (right belt = right foot) →
`calibrate_hydroelastic`. On real S001 the **vertical GRF fits to ~1%** (R k≈8.0e7 N/m³,
α≈2.4; L k≈6.7e7, α≈1.7; vert_rms≈4 N vs ~400 N mean) — achieved only when fitting Fz
alone (`horizontal_weight=0`, `free_params=("k_bed","hc_alpha")`). Friction `mu` is
friction `mu` is
excluded by design: planted stance has ~0 sliding velocity so shear is physically
unobservable (needs sliding phases → later milestone). Trajectory smoothing was tried and
HURT (added artifacts; predicted Fz std is already ~4 N — don't smooth). **136 tests pass
total.**
**M1.5 (treadmill protocol) DONE** (`io/treadmill.py` + `session.py`): `read_speedchange`
parses `Speedchange<trial>.txt` into a `TreadmillProtocol` (named boundaries START /
WALK_START / WALK_END / RUN_START / RUN_END / END = 1-based item rows 1,3,4,5,6,7; times in
seconds on the trial timeline, verified: item 11 ≈ C3D duration). `load_session` takes
`speedchange_path` and exposes `session.phase_window("walk"|"run"|"all")` → point-frame
`(lo, hi)`; the pipeline + CLI take a `phase=` arg to fit a walk/run window instead of
quiet stance. **NOTE:** per-frame contact calibration during *dynamic* walking is poor
(vertical_rms ~440 N) — stance/swing transitions + a single ground plane over the window +
cm-level marker noise dominate; this is the known limitation motivating aggregate
objectives / smoothed kinematics + mu-from-sliding (future work). A CLI
(`run_pipeline.py`) runs the whole thing on S001 and dumps a JSON report. **140 tests pass
total.**
**Robust dynamic-window calibration DONE** (`contact/stance.py` + calibration
`objective="aggregate"` + pipeline `registration="flatfoot"`): stance segmentation,
flat-foot detection + flat-foot ground registration (robust to the foot rolling/lifting),
and a per-stance sub-bin-mean aggregate objective that averages out unbiased per-frame
kinematic noise (synthetic: aggregate ~0.5% k-error vs per-frame ~4% under 0.8 mm jitter).
On real S001 **walk** this recovers *physical* plantar stiffness (R k≈7.8e6, L≈6.5e7 N/m³)
instead of the per-frame fit's degenerate floor-collapse; the residual per-frame GRF error
is now correctly attributed to reconstruction quality (cm-scale, time-varying foot
vertical error through the roll), the real limiter. **148 tests pass total.**
**Contact-in-sim DONE** (`contact/forward_sim.py`): the distributed contact model now runs
**inside a Newton MuJoCo forward simulation** — `ContactForwardSim` steps the MuJoCo
solver's forward dynamics (`mj_step`, the same engine `SolverMuJoCo` integrates) and each
step applies the **Warp-computed** distributed contact wrench as an external force
(`xfrc_applied`, net force + torque about the body COM) from the body's current world
pose+spatial-velocity, so the ground reaction *emerges* from the sim. A rigid body dropped
onto the belt settles to the **analytic** elastic-foundation equilibrium
(GRF == weight, z == -weight/(k*A)) on both CPU and the Warp backend; hydroelastic reduces
to the same equilibrium; sliding produces bounded Coulomb friction opposing motion. This
is the foundation for contact-rich forward/RRA simulation. **153 tests pass total.**

**Full-skeleton contact-rich forward sim DONE** (`contact/tracking.py`): the distributed
contact now runs on the **whole M3-exported skeleton MJCF** under the Newton MuJoCo solver
(`mj_step1`/`mj_step2` control pattern), with the Warp/NumPy contact wrench applied to
`calcn_r`/`calcn_l` each step. **Frame crux solved:** the exported MJCF is OpenSim Y-up,
but baking the Y-up→Z-up rotation `R_OS2PM` into the **free-root pose only** rigidly
rotates the whole tree into Z-up (`mjcf_qpos_zup`), so MuJoCo's body poses equal
`build_motion`'s Z-up poses exactly (validated to 1e-6 vs `foot_trajectory_from_motion`) —
the Z-up contact kernels + subject sole then apply unchanged. **Validation = the
body-weight invariant:** a *frozen standing drop* (`frozen_skeleton_xml` locks every
non-root joint at the reference pose via `<equality><joint>` so the skeleton is one rigid
body with a free floating base; MuJoCo's constraint solver transmits the distal foot load
up the locked chain to the root; the root's horizontal+orientation DOFs are held by an
inertia-scaled computed-torque PD on the root only, leaving the **vertical** DOF free)
settles to a static equilibrium where the summed two-foot vertical GRF equals the model's
total weight (elastic mean 0.986, hydroelastic 0.997, **Warp backend == NumPy** 0.986),
with the COP under each foot. This is pure force balance → **reconstruction-quality
independent**, so it validates the whole harness (frames, both feet, Warp contact, Newton
MuJoCo stepping). **Hard-won lessons (see below).** **158 tests pass total.**

### Servo/stepping lessons (contact/tracking.py) — do NOT relearn
- **MuJoCo auto-resets on divergence** (`time`→0, `qpos`→`qpos0`) and prints the
  "Nan/Inf/huge value in QACC" warning; a repeating time-reset pattern = repeated blow-up.
- A **fixed-`kp` joint PD blows up the low-inertia joints** (forearm pro/sup) because
  `wn*dt` is huge there. Fix: **inertia-scaled gains** (`kp=M_jj*wn²`, `kd=M_jj*2ζwn`
  from the mass-matrix diagonal via `mj_fullM`) → uniform, safe `wn` across all joints.
- Even inertia-scaled **diagonal PD on the full articulated floating base diverges**
  (ill-conditioned `M⁻¹K` from the 75 kg root vs 1e-3 forearm spread; worsened by the
  coupled-knee `<equality>` constraints). `implicit`/`implicitfast` do **not** help —
  they don't integrate `qfrc_applied` implicitly.
- **Kinematic teleport-freeze (overwrite qpos/qvel each step) does NOT work**: the contact
  is on the distal `calcn`, and teleporting the leg joints does not transmit the load to
  the free root, so it sinks until the one-step coupling alone balances (~8.7× weight).
  Use **`<equality>` joint locks** instead (MuJoCo solves + transmits the constraint
  forces). With frozen joints, servo **only the root** (`sim.servo_joints=False`).
- Contact must be **overdamped + started near the equilibrium penetration**
  (`ground_gap<0` = initial penetration) or the free-fall transient overshoots to huge
  penetration/GRF. Report the tail time-mean (small residual sway averages out).
- The **servo path (`hold_mode="servo"` with `servo_joints=True`) for dynamic joint
  tracking is still not stable** — needs full (not diagonal) computed torque or MuJoCo
  actuators+implicit. This is the open item for walk-phase forward tracking.

**M8 (Newton imitation env) DONE — runnable end to end.** `export/protomotions_robot.py`
turns a fit into a runnable ProtoMotions setup: `write_biomech_asset` (fitted MJCF into the
asset tree), `build_simbody_motion` (a `.motion` clip whose `rigid_body_*` align 1:1 with
the **sim** body set — MuJoCo FK over ALL bodies incl. the exporter's massless dummy
bodies, so 38 bodies not the 20 anatomical; Y-up→Z-up), `build_biomech_robot_config` +
`export_protomotions_bundle` (7 bridge tests). The fitted robot is registered as
`"biomech"` (`protomotions/robot_configs/biomech.py` → `BiomechRobotConfig`, in
`factory.py`): 31 coupled-knee sim DOFs/actions, 38 bodies, anchor `torso`, uniform
natural-frequency PD (10 Hz, ζ=2) — a **starting point to tune vs GRF**. The experiment is
`experiments/mimic_newton.py` (mirrors `examples/experiments/mimic/mlp.py`). **Real S001
subject exported** (`tools/export_s001_subject.py`): regenerates the committed asset
`protomotions/data/assets/mjcf/biomech_rajagopal.xml` **with the S001 group scales** and
builds the matching clip `data/motions/biomech_s001_walk.motion` (150 frames @ 100 fps,
pelvis ~0.96–1.02 m — physical walking height; same scales ⇒ clip geometry matches the
asset). **Confirmed runnable on the Newton simulator** via `protomotions/train_agent.py
--robot-name biomech --simulator newton ... --motion-file .../biomech_s001_walk.motion
--experiment-path projects/biomech/experiments/mimic_newton.py`: builds the sim, loads the
robot + clip (1.490 s), sets up foot contact sensors on all 6 foot bodies, captures the
CUDA graph, runs PPO rollouts + `agent.save()` (checkpoints written). **Windows fix
(needed to run at all):** Lightning defaults to DDP whose process group init picks NCCL,
which Windows lacks; `train_agent.py` now uses `DDPStrategy(process_group_backend="gloo")`
on `win32` only (real world_size=1 process group, so the agent's `torch.distributed`
collectives keep working; NCCL/DDP retained elsewhere). `max_epochs =
training_max_steps // total_envs // num_steps`, so use `--training-max-steps ≥
num_envs*num_steps` for ≥1 real epoch. Remaining GPU step (not a code gap): a full PPO
training-to-convergence run.
**Runnable-env fixes (do NOT relearn):** (1) `BiomechRobotConfig` must use
`ControlType.BUILT_IN_PD` (Newton's native PD, like every shipped robot) — the Newton
simulator's `PROPORTIONAL`/`TORQUE` `_physics_step` branches call an undefined
`_action_to_pd_targets`/`_action_to_torque_targets` and crash. (2) `build_simbody_motion`
MUST swap the free-root quat **xyzw→wxyz** before `data.qpos` for MuJoCo FK
(`dart_q_to_mjcf_qpos` emits xyzw; MuJoCo free joints read wxyz) — skipping it renders the
whole body upside down; guarded by `test_simbody_motion_is_upright`. (3) The M3 exporter
emits **no `<geom>`** (kinematic/dynamic only), so the renderer draws nothing;
`export_mjcf(visual_geoms=True)` (default in `write_biomech_asset`) adds non-colliding
capsule bones + leaf spheres (`density=0`, `contype/conaffinity=0`, explicit `<inertial>`
kept → FK/dynamics unchanged). The viewer needs `pyglet>=2.0` (pip-installed).
**189 tests pass total.** UPDATE: `export_s001_subject.py` now passes
`write_biomech_asset(..., bone_meshes=True)`, so the committed asset renders the actual
OpenSim Rajagopal bone meshes (79 visual-only STL geoms scaled per-body by the subject
group scales; capsule fallback for meshless bodies) instead of capsules. Meshes live in
`protomotions/data/assets/mesh/biomech_rajagopal/` (regenerate via
`tools/convert_bone_meshes.py`); MuJoCo compiles the asset (ngeom=nmesh=79).
**TM2OG (treadmill→overground) DONE** (`export/tm2og.py`): the S001 walk clip walked in
place (treadmill); a physics sim needs it to translate over ground. Port of Jung & Lee,
*Sensors* 21(3):786, 2021 **virtual-origin** method (Eqs. 2/6/7): the backward-moving
belt origin subtracts from body positions, i.e. `x_overground(t) = x_treadmill(t) +
∫₀ᵗ v_belt dτ` along +forward, plus a **Galilean velocity shift** `v += v_belt` (the
paper only remaps positions; a sim also needs velocities so a planted foot becomes
~stationary overground). We have **no belt markers**, so the **belt-speed log is the
ground truth** for `∫v dt` (S001 walk = constant 1.5 m/s over the window). Pure fore-aft
translation+velocity: rotations/ang-vel/DOFs untouched → foot-ground contact geometry
preserved. **Forward axis is inferred empirically, not assumed:** the clip's walking axis
is **Y** (not X), and the belt drags the stance foot in **+Y**, so overground forward is
**−Y**; `infer_travel_direction` derives this by negating the mean stance-foot horizontal
velocity (physical anchor: overground stance foot ≈ stationary), so the sign is
self-correcting. Wired via `build_simbody_motion(..., belt_speed=)` and
`tools/export_s001_subject.py` (loads the belt log, slices the cache `window`, uses the
mean of the two split belts). Regenerated clip: pelvis travels **−2.28 m in Y** (belt
∫v dt ≈ 2.235 m, ~2% over from residual reconstruction drift), stance-foot horizontal
speed drops ~1.35→0.25 m/s, min body z stays >0 (no floor clip). 9 tests in
`tests/test_tm2og.py` (displacement=∫v, monotonic progress, dir=−Y, TTD=belt distance,
stance foot stationary, rot/DOF unchanged, velocity/position consistency, frame-0
unchanged, numpy path). **198 tests pass total.**

## Directory layout

```
biomech/
  MEMORY.md            <- you are here
  README.md            M1 usage
  frames.py            frames/units conventions (M1)
  session.py           CaptureSession + load_session (M1)
  load_capture.py      CLI report (M1)
  run_pipeline.py      CLI: full reconstruction + contact calibration (real data)
  run_tests.py         no-pytest runner (M1)
  io/                  c3d.py, force_plate.py, treadmill.py (M1; treadmill=belts+protocol)
  tests/               M1 tests
  osim/                .osim parser -> SkeletonSpec (DONE, M2a; 9 tests)
  skeleton/            simmspline.py + spatial.py + functions.py + skeleton.py (DONE, M2b; 9 tests)
  fitting/             closed_form.py + ik.py + ik_initializer.py + marker_fitter.py
                       + dynamics_fitter.py + marker_map.py (real-data bridge)
  export/              mjcf.py + motion.py (DONE, M3; 13 tests) — MJCF + MotionLib export
  fitting/             ... + dynamics_fitter.py (M2e DONE; 17 tests, mass+inertia+GPU)
  contact/             elastic_foundation.py + kinematics.py + calibration.py +
                       hydroelastic.py + foot_geometry.py + pipeline.py + stance.py +
                       forward_sim.py + tracking.py
                       (M5+M6+M7+M4 + pipeline + robust dynamic cal + contact-in-sim +
                        full-skeleton tracking; 53 tests)
  docs/                all planning/memory docs (+ docs/refs/ goldens)
  models/              Rajagopal2015.osim (reproducible parser input)
  tools/nimble_golden/ one-time real-Nimble golden generator (WSL)
  reference/nimble/    vendored Nimble C++ (gitignored, read-only)
```

## Immediate next actions

- **Foot marker map + ankle-angle fix DONE** (`fitting/marker_placement.py` +
  `fitting/marker_map.py`; `tests/test_marker_placement.py`, 5 tests). The stock Rajagopal
  foot marker set was too sparse and mis-seated: only `RCAL`/`RMT5`/`RTOE` on `calcn`,
  nothing on `toes`, and `RTOE` placed at the toe tip while the S001 capture marker is on
  the **met-2 head** (~2.7 cm forward, which the fit was absorbing as a spurious foot
  pitch). The S001 C3D actually carries `HEE/HEE2/HEE3` (calcaneus triangle), `MTH1/MTH5`
  (met heads), `TOE` (met-2), `HLX` (hallux) — all 100% present in both static + dynamic.
  `place_foot_markers(spec, static_session)` runs the **OpenSim marker-placement step**:
  fit the static (Cal 101) trial with the stock markers, then express each rich foot marker
  in its owning body frame -> add `RCAL2/RCAL3/RMT1` + `RTOE_TIP` (on `toes`) and re-seat
  `RCAL/RTOE/RMT5`. Round-trips to <2.5 cm on real static markers. It also (a) **unlocks
  the MTP joint** (stock model ships `mtp_angle_{r,l}` `locked=true`) so the new toes marker
  makes `mtp_angle` observable, and (b) **re-zeros the ankle at the static neutral**
  (`register_ankle_neutral`): the fitted static stand reads ~ -10 deg ankle (matches PiG
  `RStaticPlantFlex` = 9.7/8.8 deg in `S001.mp`), a model-neutral offset that was biasing
  the *entire* dynamic ankle trace negative. The re-zero bakes `Rz(-off)` into the ankle
  joint's `T_child`, so with `q' = q - off` every body's world pose is provably unchanged
  (contact/export untouched) while the coordinate reads 0 at standing (unit-tested to
  1e-9). **Result on the S001 walk window** (`tools/check_ankle_fix.py`, A/B): ankle goes
  from entirely-negative (R mean -13, max -5; never dorsiflexes) to centred and crossing
  into dorsiflexion (R mean -4.9, max +1.6; L mean +2.4, max +12.6), MTP now fits (R +12,
  L range 35 deg), marker RMS 14.7 -> 13.9 mm. **Now WIRED IN** (2026-07):
  `run_subject_pipeline` (`enrich_foot_markers=True`, `placement_window_len=60`),
  `tools/make_s001_ik_figures.py` and `tools/export_s001_subject.py` all call
  `place_foot_markers` in the reconstruction path. The figures tool pickles the *enriched*
  spec into the cache (`spec_pickle` in `_s001_ik_cache.npz`); the exporter loads that exact
  spec so the MJCF (MTP now a hinge, not a weld -> 33 actions; ankle-neutral bake) stays
  self-consistent with the cached poses. Regenerated committed asset
  (`protomotions/data/assets/mjcf/biomech_rajagopal.xml`), motion
  (`biomech_s001_walk.motion`), and figs 09/10/11 (RMS 13.8 mm). `test_pipeline_real_s001`
  passes with enrichment on. **L/R asymmetry re-audited and confirmed NOT a bug**
  (`tools/diagnose_ankle_asymmetry.py`): raw static + dynamic foot/leg markers have 0%
  dropout and are mirror-symmetric on both sides; both subtalars locked, both MTPs unlocked
  identically; ankle pin axes are anatomically mirror-symmetric. Full-cycle phase-aligned
  ankle traces correlate 0.977 (mean R-L diff -1.1 deg, RMS 2 deg) -- per-frame overlay only
  *looks* asymmetric because L/R legs are ~50%% out of phase. Residual real asymmetry is
  small: static ankle neutral R -10.3 / L -12.9 deg (kept per-side, PiG convention), and
  offset mirror mismatches (HEE2/HEE3 ~10-13 mm, hallux ~11 mm) trace directly to the raw
  capture (e.g. the L hallux marker sits ~19 mm more lateral than R). Subtalar still
  `locked` (0) in the model; left as-is (PiG foot markers don't reliably resolve
  inversion/eversion).

- **Subtalar unlock evaluated & REJECTED** (`tools/check_subtalar_unlock.py`). With the
  enriched foot set the subtalar *is* observable and unlocking lowers marker RMS ~0.5 mm,
  but it **rails at the +/-20 deg anatomical limit 12-16%% of frames** and shifts the MTP
  ~8-10 deg -- i.e. it acts as an error sink for forefoot marker/soft-tissue error, not
  clean inversion/eversion. Registering a subtalar static neutral makes it worse (neutral
  comes out at a non-physiological +16/+13 deg; railing climbs to 32-34%%). Kept **locked**
  (both sides). NOTE: earlier claim of "only left subtalar locked" was WRONG -- both are
  locked in the stock model and there is no L/R subtalar asymmetry.

- **Thigh/shank cluster collapse DONE** (`fitting/cluster_collapse.py`;
  `tests/test_cluster_collapse.py`, 3 tests). The soft-tissue tracking plates
  (`RTH1/2/3`, `RTB1/2/3` + L) carried the largest per-marker residual (RTB3 ~64 mm) and
  fed the pose IK three mutually-inconsistent constraints per segment, dragging the solve
  (and pinning a group scale to the 0.5 bound). `collapse_clusters(spec, mapping)` adds one
  centroid marker per cluster (offset = mean of member offsets, refined by the fitter) and
  remaps it to the *set* of member capture labels; `build_observations` now averages a
  tuple-valued mapping per frame (NaN-aware). A/B (`tools/check_cluster_collapse.py`):
  joint-angle roughness improved on 7/10 lower-limb DOFs (only hip_rotation_l, the lost
  long-axis info, got rougher -- surface plates resolve that poorly anyway), angles stay
  physiological, group scales recovered to [0.80, 1.35] (was railing at 0.50), and residual
  on the retained trustworthy markers dropped 13.1 -> 7.6 mm. **Wired in default-on**
  (`run_subject_pipeline(collapse_lower_clusters=True)`, figures tool); export inherits it
  via the pickled spec. Regenerated asset/motion/figs (fig 10 median RMS now 8.6 mm).
  `test_pipeline_real_s001` still passes.

- **Correctness figures DONE** (`tools/make_tracking_figures.py` -> `docs/figures/`):
  5 figures, all viewed & verified: (1) body-weight invariant convergence (tail mean
  0.986x weight) + per-foot GRF, (2) Z-up skeleton standing on the registered ground,
  (3) distributed plantar pressure + COP per foot (Fz=362 N each; COP labelled inline,
  no legend overlap), (4) contact-law/backend parity bars (elastic/hydroelastic/Warp
  all within +-5%), (5) Y-up->Z-up frame consistency vs `build_motion` (all bodies <=1e-8 m,
  pelvis at machine precision). Re-run: `.venv/Scripts/python projects/biomech/tools/make_tracking_figures.py`.
- **IK joint-angle recovery figures DONE** (`tools/make_ik_figures.py` -> `docs/figures/`):
  honest synthetic-gait round trip (q_true -> FK -> markers + Gaussian noise -> IK from a
  NEUTRAL static init -> compare recovered joint angles to truth). 3 figures, all viewed:
  (6) per-DOF angle error: noiseless recovered to ~1e-6 deg (below 1 millidegree), 2 mm
  marker noise gives sub-degree RMS per lower-body DOF; (7) true-vs-recovered time series
  for 6 lower-body DOFs overlay near-perfectly under 2 mm noise; (8) recovered-vs-true
  scatter R^2=0.99873 on identity, and a noise sweep (0/1/2/4/8 mm) showing angle error is
  linear and UNBIASED (0 error at 0 noise; marker RMS == input sigma => IK extracts all
  signal). Perf: batched autodiff Jacobian is the cost (~0.85 s/iter/60-frames on cuda);
  noiseless converges in ~6 iters, noisy solves capped at `_MAX_ITERS=50` (angles converged
  <0.06 deg vs 120-iter ref), solves cached by sigma (single seed). Uses `device="cuda"`.
- **S001 IK speed/anthropometry benchmark STARTED** (`tools/benchmark_s001_ik.py` ->
  `docs/figures/s001_ik_benchmark_metrics.json` + figs 12/13/14). Baseline is now explicit
  and preserved: S001 walk window (1469,1619), median marker RMS 16.639 mm, ranges
  hip R/L 48.77/54.62 deg, knee R/L 69.82/72.67 deg, ankle R/L 22.28/30.26 deg, first
  wall time 357.8 s. Accepted speed variant `fast_bilevel_v1`: same scales, same RMS and
  same joint angles to numerical precision, wall time ~19-30 s (cache/run variance) = ~12-19x
  faster. This is from the fast FD Warp Jacobian path + bounded final IK (`MarkerFitConfig.final_inner`);
  full tests 158/158 pass. Throughput metric now included: warm-started native-Warp-FK/Jacobian
  dynamic IK over 200 S001 frames (`warp_warm_200_v1`) takes 0.863 s = 231.7 frames/s
  (5 LM iterations, 1158 frame-iters/s) with median RMS 16.66 mm. NO TORCH is accepted or
  kept in the code path. Static-calibration marker test DONE: excluding medial knee/ankle
  calibration markers (`RMFC/LMFC/RMMAL/LMMAL` from dynamic IK) did NOT improve accuracy —
  full no-static dynamic fit had RMS 16.80 mm vs baseline 16.64 and worse gait-range penalty;
  final-IK-only no-static had RMS 17.03 mm and inflated knee ranges. Conclusion: keep them
  for now. Static-trial-only marker-offset calibration was also tested and WORSE on dynamic
  gait (Cal fit RMS ~13.7 mm, but walk RMS ~19.8 mm and hip/knee ranges inflated), so do not
  replace dynamic calibration with static offsets alone. **Accepted accuracy improvement:**
  `robust_anatomical_v1` / robust marker weighting (anatomical landmarks high, soft-tissue
  thigh/shank clusters downweighted) on S001 walk: anatomical marker RMS improves from
  ~19.1 mm -> ~14.1 mm, knee peaks drop from ~73 deg -> ~67-69 deg, and gait range penalty
  drops from 2.67 deg -> 0.0 deg. Tradeoff is intentional: tracking/cluster marker residuals
  rise (~30 mm -> ~45 mm) because we stop letting soft-tissue cluster motion dominate bone
  angles. `robust_balanced_v2` was also benchmarked but did not supersede v1 (penalty ~0.4,
  similar anatomical/cluster residual). Next accuracy lever: robust/adaptive marker weights
  based on per-marker residual + static trial for marker identity/offset sanity, not naive
  static-only offsets. Rejected anthropometry v1 attempts are also saved for honesty:
  `anthropo_fixed_v1` (13.5 s but RMS 27.2 mm, ankle/knee worse) and `anthropo_prior_v1`
  (13.7 s but RMS 25.6 mm, left hip/knee worse, scale hit lower bound). Conclusion: do NOT
  accept naive .mp scale fixation/strong prior. Next accuracy lever is subject-specific
  marker correspondence/local-offset calibration from S001 `.mp`/static trial + robust
  marker weighting, not simply forcing segment scales.
- **S001 real-data IK figures DONE** (`tools/make_s001_ik_figures.py` -> `docs/figures/`,
  results cached in `docs/figures/_s001_ik_cache.npz`; `--fresh` to refit). Real data has NO
  ground-truth angles, so these show self-consistency/plausibility instead of recovered-vs-
  true. Methodology = standard scale-once/IK-per-frame: full bilevel `MarkerFitter` on a
  60-frame mid-slice for {scales, offsets}, then fixed-param batched `solve_marker_ik` over
  a 150-frame contiguous walk window (auto-picked best-visibility inside `phase_window("walk")`).
  Figures: (9) recovered lower-body joint angles, right vs left, with measured split-belt GRF
  stance SHADED -> bilaterally symmetric ~50%-cycle-lagged gait curves that phase-lock to the
  independent GRF; (10) marker-fit residual per-frame (median ~16.6 mm) + per-marker sorted
  bars -> bony landmarks (forearm/ACR/ASIS/PSIS) fit 3-15 mm, thigh/shank cluster+wand markers
  (LTH*/RTH*/*TB3/RUA1) 40-68 mm = the PiG<->Rajagopal marker-set mismatch + soft-tissue
  artifact, NOT the solver; (11) reconstruction filmstrip (6 stride snapshots, clearly a
  walking human on the ground). Cost: full refit ~400 s on cuda (scale FD Jacobian + IK).
  KEY honest caveat: 16.6 mm marker RMS is real-data quality -> some gait-amplitude inflation
  (knee peak ~73 deg vs ~60 normative); the deferred anthropometric prior / better marker
  correspondence is the lever. The round-trip figs 06-08 isolate solver correctness; these
  show the real reconstruction.
- **Contact rungs M5/M6/M7/M4 + the real-data pipeline are all DONE.** The full contact
  stack exists and `contact/pipeline.py` runs it end to end on real S001: fit -> export
  motion -> subject sole -> ground registration -> **calibrate hydroelastic vertical GRF
  to ~1% on real belt data**. Next real-data steps below.
- **mu (friction) calibration from sliding:** friction is unobservable in planted stance
  (sliding velocity ~0). Extend `pipeline.py` to detect sliding sub-phases (nonzero
  tangential foot velocity) and calibrate `mu_d`/`mu_s`/`v_stribeck` there. **Now
  unblocked** by M1.5 — use `phase="walk"`/`"run"` to get windows with real sliding.
    Pairs with the aggregate-objective / smoothed-kinematics work below (per-frame
    calibration on dynamic windows is currently poor).
  - **Robust dynamic-window contact calibration (DONE, `contact/stance.py` + calibration
    `objective="aggregate"` + pipeline `registration="flatfoot"`):** stance segmentation
    (`segment_contacts`), flat-foot detection (`flat_foot_mask`: planted + horizontal +
    slow + loaded) and flat-foot ground registration (`register_ground_flatfoot`, median
    lowest sole z over planted frames — robust to the foot rolling/lifting), plus an
    aggregate (per-stance sub-bin mean) calibration objective that averages out unbiased
    per-frame kinematic noise (proven on synthetic: aggregate ~0.5% k-error vs per-frame
    ~4% under 0.8 mm jitter). On the real S001 **walk** window this yields *physical*
    stiffness (R k≈7.8e6, L≈6.5e7 N/m³) instead of the degenerate floor-collapse the
    per-frame fit produced. **The remaining per-frame GRF error on dynamic windows is now
    correctly attributed to reconstruction quality** (cm-scale, time-varying foot vertical
    error through the roll), not the contact code — no single ground plane can fix it.
    Next lever there: better marker fit (anthropometric prior) or forward simulation.
- **M1.5 (treadmill protocol) DONE:** `read_speedchange` + `session.phase_window` +
  pipeline/CLI `phase=` select a walk/run/all window. **CLI `run_pipeline.py` DONE.**
- **Anthropometric prior** (deferred from M2d) to further improve the marker fit.
- **Contact inside the Newton sim (DONE, `contact/forward_sim.py` + `contact/tracking.py`):**
  the distributed contact model runs inside MuJoCo forward dynamics (`mj_step`) as an
  applied external force — first on a single rigid body (drop-to-equilibrium, CPU + Warp),
  now on the **full M3-exported skeleton** (`tracking.py`): a frozen standing drop settles
  to the **body-weight invariant** (total two-foot vertical GRF == model weight), COP under
  each foot, on CPU + Warp, elastic + hydroelastic. **Next:** dynamic **walk-phase forward
  tracking** against measured GRF — blocked on a stable full-body joint servo (diagonal PD
  diverges; needs full computed torque or MuJoCo actuators+implicit; see tracking lessons)
  AND on reconstruction quality (the cm-scale time-varying foot vertical error, the known
  limiter). Also a GPU `mujoco_warp` stepping backend for `tracking.py`.
- **M2e extensions** (mass + full 10-param inertial ID + GPU-batched residual DONE):
  remaining polish — batch the inertial **regressor across warp worlds**
  (blocked by `mjw.Model` sharing params across worlds), optional kinematic RRA.

## Regenerating goldens

See `tools/nimble_golden/README.md`. TL;DR: WSL2 Ubuntu-22.04, `~/nimble-golden` venv,
`nimblephysics==0.10.52.1` **with `numpy<2`** (numpy 2.x segfaults the bindings).

## Conventions to never break

- `projects/` is not a package: entry scripts do `sys.path.insert(0, "projects")` then
  `import biomech`.
- SI units, Z-up lab==world; OpenSim Y-up conversion only at skeleton import.
- Gold-standard `q(t)` never goes through the 18-keypoint/PyRoki path.
- `reference/nimble/` is read-only reference; never build/import it.
- Validate every Newton/MuJoCo call against pins: newton 1.0.0, warp 1.14.0,
  mujoco 3.5.0.
