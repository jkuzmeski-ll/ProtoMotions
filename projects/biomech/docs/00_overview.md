# 00 — Overview, goal, and locked decisions

## Ultimate goal

A **fully local, Windows-native** pipeline that turns mocap + instrumented-treadmill
captures into **biomechanically gold-standard** skeleton motion, then uses that
motion in **Newton** (via ProtoMotions) to research **distributed surface contact
models** (elastic-foundation → hydroelastic, "between FEA and point contact") on the
**subject's own foot geometry**.

- Ends: improve contact models for biomechanics research; train digital humans.
- Non-goals: muscle-tendon modeling.
- Focus: lower-body research, but fit the **full body** (upper-body markers exist;
  whole-body inertia is needed for correct GRF/residual physics).
- Compute: use **Warp kernels** (population / CMA-ES style, `enable_backward=False`
  where contact is non-differentiable) on the local RTX A6000.

## The pivot that defines the current phase

Originally we planned to run **Nimble** (`nimblephysics`) for the gold-standard
skeleton fit. Nimble ships **no Windows wheels** (manylinux + macOS only), so that
required WSL2/Docker. **Decision: reject the Linux runtime. Instead port Nimble's
math and methods natively to Windows on top of Newton + Warp.** More work, but it
keeps everything local, native, GPU-accelerated, and under our control.

See `20_nimble_port_plan.md` for the full plan and `21_nimble_source_map.md` for the
file-by-file port map.

## Local environment (verified 2026-07)

- OS: Windows. Shell: `sh`. Repo: `C:\Users\JO31399\DigitalHumans\ProtoMotions`.
- Python `3.10.11` in `.venv`.
- **Warp `1.14.0`**, **Newton `1.0.0`** both import and see the GPU
  (`cuda:0` = NVIDIA RTX A6000, 48 GiB, sm_86).
- Newton exposes: `eval_fk`, `eval_ik`, `eval_jacobian`, `eval_mass_matrix`,
  `ModelBuilder`, `Model/State/Control/Contacts`, a batched Warp **`newton.ik`**
  module (`IKSolver`, `IKObjectivePosition/Rotation/JointLimit`, `IKOptimizerLBFGS`,
  `IKOptimizerLM`, `IKSampler`), and solvers `SolverFeatherstone`, `SolverMuJoCo`,
  `SolverXPBD`, `SolverVBD`, `SolverSemiImplicit`, `SolverImplicitMPM`, `SolverKamino`.
- **MuJoCo solver is available and is the chosen physics backend:**
  `newton.solvers.SolverMuJoCo` imports; `mujoco 3.5.0` + `mujoco_warp` are present.
  This is GPU-batched via Warp and also gives us `mj_inverse`-style dynamics
  (mass matrix, bias forces, inverse dynamics) for the DynamicsFitter, plus the
  contact model we want to research.
- Nimble source is vendored (blobless sparse clone, **gitignored**) at
  `projects/biomech/reference/nimble/` — dirs: `dart/biomechanics`, `dart/dynamics`,
  `dart/math`, `python/nimblephysics/biomechanics`. This is **reference only** (read
  the C++ to port the math); it is not built or imported.

## Solver decision (locked)

**Use the Newton MuJoCo solver (`SolverMuJoCo`, backed by `mujoco_warp`) for all
dynamics and simulation** (dynamics fitting, tracking rollouts, and contact
research). Rationale:

- GPU-batched through Warp → matches the "accelerate the math with Warp" goal.
- Provides articulated-body dynamics quantities (mass matrix, bias/Coriolis+gravity,
  forward/inverse dynamics) so the **DynamicsFitter port sits on top of MuJoCo**
  instead of a hand-ported RNEA/CRBA.
- Its contact model is the baseline we extend toward elastic-foundation /
  hydroelastic "between FEA and point contact" methods on the subject's own foot.

Implications for the port:

- The **kinematic fit** (IKInitializer + MarkerFitter) still runs on our **own Warp
  skeleton** with SimmSpline `CustomJoint` fidelity, because MuJoCo FK cannot express
  SimmSpline-coupled DOFs cleanly and the kinematic fit needs no contact/dynamics.
- The fitted skeleton is exported to **MJCF** for the MuJoCo solver. OpenSim coupled
  DOFs (e.g. the knee's flexion-coupled translation) are approximated in MJCF via
  MuJoCo `equality`/tendon coupling or baked; foot/ankle/subtalar joints stay
  high-fidelity because they are the contact-relevant DOFs.
- Custom contact (elastic foundation → hydroelastic) is injected around/into the
  MuJoCo contact stage; validate the exact seam against `mujoco_warp` in this pin.

## Licensing

- Nimble = **MIT** ✅ safe to read, port, and vendor.
- `dart/math/SimmSpline.*` is adapted from OpenSim (**Apache-2.0**), re-licensed MIT
  inside Nimble. Our port keeps attribution (Peter Loan / OpenSim, Apache-2.0).

## Locked data/physics decisions (S001)

- Split-belt treadmill: **left belt file = left foot, right belt file = right foot**
  (belts identical here). Force plates split by x: plate0 at +x, plate1 at −x, both
  span y∈[−0.85, 0.85]; walking axis = y. Shared inner edge at x = 0.
- GRF sign: raw `Fz` is **negative in stance** (force onto plate);
  `ForcePlate.grf = -force_measured` (force **on subject**, +Z up).
- COP is `NaN` below `fz_threshold` (20 N); per-belt COP legitimately crosses x = 0
  during weight transfer — do **not** "fix".
- Only **6-axis GRF + COP** available (no plantar pressure map). This bounds contact
  validation strength.
- Frames: lab == world, **Z-up meters** (identity). OpenSim world is **Y-up meters**;
  apply `R_OS2PM = [[1,0,0],[0,0,-1],[0,1,0]]` **only** at skeleton import, never to
  lab/force data.
- Subject: 81.65 kg, 1.90 m (`S001.mp`).

## Guardrails

- `projects/` is **not** a package; entry scripts must
  `sys.path.insert(0, "projects")` then `import biomech`.
- Do **not** route the gold-standard reference through the 18-keypoint / PyRoki path
  (lossy; that path is only for non-anatomical robots like G1/H1).
- Newton API moves fast — validate every solver/collision/IK call against the pinned
  `newton==1.0.0` / `warp==1.14.0`.
- Configs in ProtoMotions are Python dataclasses; docs under `docs/source/` are the
  ground truth for ProtoMotions itself.
