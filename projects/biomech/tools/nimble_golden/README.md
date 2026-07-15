# Nimble golden-reference workflow

One-time generation of **gold-standard reference values** from *real* Nimble, so the
Windows-native port (`biomech.osim`, `biomech.skeleton`, `biomech.fitting`) can be
unit-tested for parity. Nimble itself is **not** a runtime dependency — it runs once,
in Linux (WSL2), to emit JSON goldens into `../../docs/refs/`.

## Environment (verified working)

- WSL2 distro: **Ubuntu-22.04** (`wsl.exe -d Ubuntu-22.04`), Python 3.10.12.
- venv: `~/nimble-golden` (in the WSL home, not on `/mnt/c`, for speed).
- Packages: `nimblephysics==0.10.52.1` **and `numpy<2` (we pin `numpy==1.26.4`)**.

### CRITICAL gotcha: numpy must be < 2

`nimblephysics 0.10.52.1` is built against the numpy 1.x C-ABI. With **numpy 2.x**,
passing arrays into its pybind/Eigen bindings **segfaults** on basic calls
(`setPositions`, `setGroupScales`, `setBodyScales`, marker/FK access) with no catchable
Python exception. Installing `nimblephysics` pulls numpy 2.x by default, so you **must**
downgrade afterwards:

```bash
python3 -m venv ~/nimble-golden
~/nimble-golden/bin/python -m pip install --upgrade pip
~/nimble-golden/bin/python -m pip install nimblephysics
~/nimble-golden/bin/python -m pip install 'numpy<2'   # <-- REQUIRED, else segfaults
```

Other API notes for this binding version:
- `BodyNode.getMomentOfInertia(...)` is a **setter** (takes 6 floats); there is no
  `getInertia()`. Inertia isn't needed for the kinematic port anyway.
- `getBodyScaleGroups()[i]` indexing throws; use `getScaleGroupIndex(bodyNode)` per body.
- `getJointWorldPositionsJacobianWrtBodyScales(joints)` requires the **joint list** arg.

## Model

The gold-standard target model is Nimble's bundled **Rajagopal2015** at
`nimblephysics/models/rajagopal_data/Rajagopal2015.osim`. A copy lives in the repo at
`projects/biomech/models/rajagopal_data/Rajagopal2015.osim` (the reproducible parser
input; meshes omitted — kinematics don't need them).

Inventory (this is the M2a result that bounds the skeleton port): **37 DOFs, 20 bodies,
20 joints, 66 markers, 20 scale groups**. Joint types:
`EulerFreeJoint`x1 (root `ground_pelvis`), `EulerJoint`x5, `CustomJoint1`x2 (the coupled
knees `walker_knee_r/l`, driven by `knee_angle`, axis order `XZY`; **left knee flip =
[-1,1,1]**), `RevoluteJoint`x10, `UniversalJoint`x2.

## Run

From the Windows repo root:

```bash
wsl.exe -d Ubuntu-22.04 -- bash -lc \
  "~/nimble-golden/bin/python \
   /mnt/c/Users/JO31399/DigitalHumans/ProtoMotions/projects/biomech/tools/nimble_golden/dump_goldens.py"
```

## Outputs (in `docs/refs/`, Nimble's native OpenSim Y-up meters frame)

| file | contents | validates |
|---|---|---|
| `rajagopal2015_structure.json` | bodies, joints (type/dofs/axis/flip/transforms/limits), markers, scale groups | `osim/parser.py` |
| `rajagopal2015_fk.json` | body world transforms + marker world positions at q=0 and 8 random poses | `skeleton` FK |
| `rajagopal2015_scaling.json` | same under 5 random anisotropic group scales | `skeleton` group scaling |
| `rajagopal2015_customjoint_sweep.json` | per-knee driving-DOF sweep (child body world transform, 25 steps) | SimmSpline + CustomJoint |
| `rajagopal2015_scaling_jacobian.json` | analytic d(jointWorldPos)/d(bodyScales) at 3 poses | scaling Jacobian |

## Still to golden (needs the S001 marker map, i.e. after M2 marker map exists)

Full-pipeline S001 outputs — IKInitializer joint centers, MarkerFitter `q(t)` + marker
RMS + offsets, DynamicsFitter residuals — for end-to-end parity of `biomech.fitting`.
