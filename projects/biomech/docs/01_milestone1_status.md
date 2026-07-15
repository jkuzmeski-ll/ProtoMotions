# Biomech pipeline — status

Local pipeline: mocap + instrumented-treadmill → gold-standard skeleton motion →
Newton distributed-contact research. Full plan in the handoff summary and in
`kinodynamic_human_retargeting_newton.md` / `foot_contact_modeling_mjcf_newton.md`.

## Milestone 1 — local C3D + treadmill sync loader — DONE

Code: `projects/biomech/` (package `biomech`, `numpy`-only, no `ezc3d`).

- `io/c3d.py` — dense frame-indexed C3D reader (markers m + NaN gaps; analog
  scaled at 2000 Hz; float & int storage).
- `io/force_plate.py` — per-belt GRF / world-frame COP / free moment from
  `FORCE_PLATFORM`. Validated: COP lands on the correct belt; GRF ≈ bodyweight.
- `io/treadmill.py` — Visual3D `SPEEDCHANGE` belt reader + time resampler.
- `session.py` — unified `CaptureSession`: `t_point` (100 Hz) + `t_analog`
  (2000 Hz), belt resampled onto both, events, `.mp` metadata, explicit
  `Frames`. `load_session(...)` is the entry point.
- `frames.py` — explicit LAB/FORCE_PLATE/TREADMILL/WORLD, SI units, Z-up.
- `tests/` + `run_tests.py` (no pytest): 17 tests pass on S001.
- `load_capture.py` — CLI/report. Report saved at
  `projects/data/S001/Trial101_session_report.json`.

### S001 facts locked in
- 72 markers @100 Hz, 17844 frames (178.4 s); 249 analog @2000 Hz.
- Two TYPE-2 belts: ch 1–6 at +x, ch 7–12 at −x, shared edge x=0, Z-up meters.
- Lab == world (identity); OpenSim Y-up conversion deferred to skeleton stage.
- GRF exposed as force-on-subject (+Z up); measured force is −that.
- Peak vertical GRF ~1900 N (~2.4 BW at belt 3.0 m/s); mass 81.65 kg (from `.mp`).

### Open items carried forward
- Belt-log rate is **inferred ~298 Hz** (no rate in file, separate clock) →
  confirm true rate + mocap↔belt offset, then pass `belt_rate_hz`.
- Assign each belt → left/right foot (needs subject facing direction).
- COP math assumes axis-aligned plates (true here; rotated plates flagged).
- Only 6-axis GRF+COP available (no plantar pressure map) — sets validation
  strength for later contact rungs.

## Next: Milestone 2 — local Nimble full-body fit
MarkerFitter (scale + marker offsets + IK) → DynamicsFitter (residual reduction
+ inertia from GRF) → `q(t)`, `qd(t)`, measured GRF/COP; report marker RMS +
residuals. Fit FULL body (upper-body markers present; GRF ground truth needs
whole-body inertia) even though research is lower-body-focused. Verify Nimble
licensing before vendoring.
