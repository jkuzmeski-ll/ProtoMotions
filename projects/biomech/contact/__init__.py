"""biomech.contact — distributed surface contact models for the MuJoCo solver.

Milestones M5+ (the research payload). Custom contact "between FEA and point contact"
evaluated on the subject's own foot geometry, driven by Warp kernels around/into the
Newton MuJoCo solver's contact stage:

- rung 1: elastic-foundation (Winkler) contact under prescribed kinematics (M5) —
  ``elastic_foundation.py`` (Warp kernel + NumPy reference) and ``kinematics.py``
  (drive it from a gold-standard ``MotionExportResult``): DONE.
- rung 2: batched calibration of ``(k_bed, c_bed, mu)`` against measured 6-axis
  GRF/COP (M6) — ``calibration.py`` (log-space Levenberg–Marquardt over the Warp
  forward): DONE. Also calibrates the M7 hydroelastic params (``calibrate_hydroelastic``
  with a selectable free-parameter set).
- rung 3: hydroelastic / tactile-rich contact (M7) — ``hydroelastic.py``
  (pressure-field law: spatial compliance map + hyperelastic stiffening +
  Hunt–Crossley dissipation + Stribeck friction; reduces to the Winkler bed in the
  linear limit): DONE (law; analytic soles until M4 supplies subject geometry).

Subject plantar geometry (M4) comes from ``biomech.contact.foot_geometry`` (a
subject-specific tapered ``FootSole`` in the ``calcn`` body frame, sized from a static
C3D and anchored by the scaled model foot markers), reusing the foot-calibration logic in
``data/scripts/calibrate_lower_body_elipsoid_from_static_c3d.py``.

The whole chain is stitched end to end on a real capture by ``pipeline.py``
(``run_subject_pipeline``): load session -> Plug-in-Gait->Rajagopal observations ->
IKInitializer seed -> MarkerFitter (drops marker RMS to ~1.4 cm on S001) -> ``build_motion``
-> subject ``FootSole`` -> robust per-stance ground registration -> measured split-belt
GRF (right belt = right foot) -> ``calibrate_hydroelastic``. On S001 the *vertical* GRF
fits to ~1% (k~7-8e7 N/m^3, alpha~2); friction ``mu`` is excluded by design (planted
stance has ~0 sliding velocity, so shear is physically unobservable).

For *dynamic* (walk/run) windows ``stance.py`` adds the robustness layer:
``segment_contacts`` (contiguous stance phases), ``flat_foot_mask`` +
``register_ground_flatfoot`` (register the ground from genuinely planted frames only,
robust to the foot rolling), and the calibration ``objective="aggregate"`` (per-stance
sub-bin-mean GRF, which averages out unbiased per-frame kinematic noise). These recover
physical stiffness on real walking data where a per-frame fit degenerates; the residual
per-frame GRF error is then limited by reconstruction quality, not the contact model.

``forward_sim.py`` closes the loop: it runs the **MuJoCo solver's forward dynamics** (the
same engine Newton's ``SolverMuJoCo`` integrates) and applies the Warp-computed
distributed contact wrench as an external force (``xfrc_applied``) each step, so the
ground reaction *emerges* from the simulation instead of being read from data. A rigid
body dropped onto the belt settles to the analytic elastic-foundation equilibrium
(GRF == weight, penetration == weight/(k*A)). This is the foundation for contact-rich
forward/RRA simulation on the subject's own plantar geometry.
See ``docs/11_foot_contact_modeling_newton.md`` and ``docs/20_nimble_port_plan.md``.
"""
