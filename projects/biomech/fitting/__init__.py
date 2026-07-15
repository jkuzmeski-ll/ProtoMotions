"""biomech.fitting — marker/dynamics fitting (ports of Nimble's fitters).

Milestones M2c-e. Runs on top of ``biomech.skeleton`` (kinematics) and, for dynamics,
the Newton MuJoCo solver.

- ``marker_fixer.py``    : RANSAC gap-fill / outlier rejection on raw marker traces
                           (port of ``MarkerFixer``). Run before initialization.
- ``ik_initializer.py``  : closed-form joint centers -> group scales -> poses (port of
                           ``IKInitializer``: MDS, Chang-Pollard 2006, Gamage-Lasenby
                           2002, concentric-sphere fit; StackedBody/StackedJoint
                           topology simplification). M2c.
- ``marker_fitter.py``   : bilevel {group scales, marker offsets, per-frame poses} fit
                           (port of ``MarkerFitter`` objective/priors; optimizer swapped
                           to a Warp CMA-ES/LM since IPOPT is unavailable on Windows).
                           M2d.
- ``priors.py``          : marker-offset prior + anthropometric priors
                           (``MarkerOffsetPrior``, ``Anthropometrics``).
- ``dynamics_fitter.py`` : residual reduction + mass/COM/inertia from GRF (port of
                           ``DynamicsFitter`` formulation; dynamics terms via MuJoCo
                           ``mj_inverse``/``mujoco_warp``, not a ported Featherstone).
                           M2e. Public API: ``ResidualHelper``, ``DynamicsFitter``,
                           ``Contact``, ``identify_masses``, GRF adapters.
- ``report.py``          : marker RMS/max + residual reporting (``IKErrorReport``).

Parity: validate each stage against Nimble goldens in ``docs/refs/`` (one-time Linux
reference run). See ``docs/20_nimble_port_plan.md``.
"""
