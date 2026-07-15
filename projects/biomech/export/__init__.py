"""biomech.export — fitted skeleton + motion -> MJCF and ProtoMotions MotionLib.

Milestone M3. Converts a fitted ``SkeletonSpec`` + ``q(t)`` into:

- an **MJCF** model ready for the Newton MuJoCo solver (``SolverMuJoCo``). OpenSim
  coupled DOFs (e.g. the knee's flexion-coupled translation) are approximated via MuJoCo
  ``equality``/tendon coupling or baked; the coupling error is quantified at export.
- a ProtoMotions **MotionLib** clip (``gts/grs/gvs/gavs/dps/dvs``) so the gold-standard
  motion drives ProtoMotions tracking/imitation. FK/IK per ``protomotions/components/
  pose_lib.py`` / ``motion_lib.py``.

Also the working analog of Nimble's ``SubjectOnDisk`` (``.b3d``) output.
See ``docs/20_nimble_port_plan.md`` (M3).
"""

from biomech.export.mjcf import (
    MjcfExportResult,
    dart_q_to_mjcf_qpos,
    export_mjcf,
    write_mjcf,
)
from biomech.export.motion import (
    MotionExportResult,
    R_OS2PM,
    build_motion,
    write_motion,
)
from biomech.export.subject import (
    FittedSubject,
    load_subject,
    save_subject,
)

__all__ = [
    "MjcfExportResult",
    "export_mjcf",
    "write_mjcf",
    "dart_q_to_mjcf_qpos",
    "MotionExportResult",
    "build_motion",
    "write_motion",
    "R_OS2PM",
    "FittedSubject",
    "save_subject",
    "load_subject",
]
