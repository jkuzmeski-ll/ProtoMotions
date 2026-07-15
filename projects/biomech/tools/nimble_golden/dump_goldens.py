"""Dump golden reference values from *real* Nimble for the Windows port to test against.

This is a **one-time reference generator**, not a runtime dependency. It runs under a
Linux Python that has ``nimblephysics`` installed (we use WSL2 Ubuntu-22.04), reads the
bundled ``Rajagopal2015.osim`` gold-standard model, and writes JSON goldens that the
Windows-native port (`biomech.osim`, `biomech.skeleton`) is unit-tested against for
parity.

Outputs (written to ``--out-dir``, default ``projects/biomech/docs/refs``):
  - ``rajagopal2015_structure.json``       bodies, joints (type/dofs/axis/transforms),
                                           markers, scale groups  -> tests osim/parser
  - ``rajagopal2015_fk.json``              body world transforms + marker world positions
                                           at q=0 and random poses -> tests skeleton FK
  - ``rajagopal2015_scaling.json``         same, under random group scales -> tests
                                           anisotropic segment scaling
  - ``rajagopal2015_customjoint_sweep.json`` per-CustomJoint driving-DOF sweep (child
                                           body world transform) -> tests SimmSpline +
                                           CustomJoint coupling (the gold-standard part)

Run (from the Windows repo root, via WSL):
    wsl.exe -d Ubuntu-22.04 -- bash -lc \\
      "~/nimble-golden/bin/python \\
       /mnt/c/Users/JO31399/DigitalHumans/ProtoMotions/projects/biomech/tools/nimble_golden/dump_goldens.py"

All matrices are 4x4 row-major lists in Nimble's native OpenSim (Y-up, meters) frame.
Do NOT apply the OpenSim->ProtoMotions Z-up rotation here; parity is checked in Nimble's
own frame, and the Z-up conversion is a separate downstream concern.
"""

import argparse
import json
import os

import numpy as np
import nimblephysics as nimble


def mat(x):
    return np.array(x, dtype=np.float64).tolist()


def vec(x):
    return np.array(x, dtype=np.float64).ravel().tolist()


def safe_limits(lo, hi, fallback=1.5):
    lo = np.asarray(lo, dtype=np.float64)
    hi = np.asarray(hi, dtype=np.float64)
    lo = np.where(np.isfinite(lo), lo, -fallback)
    hi = np.where(np.isfinite(hi), hi, fallback)
    return lo, hi


def marker_world_positions(f):
    """Marker world positions computed the way Nimble's authoritative API does:
    ``T_body * (bodyScale ⊙ offset)`` (Skeleton::getMarkerWorldPositions,
    Skeleton.cpp:6557). The local marker offset is scaled by the segment scale;
    this matters for the scaling golden (at unit scale it reduces to the plain
    ``T_body * offset``). The biomech port (biomech.skeleton) reproduces this."""
    out = {}
    for name, (body, off) in f.markersMap.items():
        T = np.array(body.getWorldTransform().matrix(), dtype=np.float64)
        scale = np.array(body.getScale(), dtype=np.float64).ravel()
        local = scale * np.array(off, dtype=np.float64).ravel()
        p = T[:3, :3] @ local + T[:3, 3]
        out[name] = p.tolist()
    return out


def body_transforms(sk):
    return {
        sk.getBodyNode(i).getName(): mat(sk.getBodyNode(i).getWorldTransform().matrix())
        for i in range(sk.getNumBodyNodes())
    }


def dump_structure(sk, f):
    bodies = []
    for i in range(sk.getNumBodyNodes()):
        b = sk.getBodyNode(i)
        pb = b.getParentBodyNode()
        pj = b.getParentJoint()
        entry = dict(
            index=i,
            name=b.getName(),
            parentBody=(pb.getName() if pb is not None else None),
            parentJoint=(pj.getName() if pj is not None else None),
            mass=float(b.getMass()),
            localCOM=vec(b.getLocalCOM()),
        )
        # Inertia is only needed for the (later) MuJoCo dynamics side; capture it
        # defensively since binding shapes vary.
        try:
            inertia = b.getInertia()
            entry["inertiaMoment"] = [float(v) for v in inertia.getMoment()]
            entry["inertiaSpatialTensor"] = mat(inertia.getSpatialTensor())
        except Exception:
            pass
        bodies.append(entry)

    joints = []
    for i in range(sk.getNumJoints()):
        j = sk.getJoint(i)
        d = dict(
            index=i,
            name=j.getName(),
            type=j.getType(),
            numDofs=int(j.getNumDofs()),
            dofNames=[j.getDofName(k) for k in range(j.getNumDofs())],
            dofIndicesInSkeleton=[int(j.getIndexInSkeleton(k)) for k in range(j.getNumDofs())],
            parentBody=(j.getParentBodyNode().getName() if j.getParentBodyNode() is not None else None),
            childBody=j.getChildBodyNode().getName(),
            Tparent=mat(j.getTransformFromParentBodyNode().matrix()),
            Tchild=mat(j.getTransformFromChildBodyNode().matrix()),
            posLower=[float(v) for v in j.getPositionLowerLimits()],
            posUpper=[float(v) for v in j.getPositionUpperLimits()],
        )
        for attr, key in (("getAxisOrder", "axisOrder"), ("getFlipAxisMap", "flipAxisMap")):
            fn = getattr(j, attr, None)
            if fn is not None:
                try:
                    val = fn()
                    d[key] = str(val) if key == "axisOrder" else vec(val)
                except Exception:
                    pass
        joints.append(d)

    anat = set()
    for a in ("anatomicalMarkers",):
        try:
            anat = set(getattr(f, a))
        except Exception:
            pass
    markers = []
    for name, (body, off) in f.markersMap.items():
        markers.append(
            dict(name=name, body=body.getName(), offset=vec(off), anatomical=(name in anat))
        )

    scale_groups = []
    # NOTE: scale *setters* (setGroupScales/setBodyScales) segfault natively in
    # nimblephysics 0.10.52.1, so we capture group membership read-only, per body,
    # via getScaleGroupIndex. Scaling parity is validated later against the analytic
    # Jacobian getJointWorldPositionsJacobianWrtBodyScales instead of by mutating
    # scales (see dump_scaling_jacobian).
    try:
        n_groups = int(sk.getGroupScales().shape[0]) // 3
        members = [[] for _ in range(n_groups)]
        for i in range(sk.getNumBodyNodes()):
            gi = int(sk.getScaleGroupIndex(sk.getBodyNode(i)))
            if 0 <= gi < n_groups:
                members[gi].append(sk.getBodyNode(i).getName())
        scale_groups = members
    except Exception:
        scale_groups = []

    return dict(
        model="Rajagopal2015",
        frame="opensim_y_up_meters",
        numDofs=int(sk.getNumDofs()),
        numBodies=int(sk.getNumBodyNodes()),
        numJoints=int(sk.getNumJoints()),
        groupScalesDim=int(sk.getGroupScales().shape[0]),
        bodies=bodies,
        joints=joints,
        markers=markers,
        bodyScaleGroups=scale_groups,
    )


def dump_fk(sk, f, n=8, seed=0):
    rng = np.random.default_rng(seed)
    ndofs = sk.getNumDofs()
    lo, hi = safe_limits(sk.getPositionLowerLimits(), sk.getPositionUpperLimits())
    saved = np.array(sk.getPositions(), dtype=np.float64)
    cases = []
    poses = [np.zeros(ndofs)] + [lo + (hi - lo) * rng.random(ndofs) for _ in range(n)]
    for q in poses:
        sk.setPositions(q)
        cases.append(
            dict(q=vec(q), bodyTransforms=body_transforms(sk), markers=marker_world_positions(f))
        )
    sk.setPositions(saved)
    return dict(model="Rajagopal2015", seed=seed, cases=cases)


def dump_customjoint_sweep(sk, steps=25):
    ndofs = sk.getNumDofs()
    saved = np.array(sk.getPositions(), dtype=np.float64)
    out = {}
    for i in range(sk.getNumJoints()):
        j = sk.getJoint(i)
        if not j.getType().startswith("Custom"):
            continue
        di = int(j.getIndexInSkeleton(0))
        lo = j.getPositionLowerLimit(0)
        hi = j.getPositionUpperLimit(0)
        if not np.isfinite(lo):
            lo = -2.0
        if not np.isfinite(hi):
            hi = 2.0
        child_name = j.getChildBodyNode().getName()
        samples = []
        for x in np.linspace(lo, hi, steps):
            q = np.zeros(ndofs)
            q[di] = x
            sk.setPositions(q)
            entry = dict(
                x=float(x),
                childBodyWorld=mat(j.getChildBodyNode().getWorldTransform().matrix()),
            )
            get_rel = getattr(j, "getRelativeTransform", None)
            if get_rel is not None:
                try:
                    entry["Trel"] = mat(get_rel().matrix())
                except Exception:
                    pass
            samples.append(entry)
        out[j.getName()] = dict(
            type=j.getType(),
            dofName=j.getDofName(0),
            dofIndexInSkeleton=di,
            childBody=child_name,
            samples=samples,
        )
    sk.setPositions(saved)
    return out


def dump_scaling(sk, f, n=5, seed=1):
    """Direct anisotropic-scaling golden: body transforms + marker world positions
    under random per-group scales and random poses. Tests the port's group scaling."""
    rng = np.random.default_rng(seed)
    ndofs = sk.getNumDofs()
    lo, hi = safe_limits(sk.getPositionLowerLimits(), sk.getPositionUpperLimits())
    saved_q = np.array(sk.getPositions(), dtype=np.float64)
    saved_s = np.array(sk.getGroupScales(), dtype=np.float64)
    sdim = saved_s.shape[0]
    cases = []
    for _ in range(n):
        scales = 0.85 + 0.30 * rng.random(sdim)  # anisotropic, in [0.85, 1.15]
        q = lo + (hi - lo) * rng.random(ndofs)
        sk.setGroupScales(scales)
        sk.setPositions(q)
        cases.append(
            dict(
                groupScales=vec(scales),
                q=vec(q),
                bodyTransforms=body_transforms(sk),
                markers=marker_world_positions(f),
            )
        )
    sk.setGroupScales(saved_s)
    sk.setPositions(saved_q)
    return dict(model="Rajagopal2015", seed=seed, cases=cases)


def dump_scaling_jacobian(sk):
    """Analytic d(jointWorldPositions)/d(bodyScales) at a few random poses.

    Used to validate the port's anisotropic segment scaling WITHOUT mutating scale
    state (the scale setters segfault in this Nimble build). Our port checks its own
    scaling by finite-differencing joint world positions w.r.t. body scales and
    comparing to this analytic Jacobian."""
    fn = getattr(sk, "getJointWorldPositionsJacobianWrtBodyScales", None)
    if fn is None:
        return None
    joints = [sk.getJoint(i) for i in range(sk.getNumJoints())]
    rng = np.random.default_rng(2)
    ndofs = sk.getNumDofs()
    lo, hi = safe_limits(sk.getPositionLowerLimits(), sk.getPositionUpperLimits())
    saved = np.array(sk.getPositions(), dtype=np.float64)
    cases = []
    for _ in range(3):
        q = lo + (hi - lo) * rng.random(ndofs)
        sk.setPositions(q)
        try:
            J = np.array(fn(joints), dtype=np.float64)
        except Exception as e:  # noqa: BLE001
            sk.setPositions(saved)
            return dict(error=repr(e))
        cases.append(dict(q=vec(q), jacShape=list(J.shape), jac=J.tolist()))
    sk.setPositions(saved)
    return dict(model="Rajagopal2015", note="d(jointWorldPos)/d(bodyScales)", cases=cases)


def main():
    default_out = "/mnt/c/Users/JO31399/DigitalHumans/ProtoMotions/projects/biomech/docs/refs"
    ap = argparse.ArgumentParser()
    ap.add_argument("--osim", default=None, help="path to .osim (default: bundled Rajagopal2015)")
    ap.add_argument("--out-dir", default=default_out)
    args = ap.parse_args()

    osim = args.osim or os.path.join(
        os.path.dirname(nimble.__file__), "models", "rajagopal_data", "Rajagopal2015.osim"
    )
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"nimble file: {nimble.__file__}")
    print(f"parsing: {osim}")
    f = nimble.biomechanics.OpenSimParser.parseOsim(osim)
    sk = f.skeleton
    print(f"dofs={sk.getNumDofs()} bodies={sk.getNumBodyNodes()} joints={sk.getNumJoints()}")

    outputs = {
        "rajagopal2015_structure.json": dump_structure(sk, f),
        "rajagopal2015_fk.json": dump_fk(sk, f),
        "rajagopal2015_scaling.json": dump_scaling(sk, f),
        "rajagopal2015_customjoint_sweep.json": dump_customjoint_sweep(sk),
    }
    scaling_jac = dump_scaling_jacobian(sk)
    if scaling_jac is not None:
        outputs["rajagopal2015_scaling_jacobian.json"] = scaling_jac
    for fname, data in outputs.items():
        p = os.path.join(args.out_dir, fname)
        with open(p, "w") as fh:
            json.dump(data, fh, indent=1)
        print(f"wrote {p} ({os.path.getsize(p)} bytes)")


if __name__ == "__main__":
    main()
