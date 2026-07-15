# SPDX-License-Identifier: MIT
#
# Windows-native port of Nimble's closed-form IK initialization
# (``dart/biomechanics/IKInitializer.cpp``). This produces the robust, gradient-free
# *seed* that the bilevel marker fit (M2d ``MarkerFitter``) refines: per-frame joint
# centers in world space, anisotropic per-segment (group) scales, and per-frame poses.
#
# Scope of this port (M2c):
#   * StackedBody / StackedJoint topology simplification -> for the target gait models
#     (Rajagopal-2015) there are no weld or stacked low-DOF joints, so each model joint
#     is its own stacked joint and each body its own stacked body. We build the topology
#     generically but do not (yet) collapse welds; if a future model needs it, the merge
#     step slots in here.
#   * closedFormMDSJointCenterSolver  -> DONE (MDS triangulation + rigid map +
#     coplanar-ambiguity resolution), the primary joint-center estimator.
#   * estimateGroupScalesClosedForm   -> DONE (per-body anisotropic scale from
#     joint-center / anatomical-marker distances via getLocalScale, then condensed to
#     the symmetric group-scale vector).
#   * poses                           -> DONE via the batched Warp marker IK
#     (``biomech.fitting.ik``), analogous to Nimble's ``estimatePosesWithIK``.
#
# Deferred polishing passes (refine joint centers under marker noise; not needed for
# clean data and validated separately once real S001 goldens exist):
#   prescaleBasedOnAnatomicalMarkers, closedFormPivotFindingJointCenterSolver,
#   recenterAxisJointsBasedOnBoneAngles.
#
# The heavy per-(frame, marker) work (FK, marker Jacobian, LM pose solve) runs on the
# Warp skeleton; the small dense MDS eigendecomposition per (frame, joint) is host NumPy
# (there is no Warp equivalent for arbitrary small-matrix factorizations). Batching the
# MDS distance-matrix assembly across frames is the natural future Warp target.

"""Closed-form IK initializer: joint centers -> group scales -> poses (Nimble M2c port)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from biomech.fitting import closed_form as cf
from biomech.fitting.ik import MarkerIKConfig, solve_marker_ik
from biomech.skeleton.skeleton import WarpSkeleton


@dataclass
class IKInitializerResult:
    """Outputs of :meth:`IKInitializer.run`."""

    joint_centers: dict[str, np.ndarray]  # joint name -> (F, 3), NaN where unsolved
    group_scales: np.ndarray  # (3*G,) anisotropic per-group scale
    poses: np.ndarray  # (F, ndof)
    marker_rms: np.ndarray  # (F,) per-frame weighted marker RMS (m)
    joint_names: list[str] = field(default_factory=list)


class IKInitializer:
    """Closed-form joint-center / scale / pose initializer for a :class:`WarpSkeleton`.

    Parameters
    ----------
    skel : WarpSkeleton
    observations : (F, M, 3)
        Observed marker world positions in the model's native (OpenSim Y-up, meters)
        frame, aligned to ``skel.marker_names()`` order. Missing markers are NaN.
    anatomical : (M,) bool, optional
        Which markers are anatomical landmarks (used for scaling). Defaults to each
        marker's ``fixed`` flag from the parsed model.
    min_markers_per_joint : int
        A joint is only estimated if it has at least this many adjacent markers
        (Nimble uses 3; below 3 the MDS system is under-determined).
    """

    def __init__(
        self,
        skel: WarpSkeleton,
        observations: np.ndarray,
        anatomical: np.ndarray | None = None,
        min_markers_per_joint: int = 3,
    ):
        self.skel = skel
        self.spec = skel.spec
        self.topo = skel.topo
        self.obs = np.asarray(observations, dtype=np.float64)
        assert self.obs.ndim == 3 and self.obs.shape[2] == 3, self.obs.shape
        assert self.obs.shape[1] == self.topo.num_markers, (
            f"observations M={self.obs.shape[1]} != model markers "
            f"{self.topo.num_markers}"
        )
        self.F = self.obs.shape[0]
        self.min_markers = min_markers_per_joint

        self.marker_names = skel.marker_names()
        self.joint_names = [j.name for j in self.spec.joints]
        self.body_names = skel.body_names()

        if anatomical is None:
            anatomical = np.array(
                [m.anatomical for m in self.spec.markers], dtype=bool
            )
        self.anatomical = np.asarray(anatomical, dtype=bool)

        # marker visibility per frame (finite observations)
        self.visible = np.asarray(
            np.isfinite(self.obs).all(axis=2), dtype=bool
        )  # (F, M)

        # body -> group index (from the model's scale groups)
        self.body_group = np.zeros(self.topo.num_bodies, dtype=np.int32)
        bidx = {b.name: i for i, b in enumerate(self.spec.bodies)}
        for gi, group in enumerate(self.spec.scale_groups):
            for name in group:
                self.body_group[bidx[name]] = gi
        self.num_groups = len(self.spec.scale_groups)

        self._build_neutral()
        self._build_topology()

    # ------------------------------------------------------------------
    # topology + neutral geometry
    # ------------------------------------------------------------------
    def _build_neutral(self):
        """Neutral (q=0, unit scale) joint centers, marker positions, body frames."""
        ndof = self.topo.num_dofs
        world, markers = self.skel.forward(np.zeros(ndof))
        self.neutral_body_world = world[0]  # (B, 4, 4)
        self.neutral_marker_world = markers[0]  # (M, 3)
        jc = np.zeros((self.topo.num_joints, 3), dtype=np.float64)
        for j in range(self.topo.num_joints):
            pb = int(self.topo.j_parent_body[j])
            Tp = self.topo.T_parent[j]
            if pb < 0:
                jw = Tp  # root: parent frame is world identity
            else:
                jw = self.neutral_body_world[pb] @ Tp
            jc[j] = jw[:3, 3]
        self.neutral_joint_center = jc  # (J, 3)

    def _child_body(self, j: int) -> int:
        # joints are parallel to bodies in tree order: joint j's child is body j
        return j

    def _build_topology(self):
        """Joint -> adjacent markers (+neutral squared distances) and joint->joint dists.

        Mirrors the constructor's steps 3-4 in ``IKInitializer.cpp``.
        """
        J = self.topo.num_joints
        m_body = self.topo.m_body
        # joint -> {marker_index: squared distance (neutral)}
        self.joint_markers: list[dict[int, float]] = []
        self.active_joints: list[int] = []
        for j in range(J):
            pb = int(self.topo.j_parent_body[j])
            cb = self._child_body(j)
            jc = self.neutral_joint_center[j]
            d: dict[int, float] = {}
            for mi in range(self.topo.num_markers):
                b = int(m_body[mi])
                if b == cb or (pb >= 0 and b == pb):
                    d[mi] = float(
                        np.sum((jc - self.neutral_marker_world[mi]) ** 2)
                    )
            self.joint_markers.append(d)
            if len(d) >= self.min_markers:
                self.active_joints.append(j)

        # joint -> {other_joint: squared distance} for connected joint pairs
        self.joint_joint: dict[int, dict[int, float]] = {
            j: {} for j in self.active_joints
        }
        for a in range(len(self.active_joints)):
            j1 = self.active_joints[a]
            pb1, cb1 = int(self.topo.j_parent_body[j1]), self._child_body(j1)
            for b in range(a + 1, len(self.active_joints)):
                j2 = self.active_joints[b]
                pb2, cb2 = int(self.topo.j_parent_body[j2]), self._child_body(j2)
                if pb1 == cb2 or pb2 == cb1 or (pb1 == pb2 and pb1 >= 0):
                    dd = float(
                        np.sum(
                            (
                                self.neutral_joint_center[j1]
                                - self.neutral_joint_center[j2]
                            )
                            ** 2
                        )
                    )
                    self.joint_joint[j1][j2] = dd
                    self.joint_joint[j2][j1] = dd

    # ------------------------------------------------------------------
    # stage 1: closed-form MDS joint centers
    # ------------------------------------------------------------------
    def closed_form_mds_joint_centers(self) -> list[dict[int, np.ndarray]]:
        """Per-frame joint-center estimates by MDS triangulation.

        Port of ``closedFormMDSJointCenterSolver``. Returns a list (per frame) of
        ``{joint_index: (3,) world center}``. Iterates within a frame so that already-
        solved neighbor joint centers can help triangulate remaining joints (via the
        neutral joint->joint distances).
        """
        neutral_mk = self.neutral_marker_world
        neutral_jc = self.neutral_joint_center
        results: list[dict[int, np.ndarray]] = []

        for t in range(self.F):
            solved: dict[int, np.ndarray] = {}
            last_count = -1
            while len(solved) != last_count:
                last_count = len(solved)
                for j in self.active_joints:
                    pts: list[np.ndarray] = []
                    sq: list[float] = []
                    neutral_pts: list[np.ndarray] = []
                    # adjacent visible markers
                    for mi, d2 in self.joint_markers[j].items():
                        if self.visible[t, mi]:
                            pts.append(self.obs[t, mi])
                            sq.append(d2)
                            neutral_pts.append(neutral_mk[mi])
                    # adjacent already-solved joint centers
                    for j2, d2 in self.joint_joint[j].items():
                        if j2 in solved:
                            pts.append(solved[j2])
                            sq.append(d2)
                            neutral_pts.append(neutral_jc[j2])
                    if len(pts) < 3:
                        continue

                    n = len(pts)
                    dim = n + 1
                    D = np.zeros((dim, dim), dtype=np.float64)
                    P = np.asarray(pts)
                    # pairwise squared distances between the known points
                    diff = P[:, None, :] - P[None, :, :]
                    D[:n, :n] = np.sum(diff * diff, axis=2)
                    # last point (the unknown joint center) known squared distances
                    D[:n, n] = sq
                    D[n, :n] = sq

                    cloud = cf.point_cloud_from_distance_matrix(D)  # 3 x dim
                    transformed = cf.map_point_cloud_to_data(cloud, pts)  # 3 x dim
                    jc = transformed[:, n].copy()

                    # resolve coplanar ambiguity using the neutral skeleton geometry
                    if cf.is_coplanar(neutral_pts) or cf.is_coplanar(pts):
                        jc = cf.ensure_on_same_side_of_plane(
                            neutral_pts, neutral_jc[j], pts, jc
                        )
                    solved[j] = jc
            results.append(solved)

        self.joint_centers_per_frame = results
        return results

    # ------------------------------------------------------------------
    # stage 2: closed-form group scales
    # ------------------------------------------------------------------
    def estimate_prescale(self) -> float:
        """Global uniform scale from the observed vs model anatomical-marker span.

        Port of Nimble's initial *prescale* step: before the per-group anisotropic fit,
        estimate one isotropic scale so the model's overall size matches the subject.
        It is the median over all anatomical-marker pairs of
        ``median_t ||obs_i - obs_k|| / ||model_i - model_k||`` (robust to per-frame
        noise and to a handful of mislabeled markers). Used to seed weakly-observed /
        unobserved scale axes and groups instead of the naive 1.0, which conditions the
        subsequent closed-form and IK stages on real (scaled) subjects.
        """
        anat = [mi for mi in range(self.topo.num_markers) if self.anatomical[mi]]
        if len(anat) < 4:
            # this model may not flag anatomical landmarks (Nimble sets them from a
            # separate list); fall back to all markers -- the robust median of pair
            # ratios tolerates the extra soft-tissue markers for an isotropic prescale.
            anat = list(range(self.topo.num_markers))
        ratios: list[float] = []
        for a in range(len(anat)):
            i = anat[a]
            for b in range(a + 1, len(anat)):
                k = anat[b]
                dmodel = float(
                    np.linalg.norm(
                        self.neutral_marker_world[i] - self.neutral_marker_world[k]
                    )
                )
                if dmodel < 1e-4:
                    continue
                both = self.visible[:, i] & self.visible[:, k]
                if not both.any():
                    continue
                dobs = np.linalg.norm(self.obs[both, i] - self.obs[both, k], axis=1)
                ratios.append(float(np.median(dobs)) / dmodel)
        return float(np.median(ratios)) if ratios else 1.0

    def estimate_group_scales(
        self, default_scale: float | None = None, prescale: bool = True
    ) -> np.ndarray:
        """Anisotropic per-group scales from joint-center / anatomical-marker distances.

        Port of ``estimateGroupScalesClosedForm``. For each body, collect local points
        (adjacent joint centers + anatomical markers, in the neutral body frame),
        accumulate their average observed world-space pairwise distances across frames,
        and fit the body scale with :func:`closed_form.get_local_scale`. Body scales are
        condensed into the (symmetric) group-scale vector.

        When ``prescale`` is set (default), :meth:`estimate_prescale` provides the
        isotropic fallback for weakly-observed axes and for groups with no usable marker
        pairs, instead of the naive 1.0 -- this is the deferred prescale polishing step
        and only affects axes/groups the closed form could not otherwise determine.
        """
        if default_scale is None:
            default_scale = self.estimate_prescale() if prescale else 1.0
        if not hasattr(self, "joint_centers_per_frame"):
            self.closed_form_mds_joint_centers()
        jc_frames = self.joint_centers_per_frame

        # per-group accumulation (bodies in a group share a scale); average the
        # independently-fit body scales within each group for symmetry.
        group_scale_sum = np.zeros((self.num_groups, 3), dtype=np.float64)
        group_scale_cnt = np.zeros(self.num_groups, dtype=np.int64)

        for b in range(self.topo.num_bodies):
            # adjacent joints: parent joint (if not root) + child joints
            adj_joints: list[int] = []
            parent_joint = b  # joint whose child is body b
            if int(self.topo.j_parent_body[parent_joint]) >= 0:
                if parent_joint in self.active_joints:
                    adj_joints.append(parent_joint)
            for jj in self.active_joints:
                if int(self.topo.j_parent_body[jj]) == b:
                    adj_joints.append(jj)

            Tb_inv = np.linalg.inv(self.neutral_body_world[b])
            local_points: list[np.ndarray] = []
            for jj in adj_joints:
                jw = self.neutral_joint_center[jj]
                local_points.append((Tb_inv @ np.append(jw, 1.0))[:3])

            # anatomical markers on this body
            anat_markers: list[int] = []
            for mi in range(self.topo.num_markers):
                if int(self.topo.m_body[mi]) == b and self.anatomical[mi]:
                    anat_markers.append(mi)
                    local_points.append(self.topo.m_offset[mi])

            nP = len(local_points)
            if nP < 2:
                continue

            # accumulate observed world distances between all point pairs over frames
            n_adj = len(adj_joints)
            avg = np.zeros((nP, nP), dtype=np.float64)
            cnt = np.zeros((nP, nP), dtype=np.int64)

            def _world_pos(idx: int, t: int) -> np.ndarray | None:
                if idx < n_adj:
                    jj = adj_joints[idx]
                    return jc_frames[t].get(jj, None)
                mi = anat_markers[idx - n_adj]
                return self.obs[t, mi] if self.visible[t, mi] else None

            for t in range(self.F):
                pos = [_world_pos(i, t) for i in range(nP)]
                for i in range(nP):
                    pi = pos[i]
                    if pi is None:
                        continue
                    for k in range(i + 1, nP):
                        pk = pos[k]
                        if pk is None:
                            continue
                        dist = float(np.linalg.norm(pi - pk))
                        avg[i, k] += dist
                        cnt[i, k] += 1

            pairs: list[tuple[int, int, float, float]] = []
            for i in range(nP):
                for k in range(i + 1, nP):
                    if cnt[i, k] > 0:
                        pairs.append((i, k, avg[i, k] / cnt[i, k], 1.0))

            scale = cf.get_local_scale(
                local_points, pairs, default_axis_scale=default_scale
            )
            g = int(self.body_group[b])
            group_scale_sum[g] += scale
            group_scale_cnt[g] += 1

        group_scales = np.full(
            (self.num_groups, 3), float(default_scale), dtype=np.float64
        )
        nonzero = group_scale_cnt > 0
        group_scales[nonzero] = (
            group_scale_sum[nonzero] / group_scale_cnt[nonzero, None]
        )
        self.group_scales = group_scales.reshape(-1)
        return self.group_scales

    # ------------------------------------------------------------------
    # stage 3: poses (batched Warp marker IK)
    # ------------------------------------------------------------------
    def estimate_poses(
        self, config: MarkerIKConfig | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Per-frame poses via batched marker IK with the estimated group scales.

        Analogous to ``estimatePosesWithIK``. Seeds each frame's global translation
        from the root joint-center estimate (the dominant, otherwise poorly-conditioned
        DOF) and joint angles from neutral, then runs the Warp LM marker IK.
        """
        if not hasattr(self, "group_scales"):
            self.estimate_group_scales()
        if not hasattr(self, "joint_centers_per_frame"):
            self.closed_form_mds_joint_centers()

        ndof = self.topo.num_dofs
        q_init = np.zeros((self.F, ndof), dtype=np.float64)
        # Seed root translation (DOFs 3:6 for the EulerFree root) from the root joint
        # center estimate where available, else the visible-marker centroid.
        root_joint = 0
        for t in range(self.F):
            jc = self.joint_centers_per_frame[t].get(root_joint, None)
            if jc is not None:
                q_init[t, 3:6] = jc
            elif self.visible[t].any():
                q_init[t, 3:6] = self.obs[t, self.visible[t]].mean(axis=0)

        cfg = config or MarkerIKConfig(max_iters=200)
        res = solve_marker_ik(
            self.skel,
            self.obs,  # NaNs auto-masked
            q_init,
            group_scales=self.group_scales,
            config=cfg,
        )
        self.poses = res.q
        self.marker_rms = res.marker_rms
        return self.poses, self.marker_rms

    # ------------------------------------------------------------------
    def run(self, config: MarkerIKConfig | None = None) -> IKInitializerResult:
        """Run the full closed-form pipeline and return the seed."""
        self.closed_form_mds_joint_centers()
        self.estimate_group_scales()
        self.estimate_poses(config)

        # pack joint centers into (F, 3) arrays (NaN where unsolved)
        jc_out: dict[str, np.ndarray] = {}
        for j in self.active_joints:
            arr = np.full((self.F, 3), np.nan, dtype=np.float64)
            for t in range(self.F):
                if j in self.joint_centers_per_frame[t]:
                    arr[t] = self.joint_centers_per_frame[t][j]
            jc_out[self.joint_names[j]] = arr

        return IKInitializerResult(
            joint_centers=jc_out,
            group_scales=self.group_scales,
            poses=self.poses,
            marker_rms=self.marker_rms,
            joint_names=[self.joint_names[j] for j in self.active_joints],
        )
