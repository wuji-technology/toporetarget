"""The per-frame QP: the task and its solver, the costs and constraints, and the two stage programs."""

from __future__ import annotations

import ctypes
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

import daqp
import numpy as np
from scipy import sparse as sp
from scipy.spatial.transform import Rotation as RotLib

from toporetarget.solver.hand import WRIST_DOF, WRIST_ROTATION_SLICE, HandModelProtocol, ObjectAwareHandModel
from toporetarget.solver.landmarks import MEDIAPIPE_LANDMARK_COUNT, mediapipe_chain_edges
from toporetarget.solver.laplacian import LaplacianJacobians, LaplacianKeypoints
from toporetarget.solver.settings import HandRetargetConfig


@dataclass(frozen=True)
class CostTerm:
    """A quadratic objective term in ``0.5 dq^T H dq + c^T dq`` form."""

    H: np.ndarray
    c: np.ndarray

    @classmethod
    def zeros(cls, nq: int) -> CostTerm:
        return cls(H=np.zeros((nq, nq), dtype=float), c=np.zeros(nq, dtype=float))

    @classmethod
    def sum(cls, terms, *, nq: int) -> CostTerm:
        total = cls.zeros(nq)
        H = total.H.copy()
        c = total.c.copy()
        for term in terms:
            if term is None:
                continue
            H += term.H
            c += term.c
        return cls(H=H, c=c)


@dataclass(frozen=True)
class ConstraintSet:
    """Linear inequality constraints in solver-ready ``G dq <= h`` form."""

    G: np.ndarray | None = None
    h: np.ndarray | None = None

    @classmethod
    def none(cls) -> ConstraintSet:
        return cls()

    @classmethod
    def from_rows(cls, *, G: np.ndarray, h: np.ndarray) -> ConstraintSet:
        return cls(G=G, h=h)

    @classmethod
    def stack(cls, sets) -> ConstraintSet:
        sets = [s for s in sets if s.G is not None]
        if not sets:
            return cls.none()
        return cls(G=np.vstack([s.G for s in sets]), h=np.concatenate([s.h for s in sets]))


@dataclass(frozen=True)
class StepBounds:
    """Lower and upper bounds for the QP step ``dq``."""

    lb: np.ndarray
    ub: np.ndarray


@dataclass(frozen=True)
class QpTask:
    """A complete QP task consumed by ``solve_qp_task``."""

    cost: CostTerm
    bounds: StepBounds
    constraints: ConstraintSet = field(default_factory=ConstraintSet.none)

    @property
    def H(self) -> np.ndarray:
        return self.cost.H

    @property
    def c(self) -> np.ndarray:
        return self.cost.c

    @property
    def lb(self) -> np.ndarray:
        return self.bounds.lb

    @property
    def ub(self) -> np.ndarray:
        return self.bounds.ub

    @property
    def G(self) -> np.ndarray | None:
        return self.constraints.G

    @property
    def h(self) -> np.ndarray | None:
        return self.constraints.h


@dataclass(frozen=True)
class QpResult:
    """The result of a trust-region QP solve."""

    dq: np.ndarray
    n_shrinks: int
    found: bool


@dataclass(frozen=True)
class QpSolverOptions:
    """Explicit solver options; no retarget config leaks below this boundary."""

    max_trust_shrinks: int
    hybrid_enabled: bool
    hybrid_tolerance: float
    hybrid_backstop: float
    hybrid_slack_weight: float
    hybrid_slack_violated_only: bool = False


_WARNED_HYBRID_INFEASIBLE = False

_TRUST_REGION_SHRINK: float = 0.5


class QpTaskLike(Protocol):
    H: np.ndarray
    c: np.ndarray
    lb: np.ndarray
    ub: np.ndarray
    G: np.ndarray | None
    h: np.ndarray | None


_DAQP_INFEASIBLE = -1


def _daqp(
    H: np.ndarray, c: np.ndarray, G: np.ndarray | None, h: np.ndarray | None, lb: np.ndarray, ub: np.ndarray
) -> tuple[np.ndarray | None, int]:
    """``min 0.5 dq^T H dq + c^T dq`` s.t. ``G dq <= h``, ``lb <= dq <= ub``: ``(dq or None, daqp exit flag)``."""
    if G is None or not len(G):
        G, h = np.zeros((0, len(c))), np.zeros(0)
    upper = np.concatenate([ub, h])
    lower = np.concatenate([lb, np.full(len(h), -1e30)])
    x, _, flag, _ = daqp.solve(H, c, G, upper, lower, np.zeros(len(upper), dtype=ctypes.c_int))
    return (x if flag > 0 else None), int(flag)


def _trust_region_loop(
    qp_fn: Callable[[np.ndarray, np.ndarray], tuple[np.ndarray | None, int]],
    *,
    lb: np.ndarray,
    ub: np.ndarray,
    max_shrinks: int,
    shrink: Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]] | None = None,
    extract_dq: Callable[[np.ndarray], np.ndarray] | None = None,
    failure_dq: Callable[[], np.ndarray],
) -> QpResult:
    """Solve a QP under a shrinking trust region until it is found or the budget runs out."""
    if shrink is None:

        def shrink(lb_s: np.ndarray, ub_s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            return lb_s * _TRUST_REGION_SHRINK, ub_s * _TRUST_REGION_SHRINK

    lb_s, ub_s = lb.copy(), ub.copy()
    for k in range(max_shrinks + 1):
        x, flag = qp_fn(lb_s, ub_s)
        if x is not None:
            return QpResult(x if extract_dq is None else extract_dq(x), k, True)
        if flag == _DAQP_INFEASIBLE:
            break
        lb_s, ub_s = shrink(lb_s, ub_s)
    return QpResult(failure_dq(), k + 1, False)


def solve_qp_task(task: QpTaskLike, options: QpSolverOptions) -> QpResult:
    """Solve a QP task with the configured trust-region strategy."""
    G, h = task.G, task.h
    if G is not None and h is not None and len(G):
        keep = np.maximum(G * task.lb, G * task.ub).sum(axis=1) > h
        G, h = G[keep], h[keep]
    if options.hybrid_enabled and G is not None and h is not None and len(G):
        return _solve_qp_with_slack_hybrid(task.H, task.c, G, h, task.lb, task.ub, options)

    def qp_fn(lb_s: np.ndarray, ub_s: np.ndarray):
        return _daqp(task.H, task.c, G, h, lb_s, ub_s)

    return _trust_region_loop(
        qp_fn,
        lb=task.lb,
        ub=task.ub,
        max_shrinks=options.max_trust_shrinks,
        failure_dq=lambda: np.zeros_like(task.lb),
    )


def _solve_qp_with_slack_hybrid(
    H: np.ndarray,
    c: np.ndarray,
    G: np.ndarray,
    h: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
    options: QpSolverOptions,
) -> QpResult:
    """Augment each constraint row with a bounded, heavily penalized slack variable."""
    nq = H.shape[0]
    n_rows = G.shape[0]
    slack_rows = np.flatnonzero(h < 0.0) if options.hybrid_slack_violated_only else np.arange(n_rows)
    n_pen = len(slack_rows)
    w_slack = options.hybrid_slack_weight
    slack_max = max(options.hybrid_backstop - options.hybrid_tolerance, 0.0)

    H_aug = np.zeros((nq + n_pen, nq + n_pen))
    H_aug[:nq, :nq] = H
    if n_pen > 0:
        H_aug[nq:, nq:] = w_slack * np.eye(n_pen)

    c_aug = np.concatenate([c, np.zeros(n_pen)])

    G_aug = np.hstack([G, -np.eye(n_rows)[:, slack_rows]])

    lb_aug = np.concatenate([lb, np.zeros(n_pen)])
    ub_aug = np.concatenate([ub, slack_max * np.ones(n_pen)])

    def qp_fn(lb_s: np.ndarray, ub_s: np.ndarray):
        return _daqp(H_aug, c_aug, G_aug, h, lb_s, ub_s)

    def shrink(lb_s: np.ndarray, ub_s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        new_lb = lb_s.copy()
        new_lb[:nq] *= _TRUST_REGION_SHRINK
        return new_lb, ub_s * _TRUST_REGION_SHRINK

    result = _trust_region_loop(
        qp_fn,
        lb=lb_aug,
        ub=ub_aug,
        max_shrinks=options.max_trust_shrinks,
        shrink=shrink,
        extract_dq=lambda x: x[:nq],
        failure_dq=lambda: np.zeros(nq),
    )
    if not result.found:
        global _WARNED_HYBRID_INFEASIBLE
        if not _WARNED_HYBRID_INFEASIBLE:
            warnings.warn(
                "hybrid penetration QP infeasible after trust-shrink; returning dq=0.",
                stacklevel=2,
            )
            _WARNED_HYBRID_INFEASIBLE = True
    return result


def build_edge_direction_system(
    source_pts: np.ndarray,
    edges: np.ndarray,
    point_mapper,
    nq: int,
    palm_world_rot_inv: np.ndarray | None = None,
    edge_weights: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Cosine-IK residual and Jacobian rows over an arbitrary edge set."""
    eps = 1e-8
    if edges.shape[0] == 0:
        return np.zeros(0), np.zeros((0, nq))

    uniq_idx, inverse = np.unique(edges.reshape(-1), return_inverse=True)
    inverse = inverse.reshape(edges.shape)
    K = uniq_idx.shape[0]
    pos_uniq = np.empty((K, 3))
    jac_uniq = np.empty((K, 3, nq))
    for k in range(K):
        pos_uniq[k], jac_uniq[k] = point_mapper.robot_keypoint_and_jacobian(int(uniq_idx[k]))

    e_rob = pos_uniq[inverse[:, 1]] - pos_uniq[inverse[:, 0]]
    e_jac = jac_uniq[inverse[:, 1]] - jac_uniq[inverse[:, 0]]
    if palm_world_rot_inv is not None:
        e_rob = e_rob @ palm_world_rot_inv.T
        e_jac = np.einsum("ij,ejk->eik", palm_world_rot_inv, e_jac)
        e_jac[:, :, :WRIST_DOF] = 0.0
    e_len = np.linalg.norm(e_rob, axis=1)

    e_src = source_pts[edges[:, 1]] - source_pts[edges[:, 0]]
    s_len = np.linalg.norm(e_src, axis=1)

    valid = (e_len >= eps) & (s_len >= eps)
    if not valid.any():
        return np.zeros(0), np.zeros((0, nq))

    e_rob, e_jac, e_len = e_rob[valid], e_jac[valid], e_len[valid]
    e_src, s_len = e_src[valid], s_len[valid]

    d_rob = e_rob / e_len[:, None]
    d_src = e_src / s_len[:, None]

    eye3 = np.eye(3)
    d_outer = np.einsum("ei,ej->eij", d_rob, d_rob)
    proj = (eye3[None] - d_outer) / e_len[:, None, None]
    tangent_jac = np.einsum("eij,ejk->eik", proj, e_jac)

    w = edge_weights[valid].astype(np.float64) if edge_weights is not None else np.ones(d_rob.shape[0])
    residual = (w[:, None] * (d_rob - d_src)).reshape(-1)
    jacobian = (w[:, None, None] * tangent_jac).reshape(-1, nq)
    return residual, jacobian


def palm_world_rot_inv(q: np.ndarray, floating_base: bool) -> np.ndarray | None:
    """``R_wrist^T`` for rotating world-frame robot edges into the palm frame; None without a floating wrist."""
    if not floating_base:
        return None
    return RotLib.from_euler("XYZ", q[WRIST_ROTATION_SLICE]).as_matrix().T


@dataclass(frozen=True)
class StepBoundSpec:
    step_size: float


def build_step_bounds(
    q_current: np.ndarray,
    q_lb: np.ndarray,
    q_ub: np.ndarray,
    spec: StepBoundSpec,
) -> StepBounds:
    """The trust-region box intersected with the joint limits."""
    lb = np.maximum(-spec.step_size, q_lb - q_current)
    ub = np.minimum(spec.step_size, q_ub - q_current)
    return StepBounds(lb=lb, ub=ub)


@dataclass(frozen=True)
class PenetrationSpec:
    """Non-penetration settings: enable flag, tolerance, active-set query threshold."""

    enabled: bool
    tolerance: float
    query_threshold: float = 0.05


def build_penetration_constraints(hand, q: np.ndarray, spec: PenetrationSpec) -> ConstraintSet:
    """Linearized non-penetration rows for the current frame: ``phi(q + dq) >= -tolerance``."""
    if not spec.enabled:
        return ConstraintSet.none()
    if not isinstance(hand, ObjectAwareHandModel) or not hand.has_object:
        return ConstraintSet.none()

    rows = hand.query_hand_penetration(threshold=spec.query_threshold)
    if not rows:
        return ConstraintSet.none()

    G = np.vstack([(-r[0]).reshape(1, -1) for r in rows])
    h = np.array([r[1] + spec.tolerance for r in rows])
    return ConstraintSet.from_rows(G=G, h=h)


def measure_penetration_mm(hand, q: np.ndarray, *, query_threshold: float = 0.05) -> float:
    """Max hand-object penetration depth at ``q``, in millimetres, for metrics."""
    if not isinstance(hand, ObjectAwareHandModel) or not hand.has_object:
        return 0.0
    hand.forward(q)
    pen = [-phi for _, phi, _, _ in hand.query_hand_penetration(threshold=query_threshold) if phi < 0]
    return float((max(pen) if pen else 0.0) * 1000.0)


def _lock_at_zero(q_lb: np.ndarray, q_ub: np.ndarray) -> None:
    q_lb[:WRIST_DOF] = 0.0
    q_ub[:WRIST_DOF] = 0.0


def _intersect_with_native(q_lb: np.ndarray, q_ub: np.ndarray, config: HandRetargetConfig) -> None:
    limits = np.array(
        [config.wrist.translation_limit] * 3 + [config.wrist.rotation_limit] * 3,
        dtype=float,
    )
    q_lb[:WRIST_DOF] = np.maximum(q_lb[:WRIST_DOF], -limits)
    q_ub[:WRIST_DOF] = np.minimum(q_ub[:WRIST_DOF], limits)
    if np.any(q_lb[:WRIST_DOF] > q_ub[:WRIST_DOF]):
        raise ValueError("Configured wrist bounds do not intersect the model wrist joint ranges.")


def configure_wrist_bounds(q_lb: np.ndarray, q_ub: np.ndarray, config: HandRetargetConfig) -> None:
    """Apply the configured wrist bounds in place: locked at zero on a fixed base or with motion off."""
    if not (config.model.floating_base and config.wrist.motion_enabled):
        _lock_at_zero(q_lb, q_ub)
        return
    _intersect_with_native(q_lb, q_ub, config)


@dataclass(frozen=True)
class WristZeroSpec:
    enabled: bool
    translation_weight: float
    rotation_weight: float


def build_temporal_smoothness_cost(q_current: np.ndarray, q_prev: np.ndarray, *, weight: float) -> CostTerm:
    """Penalize deviation from the previous frame's joint state."""
    nq = len(q_current)
    return CostTerm(H=weight * np.eye(nq), c=-weight * (q_prev - q_current))


def build_wrist_zero_cost(q_current: np.ndarray, spec: WristZeroSpec) -> CostTerm:
    """A soft quadratic prior keeping the floating wrist offset near zero."""
    nq = len(q_current)
    cost = CostTerm.zeros(nq)
    if not spec.enabled:
        return cost

    weights = np.array([spec.translation_weight] * 3 + [spec.rotation_weight] * 3, dtype=float)
    idx = np.nonzero(weights > 0.0)[0]
    H = cost.H.copy()
    c = cost.c.copy()
    H[idx, idx] += weights[idx]
    c[idx] += weights[idx] * q_current[idx]
    return CostTerm(H=H, c=c)


def build_psd_jitter_cost(nq: int, eps: float = 1e-12) -> CostTerm:
    """A tiny positive diagonal term for numerical PSD stability."""
    return CostTerm(H=eps * np.eye(nq), c=np.zeros(nq))


@dataclass(frozen=True)
class BoneDirSpec:
    weight: float
    topology: str
    edge_weight_mode: str
    laplacian_edge_decay_k: float | None
    floating_base: bool


def bone_dir_edges_and_weights(
    adj_list: list[list[int]],
    source_pts: np.ndarray,
    *,
    topology: str,
    edge_weight_mode: str,
    lap_decay_k: float | None,
    n_hand: int | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """The edge set and per-edge sqrt-weights for a bone-direction term."""
    if topology == "chain":
        return mediapipe_chain_edges(), None

    cap = len(source_pts) if n_hand is None else min(n_hand, len(source_pts))
    edges_list: list[tuple[int, int]] = []
    for i, neighbors in enumerate(adj_list):
        if i >= cap:
            continue
        for j in neighbors:
            if i < j < cap:
                edges_list.append((i, j))
    if not edges_list:
        return np.empty((0, 2), dtype=np.int64), None

    edges = np.array(edges_list, dtype=np.int64)
    if edge_weight_mode == "uniform":
        return edges, None
    if edge_weight_mode == "inv_count":
        return edges, np.full(edges.shape[0], 1.0 / np.sqrt(edges.shape[0]))
    if edge_weight_mode == "laplacian_aligned":
        d = np.linalg.norm(source_pts[edges[:, 1]] - source_pts[edges[:, 0]], axis=1)
        k = lap_decay_k if lap_decay_k is not None else 0.0
        return edges, np.exp(-0.5 * k * d)
    raise ValueError(f"unknown edge_weight_mode: {edge_weight_mode!r}")


def build_bone_dir_cost(
    *,
    source_pts: np.ndarray,
    adjacency: list[list[int]],
    point_mapper,
    nq: int,
    spec: BoneDirSpec,
    q_current: np.ndarray,
    n_hand: int | None = None,
) -> CostTerm | None:
    """The bone-direction (cosine IK) objective over the resolved source edges."""
    edges, edge_weights = bone_dir_edges_and_weights(
        adjacency,
        source_pts,
        topology=spec.topology,
        edge_weight_mode=spec.edge_weight_mode,
        lap_decay_k=spec.laplacian_edge_decay_k,
        n_hand=n_hand,
    )
    r_bones, J_bones = build_edge_direction_system(
        source_pts,
        edges,
        point_mapper,
        nq,
        palm_world_rot_inv(q_current, spec.floating_base),
        edge_weights,
    )
    if r_bones.size == 0:
        return None
    return CostTerm(H=spec.weight * (J_bones.T @ J_bones), c=spec.weight * (J_bones.T @ r_bones))


@dataclass(frozen=True)
class LaplacianCostSpec:
    weight: float
    length_scale: float | None


def build_laplacian_cost(
    *,
    robot_keypoints: LaplacianKeypoints,
    robot_jacobians: LaplacianJacobians,
    target_laplacian: np.ndarray,
    source_laplacian_matrix: np.ndarray,
    spec: LaplacianCostSpec,
    nq: int,
) -> CostTerm:
    """The source-fixed interaction-mesh Laplacian objective."""
    robot_pts = robot_keypoints.all_points
    n_total = robot_keypoints.n_total
    J_V = robot_jacobians.all_jacobians(n_total=n_total, nq=nq)

    L_sp = sp.csr_matrix(source_laplacian_matrix)
    J_L = sp.kron(L_sp, sp.eye(3, format="csr"), format="csr") @ J_V
    J_L_dense = J_L.toarray() if sp.issparse(J_L) else J_L

    lap0 = L_sp @ robot_pts
    residual = lap0.reshape(-1) - target_laplacian.reshape(-1)
    sqrt_w3 = np.sqrt(np.repeat(spec.weight * np.ones(n_total), 3))
    if spec.length_scale is not None and spec.length_scale > 0:
        sqrt_w3 = sqrt_w3 / float(spec.length_scale)

    J_w = sqrt_w3[:, None] * J_L_dense
    r_w = sqrt_w3 * residual
    return CostTerm(H=J_w.T @ J_w, c=J_w.T @ r_w)


@dataclass(frozen=True)
class WarmupProgramSpec:
    bone_dir: BoneDirSpec
    smooth_weight: float
    bounds: StepBoundSpec
    penetration: PenetrationSpec
    wrist_zero: WristZeroSpec


@dataclass(frozen=True)
class WarmupProgramInput:
    q_current: np.ndarray
    q_prev: np.ndarray
    landmarks_21: np.ndarray
    adjacency: list[list[int]]
    q_lb: np.ndarray
    q_ub: np.ndarray


@dataclass
class WarmupProgram:
    """Stage 1: bone-direction cosine IK, assembled once per retargeting scope."""

    hand: HandModelProtocol
    point_mapper: Any
    nq: int
    spec: WarmupProgramSpec

    def build_task(self, inputs: WarmupProgramInput) -> QpTask | None:
        """One warmup QP task from the current frame state."""
        self.hand.forward(inputs.q_current)
        bone_dir_cost = build_bone_dir_cost(
            source_pts=inputs.landmarks_21,
            adjacency=inputs.adjacency,
            point_mapper=self.point_mapper,
            nq=self.nq,
            spec=self.spec.bone_dir,
            q_current=inputs.q_current,
        )
        if bone_dir_cost is None:
            return None

        cost = CostTerm.sum(
            [
                build_temporal_smoothness_cost(inputs.q_current, inputs.q_prev, weight=self.spec.smooth_weight),
                bone_dir_cost,
                build_wrist_zero_cost(inputs.q_current, self.spec.wrist_zero),
                build_psd_jitter_cost(self.nq),
            ],
            nq=self.nq,
        )
        bounds = build_step_bounds(inputs.q_current, inputs.q_lb, inputs.q_ub, self.spec.bounds)
        constraints = build_penetration_constraints(self.hand, inputs.q_current, self.spec.penetration)
        return QpTask(cost=cost, bounds=bounds, constraints=constraints)


def _first_21(landmarks: np.ndarray) -> np.ndarray:
    if landmarks.shape[0] > MEDIAPIPE_LANDMARK_COUNT:
        return landmarks[:MEDIAPIPE_LANDMARK_COUNT]
    return landmarks


@dataclass(frozen=True)
class ObjectFrameTransform:
    """The world -> object-local transform used when ``object.use_frame`` is on."""

    R_inv: np.ndarray
    t_obj: np.ndarray

    def points_to_local(self, points: np.ndarray) -> np.ndarray:
        return (points - self.t_obj) @ self.R_inv.T

    def jacobians_to_local(self, jacobians: np.ndarray, *, n_points: int, nq: int) -> np.ndarray:
        blocks = jacobians.reshape(n_points, 3, nq)
        return np.einsum("ij,njk->nik", self.R_inv, blocks).reshape(n_points * 3, nq)


@dataclass(frozen=True)
class MeshSqpProgramSpec:
    laplacian: LaplacianCostSpec
    smooth_weight: float
    bounds: StepBoundSpec
    penetration: PenetrationSpec
    wrist_zero: WristZeroSpec
    pose_prior: BoneDirSpec | None


@dataclass(frozen=True)
class MeshSqpProgramInput:
    q_current: np.ndarray
    q_prev: np.ndarray
    adjacency: list[list[int]]
    target_laplacian: np.ndarray
    source_laplacian_matrix: np.ndarray
    object_points: np.ndarray | None
    object_frame: ObjectFrameTransform | None
    landmarks: np.ndarray
    q_lb: np.ndarray
    q_ub: np.ndarray


@dataclass
class MeshSqpProgram:
    """Stage 2: the interaction-mesh Laplacian SQP, assembled once per scope."""

    hand: HandModelProtocol
    point_mapper: Any
    nq: int
    n_keypoints: int
    spec: MeshSqpProgramSpec
    finger_clearance: Callable[[np.ndarray], ConstraintSet]

    def build_task(self, inputs: MeshSqpProgramInput) -> QpTask:
        """One interaction-mesh SQP task from the current frame state."""
        self.hand.forward(inputs.q_current)
        robot_keypoints, robot_jacobians = self.robot_laplacian_inputs(
            object_points=inputs.object_points,
            object_frame=inputs.object_frame,
        )
        laplacian_cost = build_laplacian_cost(
            robot_keypoints=robot_keypoints,
            robot_jacobians=robot_jacobians,
            target_laplacian=inputs.target_laplacian,
            source_laplacian_matrix=inputs.source_laplacian_matrix,
            spec=self.spec.laplacian,
            nq=self.nq,
        )
        cost = CostTerm.sum(
            [
                laplacian_cost,
                build_temporal_smoothness_cost(inputs.q_current, inputs.q_prev, weight=self.spec.smooth_weight),
                self.pose_prior_term(inputs),
                build_wrist_zero_cost(inputs.q_current, self.spec.wrist_zero),
                build_psd_jitter_cost(self.nq),
            ],
            nq=self.nq,
        )
        bounds = build_step_bounds(inputs.q_current, inputs.q_lb, inputs.q_ub, self.spec.bounds)
        constraints = build_penetration_constraints(self.hand, inputs.q_current, self.spec.penetration)
        constraints = ConstraintSet.stack([constraints, self.finger_clearance(inputs.q_current)])
        return QpTask(cost=cost, bounds=bounds, constraints=constraints)

    def robot_laplacian_inputs(
        self,
        *,
        object_points: np.ndarray | None,
        object_frame: ObjectFrameTransform | None,
    ) -> tuple[LaplacianKeypoints, LaplacianJacobians]:
        """Robot keypoints and Jacobians expressed in the active Laplacian frame."""
        hand_pts = self.point_mapper.robot_keypoints()
        J_hand = self.point_mapper.robot_jacobians()

        if object_frame is not None:
            hand_pts = object_frame.points_to_local(hand_pts)
            J_hand = object_frame.jacobians_to_local(J_hand, n_points=self.n_keypoints, nq=self.nq)

        return (
            LaplacianKeypoints(hand_points=hand_pts, obj_points=object_points),
            LaplacianJacobians(hand_jacobians=J_hand),
        )

    def pose_prior_term(self, inputs: MeshSqpProgramInput) -> CostTerm | None:
        """The optional bone-direction pose prior."""
        if self.spec.pose_prior is None or self.spec.pose_prior.weight <= 0:
            return None
        return build_bone_dir_cost(
            source_pts=_first_21(inputs.landmarks),
            adjacency=inputs.adjacency,
            point_mapper=self.point_mapper,
            nq=self.nq,
            spec=self.spec.pose_prior,
            q_current=inputs.q_current,
            n_hand=self.n_keypoints,
        )
