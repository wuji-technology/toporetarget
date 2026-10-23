"""The retargeter: keypoint mapping, per-frame topology, the frame optimizer and the runtime they assemble into."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
from scipy.spatial.transform import Rotation as RotLib

from toporetarget.solver.hand import HandModelProtocol, HandSpec, ObjectAwareHandModel
from toporetarget.solver.landmarks import MEDIAPIPE_LANDMARK_COUNT
from toporetarget.solver.laplacian import (
    Connections,
    LaplacianKeypoints,
    build_delaunay_connections,
    build_skeleton_connections,
    build_skeleton_plus_sphere_connections,
    calculate_laplacian_matrix,
)
from toporetarget.solver.model import MuJoCoFloatingHandModel
from toporetarget.solver.qp import (
    BoneDirSpec,
    LaplacianCostSpec,
    MeshSqpProgram,
    MeshSqpProgramInput,
    MeshSqpProgramSpec,
    ObjectFrameTransform,
    PenetrationSpec,
    QpSolverOptions,
    QpTask,
    StepBoundSpec,
    WarmupProgram,
    WarmupProgramInput,
    WarmupProgramSpec,
    WristZeroSpec,
    configure_wrist_bounds,
    measure_penetration_mm,
    solve_qp_task,
)
from toporetarget.solver.sampling import sample_uniform_object_points_from_mesh
from toporetarget.solver.selfcol import FingerClearance
from toporetarget.solver.settings import HandRetargetConfig

_NON_THUMB_MCP_MP: frozenset[int] = frozenset({5, 9, 13, 17})

_NORMAL_DEGENERATE_EPS: float = 1e-8


class PointMapper:
    """Resolve MediaPipe landmarks to source and robot keypoints (``HandSpec.keypoint_targets``)."""

    def __init__(self, config: HandRetargetConfig, hand, spec: HandSpec) -> None:
        self.config = config
        self.hand = hand
        self.spec = spec
        self.joints_mapping = spec.keypoint_targets()
        self.mp_indices = sorted(self.joints_mapping.keys())
        self.n_keypoints = len(self.mp_indices)
        self._robot_keypoint_resolver = self._build_robot_keypoint_resolver()

    def source_keypoints(self, landmarks: np.ndarray) -> np.ndarray:
        """The mapped source keypoints of one 21-landmark frame."""
        return landmarks[self.mp_indices]

    def robot_keypoint_and_jacobian(self, mediapipe_index: int) -> tuple[np.ndarray, np.ndarray]:
        """The robot keypoint mapped to one MediaPipe landmark, plus its Jacobian."""
        body_id = self._robot_keypoint_resolver[mediapipe_index]
        position = self.hand.get_body_pos_by_id(body_id)
        jacobian = self.hand.get_body_jacp_by_id(body_id)

        offset = self.config.point_mapping.mcp_surface_offset_m
        if mediapipe_index in _NON_THUMB_MCP_MP and offset != 0.0:
            position = position + offset * self._palm_back_normal()
        return position, jacobian

    def robot_keypoints(self) -> np.ndarray:
        """Current robot keypoint positions, ``(n_keypoints, 3)``."""
        pts = np.empty((self.n_keypoints, 3))
        for k, mediapipe_index in enumerate(self.mp_indices):
            pts[k], _ = self.robot_keypoint_and_jacobian(mediapipe_index)
        return pts

    def robot_jacobians(self) -> np.ndarray:
        """Stacked translational Jacobians for the current robot keypoints."""
        J = np.zeros((3 * self.n_keypoints, self.hand.nq))
        for k, mediapipe_index in enumerate(self.mp_indices):
            _, J[3 * k : 3 * k + 3] = self.robot_keypoint_and_jacobian(mediapipe_index)
        return J

    def _palm_back_normal(self) -> np.ndarray:
        """Unit palm normal pointing toward the back of the hand."""
        mapping = self.joints_mapping
        p_wrist = self.hand.get_body_pos(mapping[0])
        p_idx = self.hand.get_body_pos(mapping[5])
        p_pinky = self.hand.get_body_pos(mapping[17])
        n = np.cross(p_idx - p_wrist, p_pinky - p_wrist)
        if self.spec.side == "right":
            n = -n
        norm = np.linalg.norm(n)
        return n / norm if norm > _NORMAL_DEGENERATE_EPS else np.array([0.0, 0.0, 1.0])

    def _build_robot_keypoint_resolver(self) -> dict[int, int]:
        """Body (or site) id per keypoint, with the MCP / thumb CMC surrogates applied."""
        point_mapping = self.config.point_mapping
        surrogates = self.spec.keypoint_surrogates()
        resolver: dict[int, int] = {}
        for mediapipe_index, body in self.joints_mapping.items():
            if mediapipe_index == 1 and point_mapping.thumb_cmc_surrogate == "link2":
                body = surrogates[mediapipe_index]
            elif mediapipe_index in _NON_THUMB_MCP_MP and point_mapping.mcp_surrogate == "link2":
                body = surrogates[mediapipe_index]
            resolver[mediapipe_index] = self.hand.get_body_id(body)
        return resolver


@dataclass(frozen=True)
class ObjectFrameSpec:
    """Optional object keypoints and frame for Laplacian topology construction."""

    world_points: np.ndarray | None = None
    frame: tuple[np.ndarray, np.ndarray] | None = None
    local_points: np.ndarray | None = None


@dataclass(frozen=True)
class LaplacianTopologyOptions:
    """Topology and Laplacian-matrix configuration for one frame."""

    n_keypoints: int
    topology_mode: str
    delaunay_edge_threshold: float | None
    skeleton_sphere_radius: float
    laplacian_distance_weight_k: float | None
    use_delaunay_edge_threshold: bool = True
    laplacian_distance_weight_mode: str | None = None
    laplacian_distance_weight_cutoff_radius: float | None = None
    object_frame: ObjectFrameSpec = ObjectFrameSpec()


def _select_connections(
    keypoints: LaplacianKeypoints,
    *,
    topology_mode: str,
    delaunay_edge_threshold: float | None,
    skeleton_sphere_radius: float,
    use_delaunay_edge_threshold: bool = True,
) -> Connections:
    if topology_mode == "delaunay":
        edge_threshold = delaunay_edge_threshold if use_delaunay_edge_threshold else None
        return build_delaunay_connections(keypoints, edge_threshold=edge_threshold)
    if topology_mode == "skeleton":
        return build_skeleton_connections(keypoints)
    if topology_mode == "skeleton_plus_sphere":
        return build_skeleton_plus_sphere_connections(keypoints, radius=skeleton_sphere_radius)
    raise ValueError(f"Unknown topology mode: {topology_mode!r}")


def _build_keypoints(source_pts: np.ndarray, object_frame: ObjectFrameSpec) -> LaplacianKeypoints:
    world_points = object_frame.world_points
    if world_points is not None and len(world_points) == 0:
        world_points = None

    frame = object_frame.frame
    local_points = object_frame.local_points

    if frame is not None and local_points is not None:
        r_inv, t_obj = frame
        return LaplacianKeypoints(hand_points=(source_pts - t_obj) @ r_inv.T, obj_points=local_points)
    if world_points is not None:
        return LaplacianKeypoints(hand_points=source_pts, obj_points=world_points)
    return LaplacianKeypoints(hand_points=source_pts)


def build_laplacian_topology(
    source_pts: np.ndarray,
    options: LaplacianTopologyOptions,
) -> tuple[LaplacianKeypoints, Connections, np.ndarray, np.ndarray]:
    """Keypoints, connections, the source-fixed Laplacian target and its matrix."""
    keypoints = _build_keypoints(source_pts, options.object_frame)
    if keypoints.n_hand != options.n_keypoints:
        raise ValueError(f"Expected {options.n_keypoints} hand keypoints, got {keypoints.n_hand}")

    connections = _select_connections(
        keypoints,
        topology_mode=options.topology_mode,
        delaunay_edge_threshold=options.delaunay_edge_threshold,
        skeleton_sphere_radius=options.skeleton_sphere_radius,
        use_delaunay_edge_threshold=options.use_delaunay_edge_threshold,
    )

    source_laplacian_matrix = calculate_laplacian_matrix(
        keypoints.all_points,
        connections.adjacency,
        distance_decay_k=options.laplacian_distance_weight_k,
        distance_weight_mode=options.laplacian_distance_weight_mode,
        distance_weight_cutoff_radius=options.laplacian_distance_weight_cutoff_radius,
    )
    target_lap = source_laplacian_matrix @ keypoints.all_points
    return keypoints, connections, target_lap, source_laplacian_matrix


class FrameOptimizer:
    """Run generic frame-level QP loops and collect per-frame metrics."""

    def __init__(
        self,
        hand: HandModelProtocol,
        *,
        solver_options: QpSolverOptions,
        penetration_metric: Callable[[HandModelProtocol, np.ndarray], float] | None = None,
    ) -> None:
        self.hand = hand
        self.solver_options = solver_options
        self.nq = hand.nq
        self.penetration_metric = penetration_metric
        self.frame_np_metrics: dict[str, float | int | bool] = {}
        self._shrinks_this_frame: dict[str, int] = {}
        self._stall_iters: dict[str, int] = {}

    def begin_frame(self) -> None:
        """Reset the per-frame metrics before running the QP loops."""
        self._shrinks_this_frame = {}
        self._stall_iters = {}
        self.frame_np_metrics = {}

    def register_metrics_key(self, metrics_key: str) -> None:
        """Ensure ``metrics_key`` emits zero counters even if its loop is skipped."""
        self._shrinks_this_frame.setdefault(metrics_key, 0)
        self._stall_iters.setdefault(metrics_key, 0)

    def run_qp_loop(
        self,
        q_current: np.ndarray,
        *,
        max_iters: int,
        convergence_delta: float,
        q_lb: np.ndarray,
        q_ub: np.ndarray,
        metrics_key: str,
        build_task: Callable[[np.ndarray], QpTask | None],
    ) -> np.ndarray:
        """Run a bounded QP loop, accumulating metrics under ``metrics_key``."""
        self.register_metrics_key(metrics_key)
        for _ in range(max_iters):
            q_before = q_current.copy()
            task = build_task(q_current)
            if task is None:
                break

            result = solve_qp_task(task, self.solver_options)
            self._shrinks_this_frame[metrics_key] += result.n_shrinks
            if not result.found:
                self._stall_iters[metrics_key] += 1
                break

            q_raw = q_current + result.dq
            q_clipped = np.clip(q_raw, q_lb, q_ub)
            kkt_slack = float(np.abs(q_clipped - q_raw).max())
            if kkt_slack >= 1e-5:
                raise RuntimeError(
                    f"QP joint-limit KKT violation {kkt_slack:.2e} > 1e-5 "
                    "(solver jitter is ~1e-7; large slack means the step bounds are wrong)"
                )
            q_current = q_clipped
            if np.linalg.norm(q_current - q_before) < convergence_delta:
                break
        return q_current

    def finish_frame(self, q_current: np.ndarray) -> None:
        """Collect the per-frame metrics once the QP loops have finished."""
        final_pen_max_mm = 0.0
        if self.penetration_metric is not None:
            final_pen_max_mm = self.penetration_metric(self.hand, q_current)

        metrics: dict[str, float | int | bool] = {}
        for key in self._shrinks_this_frame:
            metrics[f"{key}_shrinks"] = self._shrinks_this_frame[key]
            metrics[f"{key}_stalls"] = self._stall_iters[key]
        metrics["final_pen_max_mm"] = final_pen_max_mm
        metrics["struct_infeas"] = any(count > 0 for count in self._stall_iters.values())
        self.frame_np_metrics = metrics


@dataclass
class RetargetRuntime:
    config: HandRetargetConfig
    hand: HandModelProtocol
    spec: HandSpec
    point_mapper: PointMapper
    optimizer: FrameOptimizer
    q_lb: np.ndarray
    q_ub: np.ndarray
    nq: int
    n_keypoints: int
    mp_indices: list[int]


@dataclass(frozen=True)
class ObjectFrameInputs:
    """Per-frame object inputs fed to ``retarget_frame``."""

    world: np.ndarray | None = None
    frame: tuple[np.ndarray, np.ndarray] | None = None
    local: np.ndarray | None = None
    pose_position: np.ndarray | None = None
    pose_quaternion_xyzw: np.ndarray | None = None


@dataclass(frozen=True)
class FramePointSetup:
    landmarks: np.ndarray
    object_inputs: ObjectFrameInputs
    source_keypoints: np.ndarray
    object_points: np.ndarray | None = None
    object_frame: ObjectFrameTransform | None = None


@dataclass(frozen=True)
class FrameTopology:
    adjacency: list[list[int]]
    target_laplacian: np.ndarray
    source_laplacian_matrix: np.ndarray


def transform_object_points(
    points_local: np.ndarray,
    position: np.ndarray,
    quaternion_xyzw: np.ndarray,
) -> np.ndarray:
    """Transform object-local points into the frame where the object has that pose."""
    rot = RotLib.from_quat(quaternion_xyzw).as_matrix()
    return (rot @ points_local.T).T + position


def build_object_frame_inputs(
    points_local: Sequence[np.ndarray] | None,
    positions: np.ndarray,
    quaternions_xyzw: np.ndarray,
    *,
    use_object_frame: bool,
) -> ObjectFrameInputs:
    """One frame's object payload for the topology and optimization stages."""
    positions = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
    quaternions_xyzw = np.asarray(quaternions_xyzw, dtype=np.float64).reshape(-1, 4)
    if not points_local:
        return ObjectFrameInputs(pose_position=positions, pose_quaternion_xyzw=quaternions_xyzw)

    world = np.concatenate(
        [
            transform_object_points(points, pos, quat)
            for points, pos, quat in zip(points_local, positions, quaternions_xyzw, strict=True)
        ]
    )
    if use_object_frame:
        if len(points_local) != 1:
            raise ValueError(f"object.use_frame needs a single object, the clip has {len(points_local)}")
        rot = RotLib.from_quat(quaternions_xyzw[0]).as_matrix()
        return ObjectFrameInputs(
            world=world,
            frame=(rot.T, positions[0]),
            local=points_local[0],
            pose_position=positions,
            pose_quaternion_xyzw=quaternions_xyzw,
        )
    return ObjectFrameInputs(
        world=world,
        pose_position=positions,
        pose_quaternion_xyzw=quaternions_xyzw,
    )


def create_runtime(config: HandRetargetConfig, spec: HandSpec) -> RetargetRuntime:
    """Create the hand model, point mapping and optimizer for a retargeter."""
    hand = MuJoCoFloatingHandModel(spec)
    point_mapper = PointMapper(config, hand, spec)
    runtime = RetargetRuntime(
        config=config,
        hand=hand,
        spec=spec,
        point_mapper=point_mapper,
        optimizer=cast(FrameOptimizer, None),
        q_lb=hand.q_lb,
        q_ub=hand.q_ub,
        nq=hand.nq,
        n_keypoints=point_mapper.n_keypoints,
        mp_indices=point_mapper.mp_indices,
    )
    runtime.optimizer = build_optimizer(runtime)
    return runtime


def build_optimizer(runtime: RetargetRuntime) -> FrameOptimizer:
    radius = runtime.config.mesh.interaction_radius_m
    return FrameOptimizer(
        runtime.hand,
        solver_options=build_solver_options(runtime.config),
        penetration_metric=lambda hand, q: measure_penetration_mm(hand, q, query_threshold=radius),
    )


def build_solver_options(config: HandRetargetConfig) -> QpSolverOptions:
    return QpSolverOptions(
        max_trust_shrinks=config.penetration.max_trust_shrinks,
        hybrid_enabled=config.penetration.hybrid.enabled,
        hybrid_tolerance=config.penetration.tolerance,
        hybrid_backstop=config.penetration.hybrid.backstop,
        hybrid_slack_weight=config.penetration.hybrid.slack_weight,
    )


def build_stage_programs(runtime: RetargetRuntime) -> tuple[WarmupProgram, MeshSqpProgram]:
    """Construct the QP program objects once for a retargeting scope."""
    return build_warmup_program(runtime), build_mesh_sqp_program(runtime)


def build_warmup_program(runtime: RetargetRuntime) -> WarmupProgram:
    """The Stage 1 bone-direction warmup program."""
    cfg = runtime.config
    return WarmupProgram(
        hand=runtime.hand,
        point_mapper=runtime.point_mapper,
        nq=runtime.nq,
        spec=WarmupProgramSpec(
            bone_dir=BoneDirSpec(
                weight=cfg.warmup.weight,
                topology=cfg.warmup.topology,
                edge_weight_mode=cfg.warmup.edge_weight_mode,
                laplacian_edge_decay_k=cfg.mesh.edge_weight.exp_decay_k,
                floating_base=cfg.model.floating_base,
            ),
            smooth_weight=cfg.smooth.weight,
            bounds=StepBoundSpec(step_size=cfg.solver.step_size),
            penetration=PenetrationSpec(
                enabled=cfg.penetration.enabled_warmup,
                tolerance=cfg.penetration.tolerance,
                query_threshold=cfg.mesh.interaction_radius_m,
            ),
            wrist_zero=_wrist_zero_spec(runtime),
        ),
    )


def build_mesh_sqp_program(runtime: RetargetRuntime) -> MeshSqpProgram:
    """The Stage 2 interaction-mesh SQP program."""
    cfg = runtime.config
    pose_prior = None
    if cfg.pose_prior.bone_dir_weight > 0:
        pose_prior = BoneDirSpec(
            weight=cfg.pose_prior.bone_dir_weight,
            topology=cfg.pose_prior.topology,
            edge_weight_mode=cfg.pose_prior.edge_weight_mode,
            laplacian_edge_decay_k=cfg.mesh.edge_weight.exp_decay_k,
            floating_base=cfg.model.floating_base,
        )

    return MeshSqpProgram(
        hand=runtime.hand,
        point_mapper=runtime.point_mapper,
        nq=runtime.nq,
        n_keypoints=runtime.n_keypoints,
        finger_clearance=FingerClearance(runtime.spec, cfg.selfcol, runtime.nq),
        spec=MeshSqpProgramSpec(
            laplacian=LaplacianCostSpec(
                weight=cfg.mesh.laplacian_weight,
                length_scale=cfg.mesh.length_scale,
            ),
            smooth_weight=cfg.smooth.weight,
            bounds=StepBoundSpec(step_size=cfg.solver.step_size),
            penetration=PenetrationSpec(
                enabled=cfg.penetration.enabled_s2,
                tolerance=cfg.penetration.tolerance,
                query_threshold=cfg.mesh.interaction_radius_m,
            ),
            wrist_zero=_wrist_zero_spec(runtime),
            pose_prior=pose_prior,
        ),
    )


def _wrist_zero_spec(runtime: RetargetRuntime) -> WristZeroSpec:
    cfg = runtime.config
    return WristZeroSpec(
        enabled=cfg.model.floating_base and cfg.wrist.motion_enabled,
        translation_weight=cfg.wrist.zero_translation_weight,
        rotation_weight=cfg.wrist.zero_rotation_weight,
    )


class InteractionMeshHandRetargeter:
    """Retarget MediaPipe-21 landmarks to the joint trajectory of a ``HandSpec`` hand."""

    def __init__(self, config: HandRetargetConfig, spec: HandSpec) -> None:
        self.runtime = create_runtime(config, spec)

    def _sync_object_pose_for_penetration(self, object_inputs: ObjectFrameInputs) -> None:
        """Update the object-aware hand state the penetration queries read."""
        if not self.runtime.config.penetration.enabled_any:
            return
        if not isinstance(self.runtime.hand, ObjectAwareHandModel) or not self.runtime.hand.has_object:
            return
        if object_inputs.pose_position is None or object_inputs.pose_quaternion_xyzw is None:
            return
        self.runtime.hand.set_object_poses(object_inputs.pose_position, object_inputs.pose_quaternion_xyzw)

    def retarget_frame(
        self,
        landmarks: np.ndarray,
        q_prev: np.ndarray,
        *,
        is_first_frame: bool = False,
        object_inputs: ObjectFrameInputs | None = None,
        q_lb: np.ndarray | None = None,
        q_ub: np.ndarray | None = None,
        warmup_program: WarmupProgram | None = None,
        mesh_sqp_program: MeshSqpProgram | None = None,
    ) -> np.ndarray:
        """Retarget one preprocessed landmark frame; returns its qpos."""
        if object_inputs is None:
            object_inputs = ObjectFrameInputs()
        self._sync_object_pose_for_penetration(object_inputs)
        if q_lb is None:
            q_lb = self.runtime.q_lb
        if q_ub is None:
            q_ub = self.runtime.q_ub
        if warmup_program is None or mesh_sqp_program is None:
            default_warmup, default_mesh_sqp = build_stage_programs(self.runtime)
            warmup_program = warmup_program or default_warmup
            mesh_sqp_program = mesh_sqp_program or default_mesh_sqp

        points = self.setup_frame_points(landmarks, object_inputs)
        topology = self.setup_frame_laplacian(points)

        self.runtime.optimizer.begin_frame()
        self.runtime.optimizer.register_metrics_key("warmup")
        self.runtime.optimizer.register_metrics_key("s2")
        q_warm = self.run_stage1_warmup(
            q_prev.copy(),
            q_prev,
            points,
            topology,
            is_first_frame=is_first_frame,
            q_lb=q_lb,
            q_ub=q_ub,
            warmup_program=warmup_program,
        )
        q_opt = self.run_stage2_mesh_sqp(
            q_warm,
            q_prev,
            points,
            topology,
            is_first_frame=is_first_frame,
            q_lb=q_lb,
            q_ub=q_ub,
            mesh_sqp_program=mesh_sqp_program,
        )
        self.runtime.optimizer.finish_frame(q_opt)
        return q_opt

    def setup_frame_points(
        self,
        landmarks: np.ndarray,
        object_inputs: ObjectFrameInputs | None = None,
    ) -> FramePointSetup:
        """Build the frame point inputs the topology and optimization stages consume."""
        if object_inputs is None:
            object_inputs = ObjectFrameInputs()
        object_frame = None
        object_points = object_inputs.world
        if object_inputs.frame is not None:
            object_frame = ObjectFrameTransform(R_inv=object_inputs.frame[0], t_obj=object_inputs.frame[1])
            object_points = object_inputs.local

        return FramePointSetup(
            landmarks=landmarks,
            object_inputs=object_inputs,
            source_keypoints=self.runtime.point_mapper.source_keypoints(landmarks),
            object_points=object_points,
            object_frame=object_frame,
        )

    def setup_frame_laplacian(self, points: FramePointSetup) -> FrameTopology:
        """Build the interaction graph and the source Laplacian for one frame."""
        cfg = self.runtime.config.mesh
        _keypoints, connections, target_lap, source_laplacian_matrix = build_laplacian_topology(
            points.source_keypoints,
            LaplacianTopologyOptions(
                n_keypoints=self.runtime.n_keypoints,
                topology_mode=cfg.topology,
                delaunay_edge_threshold=cfg.interaction_radius_m,
                skeleton_sphere_radius=cfg.skeleton_sphere_radius,
                laplacian_distance_weight_k=cfg.edge_weight.exp_decay_k,
                use_delaunay_edge_threshold=cfg.delaunay.use_edge_threshold,
                laplacian_distance_weight_mode=cfg.edge_weight.mode,
                laplacian_distance_weight_cutoff_radius=cfg.edge_weight.cutoff_radius,
                object_frame=ObjectFrameSpec(
                    world_points=points.object_inputs.world,
                    frame=points.object_inputs.frame,
                    local_points=points.object_inputs.local,
                ),
            ),
        )
        return FrameTopology(
            adjacency=connections.adjacency,
            target_laplacian=target_lap,
            source_laplacian_matrix=source_laplacian_matrix,
        )

    def run_stage1_warmup(
        self,
        q_current: np.ndarray,
        q_prev: np.ndarray,
        points: FramePointSetup,
        laplacian: FrameTopology,
        *,
        is_first_frame: bool,
        q_lb: np.ndarray,
        q_ub: np.ndarray,
        warmup_program: WarmupProgram,
    ) -> np.ndarray:
        """Stage 1: the cosine-IK warmup, under the sequence-local bounds."""
        cfg = self.runtime.config
        if not cfg.warmup.enabled:
            return q_current

        max_iters = cfg.warmup.iters_first if is_first_frame else cfg.warmup.iters
        landmarks_21 = points.landmarks[:MEDIAPIPE_LANDMARK_COUNT]

        return self.runtime.optimizer.run_qp_loop(
            q_current,
            max_iters=max_iters,
            convergence_delta=cfg.solver.warmup_convergence_delta,
            q_lb=q_lb,
            q_ub=q_ub,
            metrics_key="warmup",
            build_task=lambda q: warmup_program.build_task(
                WarmupProgramInput(
                    q_current=q,
                    q_prev=q_prev,
                    landmarks_21=landmarks_21,
                    adjacency=laplacian.adjacency,
                    q_lb=q_lb,
                    q_ub=q_ub,
                )
            ),
        )

    def run_stage2_mesh_sqp(
        self,
        q_current: np.ndarray,
        q_prev: np.ndarray,
        points: FramePointSetup,
        laplacian: FrameTopology,
        *,
        is_first_frame: bool,
        q_lb: np.ndarray,
        q_ub: np.ndarray,
        mesh_sqp_program: MeshSqpProgram,
    ) -> np.ndarray:
        """Stage 2: the interaction-mesh SQP, under the sequence-local bounds."""
        cfg = self.runtime.config
        max_iters = cfg.solver.n_iter_first if is_first_frame else cfg.solver.n_iter

        return self.runtime.optimizer.run_qp_loop(
            q_current,
            max_iters=max_iters,
            convergence_delta=cfg.solver.s2_convergence_delta,
            q_lb=q_lb,
            q_ub=q_ub,
            metrics_key="s2",
            build_task=lambda q: mesh_sqp_program.build_task(
                MeshSqpProgramInput(
                    q_current=q,
                    q_prev=q_prev,
                    adjacency=laplacian.adjacency,
                    target_laplacian=laplacian.target_laplacian,
                    source_laplacian_matrix=laplacian.source_laplacian_matrix,
                    object_points=points.object_points,
                    object_frame=points.object_frame,
                    landmarks=points.landmarks,
                    q_lb=q_lb,
                    q_ub=q_ub,
                )
            ),
        )


@dataclass(frozen=True)
class RetargeterAssets:
    """What a retargeter is built from: the hand and the object meshes."""

    hand: HandSpec
    object_mesh_paths: tuple[Path, ...] = ()


@dataclass(frozen=True)
class PreparedRetargeter:
    """A fully assembled retargeter: solver, joint bounds, stage programs, object points."""

    retargeter: InteractionMeshHandRetargeter
    q_lb: np.ndarray
    q_ub: np.ndarray
    warmup_program: WarmupProgram
    mesh_sqp_program: MeshSqpProgram
    object_points_local: list[np.ndarray]
    effective_config: HandRetargetConfig


def build_prepared_retargeter(config: HandRetargetConfig, assets: RetargeterAssets) -> PreparedRetargeter:
    """Build a retargeter plus the one-time assets for stateful hand-object streaming."""
    cfg = dataclasses.replace(config)
    cfg.revalidate_after_override()

    object_meshes = [Path(path) for path in assets.object_mesh_paths]
    if cfg.penetration.enabled_any and not object_meshes:
        raise ValueError("object meshes are required when penetration is enabled")
    for object_mesh in object_meshes:
        if not object_mesh.exists():
            raise FileNotFoundError(f"object mesh not found: {object_mesh}")

    retargeter = InteractionMeshHandRetargeter(cfg, assets.hand)
    q_lb = retargeter.runtime.q_lb.copy()
    q_ub = retargeter.runtime.q_ub.copy()
    if cfg.penetration.enabled_any and object_meshes and isinstance(retargeter.runtime.hand, ObjectAwareHandModel):
        retargeter.runtime.hand.inject_object_meshes([str(path) for path in object_meshes])
        q_lb = retargeter.runtime.hand.q_lb.copy()
        q_ub = retargeter.runtime.hand.q_ub.copy()

    configure_wrist_bounds(q_lb, q_ub, cfg)
    warmup_program, mesh_sqp_program = build_stage_programs(retargeter.runtime)

    object_points_local = [
        sample_uniform_object_points_from_mesh(
            str(object_mesh),
            sample_count=cfg.object.sample_count,
            seed=cfg.object.sampler.seed,
        )
        for object_mesh in object_meshes
    ]

    return PreparedRetargeter(
        retargeter=retargeter,
        q_lb=q_lb,
        q_ub=q_ub,
        warmup_program=warmup_program,
        mesh_sqp_program=mesh_sqp_program,
        object_points_local=object_points_local,
        effective_config=cfg,
    )
