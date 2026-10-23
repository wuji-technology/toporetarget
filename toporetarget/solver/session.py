"""``FrameSolver``: the per-frame solver, driven one ``HandObjectState`` at a time."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as RotLib

from toporetarget.solver.hand import WRIST_DOF, HandSpec
from toporetarget.solver.landmarks import MEDIAPIPE_LANDMARK_COUNT
from toporetarget.solver.retargeter import (
    InteractionMeshHandRetargeter,
    PreparedRetargeter,
    RetargeterAssets,
    build_object_frame_inputs,
    build_prepared_retargeter,
)
from toporetarget.solver.settings import HandRetargetConfig

_QUAT_EPS = 1.0e-12


def _finite(name: str, arr, shape: tuple[int, ...]) -> np.ndarray:
    a = np.asarray(arr, dtype=np.float64)
    if a.shape != shape:
        raise ValueError(f"{name}: expected shape {shape}, got {a.shape}")
    if not np.isfinite(a).all():
        raise ValueError(f"{name}: contains non-finite values")
    return a


def _finite_quat(name: str, value) -> np.ndarray:
    """A finite, normalized xyzw quaternion (a float32 input is not exactly unit length)."""
    quat = _finite(name, value, (4,))
    norm = float(np.linalg.norm(quat))
    if norm <= _QUAT_EPS:
        raise ValueError(f"{name} must be a non-zero xyzw quaternion")
    return quat / norm


@dataclass(frozen=True)
class HandObjectState:
    """One frame of input: 21 landmarks + wrist pose + object poses, all world frame (xyzw)."""

    hand_landmarks_world: np.ndarray
    wrist_t_world: np.ndarray
    wrist_q_world: np.ndarray
    object_t_world: np.ndarray
    object_q_world: np.ndarray


@dataclass(frozen=True)
class FrameState:
    """One frame of output: the robot state (world wrist pose + joint angles)."""

    qpos: np.ndarray
    finger_qpos: np.ndarray
    wrist_t_world: np.ndarray
    wrist_q_world: np.ndarray
    object_t_world: np.ndarray
    object_q_world: np.ndarray
    metrics: dict


def validate_hand_object_state(state: HandObjectState) -> HandObjectState:
    """Validate and normalize one frame of input; non-finite values fail here."""
    n_objects = len(np.asarray(state.object_t_world).reshape(-1, 3))
    object_q = _finite("object_q_world", state.object_q_world, (n_objects, 4))
    norms = np.linalg.norm(object_q, axis=1, keepdims=True)
    if np.any(norms <= _QUAT_EPS):
        raise ValueError("object_q_world must hold non-zero xyzw quaternions")
    return HandObjectState(
        hand_landmarks_world=_finite("hand_landmarks_world", state.hand_landmarks_world, (MEDIAPIPE_LANDMARK_COUNT, 3)),
        wrist_t_world=_finite("wrist_t_world", state.wrist_t_world, (3,)),
        wrist_q_world=_finite_quat("wrist_q_world", state.wrist_q_world),
        object_t_world=_finite("object_t_world", state.object_t_world, (n_objects, 3)),
        object_q_world=object_q / norms,
    )


def world_to_palm(state: HandObjectState) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """World frame -> wrist (palm) local frame."""
    origin = state.wrist_t_world
    r_palm_to_world = RotLib.from_quat(state.wrist_q_world).as_matrix()
    landmarks_palm = (state.hand_landmarks_world - origin) @ r_palm_to_world
    object_t_palm = (state.object_t_world - origin) @ r_palm_to_world
    object_q_palm = np.array(
        [
            RotLib.from_matrix(r_palm_to_world.T @ RotLib.from_quat(q).as_matrix()).as_quat()
            for q in state.object_q_world
        ]
    ).reshape(-1, 4)
    return landmarks_palm, object_t_palm, object_q_palm


def compose_retargeted_wrist_world(wrist_t_world, wrist_q_world, qpos) -> tuple[np.ndarray, np.ndarray]:
    """Compose the solver's wrist 6-DoF correction back into a world pose ``(t, q_xyzw)``."""
    q = np.asarray(qpos, dtype=np.float64)
    if q.ndim != 1 or q.shape[0] <= WRIST_DOF or not np.isfinite(q).all():
        raise ValueError(f"qpos must be a finite vector of {WRIST_DOF} wrist DoF plus the finger joints")
    t_in = _finite("wrist_t_world", wrist_t_world, (3,))
    r_input = RotLib.from_quat(_finite_quat("wrist_q_world", wrist_q_world)).as_matrix()
    solver_r = RotLib.from_euler("XYZ", q[3:6]).as_matrix()
    world_t = t_in + r_input @ q[:3]
    world_q = RotLib.from_matrix(r_input @ solver_r).as_quat()
    return world_t, world_q


class FrameSolver:
    """A stateful single-frame session; all once-per-session setup happens in the constructor."""

    def __init__(
        self,
        prepared: PreparedRetargeter,
        *,
        object_points_local: list[np.ndarray] | None = None,
    ) -> None:
        self._prepared = prepared
        self._retargeter = prepared.retargeter
        self._q_lb = np.asarray(prepared.q_lb, dtype=np.float64).copy()
        self._q_ub = np.asarray(prepared.q_ub, dtype=np.float64).copy()
        self._warmup_program = prepared.warmup_program
        self._mesh_sqp_program = prepared.mesh_sqp_program
        self._object_points_local = prepared.object_points_local if object_points_local is None else object_points_local
        self._q_prev = self._default_qpos()
        self._frame_idx = 0

    @classmethod
    def from_assets(
        cls,
        config: HandRetargetConfig,
        assets: RetargeterAssets,
        *,
        object_points_local: list[np.ndarray] | None = None,
    ) -> FrameSolver:
        """Build a session from the scene / object assets."""
        return cls(build_prepared_retargeter(config, assets), object_points_local=object_points_local)

    @classmethod
    def from_paths(
        cls,
        config: HandRetargetConfig,
        hand: HandSpec,
        object_meshes: Sequence[str | Path] = (),
        *,
        object_points_local: list[np.ndarray] | None = None,
    ) -> FrameSolver:
        """A session for one hand against the objects of the given meshes."""
        return cls.from_assets(
            config,
            RetargeterAssets(hand=hand, object_mesh_paths=tuple(Path(path) for path in object_meshes)),
            object_points_local=object_points_local,
        )

    @property
    def retargeter(self) -> InteractionMeshHandRetargeter:
        """The underlying retargeter, for diagnostics that need runtime internals."""
        return self._retargeter

    @property
    def nq(self) -> int:
        """The hand model's configuration size: the 6 wrist DoF plus the finger joints."""
        return int(self._retargeter.runtime.hand.nq)

    @property
    def object_points_local(self) -> list[np.ndarray]:
        """The interaction-mesh samples, one ``(N, 3)`` array per object in its own frame."""
        return self._object_points_local

    def step(self, state: HandObjectState) -> FrameState:
        """Solve one frame: world-frame landmarks + wrist + object -> robot state."""
        state = validate_hand_object_state(state)
        landmarks_palm, object_t_palm, object_q_palm = world_to_palm(state)
        qpos = self._solve_palm(landmarks_palm, object_t_palm, object_q_palm)
        wrist_t_world, wrist_q_world = compose_retargeted_wrist_world(state.wrist_t_world, state.wrist_q_world, qpos)
        return FrameState(
            qpos=qpos.copy(),
            finger_qpos=qpos[WRIST_DOF:].copy(),
            wrist_t_world=wrist_t_world,
            wrist_q_world=wrist_q_world,
            object_t_world=state.object_t_world.copy(),
            object_q_world=state.object_q_world.copy(),
            metrics=dict(self._retargeter.runtime.optimizer.frame_np_metrics),
        )

    def step_palm(self, landmarks_palm, object_t_palm, object_q_palm) -> np.ndarray:
        """Solve one frame whose input is already in the palm frame; returns the qpos."""
        return self._solve_palm(landmarks_palm, object_t_palm, object_q_palm)

    def _solve_palm(self, landmarks_palm, object_t_palm, object_q_palm) -> np.ndarray:
        """The one per-frame primitive: palm-frame input -> qpos, advancing ``q_prev``."""
        object_inputs = build_object_frame_inputs(
            self._object_points_local,
            object_t_palm,
            object_q_palm,
            use_object_frame=self._retargeter.runtime.config.object.use_frame,
        )
        qpos = self._retargeter.retarget_frame(
            landmarks_palm,
            self._q_prev,
            is_first_frame=self._frame_idx == 0,
            object_inputs=object_inputs,
            q_lb=self._q_lb,
            q_ub=self._q_ub,
            warmup_program=self._warmup_program,
            mesh_sqp_program=self._mesh_sqp_program,
        )
        qpos = _finite("qpos", qpos, (self.nq,))
        self._q_prev = qpos.copy()
        self._frame_idx += 1
        return qpos

    def _default_qpos(self) -> np.ndarray:
        hand = self._retargeter.runtime.hand
        return _finite("default qpos", hand.get_default_qpos(), (int(hand.nq),))
