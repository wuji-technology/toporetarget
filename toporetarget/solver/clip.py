"""The input: canonical clips on disk, the wrist pose derived from their landmarks, and palm-frame tracks."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation as RotLib

from toporetarget.config import CANONICAL_SCHEMA
from toporetarget.solver.hand import HandSpec
from toporetarget.solver.landmarks import MEDIAPIPE_LANDMARK_COUNT, estimate_frame_from_hand_points


@dataclass(frozen=True)
class ObjectPoseTrack:
    """Per-frame object pose in one explicitly documented frame."""

    position: np.ndarray
    quaternion_xyzw: np.ndarray


@dataclass(frozen=True)
class RetargetObjectTrack:
    """Object data needed by the retargeter."""

    mesh_path: str | None
    pose: ObjectPoseTrack


@dataclass(frozen=True)
class RetargetInputTrack:
    """Per-frame hand/object inputs already expressed in the retarget (palm) frame."""

    landmarks: np.ndarray
    objects: tuple[RetargetObjectTrack, ...] = ()


def build_retarget_track(
    landmarks_world: np.ndarray,
    object_position_world: np.ndarray,
    object_quaternion_world_xyzw: np.ndarray,
    wrist_quaternion_world_xyzw: np.ndarray,
    *,
    mesh_paths: Sequence[str],
) -> RetargetInputTrack:
    """Convert decoded world-frame tracks into retarget-frame (palm-frame) data."""
    n_frames = len(landmarks_world)
    n_objects = len(mesh_paths)
    landmarks = np.empty_like(np.asarray(landmarks_world, dtype=np.float64))
    object_position = np.empty((n_frames, n_objects, 3), dtype=np.float64)
    object_quaternion = np.empty((n_frames, n_objects, 4), dtype=np.float64)

    for idx in range(n_frames):
        origin = landmarks_world[idx, 0]
        align = RotLib.from_quat(wrist_quaternion_world_xyzw[idx]).as_matrix()
        landmarks[idx] = (landmarks_world[idx] - origin) @ align
        for k in range(n_objects):
            object_position[idx, k] = (object_position_world[idx, k] - origin) @ align
            object_rot_world = RotLib.from_quat(object_quaternion_world_xyzw[idx, k]).as_matrix()
            object_quaternion[idx, k] = RotLib.from_matrix(align.T @ object_rot_world).as_quat()

    return RetargetInputTrack(
        landmarks=landmarks,
        objects=tuple(
            RetargetObjectTrack(
                mesh_path=mesh_path,
                pose=ObjectPoseTrack(position=object_position[:, k], quaternion_xyzw=object_quaternion[:, k]),
            )
            for k, mesh_path in enumerate(mesh_paths)
        ),
    )


REQUIRED_CLIP_KEYS = ("mediapipe_landmarks_w", "object_pose_w")
OBJECT_MESH_NAME = "mesh.stl"


@dataclass(frozen=True)
class ClipPaths:
    """The resolved filesystem paths of one canonical clip."""

    clip_id: str
    npz_path: Path
    meta_path: Path
    assets_dir: Path


@dataclass(frozen=True)
class CanonicalClip:
    """One hand of a decoded canonical clip, already sliced to the requested frame range."""

    clip_id: str
    side: str
    fps: float
    object_names: tuple[str, ...]
    object_meshes: tuple[Path, ...]
    object_dirs: tuple[Path, ...]
    landmarks_world: np.ndarray
    object_pose_world: np.ndarray
    wrist_pose_world: np.ndarray
    start_frame: int
    end_frame: int
    source_frames: int
    meta: dict[str, Any]

    @property
    def n_frames(self) -> int:
        return int(len(self.landmarks_world))


def resolve_clip_paths(clip: str | Path, dataset_root: str | Path) -> ClipPaths:
    """Resolve a clip id, or an explicit ``.npz`` path, to its companion paths."""
    clip_s = str(clip)
    root = Path(dataset_root)

    if clip_s.endswith(".npz"):
        npz_path = Path(clip_s)
        if not npz_path.exists():
            raise FileNotFoundError(f"clip path does not exist: {clip_s}")
        clip_id = npz_path.stem
        if npz_path.parent.name == "motions":
            root = npz_path.parent.parent
    else:
        if not clip_s or Path(clip_s).name != clip_s or clip_s in {".", ".."}:
            raise ValueError(f"clip id must be a filename stem, not a path: {clip_s!r}")
        clip_id = clip_s
        npz_path = root / "motions" / f"{clip_id}.npz"

    meta_path = npz_path.with_suffix(".meta.json")
    if not npz_path.exists():
        raise FileNotFoundError(f"clip not found: {npz_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"clip meta not found: {meta_path} (expected next to {npz_path.name})")
    return ClipPaths(clip_id=clip_id, npz_path=npz_path, meta_path=meta_path, assets_dir=root / "assets")


def derive_input_wrist_pose(landmarks_world: np.ndarray, hand: HandSpec) -> np.ndarray:
    """``(T, 7)`` ``[xyz, qw qx qy qz]`` wrist poses derived from the MediaPipe landmarks."""
    landmarks = np.asarray(landmarks_world, dtype=np.float64)
    if landmarks.ndim != 3 or landmarks.shape[1:] != (MEDIAPIPE_LANDMARK_COUNT, 3):
        raise ValueError(f"landmarks_world must have shape (T, 21, 3), got {landmarks.shape}")
    output = np.empty((landmarks.shape[0], 7), dtype=np.float64)
    output[:, :3] = landmarks[:, 0, :]
    alignment = hand.mano_alignment
    for frame, points in enumerate(landmarks):
        centered = points - points[0]
        palm_to_world = estimate_frame_from_hand_points(centered) @ alignment
        quat_xyzw = RotLib.from_matrix(palm_to_world).as_quat()
        output[frame, 3:7] = quat_xyzw[[3, 0, 1, 2]]
    return output


def load_canonical_clip(
    paths: ClipPaths,
    hand: HandSpec,
    *,
    start: int = 0,
    end: int | None = None,
    max_frames: int | None = None,
) -> CanonicalClip:
    """Decode the robot hand's side of a canonical clip and derive its wrist track."""
    meta = json.loads(paths.meta_path.read_text())
    schema = str(meta.get("schema", ""))
    if schema != CANONICAL_SCHEMA:
        raise ValueError(f"{paths.meta_path}: format {schema!r}, expected {CANONICAL_SCHEMA!r}; re-import the dataset")

    hands = [str(name) for name in meta["hands"]]
    side = hand.side
    if side not in hands:
        raise ValueError(f"{paths.meta_path}: the clip has no {side} hand (it has {', '.join(hands)})")

    with np.load(paths.npz_path, allow_pickle=False) as data:
        missing = [key for key in REQUIRED_CLIP_KEYS if key not in data.files]
        if missing:
            raise ValueError(f"{paths.npz_path}: missing required keys {missing}")
        all_landmarks = np.asarray(data["mediapipe_landmarks_w"], dtype=np.float64)
        object_pose = np.asarray(data["object_pose_w"], dtype=np.float64)
        fps = float(np.asarray(data["fps"]).item()) if "fps" in data.files else float(meta["fps"])

    source_frames = len(all_landmarks)
    if all_landmarks.shape != (source_frames, len(hands), MEDIAPIPE_LANDMARK_COUNT, 3):
        raise ValueError(f"{paths.npz_path}: mediapipe_landmarks_w {all_landmarks.shape} does not match hands {hands}")
    landmarks = all_landmarks[:, hands.index(side)]
    object_names = tuple(str(name) for name in meta["objects"])
    if object_pose.shape != (source_frames, len(object_names), 7):
        raise ValueError(
            f"{paths.npz_path}: object_pose_w {object_pose.shape} != ({source_frames}, {len(object_names)}, 7)"
        )
    if not np.isfinite(landmarks).all() or not np.isfinite(object_pose).all():
        raise ValueError(f"{paths.npz_path}: landmarks and object pose must be finite (run the importer first)")

    start_frame = max(int(start), 0)
    end_frame = source_frames if end is None else min(int(end), source_frames)
    if max_frames is not None and max_frames > 0:
        end_frame = min(end_frame, start_frame + int(max_frames))
    if end_frame - start_frame < 2:
        raise ValueError(
            f"{paths.npz_path}: need at least 2 frames, got {end_frame - start_frame} "
            f"for range [{start_frame}, {end_frame})"
        )
    landmarks = landmarks[start_frame:end_frame]
    object_pose = object_pose[start_frame:end_frame]

    object_dirs = tuple((paths.assets_dir / name).resolve() for name in object_names)
    object_meshes = tuple(directory / OBJECT_MESH_NAME for directory in object_dirs)
    for object_mesh in object_meshes:
        if not object_mesh.is_file():
            raise FileNotFoundError(f"object mesh not found: {object_mesh}")

    return CanonicalClip(
        clip_id=paths.clip_id,
        side=side,
        fps=fps,
        object_names=object_names,
        object_meshes=object_meshes,
        object_dirs=object_dirs,
        landmarks_world=landmarks,
        object_pose_world=object_pose,
        wrist_pose_world=derive_input_wrist_pose(landmarks, hand),
        start_frame=start_frame,
        end_frame=end_frame,
        source_frames=source_frames,
        meta=meta,
    )
