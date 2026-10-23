"""The canonical dataset every TopoRetarget stage reads, and the helpers that write one."""

from __future__ import annotations

import contextlib
import json
import os
import shutil
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np

from toporetarget.config import CANONICAL_SCHEMA, safe_name

TARGET_FPS = 50.0
MEDIAPIPE_LANDMARK_COUNT = 21
MESH_NAME = "mesh.stl"
MAX_MESH_FACES = 200_000
ASSET_META_NAME = "asset.json"
MANIFEST_NAME = "manifest.json"
HANDEDNESS = ("left", "right")


class DatasetError(ValueError):
    """Source data that cannot become a canonical clip or asset."""


MIN_QUATERNION_NORM = 1e-12


def normalize_quaternion(value: np.ndarray, *, label: str = "quaternion") -> np.ndarray:
    """Normalize wxyz quaternions, rejecting non-finite or degenerate rows."""
    quaternion = np.asarray(value, dtype=np.float64)
    if quaternion.shape == () or quaternion.shape[-1] != 4:
        raise DatasetError(f"{label}: expected trailing quaternion dimension 4")
    norm = np.linalg.norm(quaternion, axis=-1, keepdims=True)
    if not np.isfinite(quaternion).all() or np.any(norm < MIN_QUATERNION_NORM):
        raise DatasetError(f"{label}: non-finite, zero-length, or near-zero quaternion")
    return quaternion / norm


def slerp_quaternion_wxyz(first: np.ndarray, second: np.ndarray, fraction: float) -> np.ndarray:
    """Interpolate two wxyz quaternions along the shortest spherical arc."""
    first_unit = normalize_quaternion(first, label="first quaternion")
    second_unit = normalize_quaternion(second, label="second quaternion")
    if first_unit.shape != (4,) or second_unit.shape != (4,):
        raise DatasetError("quaternion SLERP expects two individual quaternions")
    if not np.isfinite(fraction):
        raise DatasetError("quaternion SLERP fraction must be finite")

    dot = float(np.dot(first_unit, second_unit))
    if dot < 0.0:
        second_unit = -second_unit
        dot = -dot
    dot = float(np.clip(dot, -1.0, 1.0))
    if dot > 0.9995:
        return normalize_quaternion(
            first_unit + fraction * (second_unit - first_unit),
            label="interpolated quaternion",
        )
    angle = np.arccos(dot)
    return (np.sin((1.0 - fraction) * angle) * first_unit + np.sin(fraction * angle) * second_unit) / np.sin(angle)


def _normalized_source_times(source_times: np.ndarray, *, length: int | None = None) -> np.ndarray:
    source = np.asarray(source_times, dtype=np.float64)
    if source.ndim != 1 or source.shape[0] == 0:
        raise DatasetError("source_times must be one non-empty vector")
    if length is not None and source.shape[0] != length:
        raise DatasetError(f"source_times length {source.shape[0]} != values length {length}")
    if not np.isfinite(source).all() or np.any(np.diff(source) <= 0.0):
        raise DatasetError("source_times must be finite and strictly increasing")
    return source - source[0]


def resample_times_from_samples(source_times: np.ndarray, target_dt: float) -> np.ndarray:
    """Target sample times for a strictly increasing, possibly irregular source."""
    source = _normalized_source_times(source_times)
    if target_dt <= 0.0:
        raise DatasetError("target_dt must be positive")
    if source.shape[0] == 1:
        return np.zeros(1, dtype=np.float64)
    times = np.arange(0.0, source[-1], target_dt, dtype=np.float64)
    return times if times.shape[0] else np.zeros(1, dtype=np.float64)


def resample_linear_at(values: np.ndarray, times: np.ndarray, source_times: np.ndarray) -> np.ndarray:
    """Linearly interpolate a frame-major array from explicit sample times."""
    array = np.asarray(values)
    source = _normalized_source_times(source_times, length=array.shape[0])
    if array.shape[0] == 1:
        return np.repeat(array[:1], times.shape[0], axis=0).astype(np.float32)
    flat = array.reshape(array.shape[0], -1).astype(np.float64)
    output = np.empty((times.shape[0], flat.shape[1]), dtype=np.float32)
    for column in range(flat.shape[1]):
        output[:, column] = np.interp(times, source, flat[:, column])
    return output.reshape(times.shape[0], *array.shape[1:])


def resample_quat_at(values: np.ndarray, times: np.ndarray, source_times: np.ndarray) -> np.ndarray:
    """SLERP a wxyz quaternion track from explicit sample times."""
    quat = normalize_quaternion(np.asarray(values, dtype=np.float64))
    source = _normalized_source_times(source_times, length=quat.shape[0])
    if quat.shape[0] == 1:
        return np.repeat(quat[:1], times.shape[0], axis=0).astype(np.float32)
    output = np.empty((times.shape[0], 4), dtype=np.float64)
    for index, time_value in enumerate(times):
        if time_value <= source[0]:
            output[index] = quat[0]
            continue
        if time_value >= source[-1]:
            output[index] = quat[-1]
            continue
        right = int(np.searchsorted(source, time_value, side="left"))
        left = right - 1
        tau = (time_value - source[left]) / (source[right] - source[left])
        output[index] = slerp_quaternion_wxyz(quat[left], quat[right], tau)
    return normalize_quaternion(output).astype(np.float32)


def resample_pose_at(values: np.ndarray, times: np.ndarray, source_times: np.ndarray) -> np.ndarray:
    """Resample ``[xyz, wxyz]`` from explicit sample times: lerp plus SLERP."""
    pose = np.asarray(values)
    if pose.ndim != 2 or pose.shape[1] != 7:
        raise DatasetError(f"pose arrays must have shape (frames, 7), got {pose.shape}")
    position = resample_linear_at(pose[:, :3], times, source_times)
    quaternion = resample_quat_at(pose[:, 3:7], times, source_times)
    return np.concatenate([position, quaternion], axis=-1).astype(np.float32)


def zero_initial_object_xy(
    landmarks_w: np.ndarray,
    object_pose_w: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Translate one world-frame trajectory so its first object's initial XY is the origin."""
    landmarks = np.asarray(landmarks_w)
    object_pose = np.asarray(object_pose_w)
    if landmarks.ndim != 4 or landmarks.shape[2:] != (MEDIAPIPE_LANDMARK_COUNT, 3):
        raise DatasetError("landmarks_w must have shape (T, H, 21, 3)")
    if object_pose.ndim != 3 or object_pose.shape[0] != len(landmarks) or object_pose.shape[2] != 7:
        raise DatasetError("object_pose_w must have shape (T, K, 7)")

    finite_frames = np.flatnonzero(np.isfinite(object_pose[:, 0, :2]).all(axis=1))
    if finite_frames.size == 0:
        return landmarks.copy(), object_pose.copy()
    anchor_xy = object_pose[int(finite_frames[0]), 0, :2].copy()

    zeroed_landmarks = landmarks.copy()
    zeroed_object_pose = object_pose.copy()
    zeroed_landmarks[..., :2] -= anchor_xy
    zeroed_object_pose[..., :2] -= anchor_xy
    return zeroed_landmarks, zeroed_object_pose


def write_json(path: Path, value: Any) -> None:
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


@contextlib.contextmanager
def staged_dataset(out: str | Path, *, force: bool = False) -> Iterator[Path]:
    """Yield an empty staging root that replaces ``out`` in one rename on success."""
    out = Path(out).expanduser().resolve()
    if out.exists() and not force:
        raise FileExistsError(f"{out} already exists (pass --force to replace it)")
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".toporetarget-import-", dir=out.parent) as tmp:
        stage = Path(tmp) / out.name
        (stage / "motions").mkdir(parents=True)
        yield stage
        if out.exists():
            shutil.rmtree(out)
        os.replace(stage, out)


def write_clip(
    stage: Path,
    clip_id: str,
    landmarks_w: np.ndarray,
    object_pose_w: np.ndarray,
    meta: dict[str, Any],
) -> dict[str, Any]:
    """Write one clip's npz and meta.json; returns the meta as written."""
    if clip_id != safe_name(clip_id):
        raise DatasetError(f"clip id {clip_id!r} is not filesystem-safe")
    landmarks = np.asarray(landmarks_w, dtype=np.float32)
    object_pose = np.asarray(object_pose_w, dtype=np.float32)
    hands, objects = list(meta.get("hands") or []), list(meta.get("objects") or [])
    if landmarks.ndim != 4 or landmarks.shape[2:] != (MEDIAPIPE_LANDMARK_COUNT, 3) or len(landmarks) < 2:
        raise DatasetError(f"{clip_id}: landmarks must have shape (T >= 2, H, 21, 3), got {landmarks.shape}")
    if object_pose.shape != (len(landmarks), len(objects), 7):
        raise DatasetError(f"{clip_id}: object_pose_w must have shape (T, {len(objects)}, 7), got {object_pose.shape}")
    if not np.isfinite(landmarks).all() or not np.isfinite(object_pose).all():
        raise DatasetError(f"{clip_id}: landmarks and object pose must be finite")
    if not hands or len(set(hands)) != len(hands) or not set(hands) <= set(HANDEDNESS):
        raise DatasetError(f"{clip_id}: hands must be distinct entries of {HANDEDNESS}, got {hands!r}")
    if landmarks.shape[1] != len(hands):
        raise DatasetError(f"{clip_id}: {landmarks.shape[1]} landmark tracks for hands {hands!r}")
    if not objects or len(set(objects)) != len(objects):
        raise DatasetError(f"{clip_id}: objects must be distinct asset names, got {objects!r}")

    np.savez_compressed(
        stage / "motions" / f"{clip_id}.npz",
        mediapipe_landmarks_w=landmarks,
        object_pose_w=object_pose,
        fps=np.asarray(TARGET_FPS, dtype=np.float64),
    )
    written = {
        **meta,
        "schema": CANONICAL_SCHEMA,
        "clip_id": clip_id,
        "fps": TARGET_FPS,
        "frames": int(len(landmarks)),
    }
    write_json(stage / "motions" / f"{clip_id}.meta.json", written)
    return written
