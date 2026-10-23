#!/usr/bin/env python3
"""Import a keypoint capture bundle into a canonical TopoRetarget dataset.

Examples:
    toporetarget import-capture --source /path/to/bundle --out data/my_capture
    toporetarget import-capture --dataset my_capture   # an entry of motion_dataset_registry.local.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from toporetarget.config import recorded_path, safe_name
from toporetarget.datasets.canonical import (
    ASSET_META_NAME,
    CANONICAL_SCHEMA,
    MANIFEST_NAME,
    MEDIAPIPE_LANDMARK_COUNT,
    MESH_NAME,
    TARGET_FPS,
    DatasetError,
    resample_linear_at,
    resample_pose_at,
    resample_times_from_samples,
    staged_dataset,
    write_clip,
    write_json,
    zero_initial_object_xy,
)

LOG_PREFIX = "Import"
REGISTRY_NAME = "motion_dataset_registry.local.json"
LANDMARK_KEYS = ("mediapipe_landmarks_w", "hand_landmarks_w")
SOURCE_MESH_NAMES = ("mesh_med.stl", "mesh.stl")


@dataclass(frozen=True)
class SourceClip:
    """One indexed source clip at its native rate."""

    clip_id: str
    npz_path: Path
    object_asset: str


def _safe_component(value: str, *, label: str) -> str:
    """Reject a path component that could escape its directory."""
    text = str(value)
    if not text or Path(text).name != text or text in {".", ".."}:
        raise DatasetError(f"{label} must be a plain name, not a path: {value!r}")
    return text


def load_json(path: Path) -> Any:
    return json.loads(Path(path).read_text())


def discover_source_clips(source: Path) -> tuple[str, float, tuple[SourceClip, ...]]:
    """Read the bundle index; returns ``(handedness, declared_fps, clips)``."""
    index = load_json(source / "index.json")
    handedness = str(index.get("handedness", "right")).strip().lower() or "right"
    if handedness not in {"left", "right"}:
        raise DatasetError(f"unsupported handedness {handedness!r}")
    clips = index.get("clips")
    if not isinstance(clips, dict) or not clips:
        raise DatasetError(f"{source / 'index.json'}: clips must be a non-empty object")
    declared_count = index.get("clip_count")
    if declared_count is not None and int(declared_count) != len(clips):
        raise DatasetError(f"{source / 'index.json'}: clip_count does not match clips")

    records: list[SourceClip] = []
    for clip_id, metadata in sorted(clips.items()):
        _safe_component(clip_id, label="clip id")
        if not isinstance(metadata, dict):
            raise DatasetError(f"{source / 'index.json'}: invalid metadata for clip {clip_id!r}")
        relative = metadata.get("raw_npz_path", f"rawdata/{clip_id}.npz")
        npz_path = (source / relative).resolve()
        try:
            npz_path.relative_to(source.resolve())
        except ValueError as error:
            raise DatasetError(f"raw_npz_path escapes the bundle: {relative!r}") from error
        if npz_path.is_symlink():
            raise DatasetError(f"raw clip must not be a symlink: {npz_path}")
        if not npz_path.is_file():
            raise FileNotFoundError(npz_path)
        records.append(
            SourceClip(
                clip_id=clip_id,
                npz_path=npz_path,
                object_asset=_safe_component(metadata["object_asset"], label="object asset"),
            )
        )
    return handedness, float(index.get("fps", TARGET_FPS)), tuple(records)


def load_source_clip(record: SourceClip) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Decode one native-rate source clip; returns ``(landmarks, object_pose, t_s, fps)``."""
    with np.load(record.npz_path, allow_pickle=False) as data:
        landmark_key = next((key for key in LANDMARK_KEYS if key in data.files), None)
        if landmark_key is None:
            raise DatasetError(f"{record.npz_path}: no landmark array (expected one of {LANDMARK_KEYS})")
        for key in ("object_pose_w", "t_s"):
            if key not in data.files:
                raise DatasetError(f"{record.npz_path}: missing field {key!r}")
        landmarks = np.asarray(data[landmark_key], dtype=np.float64)
        object_pose = np.asarray(data["object_pose_w"], dtype=np.float64)
        timestamps = np.asarray(data["t_s"], dtype=np.float64)
        declared_fps = None
        if "fps" in data.files:
            fps_value = np.asarray(data["fps"])
            if fps_value.shape != ():
                raise DatasetError(f"{record.npz_path}: fps must be a scalar")
            declared_fps = float(fps_value.item())

    frames = len(timestamps)
    if frames <= 1 or landmarks.shape != (frames, MEDIAPIPE_LANDMARK_COUNT, 3):
        raise DatasetError(f"{record.npz_path}: invalid landmark or timestamp shape")
    if object_pose.shape != (frames, 7):
        raise DatasetError(f"{record.npz_path}: object_pose_w must have shape (T, 7)")
    if not np.isfinite(landmarks).all() or not np.isfinite(object_pose).all():
        raise DatasetError(f"{record.npz_path}: landmarks and object pose must be finite on every frame")
    if not np.isfinite(timestamps).all() or np.any(np.diff(timestamps) <= 0.0):
        raise DatasetError(f"{record.npz_path}: timestamps must be finite and strictly increasing")
    measured_fps = (frames - 1) / float(timestamps[-1] - timestamps[0])
    if not math.isfinite(measured_fps) or measured_fps <= 0.0:
        raise DatasetError(f"{record.npz_path}: invalid measured frame rate")
    if declared_fps is not None and not math.isclose(declared_fps, measured_fps, rel_tol=1e-5, abs_tol=1e-5):
        raise DatasetError(f"{record.npz_path}: fps {declared_fps} != measured fps {measured_fps}")
    return landmarks, object_pose, timestamps, measured_fps


def canonical_clip(record: SourceClip) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Resample one source clip to 50 Hz and re-origin it."""
    landmarks, object_pose, source_times, source_fps = load_source_clip(record)
    target_times = resample_times_from_samples(source_times, 1.0 / TARGET_FPS)
    canonical_landmarks = resample_linear_at(landmarks, target_times, source_times).astype(np.float32)
    canonical_object = resample_pose_at(object_pose, target_times, source_times).astype(np.float32)
    canonical_landmarks, canonical_object = zero_initial_object_xy(
        canonical_landmarks[:, None], canonical_object[:, None]
    )

    stats = {
        "source_frames": int(len(source_times)),
        "source_fps": source_fps,
        "source_duration_s": float(source_times[-1] - source_times[0]),
        "target_frames": int(len(target_times)),
    }
    return canonical_landmarks, canonical_object, stats


def copy_asset(source_asset: Path, target_asset: Path, *, mass_kg: float | None) -> dict[str, Any]:
    """Copy one object asset, keeping its mass metadata and collision sidecar."""
    import trimesh

    mesh_path = next((source_asset / name for name in SOURCE_MESH_NAMES if (source_asset / name).is_file()), None)
    if mesh_path is None:
        raise FileNotFoundError(f"no mesh under {source_asset} (expected one of {SOURCE_MESH_NAMES})")
    mesh_bytes = mesh_path.read_bytes()
    mesh_sha256 = hashlib.sha256(mesh_bytes).hexdigest()

    description: dict[str, Any] = {}
    if (source_asset / ASSET_META_NAME).is_file():
        description = load_json(source_asset / ASSET_META_NAME)
    declared_sha = description.get("mesh_sha256")
    if declared_sha is not None and declared_sha != mesh_sha256:
        raise DatasetError(f"mesh checksum mismatch: {mesh_path}")

    resolved_mass = mass_kg if mass_kg is not None else description.get("mass_kg")
    if resolved_mass is None:
        raise DatasetError(f"{source_asset}: no mass_kg (set it in asset.json or pass --mass-kg)")

    model = trimesh.load(mesh_path, force="mesh")
    if not isinstance(model, trimesh.Trimesh) or not model.vertices.size:
        raise DatasetError(f"empty asset mesh: {mesh_path}")

    target_asset.mkdir(parents=True, exist_ok=True)
    (target_asset / MESH_NAME).write_bytes(mesh_bytes)
    write_json(target_asset / ASSET_META_NAME, {"mass_kg": float(resolved_mass), "mesh_sha256": mesh_sha256})
    has_collision = (source_asset / "collision.json").is_file()
    if has_collision:
        shutil.copy2(source_asset / "collision.json", target_asset / "collision.json")

    return {
        "source": f"assets/{source_asset.name}",
        "mesh_sha256": mesh_sha256,
        "mass_kg": float(resolved_mass),
        "extents_m": [float(v) for v in model.extents],
        "compound_collision": has_collision,
        "source_metadata": description,
    }


def import_bundle(
    source: str | Path,
    out: str | Path,
    *,
    mass_kg: float | None = None,
    limit: int | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Convert one keypoint bundle into a canonical dataset; returns the manifest."""
    source = Path(source).expanduser().resolve()
    handedness, declared_fps, records = discover_source_clips(source)
    if limit is not None and limit > 0:
        records = records[:limit]

    manifest: dict[str, Any] = {
        "schema": CANONICAL_SCHEMA,
        "source": source.name,
        "source_declared_fps": declared_fps,
        "handedness": handedness,
        "target_fps": TARGET_FPS,
        "sort": "clip id ascending",
        "assets": {},
        "clips": [],
    }

    with staged_dataset(out, force=force) as stage:
        for ordinal, record in enumerate(records, start=1):
            asset_id = safe_name(record.object_asset)
            if asset_id not in manifest["assets"]:
                manifest["assets"][asset_id] = copy_asset(
                    source / "assets" / record.object_asset, stage / "assets" / asset_id, mass_kg=mass_kg
                )

            clip_id = safe_name(record.clip_id)
            landmarks, object_pose, stats = canonical_clip(record)
            write_clip(
                stage,
                clip_id,
                landmarks,
                object_pose,
                {
                    "hands": [handedness],
                    "objects": [asset_id],
                    "source": {
                        "bundle": source.name,
                        "npz": recorded_path(record.npz_path, source),
                        "sha256": hashlib.sha256(record.npz_path.read_bytes()).hexdigest(),
                        **{k: v for k, v in stats.items() if k != "target_frames"},
                    },
                },
            )
            manifest["clips"].append({"ordinal": ordinal, "clip_id": clip_id, "object_asset": asset_id, **stats})
            print(
                f"[{LOG_PREFIX}] {clip_id}: {stats['source_frames']} frames @ {stats['source_fps']:.2f} Hz "
                f"-> {stats['target_frames']} @ {TARGET_FPS:g} Hz",
                flush=True,
            )
        write_json(stage / MANIFEST_NAME, manifest)

    print(f"[{LOG_PREFIX}] imported {len(manifest['clips'])} clips and {len(manifest['assets'])} assets -> {out}")
    return manifest


def load_registry(registry: Path) -> dict[str, dict[str, Any]]:
    """The registry entries by name; none when there is no registry file."""
    return load_json(registry).get("datasets", {}) if registry.is_file() else {}


def registry_entry(registry: Path, name: str) -> dict[str, Any]:
    """One registry entry, its paths resolved against the registry file."""
    entries = load_registry(registry)
    if name not in entries:
        if not entries:
            raise KeyError(f"no dataset {name!r}: {registry} is missing or empty; pass --source and --out")
        raise KeyError(f"unknown dataset {name!r}; available: {sorted(entries)}")
    entry = dict(entries[name])
    for key in ("source", "out"):
        value = entry.get(key)
        if value is None:
            raise KeyError(f"dataset {name!r} has no {key!r}")
        path = Path(value)
        entry[key] = str(path if path.is_absolute() else (registry.resolve().parent / path).resolve())
    return entry


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=str, default=None, help="Registry entry name (see --registry).")
    parser.add_argument(
        "--registry", type=Path, default=Path(REGISTRY_NAME), help="Registry of capture bundles by name (JSON)."
    )
    parser.add_argument("--source", type=Path, default=None, help="Keypoint bundle root (overrides the registry).")
    parser.add_argument("--out", type=Path, default=None, help="Output dataset root (overrides the registry).")
    parser.add_argument(
        "--mass-kg", type=float, default=None, help="Object mass override (priority over registry and asset values)."
    )
    parser.add_argument("--limit", type=int, default=None, help="Import only the first N clips (previews).")
    parser.add_argument("--force", action="store_true", help="Replace an existing output dataset.")
    parser.add_argument("--list", action="store_true", help="List the registry entries and exit.")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.list:
        for name, entry in sorted(load_registry(args.registry).items()):
            print(f"{name}: {entry.get('description', '')}")
            print(f"  source: {entry.get('source')}")
            print(f"  out:    {entry.get('out')}")
        return

    source, out, mass_kg = args.source, args.out, args.mass_kg
    if args.dataset is not None:
        entry = registry_entry(args.registry, args.dataset)
        source = source or Path(entry["source"])
        out = out or Path(entry["out"])
        mass_kg = mass_kg if mass_kg is not None else entry.get("mass_kg")
    if source is None or out is None:
        raise SystemExit("pass --dataset, or both --source and --out")

    import_bundle(source, out, mass_kg=mass_kg, limit=args.limit, force=args.force)


if __name__ == "__main__":
    main()
