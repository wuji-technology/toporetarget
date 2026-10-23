#!/usr/bin/env python3
r"""Import a public MANO hand-object dataset into a canonical TopoRetarget dataset.

Examples:
    toporetarget import-hoi --dataset grab --raw /data/GRAB --out data/grab
    toporetarget import-hoi --dataset taco --raw /data/TACO --out data/taco \
        --units "(brush, brush, bowl)/20230919_036" --workers 4
"""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import dataclasses
import functools
import importlib.util
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from toporetarget.config import COLLISION_SIDECAR, safe_name, sha256_file
from toporetarget.datasets.canonical import (
    ASSET_META_NAME,
    CANONICAL_SCHEMA,
    MANIFEST_NAME,
    MAX_MESH_FACES,
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
from toporetarget.datasets.mano_model import DEFAULT_MANO_DIR, TIP_VERTICES, ManoModel, mediapipe_keypoints
from toporetarget.datasets.readers import DATASETS, ManoTrack, Recording, reader
from toporetarget.objects import fit_convex_sidecar

LOG = "ImportHOI"
SIDES = ("right", "left")
SURFACE_SAMPLES = 20000
MANO_CHUNK = 512
DEVIATION_BUDGET_MAX_MM = 0.15
DEVIATION_BUDGET_P99_MM = 0.10
DEVIATION_SAMPLES = 4096
DEVIATION_CANDIDATES = 16
DEVIATION_REFINE_CANDIDATES = 1024
DEVIATION_CHUNK = 1 << 20


@dataclasses.dataclass(frozen=True)
class ContactRule:
    """When a hand counts as holding an object, and how much of it a clip keeps."""

    contact_m: float
    min_contact_s: float
    gap_s: float
    pad_s: float


@functools.cache
def _mano(side: str, mano_dir: str, fix_left_shapedirs: bool) -> ManoModel:
    model = ManoModel.load(side, mano_dir)
    if side == "left" and fix_left_shapedirs:
        shapedirs = model.shapedirs.copy()
        shapedirs[:, 0, :] *= -1.0
        model = dataclasses.replace(model, shapedirs=shapedirs)
    return model


def hand_points(track: ManoTrack, side: str, rows: np.ndarray, mano_dir: str) -> tuple[np.ndarray, np.ndarray]:
    """MANO ``(keypoints (n, 21, 3), vertices (n, 778, 3))`` of the given rows, release world frame."""
    model = _mano(side, mano_dir, track.fix_left_shapedirs)
    keypoints, vertices = [], []
    for start in range(0, len(rows), MANO_CHUNK):
        chunk = rows[start : start + MANO_CHUNK]
        betas = track.betas
        if betas is not None and betas.ndim == 2:
            betas = betas[chunk]
        joints, verts = model.forward(
            track.global_orient[chunk],
            model.full_pose(
                track.hand_pose[chunk], num_pca_comps=track.num_pca_comps, flat_hand_mean=track.flat_hand_mean
            ),
            betas=betas,
            transl=None if track.transl is None else track.transl[chunk],
            v_template=track.v_template,
        )
        if track.wrist is not None:
            shift = track.wrist[chunk, None] - joints[:, :1]
            joints, verts = joints + shift, verts + shift
        keypoints.append(mediapipe_keypoints(joints, verts))
        vertices.append(verts)
    return np.concatenate(keypoints), np.concatenate(vertices)


def surface_tree(mesh) -> cKDTree:
    """A KD-tree over the mesh vertices plus dense surface samples, object frame."""
    import trimesh

    samples, _ = trimesh.sample.sample_surface(mesh, SURFACE_SAMPLES, seed=0)
    return cKDTree(np.concatenate([np.asarray(mesh.vertices), samples]))


def decimate(mesh, max_faces: int):
    """The mesh quadric-decimated to at most ``max_faces`` faces (fast-simplification)."""
    import trimesh

    welded = trimesh.Trimesh(mesh.vertices, mesh.faces)
    simple = welded.simplify_quadric_decimation(face_count=max_faces)
    if not 0 < len(simple.faces) <= max_faces:
        raise DatasetError(f"decimating {len(mesh.faces)} faces left {len(simple.faces)}, not 1..{max_faces}")
    return simple


def surface_distance(points: np.ndarray, mesh) -> np.ndarray:
    """``(n,)`` distance of each point to the mesh surface, an upper bound."""
    import trimesh

    tree = cKDTree(mesh.triangles_center)

    def measure(rows: np.ndarray, k: int) -> np.ndarray:
        _, nearest = tree.query(points[rows], k=k)
        out = np.empty(len(rows))
        step = max(1, DEVIATION_CHUNK // k)
        for start in range(0, len(rows), step):
            part = slice(start, start + step)
            triangles = mesh.triangles[nearest[part]].reshape(-1, 3, 3)
            repeated = np.repeat(points[rows[part]], k, axis=0)
            gap = np.linalg.norm(trimesh.triangles.closest_point(triangles, repeated) - repeated, axis=1)
            out[part] = gap.reshape(-1, k).min(axis=1)
        return out

    distance = measure(np.arange(len(points)), DEVIATION_CANDIDATES)
    far = np.flatnonzero(distance > np.quantile(distance, 0.99))
    distance[far] = np.minimum(distance[far], measure(far, DEVIATION_REFINE_CANDIDATES))
    return distance


def decimation_report(source, simple) -> dict[str, Any]:
    """How far a decimated mesh moved from its source, and whether its topology held."""
    import trimesh

    def points(mesh) -> np.ndarray:
        samples, _ = trimesh.sample.sample_surface(mesh, DEVIATION_SAMPLES, seed=0)
        return np.concatenate([mesh.vertices, mesh.triangles_center, samples])

    deviation = np.concatenate([surface_distance(points(source), simple), surface_distance(points(simple), source)])
    welded = trimesh.Trimesh(source.vertices, source.faces)

    def euler_numbers(mesh) -> list[int]:
        return sorted(int(part.euler_number) for part in mesh.split(only_watertight=False))

    return {
        "deviation_max_mm": float(deviation.max() * 1e3),
        "deviation_p99_mm": float(np.quantile(deviation, 0.99) * 1e3),
        "euler_numbers_kept": euler_numbers(welded) == euler_numbers(simple),
        "watertight": [bool(welded.is_watertight), bool(simple.is_watertight)],
    }


def contact_distance(vertices: np.ndarray, poses: np.ndarray, tree: cKDTree, bound: float) -> np.ndarray:
    """``(n,)`` closest hand-vertex distance to the object surface per frame, inf beyond ``bound``."""
    local = np.einsum("tji,tvj->tvi", poses[:, :3, :3], vertices - poses[:, None, :3, 3])
    lo, hi = tree.mins - bound, tree.maxes + bound
    near = ((local >= lo) & (local <= hi)).all(axis=2)
    distance = np.full(near.shape, np.inf)
    if near.any():
        distance[near] = tree.query(local[near], distance_upper_bound=bound)[0]
    return distance.min(axis=1)


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    edges = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1), strict=True))


def contact_episodes(
    contact: np.ndarray, valid: np.ndarray, times: np.ndarray, rule: ContactRule
) -> list[tuple[int, int, int, int]]:
    """``(start, end, first_contact, last_contact + 1)`` frame windows of one hand on one object."""
    held = contact & valid
    for a, b in _runs(~held):
        if 0 < a and b < len(held) and valid[a:b].all() and times[b] - times[a - 1] <= rule.gap_s:
            held[a:b] = True
    episodes = []
    tracked = _runs(valid)
    for a, b in _runs(held):
        if times[b - 1] - times[a] < rule.min_contact_s:
            continue
        lo, hi = next((s, e) for s, e in tracked if s <= a and b <= e)
        start = int(np.searchsorted(times, times[a] - rule.pad_s, side="left"))
        end = int(np.searchsorted(times, times[b - 1] + rule.pad_s, side="right"))
        episodes.append((max(start, lo), min(end, hi), a, b))
    return episodes


def _union(spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Overlapping ``[start, end)`` spans joined."""
    joined: list[list[int]] = []
    for start, end in sorted(spans):
        if joined and start < joined[-1][1]:
            joined[-1][1] = max(joined[-1][1], end)
        else:
            joined.append([start, end])
    return [(start, end) for start, end in joined]


def _pair_spans(
    contact: dict[tuple[str, str], np.ndarray],
    valid: dict[str, np.ndarray],
    times: np.ndarray,
    rule: ContactRule,
    lo: int,
    hi: int,
) -> list[tuple[int, int]]:
    """The padded contact episodes of every hand-object pair inside ``[lo, hi)``."""
    spans = []
    for (side, obj), mask in contact.items():
        tracked = valid[side][lo:hi] & valid[obj][lo:hi]
        spans.extend((lo + a, lo + b) for a, b, _, _ in contact_episodes(mask[lo:hi], tracked, times[lo:hi], rule))
    return spans


def interaction_windows(
    contact: dict[tuple[str, str], np.ndarray],
    valid: dict[str, np.ndarray],
    times: np.ndarray,
    rule: ContactRule,
) -> list[tuple[int, int]]:
    """``[start, end)`` frame windows in which some hand interacts with some object."""
    windows = []
    for start, end in _union(_pair_spans(contact, valid, times, rule, 0, len(times))):
        members = {m for pair, mask in contact.items() if mask[start:end].any() for m in pair}
        tracked = np.all([valid[m][start:end] for m in members], axis=0)
        for a, b in _runs(tracked):
            windows.extend(_union(_pair_spans(contact, valid, times, rule, start + a, start + b)))
    return windows


def pose7(poses: np.ndarray) -> np.ndarray:
    """``(T, 7)`` ``[xyz, wxyz]`` from ``(T, 4, 4)``, one quaternion hemisphere throughout."""
    quat = Rotation.from_matrix(poses[:, :3, :3]).as_quat()[:, [3, 0, 1, 2]]
    flips = np.cumprod(np.where(np.sum(quat[1:] * quat[:-1], axis=1) < 0.0, -1.0, 1.0))
    quat[1:] *= flips[:, None]
    return np.concatenate([poses[:, :3, 3], quat], axis=1)


def recording_clips(
    dataset: str, raw: Path, unit: str, *, sides: tuple[str, ...], mano_dir: str, rule: ContactRule
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """``(clips, skipped)`` of one recording; a clip carries its 50 Hz arrays and meta."""
    source = reader(dataset)
    recording: Recording = source.load(raw, unit)
    times = np.asarray(recording.times, dtype=np.float64)
    objects = list(recording.object_poses)
    trees = {obj: surface_tree(source.load_mesh(raw, obj)) for obj in objects}
    valid = {obj: np.isfinite(p).all(axis=(1, 2)) for obj, p in recording.object_poses.items()}

    keypoints: dict[str, np.ndarray] = {}
    contact: dict[tuple[str, str], np.ndarray] = {}
    for side in sides:
        track = recording.hands.get(side)
        if track is None:
            continue
        hand_valid = np.isfinite(track.global_orient).all(axis=1) & np.isfinite(track.hand_pose).all(axis=1)
        rows = np.flatnonzero(hand_valid)
        if len(rows) < 2:
            continue
        keypoints[side] = np.full((len(times), 21, 3), np.nan)
        vertices = np.full((len(times), 778, 3), np.nan)
        keypoints[side][rows], vertices[rows] = hand_points(track, side, rows, mano_dir)
        valid[side] = hand_valid
        for obj, poses in recording.object_poses.items():
            distance = np.full(len(times), np.inf)
            both = np.flatnonzero(hand_valid & valid[obj])
            if len(both):
                distance[both] = contact_distance(vertices[both], poses[both], trees[obj], rule.contact_m)
            contact[(side, obj)] = distance < rule.contact_m

    clips: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for k, (start, end) in enumerate(interaction_windows(contact, valid, times, rule)):
        clip_id = safe_name(f"{recording.name}__{k:02d}")
        touched = {pair: int(mask[start:end].sum()) for pair, mask in contact.items() if mask[start:end].any()}
        clip_objects = [obj for obj in objects if any(o == obj for _, o in touched)]
        clip_sides = [side for side in keypoints if valid[side][start:end].all()]
        world = np.tile(np.eye(4), (end - start, len(clip_objects), 1, 1))
        for j, obj in enumerate(clip_objects):
            poses = recording.object_poses[obj][start:end]
            world[:, j, :3, :3] = recording.to_zup @ poses[:, :3, :3]
            world[:, j, :3, 3] = poses[:, :3, 3] @ recording.to_zup.T
        landmarks = np.stack([keypoints[side][start:end] for side in clip_sides], axis=1) @ recording.to_zup.T

        source_times = times[start:end]
        target_times = resample_times_from_samples(source_times, 1.0 / TARGET_FPS)
        landmarks = resample_linear_at(landmarks, target_times, source_times)
        object_pose = np.stack(
            [resample_pose_at(pose7(world[:, j]), target_times, source_times) for j in range(len(clip_objects))],
            axis=1,
        )
        landmarks, object_pose = zero_initial_object_xy(landmarks, object_pose)
        if len(landmarks) < 2:
            skipped.append({"clip_id": clip_id, "reason": "shorter than two frames at 50 Hz"})
            continue
        clips.append(
            {
                "clip_id": clip_id,
                "landmarks": landmarks,
                "object_pose": object_pose,
                "object_ids": dict(zip((safe_name(obj) for obj in clip_objects), clip_objects, strict=True)),
                "meta": {
                    "hands": clip_sides,
                    "objects": [safe_name(obj) for obj in clip_objects],
                    "source": {
                        "dataset": dataset,
                        "raw": Path(raw).name,
                        "unit": unit,
                        "recording": recording.name,
                        "object_ids": clip_objects,
                        "source_frames": [int(start), int(end)],
                        "source_fps": float((end - start - 1) / (source_times[-1] - source_times[0])),
                        "source_duration_s": float(source_times[-1] - source_times[0]),
                        "contact_frames": [
                            {"hand": side, "object": safe_name(obj), "frames": frames}
                            for (side, obj), frames in sorted(touched.items())
                        ],
                    },
                },
            }
        )
    return clips, skipped


def write_asset(
    dataset: str, raw: Path, object_id: str, target: Path, *, collision: str, convex_threshold: float, max_parts: int
) -> dict[str, Any]:
    """One object's mesh (and convex sidecar) in the canonical asset layout, compiled once by MuJoCo."""
    from toporetarget.solver import check_objects

    mesh = reader(dataset).load_mesh(raw, object_id)
    source_faces = len(mesh.faces)
    decimation = None
    if source_faces > MAX_MESH_FACES:
        source, mesh = mesh, decimate(mesh, MAX_MESH_FACES)
        decimation = decimation_report(source, mesh)
    target.mkdir(parents=True, exist_ok=True)
    mesh.export(target / MESH_NAME)
    sidecar = None
    if collision == "convex":
        sidecar = fit_convex_sidecar(mesh, target, threshold=convex_threshold, max_parts=max_parts)
        if sidecar is not None:
            write_json(target / COLLISION_SIDECAR, sidecar)
    description = {
        "mesh_sha256": sha256_file(target / MESH_NAME),
        "source": {"dataset": dataset, "object_id": object_id},
    }
    write_json(target / ASSET_META_NAME, description)
    compile_error = None
    try:
        check_objects([target / MESH_NAME])
    except ValueError as error:
        compile_error = " | ".join(line.strip() for line in str(error).splitlines() if line.strip())
    entry = {
        **description,
        "vertices": int(len(mesh.vertices)),
        "faces": int(len(mesh.faces)),
        "source_faces": int(source_faces),
        "extents_m": [float(v) for v in mesh.extents],
        "collision_parts": 1 if sidecar is None else len(sidecar["geoms"]),
    }
    if decimation is not None:
        entry["decimation"] = decimation
    if compile_error is not None:
        entry["compile_error"] = compile_error
    return entry


def mano_files(mano_dir: Path) -> dict[str, str]:
    return {side: sha256_file(Path(mano_dir) / f"MANO_{side.upper()}.pkl") for side in SIDES}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=DATASETS, required=True, help="Which release --raw holds.")
    parser.add_argument("--raw", type=Path, required=True, help="The release root, in its published layout.")
    parser.add_argument("--out", type=Path, default=None, help="Canonical dataset root to write.")
    parser.add_argument("--units", nargs="*", default=None, help="Import only these recordings (see --list).")
    parser.add_argument("--limit", type=int, default=None, help="Import only the first N recordings.")
    parser.add_argument("--list", action="store_true", help="List the recordings under --raw and exit.")
    parser.add_argument("--sides", nargs="+", choices=SIDES, default=list(SIDES), help="Hands to import.")
    parser.add_argument("--mano-dir", type=Path, default=DEFAULT_MANO_DIR, help="MANO_{RIGHT,LEFT}.pkl directory.")
    parser.add_argument("--contact-mm", type=float, default=10.0, help="Hand-surface distance that counts as contact.")
    parser.add_argument("--min-contact-s", type=float, default=1.0, help="Shortest contact episode kept.")
    parser.add_argument("--gap-s", type=float, default=0.5, help="Longest contact gap bridged inside one episode.")
    parser.add_argument("--pad-s", type=float, default=0.5, help="Frames kept before and after each episode.")
    parser.add_argument(
        "--collision",
        choices=("convex", "hull"),
        default="convex",
        help="convex: CoACD parts for concave objects; hull: MuJoCo's convex hull of the mesh.",
    )
    parser.add_argument("--convex-threshold", type=float, default=0.05, help="CoACD concavity threshold.")
    parser.add_argument("--max-parts", type=int, default=16, help="Most convex parts per object.")
    parser.add_argument(
        "--workers",
        type=int,
        default=min(4, max(1, (os.cpu_count() or 2) // 2)),
        help="Parallel recordings (default half the CPUs, 1 to 4; each holds a recording and the MANO models).",
    )
    parser.add_argument("--force", action="store_true", help="Replace an existing output dataset.")
    return parser.parse_args()


def main():
    args = parse_args()
    raw = Path(args.raw).expanduser().resolve()
    source = reader(args.dataset)
    available = source.units(raw)
    if args.list:
        print("\n".join(available))
        return
    if args.out is None:
        raise SystemExit("pass --out, the canonical dataset root to write")
    needed = [*source.REQUIRES, *(["coacd"] if args.collision == "convex" else [])]
    missing_packages = [name for name in needed if importlib.util.find_spec(name) is None]
    if missing_packages:
        hints = {
            "torch": "the release pickles hold PyTorch tensors; pip install torch (the CPU build is enough)",
            "coacd": "convex decomposition; pip install coacd, or import with --collision hull",
        }
        raise SystemExit("\n".join(f"missing {name}: {hints[name]}" for name in missing_packages))
    units = list(args.units) if args.units else available
    missing = sorted(set(units) - set(available))
    if missing:
        raise SystemExit(f"not in {raw}: {missing}")
    if args.limit is not None and args.limit > 0:
        units = units[: args.limit]
    if not units:
        raise SystemExit(f"no recordings under {raw}")
    mano_dir = Path(args.mano_dir).expanduser().resolve()
    ManoModel.load("right", mano_dir), ManoModel.load("left", mano_dir)
    rule = ContactRule(
        contact_m=args.contact_mm * 1e-3,
        min_contact_s=args.min_contact_s,
        gap_s=args.gap_s,
        pad_s=args.pad_s,
    )

    started = time.perf_counter()
    print(f"[{LOG}] {args.dataset}: {len(units)} recordings in {raw}, workers={args.workers}", flush=True)
    manifest: dict[str, Any] = {
        "schema": CANONICAL_SCHEMA,
        "source": raw.name,
        "source_dataset": args.dataset,
        "handedness": "both" if len(args.sides) == 2 else args.sides[0],
        "target_fps": TARGET_FPS,
        "sort": "clip id ascending",
        "mano": {"sha256": mano_files(mano_dir), "tip_vertices": TIP_VERTICES},
        "contact_rule": dataclasses.asdict(rule),
        "collision": args.collision,
        "assets": {},
        "clips": [],
        "skipped": [],
    }
    with staged_dataset(args.out, force=args.force) as stage:
        clips_by_object: dict[str, str] = {}
        with futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
            jobs = {
                pool.submit(
                    recording_clips, args.dataset, raw, unit, sides=tuple(args.sides), mano_dir=str(mano_dir), rule=rule
                ): unit
                for unit in units
            }
            for done, job in enumerate(futures.as_completed(jobs), start=1):
                unit = jobs[job]
                try:
                    clips, skipped = job.result()
                except Exception as error:
                    manifest["skipped"].append({"unit": unit, "reason": f"{type(error).__name__}: {error}"})
                    print(f"[{LOG}] [{done}/{len(units)}] {unit}: FAILED {error}", flush=True)
                    continue
                for clip in clips:
                    meta = write_clip(stage, clip["clip_id"], clip["landmarks"], clip["object_pose"], clip["meta"])
                    clips_by_object.update(clip["object_ids"])
                    manifest["clips"].append(
                        {
                            "clip_id": clip["clip_id"],
                            "hands": meta["hands"],
                            "objects": meta["objects"],
                            "target_frames": meta["frames"],
                            "unit": unit,
                        }
                    )
                manifest["skipped"].extend({"unit": unit, **row} for row in skipped)
                print(
                    f"[{LOG}] [{done}/{len(units)}] {unit}: {len(clips)} clips"
                    + (f", {len(skipped)} skipped" if skipped else ""),
                    flush=True,
                )

        with futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
            jobs = {
                pool.submit(
                    write_asset,
                    args.dataset,
                    raw,
                    object_id,
                    stage / "assets" / asset_id,
                    collision=args.collision,
                    convex_threshold=args.convex_threshold,
                    max_parts=args.max_parts,
                ): asset_id
                for asset_id, object_id in sorted(clips_by_object.items())
            }
            for job in futures.as_completed(jobs):
                asset_id = jobs[job]
                entry = manifest["assets"][asset_id] = job.result()
                note = ""
                if "decimation" in entry:
                    report = entry["decimation"]
                    note = (
                        f", decimated {entry['source_faces']} -> {entry['faces']} faces, surface deviation max "
                        f"{report['deviation_max_mm']:.3f} mm, p99 {report['deviation_p99_mm']:.3f} mm"
                    )
                    if (
                        report["deviation_max_mm"] > DEVIATION_BUDGET_MAX_MM
                        or report["deviation_p99_mm"] > DEVIATION_BUDGET_P99_MM
                    ):
                        note += f" -- WARNING: over the {DEVIATION_BUDGET_MAX_MM} / {DEVIATION_BUDGET_P99_MM} mm budget"
                    if not report["euler_numbers_kept"]:
                        note += " -- WARNING: a connected component changed its Euler number"
                if "compile_error" in entry:
                    note += f" -- WARNING: MuJoCo does not compile it, its clips will fail: {entry['compile_error']}"
                print(f"[{LOG}] asset {asset_id}: {entry['collision_parts']} collision parts{note}", flush=True)

        manifest["clips"].sort(key=lambda row: row["clip_id"])
        manifest["clip_count"] = len(manifest["clips"])
        manifest["frame_count"] = sum(row["target_frames"] for row in manifest["clips"])
        write_json(stage / MANIFEST_NAME, manifest)

    hands = {side: sum(1 for row in manifest["clips"] if side in row["hands"]) for side in SIDES}
    print(
        f"[{LOG}] {manifest['clip_count']} clips ({hands['right']} with the right hand, "
        f"{hands['left']} with the left), "
        f"{manifest['frame_count'] / TARGET_FPS:.0f} s at {TARGET_FPS:g} Hz, {len(manifest['assets'])} assets, "
        f"{len(manifest['skipped'])} skipped, {time.perf_counter() - started:.0f} s -> {Path(args.out).resolve()}"
    )


if __name__ == "__main__":
    main()
