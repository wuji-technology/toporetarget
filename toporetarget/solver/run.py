"""Solving one hand of one clip: config assembly, the solve loop, the result files and the command line."""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation as RotLib

from toporetarget.config import (
    CANONICAL_SCHEMA,
    COLLISION_SIDECAR,
    RESULT_SCHEMA,
    RETARGET_METHOD,
    diagnostics_path,
    load_config,
    recorded_asset_path,
    recorded_path,
    resolve_path,
    solver_settings_digest,
)
from toporetarget.config import (
    section as config_section,
)
from toporetarget.dataset import CanonicalDataset
from toporetarget.solver.clip import CanonicalClip, build_retarget_track, load_canonical_clip, resolve_clip_paths
from toporetarget.solver.hand import WRIST_DOF, HandSpec
from toporetarget.solver.landmarks import FINGERTIP_LANDMARKS
from toporetarget.solver.materialize import FloatingSceneFK
from toporetarget.solver.model import MuJoCoFloatingHandModel
from toporetarget.solver.qp import PenetrationSpec
from toporetarget.solver.sampling import resolve_sequence_object_points
from toporetarget.solver.selfcol import ObjectClearance, format_penetration, optimize_trajectory
from toporetarget.solver.session import FrameSolver
from toporetarget.solver.settings import LOG_PREFIX, HandRetargetConfig, apply_config_data, set_config_value

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - progress bars are optional

    def tqdm(iterable, **_kwargs):
        return iterable


def build_hand_config(config: dict[str, Any], overrides: Sequence[str] = ()) -> HandRetargetConfig:
    """A ``HandRetargetConfig`` from the merged config's ``hand_solver`` block, with ``--set`` overrides applied."""
    block = config_section(config, "hand_solver")
    if not block:
        raise ValueError("config has no hand_solver block (the HandRetargetConfig sections)")
    cfg = HandRetargetConfig()
    apply_config_data(cfg, dict(block))
    for kv in overrides:
        key, sep, raw = kv.partition("=")
        if not sep:
            raise KeyError(f"--set expects KEY=VAL, got {kv!r}")
        set_config_value(cfg, key, _parse_override(raw))
    cfg.revalidate_after_override()
    return cfg


def _parse_override(raw: str) -> Any:
    """Parse an override value as JSON, falling back to the raw string."""
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def solve_clip(
    clip: CanonicalClip,
    cfg: HandRetargetConfig,
    hand: HandSpec,
) -> tuple[np.ndarray, list[np.ndarray], list[dict[str, Any]], FrameSolver, np.ndarray, np.ndarray]:
    """Solve one hand of a clip frame by frame, against all of the clip's objects."""
    track = build_retarget_track(
        clip.landmarks_world,
        clip.object_pose_world[..., :3],
        clip.object_pose_world[..., [4, 5, 6, 3]],
        clip.wrist_pose_world[:, [4, 5, 6, 3]],
        mesh_paths=[str(path) for path in clip.object_meshes],
    )
    object_position = np.stack([obj.pose.position for obj in track.objects], axis=1)
    object_quaternion = np.stack([obj.pose.quaternion_xyzw for obj in track.objects], axis=1)

    object_points_local = resolve_sequence_object_points(track, cfg.object)
    solver = FrameSolver.from_paths(cfg, hand, clip.object_meshes, object_points_local=object_points_local)

    qpos = np.empty((clip.n_frames, solver.nq), dtype=np.float64)
    metrics: list[dict[str, Any]] = []
    for frame in tqdm(range(clip.n_frames), desc=f"{LOG_PREFIX} {clip.clip_id} {clip.side}"):
        qpos[frame] = solver.step_palm(track.landmarks[frame], object_position[frame], object_quaternion[frame])
        metrics.append(dict(solver.retargeter.runtime.optimizer.frame_np_metrics))
    return qpos, object_points_local, metrics, solver, object_position, object_quaternion


def world_fingertips(
    hand: MuJoCoFloatingHandModel,
    qpos: np.ndarray,
    clip: CanonicalClip,
) -> np.ndarray:
    """The solved fingertips lifted into the world frame, ``(T, 5, 3)``."""
    tip_names = list(hand.spec.tip_sites)
    tips_solver = np.empty((len(qpos), len(tip_names), 3))
    for t in range(len(qpos)):
        hand.forward(qpos[t])
        tips_solver[t] = hand.get_body_positions(tip_names)
    R_in = RotLib.from_quat(clip.wrist_pose_world[:, [4, 5, 6, 3]]).as_matrix()
    return clip.wrist_pose_world[:, None, :3] + np.einsum("tij,tnj->tni", R_in, tips_solver)


def stack_metrics(metrics: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    """Per-frame optimizer metrics as one array per key."""
    keys = sorted({key for row in metrics for key in row})
    return {key: np.asarray([row.get(key, 0) for row in metrics]) for key in keys}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True, help="Merged runtime JSON config (robot + defaults).")
    parser.add_argument("--data", type=Path, default=None, help="Dataset root, or one motions/<clip>.npz.")
    parser.add_argument("--clip", type=str, default=None, help="Clip id (default motion.clip, else the first clip).")
    parser.add_argument("--out", type=Path, default=None, help="Result .npz (default retarget.out).")
    parser.add_argument("--scene", type=Path, default=None, help="Retarget scene XML (default robot.scene).")
    parser.add_argument(
        "--hand-xml", type=Path, default=None, help="Bare hand MJCF for the selfcol pass (default robot.xml)."
    )
    parser.add_argument("--start", type=int, default=None, help="First source frame (inclusive).")
    parser.add_argument("--end", type=int, default=None, help="Source frame end (exclusive).")
    parser.add_argument("--max-frames", type=int, default=None, help="Limit the solved frames (smoke runs).")
    parser.add_argument(
        "--set",
        dest="setting_overrides",
        action="append",
        default=[],
        metavar="KEY=VAL",
        help="Override one HandRetargetConfig field by dotted path, e.g. --set mesh.laplacian_weight=200",
    )
    return parser.parse_args()


def retarget_hand(
    config: dict[str, Any],
    *,
    data: str | Path | None = None,
    clip_id: str | None = None,
    out: str | Path | None = None,
    scene: str | Path | None = None,
    hand_xml: str | Path | None = None,
    start: int | None = None,
    end: int | None = None,
    max_frames: int | None = None,
    overrides: Sequence[str] = (),
) -> Path:
    """Solve the robot config's hand of one canonical clip and write its result."""
    motion = config_section(config, "motion")

    dataset_value = data if data is not None else motion.get("data")
    if dataset_value is None:
        raise ValueError("no dataset: pass --data or set motion.data")
    dataset, named = CanonicalDataset.locate(resolve_path(dataset_value, config), clip_id)
    clip_id = dataset.resolve_clip(named or clip_id or motion.get("clip"))
    paths = resolve_clip_paths(clip_id, dataset.root)
    hand = HandSpec.from_config(config, scene=scene, hand_xml=hand_xml)

    start = int(start if start is not None else motion.get("start", 0) or 0)
    end_value = end if end is not None else motion.get("end", -1)
    end = None if end_value is None or int(end_value) < 0 else int(end_value)
    max_frames_value = max_frames if max_frames is not None else motion.get("max_frames", 0)
    max_frames = None if not max_frames_value or int(max_frames_value) <= 0 else int(max_frames_value)

    out_value = out or config_section(config, "retarget").get("out")
    if out_value is None:
        raise ValueError("no output path: pass --out or set retarget.out")
    out_path = Path(resolve_path(out_value, config)).resolve()

    clip = load_canonical_clip(paths, hand, start=start, end=end, max_frames=max_frames)
    cfg = build_hand_config(config, overrides)

    compound = [
        name
        for name, directory in zip(clip.object_names, clip.object_dirs, strict=True)
        if (directory / COLLISION_SIDECAR).exists()
    ]
    print(
        f"[{LOG_PREFIX}] clip={clip.clip_id} dataset={dataset.root} robot={hand.name} side={hand.side}\n"
        f"[{LOG_PREFIX}] scene={hand.scene_xml} objects={', '.join(clip.object_names)}\n"
        f"[{LOG_PREFIX}] frames=[{clip.start_frame}, {clip.end_frame}) of {clip.source_frames} @ {clip.fps:g} fps "
        f"compound_collision={', '.join(compound) or 'none'}\n"
        f"[{LOG_PREFIX}] laplacian_w={cfg.mesh.laplacian_weight:g} object_points={cfg.object.sample_count} "
        f"sampler={cfg.object.sampler.mode} penetration={cfg.penetration.enabled_s2} "
        f"wrist_motion={cfg.wrist.motion_enabled}"
    )

    solve_start = time.perf_counter()
    qpos, object_points_local, metrics, solver, object_position, object_quaternion = solve_clip(clip, cfg, hand)
    solve_time = time.perf_counter() - solve_start
    if qpos.shape[1] != hand.nq:
        raise RuntimeError(f"solver returned nq={qpos.shape[1]}, expected {hand.nq}")

    clearance = ObjectClearance(
        hand=solver.retargeter.runtime.hand,
        wrist_qpos=qpos[:, :WRIST_DOF].copy(),
        object_position=object_position,
        object_quaternion_xyzw=object_quaternion,
        spec=PenetrationSpec(
            enabled=cfg.penetration.enabled_s2,
            tolerance=cfg.penetration.tolerance,
            query_threshold=cfg.mesh.interaction_radius_m,
        ),
    )
    cleaned, selfcol_info = optimize_trajectory(qpos[:, WRIST_DOF:], hand=hand, config=cfg.selfcol, clearance=clearance)
    qpos[:, WRIST_DOF:] = cleaned
    n = max(selfcol_info["n"], 1)
    print(
        f"[{LOG_PREFIX}] selfcol {format_penetration('before', selfcol_info['phi_before'])}\n"
        f"[{LOG_PREFIX}] selfcol {format_penetration('after ', selfcol_info['phi_after'])}\n"
        f"[{LOG_PREFIX}] selfcol link drift mean {selfcol_info['drift_mean']:.2f} / "
        f"p95 {selfcol_info['drift_p95']:.2f} / max {selfcol_info['drift_max']:.2f} mm; "
        f"windows {selfcol_info['windows']} iters {selfcol_info['iters']} "
        f"solve {selfcol_info['solve_s']:.1f}s ({1000 * selfcol_info['solve_s'] / n:.0f} ms/frame)"
    )

    penetration_mm = clearance.penetration_mm(qpos[:, WRIST_DOF:])

    tips_world = world_fingertips(solver.retargeter.runtime.hand, qpos, clip)
    src_tips = clip.landmarks_world[:, list(FINGERTIP_LANDMARKS)]
    tip_err = np.linalg.norm(tips_world - src_tips, axis=2)
    tip_err_mean, tip_err_max = float(tip_err.mean()), float(tip_err.max())

    scene_fk = FloatingSceneFK(hand, clip.object_meshes, contact_margin_m=cfg.mesh.contact_margin_m)
    materialized = scene_fk.materialize(qpos, clip.wrist_pose_world, clip.object_pose_world)
    qpos_scene = materialized["scene_qpos"]
    contact = materialized["link_object_contact"]

    frame_ids = np.arange(clip.start_frame, clip.end_frame, dtype=np.int32)
    payload: dict[str, Any] = {
        "qpos": np.asarray(qpos_scene, dtype=np.float64),
        "fps": np.asarray([clip.fps], dtype=np.float32),
        "frame_ids": frame_ids,
        "robot_xml": np.asarray(recorded_asset_path(hand.scene_xml)),
        "robot_name": np.asarray(hand.name),
        "robot_joint_names": np.asarray(scene_fk.scene_joint_names(), dtype=str),
        "robot_wrist_body": np.asarray(hand.wrist_body),
        "robot_root_body": np.asarray(hand.root_body),
        "source_data": np.asarray(recorded_path(paths.npz_path, dataset.root)),
        "source_dataset": np.asarray(dataset.name),
        "source_sequence_key": np.asarray(clip.clip_id),
        "source_format": np.asarray(CANONICAL_SCHEMA),
        "source_object_dirs": np.asarray([recorded_path(d, dataset.root) for d in clip.object_dirs]),
        "result_schema": np.asarray(RESULT_SCHEMA),
        "settings_digest": np.asarray(solver_settings_digest(config, list(overrides))),
    }
    diagnostics: dict[str, Any] = {
        "retarget_method": np.asarray(RETARGET_METHOD),
        "result_schema": np.asarray(RESULT_SCHEMA),
        "hand_side": np.asarray(hand.side),
        "hand_spec_json": np.asarray(hand.to_json()),
        "result_path": np.asarray(out_path.name),
        "qpos_solver": np.asarray(qpos, dtype=np.float64),
        "mediapipe_landmarks_w": np.asarray(clip.landmarks_world, dtype=np.float32),
        "object_pose_w": materialized["object_pose_w"],
        "reference_qpos": materialized["reference_qpos"],
        "wrist_pose_w": materialized["wrist_pose_w"],
        "link_pos_w": materialized["link_pos_w"],
        "link_object_contact": contact,
        "joint_names": np.asarray(hand.joint_names, dtype=str),
        "link_names": np.asarray(hand.link_names, dtype=str),
        "input_wrist_pose_w": np.asarray(clip.wrist_pose_world, dtype=np.float64),
        "object_names": np.asarray(clip.object_names),
        "object_meshes": np.asarray([recorded_path(path, dataset.root) for path in clip.object_meshes]),
        "object_points_local": np.asarray(object_points_local, dtype=np.float32),
        "hand_object_penetration_mm": np.asarray(penetration_mm, dtype=np.float32),
        "tips_world": np.asarray(tips_world, dtype=np.float32),
        "tip_error_m": np.asarray(tip_err, dtype=np.float32),
        "tip_err_mean": np.asarray([tip_err_mean]),
        "tip_err_max": np.asarray([tip_err_max]),
        "contact_frames_frac": np.asarray([float(contact.any(axis=1).mean())]),
        "source_fps": np.asarray([clip.fps], dtype=np.float32),
        "source_start_frame": np.asarray([clip.start_frame], dtype=np.int32),
        "source_end_frame": np.asarray([clip.end_frame], dtype=np.int32),
        "frame_id_domain": np.asarray("canonical_clip_frame"),
        "solve_time_s": np.asarray([solve_time]),
        "hand_config_json": np.asarray(json.dumps(asdict(cfg), sort_keys=True, default=str)),
    }
    for key, value in stack_metrics(metrics).items():
        diagnostics[f"metric_{key}"] = value
    if selfcol_info:
        diagnostics["selfcol_phi_before_m"] = np.asarray(selfcol_info["phi_before"], dtype=np.float32)
        diagnostics["selfcol_phi_after_m"] = np.asarray(selfcol_info["phi_after"], dtype=np.float32)
        diagnostics["selfcol_drift_mm"] = np.asarray(
            [selfcol_info["drift_mean"], selfcol_info["drift_p95"], selfcol_info["drift_max"]]
        )
        diagnostics["selfcol_windows"] = np.asarray([selfcol_info["windows"]], dtype=np.int32)
        diagnostics["selfcol_iters"] = np.asarray([selfcol_info["iters"]], dtype=np.int32)
        diagnostics["selfcol_solve_s"] = np.asarray([selfcol_info["solve_s"]])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **payload)
    np.savez_compressed(diagnostics_path(out_path), **diagnostics)

    print(
        f"[{LOG_PREFIX}] saved {out_path} qpos={qpos_scene.shape} "
        f"solve_time={solve_time:.1f}s ({1e3 * solve_time / len(qpos_scene):.1f} ms/frame) "
        f"tip_err mean={tip_err_mean * 1e3:.1f}mm max={tip_err_max * 1e3:.1f}mm "
        f"hand-obj penetration mean={penetration_mm.mean():.2f}mm max={penetration_mm.max():.2f}mm "
        f"contact frames={contact.any(axis=1).mean():.0%}"
    )
    print(f"[{LOG_PREFIX}] diagnostics: {diagnostics_path(out_path)}")
    return out_path


def main():
    args = parse_args()
    retarget_hand(
        load_config(args.config, use_default_extends=False),
        data=args.data,
        clip_id=args.clip,
        out=args.out,
        scene=args.scene,
        hand_xml=args.hand_xml,
        start=args.start,
        end=args.end,
        max_frames=args.max_frames,
        overrides=args.setting_overrides,
    )


if __name__ == "__main__":
    main()
