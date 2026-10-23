"""Solve every clip of a canonical dataset in parallel.

Examples:
    toporetarget batch --data data/grab --out output/grab --workers 8
"""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import json
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from toporetarget.config import (
    DEFAULT_CONFIG,
    diagnostics_path,
    example_robot_config,
    load_layered_config,
    motion_frame_range,
    read_npz,
    result_compatible,
    result_dir_name,
    result_filename,
    robot_config,
    runtime_config,
    safe_name,
    solver_settings_digest,
)
from toporetarget.dataset import CanonicalDataset
from toporetarget.pipeline import solver_command
from toporetarget.solver import OBJECT_MESH_NAME, check_objects

LOG = "Batch"
DEFAULT_ROBOT_CONFIGS = [example_robot_config(side) for side in ("right", "left")]


@dataclass(frozen=True)
class Robot:
    """One robot hand of the batch: its name, runtime config and result directory."""

    name: str
    config_path: Path
    runtime: dict[str, Any]
    result_dir: Path


def load_robots(
    config_paths: list[Path], defaults: Path, dataset: CanonicalDataset, out_root: Path
) -> dict[str, Robot]:
    """Hand side -> robot, one runtime config written per robot."""
    robots: dict[str, Robot] = {}
    for path in config_paths:
        config = load_layered_config(path, defaults)
        robot = robot_config(config)
        side = str(robot.get("hand_side", ""))
        if side not in ("left", "right"):
            raise SystemExit(f"{path}: robot.hand_side must be left or right, got {side!r}")
        if side in robots:
            raise SystemExit(f"two robot configs for the {side} hand: {robots[side].config_path} and {path}")
        name = safe_name(str(robot["name"]))
        runtime = runtime_config(config)
        runtime.setdefault("motion", {})["data"] = str(dataset.root)
        config_path = out_root / f"batch_runtime_config_{name}.json"
        config_path.write_text(json.dumps(runtime, indent=2, sort_keys=True) + "\n")
        result_dir = out_root / result_dir_name(name)
        result_dir.mkdir(parents=True, exist_ok=True)
        robots[side] = Robot(name=name, config_path=config_path, runtime=runtime, result_dir=result_dir)
    return robots


def measure(result_path: Path) -> dict[str, Any]:
    """Quality metrics of one result, read from its diagnostics sidecar."""
    diag = read_npz(diagnostics_path(result_path))
    penetration = np.asarray(diag["hand_object_penetration_mm"], dtype=np.float64)
    stalls = np.asarray(diag["metric_struct_infeas"] if "metric_struct_infeas" in diag else [], dtype=bool)
    overlap = np.asarray(diag["selfcol_phi_after_m"], dtype=np.float64) if "selfcol_phi_after_m" in diag else None
    tip_error = np.asarray(diag["tip_error_m"], dtype=np.float64).mean(axis=1)
    in_contact = np.asarray(diag["link_object_contact"], dtype=bool).any(axis=1)
    metrics = {
        "frames": int(len(penetration)),
        "tip_error_mean_mm": float(np.asarray(diag["tip_err_mean"]).reshape(-1)[0]) * 1e3,
        "tip_error_max_mm": float(np.asarray(diag["tip_err_max"]).reshape(-1)[0]) * 1e3,
        "tip_error_contact_mean_mm": float(tip_error[in_contact].mean()) * 1e3 if in_contact.any() else None,
        "contact_frames_frac": float(np.asarray(diag["contact_frames_frac"]).reshape(-1)[0]),
    }
    metrics["penetration_max_mm"] = float(penetration.max())
    metrics["selfcol_max_mm"] = float(max(0.0, -overlap.min()) * 1e3) if overlap is not None and overlap.size else 0.0
    metrics["stall_frac"] = float(stalls.mean()) if stalls.size else 0.0
    return metrics


def asset_error(dataset: CanonicalDataset, name: str) -> str | None:
    """Why MuJoCo does not compile one object asset of the dataset; None when it does."""
    try:
        check_objects([dataset.asset_dir(name) / OBJECT_MESH_NAME])
    except ValueError as error:
        return " | ".join(line.strip() for line in str(error).splitlines() if line.strip())
    return None


def solve_one(
    clip_id: str,
    hand: str,
    *,
    config_path: Path,
    dataset: CanonicalDataset,
    result_path: Path,
    log_path: Path,
    overrides: list[str],
) -> dict[str, Any]:
    """Run the solver for one hand of a clip as a subprocess, capturing its log."""
    cmd = solver_command(config_path, dataset, clip_id, result_path, overrides=overrides)

    started = time.perf_counter()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(cmd, capture_output=True, text=True)
    log_path.write_text(completed.stdout + completed.stderr)
    record: dict[str, Any] = {
        "clip_id": clip_id,
        "hand": hand,
        "result": str(result_path),
        "log": str(log_path),
        "elapsed_s": round(time.perf_counter() - started, 2),
        "returncode": completed.returncode,
        "status": "ok" if completed.returncode == 0 else "failed",
    }
    if completed.returncode != 0:
        tail = (completed.stderr or completed.stdout or "").strip().splitlines()[-5:]
        record["error"] = "\n".join(tail)
    return record


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--config",
        type=Path,
        action="append",
        default=None,
        help="Robot (hand) config, once per hand (default: the Wuji Hand 2 right and left configs).",
    )
    parser.add_argument("--defaults", type=Path, default=DEFAULT_CONFIG, help="Data and solver defaults.")
    parser.add_argument("--data", type=Path, required=True, help="Canonical dataset root.")
    parser.add_argument("--out", type=Path, required=True, help="Output root; results land in <out>/<robot>_retarget.")
    parser.add_argument("--workers", type=int, default=4, help="Parallel solver processes.")
    parser.add_argument("--clips", nargs="*", default=None, help="Solve only these clip ids.")
    parser.add_argument("--limit", type=int, default=None, help="Solve only the first N clips.")
    parser.add_argument("--force", action="store_true", help="Re-solve clips that already have a compatible result.")
    parser.add_argument(
        "--set",
        dest="setting_overrides",
        action="append",
        default=[],
        metavar="KEY=VAL",
        help="Override one HandRetargetConfig field (dotted path, repeatable).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.workers <= 0:
        raise SystemExit("--workers must be positive")

    dataset = CanonicalDataset.at(args.data)
    out_root = Path(args.out).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    robots = load_robots(args.config or DEFAULT_ROBOT_CONFIGS, args.defaults, dataset, out_root)

    clips = args.clips or dataset.clips()
    if args.limit is not None and args.limit > 0:
        clips = clips[: args.limit]
    if not clips:
        raise SystemExit(f"no clips under {dataset.motions}")

    pending: list[tuple[str, str, Robot]] = []
    records: list[dict[str, Any]] = []
    for clip_id in clips:
        for side in dataset.hands(clip_id):
            robot = robots.get(side)
            if robot is None:
                error = f"no robot config for the {side} hand"
                records.append({"clip_id": clip_id, "hand": side, "status": "no_robot", "error": error})
                continue
            frame_range = motion_frame_range(robot.runtime["motion"], dataset, clip_id)
            digest = solver_settings_digest(robot.runtime, args.setting_overrides)
            existing = robot.result_dir / result_filename(clip_id, robot.name)
            if not args.force and result_compatible(existing, dataset, clip_id, frame_range, digest):
                records.append({"clip_id": clip_id, "hand": side, "result": str(existing), "status": "skipped"})
                continue
            pending.append((clip_id, side, robot))

    assets = sorted({name for clip_id, _, _ in pending for name in dataset.objects(clip_id)})
    with futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        broken = {
            name: error
            for name, error in zip(assets, pool.map(lambda name: asset_error(dataset, name), assets), strict=True)
            if error is not None
        }
    for name, error in sorted(broken.items()):
        print(f"[{LOG}] asset {name}: {error}", flush=True)
    for clip_id, side, _ in [job for job in pending if broken.keys() & set(dataset.objects(job[0]))]:
        error = "; ".join(f"asset {name}: {broken[name]}" for name in dataset.objects(clip_id) if name in broken)
        records.append({"clip_id": clip_id, "hand": side, "status": "failed", "error": error})
    pending = [job for job in pending if not broken.keys() & set(dataset.objects(job[0]))]

    unsolved = sum(1 for row in records if row["status"] == "no_robot")
    print(
        f"[{LOG}] dataset={dataset.root} robots={', '.join(f'{side}={r.name}' for side, r in robots.items())} "
        f"clips={len(clips)} hands={len(records) + len(pending)} pending={len(pending)} "
        f"skipped={sum(1 for row in records if row['status'] == 'skipped')} no_robot={unsolved} "
        f"asset_failed={sum(1 for row in records if row['status'] == 'failed')} workers={args.workers}"
    )

    started = time.perf_counter()
    with futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        jobs = {
            pool.submit(
                solve_one,
                clip_id,
                side,
                config_path=robot.config_path,
                dataset=dataset,
                result_path=robot.result_dir / result_filename(clip_id, robot.name),
                log_path=out_root / "logs" / f"{safe_name(clip_id)}_{robot.name}.log",
                overrides=list(args.setting_overrides),
            ): clip_id
            for clip_id, side, robot in pending
        }
        done = 0
        for job in futures.as_completed(jobs):
            record = job.result()
            records.append(record)
            done += 1
            mark = "ok" if record["status"] == "ok" else "FAILED"
            print(
                f"[{LOG}] [{done}/{len(pending)}] {record['clip_id']} {record['hand']}: {mark} "
                f"({record['elapsed_s']:.1f}s)",
                flush=True,
            )

    records.sort(key=lambda row: (row["clip_id"], row["hand"]))
    solved = [row for row in records if row["status"] in ("ok", "skipped")]
    for row in solved:
        row["metrics"] = measure(Path(row["result"]))
    failures = [row for row in records if row["status"] == "failed"]
    summary = {
        "dataset": str(dataset.root),
        "robots": {side: {"name": r.name, "result_dir": str(r.result_dir)} for side, r in robots.items()},
        "elapsed_s": round(time.perf_counter() - started, 2),
        "counts": {
            "total": len(records),
            "ok": sum(1 for row in records if row["status"] == "ok"),
            "skipped": sum(1 for row in records if row["status"] == "skipped"),
            "failed": len(failures),
            "no_robot": unsolved,
        },
        "clips": records,
    }
    (out_root / "batch_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    (out_root / "batch_failures.json").write_text(json.dumps(failures, indent=2, sort_keys=True) + "\n")

    counts = summary["counts"]
    print(
        f"[{LOG}] done in {summary['elapsed_s']:.0f}s: ok={counts['ok']} "
        f"skipped={counts['skipped']} failed={counts['failed']}"
    )
    if solved:
        metrics = [row["metrics"] for row in solved]
        in_contact = [m["tip_error_contact_mean_mm"] for m in metrics if m["tip_error_contact_mean_mm"] is not None]
        print(
            f"[{LOG}] {len(solved)} results: fingertip error per hand and clip median "
            f"{np.median([m['tip_error_mean_mm'] for m in metrics]):.1f} mm "
            f"({np.median(in_contact) if in_contact else float('nan'):.1f} mm in contact), "
            f"hand-object penetration max {max(m['penetration_max_mm'] for m in metrics):.2f} mm, "
            f"finger overlap max {max(m['selfcol_max_mm'] for m in metrics):.2f} mm, "
            f"QP stalls max {max(m['stall_frac'] for m in metrics):.1%} of frames"
        )
    print(f"[{LOG}] summary: {out_root / 'batch_summary.json'}")
    if failures:
        print(f"[{LOG}] failures: {out_root / 'batch_failures.json'}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
