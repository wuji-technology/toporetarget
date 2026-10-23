"""Run the TopoRetarget hand pipeline on one clip and play the result."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from toporetarget.config import (
    DEFAULT_CONFIG,
    bool_value,
    clean_config,
    default_result_path,
    example_robot_config,
    list_of_args,
    load_config,
    load_layered_config,
    motion_frame_range,
    resolve_path,
    result_mismatch,
    robot_config,
    runtime_config,
    safe_name,
    section,
    solver_settings_digest,
)
from toporetarget.config import (
    RETARGET_METHOD as METHOD,
)
from toporetarget.dataset import CanonicalDataset

PYTHON = sys.executable
LOG = "Pipeline"

VIEW_FALSE_FLAGS = {
    "show_object": "--no-show-object",
    "show_source_landmarks": "--no-show-source-landmarks",
    "show_contact_links": "--no-show-contact-links",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--config", type=Path, default=None, help="Robot (hand) config (default: every hand of the clip)."
    )
    parser.add_argument("--defaults", type=Path, default=DEFAULT_CONFIG, help="Data and solver defaults.")
    parser.add_argument(
        "--stage",
        choices=["all", "prepare", "retarget", "view"],
        default="all",
        help="prepare stops after resolving the dataset and clip; view replays a result.",
    )
    parser.add_argument("--data", type=Path, default=None, help="Dataset root, or one motions/<clip>.npz.")
    parser.add_argument("--clip", type=str, default=None, help="Clip id (default motion.clip, else the first clip).")
    parser.add_argument("--out", type=Path, default=None, help="Override the result .npz (one hand only).")
    parser.add_argument("--start", type=int, default=None, help="First source frame (inclusive).")
    parser.add_argument("--end", type=int, default=None, help="Source frame end (exclusive).")
    parser.add_argument("--max-frames", type=int, default=None, help="Limit the solved frames (smoke runs).")
    parser.add_argument(
        "--set",
        dest="setting_overrides",
        action="append",
        default=[],
        metavar="KEY=VAL",
        help="Override one HandRetargetConfig field (dotted path, repeatable).",
    )
    parser.add_argument("--force-retarget", action="store_true", help="Rebuild a compatible existing result.")
    parser.add_argument("--skip-view", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def run_command(cmd: list[str], *, dry_run: bool) -> None:
    print(f"[{LOG}] " + " ".join(str(part) for part in cmd), flush=True)
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def resolve_dataset(config: dict[str, Any], args) -> tuple[CanonicalDataset, str]:
    """The dataset and clip the arguments name, falling back to the ``motion`` block, then to the first clip."""
    motion = section(config, "motion")
    value = args.data if args.data is not None else motion.get("data")
    if value is None:
        raise ValueError("no dataset: pass --data or set motion.data in the defaults")
    dataset, named = CanonicalDataset.locate(resolve_path(value, config), args.clip)
    clip = named or args.clip or motion.get("clip")
    return dataset, dataset.resolve_clip(None if clip is None else str(clip))


def example_robot_configs(dataset: CanonicalDataset, clip_id: str) -> list[Path]:
    """The packaged robot config of each hand the clip holds."""
    return [example_robot_config(hand) for hand in dataset.hands(clip_id)]


def solver_command(
    config_path: Path,
    dataset: CanonicalDataset,
    clip_id: str,
    result_path: Path,
    *,
    overrides: list[str],
) -> list[str]:
    """The ``python -m toporetarget.solver`` command line for one hand of one clip (the batch runs the same)."""
    cmd = [PYTHON, "-u", "-m", "toporetarget.solver", "--config", str(config_path)]
    cmd += ["--data", str(dataset.root), "--clip", str(clip_id), "--out", str(result_path)]
    for kv in overrides:
        cmd.extend(["--set", kv])
    return cmd


def make_runtime_config(
    base_config: dict[str, Any],
    dataset: CanonicalDataset,
    clip_id: str,
    work_dir: Path,
    args,
) -> tuple[dict[str, Any], Path]:
    """Write the merged runtime config the solver reads, and return it."""
    config = runtime_config(base_config)
    motion = config.setdefault("motion", {})
    motion["data"] = str(dataset.root)
    motion["clip"] = str(clip_id)
    for key in ("start", "end", "max_frames"):
        value = getattr(args, key)
        if value is not None:
            motion[key] = value
    retarget = config.setdefault("retarget", {})
    if args.out is not None:
        retarget["out"] = str(Path(args.out).resolve())
    elif not retarget.get("out"):
        retarget["out"] = str(default_result_path(dataset.name, str(robot_config(config)["name"]), clip_id))

    robot_name = safe_name(str(robot_config(config)["name"]))
    temp_path = work_dir / "configs" / f"{safe_name(clip_id)}_{METHOD}_{robot_name}_runtime_config.json"
    config["_config_path"] = str(temp_path)
    config["_config_dir"] = str(temp_path.parent)
    if args.dry_run:
        print(f"[{LOG}] would write runtime config: {temp_path}")
    else:
        temp_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path.write_text(json.dumps(clean_config(config), indent=2, sort_keys=True) + "\n")
    return config, temp_path


def run_solver(config: dict[str, Any], config_path: Path, dataset: CanonicalDataset, clip_id: str, args) -> Path:
    result_path = Path(config["retarget"]["out"])
    if not args.force_retarget:
        frame_range = motion_frame_range(section(config, "motion"), dataset, clip_id)
        digest = solver_settings_digest(config, args.setting_overrides)
        reason = result_mismatch(result_path, dataset, clip_id, frame_range, digest)
        if reason is None:
            print(f"[{LOG}] reuse retarget result: {result_path}")
            return result_path
        if result_path.exists():
            print(f"[{LOG}] rebuild retarget result: {result_path}: {reason}")

    cmd = solver_command(config_path, dataset, clip_id, result_path, overrides=args.setting_overrides)
    run_command(cmd, dry_run=args.dry_run)
    return result_path


def visualize_results(config: dict[str, Any], result_paths: list[Path], *, dry_run: bool) -> None:
    view = section(config, "view")
    if view.get("enabled", True) is False:
        print(f"[{LOG}] view disabled by config.")
        return
    view_args: dict[str, Any] = {
        "result": [str(path) for path in result_paths],
        "play": bool_value(view.get("play"), True),
    }
    for key, value in (view.get("extra_args", {}) or {}).items():
        view_args[key] = value
    cmd = [PYTHON, "-m", "toporetarget.viewer", *list_of_args(view_args)]
    for key, flag in VIEW_FALSE_FLAGS.items():
        if view_args.get(key) is False:
            cmd.append(flag)
    run_command(cmd, dry_run=dry_run)


def run_pipeline(args, work_dir: Path) -> None:
    defaults = load_config(args.defaults, use_default_extends=False)
    dataset, clip_id = resolve_dataset(defaults, args)
    robot_configs = [args.config] if args.config is not None else example_robot_configs(dataset, clip_id)
    if args.out is not None and len(robot_configs) > 1:
        raise ValueError(
            f"--out names one result, but {clip_id} has {len(robot_configs)} hands; pick one with --config"
        )

    configs: list[dict[str, Any]] = []
    result_paths: list[Path] = []
    for robot_config_path in robot_configs:
        base_config = load_layered_config(robot_config_path, args.defaults)
        robot = robot_config(base_config)
        config, config_path = make_runtime_config(base_config, dataset, clip_id, work_dir, args)
        print(
            f"[{LOG}] robot={robot['name']} dataset={dataset.root} clip={clip_id}\n"
            f"[{LOG}] scene={resolve_path(robot['scene'], base_config)} "
            f"hand={resolve_path(robot['xml'], base_config)}\n"
            f"[{LOG}] runtime config={config_path} result={config['retarget']['out']}",
            flush=True,
        )
        if args.stage == "prepare":
            continue
        result_path = Path(config["retarget"]["out"])
        if args.stage in {"all", "retarget"}:
            result_path = run_solver(config, config_path, dataset, clip_id, args)
        configs.append(config)
        result_paths.append(result_path)

    if args.stage == "view" or (args.stage == "all" and not args.skip_view):
        missing = [path for path in result_paths if not path.exists()]
        if missing and not args.dry_run:
            raise FileNotFoundError(f"retarget result not found: {missing[0]}; run --stage retarget first")
        visualize_results(configs[0], result_paths, dry_run=args.dry_run)


def main():
    args = parse_args()
    with tempfile.TemporaryDirectory(prefix="toporetarget_pipeline_") as work_dir:
        run_pipeline(args, Path(work_dir))


if __name__ == "__main__":
    main()
