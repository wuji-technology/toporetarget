#!/usr/bin/env python3
"""Merge several canonical TopoRetarget datasets into one dataset root.

Examples:
    toporetarget import-capture --source /path/to/bundle1 --out data/_staging/bundle1
    toporetarget import-capture --source /path/to/bundle2 --out data/_staging/bundle2
    toporetarget merge --out data/merged \
        data/_staging/bundle1 data/_staging/bundle2
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from toporetarget.config import CANONICAL_SCHEMA, sha256_file

LOG_PREFIX = "Merge"


class MergeError(ValueError):
    """Two inputs cannot be joined into one canonical dataset."""


def asset_fingerprint(asset_dir: Path) -> dict[str, str]:
    """Digest every file of one asset directory, convex parts included, so a mismatch is provable."""
    return {
        path.relative_to(asset_dir).as_posix(): sha256_file(path)
        for path in sorted(asset_dir.rglob("*"))
        if path.is_file()
    }


def read_dataset(root: Path) -> dict[str, Any]:
    """Load one canonical dataset manifest, rejecting a non-dataset directory."""
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise MergeError(f"{root} is not a canonical dataset (no manifest.json)")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != CANONICAL_SCHEMA:
        raise MergeError(f"{manifest_path}: expected schema {CANONICAL_SCHEMA!r}")
    if not (root / "motions").is_dir():
        raise MergeError(f"{root} is not a canonical dataset (no motions/ directory)")
    return manifest


def merge_datasets(inputs: list[Path], out: Path, *, label: str | None = None, force: bool = False) -> dict[str, Any]:
    """Join canonical datasets into ``out``; returns the merged manifest."""
    if not inputs:
        raise MergeError("no input datasets")
    out = Path(out).expanduser().resolve()
    inputs = [Path(root).expanduser().resolve() for root in inputs]
    for root in inputs:
        if out == root or out.is_relative_to(root) or root.is_relative_to(out):
            raise MergeError(f"--out {out} overlaps the input {root}")
    if out.exists() and not force:
        raise FileExistsError(f"{out} already exists (pass --force to replace it)")

    merged: dict[str, Any] = {
        "schema": CANONICAL_SCHEMA,
        "dataset": label or out.name,
        "merged_from": [],
        "handedness": None,
        "target_fps": None,
        "sort": "clip id ascending",
        "assets": {},
        "clips": [],
    }
    fingerprints: dict[str, dict[str, str]] = {}
    clip_origin: dict[str, Path] = {}

    staging = out.parent / f".{out.name}.merge"
    if staging.exists():
        shutil.rmtree(staging)
    (staging / "motions").mkdir(parents=True)
    try:
        for root in inputs:
            manifest = read_dataset(root)

            if merged["handedness"] is None:
                merged["handedness"] = manifest["handedness"]
                merged["target_fps"] = float(manifest["target_fps"])
            if manifest["handedness"] != merged["handedness"]:
                merged["handedness"] = "both"
            if float(manifest["target_fps"]) != merged["target_fps"]:
                raise MergeError(f"{root}: target_fps {manifest['target_fps']} != {merged['target_fps']}")

            for asset_id in sorted(manifest["assets"]):
                source_asset = root / "assets" / asset_id
                fingerprint = asset_fingerprint(source_asset)
                if asset_id in fingerprints:
                    if fingerprints[asset_id] != fingerprint:
                        raise MergeError(
                            f"object asset {asset_id!r} differs between inputs; {root} does not match "
                            f"the first dataset that supplied it"
                        )
                else:
                    fingerprints[asset_id] = fingerprint
                    shutil.copytree(source_asset, staging / "assets" / asset_id)
                    merged["assets"][asset_id] = manifest["assets"][asset_id]

            for clip in manifest["clips"]:
                clip_id = clip["clip_id"]
                if clip_id in clip_origin:
                    raise MergeError(
                        f"duplicate clip id {clip_id!r} in {root} (already supplied by {clip_origin[clip_id]})"
                    )
                clip_origin[clip_id] = root
                for suffix in (".npz", ".meta.json"):
                    shutil.copy2(root / "motions" / f"{clip_id}{suffix}", staging / "motions" / f"{clip_id}{suffix}")
                merged["clips"].append({**clip, "source_bundle": manifest.get("source"), "source_dataset": root.name})

            merged["merged_from"].append(
                {
                    "dataset": root.name,
                    "bundle": manifest.get("source"),
                    "clips": len(manifest["clips"]),
                    "source_declared_fps": manifest.get("source_declared_fps"),
                }
            )

        merged["clips"].sort(key=lambda clip: clip["clip_id"])
        for ordinal, clip in enumerate(merged["clips"], start=1):
            clip["ordinal"] = ordinal
        merged["clip_count"] = len(merged["clips"])
        merged["frame_count"] = sum(int(clip["target_frames"]) for clip in merged["clips"])
        merged["duration_s"] = round(merged["frame_count"] / merged["target_fps"], 3)

        (staging / "manifest.json").write_text(json.dumps(merged, indent=2, sort_keys=True) + "\n")
        if out.exists():
            shutil.rmtree(out)
        staging.replace(out)
    finally:
        if staging.exists():
            shutil.rmtree(staging)

    print(
        f"[{LOG_PREFIX}] {merged['clip_count']} clips from {len(inputs)} datasets, "
        f"{merged['frame_count']} frames @ {merged['target_fps']:g} Hz "
        f"({merged['duration_s']:.1f} s), assets {sorted(merged['assets'])} -> {out}",
        flush=True,
    )
    return merged


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("inputs", type=Path, nargs="+", help="Canonical dataset roots to merge.")
    parser.add_argument("--out", type=Path, required=True, help="Merged dataset root.")
    parser.add_argument("--label", type=str, default=None, help="Dataset name recorded in the manifest.")
    parser.add_argument("--force", action="store_true", help="Replace an existing output dataset.")
    return parser.parse_args()


def main():
    args = parse_args()
    merge_datasets(args.inputs, args.out, label=args.label, force=args.force)


if __name__ == "__main__":
    main()
