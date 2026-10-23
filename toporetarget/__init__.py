"""TopoRetarget: object-centric interaction-mesh retargeting of human hand-object motion to a Wuji Hand 2."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

_LAZY = {
    "toporetarget.solver": (
        "HandRetargetConfig",
        "HandSpec",
        "FingerSpec",
        "FrameSolver",
        "FrameState",
        "HandObjectState",
        "CanonicalClip",
        "build_hand_config",
        "derive_input_wrist_pose",
        "load_canonical_clip",
        "resolve_clip_paths",
        "retarget_hand",
        "solve_clip",
    ),
    "toporetarget.dataset": ("CanonicalDataset",),
    "toporetarget.config": (
        "DEFAULT_CONFIG",
        "ROBOT_CONFIG_DIR",
        "diagnostics_path",
        "load_config",
        "load_layered_config",
        "runtime_config",
    ),
}
_MODULE_OF = {name: module for module, names in _LAZY.items() for name in names}

__all__ = ["retarget", "example_robot_config", *sorted(_MODULE_OF)]


def __getattr__(name: str) -> Any:
    module = _MODULE_OF.get(name)
    if module is None:
        raise AttributeError(f"module 'toporetarget' has no attribute {name!r}")
    import importlib

    value = getattr(importlib.import_module(module), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))


def example_robot_config(hand: str) -> Path:
    """The Wuji Hand 2 robot config of one hand side (``"left"`` or ``"right"``)."""
    from toporetarget.config import example_robot_config as robot_config_path

    return robot_config_path(hand)


def retarget(
    data: str | Path,
    clip: str | None = None,
    *,
    hands: Sequence[str] | None = None,
    config: str | Path | None = None,
    defaults: str | Path | None = None,
    out: str | Path | None = None,
    start: int | None = None,
    end: int | None = None,
    max_frames: int | None = None,
    overrides: Sequence[str] = (),
) -> list[Path]:
    """Retarget the hands of one canonical clip in this process and return the result paths.

    Args:
        data: A canonical dataset root, or one ``motions/<clip>.npz``.
        clip: The clip id; default the dataset's first clip.
        hands: The hands to solve, ``"right"`` and/or ``"left"``; default every hand the clip holds.
        config: One robot config to solve on, instead of the example config of each hand's side.
        defaults: The data and solver settings file; default the package's ``defaults.json``.
        out: The result path, for one hand only; default
            ``output/<dataset>/<robot>_retarget/<clip>_toporetarget_<robot>.npz``.
        start: The first source frame solved (inclusive).
        end: The source frame the solve stops before (exclusive).
        max_frames: The most frames solved, from ``start``.
        overrides: ``KEY=VAL`` settings, as ``--set`` takes them.
    """
    from toporetarget.config import (
        DEFAULT_CONFIG,
        default_result_path,
        load_layered_config,
        robot_config,
        runtime_config,
    )
    from toporetarget.dataset import CanonicalDataset
    from toporetarget.solver import retarget_hand

    dataset, named = CanonicalDataset.locate(data, clip)
    clip = dataset.resolve_clip(named or clip)

    if config is not None:
        if hands is not None:
            raise ValueError("pass either config (one robot) or hands, not both")
        robot_configs = [Path(config)]
    else:
        clip_hands = dataset.hands(clip)
        chosen = list(clip_hands if hands is None else hands)
        missing = [hand for hand in chosen if hand not in clip_hands]
        if missing:
            raise ValueError(f"clip {clip!r} holds the hands {clip_hands}, not {missing}")
        robot_configs = [example_robot_config(hand) for hand in chosen]
    if out is not None and len(robot_configs) != 1:
        raise ValueError(f"out names one result, but {len(robot_configs)} hands are solved; pick one")

    results: list[Path] = []
    for robot_config_path in robot_configs:
        merged = runtime_config(load_layered_config(robot_config_path, Path(defaults or DEFAULT_CONFIG)))
        robot_name = str(robot_config(merged)["name"])
        result = Path(out).resolve() if out is not None else default_result_path(dataset.name, robot_name, clip)
        results.append(
            retarget_hand(
                merged,
                data=dataset.root,
                clip_id=clip,
                out=result,
                start=start,
                end=end,
                max_frames=max_frames,
                overrides=list(overrides),
            )
        )
    return results
