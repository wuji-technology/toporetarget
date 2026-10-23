"""Layered JSON/YAML config loading shared by the TopoRetarget commands."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

from toporetarget.dataset import CanonicalDataset

PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = PACKAGE_DIR / "defaults.json"
ROBOT_CONFIG_DIR = PACKAGE_DIR / "robots"
ASSETS_DIR = PACKAGE_DIR / "assets"


def example_robot_config(side: str) -> Path:
    """The robot config the package ships for one hand side (``"left"`` or ``"right"``)."""
    if side not in ("left", "right"):
        raise ValueError(f"hand side must be 'left' or 'right', got {side!r}")
    return ROBOT_CONFIG_DIR / f"wujihand2_{side}.json"


def _load_config_file(path: Path) -> dict[str, Any]:
    path = Path(path)
    text = path.read_text()
    if path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except ImportError as exc:
            raise ImportError(
                f"{path} is YAML, but PyYAML is not installed. Use a JSON config or install PyYAML."
            ) from exc
        data = yaml.safe_load(text)
    else:
        data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError(f"Config must be an object: {path}")
    return data


def _resolve_config_path(path: str | Path) -> Path:
    raw_path = Path(path)
    if raw_path.exists():
        return raw_path.resolve()
    if raw_path.is_absolute():
        return raw_path.resolve()
    candidate = ROBOT_CONFIG_DIR / raw_path.name
    if candidate.exists():
        return candidate.resolve()
    return raw_path.resolve()


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if key in {"extends", "_config_path", "_config_dir"}:
            continue
        current = merged.get(key)
        if isinstance(current, dict) and isinstance(value, dict):
            merged[key] = _deep_merge(current, value)
        else:
            merged[key] = value
    return merged


def _resolve_extend_path(value: str | Path, child_path: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path.resolve()
    candidate = child_path.resolve().parent / path
    if candidate.exists() or str(value).startswith("."):
        return candidate.resolve()
    return (PACKAGE_DIR / path).resolve()


def load_config(path: Path, use_default_extends: bool = True) -> dict[str, Any]:
    """Load one config, applying its ``extends`` chain (and the package's ``defaults.json``)."""
    path = _resolve_config_path(path)
    data = _load_config_file(path)
    extends = data.get("extends")
    if not extends and use_default_extends and path != DEFAULT_CONFIG and DEFAULT_CONFIG.exists():
        extends = str(DEFAULT_CONFIG)
    if extends:
        extend_values = extends if isinstance(extends, list) else [extends]
        merged: dict[str, Any] = {}
        for extend_value in extend_values:
            if not isinstance(extend_value, (str, Path)):
                raise ValueError(f"Config extends entries must be paths: {path}")
            parent = load_config(_resolve_extend_path(extend_value, path), use_default_extends=False)
            merged = _deep_merge(merged, parent)
        data = _deep_merge(merged, data)
    data["_config_path"] = str(path)
    data["_config_dir"] = str(path.parent)
    return data


def load_layered_config(config_path: Path, defaults_path: Path | None) -> dict[str, Any]:
    """Merge ``--defaults`` (task) under ``--config`` (robot), the usual entry point."""
    robot = load_config(config_path, use_default_extends=defaults_path is None)
    if defaults_path is None:
        return robot
    defaults = load_config(defaults_path, use_default_extends=False)
    merged = _deep_merge(defaults, robot)
    merged["_config_path"] = robot["_config_path"]
    merged["_config_dir"] = robot["_config_dir"]
    return merged


def clean_config(config: dict[str, Any]) -> dict[str, Any]:
    """Drop the private ``_config_*`` bookkeeping keys before serializing."""
    return {key: value for key, value in config.items() if not key.startswith("_config")}


def config_dir(config: dict[str, Any]) -> Path:
    return Path(config.get("_config_dir", Path.cwd())).resolve()


def resolve_path(value: str | Path | None, config: dict[str, Any], default: str | Path | None = None) -> Path | None:
    """Resolve a config path against the config's directory, then the working directory."""
    raw = default if value is None else value
    if raw is None:
        return None
    path = Path(raw)
    if path.is_absolute():
        return path.resolve()
    candidate = config_dir(config) / path
    if candidate.exists() or str(raw).startswith("."):
        return candidate.resolve()
    return path.resolve()


def section(config: dict[str, Any], name: str) -> dict[str, Any]:
    value = config.get(name, {})
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"Config section {name!r} must be an object.")
    return value


def cfg_get(config: dict[str, Any], dotted: str, default: Any = None) -> Any:
    """Read a dotted path out of a merged config, returning ``default`` if absent."""
    node: Any = config
    for key in dotted.split("."):
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    return node


def robot_config(config: dict[str, Any]) -> dict[str, Any]:
    robot = dict(section(config, "robot"))
    if not robot.get("name"):
        raise ValueError("Config requires robot.name.")
    if not robot.get("xml"):
        raise ValueError("Config requires robot.xml (the hand MJCF).")
    if not robot.get("scene"):
        raise ValueError("Config requires robot.scene (the single-hand retarget scene XML).")
    return robot


def bool_value(value: Any, default: bool) -> bool:
    return default if value is None else bool(value)


def list_of_args(values: dict[str, Any]) -> list[str]:
    """Turn a dict of view/solver options into ``--flag value`` CLI arguments."""
    args: list[str] = []
    for key, value in values.items():
        flag = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            if value:
                args.append(flag)
            continue
        if value is None:
            continue
        if isinstance(value, (list, tuple)):
            args.append(flag)
            args.extend(str(item) for item in value)
            continue
        args.extend([flag, str(value)])
    return args


def safe_name(value: str) -> str:
    """Filesystem-safe token used in result paths and dataset directory names."""
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in str(value))


RUNTIME_PATH_KEYS: tuple[tuple[str, str], ...] = (("robot", "scene"), ("robot", "xml"))


def runtime_config(config: dict[str, Any]) -> dict[str, Any]:
    """A cleaned deep copy of ``config`` with every path key absolutized."""
    runtime = copy.deepcopy(clean_config(config))
    for parent, key in RUNTIME_PATH_KEYS:
        block = runtime.get(parent)
        if isinstance(block, dict) and block.get(key):
            block[key] = str(resolve_path(block[key], config))
    return runtime


RETARGET_METHOD = "toporetarget"
CANONICAL_SCHEMA = "toporetarget_canonical"
COLLISION_SIDECAR = "collision.json"
RESULT_SCHEMA = "toporetarget_result"
DIAGNOSTICS_SUFFIX = "_diagnostics.npz"


def result_dir_name(robot_name: str) -> str:
    """``<robot>_retarget``, the directory a robot's results land in."""
    return f"{safe_name(robot_name)}_retarget"


def result_filename(clip_id: str, robot_name: str) -> str:
    """``<clip>_toporetarget_<robot>.npz``, the result layout every command shares."""
    return f"{safe_name(clip_id)}_{RETARGET_METHOD}_{safe_name(robot_name)}.npz"


def default_result_path(dataset_name: str, robot_name: str, clip_id: str) -> Path:
    """``output/<dataset>/<robot>_retarget/<clip>_toporetarget_<robot>.npz`` in the working directory."""
    return Path.cwd() / "output" / dataset_name / result_dir_name(robot_name) / result_filename(clip_id, robot_name)


def recorded_path(path: Path, root: Path) -> str:
    """``path`` relative to ``root``, the form a result records its inputs in."""
    try:
        return str(Path(path).resolve().relative_to(Path(root).resolve()))
    except ValueError:
        raise ValueError(f"{path} is not under {root}, so it cannot be recorded relative to it") from None


def recorded_asset_path(path: Path) -> str:
    """A robot scene or hand MJCF as a result records it."""
    for root in (ASSETS_DIR, Path.cwd()):
        try:
            return recorded_path(path, root)
        except ValueError:
            continue
    raise ValueError(f"{path} is neither under {ASSETS_DIR} nor under the working directory {Path.cwd()}")


def diagnostics_path(result_path: Path) -> Path:
    """The ``<result>_diagnostics.npz`` sidecar next to a result ``.npz``."""
    result_path = Path(result_path)
    return result_path.with_name(result_path.stem + DIAGNOSTICS_SUFFIX)


def read_npz(path: str | Path) -> dict[str, Any]:
    """Read result or diagnostics arrays without pickle unless TOPORETARGET_ALLOW_PICKLE=1 explicitly opts in."""
    import os

    import numpy as np

    try:
        with np.load(path, allow_pickle=False) as data:
            return {key: data[key] for key in data.files}
    except ValueError as exc:
        if "allow_pickle" not in str(exc):
            raise
        if os.environ.get("TOPORETARGET_ALLOW_PICKLE") == "1":
            print(f"[{RETARGET_METHOD}] reading {path} with pickle enabled (TOPORETARGET_ALLOW_PICKLE=1)")
            with np.load(path, allow_pickle=True) as data:
                return {key: data[key] for key in data.files}
        raise ValueError(
            f"{path} holds pickled object arrays, which TopoRetarget does not load; "
            "if it is an older TopoRetarget result, solve the clip again, "
            "or set TOPORETARGET_ALLOW_PICKLE=1 to read a file you trust"
        ) from exc


def discover_results(result_dir: str | Path) -> list[Path]:
    """Every result ``.npz`` in a directory, excluding the diagnostics sidecars."""
    return sorted(p for p in Path(result_dir).glob("*.npz") if not p.stem.endswith("_diagnostics"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def solver_settings_digest(config: dict[str, Any], overrides: list[str]) -> str:
    """Digest of what a solve depends on besides its clip and object asset."""
    payload = {"hand_solver": config.get("hand_solver") or {}, "set": list(overrides)}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def motion_frame_range(motion: dict[str, Any], dataset: CanonicalDataset, clip_id: str) -> tuple[int, int]:
    """The ``[start, end)`` the solver takes for one clip under a ``motion`` block."""
    frames = dataset.frames(clip_id)
    start = max(int(motion.get("start", 0) or 0), 0)
    end_value = motion.get("end", -1)
    end = frames if end_value is None or int(end_value) < 0 else min(int(end_value), frames)
    max_frames = int(motion.get("max_frames", 0) or 0)
    if max_frames > 0:
        end = min(end, start + max_frames)
    return start, end


def result_mismatch(
    result_path: Path, dataset: CanonicalDataset, clip_id: str, frame_range: tuple[int, int], settings_digest: str
) -> str | None:
    """Why ``result_path`` is NOT a current solve of ``clip_id``; None when it is."""
    import numpy as np

    result_path = Path(result_path)
    sidecar = diagnostics_path(result_path)
    if not result_path.exists():
        return "no result yet"
    if not sidecar.exists():
        return f"diagnostics sidecar missing: {sidecar}"
    try:
        data = read_npz(result_path)
        saved = str(np.asarray(data["source_sequence_key"]).item())
        if saved != str(clip_id):
            return f"saved clip={saved}, expected {clip_id}"
        schema = str(np.asarray(data["result_schema"]).item())
        if schema != RESULT_SCHEMA:
            return f"result schema {schema}, expected {RESULT_SCHEMA}"
        frame_ids = np.asarray(data["frame_ids"])
        saved_range = None if frame_ids.size == 0 else (int(frame_ids[0]), int(frame_ids[-1]) + 1)
        if saved_range != tuple(frame_range):
            return f"solved frames {saved_range}, expected {tuple(frame_range)}"
        if str(np.asarray(data["settings_digest"]).item()) != settings_digest:
            return "solved with other settings"
        if str(np.asarray(read_npz(sidecar)["retarget_method"]).item()) != RETARGET_METHOD:
            return "the result was produced by another method"
        newest = max(dataset.clip_inputs(clip_id), key=lambda p: p.stat().st_mtime_ns)
        if newest.stat().st_mtime_ns > result_path.stat().st_mtime_ns:
            return f"{newest.name} is newer than the result"
    except Exception as exc:
        return f"unreadable result: {exc}"
    return None


def result_compatible(
    result_path: Path, dataset: CanonicalDataset, clip_id: str, frame_range: tuple[int, int], settings_digest: str
) -> bool:
    """True when ``result_path`` is a current solve of ``clip_id`` (see ``result_mismatch``)."""
    return result_mismatch(result_path, dataset, clip_id, frame_range, settings_digest) is None
