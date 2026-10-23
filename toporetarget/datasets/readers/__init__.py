"""Readers for the public MANO hand-object datasets ``import_hoi.py`` imports."""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from types import ModuleType

import numpy as np

DATASETS = ("grab", "taco", "oakink2", "hot3d")

YUP_TO_ZUP = np.array([[-1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]])


@dataclass(frozen=True)
class ManoTrack:
    """One hand's MANO fit, in the release's world frame."""

    global_orient: np.ndarray
    hand_pose: np.ndarray
    num_pca_comps: int | None
    flat_hand_mean: bool
    betas: np.ndarray | None = None
    transl: np.ndarray | None = None
    wrist: np.ndarray | None = None
    v_template: np.ndarray | None = None
    fix_left_shapedirs: bool = False


@dataclass(frozen=True)
class Recording:
    """One recording: per-hand MANO tracks and per-object poses on one time base."""

    name: str
    times: np.ndarray
    hands: dict[str, ManoTrack]
    object_poses: dict[str, np.ndarray]
    to_zup: np.ndarray = field(default_factory=lambda: np.eye(3))


def reader(name: str) -> ModuleType:
    if name not in DATASETS:
        raise ValueError(f"unknown dataset {name!r}; supported: {', '.join(DATASETS)}")
    return importlib.import_module(f"toporetarget.datasets.readers.{name}")


def poses_from_rotvec(rotvec: np.ndarray, transl: np.ndarray) -> np.ndarray:
    """``(T, 4, 4)`` rigid poses from axis-angle rotations and translations."""
    from scipy.spatial.transform import Rotation

    pose = np.tile(np.eye(4), (len(transl), 1, 1))
    pose[:, :3, :3] = Rotation.from_rotvec(np.asarray(rotvec, dtype=np.float64)).as_matrix()
    pose[:, :3, 3] = transl
    return pose
