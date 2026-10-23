"""TACO (https://taco2024.github.io)."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import trimesh

from toporetarget.datasets.readers import ManoTrack, Recording

REQUIRES = ("torch",)

FPS = 30.0


def units(raw: Path) -> list[str]:
    root = Path(raw) / "Hand_Poses"
    return sorted(f"{d.parent.name}/{d.name}" for d in root.glob("*/*") if d.is_dir())


def _pickle(path: Path):
    with path.open("rb") as stream:
        return pickle.load(stream)


def load(raw: Path, unit: str) -> Recording:
    raw = Path(raw)
    triplet, sequence = unit.split("/")
    hands = {}
    for side in ("right", "left"):
        hand_dir = raw / "Hand_Poses" / unit
        if not (hand_dir / f"{side}_hand.pkl").is_file():
            continue
        frames = _pickle(hand_dir / f"{side}_hand.pkl")
        keys = sorted(frames)
        pose = np.stack([frames[k]["hand_pose"].numpy() for k in keys]).astype(np.float64)
        hands[side] = ManoTrack(
            global_orient=pose[:, :3],
            hand_pose=pose[:, 3:],
            num_pca_comps=None,
            flat_hand_mean=True,
            betas=_pickle(hand_dir / f"{side}_hand_shape.pkl")["hand_shape"].numpy().astype(np.float64),
            wrist=np.stack([frames[k]["hand_trans"].numpy() for k in keys]).astype(np.float64),
        )
    object_files = sorted((raw / "Object_Poses" / unit).glob("*_*.npy"))
    object_poses = {path.stem.split("_", 1)[1]: np.load(path).astype(np.float64) for path in object_files}
    if len(object_poses) != len(object_files):
        raise ValueError(f"{unit}: two object tracks share one object id")
    lengths = {len(pose) for pose in object_poses.values()} | {len(hand.wrist) for hand in hands.values()}
    if len(lengths) != 1:
        raise ValueError(f"{unit}: hand and object tracks differ in length {sorted(lengths)}")
    words = [w.replace(" ", "_") for w in triplet.strip("()").split(", ")]
    return Recording(
        name="__".join([*words, sequence]),
        times=np.arange(lengths.pop()) / FPS,
        hands=hands,
        object_poses=object_poses,
    )


def load_mesh(raw: Path, object_id: str) -> trimesh.Trimesh:
    path = Path(raw) / "object_models_released" / f"{object_id}_cm.obj"
    mesh = trimesh.load(path, force="mesh", process=False, skip_materials=True)
    mesh.apply_scale(0.01)
    return mesh
