"""OakInk2 (https://oakink.net/v2)."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

from toporetarget.datasets.readers import YUP_TO_ZUP, ManoTrack, Recording

REQUIRES = ("torch",)

FPS = 120.0


def units(raw: Path) -> list[str]:
    return sorted(p.stem for p in (Path(raw) / "anno_preview").glob("*.pkl"))


def _mesh_path(raw: Path, object_id: str) -> Path:
    return Path(raw) / "object_repair" / "align_ds" / object_id / "model.obj"


def load(raw: Path, unit: str) -> Recording:
    raw = Path(raw)
    with (raw / "anno_preview" / f"{unit}.pkl").open("rb") as stream:
        anno = pickle.load(stream)
    frames = list(anno["mocap_frame_id_list"])
    mano = [anno["raw_mano"][f] for f in frames]
    hands = {}
    for side, prefix in (("right", "rh"), ("left", "lh")):
        quat_wxyz = np.concatenate([m[f"{prefix}__pose_coeffs"].numpy() for m in mano]).astype(np.float64)
        rotvec = Rotation.from_quat(quat_wxyz.reshape(-1, 4)[:, [1, 2, 3, 0]]).as_rotvec().reshape(len(frames), 16, 3)
        hands[side] = ManoTrack(
            global_orient=rotvec[:, 0],
            hand_pose=rotvec[:, 1:].reshape(len(frames), 45),
            num_pca_comps=None,
            flat_hand_mean=True,
            betas=np.concatenate([m[f"{prefix}__betas"].numpy() for m in mano]).astype(np.float64),
            wrist=np.concatenate([m[f"{prefix}__tsl"].numpy() for m in mano]).astype(np.float64),
        )
    object_poses = {
        str(obj): np.stack([anno["obj_transf"][obj][f] for f in frames]).astype(np.float64)
        for obj in anno["obj_transf"]
        if _mesh_path(raw, str(obj)).is_file()
    }
    return Recording(
        name=unit,
        times=np.asarray(frames, dtype=np.float64) / FPS,
        hands=hands,
        object_poses=object_poses,
        to_zup=YUP_TO_ZUP,
    )


def load_mesh(raw: Path, object_id: str) -> trimesh.Trimesh:
    return trimesh.load(_mesh_path(raw, object_id), force="mesh", process=False, skip_materials=True)
