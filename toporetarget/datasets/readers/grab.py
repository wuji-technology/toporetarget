"""GRAB (https://grab.is.tue.mpg.de)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh

from toporetarget.datasets.readers import ManoTrack, Recording, poses_from_rotvec

REQUIRES: tuple[str, ...] = ()

NUM_PCA_COMPS = 24


def units(raw: Path) -> list[str]:
    root = Path(raw) / "grab"
    return sorted(str(p.relative_to(root)) for p in root.glob("*/*.npz"))


def load(raw: Path, unit: str) -> Recording:
    raw = Path(raw)
    with np.load(raw / "grab" / unit, allow_pickle=True) as data:
        fields = {key: data[key].item() for key in ("framerate", "n_comps", "obj_name", "rhand", "lhand", "object")}
    if int(fields["n_comps"]) != NUM_PCA_COMPS:
        raise ValueError(f"{unit}: {fields['n_comps']} MANO PCA components, expected {NUM_PCA_COMPS}")
    hands = {}
    for side in ("right", "left"):
        fit = fields[f"{side[0]}hand"]
        params = fit["params"]
        template = trimesh.load(raw / fit["vtemp"], process=False)
        hands[side] = ManoTrack(
            global_orient=np.asarray(params["global_orient"], dtype=np.float64),
            hand_pose=np.asarray(params["hand_pose"], dtype=np.float64),
            num_pca_comps=NUM_PCA_COMPS,
            flat_hand_mean=True,
            transl=np.asarray(params["transl"], dtype=np.float64),
            v_template=np.asarray(template.vertices, dtype=np.float64),
        )
    obj = fields["object"]["params"]
    object_pose = poses_from_rotvec(obj["global_orient"], obj["transl"])
    object_pose[:, :3, :3] = np.swapaxes(object_pose[:, :3, :3], 1, 2)
    subject, sequence = Path(unit).parent.name, Path(unit).stem
    return Recording(
        name=f"{subject}_{sequence}",
        times=np.arange(len(object_pose)) / float(fields["framerate"]),
        hands=hands,
        object_poses={str(fields["obj_name"]): object_pose},
    )


def load_mesh(raw: Path, object_id: str) -> trimesh.Trimesh:
    path = Path(raw) / "tools" / "object_meshes" / "contact_meshes" / f"{object_id}.ply"
    return trimesh.load(path, force="mesh", process=False)
