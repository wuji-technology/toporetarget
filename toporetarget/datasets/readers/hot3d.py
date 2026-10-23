"""HOT3D (https://facebookresearch.github.io/hot3d)."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

from toporetarget.datasets.readers import ManoTrack, Recording

REQUIRES: tuple[str, ...] = ()

NUM_PCA_COMPS = 15
FRAME_SPACING_NS = 10_000_000
HAND_KEYS = {"left": "0", "right": "1"}
QUEST3_TO_ZUP = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])


def units(raw: Path) -> list[str]:
    root = Path(raw) / "dataset"
    return sorted(
        p.name
        for p in root.iterdir()
        if (p / "mano_hand_pose_trajectory.jsonl").is_file()
        and json.loads((p / "metadata.json").read_text()).get("have_hand_object_pose_gt")
    )


def _instances(raw: Path) -> dict[str, str]:
    """Map each object uid to its instance name."""
    table = json.loads((Path(raw) / "_assets" / "assets" / "instance.json").read_text())
    return {uid: entry["instance_name"] for uid, entry in table.items()}


def load(raw: Path, unit: str) -> Recording:
    raw = Path(raw)
    recording = raw / "dataset" / unit
    meta = json.loads((recording / "metadata.json").read_text())
    names = _instances(raw)

    hand_rows: dict[int, dict] = {}
    with (recording / "mano_hand_pose_trajectory.jsonl").open() as stream:
        for line in stream:
            entry = json.loads(line)
            hand_rows.setdefault(int(entry["timestamp_ns"]), entry["hand_poses"])
    object_rows: dict[int, dict[str, np.ndarray]] = {}
    with (recording / "dynamic_objects.csv").open() as stream:
        for row in csv.DictReader(stream):
            pose = np.eye(4)
            pose[:3, :3] = Rotation.from_quat([float(row[f"q_wo_{a}"]) for a in "xyzw"]).as_matrix()
            pose[:3, 3] = [float(row[f"t_wo_{a}[m]"]) for a in "xyz"]
            object_rows.setdefault(int(row["timestamp[ns]"]), {})[names[row["object_uid"]]] = pose

    stamps, last = [], None
    for stamp in sorted(hand_rows):
        if last is None or stamp - last >= FRAME_SPACING_NS:
            stamps.append(stamp)
            last = stamp

    n_frames = len(stamps)
    hands = {}
    for side, key in HAND_KEYS.items():
        rows = [hand_rows[t].get(key) for t in stamps]
        present = [row for row in rows if row is not None]
        if not present:
            continue
        orient = np.full((n_frames, 3), np.nan)
        transl = np.full((n_frames, 3), np.nan)
        pose = np.full((n_frames, NUM_PCA_COMPS), np.nan)
        for i, row in enumerate(rows):
            if row is None:
                continue
            quat_wxyz = np.asarray(row["wrist_xform"]["q_wxyz"], dtype=np.float64)
            orient[i] = Rotation.from_quat(quat_wxyz[[1, 2, 3, 0]]).as_rotvec()
            transl[i] = row["wrist_xform"]["t_xyz"]
            pose[i] = row["pose"]
        hands[side] = ManoTrack(
            global_orient=orient,
            hand_pose=pose,
            num_pca_comps=NUM_PCA_COMPS,
            flat_hand_mean=False,
            betas=np.asarray(present[0]["betas"], dtype=np.float64)[:10],
            transl=transl,
            fix_left_shapedirs=True,
        )

    object_poses = {}
    for uid in meta["object_uids"]:
        name = names[str(uid)]
        track = np.full((n_frames, 4, 4), np.nan)
        for i, stamp in enumerate(stamps):
            pose = object_rows.get(stamp, {}).get(name)
            if pose is not None:
                track[i] = pose
        object_poses[name] = track

    stamps_ns = np.asarray(stamps, dtype=np.int64)
    return Recording(
        name=unit,
        times=(stamps_ns - stamps_ns[0]) * 1e-9,
        hands=hands,
        object_poses=object_poses,
        to_zup=QUEST3_TO_ZUP if meta["headset"] == "Quest3" else np.eye(3),
    )


def load_mesh(raw: Path, object_id: str) -> trimesh.Trimesh:
    uid = {name: uid for uid, name in _instances(raw).items()}[object_id]
    return trimesh.load(Path(raw) / "_assets" / "assets" / f"{uid}.glb", force="mesh", process=False)
