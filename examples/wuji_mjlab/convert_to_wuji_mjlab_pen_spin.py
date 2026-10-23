"""Convert single-pen, right-hand, 50 Hz results for wuji-mjlab pen_spin only.

Compatible with wuji-mjlab v2026.10.22 (b2b98d7). This checkout-only example
supports results solved from the pen-spin raw capture package.
The original --source-clip and --asset-json are required to restore the capture's
world origin and pen coordinates. Requires NumPy, MuJoCo and wuji-description.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HAND_XML = ROOT / "toporetarget/assets/wuji-description/hand2/hand2_2/body/mjcf/right.xml"
BODY = {
    "r_thumb_proximal": "right_finger1_link1",
    "r_thumb_proximal_abd": "right_finger1_link2",
    "r_thumb_middle": "right_finger1_link3",
    "r_thumb_distal": "right_finger1_link4",
    "r_index_finger_proximal": "right_finger2_link1",
    "r_index_finger_proximal_abd": "right_finger2_link2",
    "r_index_finger_middle": "right_finger2_link3",
    "r_index_finger_distal": "right_finger2_link4",
    "r_middle_finger_proximal": "right_finger3_link1",
    "r_middle_finger_proximal_abd": "right_finger3_link2",
    "r_middle_finger_middle": "right_finger3_link3",
    "r_middle_finger_distal": "right_finger3_link4",
    "r_ring_finger_proximal": "right_finger4_link1",
    "r_ring_finger_proximal_abd": "right_finger4_link2",
    "r_ring_finger_middle": "right_finger4_link3",
    "r_ring_finger_distal": "right_finger4_link4",
    "r_pinky_proximal": "right_finger5_link1",
    "r_pinky_proximal_abd": "right_finger5_link2",
    "r_pinky_middle": "right_finger5_link3",
    "r_pinky_distal": "right_finger5_link4",
}
JOINT = {
    "r_thumb_cmc_flex": "right_finger1_joint1",
    "r_thumb_cmc_abd": "right_finger1_joint2",
    "r_thumb_mcp": "right_finger1_joint3",
    "r_thumb_ip": "right_finger1_joint4",
    "r_index_finger_mcp_flex": "right_finger2_joint1",
    "r_index_finger_mcp_abd": "right_finger2_joint2",
    "r_index_finger_pip": "right_finger2_joint3",
    "r_index_finger_dip": "right_finger2_joint4",
    "r_middle_finger_mcp_flex": "right_finger3_joint1",
    "r_middle_finger_mcp_abd": "right_finger3_joint2",
    "r_middle_finger_pip": "right_finger3_joint3",
    "r_middle_finger_dip": "right_finger3_joint4",
    "r_ring_finger_mcp_flex": "right_finger4_joint1",
    "r_ring_finger_mcp_abd": "right_finger4_joint2",
    "r_ring_finger_pip": "right_finger4_joint3",
    "r_ring_finger_dip": "right_finger4_joint4",
    "r_pinky_mcp_flex": "right_finger5_joint1",
    "r_pinky_mcp_abd": "right_finger5_joint2",
    "r_pinky_pip": "right_finger5_joint3",
    "r_pinky_dip": "right_finger5_joint4",
}
TASK_JOINTS = tuple(f"right_finger{f}_joint{j}" for f in range(1, 6) for j in range(1, 5))
TASK_LINKS = tuple(f"right_finger{f}_link{k}" for f in range(1, 6) for k in range(1, 5))


def order(names, table, expected):
    mapped = [table.get(name) for name in names]
    if len(mapped) != len(expected) or set(mapped) != set(expected):
        raise ValueError(f"Unsupported right-hand joint/link names: {names}; use the Wuji Hand 2 model")
    return [mapped.index(name) for name in expected]


def quat_mul(a, b):
    aw, ax, ay, az = np.moveaxis(a, -1, 0)
    bw, bx, by, bz = np.moveaxis(b, -1, 0)
    return np.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        axis=-1,
    )


def quat_from_matrix(m):
    w = np.sqrt(max(0.0, 1.0 + m[0, 0] + m[1, 1] + m[2, 2])) / 2.0
    if w < 1e-8:
        # Antisymmetric terms cannot determine component signs at a half turn.
        quat = np.empty(4)
        mujoco.mju_mat2Quat(quat, m.ravel())
        return quat
    x = np.sqrt(max(0.0, 1.0 + m[0, 0] - m[1, 1] - m[2, 2])) / 2.0
    y = np.sqrt(max(0.0, 1.0 - m[0, 0] + m[1, 1] - m[2, 2])) / 2.0
    z = np.sqrt(max(0.0, 1.0 - m[0, 0] - m[1, 1] + m[2, 2])) / 2.0
    return np.array(
        [w, np.copysign(x, m[2, 1] - m[1, 2]), np.copysign(y, m[0, 2] - m[2, 0]), np.copysign(z, m[1, 0] - m[0, 1])]
    )


def quat_rotate(q, v):
    qv = np.concatenate([np.zeros(v.shape[:-1] + (1,)), v], axis=-1)
    conj = q * np.array([1.0, -1.0, -1.0, -1.0])
    return quat_mul(quat_mul(q, qv), conj)[..., 1:]


def palm_in_mount(hand_xml):
    model = mujoco.MjModel.from_xml_path(str(hand_xml))
    palm = model.body("r_wrist")
    if model.body(int(palm.parentid[0])).name != "r_mount":
        raise ValueError("Unsupported hand MJCF: r_wrist must be a direct child of r_mount")
    return np.asarray(palm.pos, np.float64), np.asarray(palm.quat, np.float64)


def pen_rotation(name, asset_json):
    metadata = json.loads(asset_json.read_text())
    if "source_to_capture_rotation" in metadata:
        rotation = np.asarray(metadata["source_to_capture_rotation"], np.float64)
    elif name == "pen":
        rotation = np.eye(3)
    else:
        raise ValueError("Unsupported pen coordinates: pass the original capture --asset-json with its rotation")
    if (
        rotation.shape != (3, 3)
        or not np.isfinite(rotation).all()
        or not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-8)
        or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-8)
    ):
        raise ValueError("source_to_capture_rotation must be a finite proper 3x3 rotation")
    return quat_from_matrix(rotation)


def component(value):
    if not value or value in {".", ".."} or any(c in value for c in "/\\\0"):
        raise ValueError(f"Invalid clip/group ID: {value!r}; use a single filename component")
    return value


def convert_result(result, *, source_clip, asset_json, hand_xml=HAND_XML):
    diagnostics = result.with_name(f"{result.stem}_diagnostics.npz")
    with np.load(result, allow_pickle=False) as data:
        fps = float(data["fps"].item())
        frame_ids = np.asarray(data["frame_ids"])
        clip_id = component(str(data["source_sequence_key"].item()))
    with np.load(diagnostics, allow_pickle=False) as diag:
        if str(diag["hand_side"].item()) != "right":
            raise ValueError("Only the right hand is supported by mjlab pen_spin; solve the right hand instead")
        objects = diag["object_names"].tolist()
        if len(objects) != 1:
            raise ValueError(f"Only a single object is supported by mjlab pen_spin; got {len(objects)} objects")
        if objects[0] not in {"pen", "pen250"}:
            raise ValueError(f"Unsupported object {objects[0]!r}: mjlab pen_spin supports only pen or pen250")
        if fps != 50.0 or float(diag["source_fps"].item()) != 50.0:
            raise ValueError("Only 50 Hz results are supported; import the capture at 50 Hz and solve again")
        joints = diag["joint_names"].tolist()
        links = diag["link_names"].tolist()
        qpos = np.asarray(diag["reference_qpos"], np.float64)
        root = np.asarray(diag["wrist_pose_w"], np.float64)
        link_pos = np.asarray(diag["link_pos_w"], np.float64)
        obj = np.asarray(diag["object_pose_w"], np.float64)
        landmarks = np.asarray(diag["mediapipe_landmarks_w"], np.float64)
        start = int(diag["source_start_frame"].item())
        end = int(diag["source_end_frame"].item())
    frames = len(qpos)
    if obj.shape != (frames, 1, 7):
        raise ValueError("Only a single object track with shape (T, 1, 7) is supported")
    if frames < 2 or not np.array_equal(frame_ids, np.arange(start, end)) or len(frame_ids) != frames:
        raise ValueError("Result and diagnostics must describe the same consecutive clip frames (at least two)")
    if not links or links[0] != "r_mount":
        raise ValueError("Unsupported wrist root: expected the right hand's r_mount, not a palm-rooted result")
    for name, array, shape in (
        ("reference_qpos", qpos, (frames, 20)),
        ("wrist_pose_w", root, (frames, 7)),
        ("link_pos_w", link_pos, (frames, 21, 3)),
        ("mediapipe_landmarks_w", landmarks, (frames, 21, 3)),
        ("object_pose_w", obj, (frames, 1, 7)),
    ):
        if array.shape != shape or not np.isfinite(array).all():
            raise ValueError(f"{name} must be finite with shape {shape}")
    joint_order = order(joints, JOINT, TASK_JOINTS)
    link_order = order(links[1:], BODY, TASK_LINKS)
    q_s = pen_rotation(objects[0], asset_json)
    offset_pos, offset_quat = palm_in_mount(hand_xml)
    with np.load(source_clip, allow_pickle=False) as source:
        times = np.asarray(source["t_s"], np.float64)
        original = np.asarray(source["mediapipe_landmarks_w"])
    if (
        times.ndim != 1
        or len(times) < end
        or not np.isfinite(times).all()
        or not np.allclose(np.diff(times), 0.02, rtol=0, atol=1e-8)
        or original.shape != (len(times), 21, 3)
    ):
        raise ValueError("--source-clip must be the matching original 50 Hz capture with t_s and (T, 21, 3) landmarks")
    original = original[frame_ids]
    delta = original.astype(np.float64) - landmarks
    shift = delta[0, 0]
    if not np.isfinite(delta).all() or np.abs(delta - shift).max() >= 1e-6:
        raise ValueError(
            "--source-clip does not match the result: landmarks must differ only by a constant translation"
        )
    landmarks = original
    # r_mount and the palm are not the same frame, even when all array shapes match.
    palm_quat = quat_mul(root[:, 3:], offset_quat)
    palm_pos = root[:, :3] + quat_rotate(root[:, 3:], offset_pos) + shift
    fingers = link_pos[:, 1:][:, link_order] + shift
    arrays = {
        "mediapipe_landmarks_w": landmarks,
        "object_pose_w": np.concatenate([obj[:, 0, :3] + shift, quat_mul(obj[:, 0, 3:], q_s)], axis=1),
        "reference_qpos": qpos[:, joint_order],
        "wrist_pose_w": np.concatenate([palm_pos, palm_quat], axis=1),
        "link_pos_w": np.concatenate([palm_pos[:, None], fingers], axis=1),
    }
    return clip_id, {key: value.astype(np.float32) for key, value in arrays.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", nargs="+", type=Path, help="Result NPZs, each beside its _diagnostics.npz sidecar")
    parser.add_argument("--out", required=True, type=Path, help="New clip group directory; existing paths are refused")
    parser.add_argument("--group-name", help="Display name in index.json (default: output directory name)")
    parser.add_argument(
        "--hand-xml", type=Path, default=HAND_XML, help="Wuji Hand 2 right.xml (default: repository submodule)"
    )
    parser.add_argument("--asset-json", type=Path, required=True, help="Original raw package's pen asset.json")
    parser.add_argument(
        "--source-clip",
        type=Path,
        action="append",
        required=True,
        help="Original raw package's 50 Hz capture NPZ, once per result in order",
    )
    args = parser.parse_args()
    try:
        group_id = component(args.out.name)
        if args.out.exists():
            raise ValueError(f"Output already exists: {args.out}; choose a new group directory")
        if len(args.source_clip) != len(args.results):
            raise ValueError("Pass one --source-clip per result, in the same order")
        clips = [
            convert_result(path, source_clip=source, asset_json=args.asset_json, hand_xml=args.hand_xml)
            for path, source in zip(args.results, args.source_clip, strict=True)
        ]
        if len({clip_id for clip_id, _ in clips}) != len(clips):
            raise ValueError("Duplicate clip IDs in the results; write separate groups")
        args.out.mkdir(parents=True)
        entries = []
        for clip_id, arrays in clips:
            path = args.out / f"{clip_id}.npz"
            np.savez(path, **arrays)
            entries.append(
                {"clip_id": clip_id, "file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            )
        index = {
            "schema": "wuji_pen_spin_clips:1",
            "group_id": group_id,
            "group_name": args.group_name or group_id,
            "frame_dt": 0.02,
            "clip_count": len(entries),
            "clips": entries,
        }
        (args.out / "index.json").write_text(json.dumps(index, indent=2) + "\n")
        print(f"{group_id}: {len(entries)} clips, 50 Hz -> {args.out}")
    except (OSError, ValueError, KeyError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
