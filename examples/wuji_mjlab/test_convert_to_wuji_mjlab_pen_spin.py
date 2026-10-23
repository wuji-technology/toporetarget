import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).with_name("convert_to_wuji_mjlab_pen_spin.py")
ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def converter():
    assert SCRIPT.is_file(), "The standalone pen_spin converter is missing"
    spec = importlib.util.spec_from_file_location("pen_spin_converter", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def result(tmp_path):
    robot = json.loads((ROOT / "toporetarget/robots/wujihand2_right.json").read_text())["robot"]
    path = tmp_path / "sample.npz"
    np.savez(path, fps=[50.0], frame_ids=[0, 1], source_sequence_key="sample")
    data = {
        "hand_side": np.asarray("right"),
        "object_names": np.asarray(["pen250"]),
        "joint_names": np.asarray([name for finger in robot["hand"]["fingers"] for name in finger["joints"]]),
        "link_names": np.asarray(
            ["r_mount"] + [name for finger in robot["hand"]["fingers"] for name in finger["links"]]
        ),
        "reference_qpos": np.tile(np.arange(20, dtype=float), (2, 1)),
        "wrist_pose_w": np.tile([0, 0, 0, 1, 0, 0, 0], (2, 1)),
        "object_pose_w": np.tile([0, 0, 1, 1, 0, 0, 0], (2, 1, 1)),
        "link_pos_w": np.zeros((2, 21, 3)),
        "mediapipe_landmarks_w": np.zeros((2, 21, 3), dtype=np.float32),
        "source_fps": np.asarray([50.0]),
        "source_start_frame": np.asarray([0]),
        "source_end_frame": np.asarray([2]),
    }
    np.savez(path.with_name("sample_diagnostics.npz"), **data)
    source = tmp_path / "capture.npz"
    np.savez(source, mediapipe_landmarks_w=np.zeros((3, 21, 3), np.float32), t_s=np.arange(3) / 50)
    asset = tmp_path / "asset.json"
    asset.write_text(json.dumps({"source_to_capture_rotation": [[0, 0, -1], [0, 1, 0], [1, 0, 0]]}))
    return path, data, source, asset


def test_palm_root_pen_axis_and_task_arrays(converter, result):
    path, _, source, asset = result
    clip_id, arrays = converter.convert_result(path, source_clip=source, asset_json=asset)
    assert clip_id == "sample"
    assert set(arrays) == {"reference_qpos", "wrist_pose_w", "object_pose_w", "link_pos_w", "mediapipe_landmarks_w"}
    np.testing.assert_allclose(arrays["wrist_pose_w"][0, :3], [0.00300045189842, 0, -0.0284982808512])
    np.testing.assert_allclose(arrays["object_pose_w"][0], [0, 0, 1, 2**-0.5, 0, -(2**-0.5), 0], atol=1e-7)
    np.testing.assert_array_equal(arrays["link_pos_w"][:, 0], arrays["wrist_pose_w"][:, :3])
    np.testing.assert_array_equal(arrays["reference_qpos"][0], np.arange(20))
    assert all(array.dtype == np.float32 for array in arrays.values())


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("hand_side", np.asarray("left"), "right hand"),
        ("object_names", np.asarray(["pen250", "pen"]), "single object"),
        ("object_names", np.asarray(["cube"]), "pen"),
        ("object_pose_w", np.zeros((2, 2, 7)), "single object"),
        ("source_fps", np.asarray([25.0]), "50 Hz"),
    ],
)
def test_rejects_unsupported_input_before_output(converter, result, tmp_path, field, value, message):
    path, data, source, asset = result
    data[field] = value
    np.savez(path.with_name("sample_diagnostics.npz"), **data)
    out = tmp_path / "clips"
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            str(path),
            "--out",
            str(out),
            "--source-clip",
            str(source),
            "--asset-json",
            str(asset),
        ],
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
    assert message in completed.stderr
    assert not out.exists()


def test_reorders_joint_and_link_names(converter, result):
    path, data, source, asset = result
    data["joint_names"] = data["joint_names"][::-1]
    data["reference_qpos"] = data["reference_qpos"][:, ::-1]
    data["link_names"][1:] = data["link_names"][1:][::-1]
    data["link_pos_w"][:, 1:, 0] = np.arange(20)[::-1]
    np.savez(path.with_name("sample_diagnostics.npz"), **data)
    _, arrays = converter.convert_result(path, source_clip=source, asset_json=asset)
    np.testing.assert_array_equal(arrays["reference_qpos"][0], np.arange(20))
    np.testing.assert_array_equal(arrays["link_pos_w"][0, 1:, 0], np.arange(20))


def test_restores_capture_origin_without_published_clip(converter, result, tmp_path):
    path, _, source, asset = result
    landmarks = np.broadcast_to(np.asarray([0.25, -0.125, 0], np.float32), (3, 21, 3))
    np.savez(source, mediapipe_landmarks_w=landmarks, t_s=np.arange(3) / 50)
    _, arrays = converter.convert_result(path, source_clip=source, asset_json=asset)
    np.testing.assert_array_equal(arrays["mediapipe_landmarks_w"], landmarks[:2])
    np.testing.assert_array_equal(arrays["object_pose_w"][0, :3], [0.25, -0.125, 1])


def test_writes_loadable_index_without_overwrite(converter, result, tmp_path):
    path, _, source, asset = result
    out = tmp_path / "example_group"
    command = [
        sys.executable,
        str(SCRIPT),
        str(path),
        "--out",
        str(out),
        "--source-clip",
        str(source),
        "--asset-json",
        str(asset),
    ]
    subprocess.run(command, check=True, capture_output=True, text=True)
    index = json.loads((out / "index.json").read_text())
    npz = out / "sample.npz"
    assert index == {
        "schema": "wuji_pen_spin_clips:1",
        "group_id": "example_group",
        "group_name": "example_group",
        "frame_dt": 0.02,
        "clip_count": 1,
        "clips": [{"clip_id": "sample", "file": "sample.npz", "sha256": hashlib.sha256(npz.read_bytes()).hexdigest()}],
    }
    before = npz.read_bytes()
    assert subprocess.run(command, capture_output=True).returncode != 0
    assert npz.read_bytes() == before


def test_mixed_axis_half_turn_keeps_rotation(converter):
    matrix = np.array([[0.0, -1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])
    quaternion = converter.quat_from_matrix(matrix)
    np.testing.assert_allclose(converter.quat_rotate(quaternion, np.eye(3)), matrix.T, atol=1e-14)


@pytest.mark.parametrize("missing", ["--source-clip", "--asset-json"])
def test_requires_original_capture_and_asset(converter, result, tmp_path, missing):
    path, _, source, asset = result
    out = tmp_path / "missing_capture"
    command = [sys.executable, str(SCRIPT), str(path), "--out", str(out)]
    for flag, value in (("--source-clip", source), ("--asset-json", asset)):
        if flag != missing:
            command.extend([flag, str(value)])
    completed = subprocess.run(command, capture_output=True, text=True)
    assert completed.returncode == 2
    assert missing in completed.stderr
    assert not out.exists()
