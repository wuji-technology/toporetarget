"""End to end on a few frames: the Python API writes a result the drivers recognise as current."""

from __future__ import annotations

import json

import numpy as np

import toporetarget as tr
from toporetarget.config import RESULT_SCHEMA, motion_frame_range, read_npz, result_mismatch, solver_settings_digest
from toporetarget.dataset import CanonicalDataset
from toporetarget.viewer import load_clip


def _strings(value):
    """Every string in a nested value, and in the JSON a string holds."""
    if isinstance(value, str):
        yield value
        try:
            yield from _strings(json.loads(value))
        except ValueError:
            pass
    elif isinstance(value, dict):
        for item in value.values():
            yield from _strings(item)
    elif isinstance(value, list | tuple):
        for item in value:
            yield from _strings(item)


def _absolute_paths(arrays: dict) -> list[str]:
    return [
        f"{key}: {text}"
        for key, array in arrays.items()
        if np.asarray(array).dtype.kind == "U"
        for text in _strings(np.asarray(array).tolist())
        if text.startswith("/")
    ]


def test_retarget_writes_result_and_diagnostics(tmp_path, pen_dataset, right_config):
    out = tmp_path / "pen.npz"
    overrides = ["solver.n_iter=2", "solver.n_iter_first=2"]
    (result,) = tr.retarget(pen_dataset, "pen_spin_029", max_frames=6, out=out, overrides=overrides)
    assert result == out
    with np.load(result, allow_pickle=False) as data:
        assert str(data["result_schema"]) == RESULT_SCHEMA
        assert data["qpos"].shape == (6, 7 + 20)
        assert str(data["robot_wrist_body"]) == "wuji_wrist"
        assert str(data["source_dataset"]) == "pen_spin"
        assert list(data["frame_ids"]) == list(range(6))
    with np.load(tr.diagnostics_path(result), allow_pickle=False) as diag:
        assert diag["link_pos_w"].shape == (6, 21, 3)
        assert diag["link_object_contact"].shape == (6, 21)
        assert diag["reference_qpos"].shape == (6, 20)
        assert np.allclose(diag["link_pos_w"][:, 0], diag["wrist_pose_w"][:, :3])
        spec = json.loads(str(diag["hand_spec_json"]))
        assert spec["name"] == "wujihand2_right"

    dataset = CanonicalDataset.at(pen_dataset)
    frame_range = motion_frame_range({"max_frames": 6}, dataset, "pen_spin_029")
    digest = solver_settings_digest(right_config, overrides)
    assert result_mismatch(result, dataset, "pen_spin_029", frame_range, digest) is None
    other = solver_settings_digest(right_config, [*overrides, "solver.n_iter=3"])
    assert result_mismatch(result, dataset, "pen_spin_029", frame_range, other) == "solved with other settings"

    clip = load_clip([result])
    assert clip.hands[0].side == "right" and clip.n_frames == 6

    assert _absolute_paths(read_npz(result)) == []
    assert _absolute_paths(read_npz(tr.diagnostics_path(result))) == []


def test_example_dataset_records_no_absolute_path(pen_dataset):
    files = [pen_dataset / "manifest.json", *sorted(pen_dataset.glob("motions/*.meta.json"))]
    found = [f"{path.name}: {text}" for path in files for text in _strings(json.loads(path.read_text()))]
    assert [entry for entry in found if entry.split(": ", 1)[1].startswith("/")] == []
