"""CanonicalDataset: clip discovery, metadata and the motions/<clip>.npz shorthand."""

from __future__ import annotations

import pytest

from toporetarget.dataset import CanonicalDataset


def test_clips_hands_and_objects(pen_dataset):
    dataset = CanonicalDataset.at(pen_dataset)
    assert dataset.name == "pen_spin"
    assert dataset.clips() == ["pen_spin_029"]
    assert dataset.hands("pen_spin_029") == ["right"]
    assert dataset.objects("pen_spin_029") == ["pen250"]
    assert dataset.frames("pen_spin_029") == 231
    inputs = dataset.clip_inputs("pen_spin_029")
    assert dataset.clip_npz("pen_spin_029") in inputs
    assert any(path.name == "mesh.stl" for path in inputs)


def test_locate_accepts_a_clip_file(pen_dataset):
    dataset, clip = CanonicalDataset.locate(pen_dataset / "motions" / "pen_spin_029.npz")
    assert dataset.root == pen_dataset.resolve() and clip == "pen_spin_029"
    with pytest.raises(ValueError, match="names clip"):
        CanonicalDataset.locate(pen_dataset / "motions" / "pen_spin_029.npz", clip="other")


def test_resolve_clip_defaults_to_the_first(pen_dataset):
    dataset = CanonicalDataset.at(pen_dataset)
    assert dataset.resolve_clip(None) == dataset.clips()[0]
    with pytest.raises(FileNotFoundError):
        dataset.resolve_clip("no_such_clip")


def test_not_a_dataset(tmp_path):
    with pytest.raises(FileNotFoundError, match="not a canonical dataset"):
        CanonicalDataset.at(tmp_path)
