"""Golden run: frames 75-130 of the pen clip, where the fingers cross, against a stored solve."""

from __future__ import annotations

from pathlib import Path

import numpy as np

import toporetarget as tr
from toporetarget.config import read_npz

GOLDEN = Path(__file__).parent / "golden" / "pen_spin_029_75_130_qpos.npy"
TOLERANCE = 1e-4


def test_pen_clip_matches_the_golden_solve(tmp_path, pen_dataset):
    (result,) = tr.retarget(pen_dataset, "pen_spin_029", start=75, end=130, out=tmp_path / "pen.npz")
    qpos = read_npz(result)["qpos"]
    golden = np.load(GOLDEN, allow_pickle=False)
    assert qpos.shape == golden.shape
    worst = np.abs(qpos - golden).max()
    assert worst <= TOLERANCE, f"qpos moved by up to {worst:.2e} from the golden solve"

    phi_before = read_npz(tr.diagnostics_path(result))["selfcol_phi_before_m"]
    assert phi_before.min() >= 0.0
