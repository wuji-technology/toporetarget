"""FrameSolver: the per-frame session on the pen example, driven with world-frame input."""

from __future__ import annotations

import numpy as np

import toporetarget as tr
from toporetarget.solver.hand import WRIST_DOF

XYZW = [4, 5, 6, 3]


def test_steps_on_the_pen_clip(right_config, right_hand, pen_dataset):
    paths = tr.resolve_clip_paths("pen_spin_029", pen_dataset)
    clip = tr.load_canonical_clip(paths, right_hand, max_frames=4)
    cfg = tr.build_hand_config(right_config)
    solver = tr.FrameSolver.from_paths(cfg, right_hand, clip.object_meshes)
    assert solver.nq == right_hand.nq

    limits = solver.retargeter.runtime.hand
    for frame in range(clip.n_frames):
        wrist = clip.wrist_pose_world[frame]
        objects = clip.object_pose_world[frame]
        state = solver.step(
            tr.HandObjectState(clip.landmarks_world[frame], wrist[:3], wrist[XYZW], objects[:, :3], objects[:, XYZW])
        )
        assert state.qpos.shape == (right_hand.nq,) and np.isfinite(state.qpos).all()
        assert state.finger_qpos.shape == (right_hand.finger_dof,)
        fingers = state.qpos[WRIST_DOF:]
        assert np.all(fingers >= limits.q_lb[WRIST_DOF:] - 1e-9) and np.all(fingers <= limits.q_ub[WRIST_DOF:] + 1e-9)
        assert np.all(np.abs(state.qpos[:3]) <= cfg.wrist.translation_limit + 1e-9)
        assert np.all(np.abs(state.qpos[3:WRIST_DOF]) <= cfg.wrist.rotation_limit + 1e-9)
        assert np.isclose(np.linalg.norm(state.wrist_q_world), 1.0)
    assert state.metrics["final_pen_max_mm"] <= cfg.penetration.tolerance * 1e3 + 0.5
