"""The self-collision pass: a crossing of two fingers is measured as penetration and removed."""

from __future__ import annotations

import mujoco
import numpy as np

from toporetarget.solver.selfcol import optimize_trajectory


def test_removes_a_finger_crossing(right_hand):
    model = mujoco.MjModel.from_xml_path(str(right_hand.hand_xml))
    lo, hi = model.jnt_range[:, 0], model.jnt_range[:, 1]
    names = list(right_hand.joint_names)
    q = (lo + hi) / 2
    q[names.index("r_index_finger_mcp_abd")] = lo[names.index("r_index_finger_mcp_abd")]
    q[names.index("r_middle_finger_mcp_abd")] = hi[names.index("r_middle_finger_mcp_abd")]
    q_ref = np.repeat(q[None], 4, axis=0)

    q_clean, info = optimize_trajectory(q_ref, hand=right_hand)

    assert info["phi_before"].min() < -5e-3
    assert info["phi_after"].min() >= -1e-6
    assert q_clean.shape == q_ref.shape
    assert np.all(q_clean >= lo - 1e-9) and np.all(q_clean <= hi + 1e-9)
