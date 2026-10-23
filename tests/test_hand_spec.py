"""HandSpec: the packaged hands load, agree with their MJCF, and a wrong name is refused by name."""

from __future__ import annotations

import copy

import mujoco
import numpy as np
import pytest

import toporetarget as tr
from toporetarget.solver.hand import WRIST_DOF, HandSpec

PALM_ALIGN_RIGHT = np.array(
    [
        [-0.205760758910, +0.974549564585, -0.088969973925],
        [+0.976653679205, +0.210235149529, +0.044144906811],
        [+0.061726015483, -0.077809562845, -0.995055561736],
    ]
)


@pytest.mark.parametrize("side", ["right", "left"])
def test_packaged_hand_matches_its_mjcf(side):
    config = tr.runtime_config(tr.load_layered_config(tr.example_robot_config(side), tr.DEFAULT_CONFIG))
    hand = HandSpec.from_config(config)
    assert hand.side == side and hand.name == f"wujihand2_{side}"
    assert hand.finger_dof == 20 and hand.nq == WRIST_DOF + 20
    assert len(hand.link_names) == 21 and hand.link_names[0] == hand.root_body
    assert len(hand.tip_sites) == 5

    model = mujoco.MjModel.from_xml_path(str(hand.hand_xml))
    assert tuple(model.joint(i).name for i in range(model.njnt)) == hand.joint_names
    assert len(hand.keypoint_targets()) == 21
    assert np.allclose(hand.palm_align @ hand.palm_align.T, np.eye(3), atol=1e-12)
    if side == "right":
        assert np.abs(hand.palm_align - PALM_ALIGN_RIGHT).max() < 1e-9


def test_wrong_body_name_is_reported_by_name(right_config):
    config = copy.deepcopy(right_config)
    config["robot"]["hand"]["fingers"][1]["links"][2] = "r_index_finger_nowhere"
    with pytest.raises(ValueError, match="r_index_finger_nowhere"):
        HandSpec.from_config(config)


def test_joint_order_must_match_the_mjcf(right_config):
    config = copy.deepcopy(right_config)
    thumb = config["robot"]["hand"]["fingers"][0]["joints"]
    thumb[0], thumb[1] = thumb[1], thumb[0]
    with pytest.raises(ValueError, match="in its order"):
        HandSpec.from_config(config)


def test_missing_hand_block_points_at_the_reference(right_config):
    config = copy.deepcopy(right_config)
    del config["robot"]["hand"]
    with pytest.raises(ValueError, match="robot.hand"):
        HandSpec.from_config(config)


def test_spec_round_trips_through_json(right_hand):
    data = right_hand.to_dict()
    assert data["fingers"][0]["tip_site"] == "r_thumb_tip"
    assert np.allclose(np.asarray(data["palm_align"]), right_hand.palm_align)
