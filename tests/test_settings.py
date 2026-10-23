"""HandRetargetConfig: the hand_solver block and --set overrides are validated, not silently accepted."""

from __future__ import annotations

import pytest

from toporetarget.solver import HandRetargetConfig, build_hand_config
from toporetarget.solver.settings import set_config_value


def test_defaults_file_matches_the_dataclass_defaults(right_config):
    cfg = build_hand_config(right_config)
    assert cfg == HandRetargetConfig()


def test_override_changes_one_field(right_config):
    cfg = build_hand_config(right_config, ["mesh.laplacian_weight=200", "penetration.enabled_warmup=false"])
    assert cfg.mesh.laplacian_weight == 200
    assert cfg.penetration.enabled_warmup is False


def test_unknown_key_is_refused(right_config):
    with pytest.raises(TypeError, match="unknown config override"):
        build_hand_config(right_config, ["mesh.laplacian_weigth=200"])


def test_type_mismatch_is_refused():
    cfg = HandRetargetConfig()
    with pytest.raises(TypeError, match="boolean"):
        set_config_value(cfg, "penetration.enabled_warmup", 1)


def test_interaction_radius_must_cover_one_sqp_step(right_config):
    with pytest.raises(ValueError, match="interaction_radius_m"):
        build_hand_config(right_config, ["mesh.interaction_radius_m=0.02"])


def test_key_value_syntax_is_checked(right_config):
    with pytest.raises(KeyError, match="KEY=VAL"):
        build_hand_config(right_config, ["mesh.laplacian_weight"])
