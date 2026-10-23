"""Shared fixtures: the packaged robot configs and the example clip that ships in data/pen_spin."""

from __future__ import annotations

from pathlib import Path

import pytest

import toporetarget as tr

REPO = Path(__file__).resolve().parents[1]
EXAMPLES = REPO / "data"


@pytest.fixture(scope="session")
def right_config() -> dict:
    return tr.runtime_config(tr.load_layered_config(tr.example_robot_config("right"), tr.DEFAULT_CONFIG))


@pytest.fixture(scope="session")
def right_hand(right_config) -> tr.HandSpec:
    return tr.HandSpec.from_config(right_config)


@pytest.fixture(scope="session")
def pen_dataset() -> Path:
    return EXAMPLES / "pen_spin"
