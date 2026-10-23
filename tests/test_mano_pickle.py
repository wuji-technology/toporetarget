"""The MANO reader permits allowlisted globals and chumpy state holders, and rejects other globals."""

from __future__ import annotations

import io
import os
import pickle

import numpy as np
import pytest

from toporetarget.datasets.mano_model import _ManoUnpickler


def _load(payload) -> object:
    return _ManoUnpickler(io.BytesIO(pickle.dumps(payload, protocol=2)), encoding="latin1").load()


class _RunsCode:
    def __reduce__(self):
        return (os.getcwd, ())


def test_arrays_load():
    data = _load({"v_template": np.zeros((3, 3)), "kintree_table": np.arange(4).reshape(2, 2)})
    assert data["v_template"].shape == (3, 3)


def test_other_classes_are_refused():
    with pytest.raises(pickle.UnpicklingError):
        _load({"v_template": _RunsCode()})
