"""The MANO hand model in NumPy, and its 21 MediaPipe-ordered keypoints."""

from __future__ import annotations

import os
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

DEFAULT_MANO_DIR = Path(os.environ.get("TOPORETARGET_MANO_DIR") or Path.home() / ".cache" / "toporetarget" / "mano")

TIP_VERTICES = {"thumb": 745, "index": 317, "middle": 444, "ring": 556, "pinky": 673}

MEDIAPIPE_FROM_MANO = np.array(
    [0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20],
    dtype=np.intp,
)


class _ChumpyStub:
    """Stands in for a chumpy object while unpickling; keeps its state only."""

    def __init__(self, *args, **kwargs) -> None:
        pass

    def __setstate__(self, state) -> None:
        self.__dict__.update(state)


_MANO_GLOBALS = {
    ("numpy", "ndarray"),
    ("numpy", "dtype"),
    ("numpy.core.multiarray", "_reconstruct"),
    ("numpy._core.multiarray", "_reconstruct"),
    ("numpy.core.multiarray", "scalar"),
    ("numpy._core.multiarray", "scalar"),
    ("scipy.sparse", "csc_matrix"),
    ("scipy.sparse", "csr_matrix"),
    ("builtins", "set"),
    ("builtins", "frozenset"),
    ("__builtin__", "set"),
    ("__builtin__", "frozenset"),
    ("_codecs", "encode"),
}


class _ManoUnpickler(pickle.Unpickler):
    """Decode allowlisted NumPy, sparse, set and codec globals; replace chumpy classes with state holders."""

    def find_class(self, module: str, name: str):
        if module.startswith("chumpy"):
            return type(name, (_ChumpyStub,), {})
        if module.startswith("scipy.sparse."):
            module = "scipy.sparse"
        if (module, name) not in _MANO_GLOBALS:
            raise pickle.UnpicklingError(
                f"{module}.{name} is not part of a MANO model file; "
                "use the MANO_RIGHT.pkl / MANO_LEFT.pkl downloaded from mano.is.tue.mpg.de"
            )
        return super().find_class(module, name)


def _array(value) -> np.ndarray:
    """A plain array from a pickled MANO entry: ndarray, sparse matrix or chumpy node."""
    if hasattr(value, "toarray"):
        return value.toarray()
    if isinstance(value, _ChumpyStub):
        if "x" in value.__dict__:
            return np.asarray(value.x)
        return _array(value.a).ravel()[value.idxs].reshape(value.preferred_shape)
    return np.asarray(value)


@dataclass(frozen=True)
class ManoModel:
    """One hand's MANO model, as arrays."""

    side: str
    v_template: np.ndarray
    shapedirs: np.ndarray
    posedirs: np.ndarray
    j_regressor: np.ndarray
    weights: np.ndarray
    parents: np.ndarray
    hands_components: np.ndarray
    hands_mean: np.ndarray
    faces: np.ndarray

    @classmethod
    def load(cls, side: str, mano_dir: str | Path | None = None) -> ManoModel:
        path = Path(mano_dir or DEFAULT_MANO_DIR) / f"MANO_{side.upper()}.pkl"
        if not path.is_file():
            raise FileNotFoundError(
                f"{path} not found. Download the MANO models (mano.is.tue.mpg.de, 'Models & Code') and put "
                f"MANO_RIGHT.pkl and MANO_LEFT.pkl in {DEFAULT_MANO_DIR} (`toporetarget download mano` does), "
                "or pass --mano-dir."
            )
        with path.open("rb") as stream:
            data = _ManoUnpickler(stream, encoding="latin1").load()
        parents = np.asarray(data["kintree_table"], dtype=np.int64)[0].copy()
        parents[0] = -1
        return cls(
            side=side,
            v_template=_array(data["v_template"]).astype(np.float64),
            shapedirs=_array(data["shapedirs"]).astype(np.float64),
            posedirs=_array(data["posedirs"]).astype(np.float64),
            j_regressor=_array(data["J_regressor"]).astype(np.float64),
            weights=_array(data["weights"]).astype(np.float64),
            parents=parents,
            hands_components=_array(data["hands_components"]).astype(np.float64),
            hands_mean=_array(data["hands_mean"]).astype(np.float64),
            faces=np.asarray(data["f"], dtype=np.int64),
        )

    def full_pose(self, hand_pose: np.ndarray, *, num_pca_comps: int | None, flat_hand_mean: bool) -> np.ndarray:
        """``(T, 45)`` axis-angle finger pose from the dataset's parameterisation."""
        pose = np.asarray(hand_pose, dtype=np.float64)
        if num_pca_comps is not None:
            pose = pose @ self.hands_components[:num_pca_comps]
        if not flat_hand_mean:
            pose = pose + self.hands_mean
        return pose

    def forward(
        self,
        global_orient: np.ndarray,
        finger_pose: np.ndarray,
        *,
        betas: np.ndarray | None = None,
        transl: np.ndarray | None = None,
        v_template: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """``(joints (T, 16, 3), vertices (T, 778, 3))`` for axis-angle poses."""
        global_orient = np.asarray(global_orient, dtype=np.float64).reshape(-1, 3)
        n_frames = len(global_orient)
        pose = np.concatenate([global_orient, np.asarray(finger_pose, dtype=np.float64).reshape(n_frames, 45)], 1)

        template = self.v_template if v_template is None else np.asarray(v_template, dtype=np.float64)
        shaped = np.broadcast_to(template, (n_frames, 778, 3))
        if betas is not None:
            betas = np.broadcast_to(np.asarray(betas, dtype=np.float64), (n_frames, self.shapedirs.shape[-1]))
            shaped = shaped + np.einsum("vck,tk->tvc", self.shapedirs, betas)
        joints_rest = np.einsum("jv,tvc->tjc", self.j_regressor, shaped)

        rot = Rotation.from_rotvec(pose.reshape(-1, 3)).as_matrix().reshape(n_frames, 16, 3, 3)
        pose_feature = (rot[:, 1:] - np.eye(3)).reshape(n_frames, 135)
        posed = shaped + np.einsum("vck,tk->tvc", self.posedirs, pose_feature)

        world_rot = np.empty_like(rot)
        world_pos = np.empty_like(joints_rest)
        world_rot[:, 0], world_pos[:, 0] = rot[:, 0], joints_rest[:, 0]
        for j in range(1, 16):
            parent = self.parents[j]
            offset = joints_rest[:, j] - joints_rest[:, parent]
            world_rot[:, j] = world_rot[:, parent] @ rot[:, j]
            world_pos[:, j] = world_pos[:, parent] + np.einsum("tij,tj->ti", world_rot[:, parent], offset)

        skin_pos = world_pos - np.einsum("tjab,tjb->tja", world_rot, joints_rest)
        blend_rot = np.einsum("vj,tjab->tvab", self.weights, world_rot)
        blend_pos = np.einsum("vj,tja->tva", self.weights, skin_pos)
        vertices = np.einsum("tvab,tvb->tva", blend_rot, posed) + blend_pos

        joints = world_pos
        if transl is not None:
            shift = np.asarray(transl, dtype=np.float64).reshape(n_frames, 1, 3)
            joints, vertices = joints + shift, vertices + shift
        return joints, vertices


def mediapipe_keypoints(joints: np.ndarray, vertices: np.ndarray) -> np.ndarray:
    """``(T, 21, 3)`` MediaPipe-ordered keypoints from MANO joints and vertices."""
    tips = vertices[:, list(TIP_VERTICES.values())]
    return np.concatenate([joints, tips], axis=1)[:, MEDIAPIPE_FROM_MANO]
