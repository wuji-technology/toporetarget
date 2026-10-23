"""The captured hand: the 21 MediaPipe landmarks, their finger chains and the MANO wrist frame they define."""

from __future__ import annotations

import numpy as np

MEDIAPIPE_LANDMARK_COUNT = 21
MEDIAPIPE_FINGER_CHAINS: tuple[tuple[int, ...], ...] = (
    (0, 1, 2, 3, 4),
    (0, 5, 6, 7, 8),
    (0, 9, 10, 11, 12),
    (0, 13, 14, 15, 16),
    (0, 17, 18, 19, 20),
)
FINGER_ROOT_LANDMARKS = tuple(chain[1] for chain in MEDIAPIPE_FINGER_CHAINS)
FINGERTIP_LANDMARKS = tuple(chain[-1] for chain in MEDIAPIPE_FINGER_CHAINS)


def mediapipe_chain_edges() -> np.ndarray:
    """The 20 ``(parent, child)`` edges along the 5 MediaPipe finger chains."""
    edges = [(chain[k], chain[k + 1]) for chain in MEDIAPIPE_FINGER_CHAINS for k in range(4)]
    return np.array(edges, dtype=np.int64)


OPERATOR2MANO_RIGHT = np.array(
    [
        [0, 0, -1],
        [-1, 0, 0],
        [0, 1, 0],
    ]
)

OPERATOR2MANO_LEFT = np.array(
    [
        [0, 0, -1],
        [1, 0, 0],
        [0, -1, 0],
    ]
)

OPERATOR2MANO = {"right": OPERATOR2MANO_RIGHT, "left": OPERATOR2MANO_LEFT}


def estimate_frame_from_hand_points(keypoint_3d_array: np.ndarray) -> np.ndarray:
    """The ``(3, 3)`` wrist orientation frame (MANO convention) from ``(21, 3)`` keypoints."""
    assert keypoint_3d_array.shape == (21, 3)
    points = keypoint_3d_array[[0, 5, 9], :]

    x_vector = points[0] - points[2]

    points = points - np.mean(points, axis=0, keepdims=True)
    u, s, v = np.linalg.svd(points)

    normal = v[2, :]

    x = x_vector - np.sum(x_vector * normal) * normal
    x = x / np.linalg.norm(x)
    z = np.cross(x, normal)

    if np.sum(z * (points[1] - points[2])) < 0:
        normal *= -1
        z *= -1
    frame = np.stack([x, normal, z], axis=1)
    return frame
