"""The interaction mesh: Delaunay topology, edge weights and the Laplacian over hand keypoints and object samples."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import Delaunay

from toporetarget.solver.landmarks import FINGER_ROOT_LANDMARKS


def create_interaction_mesh(vertices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Build a tetrahedral mesh over keypoints with Delaunay triangulation."""
    tri = Delaunay(vertices)
    return vertices, tri.simplices


def get_adjacency_list(tetrahedra: np.ndarray, num_vertices: int) -> list[list[int]]:
    """Adjacency list from tetrahedra: two vertices are neighbours if they share one."""
    adj: list[set[int]] = [set() for _ in range(num_vertices)]
    for tet in tetrahedra:
        for i in range(4):
            for j in range(i + 1, 4):
                u, v = tet[i], tet[j]
                adj[u].add(v)
                adj[v].add(u)
    return [list(s) for s in adj]


def _flatten_adjacency(adj_list: list[list[int]]) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Flatten adjacency to COO ``(i_flat, j_flat)`` plus per-row sizes."""
    row_sizes = [len(row) for row in adj_list]
    total = sum(row_sizes)
    i_flat = np.empty(total, dtype=np.intp)
    j_flat = np.empty(total, dtype=np.intp)
    cursor = 0
    for i, row in enumerate(adj_list):
        k = len(row)
        if k:
            j_flat[cursor : cursor + k] = row
            i_flat[cursor : cursor + k] = i
            cursor += k
    return i_flat, j_flat, row_sizes


def filter_adjacency_by_distance(
    adj_list: list[list[int]],
    vertices: np.ndarray,
    threshold: float,
) -> list[list[int]]:
    """Drop edges longer than ``threshold`` (metres) from an adjacency list."""
    n = len(adj_list)
    i_flat, j_flat, row_sizes = _flatten_adjacency(adj_list)
    if i_flat.size == 0:
        return [[] for _ in range(n)]

    dists = np.linalg.norm(vertices[j_flat] - vertices[i_flat], axis=1)
    keep = dists < threshold

    filtered: list[list[int]] = []
    start = 0
    for k in row_sizes:
        end = start + k
        filtered.append([] if k == 0 else j_flat[start:end][keep[start:end]].tolist())
        start = end
    return filtered


def calculate_laplacian_matrix(
    vertices: np.ndarray,
    adj_list: list[list[int]],
    epsilon: float = 1e-6,
    distance_decay_k: float | None = None,
    *,
    distance_weight_mode: str | None = None,
    distance_weight_cutoff_radius: float | None = None,
) -> np.ndarray:
    """The ``(N, N)`` Laplacian matrix with ``L[i,i] = 1`` and ``L[i,j] = -w_ij / sum_j w_ij``."""
    n = len(vertices)
    L = np.zeros((n, n))
    i_flat, j_flat, _ = _flatten_adjacency(adj_list)
    if i_flat.size == 0:
        return L

    mode = distance_weight_mode
    if mode is None:
        mode = "exp" if distance_decay_k is not None else "uniform"

    if mode == "uniform":
        deg = np.zeros(n, dtype=np.int64)
        np.add.at(deg, i_flat, 1)
        w = 1.0 / deg[i_flat].astype(np.float64)
        L[i_flat, j_flat] = -w
        rows_with_neighbors = np.unique(i_flat)
        L[rows_with_neighbors, rows_with_neighbors] = 1.0
        return L

    if mode == "exp":
        raw_w = _exp_decay_weights(vertices, i_flat, j_flat, distance_decay_k)
        _assign_normalized_weights(L, i_flat, j_flat, raw_w, n, epsilon)
        return L

    if mode != "compact_only_exp":
        raise ValueError(f"Unknown Laplacian distance weight mode: {mode!r}")
    if distance_weight_cutoff_radius is None or distance_weight_cutoff_radius <= 0:
        raise ValueError("distance_weight_cutoff_radius must be > 0 for compact_only_exp Laplacian weighting")

    dists = np.linalg.norm(vertices[j_flat] - vertices[i_flat], axis=1)
    raw_w = np.exp(-distance_decay_k * dists) if distance_decay_k is not None else np.ones_like(dists)
    x = dists / float(distance_weight_cutoff_radius)
    raw_w *= np.where(x < 1.0, (1.0 - x * x) ** 2, 0.0)
    _assign_normalized_weights_skip_inactive(L, i_flat, j_flat, raw_w, n, epsilon)
    return L


def _exp_decay_weights(
    vertices: np.ndarray,
    i_flat: np.ndarray,
    j_flat: np.ndarray,
    distance_decay_k: float | None,
) -> np.ndarray:
    """Per-edge ``exp(-k ||e_ij||)`` weights (ones when ``k`` is None)."""
    dists = np.linalg.norm(vertices[j_flat] - vertices[i_flat], axis=1)
    if distance_decay_k is None:
        return np.ones_like(dists)
    return np.exp(-distance_decay_k * dists)


def _assign_normalized_weights(
    L: np.ndarray,
    i_flat: np.ndarray,
    j_flat: np.ndarray,
    raw_w: np.ndarray,
    n: int,
    epsilon: float,
) -> None:
    """Write ``-w_ij / row_sum`` into ``L``; underflowing rows divide by 1.0."""
    row_w_sum = np.zeros(n)
    np.add.at(row_w_sum, i_flat, raw_w)
    denom = np.where(row_w_sum[i_flat] > epsilon, row_w_sum[i_flat], 1.0)
    w = raw_w / denom
    L[i_flat, j_flat] = -w
    rows_with_neighbors = np.unique(i_flat)
    L[rows_with_neighbors, rows_with_neighbors] = 1.0


def _assign_normalized_weights_skip_inactive(
    L: np.ndarray,
    i_flat: np.ndarray,
    j_flat: np.ndarray,
    raw_w: np.ndarray,
    n: int,
    epsilon: float,
) -> None:
    """Write ``-w_ij / row_sum`` into ``L``, leaving fully inactive rows at zero."""
    row_w_sum = np.zeros(n)
    np.add.at(row_w_sum, i_flat, raw_w)
    active_edges = row_w_sum[i_flat] > epsilon
    if not np.any(active_edges):
        return
    w = raw_w[active_edges] / row_w_sum[i_flat[active_edges]]
    L[i_flat[active_edges], j_flat[active_edges]] = -w
    active_rows = np.flatnonzero(row_w_sum > epsilon)
    L[active_rows, active_rows] = 1.0


def get_skeleton_adjacency(n_keypoints: int = 21) -> list[list[int]]:
    """Hand-skeleton topology: wrist-to-each-MCP plus one chain per finger."""
    adj: list[list[int]] = [[] for _ in range(n_keypoints)]

    def _add(i: int, j: int) -> None:
        adj[i].append(j)
        adj[j].append(i)

    for root in FINGER_ROOT_LANDMARKS:
        _add(0, root)
    for start in FINGER_ROOT_LANDMARKS:
        for k in range(3):
            _add(start + k, start + k + 1)
    return adj


def _normalize_object_points(obj_points: np.ndarray | None) -> np.ndarray | None:
    if obj_points is None or len(obj_points) == 0:
        return None
    return obj_points


@dataclass(frozen=True)
class LaplacianKeypoints:
    """Hand and optional object points, in the fixed hand-first order."""

    hand_points: np.ndarray
    obj_points: np.ndarray | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "obj_points", _normalize_object_points(self.obj_points))

    @property
    def all_points(self) -> np.ndarray:
        if self.obj_points is None:
            return self.hand_points
        return np.vstack([self.hand_points, self.obj_points])

    @property
    def n_hand(self) -> int:
        return len(self.hand_points)

    @property
    def n_obj(self) -> int:
        return 0 if self.obj_points is None else len(self.obj_points)

    @property
    def n_total(self) -> int:
        return self.n_hand + self.n_obj


@dataclass(frozen=True)
class LaplacianJacobians:
    """Robot-side Jacobians for the Laplacian keypoints."""

    hand_jacobians: np.ndarray
    obj_jacobians: np.ndarray | None = None

    def all_jacobians(self, n_total: int, nq: int) -> np.ndarray:
        """Stacked Jacobians, with zero rows for the fixed object points."""
        full = np.zeros((3 * n_total, nq))
        full[: len(self.hand_jacobians)] = self.hand_jacobians
        if self.obj_jacobians is not None:
            start = len(self.hand_jacobians)
            full[start : start + len(self.obj_jacobians)] = self.obj_jacobians
        return full


@dataclass(frozen=True)
class Connections:
    """The interaction-mesh adjacency over ``[hand keypoints ; object points]``."""

    adjacency: list[list[int]]


def _with_padded_object_vertices(adjacency: list[list[int]], source_keypoints: LaplacianKeypoints) -> list[list[int]]:
    """``adjacency`` extended with empty rows for the object vertices."""
    n_extra = source_keypoints.n_total - source_keypoints.n_hand
    if n_extra <= 0:
        return adjacency
    return adjacency + [[] for _ in range(n_extra)]


def _add_undirected_edge(adjacency: list[list[int]], i: int, j: int) -> None:
    """Insert an undirected ``(i, j)`` edge in place (idempotent)."""
    if j not in adjacency[i]:
        adjacency[i].append(j)
    if i not in adjacency[j]:
        adjacency[j].append(i)


def _with_radius_edges(adjacency: list[list[int]], points: np.ndarray, radius: float) -> list[list[int]]:
    """A deep copy of ``adjacency`` augmented with every edge shorter than ``radius``."""
    result = [list(neighbors) for neighbors in adjacency]
    n_points = len(points)
    for i in range(n_points):
        for j in range(i + 1, n_points):
            if np.linalg.norm(points[i] - points[j]) < radius:
                _add_undirected_edge(result, i, j)
    return result


def build_delaunay_connections(
    source_keypoints: LaplacianKeypoints,
    *,
    edge_threshold: float | None,
) -> Connections:
    """Delaunay adjacency over all hand and object keypoints."""
    points = source_keypoints.all_points
    _, simplices = create_interaction_mesh(points)
    adjacency = get_adjacency_list(simplices, source_keypoints.n_total)
    if edge_threshold is not None:
        adjacency = filter_adjacency_by_distance(adjacency, points, edge_threshold)
    return Connections(adjacency)


def build_skeleton_connections(source_keypoints: LaplacianKeypoints) -> Connections:
    """MediaPipe hand-skeleton adjacency, leaving the object vertices isolated."""
    adjacency = get_skeleton_adjacency(source_keypoints.n_hand)
    adjacency = _with_padded_object_vertices(adjacency, source_keypoints)
    return Connections(adjacency)


def build_skeleton_plus_sphere_connections(source_keypoints: LaplacianKeypoints, *, radius: float) -> Connections:
    """Hand-skeleton adjacency plus every source-space edge below ``radius``."""
    adjacency = get_skeleton_adjacency(source_keypoints.n_hand)
    adjacency = _with_padded_object_vertices(adjacency, source_keypoints)
    adjacency = _with_radius_edges(adjacency, source_keypoints.all_points, radius)
    return Connections(adjacency)
