"""Object surface sampling: uniform, curvature-weighted and fingertip-proximity-weighted."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from toporetarget.solver.clip import ObjectPoseTrack, RetargetInputTrack, RetargetObjectTrack
from toporetarget.solver.landmarks import FINGERTIP_LANDMARKS
from toporetarget.solver.settings import OBJECT_SAMPLER_MODES, ObjectConfig


@dataclass(frozen=True)
class ObjectSamplerSpec:
    """Sampler mode and hyperparameters for object surface point selection."""

    mode: str = "proxcurv"
    sample_count: int = 50
    seed: int = 0
    sigma: float = 0.010
    alpha: float = 30.0
    tip_indices: tuple[int, ...] = FINGERTIP_LANDMARKS


@dataclass(frozen=True)
class ObjectSamplingInput:
    """Per-sequence sampler inputs: the object track and the retarget-frame landmarks."""

    object_track: RetargetObjectTrack
    landmarks: np.ndarray


def resolve_sequence_object_points(track: RetargetInputTrack, object_cfg: ObjectConfig) -> list[np.ndarray]:
    """Resolve one sequence's object-local point cloud per object from the configured sampler."""
    sampler = object_cfg.sampler
    spec = ObjectSamplerSpec(
        mode=sampler.mode,
        sample_count=object_cfg.sample_count,
        seed=sampler.seed,
        sigma=sampler.sigma,
        alpha=sampler.alpha,
    )
    return [sample_object_points(ObjectSamplingInput(obj, track.landmarks), spec) for obj in track.objects]


def sample_object_points(inputs: ObjectSamplingInput, spec: ObjectSamplerSpec) -> np.ndarray | None:
    """Sample object-local surface points according to ``spec``."""
    object_track = inputs.object_track
    if spec.mode not in OBJECT_SAMPLER_MODES:
        raise ValueError(f"unsupported object sampler mode: {spec.mode!r}")
    mesh_path = object_track.mesh_path
    if mesh_path is None:
        return None

    mesh = _load_mesh(mesh_path)
    if spec.mode == "base":
        return sample_base(mesh, spec.sample_count, spec.seed)
    if spec.mode == "curv":
        return sample_curvature_weighted(mesh, spec.sample_count, spec.seed)

    pose = object_track.pose
    if pose is None:
        return sample_curvature_weighted(mesh, spec.sample_count, spec.seed)
    tips_obj_local = finger_tips_object_local(inputs.landmarks, pose, tip_indices=spec.tip_indices)
    return sample_proxcurv(
        mesh,
        spec.sample_count,
        spec.seed,
        tips_obj_local,
        sigma=spec.sigma,
        alpha=spec.alpha,
    )


def sample_base(mesh, count: int, seed: int) -> np.ndarray:
    """Uniformly sample points on a mesh surface."""
    import trimesh

    mesh = _as_trimesh(mesh)
    points, _ = trimesh.sample.sample_surface(mesh, count, seed=seed)
    return np.asarray(points, dtype=np.float64)


def sample_uniform_object_points_from_mesh(mesh_path: str, *, sample_count: int, seed: int) -> np.ndarray:
    """Uniformly sample object-local points straight from a mesh path."""
    return sample_base(_load_mesh(str(mesh_path)), sample_count, seed)


def sample_curvature_weighted(mesh, count: int, seed: int) -> np.ndarray:
    """Sample surface points weighted by per-face curvature saliency."""
    mesh = _as_trimesh(mesh)
    return _sample_weighted_surface(mesh, count, seed, _face_curvature_saliency(mesh))


def sample_proxcurv(
    mesh,
    count: int,
    seed: int,
    tips_obj_local: np.ndarray,
    *,
    sigma: float,
    alpha: float,
) -> np.ndarray:
    """Sample surface points with curvature plus fingertip-proximity weighting."""
    mesh = _as_trimesh(mesh)
    curvature = _face_curvature_saliency(mesh)
    proximity = _face_proximity_saliency(mesh, tips_obj_local, sigma=sigma)
    saliency = 1.0 + curvature + float(alpha) * proximity
    return _sample_weighted_surface(mesh, count, seed, saliency)


def finger_tips_object_local(
    landmarks: np.ndarray,
    object_pose: ObjectPoseTrack,
    *,
    tip_indices: tuple[int, ...],
) -> np.ndarray:
    """Transform the selected retarget-frame fingertips into object-local coordinates."""
    landmarks = np.asarray(landmarks, dtype=np.float64)
    positions = np.asarray(object_pose.position, dtype=np.float64)
    quaternions = np.asarray(object_pose.quaternion_xyzw, dtype=np.float64)
    tip_idx = np.asarray(tip_indices, dtype=np.int64)

    if landmarks.ndim == 2:
        pos, quat = _pose_value_at(positions, quaternions, 0)
        rot = _quat_xyzw_to_matrix(quat)
        return np.asarray((landmarks[tip_idx] - pos) @ rot, dtype=np.float64)

    if landmarks.ndim != 3:
        raise ValueError(f"landmarks must have shape (21, 3) or (T, 21, 3), got {landmarks.shape}")

    transformed: list[np.ndarray] = []
    for frame_idx in range(landmarks.shape[0]):
        pos, quat = _pose_value_at(positions, quaternions, frame_idx)
        rot = _quat_xyzw_to_matrix(quat)
        transformed.append((landmarks[frame_idx, tip_idx] - pos) @ rot)
    return np.asarray(np.vstack(transformed), dtype=np.float64)


def _face_curvature_saliency(mesh) -> np.ndarray:
    """Non-negative per-face saliency from the face-adjacency dihedral angles."""
    mesh = _as_trimesh(mesh)
    saliency = np.zeros(len(mesh.faces), dtype=np.float64)
    if len(mesh.faces) == 0:
        return saliency

    adjacency = np.asarray(getattr(mesh, "face_adjacency", np.empty((0, 2), dtype=np.int64)), dtype=np.int64)
    angles = np.asarray(getattr(mesh, "face_adjacency_angles", np.empty(0)), dtype=np.float64)
    if adjacency.size == 0 or angles.size == 0:
        return np.ones(len(mesh.faces), dtype=np.float64)

    angles = np.abs(angles)
    np.add.at(saliency, adjacency[:, 0], angles)
    np.add.at(saliency, adjacency[:, 1], angles)
    return _normalize_saliency(saliency)


def _face_proximity_saliency(mesh, tips_obj_local: np.ndarray, *, sigma: float) -> np.ndarray:
    """Per-face saliency from Gaussian proximity to the fingertip points."""
    mesh = _as_trimesh(mesh)
    tips = np.asarray(tips_obj_local, dtype=np.float64)
    if len(mesh.faces) == 0 or tips.size == 0:
        return np.zeros(len(mesh.faces), dtype=np.float64)
    if tips.ndim != 2 or tips.shape[1] != 3:
        raise ValueError(f"tips_obj_local must have shape (N, 3), got {tips.shape}")
    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma}")

    centers = np.asarray(mesh.triangles_center, dtype=np.float64)
    min_dist2 = np.full(len(centers), np.inf, dtype=np.float64)
    tip_chunk = 512
    for start in range(0, len(tips), tip_chunk):
        diff = centers[:, None, :] - tips[None, start : start + tip_chunk, :]
        np.minimum(min_dist2, np.min(np.sum(diff * diff, axis=2), axis=1), out=min_dist2)
    saliency = np.exp(-0.5 * min_dist2 / (float(sigma) * float(sigma)))
    return _normalize_saliency(saliency)


def _draw_barycentric(count: int, rng: np.random.Generator) -> np.ndarray:
    """Uniformly distributed barycentric weights for triangles."""
    if count < 0:
        raise ValueError(f"count must be non-negative, got {count}")
    uv = rng.random((count, 2))
    sqrt_u = np.sqrt(uv[:, 0])
    return np.column_stack((1.0 - sqrt_u, sqrt_u * (1.0 - uv[:, 1]), sqrt_u * uv[:, 1]))


def _sample_weighted_surface(mesh, count: int, seed: int, face_saliency: np.ndarray) -> np.ndarray:
    """Sample surface points with faces drawn proportional to area times saliency."""
    mesh = _as_trimesh(mesh)
    if count < 0:
        raise ValueError(f"count must be non-negative, got {count}")
    if count == 0:
        return np.empty((0, 3), dtype=np.float64)
    if len(mesh.faces) == 0:
        raise ValueError("mesh must contain at least one face")

    rng = np.random.default_rng(seed)
    weights = np.asarray(mesh.area_faces, dtype=np.float64) * np.asarray(face_saliency, dtype=np.float64)
    if weights.shape != (len(mesh.faces),):
        raise ValueError(f"face saliency shape {weights.shape} does not match {len(mesh.faces)} faces")
    if not np.all(np.isfinite(weights)) or float(np.sum(weights)) <= 0.0:
        weights = np.asarray(mesh.area_faces, dtype=np.float64)
    weights = weights / float(np.sum(weights))

    face_indices = rng.choice(len(mesh.faces), size=count, replace=True, p=weights)
    bary = _draw_barycentric(count, rng)
    triangles = np.asarray(mesh.triangles, dtype=np.float64)[face_indices]
    return np.einsum("nf,nfd->nd", bary, triangles).astype(np.float64, copy=False)


def _normalize_saliency(saliency: np.ndarray) -> np.ndarray:
    """Clamp non-finite/negative values to zero and rescale the max to one."""
    saliency = np.asarray(saliency, dtype=np.float64)
    saliency = np.where(np.isfinite(saliency) & (saliency > 0.0), saliency, 0.0)
    total = float(np.max(saliency)) if saliency.size else 0.0
    if total <= 0.0:
        return saliency
    return saliency / total


def _load_mesh(mesh_path: str):
    """Load a mesh file as a single concatenated ``trimesh.Trimesh``."""
    import trimesh

    return _as_trimesh(trimesh.load(str(mesh_path), force="mesh"))


def _as_trimesh(mesh):
    """Coerce a mesh or scene into a single ``trimesh.Trimesh``."""
    import trimesh

    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"expected trimesh.Trimesh, got {type(mesh).__name__}")
    return mesh


def _pose_value_at(positions: np.ndarray, quaternions: np.ndarray, frame_idx: int) -> tuple[np.ndarray, np.ndarray]:
    """The ``(position, quaternion)`` at ``frame_idx``, clamped to the last frame."""
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError(f"object pose position must have shape (T, 3), got {positions.shape}")
    if quaternions.ndim != 2 or quaternions.shape[1] != 4:
        raise ValueError(f"object pose quaternion must have shape (T, 4), got {quaternions.shape}")
    if len(positions) == 0 or len(quaternions) == 0:
        raise ValueError("object pose tracks must contain at least one frame")
    pose_idx = min(frame_idx, len(positions) - 1)
    return positions[pose_idx], quaternions[pose_idx]


def _quat_xyzw_to_matrix(q: np.ndarray) -> np.ndarray:
    """An xyzw quaternion as a 3x3 rotation matrix, after normalization."""
    q = np.asarray(q, dtype=np.float64)
    norm = float(np.linalg.norm(q))
    if norm <= 0.0:
        raise ValueError("object quaternion has zero norm")
    x, y, z, w = q / norm
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )
