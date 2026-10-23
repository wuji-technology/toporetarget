"""``HandRetargetConfig``: the solver's settings as validated nested dataclasses, with dotted overrides."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any, Literal, get_args

LOG_PREFIX = "TopoRetarget"

BaseMode = Literal["fixed", "floating"]
TopologyMode = Literal["delaunay", "skeleton", "skeleton_plus_sphere"]
EdgeWeightMode = Literal["uniform", "exp", "compact_only_exp"]
McpSurrogate = Literal["link1", "link2"]
BoneDirTopology = Literal["chain", "graph"]
BoneDirEdgeWeightMode = Literal["uniform", "inv_count", "laplacian_aligned"]
ObjectSamplerMode = Literal["base", "curv", "proxcurv"]

BASE_MODES = get_args(BaseMode)
TOPOLOGY_MODES = get_args(TopologyMode)
LAPLACIAN_DISTANCE_WEIGHT_MODES = get_args(EdgeWeightMode)
MCP_SURROGATES = get_args(McpSurrogate)
BONE_DIR_TOPOLOGIES = get_args(BoneDirTopology)
BONE_DIR_EDGE_WEIGHT_MODES = get_args(BoneDirEdgeWeightMode)
OBJECT_SAMPLER_MODES = get_args(ObjectSamplerMode)


def _check_positive(name: str, v: float) -> None:
    if v <= 0:
        raise ValueError(f"{name} must be > 0, got {v}")


def _check_positive_int(name: str, v: int) -> None:
    if not isinstance(v, int) or isinstance(v, bool) or v <= 0:
        raise ValueError(f"{name} must be a positive integer, got {v!r}")


def _check_nonneg_int(name: str, v: int) -> None:
    if not isinstance(v, int) or isinstance(v, bool) or v < 0:
        raise ValueError(f"{name} must be a non-negative integer, got {v!r}")


def _check_nonneg(name: str, v: float) -> None:
    if v < 0:
        raise ValueError(f"{name} must be >= 0, got {v}")


def _check_optional_positive(name: str, v: float | None) -> None:
    if v is not None and v <= 0:
        raise ValueError(f"{name} must be > 0 when set, got {v}")


def _check_choice(name: str, v: str, allowed: tuple[str, ...]) -> None:
    if v not in allowed:
        raise ValueError(f"{name} must be one of {allowed}, got {v!r}")


@dataclass
class ModelConfig:
    """The base mode: ``floating`` solves the 6 virtual wrist DoF, ``fixed`` locks them at zero."""

    base_mode: str = "floating"

    def __post_init__(self) -> None:
        _check_choice("model.base_mode", self.base_mode, BASE_MODES)

    @property
    def floating_base(self) -> bool:
        return self.base_mode == "floating"


@dataclass
class SolverConfig:
    """SQP solver configuration: step size, iteration counts, convergence deltas."""

    step_size: float = 0.1
    n_iter_first: int = 20
    n_iter: int = 10
    warmup_convergence_delta: float = 1e-3
    s2_convergence_delta: float = 1e-3

    def __post_init__(self) -> None:
        _check_positive("solver.step_size", self.step_size)
        _check_positive_int("solver.n_iter_first", self.n_iter_first)
        _check_positive_int("solver.n_iter", self.n_iter)
        _check_nonneg("solver.warmup_convergence_delta", self.warmup_convergence_delta)
        _check_nonneg("solver.s2_convergence_delta", self.s2_convergence_delta)


@dataclass
class WarmupConfig:
    """Bone-direction warmup stage: weight, iteration counts, topology."""

    enabled: bool = True
    weight: float = 1.0
    iters_first: int = 20
    iters: int = 1
    topology: str = "chain"
    edge_weight_mode: str = "inv_count"

    def __post_init__(self) -> None:
        _check_positive_int("warmup.iters_first", self.iters_first)
        _check_positive_int("warmup.iters", self.iters)
        _check_nonneg("warmup.weight", self.weight)
        _check_choice("warmup.topology", self.topology, BONE_DIR_TOPOLOGIES)
        _check_choice("warmup.edge_weight_mode", self.edge_weight_mode, BONE_DIR_EDGE_WEIGHT_MODES)


@dataclass
class PosePriorConfig:
    """Bone-direction pose prior: weight, topology, edge-weight mode."""

    bone_dir_weight: float = 0.1
    topology: str = "chain"
    edge_weight_mode: str = "inv_count"

    def __post_init__(self) -> None:
        _check_nonneg("pose_prior.bone_dir_weight", self.bone_dir_weight)
        _check_choice("pose_prior.topology", self.topology, BONE_DIR_TOPOLOGIES)
        _check_choice("pose_prior.edge_weight_mode", self.edge_weight_mode, BONE_DIR_EDGE_WEIGHT_MODES)


@dataclass
class SmoothConfig:
    """Temporal-smoothing regularizer weight."""

    weight: float = 2.5

    def __post_init__(self) -> None:
        _check_nonneg("smooth.weight", self.weight)


@dataclass
class DelaunayConfig:
    """Delaunay interaction mesh: whether ``MeshConfig.interaction_radius_m`` cuts its long edges."""

    use_edge_threshold: bool = False


@dataclass
class EdgeWeightConfig:
    """Laplacian edge weights: mode, exponential decay, compact cutoff radius."""

    mode: str = "exp"
    exp_decay_k: float | None = 30.0
    cutoff_radius: float | None = None

    def __post_init__(self) -> None:
        _check_choice("mesh.edge_weight.mode", self.mode, LAPLACIAN_DISTANCE_WEIGHT_MODES)
        _check_optional_positive("mesh.edge_weight.exp_decay_k", self.exp_decay_k)
        _check_optional_positive("mesh.edge_weight.cutoff_radius", self.cutoff_radius)
        if self.mode == "compact_only_exp" and self.cutoff_radius is None:
            raise ValueError("mesh.edge_weight.cutoff_radius is required for compact_only_exp")


@dataclass
class MeshConfig:
    """Interaction mesh: topology, Laplacian weight, contact radii, sub-configs."""

    topology: str = "delaunay"
    laplacian_weight: float = 500.0
    interaction_radius_m: float = 0.05
    length_scale: float | None = None
    delaunay: DelaunayConfig = field(default_factory=DelaunayConfig)
    edge_weight: EdgeWeightConfig = field(default_factory=EdgeWeightConfig)
    skeleton_sphere_radius: float = 0.01
    contact_margin_m: float = 0.002

    def __post_init__(self) -> None:
        _check_choice("mesh.topology", self.topology, TOPOLOGY_MODES)
        _check_nonneg("mesh.laplacian_weight", self.laplacian_weight)
        _check_positive("mesh.interaction_radius_m", self.interaction_radius_m)
        _check_optional_positive("mesh.length_scale", self.length_scale)
        _check_positive("mesh.skeleton_sphere_radius", self.skeleton_sphere_radius)
        _check_nonneg("mesh.contact_margin_m", self.contact_margin_m)


@dataclass
class ObjectSamplerConfig:
    """Object surface-point sampler: mode, seed, proximity sigma, proximity alpha."""

    mode: str = "proxcurv"
    seed: int = 0
    sigma: float = 0.010
    alpha: float = 30.0

    def __post_init__(self) -> None:
        _check_choice("object.sampler.mode", self.mode, OBJECT_SAMPLER_MODES)
        _check_nonneg_int("object.sampler.seed", self.seed)
        _check_positive("object.sampler.sigma", self.sigma)
        _check_nonneg("object.sampler.alpha", self.alpha)


@dataclass
class ObjectConfig:
    """Object configuration: sample count, object-frame usage, nested sampler."""

    sample_count: int = 50
    use_frame: bool = False
    sampler: ObjectSamplerConfig = field(default_factory=ObjectSamplerConfig)

    def __post_init__(self) -> None:
        _check_positive_int("object.sample_count", self.sample_count)
        self.sampler.__post_init__()


@dataclass
class HybridPenetrationConfig:
    """Hybrid soft/hard penetration constraint: enable, backstop, slack weight."""

    enabled: bool = True
    backstop: float = 0.030
    slack_weight: float = 1.0e5

    def __post_init__(self) -> None:
        _check_nonneg("penetration.hybrid.backstop", self.backstop)
        _check_positive("penetration.hybrid.slack_weight", self.slack_weight)


@dataclass
class PenetrationConfig:
    """Non-penetration constraint: warmup/S2 enables, tolerance, trust shrinks, hybrid."""

    enabled_warmup: bool = True
    enabled_s2: bool = True
    tolerance: float = 1e-3
    max_trust_shrinks: int = 3
    hybrid: HybridPenetrationConfig = field(default_factory=HybridPenetrationConfig)

    def __post_init__(self) -> None:
        _check_nonneg("penetration.tolerance", self.tolerance)
        _check_nonneg("penetration.max_trust_shrinks", self.max_trust_shrinks)
        if self.hybrid.enabled and self.hybrid.backstop < self.tolerance:
            raise ValueError("penetration.hybrid.backstop must be >= penetration.tolerance when hybrid is enabled")

    @property
    def enabled_any(self) -> bool:
        return self.enabled_warmup or self.enabled_s2


@dataclass
class WristConfig:
    """Floating-wrist DoF: motion enable, translation/rotation limits, zero-pose weights."""

    motion_enabled: bool = True
    translation_limit: float = 0.05
    rotation_limit: float = 0.7853981633974483
    zero_translation_weight: float = 100.0
    zero_rotation_weight: float = 1.0

    def __post_init__(self) -> None:
        _check_nonneg("wrist.translation_limit", self.translation_limit)
        _check_nonneg("wrist.rotation_limit", self.rotation_limit)
        _check_nonneg("wrist.zero_translation_weight", self.zero_translation_weight)
        _check_nonneg("wrist.zero_rotation_weight", self.zero_rotation_weight)


@dataclass
class PointMappingConfig:
    """Keypoint-to-link mapping: MCP / thumb-CMC surrogates and MCP surface offset."""

    mcp_surrogate: str = "link2"
    mcp_surface_offset_m: float = 0.0
    thumb_cmc_surrogate: str = "link2"

    def __post_init__(self) -> None:
        _check_choice("point_mapping.mcp_surrogate", self.mcp_surrogate, MCP_SURROGATES)
        _check_choice("point_mapping.thumb_cmc_surrogate", self.thumb_cmc_surrogate, MCP_SURROGATES)


@dataclass
class SelfColConfig:
    """Finger self-collision: rows in the per-frame solve and the whole-trajectory cleanup pass."""

    margin: float = 0.001
    radius_mode: str = "diag"
    max_window: int = 40
    overlap: int = 10
    max_iters: int = 8
    lam_smooth: float = 2.5
    dq_max: float = 0.1
    tol: float = 0.0005

    def __post_init__(self) -> None:
        _check_nonneg("selfcol.margin", self.margin)
        _check_choice("selfcol.radius_mode", self.radius_mode, ("diag", "aabb"))
        _check_positive_int("selfcol.max_window", self.max_window)
        _check_nonneg_int("selfcol.overlap", self.overlap)
        _check_positive_int("selfcol.max_iters", self.max_iters)
        _check_nonneg("selfcol.lam_smooth", self.lam_smooth)
        _check_positive("selfcol.dq_max", self.dq_max)
        _check_nonneg("selfcol.tol", self.tol)
        if self.overlap >= self.max_window:
            raise ValueError("selfcol.overlap must be smaller than selfcol.max_window")


@dataclass
class HandRetargetConfig:
    """Structured configuration for the retargeting pipeline."""

    model: ModelConfig = field(default_factory=ModelConfig)
    solver: SolverConfig = field(default_factory=SolverConfig)
    warmup: WarmupConfig = field(default_factory=WarmupConfig)
    pose_prior: PosePriorConfig = field(default_factory=PosePriorConfig)
    smooth: SmoothConfig = field(default_factory=SmoothConfig)
    mesh: MeshConfig = field(default_factory=MeshConfig)
    object: ObjectConfig = field(default_factory=ObjectConfig)
    penetration: PenetrationConfig = field(default_factory=PenetrationConfig)
    wrist: WristConfig = field(default_factory=WristConfig)
    point_mapping: PointMappingConfig = field(default_factory=PointMappingConfig)
    selfcol: SelfColConfig = field(default_factory=SelfColConfig)

    def __post_init__(self) -> None:
        if self.penetration.enabled_any:
            min_radius = self.penetration.tolerance + 0.030
            if self.mesh.interaction_radius_m < min_radius:
                raise ValueError(
                    f"mesh.interaction_radius_m={self.mesh.interaction_radius_m:.4f} m is below the "
                    f"SQP single-step safety bound (penetration.tolerance + 0.030 = {min_radius:.4f} m). "
                    "Increase the radius."
                )

    def revalidate_after_override(self) -> None:
        """Re-run every section's validation, nested ones first, after overrides mutate fields in place."""

        def validate(section: object) -> None:
            for f in dataclasses.fields(section):
                child = getattr(section, f.name)
                if is_dataclass(child):
                    validate(child)
            getattr(section, "__post_init__", lambda: None)()

        validate(self)


def _dataclass_field_names(target: object) -> set[str]:
    return {f.name for f in fields(target)}


def _unknown_message(path: tuple[str, ...] | list[str]) -> str:
    return f"unknown config override {'.'.join(path)!r}; use dotted structured paths"


def _validate_leaf_type(current: object, value: Any, path: tuple[str, ...]) -> None:
    if isinstance(current, bool) and not isinstance(value, bool):
        raise TypeError(f"Config key {'.'.join(path)} must be a boolean, got {type(value).__name__}")
    if isinstance(value, dict) and not isinstance(current, dict):
        raise TypeError(f"Config key {'.'.join(path)} must not be a mapping")


def _set_dataclass_leaf(target: object, key: str, value: Any, *, path: tuple[str, ...], unknown_error) -> None:
    if key not in _dataclass_field_names(target):
        raise unknown_error(_unknown_message(path))
    current = getattr(target, key)
    if is_dataclass(current):
        if not isinstance(value, dict):
            raise TypeError(f"Config key {'.'.join(path)} must be a mapping")
        _apply_mapping(current, value, path=path)
        return
    _validate_leaf_type(current, value, path)
    setattr(target, key, value)


def _set_leaf(target: object, key: str, value: Any, *, path: tuple[str, ...], unknown_error) -> None:
    if is_dataclass(target):
        _set_dataclass_leaf(target, key, value, path=path, unknown_error=unknown_error)
        return
    raise TypeError(f"Config key {'.'.join(path)} cannot set a value on a non-mapping target")


def _apply_mapping(target: object, data: dict[str, Any], *, path: tuple[str, ...]) -> None:
    if not is_dataclass(target):
        raise TypeError(f"Config key {'.'.join(path)} must not contain nested keys")
    known_fields = _dataclass_field_names(target)
    for key, value in data.items():
        key_path = (*path, key)
        if key not in known_fields:
            raise KeyError(f"Unknown config key: {'.'.join(key_path)}")
        current = getattr(target, key)
        if is_dataclass(current):
            if not isinstance(value, dict):
                raise TypeError(f"Config key {'.'.join(key_path)} must be a mapping")
            _apply_mapping(current, value, path=key_path)
        else:
            _set_leaf(target, key, value, path=key_path, unknown_error=KeyError)


def apply_config_data(config: object, data: dict[str, Any]) -> None:
    """Apply a JSON/YAML mapping onto a dataclass-backed config object."""
    _apply_mapping(config, data, path=())


def set_config_value(config: object, dotted_path: str, value: Any) -> None:
    """Apply one dotted override (``mesh.laplacian_weight=200``) onto a config object."""
    if not dotted_path:
        raise TypeError("unknown config override ''; use dotted structured paths")
    target = config
    parts = dotted_path.split(".")
    for attr in parts[:-1]:
        if not is_dataclass(target) or attr not in _dataclass_field_names(target):
            raise TypeError(_unknown_message(parts))
        target = getattr(target, attr)
    _set_leaf(target, parts[-1], value, path=tuple(parts), unknown_error=TypeError)
