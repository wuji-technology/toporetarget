"""Whole-trajectory finger self-collision cleanup: capsule proxies, the window SQP and the sliding pass."""

from __future__ import annotations

import itertools
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import mujoco
import numpy as np

from toporetarget.solver.hand import WRIST_DOF, HandSpec, body_link_index
from toporetarget.solver.qp import (
    ConstraintSet,
    CostTerm,
    PenetrationSpec,
    QpSolverOptions,
    QpTask,
    StepBounds,
    build_penetration_constraints,
    measure_penetration_mm,
    solve_qp_task,
)
from toporetarget.solver.settings import SelfColConfig

_AXIS_TO_Z_QUAT = {
    0: np.array([0.70710678, 0.0, 0.70710678, 0.0]),
    1: np.array([0.70710678, -0.70710678, 0.0, 0.0]),
    2: np.array([1.0, 0.0, 0.0, 0.0]),
}


def link_collision_geoms(model, link_ids: Sequence[int]) -> list[list[int]]:
    """Per entry of ``link_ids``, the collision mesh geoms of that link (``body_link_index``)."""
    owners = body_link_index(model, link_ids)
    geoms: list[list[int]] = [[] for _ in link_ids]
    for geom in range(model.ngeom):
        link = owners.get(int(model.geom_bodyid[geom]))
        if link is None or model.geom_type[geom] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        if model.geom_contype[geom] or model.geom_conaffinity[geom]:
            geoms[link].append(geom)
    return geoms


def _mesh_verts_in_link(model, data, geom_id: int, link_id: int) -> np.ndarray:
    """A mesh geom's vertices in the frame of ``link_id`` (forward kinematics must have run)."""
    mesh_id = int(model.geom_dataid[geom_id])
    adr, num = int(model.mesh_vertadr[mesh_id]), int(model.mesh_vertnum[mesh_id])
    v_geom = model.mesh_vert[adr : adr + num].reshape(-1, 3).astype(np.float64)
    v_world = v_geom @ data.geom_xmat[geom_id].reshape(3, 3).T + data.geom_xpos[geom_id]
    return (v_world - data.xpos[link_id]) @ data.xmat[link_id].reshape(3, 3)


def _fit_capsule_local(verts_body: np.ndarray, radius_mode: str = "diag") -> tuple[int, np.ndarray, float, float]:
    """Fit a capsule to body-frame vertices: ``(longest_axis, center, radius, half_length)``."""
    lo, hi = verts_body.min(0), verts_body.max(0)
    ext = hi - lo
    axis = int(np.argmax(ext))
    center = (hi + lo) / 2.0
    perp = [i for i in range(3) if i != axis]
    if radius_mode == "diag":
        radial = np.sqrt(((verts_body[:, perp] - center[perp]) ** 2).sum(axis=1))
        radius = float(radial.max())
    else:
        radius = float(max(ext[perp]) / 2.0)
    half_len = max(float(ext[axis] / 2.0 - radius), 1e-4)
    return axis, center, radius, half_len


def build_selfcol_model(
    mjcf_path: str,
    *,
    margin_m: float = 0.001,
    links_per_finger: Sequence[Sequence[str]],
    radius_mode: str = "diag",
) -> tuple[Any, list[int], list[int], list[list[int]]]:
    """Compile the bare hand MJCF with one inert capsule proxy per link of ``links_per_finger``."""
    links = tuple(link for finger in links_per_finger for link in finger)
    finger_of = [index for index, finger in enumerate(links_per_finger) for _ in finger]
    spec = mujoco.MjSpec.from_file(mjcf_path)
    raw = spec.compile()
    raw_data = mujoco.MjData(raw)
    mujoco.mj_kinematics(raw, raw_data)
    link_ids = []
    for name in links:
        bid = mujoco.mj_name2id(raw, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid < 0:
            raise ValueError(f"finger link {name!r} not in MJCF")
        link_ids.append(bid)
    fits: dict[str, tuple[int, np.ndarray, float, float]] = {}
    for name, bid, geoms in zip(links, link_ids, link_collision_geoms(raw, link_ids), strict=True):
        if not geoms:
            raise ValueError(f"no collision mesh on {name!r} to fit a proxy from")
        verts = np.concatenate([_mesh_verts_in_link(raw, raw_data, g, bid) for g in geoms])
        fits[name] = _fit_capsule_local(verts, radius_mode=radius_mode)

    proxy_names: list[str] = []
    for name in links:
        axis, center, radius, half_len = fits[name]
        g = spec.body(name).add_geom()
        g.name = f"{name}_selfproxy"
        g.type = mujoco.mjtGeom.mjGEOM_CAPSULE
        g.size = [radius + margin_m, half_len, 0.0]
        g.pos = center
        g.quat = _AXIS_TO_Z_QUAT[axis]
        g.contype = 0
        g.conaffinity = 0
        g.rgba = [1.0, 0.3, 0.3, 0.4]
        proxy_names.append(g.name)

    model = spec.compile()
    gids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, n) for n in proxy_names]
    bids = [int(model.geom_bodyid[g]) for g in gids]
    link_geoms = link_collision_geoms(model, bids)
    model.geom_contype[:] = 0
    model.geom_conaffinity[:] = 0
    all_fingers = (1 << len(links_per_finger)) - 1
    for index, geoms in enumerate(link_geoms):
        finger_bit = 1 << finger_of[index]
        model.geom_contype[geoms] = finger_bit
        model.geom_conaffinity[geoms] = all_fingers & ~finger_bit
    return model, gids, bids, link_geoms


def proxy_distance(model, data, geom_i: int, geom_j: int, *, distmax: float) -> tuple[float, np.ndarray]:
    """Signed distance and witness points between two proxy geoms at the current qpos."""
    fromto = np.zeros(6)
    phi = mujoco.mj_geomDistance(model, data, geom_i, geom_j, distmax, fromto)
    return float(phi), fromto


def proxy_jacobian(model, data, geom_i: int, geom_j: int, phi: float, fromto: np.ndarray) -> np.ndarray:
    """``d(phi)/dq`` as the sign-flipped contact-normal projection of the relative witness Jacobian."""
    p_a, p_b = fromto[:3], fromto[3:]
    diff = p_a - p_b
    nrm = float(np.linalg.norm(diff))
    if nrm < 1e-12:
        return np.zeros(model.nv)
    nhat = (1.0 if phi >= 0.0 else -1.0) * diff / nrm
    jac_a = np.zeros((3, model.nv))
    jac_b = np.zeros((3, model.nv))
    mujoco.mj_jac(model, data, jac_a, None, p_a, int(model.geom_bodyid[geom_i]))
    mujoco.mj_jac(model, data, jac_b, None, p_b, int(model.geom_bodyid[geom_j]))
    return nhat @ (jac_a - jac_b)


def proxy_pairs(finger_of: Sequence[int]) -> list[tuple[int, int]]:
    """All unordered proxy index pairs that belong to different fingers (``finger_of[i]`` is proxy ``i``'s finger)."""
    return [(i, j) for i, j in itertools.combinations(range(len(finger_of)), 2) if finger_of[i] != finger_of[j]]


@dataclass(frozen=True)
class SelfColSpec:
    """Finger self-collision constraint parameters."""

    tol: float = 0.001
    query_band: float = 0.015
    distmax: float = 0.020
    row_scale: float = 100.0
    residual: float = 1e-4
    object_residual: float = 1e-5

    def __post_init__(self) -> None:
        if not (self.distmax > self.query_band > 0.0):
            raise ValueError("require distmax > query_band > 0")
        if self.residual < 0.0 or self.object_residual < 0.0:
            raise ValueError("require residual >= 0 and object_residual >= 0")


def build_self_constraints(model, data, proxy_gids: list[int], finger_of: Sequence[int], spec: SelfColSpec):
    """Linearized active cross-finger non-penetration rows at the current qpos."""
    g_rows: list[np.ndarray] = []
    h_rows: list[float] = []
    active: list[tuple[int, int]] = []
    for i, j in proxy_pairs(finger_of):
        gi, gj = proxy_gids[i], proxy_gids[j]
        phi, fromto = proxy_distance(model, data, gi, gj, distmax=spec.distmax)
        if phi >= spec.query_band or np.linalg.norm(fromto) < 1e-9:
            continue
        row = proxy_jacobian(model, data, gi, gj, phi, fromto)
        g_rows.append(-row / spec.row_scale)
        h_rows.append((phi + spec.tol) / spec.row_scale)
        active.append((i, j))
    if not g_rows:
        return ConstraintSet.none(), []
    return ConstraintSet.from_rows(G=np.asarray(g_rows), h=np.asarray(h_rows)), active


class FingerClearance:
    """Cross-finger non-penetration rows of the per-frame solve, at the hand's capsule proxies."""

    def __init__(self, hand: HandSpec, config: SelfColConfig, nq: int) -> None:
        self.model, self.proxy_gids, _, _ = build_selfcol_model(
            str(hand.hand_xml),
            margin_m=config.margin,
            links_per_finger=hand.selfcol_links,
            radius_mode=config.radius_mode,
        )
        self.data = mujoco.MjData(self.model)
        self.finger_of = [index for index, finger in enumerate(hand.selfcol_links) for _ in finger]
        self.spec = SelfColSpec(tol=config.tol, row_scale=1.0)
        self.offset = nq - self.model.nq

    def __call__(self, q: np.ndarray) -> ConstraintSet:
        self.data.qpos[:] = q[self.offset :]
        mujoco.mj_forward(self.model, self.data)
        cset = build_self_constraints(self.model, self.data, self.proxy_gids, self.finger_of, self.spec)[0]
        if cset.G is None:
            return cset
        return ConstraintSet.from_rows(G=np.hstack([np.zeros((len(cset.G), self.offset)), cset.G]), h=cset.h)


@dataclass(frozen=True)
class ObjectClearance:
    """The hand-object non-penetration rows of the per-frame solve, for the finger-only window QP."""

    hand: Any
    wrist_qpos: np.ndarray
    object_position: np.ndarray
    object_quaternion_xyzw: np.ndarray
    spec: PenetrationSpec

    def _forward(self, frame: int, fingers: np.ndarray) -> np.ndarray:
        q = np.concatenate([self.wrist_qpos[frame], fingers])
        self.hand.set_object_poses(self.object_position[frame], self.object_quaternion_xyzw[frame])
        self.hand.forward(q)
        return q

    def constraints(self, frame: int, fingers: np.ndarray) -> ConstraintSet:
        """Rows over the finger joints only, left in metres."""
        cset = build_penetration_constraints(self.hand, self._forward(frame, fingers), self.spec)
        if cset.G is None:
            return cset
        return ConstraintSet.from_rows(G=cset.G[:, WRIST_DOF:], h=cset.h)

    def penetration_mm(self, fingers: np.ndarray) -> np.ndarray:
        """Per-frame max hand-object penetration depth of a ``(T, 20)`` finger trajectory."""
        return np.asarray(
            [
                measure_penetration_mm(
                    self.hand, self._forward(frame, fingers[frame]), query_threshold=self.spec.query_threshold
                )
                for frame in range(len(fingers))
            ]
        )


def _second_diff(n_frames: int) -> np.ndarray:
    """The ``(n_frames-2, n_frames)`` second-difference operator (stencil ``[1, -2, 1]``)."""
    if n_frames < 3:
        return np.zeros((0, n_frames))
    d2 = np.zeros((n_frames - 2, n_frames))
    for t in range(n_frames - 2):
        d2[t, t] = 1.0
        d2[t, t + 1] = -2.0
        d2[t, t + 2] = 1.0
    return d2


def build_window_cost(
    q_cur: np.ndarray,
    q_ref: np.ndarray,
    *,
    w_fid: float,
    lam_smooth: float = 0.0,
    nq: int,
) -> CostTerm:
    """The window CostTerm for the flattened step ``dq`` of shape ``(W nq,)``."""
    if not np.isfinite(lam_smooth) or lam_smooth < 0.0:
        raise ValueError(f"lam_smooth must be finite and non-negative, got {lam_smooth!r}")
    n_frames = q_cur.shape[0]
    n = n_frames * nq
    resid = (q_cur - q_ref).reshape(-1)
    hessian = 2.0 * w_fid * np.eye(n)
    grad = 2.0 * w_fid * resid

    d2 = _second_diff(n_frames)
    if d2.shape[0] > 0:
        s_frame = lam_smooth * (d2.T @ d2)
        smooth = np.kron(s_frame, np.eye(nq))
        hessian += 2.0 * smooth
        grad += 2.0 * (smooth @ q_cur.reshape(-1))

    return CostTerm(H=hessian, c=grad)


_HARD_OPTS = QpSolverOptions(
    max_trust_shrinks=3,
    hybrid_enabled=False,
    hybrid_tolerance=1e-3,
    hybrid_backstop=0.030,
    hybrid_slack_weight=1.0e5,
)
_HYBRID_OPTS = QpSolverOptions(
    max_trust_shrinks=3,
    hybrid_enabled=True,
    hybrid_tolerance=1e-3,
    hybrid_backstop=0.030,
    hybrid_slack_weight=1.0e7,
    hybrid_slack_violated_only=True,
)


def joint_position_bounds(model) -> tuple[np.ndarray, np.ndarray]:
    """Hard joint limits of the scalar-joint hand model used by the self-collision SQP."""
    if model.nq != model.njnt or model.nv != model.nq:
        raise ValueError("self-collision cleanup requires scalar hand joints")
    lower = np.full(model.nq, -np.inf)
    upper = np.full(model.nq, np.inf)
    for j in range(model.njnt):
        if model.jnt_limited[j]:
            lower[model.jnt_qposadr[j]], upper[model.jnt_qposadr[j]] = model.jnt_range[j]
    return lower, upper


def cap_min_phi(model, data, proxy_gids: list[int], finger_of: Sequence[int], q: np.ndarray) -> float:
    """Min capsule-proxy signed distance over all cross-finger pairs at ``q``."""
    data.qpos[:] = q
    mujoco.mj_forward(model, data)
    return min(
        proxy_distance(model, data, proxy_gids[i], proxy_gids[j], distmax=0.05)[0] for i, j in proxy_pairs(finger_of)
    )


def optimize_window(
    model,
    data,
    proxy_gids: list[int],
    finger_of: Sequence[int],
    q_ref_window: np.ndarray,
    *,
    q_init: np.ndarray | None = None,
    spec: SelfColSpec,
    clearance: ObjectClearance | None = None,
    frame_offset: int = 0,
    w_fid: float = 1.0,
    lam_smooth: float = 2.5,
    dq_max: float = 0.1,
    clamp_front: int = 2,
    clamp_back: int = 2,
    max_iters: int = 12,
    conv: float = 1e-4,
) -> tuple[np.ndarray, int]:
    """SQP-optimize one window; fidelity vs ``q_ref_window``, init from ``q_init``."""
    n_frames, nq = q_ref_window.shape
    lower, upper = joint_position_bounds(model)
    q_cur = np.clip(q_ref_window if q_init is None else q_init, lower, upper)
    clamp = set(range(clamp_front)) | set(range(n_frames - clamp_back, n_frames))

    n_iters = 0
    for it in range(max_iters + 1):
        g_rows: list[np.ndarray] = []
        h_rows: list[float] = []
        worst_self = worst_object = 0.0
        for t in range(n_frames):
            if t in clamp:
                continue
            data.qpos[:] = q_cur[t]
            mujoco.mj_forward(model, data)
            self_set = build_self_constraints(model, data, proxy_gids, finger_of, spec)[0]
            if self_set.G is not None:
                worst_self = min(worst_self, float(self_set.h.min()) * spec.row_scale)
            csets = [self_set]
            if clearance is not None:
                object_set = clearance.constraints(frame_offset + t, q_cur[t])
                if object_set.G is not None:
                    worst_object = min(worst_object, float(object_set.h.min()))
                csets.append(object_set)
            for cset in csets:
                if cset.G is None:
                    continue
                for g_local, h_local in zip(cset.G, cset.h, strict=True):
                    row = np.zeros(n_frames * nq)
                    row[t * nq : (t + 1) * nq] = g_local
                    g_rows.append(row)
                    h_rows.append(float(h_local))

        if not g_rows or (worst_self >= -spec.residual and worst_object >= -spec.object_residual):
            break
        if it == 0:
            object_floor = min(worst_object, 0.0) - spec.object_residual
        if worst_object >= object_floor:
            q_held = q_cur
        if it == max_iters:
            q_cur = q_held
            break

        n_iters = it + 1
        cost = build_window_cost(q_cur, q_ref_window, w_fid=w_fid, lam_smooth=lam_smooth, nq=nq)
        lb = np.maximum(lower - q_cur, -dq_max).ravel()
        ub = np.minimum(upper - q_cur, dq_max).ravel()
        for t in clamp:
            lb[t * nq : (t + 1) * nq] = 0.0
            ub[t * nq : (t + 1) * nq] = 0.0
        constraints = ConstraintSet.from_rows(G=np.asarray(g_rows), h=np.asarray(h_rows))
        task = QpTask(cost=cost, bounds=StepBounds(lb=lb, ub=ub), constraints=constraints)
        result = solve_qp_task(task, _HARD_OPTS)
        hard_feasible = result.found
        if not hard_feasible:
            result = solve_qp_task(task, _HYBRID_OPTS)
        dq = result.dq.reshape(n_frames, nq)
        q_cur = np.clip(q_cur + dq, lower, upper)
        if hard_feasible and np.abs(dq).max() < conv:
            break
    return q_cur, n_iters


def optimize_clip_sliding(
    model,
    data,
    proxy_gids: list[int],
    finger_of: Sequence[int],
    q_ref: np.ndarray,
    *,
    spec: SelfColSpec,
    clearance: ObjectClearance | None = None,
    w_fid: float = 1.0,
    lam_smooth: float = 2.5,
    dq_max: float = 0.1,
    max_w: int = 40,
    overlap: int = 10,
    edge_pad: int = 2,
    max_iters: int = 12,
) -> tuple[np.ndarray, int, int]:
    """Fixed-size sliding-window continuation SQP over the whole clip."""
    n = len(q_ref)
    lower, upper = joint_position_bounds(model)
    q_opt = np.clip(q_ref, lower, upper)
    step = max(1, max_w - overlap)
    cs = 0
    n_win = 0
    total_iters = 0
    while cs < n:
        ce = min(cs + max_w, n)
        colliding = any(cap_min_phi(model, data, proxy_gids, finger_of, q_opt[t]) < 0 for t in range(cs, ce))
        if colliding:
            q_w, it = optimize_window(
                model,
                data,
                proxy_gids,
                finger_of,
                q_ref[cs:ce],
                q_init=q_opt[cs:ce],
                spec=spec,
                clearance=clearance,
                frame_offset=cs,
                w_fid=w_fid,
                lam_smooth=lam_smooth,
                dq_max=dq_max,
                clamp_front=edge_pad if cs > 0 else 0,
                clamp_back=edge_pad if ce < n else 0,
                max_iters=max_iters,
            )
            q_opt[cs:ce] = q_w
            n_win += 1
            total_iters += it
        if ce == n:
            break
        cs += step
    return q_opt, n_win, total_iters


def _mesh_pen(model, data, link_geoms: list[list[int]], finger_of: Sequence[int], traj: np.ndarray) -> np.ndarray:
    """Per-frame min cross-finger mesh gap (m); negative means interpenetration."""
    if model.opt.disableflags & mujoco.mjtDisableBit.mjDSBL_CONTACT:
        raise RuntimeError(
            "mjDSBL_CONTACT is set on this model, so data.ncon stays 0 no matter how deeply the "
            "fingertips overlap and penetration would silently read as zero -- the one direction "
            "of error this function must not have"
        )
    finger_of_geom = {int(g): finger_of[index] for index, geoms in enumerate(link_geoms) for g in geoms}
    pairs = [
        (a, b) for a, b in itertools.combinations(sorted(finger_of_geom), 2) if finger_of_geom[a] != finger_of_geom[b]
    ]
    distmax = 0.05
    out = np.empty(len(traj))
    ft = np.zeros(6)
    for t in range(len(traj)):
        data.qpos[:] = traj[t]
        mujoco.mj_forward(model, data)
        deepest = 0.0
        for c in range(data.ncon):
            con = data.contact[c]
            finger_a, finger_b = finger_of_geom.get(int(con.geom1)), finger_of_geom.get(int(con.geom2))
            if finger_a is not None and finger_b is not None and finger_a != finger_b:
                deepest = min(deepest, float(con.dist))
        if deepest < 0.0:
            out[t] = deepest
            continue
        gap = distmax
        for a, b in pairs:
            centres = np.linalg.norm(data.geom_xpos[a] - data.geom_xpos[b])
            if centres - model.geom_rbound[a] - model.geom_rbound[b] < gap:
                gap = min(gap, mujoco.mj_geomDistance(model, data, a, b, distmax, ft))
        out[t] = max(0.0, gap)
    return out


def _link_positions(model, data, bids: list[int], q: np.ndarray) -> np.ndarray:
    data.qpos[:] = q
    mujoco.mj_forward(model, data)
    return np.array([data.xpos[b] for b in bids])


def optimize_trajectory(
    q_ref: np.ndarray,
    *,
    hand: HandSpec,
    config: SelfColConfig | None = None,
    clearance: ObjectClearance | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Remove cross-finger self-collision from a ``(T, finger_dof)`` finger trajectory of ``hand``."""
    cfg = config or SelfColConfig()
    model, gids, bids, link_geoms = build_selfcol_model(
        str(hand.hand_xml),
        margin_m=cfg.margin,
        links_per_finger=hand.selfcol_links,
        radius_mode=cfg.radius_mode,
    )
    finger_of = [index for index, finger in enumerate(hand.selfcol_links) for _ in finger]
    data = mujoco.MjData(model)
    n = len(q_ref)

    phi_before = _mesh_pen(model, data, link_geoms, finger_of, q_ref)
    t0 = time.perf_counter()
    q_clean, n_win, iters = optimize_clip_sliding(
        model,
        data,
        gids,
        finger_of,
        q_ref,
        spec=SelfColSpec(tol=cfg.tol),
        clearance=clearance,
        lam_smooth=cfg.lam_smooth,
        dq_max=cfg.dq_max,
        max_w=cfg.max_window,
        overlap=cfg.overlap,
        edge_pad=2,
        max_iters=cfg.max_iters,
    )
    solve_s = time.perf_counter() - t0
    phi_after = _mesh_pen(model, data, link_geoms, finger_of, q_clean)

    drift = (
        np.concatenate(
            [
                np.linalg.norm(
                    _link_positions(model, data, bids, q_clean[t]) - _link_positions(model, data, bids, q_ref[t]),
                    axis=1,
                )
                for t in range(n)
            ]
        )
        * 1000.0
    )
    info = {
        "n": n,
        "windows": n_win,
        "iters": iters,
        "solve_s": solve_s,
        "phi_before": phi_before,
        "phi_after": phi_after,
        "drift_mean": float(drift.mean()),
        "drift_p95": float(np.percentile(drift, 95)),
        "drift_max": float(drift.max()),
    }
    return q_clean, info


def format_penetration(tag: str, phi: np.ndarray) -> str:
    """One human-readable before/after penetration line."""
    n = max(len(phi), 1)
    pen = -phi[phi < 0] * 1000.0
    return (
        f"{tag}: max-pen {(pen.max() if len(pen) else 0):.2f}mm  "
        f"mean-pen {(pen.mean() if len(pen) else 0):.2f}mm  "
        f"pen-frames {100 * len(pen) / n:.1f}%"
    )
