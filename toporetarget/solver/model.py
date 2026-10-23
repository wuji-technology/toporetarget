"""The hand in MuJoCo: the scene with its virtual wrist, the hand-object collision model and the FK backend."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from toporetarget.config import COLLISION_SIDECAR
from toporetarget.solver.hand import WRIST_DOF, HandPenetrationContact, HandSpec


def _set_zero_joint_damping(joint) -> None:
    """Zero MjSpec joint damping across MuJoCo versions (scalar before 3.6, length-3 vector after)."""
    try:
        joint.damping = 0.0
    except (TypeError, ValueError):
        joint.damping = [0.0, 0.0, 0.0]


_WRIST_INERTIAL: dict[str, Any] = {
    "mass": 0.05,
    "pos": [0, 0, 0],
    "inertia": [0.0001, 0.0001, 0.0001],
}

_WRIST6DOF_SLIDE: dict[str, Any] = {
    "range": [-0.5, 0.5],
    "kp": 1600.0,
    "kv": 60.0,
    "forcerange": [-100, 100],
    "armature": 0.01,
}

_WRIST6DOF_HINGE: dict[str, Any] = {
    "range": [-3.14159, 3.14159],
    "kp": 15.0,
    "kv": 0.5,
    "forcerange": [-50, 50],
    "armature": 0.01,
}

_WRIST_SLIDE_NAMES: list[str] = ["wrist_tx", "wrist_ty", "wrist_tz"]
_WRIST_HINGE_NAMES: list[str] = ["wrist_rx", "wrist_ry", "wrist_rz"]
_WRIST_AXES: list[list[int]] = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

_PALM_FRAME_OFFSET: tuple[float, float, float] = (0.0, 0.0, -0.05)


def load_scene_model(hand: HandSpec) -> mujoco.MjModel:
    """Compile the hand's single-hand scene with the standard MjSpec injections applied."""
    spec = mujoco.MjSpec.from_file(str(hand.scene_xml))
    apply_standard_hand_injections(spec, hand)
    model = spec.compile()
    hand.check_scene_model(model)
    return model


def apply_standard_hand_injections(spec: mujoco.MjSpec, hand: HandSpec) -> None:
    """Re-apply the standard single-hand injection set to an existing MjSpec."""
    pre_qpos, pre_ctrl = _pre_injection_home_keyframe(spec)
    _inject_wrist_frame(spec, hand)
    _inject_wrist6dof_mode(spec, hand.wrist_body)
    if pre_qpos is not None:
        _reindex_home_keyframe(spec, pre_qpos, pre_ctrl, hand.finger_dof)


def _pre_injection_home_keyframe(spec: mujoco.MjSpec) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Snapshot the ``home`` keyframe before joints are injected, if it exists."""
    for k in spec.keys:
        if k.name == "home":
            return np.array(k.qpos, dtype=np.float64), np.array(k.ctrl, dtype=np.float64)
    return None, None


def _inject_wrist_frame(spec: mujoco.MjSpec, hand: HandSpec) -> None:
    """Inject ``palm_frame_body`` with RGB axis markers (visual reference only)."""
    hand_root = spec.body(hand.root_body)

    frame_body = hand_root.add_body()
    frame_body.name = "palm_frame_body"
    frame_body.pos = list(_PALM_FRAME_OFFSET)

    frame_site = frame_body.add_site()
    frame_site.name = "palm_frame"
    frame_site.pos = [0, 0, 0]
    frame_site.size = [0.005, 0, 0]
    frame_site.rgba = [1, 1, 0, 1]
    frame_site.group = 0

    sqrt2_2 = 0.7071068
    for axis_name, pos, quat, rgba in [
        ("x", [0.02, 0, 0], [sqrt2_2, 0, sqrt2_2, 0], [1, 0, 0, 0.9]),
        ("y", [0, 0.02, 0], [sqrt2_2, sqrt2_2, 0, 0], [0, 1, 0, 0.9]),
        ("z", [0, 0, 0.02], [1, 0, 0, 0], [0, 0, 1, 0.9]),
    ]:
        g = frame_body.add_geom()
        g.name = f"palm_frame_{axis_name}"
        g.type = mujoco.mjtGeom.mjGEOM_CYLINDER
        g.pos = pos
        g.quat = quat
        g.size = [0.003, 0.02, 0]
        g.rgba = rgba
        g.contype = 0
        g.conaffinity = 0
        g.group = 0
        g.mass = 0


def _inject_wrist6dof_mode(spec: mujoco.MjSpec, wrist_body_name: str) -> None:
    """Inject the 6-DoF virtual wrist (3 slide + 3 hinge) with PD position actuators into the named scene body."""
    wrist_body = spec.body(wrist_body_name)

    wrist_body.mass = _WRIST_INERTIAL["mass"]
    wrist_body.ipos = _WRIST_INERTIAL["pos"]
    wrist_body.inertia = _WRIST_INERTIAL["inertia"]
    wrist_body.explicitinertial = True

    for name, axis in zip(_WRIST_SLIDE_NAMES, _WRIST_AXES, strict=True):
        jnt = wrist_body.add_joint()
        jnt.name = name
        jnt.type = mujoco.mjtJoint.mjJNT_SLIDE
        jnt.axis = axis
        jnt.range = _WRIST6DOF_SLIDE["range"]
        _set_zero_joint_damping(jnt)
        jnt.armature = _WRIST6DOF_SLIDE["armature"]

    for name, axis in zip(_WRIST_HINGE_NAMES, _WRIST_AXES, strict=True):
        jnt = wrist_body.add_joint()
        jnt.name = name
        jnt.type = mujoco.mjtJoint.mjJNT_HINGE
        jnt.axis = axis
        jnt.range = _WRIST6DOF_HINGE["range"]
        _set_zero_joint_damping(jnt)
        jnt.armature = _WRIST6DOF_HINGE["armature"]

    joint_cfgs = [(n, _WRIST6DOF_SLIDE) for n in _WRIST_SLIDE_NAMES]
    joint_cfgs += [(n, _WRIST6DOF_HINGE) for n in _WRIST_HINGE_NAMES]
    for joint_name, cfg in joint_cfgs:
        act = spec.add_actuator()
        act.name = f"{joint_name}_act"
        act.target = joint_name
        act.trntype = mujoco.mjtTrn.mjTRN_JOINT
        act.gaintype = mujoco.mjtGain.mjGAIN_FIXED
        act.gainprm[0] = cfg["kp"]
        act.biastype = mujoco.mjtBias.mjBIAS_AFFINE
        act.biasprm[1] = -cfg["kp"]
        act.biasprm[2] = -cfg["kv"]
        act.forcerange = cfg["forcerange"]
        act.ctrlrange = cfg["range"]


def _reindex_home_keyframe(spec: mujoco.MjSpec, pre_qpos: np.ndarray, pre_ctrl: np.ndarray, finger_dof: int) -> None:
    """Remap the ``home`` keyframe from the pre- to the post-injection layout."""
    new_qpos = np.concatenate([np.zeros(WRIST_DOF), pre_qpos[:finger_dof]])
    new_ctrl = np.concatenate([pre_ctrl[:finger_dof], np.zeros(WRIST_DOF)])
    for k in spec.keys:
        if k.name == "home":
            k.qpos = new_qpos
            k.ctrl = new_ctrl
            return


def add_object_collision(spec, body, mesh_path, *, prefix, contype, conaffinity):
    """Add the object's collision geoms to ``body``, preferring the sidecar parts."""
    path = Path(mesh_path).resolve()
    sidecar = path.with_name(COLLISION_SIDECAR)
    geoms = []
    if sidecar.exists():
        parts = json.loads(sidecar.read_text()).get("geoms")
        if not parts:
            raise ValueError(f"Empty collision geometry: {sidecar}")
        for i, part in enumerate(parts):
            kind = part["type"]
            if kind == "mesh":
                part_path = (sidecar.parent / part["file"]).resolve()
                if not part_path.is_relative_to(sidecar.parent) or not part_path.is_file():
                    raise ValueError(f"Invalid convex part {part['file']!r}: {sidecar}")
                mesh = spec.add_mesh()
                mesh.name, mesh.file = f"{prefix}_part_{i}", str(part_path)
                geom = body.add_geom()
                geom.name = f"{prefix}_geom" + (f"_{i}" if i else "")
                geom.type = mujoco.mjtGeom.mjGEOM_MESH
                geom.meshname = mesh.name
                geoms.append(geom)
                continue
            if kind not in ("box", "cylinder"):
                raise ValueError(f"Unsupported primitive: {kind}")
            size = np.asarray(part["size"], dtype=float)
            pos = np.asarray(part["pos"], dtype=float)
            quat = np.asarray(part["quat"], dtype=float)
            if (
                size.shape != (3,)
                or pos.shape != (3,)
                or quat.shape != (4,)
                or not all(np.isfinite(v).all() for v in (size, pos, quat))
                or np.any(size[: 3 if kind == "box" else 2] <= 0)
                or not np.isclose(np.linalg.norm(quat), 1)
            ):
                raise ValueError(f"Invalid primitive geometry: {sidecar}")
            geom = body.add_geom()
            geom.name = f"{prefix}_geom" + (f"_{i}" if i else "")
            geom.type = getattr(mujoco.mjtGeom, f"mjGEOM_{kind.upper()}")
            geom.size, geom.pos, geom.quat = size, pos, quat
            geoms.append(geom)
    else:
        mesh = spec.add_mesh()
        mesh.name, mesh.file = f"{prefix}_mesh", str(path)
        geom = body.add_geom()
        geom.name = f"{prefix}_geom"
        geom.type = mujoco.mjtGeom.mjGEOM_MESH
        geom.meshname = mesh.name
        geoms.append(geom)
    for geom in geoms:
        geom.contype, geom.conaffinity = contype, conaffinity
    return geoms


HAND_CONTYPE_BIT: int = 1

OBJECT_CONTYPE_BIT: int = 2

OBJECT_BODY = "retarget_object"
OBJECT_GEOM_PREFIX = "retarget_obj"


@dataclass(frozen=True)
class ObjectCollisionModel:
    """A recompiled MuJoCo model plus the cached collision ids."""

    model: Any
    data: Any
    q_lb: np.ndarray
    q_ub: np.ndarray
    hand_col_geom_ids: list[int]
    obj_geom_ids: list[int]


def inject_object_collision_model(hand: HandSpec, mesh_paths: Sequence[str]) -> ObjectCollisionModel:
    """Compile the hand's scene with one mocap collision body per object."""
    spec = _load_object_spec(hand, mesh_paths)
    return _compile_and_cache(spec, hand)


def set_object_poses(data, positions: np.ndarray, quats_xyzw: np.ndarray) -> None:
    """Set the object mocap poses, ``(K, 3)`` and scipy xyzw ``(K, 4)``, in object order."""
    for k, (pos, quat_xyzw) in enumerate(zip(positions, quats_xyzw, strict=True)):
        data.mocap_pos[k] = pos
        data.mocap_quat[k] = [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]]


def query_hand_penetration(
    model,
    data,
    hand_col_geom_ids: list[int],
    obj_geom_ids: list[int],
    threshold: float,
    fromto: np.ndarray,
    jacp: np.ndarray,
) -> list[HandPenetrationContact]:
    """Signed-distance rows for every hand collision geom near an object."""
    mujoco.mj_collision(model, data)
    touching = {frozenset((int(data.contact[i].geom1), int(data.contact[i].geom2))) for i in range(data.ncon)}
    results: list[HandPenetrationContact] = []
    for part_id in obj_geom_ids:
        results.extend(_query_part(model, data, hand_col_geom_ids, part_id, threshold, fromto, jacp, touching))
    return results


def _query_part(model, data, hand_col_geom_ids, obj_geom_id, threshold, fromto, jacp, touching):
    """One object collision part against every active hand geom."""
    results: list[HandPenetrationContact] = []
    part_pos, part_radius = data.geom_xpos[obj_geom_id], model.geom_rbound[obj_geom_id]
    for gid in hand_col_geom_ids:
        if part_radius > 0.0 and model.geom_rbound[gid] > 0.0:
            gap = np.linalg.norm(data.geom_xpos[gid] - part_pos) - model.geom_rbound[gid] - part_radius
            if gap > threshold:
                continue
        phi = mujoco.mj_geomDistance(model, data, gid, obj_geom_id, threshold, fromto)
        if phi > threshold or (phi < 0.0 and frozenset((gid, obj_geom_id)) not in touching):
            continue

        p_hand = fromto[:3]
        p_obj = fromto[3:]
        diff = p_hand - p_obj
        dist = np.linalg.norm(diff)
        if dist < 1e-10:
            continue

        sign = 1.0 if phi >= 0.0 else -1.0
        nhat = sign * (diff / dist)

        body_id = int(model.geom_bodyid[gid])
        mujoco.mj_jac(model, data, jacp, None, p_hand, body_id)
        results.append(HandPenetrationContact((nhat @ jacp).copy(), float(phi), int(gid), body_id))
    return results


def collect_hand_collision_geoms(model, wrist_body: str) -> list[int]:
    """Active hand collision geom ids in the wrist body's subtree."""
    wrist_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, wrist_body)
    if wrist_bid < 0:
        return []

    hand_bodies: set[int] = set()
    stack = [wrist_bid]
    while stack:
        body_id = stack.pop()
        hand_bodies.add(body_id)
        for child in range(model.nbody):
            if model.body_parentid[child] == body_id:
                stack.append(child)

    geom_ids: list[int] = []
    for geom_id in range(model.ngeom):
        if model.geom_bodyid[geom_id] not in hand_bodies:
            continue
        if not (model.geom_contype[geom_id] & HAND_CONTYPE_BIT):
            continue
        if not (model.geom_conaffinity[geom_id] & OBJECT_CONTYPE_BIT):
            continue
        geom_ids.append(int(geom_id))
    return geom_ids


def _load_object_spec(hand: HandSpec, mesh_paths: Sequence[str]):
    """Load the hand's scene and bolt on one mocap body per object, with its geoms."""
    spec = mujoco.MjSpec.from_file(str(hand.scene_xml))
    apply_standard_hand_injections(spec, hand)
    _enable_hand_collision_geoms(spec.body(hand.wrist_body), hand.collision_excluded_links)
    for k, mesh_path in enumerate(mesh_paths):
        _build_mocap_object_geom(spec, mesh_path, k)
    return spec


_GEOM_DISTANCE_CHECKED = False


def check_geom_distance_convention() -> None:
    """Refuse a MuJoCo whose ``mj_geomDistance`` returns ``fromto`` reversed."""
    global _GEOM_DISTANCE_CHECKED
    if _GEOM_DISTANCE_CHECKED:
        return
    model = mujoco.MjModel.from_xml_string(
        '<mujoco><asset><mesh name="cube" vertex="-.01 -.01 -.01 .01 -.01 -.01 -.01 .01 -.01 .01 .01 -.01 '
        '-.01 -.01 .01 .01 -.01 .01 -.01 .01 .01 .01 .01 .01"/></asset><worldbody>'
        '<body><geom type="mesh" mesh="cube"/></body>'
        '<body pos="0.03 0 0"><geom type="box" size="0.01 0.01 0.01"/></body></worldbody></mujoco>'
    )
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    fromto = np.zeros(6)
    mujoco.mj_geomDistance(model, data, 0, 1, 0.1, fromto)
    if not fromto[0] < fromto[3]:
        raise RuntimeError(
            f"MuJoCo {mujoco.__version__} returns mj_geomDistance's fromto with the second geom's point "
            "first for a mesh against a primitive, so the hand-object constraint would push the fingers "
            "into the object. Install mujoco>=3.4 (the 3.3 series has this)."
        )
    _GEOM_DISTANCE_CHECKED = True


def _compile_and_cache(spec, hand: HandSpec) -> ObjectCollisionModel:
    """Compile ``spec`` into an MjModel/MjData pair and pre-resolve the geom ids."""
    check_geom_distance_convention()
    model = spec.compile()
    hand.check_scene_model(model)
    data = mujoco.MjData(model)
    hand_col_geom_ids = collect_hand_collision_geoms(model, hand.wrist_body)
    if not hand_col_geom_ids:
        raise ValueError(
            "no hand collision geoms found -- the hand XML must author collision geoms with "
            "nonzero contype/conaffinity (see _enable_hand_collision_geoms)"
        )
    obj_geom_ids = [
        g
        for g in range(model.ngeom)
        if model.geom_contype[g] & OBJECT_CONTYPE_BIT and model.body_mocapid[model.geom_bodyid[g]] >= 0
    ]
    return ObjectCollisionModel(
        model=model,
        data=data,
        q_lb=model.jnt_range[:, 0].copy(),
        q_ub=model.jnt_range[:, 1].copy(),
        hand_col_geom_ids=hand_col_geom_ids,
        obj_geom_ids=obj_geom_ids,
    )


def _build_mocap_object_geom(spec, mesh_path: str, index: int) -> None:
    """Add the mocap body of object ``index``, holding its collision geometry in place."""
    obj_body = spec.worldbody.add_body()
    obj_body.name = f"{OBJECT_BODY}{index}"
    obj_body.mocap = True
    add_object_collision(
        spec,
        obj_body,
        mesh_path,
        prefix=f"{OBJECT_GEOM_PREFIX}{index}",
        contype=OBJECT_CONTYPE_BIT,
        conaffinity=HAND_CONTYPE_BIT,
    )


def _enable_hand_collision_geoms(body, excluded_links: frozenset[str], *, excluded: bool = False) -> None:
    """Retag the hand's collision geoms for the hand-vs-object bit pairing."""
    if body.name in excluded_links:
        excluded = True
    elif body.joints:
        excluded = False
    for geom in body.geoms:
        has_intent = geom.contype != 0 or geom.conaffinity != 0
        if has_intent and not excluded and not geom.name.startswith("palm_frame_"):
            geom.contype = HAND_CONTYPE_BIT
            geom.conaffinity = OBJECT_CONTYPE_BIT
        else:
            geom.contype = 0
            geom.conaffinity = 0
    for child in body.bodies:
        _enable_hand_collision_geoms(child, excluded_links, excluded=excluded)


class MuJoCoFloatingHandModel:
    """The hand of a ``HandSpec`` in MuJoCo, with the 6-DoF virtual wrist in front of its finger joints."""

    def __init__(self, hand: HandSpec) -> None:
        self.spec = hand
        self.has_object = False
        self.model = load_scene_model(hand)
        self.data = mujoco.MjData(self.model)
        self.nq = self.model.nq
        self.nv = self.model.nv

        self.q_lb = self.model.jnt_range[:, 0].copy()
        self.q_ub = self.model.jnt_range[:, 1].copy()

        self._body_ids: dict[str, int] = {}
        self._site_ids: dict[str, int] = {}
        self._rebuild_caches()

        self._jacp_buf = np.zeros((3, self.nv))
        self._pen_fromto_buf = np.zeros(6)
        self._hand_col_geom_ids: list[int] = []
        self._obj_geom_ids: list[int] = []

    def forward(self, q: np.ndarray) -> None:
        """Set qpos and run forward kinematics."""
        self.data.qpos[: self.nq] = q
        mujoco.mj_forward(self.model, self.data)

    def get_body_id(self, body_name: str) -> int:
        if body_name not in self._body_ids:
            raise ValueError(f"Body {body_name!r} not found. Available: {list(self._body_ids)}")
        return self._body_ids[body_name]

    def get_body_pos(self, body_name: str) -> np.ndarray:
        """World position of a body, or of a fingertip site."""
        return self._resolve_xpos(self.get_body_id(body_name))

    def get_body_positions(self, body_names: list[str]) -> np.ndarray:
        positions = np.zeros((len(body_names), 3))
        for i, name in enumerate(body_names):
            positions[i] = self.get_body_pos(name)
        return positions

    def get_body_pos_by_id(self, bid: int) -> np.ndarray:
        """World position by pre-resolved id; a negative id is a site-backed virtual body."""
        return self._resolve_xpos(bid)

    def get_body_jacp_by_id(self, bid: int) -> np.ndarray:
        """World-frame translational Jacobian by pre-resolved id (negative ids route to sites)."""
        return self._jacp_into(bid)

    def get_default_qpos(self) -> np.ndarray:
        """Wrist at zero, fingers at mid-range."""
        q = np.zeros(self.nq)
        q[WRIST_DOF:] = (self.q_lb[WRIST_DOF:] + self.q_ub[WRIST_DOF:]) / 2.0
        return q

    def inject_object_meshes(self, mesh_paths: Sequence[str]) -> None:
        """Rebuild the model with the objects' collision geometry for distance queries."""
        collision_model = inject_object_collision_model(self.spec, mesh_paths)
        self.model = collision_model.model
        self.data = collision_model.data

        self.q_lb = collision_model.q_lb
        self.q_ub = collision_model.q_ub
        self._rebuild_caches()

        self._hand_col_geom_ids = collision_model.hand_col_geom_ids
        self._obj_geom_ids = collision_model.obj_geom_ids
        self.has_object = True

    def set_object_poses(self, positions: np.ndarray, quats_xyzw: np.ndarray) -> None:
        """Set the object mocap poses; call before ``forward`` / distance queries."""
        set_object_poses(self.data, positions, quats_xyzw)

    def query_hand_penetration(self, threshold: float = 0.05) -> list[HandPenetrationContact]:
        """Signed distance and contact-normal Jacobian for hand x object pairs within ``threshold``."""
        if not self.has_object:
            return []
        return query_hand_penetration(
            self.model,
            self.data,
            self._hand_col_geom_ids,
            self._obj_geom_ids,
            threshold,
            self._pen_fromto_buf,
            self._jacp_buf,
        )

    def _resolve_xpos(self, bid: int) -> np.ndarray:
        """``data.xpos`` at ``bid``, or ``site_xpos`` when ``bid`` encodes a site."""
        if bid < 0:
            return self.data.site_xpos[-(bid + 1)].copy()
        return self.data.xpos[bid].copy()

    def _jacp_into(self, bid: int) -> np.ndarray:
        """Translational Jacobian for ``bid`` computed into the shared buffer."""
        jacp = self._jacp_buf
        if bid < 0:
            mujoco.mj_jacSite(self.model, self.data, jacp, None, -(bid + 1))
        else:
            mujoco.mj_jacBody(self.model, self.data, jacp, None, bid)
        return jacp.copy()

    def _rebuild_caches(self) -> None:
        """Rebuild the body/site id caches, the fingertip sites included."""
        self._body_ids.clear()
        for i in range(self.model.nbody):
            name = self.model.body(i).name
            if name:
                self._body_ids[name] = i

        self._site_ids.clear()
        for i in range(self.model.nsite):
            name = self.model.site(i).name
            if name:
                self._site_ids[name] = i

        for site_name in self.spec.tip_sites:
            if site_name in self._site_ids:
                self._body_ids[site_name] = -(self._site_ids[site_name] + 1)
