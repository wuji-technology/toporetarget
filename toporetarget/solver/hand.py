"""The robot hand: ``HandSpec``, the names the solver drives it by, and the interface a hand model exposes."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple, Protocol, runtime_checkable

import mujoco
import numpy as np

from toporetarget.config import recorded_asset_path, resolve_path, robot_config
from toporetarget.solver.landmarks import (
    MEDIAPIPE_FINGER_CHAINS,
    MEDIAPIPE_LANDMARK_COUNT,
    OPERATOR2MANO,
    estimate_frame_from_hand_points,
)

SIDES = ("left", "right")
WRIST_DOF = 6
WRIST_ROTATION_SLICE: slice = slice(3, 6)

_REFERENCE = "toporetarget/robots/wujihand2_right.json"


@dataclass(frozen=True)
class FingerSpec:
    """One finger: its link bodies from the knuckle out, its joints in MJCF order, and its fingertip site."""

    name: str
    links: tuple[str, ...]
    joints: tuple[str, ...]
    tip_site: str
    keypoint_links: tuple[int, int, int] = (0, 2, 3)

    def __post_init__(self) -> None:
        if not self.links or not self.joints or not self.tip_site:
            raise ValueError(f"finger {self.name!r} needs links, joints and a tip_site")
        if len(self.keypoint_links) != 3 or any(not 0 <= k < len(self.links) for k in self.keypoint_links):
            raise ValueError(
                f"finger {self.name!r}: keypoint_links {self.keypoint_links} must index its {len(self.links)} links"
            )

    @property
    def root_link(self) -> str:
        """The link the root landmark maps to in the plain mapping (``"link1"``)."""
        return self.links[self.keypoint_links[0]]

    @property
    def surrogate_link(self) -> str:
        """The link the root landmark maps to under the ``"link2"`` surrogate."""
        return self.links[1] if len(self.links) > 1 else self.links[0]


@dataclass(frozen=True, eq=False)
class HandSpec:
    """A robot hand by name, plus the scene and MJCF it is built from."""

    name: str
    side: str
    scene_xml: Path
    hand_xml: Path
    root_body: str
    wrist_body: str
    fingers: tuple[FingerSpec, ...]
    palm_align: np.ndarray
    collision_excluded_links: frozenset[str]
    selfcol_links: tuple[tuple[str, ...], ...]

    def __post_init__(self) -> None:
        if self.side not in SIDES:
            raise ValueError(f"hand side must be one of {SIDES}, got {self.side!r}")
        if len(self.fingers) != len(MEDIAPIPE_FINGER_CHAINS):
            raise ValueError(
                f"{self.name}: {len(MEDIAPIPE_FINGER_CHAINS)} fingers are needed, thumb to little finger, "
                f"got {len(self.fingers)}"
            )
        if len(self.selfcol_links) != len(self.fingers):
            raise ValueError(f"{self.name}: selfcol_links lists {len(self.selfcol_links)} fingers")
        names = [self.root_body, *self.link_names[1:], *self.joint_names, *self.tip_sites]
        duplicates = sorted({n for n in names if names.count(n) > 1})
        if duplicates:
            raise ValueError(f"{self.name}: names used twice: {duplicates}")
        palm_align = np.asarray(self.palm_align, dtype=np.float64)
        if palm_align.shape != (3, 3) or not np.allclose(palm_align @ palm_align.T, np.eye(3), atol=1e-6):
            raise ValueError(f"{self.name}: palm_align must be a (3, 3) rotation matrix")
        object.__setattr__(self, "palm_align", palm_align)
        object.__setattr__(self, "scene_xml", Path(self.scene_xml))
        object.__setattr__(self, "hand_xml", Path(self.hand_xml))

    @property
    def finger_dof(self) -> int:
        """The number of finger joints."""
        return sum(len(finger.joints) for finger in self.fingers)

    @property
    def nq(self) -> int:
        """The solver's decision variable: the virtual wrist plus the finger joints."""
        return WRIST_DOF + self.finger_dof

    @property
    def joint_names(self) -> tuple[str, ...]:
        """The finger joints, in the MJCF's qpos order."""
        return tuple(joint for finger in self.fingers for joint in finger.joints)

    @property
    def link_names(self) -> tuple[str, ...]:
        """The root body followed by every finger's links, knuckle out: the links the results record."""
        return (self.root_body, *(link for finger in self.fingers for link in finger.links))

    @property
    def tip_sites(self) -> tuple[str, ...]:
        """The fingertip sites, one per finger."""
        return tuple(finger.tip_site for finger in self.fingers)

    def keypoint_targets(self) -> dict[int, str]:
        """MediaPipe landmark -> the body (or fingertip site) it maps to, the plain mapping."""
        targets = {0: self.root_body}
        for finger, chain in zip(self.fingers, MEDIAPIPE_FINGER_CHAINS, strict=True):
            root, middle, distal, tip = chain[1:]
            targets[root] = finger.links[finger.keypoint_links[0]]
            targets[middle] = finger.links[finger.keypoint_links[1]]
            targets[distal] = finger.links[finger.keypoint_links[2]]
            targets[tip] = finger.tip_site
        return dict(sorted(targets.items()))

    def keypoint_surrogates(self) -> dict[int, str]:
        """Each finger's root landmark -> the finger's ``"link2"`` surrogate link."""
        pairs = zip(self.fingers, MEDIAPIPE_FINGER_CHAINS, strict=True)
        return {chain[1]: finger.surrogate_link for finger, chain in pairs}

    @property
    def mano_alignment(self) -> np.ndarray:
        """The operator -> MANO matrix pre-composed with the palm alignment."""
        return OPERATOR2MANO[self.side] @ self.palm_align

    def check_hand_model(self, model: mujoco.MjModel) -> None:
        """Raise unless the compiled hand MJCF has every named body, joint and site, in the spec's joint order."""
        bodies = {model.body(i).name for i in range(model.nbody)}
        sites = {model.site(i).name for i in range(model.nsite)}
        joints = [model.joint(i).name for i in range(model.njnt)]
        missing_bodies = [name for name in self.link_names if name not in bodies]
        if missing_bodies:
            raise ValueError(f"{self.name}: the hand MJCF {self.hand_xml} has no body {missing_bodies}")
        missing_sites = [name for name in self.tip_sites if name not in sites]
        if missing_sites:
            raise ValueError(f"{self.name}: the hand MJCF {self.hand_xml} has no site {missing_sites}")
        groups = (self.collision_excluded_links, *self.selfcol_links)
        unknown_links = sorted(name for group in groups for name in group if name not in bodies)
        if unknown_links:
            raise ValueError(f"{self.name}: the hand MJCF {self.hand_xml} has no body {unknown_links}")
        if tuple(joints) != self.joint_names:
            raise ValueError(
                f"{self.name}: robot.hand lists the joints {list(self.joint_names)} but the hand MJCF declares "
                f"{joints}; list every joint of the MJCF, in its order"
            )

    def check_scene_model(self, model: mujoco.MjModel) -> None:
        """Raise unless the compiled scene holds the wrist body and the hand under it, with the expected layout."""
        wrist = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, self.wrist_body)
        if wrist < 0:
            raise ValueError(f"{self.name}: the scene {self.scene_xml} has no body {self.wrist_body!r} for the wrist")
        joints = [model.joint(i).name for i in range(model.njnt)]
        if model.nq != self.nq or tuple(joints[-self.finger_dof :]) != self.joint_names:
            raise ValueError(
                f"{self.name}: the scene's joints are {joints}; expected {WRIST_DOF} wrist joints followed by "
                f"{list(self.joint_names)}"
            )

    @classmethod
    def from_config(
        cls, config: dict[str, Any], *, scene: str | Path | None = None, hand_xml: str | Path | None = None
    ) -> HandSpec:
        """The hand a merged config's ``robot`` block describes."""
        robot = robot_config(config)
        block = robot.get("hand")
        if not isinstance(block, dict):
            raise ValueError(f"robot.hand must describe the hand's bodies, joints and sites (see {_REFERENCE})")
        side = str(robot.get("hand_side", "")).strip().lower()
        if side not in SIDES:
            raise ValueError(f"robot.hand_side must be one of {SIDES}, got {robot.get('hand_side')!r}")
        scene_xml = Path(scene or resolve_path(robot["scene"], config)).resolve()
        if not scene_xml.is_file():
            raise FileNotFoundError(f"retarget scene XML not found: {scene_xml}")
        hand_path = Path(hand_xml or resolve_path(robot["xml"], config)).resolve()
        if not hand_path.is_file():
            hint = ""
            if "wuji-description" in hand_path.parts:
                hint = "; fetch the wuji-description submodule with `git submodule update --init`"
            raise FileNotFoundError(f"hand MJCF not found: {hand_path}{hint}")
        try:
            fingers = tuple(
                FingerSpec(
                    name=str(finger["name"]),
                    links=tuple(str(x) for x in finger["links"]),
                    joints=tuple(str(x) for x in finger["joints"]),
                    tip_site=str(finger["tip_site"]),
                    keypoint_links=tuple(int(k) for k in finger.get("keypoint_links", (0, 2, 3))),
                )
                for finger in block["fingers"]
            )
            root_body = str(block["root_body"])
            wrist_body = str(block["wrist_body"])
        except KeyError as exc:
            raise ValueError(f"robot.hand is missing {exc} (see {_REFERENCE})") from None
        excluded = block.get("collision_excluded_links", [finger.links[0] for finger in fingers])
        selfcol = block.get("selfcol_links", [finger.links[1:] for finger in fingers])
        palm_align = block.get("palm_align")
        if palm_align is None:
            model = mujoco.MjModel.from_xml_path(str(hand_path))
            palm_align = derive_palm_align(model, root_body, fingers, side)
        spec = cls(
            name=str(robot["name"]),
            side=side,
            scene_xml=scene_xml,
            hand_xml=hand_path,
            root_body=root_body,
            wrist_body=wrist_body,
            fingers=fingers,
            palm_align=np.asarray(palm_align, dtype=np.float64),
            collision_excluded_links=frozenset(str(x) for x in excluded),
            selfcol_links=tuple(tuple(str(x) for x in group) for group in selfcol),
        )
        spec.check_hand_model(mujoco.MjModel.from_xml_path(str(hand_path)))
        return spec

    def to_dict(self) -> dict[str, Any]:
        """The spec as JSON-ready data; packaged paths relative to the assets (recorded in a result's diagnostics)."""
        return {
            "name": self.name,
            "side": self.side,
            "scene_xml": recorded_asset_path(self.scene_xml),
            "hand_xml": recorded_asset_path(self.hand_xml),
            "root_body": self.root_body,
            "wrist_body": self.wrist_body,
            "fingers": [
                {
                    "name": finger.name,
                    "links": list(finger.links),
                    "joints": list(finger.joints),
                    "tip_site": finger.tip_site,
                    "keypoint_links": list(finger.keypoint_links),
                }
                for finger in self.fingers
            ],
            "palm_align": self.palm_align.tolist(),
            "collision_excluded_links": sorted(self.collision_excluded_links),
            "selfcol_links": [list(group) for group in self.selfcol_links],
        }

    def to_json(self) -> str:
        """``to_dict`` as sorted JSON."""
        return json.dumps(self.to_dict(), sort_keys=True)


def body_link_index(model, link_ids: Sequence[int]) -> dict[int, int]:
    """Body id -> index into ``link_ids`` of the link that body belongs to."""
    index = {int(body): i for i, body in enumerate(link_ids)}
    owners: dict[int, int] = {}
    for body in range(model.nbody):
        current = body
        while current > 0 and current not in index and model.body_jntnum[current] == 0:
            current = int(model.body_parentid[current])
        if current in index:
            owners[body] = index[current]
    return owners


def derive_palm_align(model: mujoco.MjModel, root_body: str, fingers: Sequence[FingerSpec], side: str) -> np.ndarray:
    """The MANO anatomical frame -> root frame rotation, from the hand MJCF."""
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    root = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, root_body)
    if root < 0:
        raise ValueError(f"the hand MJCF has no root body {root_body!r}")
    rotation = data.xmat[root].reshape(3, 3)
    origin = data.xpos[root]
    points = np.zeros((MEDIAPIPE_LANDMARK_COUNT, 3))
    for landmark, finger in ((5, fingers[1]), (9, fingers[2])):
        body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, finger.root_link)
        if body < 0:
            raise ValueError(f"the hand MJCF has no body {finger.root_link!r} ({finger.name} root link)")
        points[landmark] = (data.xpos[body] - origin) @ rotation
    frame = estimate_frame_from_hand_points(points)
    return OPERATOR2MANO[side].T @ frame.T


class HandPenetrationContact(NamedTuple):
    """One ``hand_geom x object`` contact row from ``query_hand_penetration``."""

    contact_jacobian: np.ndarray
    phi: float
    geom_id: int
    body_id: int


@runtime_checkable
class HandModelProtocol(Protocol):
    """Forward-kinematics + Jacobian surface every hand backend must expose."""

    nq: int
    nv: int
    q_lb: np.ndarray
    q_ub: np.ndarray

    def forward(self, q: np.ndarray) -> None:
        """Set the configuration ``q`` and run forward kinematics."""

    def get_body_pos(self, body_name: str) -> np.ndarray:
        """The world position of the named body."""

    def get_body_positions(self, body_names: list[str]) -> np.ndarray:
        """The world positions of the named bodies, ``(len(body_names), 3)``."""

    def get_body_pos_by_id(self, bid: int) -> np.ndarray:
        """The world position of a pre-resolved body id."""

    def get_body_jacp_by_id(self, bid: int) -> np.ndarray:
        """The world-frame translational Jacobian, ``(3, nv)``, of a pre-resolved body id."""

    def get_default_qpos(self) -> np.ndarray:
        """The rest configuration, ``(nq,)``."""


@runtime_checkable
class ObjectAwareHandModel(Protocol):
    """Optional capability for backends that host object collision meshes."""

    has_object: bool

    def inject_object_meshes(self, mesh_paths: Sequence[str]) -> None:
        """Rebuild the model with the objects' collision geometry; sets ``has_object``."""

    def set_object_poses(self, positions: np.ndarray, quats_xyzw: np.ndarray) -> None:
        """Place the objects, ``(K, 3)`` positions and ``(K, 4)`` xyzw quaternions."""

    def query_hand_penetration(self, threshold: float) -> list[HandPenetrationContact]:
        """The hand-object pairs within ``threshold`` metres, with signed distance and Jacobian."""
