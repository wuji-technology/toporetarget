"""The floating-wrist playback scene: FK, per-link positions and the contact mask of a solved trajectory."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as RotLib

from toporetarget.config import COLLISION_SIDECAR
from toporetarget.solver.hand import WRIST_DOF, HandSpec, body_link_index
from toporetarget.solver.model import add_object_collision

VIZ_OBJECT_BODY = "viz_object"
VIZ_OBJECT_PREFIX = "viz_obj"


def add_viz_objects(spec, object_mesh_paths: Sequence[str | Path]) -> list[str]:
    """Add one mocap body per object to ``spec``; returns their body names, in mocap order."""
    names = []
    for k, object_mesh_path in enumerate(object_mesh_paths):
        prefix = f"{VIZ_OBJECT_PREFIX}{k}"
        body = spec.worldbody.add_body()
        body.name = f"{VIZ_OBJECT_BODY}{k}"
        body.mocap = True
        geoms = add_object_collision(spec, body, object_mesh_path, prefix=prefix, contype=2, conaffinity=1)
        compound = Path(object_mesh_path).resolve().with_name(COLLISION_SIDECAR).exists()
        for geom in geoms:
            geom.rgba = [0.6, 0.4, 0.2, 0.5]
            geom.group = 3 if compound else 0
        if compound:
            mesh = spec.add_mesh()
            mesh.name, mesh.file = f"{prefix}_visual_mesh", str(Path(object_mesh_path).resolve())
            visual = body.add_geom()
            visual.name = f"{prefix}_visual"
            visual.type = mujoco.mjtGeom.mjGEOM_MESH
            visual.meshname = mesh.name
            visual.contype = visual.conaffinity = 0
            visual.rgba = [0.6, 0.4, 0.2, 0.5]
            visual.group = 0
        names.append(body.name)
    return names


def check_objects(object_mesh_paths: Sequence[str | Path]) -> None:
    """Compile the objects alone, as every solver scene adds them; raises MuJoCo's ``ValueError``."""
    spec = mujoco.MjSpec()
    add_viz_objects(spec, object_mesh_paths)
    spec.compile()


def build_floating_scene(scene_xml: str | Path, wrist_body: str, object_mesh_paths: Sequence[str | Path]):
    """A floating-wrist single-hand scene plus one ``viz_object<k>`` mocap body per object."""
    spec = mujoco.MjSpec.from_file(str(scene_xml))
    spec.body(wrist_body).add_freejoint()
    add_viz_objects(spec, object_mesh_paths)
    model = spec.compile()
    return model, mujoco.MjData(model)


def freejoint_qpos_slice(model) -> slice:
    """The qpos range of the model's single freejoint (7: 3 translation + wxyz)."""
    for j in range(model.njnt):
        if int(model.jnt_type[j]) == int(mujoco.mjtJoint.mjJNT_FREE):
            adr = int(model.jnt_qposadr[j])
            return slice(adr, adr + 7)
    raise RuntimeError("no freejoint in the compiled scene")


def finger_qpos_addrs(model, joint_names) -> list[int]:
    """The qpos address of each named joint, in ``joint_names`` order."""
    out: list[int] = []
    for name in joint_names:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise RuntimeError(f"joint {name!r} not in the scene")
        out.append(int(model.jnt_qposadr[jid]))
    return out


def link_body_ids(model, link_names) -> list[int]:
    """The body id of each named link, in ``link_names`` order."""
    ids: list[int] = []
    for name in link_names:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid < 0:
            raise RuntimeError(f"link body {name!r} not in the scene")
        ids.append(bid)
    return ids


class FloatingSceneFK:
    """One compiled floating-wrist scene plus its indices: build once, reuse per frame."""

    def __init__(
        self,
        hand: HandSpec,
        object_mesh_paths: Sequence[str | Path],
        *,
        contact_margin_m: float = 0.0,
    ) -> None:
        self.joint_names = hand.joint_names
        self.link_names = hand.link_names
        self.model, self.data = build_floating_scene(hand.scene_xml, hand.wrist_body, object_mesh_paths)
        self.wrist_slice = freejoint_qpos_slice(self.model)
        self.finger_addrs = finger_qpos_addrs(self.model, self.joint_names)
        self.link_ids = link_body_ids(self.model, self.link_names)
        self.link_of_body = body_link_index(self.model, self.link_ids)
        self.palm_bid = self.link_ids[0]
        self.object_bids = {b for b in range(self.model.nbody) if self.model.body_mocapid[b] >= 0}
        self.object_geom_ids = [
            g for g in range(self.model.ngeom) if int(self.model.geom_bodyid[g]) in self.object_bids
        ]
        if contact_margin_m > 0.0:
            for g in self.object_geom_ids:
                self.model.geom_margin[g] = contact_margin_m

    def per_link_contact(self) -> np.ndarray:
        """OR-aggregate the hand-object contacts into a per-link mask (a welded body counts for its link)."""
        mask = np.zeros(len(self.link_ids), dtype=bool)
        if not self.object_bids or self.data.ncon <= 0:
            return mask
        gb = self.model.geom_bodyid
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            b1, b2 = int(gb[c.geom[0]]), int(gb[c.geom[1]])
            hand_b = b2 if b1 in self.object_bids else (b1 if b2 in self.object_bids else None)
            link = self.link_of_body.get(hand_b) if hand_b is not None else None
            if link is not None:
                mask[link] = True
        return mask

    def set_frame(self, world_wrist_t, world_wrist_q_wxyz, fingers, object_pose_w) -> None:
        """Place one frame's qpos and the ``(K, 7)`` object mocap poses, then run FK and collision."""
        ws = self.wrist_slice
        self.data.qpos[ws.start : ws.start + 3] = world_wrist_t
        self.data.qpos[ws.start + 3 : ws.stop] = world_wrist_q_wxyz
        for k, a in enumerate(self.finger_addrs):
            self.data.qpos[a] = fingers[k]
        self.data.mocap_pos[:] = object_pose_w[:, :3]
        self.data.mocap_quat[:] = object_pose_w[:, 3:7]
        mujoco.mj_forward(self.model, self.data)
        mujoco.mj_collision(self.model, self.data)

    def palm_pose(self) -> np.ndarray:
        """This frame's FK pose of the hand root as ``[x y z, qw qx qy qz]``."""
        return np.concatenate([self.data.xpos[self.palm_bid], self.data.xquat[self.palm_bid]])

    def link_positions(self) -> np.ndarray:
        return np.asarray(self.data.xpos[self.link_ids], np.float64).reshape(len(self.link_ids), 3)

    def scene_qpos(self) -> np.ndarray:
        """A copy of the full scene qpos as currently posed."""
        return np.asarray(self.data.qpos, dtype=np.float64).copy()

    def scene_joint_names(self) -> list[str]:
        """The non-free joint names of the scene, in qpos-address order."""
        named = []
        for j in range(self.model.njnt):
            if int(self.model.jnt_type[j]) == int(mujoco.mjtJoint.mjJNT_FREE):
                continue
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, j)
            named.append((int(self.model.jnt_qposadr[j]), name))
        return [name for _, name in sorted(named)]

    @staticmethod
    def lift_wrist_to_world(qpos: np.ndarray, wrist_pose_w: np.ndarray):
        """The solver's wrist 6-DoF increment plus the input wrist pose -> world freejoint pose."""
        qpos = np.asarray(qpos, dtype=np.float64)
        wrist_pose_w = np.asarray(wrist_pose_w, dtype=np.float64)
        r_in = RotLib.from_quat(wrist_pose_w[:, [4, 5, 6, 3]]).as_matrix()
        world_t = wrist_pose_w[:, :3] + np.einsum("tij,tj->ti", r_in, qpos[:, :3])
        world_r = r_in @ RotLib.from_euler("XYZ", qpos[:, 3:6]).as_matrix()
        world_q_wxyz = RotLib.from_matrix(world_r).as_quat()[:, [3, 0, 1, 2]]
        return world_t, world_q_wxyz, qpos[:, WRIST_DOF:]

    def materialize(self, qpos, wrist_pose_w, object_pose_w) -> dict[str, np.ndarray]:
        """``qpos (T, nq)`` and ``object_pose_w (T, K, 7)`` -> the per-frame arrays plus the scene qpos."""
        qpos = np.asarray(qpos, dtype=np.float64)
        object_pose_w = np.asarray(object_pose_w, dtype=np.float64)
        n_link = len(self.link_ids)
        world_t, world_q, fingers = self.lift_wrist_to_world(qpos, wrist_pose_w)
        t_n = qpos.shape[0]

        ref = np.empty((t_n, len(self.joint_names)), np.float32)
        wpose = np.empty((t_n, 7), np.float32)
        lpos = np.empty((t_n, n_link, 3), np.float32)
        contact = np.zeros((t_n, n_link), bool)
        scene_qpos = np.empty((t_n, self.model.nq), np.float64)
        for t in range(t_n):
            self.set_frame(world_t[t], world_q[t], fingers[t], object_pose_w[t])
            ref[t] = fingers[t].astype(np.float32)
            wpose[t] = self.palm_pose()
            lpos[t] = self.link_positions().astype(np.float32)
            contact[t] = self.per_link_contact()
            scene_qpos[t] = self.scene_qpos()

        return {
            "reference_qpos": ref,
            "wrist_pose_w": wpose,
            "object_pose_w": object_pose_w.astype(np.float32),
            "link_pos_w": lpos,
            "link_object_contact": contact,
            "scene_qpos": scene_qpos,
            "joint_names": np.array(self.joint_names),
            "link_names": np.array(self.link_names),
        }
