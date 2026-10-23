"""Replay TopoRetarget results in the browser, or render them to video.

Examples:
    toporetarget view \
      --result output/grab/wujihand2_right_retarget/<clip>_toporetarget_wujihand2_right.npz --play
    toporetarget view --result-dir output/grab --play
    toporetarget view --result-dir output/grab --video output/grab/videos
    toporetarget view --result-dir output/grab --data data/grab
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from toporetarget.config import ASSETS_DIR, RETARGET_METHOD, diagnostics_path, discover_results, read_npz
from toporetarget.solver import OBJECT_MESH_NAME, add_viz_objects, mediapipe_chain_edges

LOG = "Viewer"
DEFAULT_CONTACT_RGBA = (0.95, 0.35, 0.15, 0.9)
DEFAULT_LANDMARK_RGBA = (0.2, 0.7, 1.0, 0.8)
CAMERA_FOLLOW = 0.1
FLOOR_CLEARANCE = 0.02
VISIBLE_GEOM_GROUPS = (0, 1)
RESULT_INFIX = f"_{RETARGET_METHOD}_"


@dataclass
class HandResult:
    """One hand's result plus its diagnostics sidecar."""

    path: Path
    qpos: np.ndarray
    fps: float
    frame_ids: np.ndarray
    scene_xml: Path
    robot_name: str
    wrist_body: str
    side: str
    joint_names: list[str]
    object_meshes: list[Path]
    object_pose_w: np.ndarray | None
    landmarks_w: np.ndarray | None
    link_pos_w: np.ndarray | None
    link_contact: np.ndarray | None
    tip_error_mm: np.ndarray | None
    penetration_mm: np.ndarray | None


@dataclass
class ClipView:
    """The results of one clip, one per solved hand, on one timeline."""

    clip_id: str
    hands: list[HandResult]
    object_meshes: list[Path]
    object_pose_w: np.ndarray
    fps: float

    @property
    def n_frames(self) -> int:
        return int(len(self.hands[0].qpos))


def _scalar(data, key: str, default: Any = None) -> Any:
    if key not in data:
        return default
    value = np.asarray(data[key])
    return value.item() if value.shape == () or value.size == 1 else value


def clip_id_of(result_path: Path) -> str:
    """The clip id a result file is named after: ``<clip>_toporetarget_<robot>.npz``."""
    return Path(result_path).stem.rsplit(RESULT_INFIX, 1)[0]


def dataset_roots(dataset_name: str) -> list[Path]:
    """Where a dataset of that name sits under the working directory: ``data/<name>``."""
    return [Path.cwd() / "data" / dataset_name]


def load_hand_result(result_path: str | Path, data_root: Path | None = None) -> HandResult:
    """Load one result ``.npz`` and its diagnostics sidecar."""
    result_path = Path(result_path).resolve()
    data = read_npz(result_path)
    qpos = np.asarray(data["qpos"], dtype=np.float64)
    fps = float(np.asarray(data["fps"]).reshape(-1)[0])
    frame_ids = np.asarray(data["frame_ids"])
    scene_xml = Path(str(_scalar(data, "robot_xml")))
    robot_name = str(_scalar(data, "robot_name", scene_xml.stem))
    wrist_body = str(_scalar(data, "robot_wrist_body", ""))
    joint_names = [str(name) for name in np.asarray(data["robot_joint_names"])]
    object_dirs = [Path(str(d)) for d in np.asarray(data["source_object_dirs"]).reshape(-1)]
    source_data = Path(str(_scalar(data, "source_data", "")))
    dataset_name = str(_scalar(data, "source_dataset", source_data.parent.parent.name))

    diag: dict[str, np.ndarray] = {}
    side = ""
    sidecar = diagnostics_path(result_path)
    if sidecar.exists():
        sidecar_data = read_npz(sidecar)
        side = str(_scalar(sidecar_data, "hand_side", ""))
        for key in (
            "object_pose_w",
            "mediapipe_landmarks_w",
            "link_pos_w",
            "link_object_contact",
            "tip_error_m",
            "hand_object_penetration_mm",
        ):
            if key in sidecar_data:
                diag[key] = np.asarray(sidecar_data[key])

    if data_root is None and not all((d / OBJECT_MESH_NAME).is_file() for d in object_dirs):
        data_root = next((root for root in dataset_roots(dataset_name) if (root / "motions").is_dir()), None)
    if data_root is not None:
        object_dirs = [Path(data_root) / "assets" / d.name for d in object_dirs]
    object_meshes = [d / OBJECT_MESH_NAME for d in object_dirs]
    for object_mesh in object_meshes:
        if not object_mesh.is_file():
            raise FileNotFoundError(
                f"object mesh not found: {object_mesh} (solved on another machine? pass --data <dataset>)"
            )
    if not scene_xml.is_absolute():
        scene_xml = next(
            (root / scene_xml for root in (ASSETS_DIR, Path.cwd()) if (root / scene_xml).is_file()),
            ASSETS_DIR / scene_xml,
        )
    if not scene_xml.is_file():
        scene_xml = ASSETS_DIR / "scenes" / scene_xml.name
    if not scene_xml.is_file():
        raise FileNotFoundError(f"scene XML not found: {scene_xml}")
    if qpos.shape[1] != 7 + len(joint_names):
        raise ValueError(f"{result_path}: qpos width {qpos.shape[1]} != 7 + {len(joint_names)} joints")
    if not wrist_body:
        wrist_body = scene_wrist_body(scene_xml)

    def as_f64(key: str) -> np.ndarray | None:
        return diag[key].astype(np.float64) if key in diag else None

    tip_error_m = as_f64("tip_error_m")
    return HandResult(
        path=result_path,
        qpos=qpos,
        fps=fps,
        frame_ids=frame_ids,
        scene_xml=scene_xml,
        robot_name=robot_name,
        wrist_body=wrist_body,
        side=side,
        joint_names=joint_names,
        object_meshes=object_meshes,
        object_pose_w=as_f64("object_pose_w"),
        landmarks_w=as_f64("mediapipe_landmarks_w"),
        link_pos_w=as_f64("link_pos_w"),
        link_contact=diag.get("link_object_contact"),
        tip_error_mm=None if tip_error_m is None else 1000.0 * tip_error_m.mean(axis=1),
        penetration_mm=as_f64("hand_object_penetration_mm"),
    )


def load_clip(result_paths: list[Path], data_root: Path | None = None) -> ClipView:
    """The results of one clip on one timeline; they have to cover the same frames."""
    hands = [load_hand_result(path, data_root) for path in result_paths]
    first = hands[0]
    for hand in hands[1:]:
        if not np.array_equal(hand.frame_ids, first.frame_ids):
            raise ValueError(f"{hand.path.name} and {first.path.name} cover different frames of the clip")
    if first.object_pose_w is None:
        raise ValueError(f"{first.path}: no diagnostics sidecar with the object poses")
    return ClipView(
        clip_id=clip_id_of(first.path),
        hands=hands,
        object_meshes=first.object_meshes,
        object_pose_w=first.object_pose_w,
        fps=first.fps,
    )


def group_results(paths: list[Path]) -> dict[str, list[Path]]:
    """Results by clip id, in clip order; within a clip, right hand first."""
    clips: dict[str, list[Path]] = {}
    for path in sorted(paths, key=lambda p: (clip_id_of(p), "left" in p.stem.rsplit(RESULT_INFIX, 1)[-1])):
        clips.setdefault(clip_id_of(path), []).append(path)
    return clips


def batch_result_dirs(path: Path) -> list[Path]:
    """The ``<robot>_retarget`` directories of the batch ``path`` is, or belongs to."""
    path = Path(path).resolve()
    root = path.parent if path.name.endswith("_retarget") else path
    dirs = sorted(d for d in root.glob("*_retarget") if d.is_dir())
    return dirs or [path]


def clips_for_results(result_paths: list[Path]) -> dict[str, list[Path]]:
    """The clips of the given results, each with its other hands' results from the same batch."""
    found: list[Path] = []
    for path in result_paths:
        path = Path(path).resolve()
        for directory in batch_result_dirs(path.parent):
            found.extend(directory.glob(f"{clip_id_of(path)}{RESULT_INFIX}*.npz"))
        found.append(path)
    return group_results([p for p in set(found) if not p.stem.endswith("_diagnostics")])


def describe(clip: ClipView) -> str:
    lines = [
        f"[{LOG}] {clip.clip_id}: {clip.n_frames} frames at {clip.fps:g} fps, "
        f"objects {', '.join(path.parent.name for path in clip.object_meshes)}"
    ]
    lines += [f"[{LOG}]   {hand.side or '?'} hand: {hand.robot_name} ({hand.path.name})" for hand in clip.hands]
    return "\n".join(lines)


@dataclass
class Playback:
    """One compiled scene of a clip: every hand and every object."""

    model: Any
    data: Any
    qpos_addrs: list[np.ndarray]
    hand_roots: list[int]
    object_bodies: list[int]


def scene_wrist_body(scene_xml: Path) -> str:
    """The body of a single-hand scene that carries the hand: the one body under its worldbody."""
    bodies = [body.name for body in mujoco.MjSpec.from_file(str(scene_xml)).worldbody.bodies]
    if len(bodies) != 1:
        raise ValueError(f"{scene_xml}: expected one body under the worldbody to carry the hand, found {bodies}")
    return bodies[0]


def playback_model(clip: ClipView, show_object: bool = True) -> Playback:
    """The clip's hand scenes side by side plus one mocap body per object."""
    spec = mujoco.MjSpec.from_file(str(clip.hands[0].scene_xml))
    spec.body(clip.hands[0].wrist_body).add_freejoint()
    prefixes = [""]
    for index, hand in enumerate(clip.hands[1:], start=1):
        child = mujoco.MjSpec.from_file(str(hand.scene_xml))
        child.body(hand.wrist_body).add_freejoint()
        prefixes.append(f"hand{index}_")
        spec.worldbody.add_frame().attach_body(child.body(hand.wrist_body), prefixes[-1], "")
    add_viz_objects(spec, clip.object_meshes)
    model = spec.compile()

    qpos_addrs, hand_roots = [], []
    for hand, prefix in zip(clip.hands, prefixes, strict=True):
        root = model.body(prefix + hand.wrist_body).id
        free = next(
            j
            for j in range(model.njnt)
            if model.jnt_bodyid[j] == root and model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE
        )
        start = int(model.jnt_qposadr[free])
        fingers = [int(model.jnt_qposadr[model.joint(prefix + name).id]) for name in hand.joint_names]
        qpos_addrs.append(np.array([*range(start, start + 7), *fingers]))
        hand_roots.append(root)
    object_bodies = [b for b in range(model.nbody) if model.body_mocapid[b] >= 0]
    object_geoms = [g for g in range(model.ngeom) if int(model.geom_bodyid[g]) in object_bodies]
    if not show_object:
        model.geom_rgba[object_geoms, 3] = 0.0

    floor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    if floor >= 0:
        lowest = min(
            (hand.link_pos_w[..., 2] if hand.link_pos_w is not None else hand.qpos[:, 2]).min() for hand in clip.hands
        )
        if clip.object_pose_w.size:
            lowest = min(lowest, clip.object_pose_w[..., 2].min() - model.geom_rbound[object_geoms].max())
        model.geom_pos[floor, 2] = lowest - FLOOR_CLEARANCE
    return Playback(model, mujoco.MjData(model), qpos_addrs, hand_roots, object_bodies)


def camera_targets(clip: ClipView) -> np.ndarray:
    """``(T, 3)`` the point the camera follows: the centre of the palms and the objects."""
    points = [hand.qpos[:, None, :3] for hand in clip.hands] + [clip.object_pose_w[..., :3]]
    return np.concatenate(points, axis=1).mean(axis=1)


def set_frame(playback: Playback, clip: ClipView, frame: int) -> None:
    for hand, addrs in zip(clip.hands, playback.qpos_addrs, strict=True):
        playback.data.qpos[addrs] = hand.qpos[frame]
    playback.data.mocap_pos[:] = clip.object_pose_w[frame, :, :3]
    playback.data.mocap_quat[:] = clip.object_pose_w[frame, :, 3:7]
    mujoco.mj_forward(playback.model, playback.data)


def contact_points(hand: HandResult, frame: int) -> np.ndarray:
    """``(n, 3)`` the links of one hand touching an object on one frame."""
    if hand.link_contact is None or hand.link_pos_w is None or frame >= len(hand.link_contact):
        return np.zeros((0, 3))
    return hand.link_pos_w[frame][np.asarray(hand.link_contact[frame], dtype=bool)]


def _geom_surface(model, geom: int):
    """One geom's surface in its own frame as a trimesh, or None for a plane."""
    import trimesh

    kind = int(model.geom_type[geom])
    size = model.geom_size[geom]
    if kind == mujoco.mjtGeom.mjGEOM_MESH:
        mesh = int(model.geom_dataid[geom])
        v0, nv = int(model.mesh_vertadr[mesh]), int(model.mesh_vertnum[mesh])
        f0, nf = int(model.mesh_faceadr[mesh]), int(model.mesh_facenum[mesh])
        return trimesh.Trimesh(model.mesh_vert[v0 : v0 + nv], model.mesh_face[f0 : f0 + nf], process=False)
    if kind == mujoco.mjtGeom.mjGEOM_BOX:
        return trimesh.creation.box(extents=2.0 * size)
    if kind == mujoco.mjtGeom.mjGEOM_SPHERE:
        return trimesh.creation.icosphere(subdivisions=2, radius=size[0])
    if kind == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
        surface = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
        surface.apply_scale(size)
        return surface
    if kind == mujoco.mjtGeom.mjGEOM_CAPSULE:
        return trimesh.creation.capsule(height=2.0 * size[1], radius=size[0])
    if kind == mujoco.mjtGeom.mjGEOM_CYLINDER:
        return trimesh.creation.cylinder(radius=size[0], height=2.0 * size[1])
    return None


def _rgb255(rgba) -> tuple[int, int, int]:
    return tuple(int(round(255 * c)) for c in rgba[:3])


@dataclass
class _HandOverlay:
    """The viser handles of one hand's source keypoints and contacting links."""

    keypoints: Any = None
    bones: Any = None
    contacts: Any = None


class BrowserViewer:
    """A viser server replaying one clip at a time, picked from ``clips``."""

    def __init__(self, clips: dict[str, list[Path]], args):
        import viser

        self.clips = clips
        self.clip_names = list(clips)
        self.args = args
        self.server = viser.ViserServer(host=args.host, port=args.port)
        self.server.scene.set_up_direction("+z")
        self.edges = mediapipe_chain_edges()
        self.clip_view: ClipView | None = None
        self.shown_frame = -1
        self.focus = np.zeros(3)
        self._pending: int | None = 0
        self._camera_placed = False
        self.redraw = False
        self._build_gui()

    def _build_gui(self) -> None:
        gui = self.server.gui
        names = self.clip_names
        with gui.add_folder("Clip"):
            self.clip = gui.add_dropdown("Clip", options=names, initial_value=names[0], visible=len(names) > 1)
            self.prev_button = gui.add_button("Previous", visible=len(names) > 1)
            self.next_button = gui.add_button("Next", visible=len(names) > 1)
            self.info = gui.add_markdown("")
        with gui.add_folder("Playback"):
            self.playing = gui.add_checkbox("Playing", initial_value=bool(self.args.play))
            self.frame = gui.add_slider("Frame", min=0, max=1, step=1, initial_value=0)
            self.speed = gui.add_slider("Speed", min=0.1, max=2.0, step=0.1, initial_value=1.0)
            self.loop = gui.add_checkbox("Loop", initial_value=bool(self.args.loop))
            self.follow = gui.add_checkbox("Follow the hands", initial_value=True)
            self.frame_info = gui.add_markdown("")
        with gui.add_folder("Show"):
            self.show_robot = gui.add_checkbox("Robot hands", initial_value=True)
            self.show_object = gui.add_checkbox("Objects", initial_value=bool(self.args.show_object))
            self.show_landmarks = gui.add_checkbox(
                "Source keypoints", initial_value=bool(self.args.show_source_landmarks)
            )
            self.show_contacts = gui.add_checkbox("Contact links", initial_value=bool(self.args.show_contact_links))

        self.clip.on_update(lambda _: self._request(names.index(self.clip.value)))
        self.prev_button.on_click(lambda _: self._step_clip(-1))
        self.next_button.on_click(lambda _: self._step_clip(+1))
        for toggle in (self.show_robot, self.show_object, self.show_landmarks, self.show_contacts):
            toggle.on_update(lambda _: self._apply_visibility())

    def _request(self, index: int) -> None:
        self._pending = index

    def _step_clip(self, step: int) -> None:
        index = (self.clip_names.index(self.clip.value) + step) % len(self.clip_names)
        self.clip.value = self.clip_names[index]

    def _load(self, index: int) -> None:
        clip = load_clip(self.clips[self.clip_names[index]], self.args.data)
        print(describe(clip))
        playback = playback_model(clip)
        set_frame(playback, clip, 0)
        model, data = playback.model, playback.data
        scene = self.server.scene
        scene.reset()
        targets = camera_targets(clip)
        self.focus = targets[0].copy()
        self.root = scene.add_frame("/world", show_axes=False, position=-self.focus)

        floor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        floor_z = float(model.geom_pos[floor, 2]) if floor >= 0 else float(targets[:, 2].min()) - 0.1
        scene.add_grid(
            "/world/floor",
            cell_size=0.05,
            section_size=0.25,
            infinite_grid=True,
            fade_distance=3.0,
            position=(float(self.focus[0]), float(self.focus[1]), floor_z),
        )

        self.bodies: list[tuple[int, Any]] = []
        self.robot_frames: list[Any] = []
        self.object_frames: list[Any] = []
        for body in range(1, model.nbody):
            geoms = [
                g
                for g in range(model.ngeom)
                if int(model.geom_bodyid[g]) == body
                and int(model.geom_group[g]) in VISIBLE_GEOM_GROUPS
                and model.geom_rgba[g, 3] > 0.0
            ]
            surfaces = [(g, _geom_surface(model, g)) for g in geoms]
            surfaces = [(g, s) for g, s in surfaces if s is not None]
            if not surfaces:
                continue
            name = f"/world/body_{body}"
            handle = scene.add_frame(
                name, show_axes=False, position=data.xpos[body].copy(), wxyz=data.xquat[body].copy()
            )
            for g, surface in surfaces:
                alpha = float(model.geom_rgba[g, 3])
                scene.add_mesh_simple(
                    f"{name}/geom_{g}",
                    vertices=np.asarray(surface.vertices, dtype=np.float32),
                    faces=np.asarray(surface.faces, dtype=np.uint32),
                    color=_rgb255(model.geom_rgba[g]),
                    opacity=alpha if alpha < 1.0 else None,
                    position=model.geom_pos[g].copy(),
                    wxyz=model.geom_quat[g].copy(),
                )
            self.bodies.append((body, handle))
            if body in playback.object_bodies:
                self.object_frames.append(handle)
            elif int(model.body_rootid[body]) in playback.hand_roots:
                self.robot_frames.append(handle)

        self.overlays: list[_HandOverlay] = []
        for index, hand in enumerate(clip.hands):
            overlay = _HandOverlay()
            if hand.landmarks_w is not None:
                points = hand.landmarks_w[0].astype(np.float32)
                overlay.keypoints = scene.add_point_cloud(
                    f"/world/source/{index}/keypoints",
                    points=points,
                    colors=_rgb255(DEFAULT_LANDMARK_RGBA),
                    point_size=0.008,
                    point_shape="circle",
                    precision="float32",
                )
                overlay.bones = scene.add_line_segments(
                    f"/world/source/{index}/bones",
                    points=points[self.edges],
                    colors=_rgb255(DEFAULT_LANDMARK_RGBA),
                    thickness=0.0015,
                )
            if hand.link_contact is not None and hand.link_pos_w is not None:
                overlay.contacts = scene.add_point_cloud(
                    f"/world/contacts/{index}",
                    points=np.zeros((1, 3), dtype=np.float32),
                    colors=_rgb255(DEFAULT_CONTACT_RGBA),
                    point_size=0.012,
                    point_shape="circle",
                    precision="float32",
                    visible=False,
                )
            self.overlays.append(overlay)

        self.playback, self.clip_view, self.targets = playback, clip, targets
        self.frame.max = max(clip.n_frames - 1, 1)
        self.frame.value = 0
        self.shown_frame = -1
        self.info.content = self._clip_summary(clip)
        self._apply_visibility()
        if not self._camera_placed:
            self._initial_camera()
            self._camera_placed = True

    def _clip_summary(self, clip: ClipView) -> str:
        lines = [
            f"**{clip.clip_id}**",
            f"{clip.n_frames} frames at {clip.fps:g} Hz",
            f"objects: {', '.join(path.parent.name for path in clip.object_meshes)}",
        ]
        for hand in clip.hands:
            line = f"{hand.side} hand ({hand.robot_name})"
            if hand.tip_error_mm is not None:
                line += (
                    f": fingertip error mean {hand.tip_error_mm.mean():.1f} mm, worst {hand.tip_error_mm.max():.1f} mm"
                )
            if hand.penetration_mm is not None:
                line += f", penetration max {hand.penetration_mm.max():.2f} mm"
            lines.append(line)
        return "  \n".join(lines)

    def _apply_visibility(self) -> None:
        if self.clip_view is None:
            return
        for handle in self.robot_frames:
            handle.visible = bool(self.show_robot.value)
        for handle in self.object_frames:
            handle.visible = bool(self.show_object.value)
        for overlay in self.overlays:
            for handle in (overlay.keypoints, overlay.bones):
                if handle is not None:
                    handle.visible = bool(self.show_landmarks.value)
        self.redraw = True

    def _render(self, frame: int) -> None:
        clip, playback = self.clip_view, self.playback
        set_frame(playback, clip, frame)
        if self.follow.value and frame != self.shown_frame:
            if self.playing.value and 0 < frame - self.shown_frame <= 10:
                self.focus += CAMERA_FOLLOW * (self.targets[frame] - self.focus)
            else:
                self.focus = self.targets[frame].copy()
        with self.server.atomic():
            self.root.position = -self.focus
            for body, handle in self.bodies:
                handle.position = playback.data.xpos[body].copy()
                handle.wxyz = playback.data.xquat[body].copy()
            for hand, overlay in zip(clip.hands, self.overlays, strict=True):
                if overlay.keypoints is not None and frame < len(hand.landmarks_w):
                    points = hand.landmarks_w[frame].astype(np.float32)
                    overlay.keypoints.points = points
                    overlay.bones.points = points[self.edges]
                if overlay.contacts is not None:
                    touching = contact_points(hand, frame)
                    overlay.contacts.visible = bool(self.show_contacts.value) and len(touching) > 0
                    if len(touching):
                        overlay.contacts.points = touching.astype(np.float32)
                        overlay.contacts.colors = np.tile(np.uint8(_rgb255(DEFAULT_CONTACT_RGBA)), (len(touching), 1))
        readout = [f"frame {frame} / {clip.n_frames - 1}, t = {frame / clip.fps:.2f} s"]
        for hand in clip.hands:
            parts = []
            if hand.tip_error_mm is not None:
                parts.append(f"fingertip error {hand.tip_error_mm[frame]:.1f} mm")
            if hand.penetration_mm is not None:
                parts.append(f"penetration {hand.penetration_mm[frame]:.2f} mm")
            if parts:
                readout.append(f"{hand.side}: " + ", ".join(parts))
        self.frame_info.content = "  \n".join(readout)
        self.shown_frame = frame

    def _initial_camera(self) -> None:
        """Place new clients' camera like the scene's default MuJoCo view."""
        model = self.playback.model
        azimuth = np.deg2rad(float(model.vis.global_.azimuth))
        elevation = np.deg2rad(float(model.vis.global_.elevation))
        forward = np.array(
            [np.cos(elevation) * np.cos(azimuth), np.cos(elevation) * np.sin(azimuth), np.sin(elevation)]
        )
        self.server.initial_camera.position = tuple(-self.args.camera_distance * forward)
        self.server.initial_camera.look_at = (0.0, 0.0, 0.0)

    def run(self) -> None:
        clock = time.perf_counter()
        advance = 0.0
        print(f"[{LOG}] open http://localhost:{self.server.get_port()} in a browser (Ctrl+C to quit)")
        try:
            while True:
                if self._pending is not None:
                    index, self._pending = self._pending, None
                    try:
                        self._load(index)
                    except (OSError, ValueError) as exc:
                        print(f"[{LOG}] {self.clip_names[index]}: {exc}")
                        self.info.content = f"**{self.clip_names[index]}** could not be loaded:  \n{exc}"
                now = time.perf_counter()
                elapsed, clock = now - clock, now
                clip = self.clip_view
                if clip is None:
                    time.sleep(0.05)
                    continue
                if self.playing.value:
                    advance += elapsed * clip.fps * float(self.speed.value)
                    step, advance = int(advance), advance - int(advance)
                    if step:
                        frame = int(self.frame.value) + step
                        if frame >= clip.n_frames:
                            if self.loop.value:
                                frame %= clip.n_frames
                            else:
                                frame = clip.n_frames - 1
                                self.playing.value = False
                        self.frame.value = frame
                else:
                    advance = 0.0
                frame = min(int(self.frame.value), clip.n_frames - 1)
                if frame != self.shown_frame or self.redraw:
                    self.redraw = False
                    self._render(frame)
                time.sleep(max(0.0, 1.0 / clip.fps - (time.perf_counter() - now)))
        except KeyboardInterrupt:
            pass
        finally:
            self.server.stop()


def _add_sphere(scene, position: np.ndarray, radius: float, rgba: tuple[float, float, float, float]) -> None:
    """Append one debug sphere to a MuJoCo scene, if it still has room."""
    if scene.ngeom >= scene.maxgeom:
        return
    mujoco.mjv_initGeom(
        scene.geoms[scene.ngeom],
        int(mujoco.mjtGeom.mjGEOM_SPHERE),
        np.array([radius, 0.0, 0.0]),
        np.asarray(position, dtype=np.float64),
        np.eye(3).reshape(-1),
        np.asarray(rgba, dtype=np.float32),
    )
    scene.ngeom += 1


def draw_overlays(scene, clip: ClipView, frame: int, *, show_landmarks: bool, show_contacts: bool) -> None:
    """Append every hand's captured keypoints and contacting links of one frame to a MuJoCo scene."""
    for hand in clip.hands:
        if show_landmarks and hand.landmarks_w is not None and frame < len(hand.landmarks_w):
            for point in hand.landmarks_w[frame]:
                _add_sphere(scene, point, 0.004, DEFAULT_LANDMARK_RGBA)
        if show_contacts:
            for point in contact_points(hand, frame):
                _add_sphere(scene, point, 0.006, DEFAULT_CONTACT_RGBA)


def play_mujoco(clip: ClipView, args) -> None:
    """Open the native MuJoCo viewer on one clip and step through its frames."""
    import mujoco.viewer

    playback = playback_model(clip, args.show_object)
    targets = camera_targets(clip)
    dt = 1.0 / max(clip.fps, 1e-6)
    frame = 0
    playing = bool(args.play)
    with mujoco.viewer.launch_passive(playback.model, playback.data) as viewer:
        with viewer.lock():
            viewer.cam.lookat[:] = targets[0]
            viewer.cam.distance = args.camera_distance
            viewer.opt.geomgroup[:] = [int(group in VISIBLE_GEOM_GROUPS) for group in range(len(viewer.opt.geomgroup))]
        while viewer.is_running():
            start = time.perf_counter()
            set_frame(playback, clip, frame)
            with viewer.lock():
                viewer.cam.lookat[:] += CAMERA_FOLLOW * (targets[frame] - viewer.cam.lookat)
            viewer.user_scn.ngeom = 0
            draw_overlays(
                viewer.user_scn,
                clip,
                frame,
                show_landmarks=args.show_source_landmarks,
                show_contacts=args.show_contact_links,
            )
            viewer.sync()

            if playing:
                frame += 1
                if frame >= clip.n_frames:
                    if not args.loop:
                        playing = False
                        frame = clip.n_frames - 1
                    else:
                        frame = 0
            remaining = dt - (time.perf_counter() - start)
            if remaining > 0:
                time.sleep(remaining)


def render_video(clip: ClipView, args, path: Path) -> None:
    """Render one clip offscreen to an mp4, the camera following the hand-object centre."""
    import imageio.v2 as imageio

    playback = playback_model(clip, args.show_object)
    model = playback.model
    target = camera_targets(clip)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = target[0]
    camera.distance = args.camera_distance
    camera.azimuth = model.vis.global_.azimuth
    camera.elevation = model.vis.global_.elevation
    option = mujoco.MjvOption()
    option.geomgroup[:] = [int(group in VISIBLE_GEOM_GROUPS) for group in range(len(option.geomgroup))]
    path.parent.mkdir(parents=True, exist_ok=True)
    with (
        mujoco.Renderer(model, height=args.height, width=args.width) as renderer,
        imageio.get_writer(path, fps=clip.fps, macro_block_size=1) as writer,
    ):
        for frame in range(clip.n_frames):
            set_frame(playback, clip, frame)
            camera.lookat[:] += CAMERA_FOLLOW * (target[frame] - camera.lookat)
            renderer.update_scene(playback.data, camera, scene_option=option)
            draw_overlays(
                renderer.scene,
                clip,
                frame,
                show_landmarks=args.show_source_landmarks,
                show_contacts=args.show_contact_links,
            )
            writer.append_data(renderer.render())
    print(f"[{LOG}] wrote {path}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--result", type=Path, nargs="+", default=None, help="Result .npz files of the clips to show.")
    parser.add_argument(
        "--result-dir", type=Path, default=None, help="A batch output, or one of its <robot>_retarget directories."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=None,
        help="Dataset the results were solved from (for results from another machine).",
    )
    parser.add_argument(
        "--viewer",
        choices=("viser", "mujoco"),
        default="viser",
        help="viser: a web viewer to open in a browser (default); mujoco: the native MuJoCo window.",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Address the web viewer listens on (default: this machine only; 0.0.0.0 opens it to the network).",
    )
    parser.add_argument("--port", type=int, default=8080, help="Port of the web viewer.")
    parser.add_argument("--play", action="store_true", help="Start playing instead of holding the first frame.")
    parser.add_argument("--loop", action="store_true", default=True, help="Restart at the end of the clip.")
    parser.add_argument("--no-loop", dest="loop", action="store_false")
    parser.add_argument("--show-object", action="store_true", default=True)
    parser.add_argument("--no-show-object", dest="show_object", action="store_false")
    parser.add_argument("--show-source-landmarks", action="store_true", default=True)
    parser.add_argument("--no-show-source-landmarks", dest="show_source_landmarks", action="store_false")
    parser.add_argument("--show-contact-links", action="store_true", default=True)
    parser.add_argument("--no-show-contact-links", dest="show_contact_links", action="store_false")
    parser.add_argument("--camera-distance", type=float, default=0.6)
    parser.add_argument(
        "--video",
        type=Path,
        default=None,
        help="Render to this .mp4 (a directory for several clips) instead of opening a viewer.",
    )
    parser.add_argument("--width", type=int, default=960, help="Video width in pixels.")
    parser.add_argument("--height", type=int, default=720, help="Video height in pixels.")
    parser.add_argument("--dry-run", action="store_true", help="Load and report, without opening a viewer.")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.result is None and args.result_dir is None:
        raise SystemExit("pass --result or --result-dir")

    if args.result is not None:
        clips = clips_for_results(args.result)
    else:
        clips = group_results([p for d in batch_result_dirs(args.result_dir) for p in discover_results(d)])
    if not clips:
        raise SystemExit(f"no results under {args.result_dir}")

    if args.dry_run:
        for paths in clips.values():
            print(describe(load_clip(paths, args.data)))
        return
    if args.video is None and args.viewer == "viser":
        BrowserViewer(clips, args).run()
        return
    for clip_id, paths in clips.items():
        clip = load_clip(paths, args.data)
        print(describe(clip))
        if args.video is None:
            play_mujoco(clip, args)
        elif len(clips) == 1 and args.video.suffix == ".mp4":
            render_video(clip, args, args.video)
        else:
            render_video(clip, args, args.video / f"{clip_id}.mp4")


if __name__ == "__main__":
    main()
