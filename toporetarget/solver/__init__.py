"""The TopoRetarget solver: object-centric interaction-mesh SQP retargeting to a floating robot hand."""

from __future__ import annotations

from toporetarget.solver.clip import (
    OBJECT_MESH_NAME,
    CanonicalClip,
    derive_input_wrist_pose,
    load_canonical_clip,
    resolve_clip_paths,
)
from toporetarget.solver.hand import (
    WRIST_DOF,
    FingerSpec,
    HandSpec,
)
from toporetarget.solver.landmarks import (
    mediapipe_chain_edges,
)
from toporetarget.solver.materialize import (
    add_viz_objects,
    check_objects,
)
from toporetarget.solver.model import (
    add_object_collision,
)
from toporetarget.solver.run import (
    build_hand_config,
    main,
    retarget_hand,
    solve_clip,
)
from toporetarget.solver.session import (
    FrameSolver,
    FrameState,
    HandObjectState,
)
from toporetarget.solver.settings import (
    HandRetargetConfig,
)

__all__ = [
    "CanonicalClip",
    "FingerSpec",
    "FrameSolver",
    "FrameState",
    "HandObjectState",
    "HandRetargetConfig",
    "HandSpec",
    "OBJECT_MESH_NAME",
    "WRIST_DOF",
    "add_object_collision",
    "add_viz_objects",
    "build_hand_config",
    "check_objects",
    "derive_input_wrist_pose",
    "load_canonical_clip",
    "main",
    "mediapipe_chain_edges",
    "resolve_clip_paths",
    "retarget_hand",
    "solve_clip",
]
