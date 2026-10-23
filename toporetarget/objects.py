"""Prepare an object asset, optionally with a compound collision sidecar.

Examples:
    toporetarget prepare-object --asset data/pen_spin/assets/pen250
    toporetarget prepare-object --asset <dir> --pen --shaft-radius 0.004
    toporetarget prepare-object --asset <dir> --convex
    toporetarget prepare-object --asset <dir> --check
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from toporetarget.config import COLLISION_SIDECAR

LOG = "Asset"
MESH_NAME = "mesh.stl"
META_NAME = "asset.json"
CONVEX_PART_DIR = "collision"
IDENTITY_QUAT = [1.0, 0.0, 0.0, 0.0]

_AXIS_TO_Z_QUAT = {
    0: [0.70710678, 0.0, 0.70710678, 0.0],
    1: [0.70710678, -0.70710678, 0.0, 0.0],
    2: IDENTITY_QUAT,
}


def load_mesh(mesh_path: Path):
    import trimesh

    mesh = trimesh.load(mesh_path, force="mesh")
    if not isinstance(mesh, trimesh.Trimesh) or not mesh.vertices.size:
        raise ValueError(f"empty or unreadable mesh: {mesh_path}")
    return mesh


def describe_mesh(mesh) -> dict[str, Any]:
    lo, hi = mesh.bounds
    return {
        "vertices": int(len(mesh.vertices)),
        "faces": int(len(mesh.faces)),
        "extents_m": [float(v) for v in mesh.extents],
        "bounds_m": [[float(v) for v in lo], [float(v) for v in hi]],
        "long_axis": int(np.argmax(mesh.extents)),
        "hull_extents_m": [float(v) for v in mesh.convex_hull.extents],
    }


_SHAFT_SAMPLE_FRACTION = 0.5


def fit_pen_sidecar(mesh, *, shaft_radius: float | None, cap_fraction: float) -> dict[str, Any]:
    """Fit a shaft cylinder plus two end-cap boxes to a slender mesh."""
    extents = np.asarray(mesh.extents, dtype=float)
    axis = int(np.argmax(extents))
    perp = [k for k in range(3) if k != axis]
    lo, hi = np.asarray(mesh.bounds[0], float), np.asarray(mesh.bounds[1], float)
    center = (lo + hi) / 2.0
    half_len = float(extents[axis] / 2.0)

    if not 0.0 < cap_fraction < 0.5:
        raise ValueError(f"--cap-fraction must be in (0, 0.5), got {cap_fraction}")
    cap_half = half_len * cap_fraction

    if shaft_radius is None:
        verts = np.asarray(mesh.vertices, dtype=float)
        middle = verts[np.abs(verts[:, axis] - center[axis]) < half_len * _SHAFT_SAMPLE_FRACTION]
        if len(middle) < 8:
            raise SystemExit(
                f"cannot infer the shaft radius: only {len(middle)} vertices lie within the middle "
                f"{_SHAFT_SAMPLE_FRACTION:.0%} of the long axis. Pass --shaft-radius explicitly."
            )
        radial = np.linalg.norm(middle[:, perp] - center[perp], axis=1)
        shaft_radius = float(np.percentile(radial, 95))
    cap_half_extents = [float(extents[k] / 2.0) for k in perp]

    def cap_pos(sign: int) -> list[float]:
        pos = center.copy()
        pos[axis] = center[axis] + sign * (half_len - cap_half)
        return [float(v) for v in pos]

    cap_size = [0.0, 0.0, 0.0]
    cap_size[perp[0]], cap_size[perp[1]], cap_size[axis] = (
        cap_half_extents[0],
        cap_half_extents[1],
        cap_half,
    )

    return {
        "geoms": [
            {
                "type": "cylinder",
                "size": [round(float(shaft_radius), 6), round(half_len, 6), 0.0],
                "pos": [round(float(v), 6) for v in center],
                "quat": _AXIS_TO_Z_QUAT[axis],
            },
            {
                "type": "box",
                "size": [round(v, 6) for v in cap_size],
                "pos": [round(v, 6) for v in cap_pos(+1)],
                "quat": IDENTITY_QUAT,
            },
            {
                "type": "box",
                "size": [round(v, 6) for v in cap_size],
                "pos": [round(v, 6) for v in cap_pos(-1)],
                "quat": IDENTITY_QUAT,
            },
        ],
    }


def mujoco_reads_mesh(mesh_path: Path) -> bool:
    """Whether MuJoCo compiles ``mesh_path`` as the mesh of a body."""
    import mujoco

    spec = mujoco.MjSpec()
    mesh = spec.add_mesh()
    mesh.name, mesh.file = "part", str(mesh_path)
    geom = spec.worldbody.add_body().add_geom()
    geom.type, geom.meshname = mujoco.mjtGeom.mjGEOM_MESH, "part"
    try:
        spec.compile()
    except ValueError:
        return False
    return True


def fit_convex_sidecar(mesh, asset_dir: Path, *, threshold: float, max_parts: int) -> dict[str, Any] | None:
    """Decompose ``mesh`` with CoACD and write its convex parts under ``asset_dir/collision``.

    A part MuJoCo does not compile, a sliver too thin to have a volume, is left out.
    """
    import coacd
    import trimesh

    coacd.set_log_level("error")
    parts = coacd.run_coacd(
        coacd.Mesh(np.asarray(mesh.vertices, dtype=np.float64), np.asarray(mesh.faces, dtype=np.int64)),
        threshold=threshold,
        max_convex_hull=max_parts,
        seed=0,
    )
    part_dir = asset_dir / CONVEX_PART_DIR
    shutil.rmtree(part_dir, ignore_errors=True)
    part_dir.mkdir(parents=True)
    geoms = []
    for vertices, faces in parts:
        path = part_dir / f"part_{len(geoms):02d}.stl"
        trimesh.Trimesh(vertices, faces, process=False).export(path)
        if mujoco_reads_mesh(path):
            geoms.append({"type": "mesh", "file": f"{CONVEX_PART_DIR}/{path.name}"})
        else:
            path.unlink()
    if len(geoms) <= 1:
        shutil.rmtree(part_dir)
        return None
    return {"geoms": geoms}


def check_sidecar(asset_dir: Path, mesh) -> dict[str, Any]:
    """Compile the sidecar into a throwaway scene and report the resulting geoms."""
    import mujoco

    from toporetarget.solver import add_object_collision

    spec = mujoco.MjSpec()
    body = spec.worldbody.add_body()
    body.name = "asset_check"
    body.mocap = True
    geoms = add_object_collision(spec, body, asset_dir / MESH_NAME, prefix="check", contype=2, conaffinity=1)
    names = [g.name for g in geoms]
    model = spec.compile()
    described = []
    for name in names:
        geom = model.geom(name)
        described.append(
            {
                "name": name,
                "type": int(model.geom_type[geom.id]),
                "size_m": [float(v) for v in model.geom_size[geom.id]],
                "pos_m": [float(v) for v in model.geom_pos[geom.id]],
            }
        )
    hull_extents = np.asarray(mesh.convex_hull.extents, dtype=float)
    return {
        "compiled_geoms": described,
        "compound": len(described) > 1,
        "mesh_extents_m": [float(v) for v in mesh.extents],
        "hull_extents_m": [float(v) for v in hull_extents],
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--asset", type=Path, required=True, help="Asset directory holding mesh.stl.")
    parser.add_argument("--pen", action="store_true", help="Fit a shaft-plus-two-caps collision sidecar.")
    parser.add_argument("--convex", action="store_true", help="Decompose the mesh into convex parts (CoACD).")
    parser.add_argument("--convex-threshold", type=float, default=0.05, help="CoACD concavity threshold.")
    parser.add_argument("--max-parts", type=int, default=16, help="Most convex parts CoACD may produce.")
    parser.add_argument("--shaft-radius", type=float, default=None, help="Shaft radius in metres (default: inferred).")
    parser.add_argument(
        "--cap-fraction", type=float, default=0.08, help="End-cap length as a fraction of the half length."
    )
    parser.add_argument("--mass-kg", type=float, default=None, help="Write this mass into asset.json.")
    parser.add_argument("--check", action="store_true", help="Compile the collision geometry and report it.")
    parser.add_argument("--force", action="store_true", help="Overwrite an existing collision.json.")
    return parser.parse_args()


def main():
    args = parse_args()
    asset_dir = Path(args.asset).expanduser().resolve()
    mesh_path = asset_dir / MESH_NAME
    if not mesh_path.is_file():
        raise SystemExit(f"no {MESH_NAME} under {asset_dir}")
    mesh = load_mesh(mesh_path)
    sidecar_path = asset_dir / COLLISION_SIDECAR

    print(f"[{LOG}] {asset_dir}")
    print(f"[{LOG}] mesh: {json.dumps(describe_mesh(mesh), sort_keys=True)}")

    if args.mass_kg is not None:
        meta = json.loads((asset_dir / META_NAME).read_text()) if (asset_dir / META_NAME).is_file() else {}
        meta["mass_kg"] = float(args.mass_kg)
        (asset_dir / META_NAME).write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
        print(f"[{LOG}] wrote {META_NAME} (mass_kg={args.mass_kg})")

    if args.pen and args.convex:
        raise SystemExit("--pen and --convex are two different sidecars; pick one")
    if args.pen or args.convex:
        if sidecar_path.exists() and not args.force:
            raise SystemExit(f"{sidecar_path} already exists (pass --force to replace it)")
        if args.pen:
            sidecar = fit_pen_sidecar(mesh, shaft_radius=args.shaft_radius, cap_fraction=args.cap_fraction)
        else:
            sidecar = fit_convex_sidecar(mesh, asset_dir, threshold=args.convex_threshold, max_parts=args.max_parts)
        if sidecar is None:
            sidecar_path.unlink(missing_ok=True)
            print(f"[{LOG}] the mesh is one convex part; no sidecar needed")
        else:
            sidecar_path.write_text(json.dumps(sidecar, indent=2) + "\n")
            print(f"[{LOG}] wrote {COLLISION_SIDECAR} with {len(sidecar['geoms'])} parts")

    if sidecar_path.exists():
        print(f"[{LOG}] sidecar: {sidecar_path}")
    else:
        print(f"[{LOG}] no sidecar: the whole mesh will collide as one convex hull")

    if args.check:
        report = check_sidecar(asset_dir, mesh)
        print(f"[{LOG}] collision check: {json.dumps(report, indent=2, sort_keys=True)}")


if __name__ == "__main__":
    main()
