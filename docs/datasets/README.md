# Datasets

TopoRetarget reads one format, the *canonical dataset*, and the dataset
commands produce it: `toporetarget download` fetches what can be fetched
without a browser (the MANO models with your account, the OakInk2 release),
`toporetarget import-hoi` imports the public MANO hand-object datasets and
`toporetarget import-capture` keypoint capture bundles (`toporetarget
<command> --help` lists the options; the code is `toporetarget/datasets/`).
Datasets live under `data/`, one directory each; only the pen-spin example
`data/pen_spin` is committed, imported datasets land next to it and git
ignores them.

Download every release and the MANO models only from the official pages linked
below. GRAB, TACO and OakInk2 store their annotations as Python pickles, which
can run code when they are read; the importers read them as the releases ship
them, so a file from anywhere else should not be imported. The MANO reader
builds only what a MANO model file holds -- NumPy arrays, dtypes and scalars,
SciPy CSC/CSR sparse matrices, sets, and the stored state of its chumpy
objects, which become plain attribute holders -- and refuses a file that asks
for anything else.

| Source | Importer | Guide |
| --- | --- | --- |
| [GRAB](https://grab.is.tue.mpg.de) | `toporetarget import-hoi --dataset grab` | [grab.md](grab.md) |
| [TACO](https://taco2024.github.io) | `toporetarget import-hoi --dataset taco` | [taco.md](taco.md) |
| [OakInk2](https://oakink.net/v2) | `toporetarget import-hoi --dataset oakink2` | [oakink2.md](oakink2.md) |
| [HOT3D](https://facebookresearch.github.io/hot3d) | `toporetarget import-hoi --dataset hot3d` | [hot3d.md](hot3d.md) |
| Keypoint capture bundle | `toporetarget import-capture` | [below](#keypoint-capture-bundles) |

## The canonical dataset

```text
data/<dataset>/
├── manifest.json               import provenance and settings
├── assets/<asset>/
│   ├── mesh.stl                the sampling and collision mesh, object frame, metres
│   ├── asset.json              {"mesh_sha256", ["mass_kg"], ...}
│   ├── collision.json          optional compound collision sidecar
│   └── collision/part_*.stl    convex parts the sidecar names, if any
└── motions/
    ├── <clip>.npz              mediapipe_landmarks_w (T,H,21,3), object_pose_w (T,K,7), fps
    └── <clip>.meta.json        schema, clip_id, fps, hands, objects, frames, source
```

One clip is one stretch of a recording with its `H` hands and `K` objects:
`hands` in the meta names the hands in the order of the landmark array's
second axis, and `objects` the asset of each object pose track. Everything is
world frame, Z-up, metres, `[x y z qw qx qy qz]`, at 50 Hz, and every clip is
translated so its first object starts at XY zero (Z, relative positions and
orientations are unchanged). The 21 landmarks follow the MediaPipe hand
order: wrist; thumb CMC, MCP, IP, tip; then MCP, PIP, DIP, tip of the index,
middle, ring and little finger. Robot fields are not written here; they come
from the retarget stage.

The importers do not apply a clip-quality filter. Each bundle index entry
describes one clip, and MANO annotations are split into stretches in which
their hands interact with objects. Such a clip holds both hands whenever both
are tracked throughout it, the idle one included, and every object a hand
touches in it: the two arms of a pair of tongs, tracked as two objects, or a
knife and the bread it cuts stay in one clip.

`toporetarget batch` attempts each requested hand with the robot config of
that side, against all of the clip's objects. Asset compilation failures are
reported, and the affected hands are not solved. The viewer shows a clip's
hands together.

## Compound object collision

MuJoCo collides a single mesh geom as its convex hull. That fills a bowl or a
mug solid, and it holds a finger off the shaft of a pen with wider end caps.
A `collision.json` sidecar next to `mesh.stl` replaces the one hull with
several parts, each its own geom: `box` and `cylinder` primitives, or `mesh`
convex parts. Surface points are still sampled
from `mesh.stl`; only the non-penetration constraint and the playback scene use
the parts.

```bash
# convex decomposition with CoACD (pip install coacd), for concave objects
toporetarget prepare-object --asset data/<dataset>/assets/<asset> --convex
# a shaft cylinder plus two end-cap boxes, for a pen
toporetarget prepare-object --asset <dir> --pen --shaft-radius 0.00745
# compile the result and report its geoms
toporetarget prepare-object --asset <dir> --check
```

`toporetarget import-hoi` runs the convex decomposition for every object it
writes (`--collision convex`, the default). A convex object comes out as one
part and gets no sidecar, and a part MuJoCo does not compile, a sliver with no
volume (one of TACO object 168's is 40 µm across), is left out. For the bundled pen250 geometry
(7.45 mm shaft, 18 mm square heads), the hull sits 9.0 mm from the shaft axis
face-on and 12.7 mm at its corners, the primitive sidecar at 7.45 mm; on a
TACO bowl the mesh fills 6 % of its hull, and 16 convex parts follow its inside.

Before that it quadric-decimates a mesh of more than 200 000 faces, the most
MuJoCo's STL decoder reads, to that count (`fast-simplification`, through
trimesh). The manifest entry keeps the source face count and, under
`decimation`, how far the surface moved (maximum and 99th percentile, both
ways, over vertices, face centroids and surface samples) and whether the sorted
lists of connected-component Euler numbers match before and after simplification;
the import log warns past 0.15 mm / 0.10 mm or on a topology change, and writes the mesh either way. Of the four
public releases only the HOT3D birdhouse toy (385 843 faces) needs it: 0.67 mm
maximum, 0.18 mm p99, and its 11 components come out as 40. Every asset is
then compiled once by MuJoCo; one that does not compile is still written, with
`compile_error` in its manifest entry and a warning in the log.
`toporetarget batch` compiles the objects of its pending solves again before
starting and fails the hands of their clips at once, so a mesh MuJoCo cannot
read costs seconds, not the end of every solve that uses it.

## Keypoint capture bundles

A keypoint bundle contains 21 MediaPipe-ordered landmarks and an object pose
per frame, plus timestamps and object assets:

```text
<bundle>/
├── index.json                  {handedness, fps, clip_count, clips}
├── rawdata/<clip>.npz          mediapipe_landmarks_w, object_pose_w, t_s [, fps]
└── assets/<asset>/
    ├── mesh_med.stl            (or mesh.stl)
    ├── asset.json              {"mass_kg": ..., "mesh_sha256": ...}
    └── collision.json          optional compound collision sidecar
```

Each `index.json` clip entry gives `raw_npz_path`, `object_asset` and
`frame_count`. Per clip the npz carries the native-rate world-frame
observations: `mediapipe_landmarks_w` `(T, 21, 3)`, `object_pose_w` `(T, 7)` as
`[xyz, qw qx qy qz]`, and the `t_s` timestamps the importer resamples against.
`wrist_pose_w` may be present but is not read: the retargeter derives the wrist
frame from the landmarks itself, so one convention holds for every source.

```bash
toporetarget import-capture --source /path/to/bundle --out data/my_capture
# or by name, from motion_dataset_registry.local.json in the working directory
toporetarget import-capture --list
toporetarget import-capture --dataset my_capture
```

A registry entry names a bundle so it can be imported by name:

```json
{
  "datasets": {
    "my_capture": {
      "description": "what the bundle holds",
      "source": "/path/to/bundle",
      "out": "../data/my_capture",
      "mass_kg": 0.025
    }
  }
}
```

The registry is `motion_dataset_registry.local.json` in the working
directory (`--registry` names another file); write it yourself if you want
one. Git ignores it, so entries for your own bundles stay out of the
repository. Relative paths resolve against it. An explicit `--mass-kg` takes
priority over the registry's `mass_kg`; either overrides the asset's mass.
When neither is supplied, the bundle's asset must declare `mass_kg`.

Per clip, in order, the importer

1. reads the native-rate observations and measures the frame rate from `t_s`;
   an NPZ's own `fps`, if present, must agree with that rate, and non-finite
   landmarks or poses are rejected. The index's `fps` is only recorded as
   `source_declared_fps` in the output manifest, not checked against `t_s`;
2. resamples to 50 Hz against the source timestamps (linear for landmarks,
   linear + SLERP for the object pose);
3. translates the trajectory so the initial object XY sits at the origin.

A bundle clip is one hand (the bundle's `handedness`) and one object.

### Merging datasets

Import each bundle, then merge the resulting canonical datasets:

```bash
toporetarget merge --out data/merged \
  data/_staging/bundle1 data/_staging/bundle2
```

The merge copies every clip npz and `meta.json` verbatim -- nothing is
resampled a second time -- and refuses rather than guess when two
inputs disagree about the target rate, a clip id or an object asset's bytes.
The merged `manifest.json` keeps each clip's originating dataset.
