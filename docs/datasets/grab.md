# GRAB

[GRAB](https://grab.is.tue.mpg.de) records ten subjects grasping and using 51
objects, both hands fitted with MANO at 120 Hz. Register on the GRAB website,
accept its license and download, from its download page once signed in, the
sequences of every subject you want plus the object and subject meshes; unpack
them with GRAB's own `grab/unzip_grab.py`
([github.com/otaheri/GRAB](https://github.com/otaheri/GRAB)), which puts each
archive where its name says. TopoRetarget reads the release as it unpacks:

```text
<raw>/
├── grab/s1 ... s10/<sequence>.npz
└── tools/
    ├── object_meshes/contact_meshes/<object>.ply
    └── subject_meshes/{male,female}/s<N>_{rhand,lhand}.ply
```

Each sequence names the hand templates it was fitted on
(`tools/subject_meshes/.../s1_rhand.ply`, `s1_lhand.ply`); both hands need
theirs, so make sure the left-hand templates were unpacked too.

```bash
toporetarget import-hoi --dataset grab --raw /data/GRAB --out data/grab
# a few sequences only
toporetarget import-hoi --dataset grab --raw /data/GRAB --out data/grab \
  --units s1/mug_drink_1.npz s1/apple_lift.npz
toporetarget batch --data data/grab --out output/grab --workers 8
```

## How it is read

- Hands: 24 MANO PCA components on the flat mean, on the subject's own hand
  template instead of shape coefficients, placed by the smplx translation.
- Object: axis-angle rotation and translation per frame; the meshes are in
  metres. GRAB's object model rotates row vectors (`v @ R` in its
  `tools/objectmodel.py`), so the object-to-world rotation is the transpose of
  the axis-angle's matrix. Read the other way, a held object turns against the
  hand holding it: the hand stays on a mug only while the mug is upright. The table is not imported: TopoRetarget solves the hand against the
  manipulated object only.
- World: Z-up already.
- A unit is `<subject>/<sequence>.npz`; a recording is named
  `<subject>_<sequence>`, so a clip is e.g. `s1_mug_drink_1__00`.

Most GRAB sequences approach, grasp, use and put down one object, sometimes
passing it from hand to hand. A sequence becomes one clip with both hands, or
two when the object rests untouched for a while between two grasps.

## What the release yields

All 1,335 sequences import without a failure, every one into at least one
clip: 1,364 clips, every one with both hands, 2.8 h at 50 Hz, on 51 objects.
With `--collision hull` that takes 22 minutes on 2 cores; the default convex
decomposition adds roughly half a minute of CoACD per object.

GRAB's object tracking jumps in places: 15 clips hold a step in which the
object turns by 62-179° from one 50 Hz frame to the next, mostly symmetric
objects such as the small pyramid flipping between look-alike poses. They are
imported as recorded, like every other clip.
