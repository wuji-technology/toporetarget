# data

Every dataset lives here as one directory in the canonical format
([docs/datasets/](../docs/datasets/README.md)): `assets/<asset>/mesh.stl`
plus `motions/<clip>.npz` and its `.meta.json`.

`pen_spin/` ships with the repository: one right hand spinning a 250 mm pen
(`pen250`) for 231 frames, a keypoint capture of our own. It needs no
download and no MANO registration:

```bash
toporetarget run --data data/pen_spin --clip pen_spin_029   # solve, then view at http://localhost:8080
```

Everything else under `data/` is git-ignored and is written by the importers
from your own download of each release:

```bash
toporetarget import-hoi --dataset grab --raw /data/GRAB --out data/grab
toporetarget import-capture --source /path/to/bundle --out data/my_capture
```

Results go to `output/<dataset>/<robot>_retarget/`; the viewer finds a
result's dataset here by name, so `toporetarget view --result-dir output/grab`
works from the checkout without `--data`.
