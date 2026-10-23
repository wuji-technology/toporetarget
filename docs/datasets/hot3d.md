# HOT3D

[HOT3D](https://facebookresearch.github.io/hot3d) records participants
handling 33 objects in kitchen, office and living-room scenes, with Aria glasses
and Quest 3 headsets; hands are fitted with MANO and objects tracked by motion
capture. Sign up on the HOT3D website, accept its license and take the URL
lists it issues (`Hot3DAria_download_urls.json`, `Hot3DQuest_download_urls.json`,
`Hot3DAssets_download_urls.json`); HOT3D's own downloader
(`hot3d/data_downloader/dataset_downloader_base_main.py` in
[facebookresearch/hot3d](https://github.com/facebookresearch/hot3d)) fetches
the recordings and the object library with them. TopoRetarget reads the
recordings that have hand and object ground truth:

```text
<raw>/
├── dataset/<recording>/
│   ├── mano_hand_pose_trajectory.jsonl
│   ├── dynamic_objects.csv
│   └── metadata.json
└── _assets/assets/
    ├── <uid>.glb
    └── instance.json
```

```bash
toporetarget import-hoi --dataset hot3d --raw /data/HOT3D --out data/hot3d
toporetarget import-hoi --dataset hot3d --raw /data/HOT3D --out data/hot3d \
  --units P0001_624f2ba9 P0002_1464cbdc
```

## How it is read

- Hands: 15 MANO PCA components on MANO's mean pose, with the participant's
  shape coefficients, placed by the smplx translation (`wrist_xform`). Hand
  `"1"` is the right hand, `"0"` the left.
- The left hand's shape directions are corrected, as HOT3D's own MANO layer
  does (`MANO_LEFT.pkl` ships the right hand's x shape directions unmirrored,
  [smplx#48](https://github.com/vchoutas/smplx/issues/48)). HOT3D's
  participants have large shape coefficients, so it matters: without the
  correction the left hands hold their objects with markedly fewer vertices in
  contact.
- Objects: one pose per object and timestamp in `dynamic_objects.csv`; the
  meshes are in metres. Object ids are HOT3D's instance names (`mug_white`,
  `spoon_wooden`, ...), looked up in `instance.json`.
- Time: the streams of one capture carry timestamps a few milliseconds apart,
  so a timestamp less than 10 ms after the last one kept counts as the same
  frame and is dropped, which leaves HOT3D's 30 Hz. A clip ends where a hand or an
  object it touches is untracked; a hand that touches nothing in it and drops
  out of tracking is left out of that clip.
- World: Aria recordings are Z-up; Quest 3 recordings are Y-up and turned Z-up.

A recording runs for minutes and touches several objects with both hands, so
it yields several clips, each with the objects handled in it.

## What the release yields

Every recording with ground truth imports: 3,108 clips (3,091 with the right
hand, 3,030 with the left), 7.4 h at 50 Hz, on 33 objects. With
`--collision hull` that takes two minutes on 28 cores.
