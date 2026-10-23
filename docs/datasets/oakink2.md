# OakInk2

[OakInk2](https://oakink.net/v2) records bimanual object manipulation for
long multi-step tasks, both hands fitted with MANO on 120 Hz motion capture.
TopoRetarget reads the preview annotations and the repaired object models of
the release on Hugging Face (`kelvin34501/OakInk-v2`, CC BY-SA 4.0), which
`toporetarget download` fetches -- all 627 sequences (37 GB) or the ones named:

```bash
toporetarget download oakink2 --raw /data/OakInk2
toporetarget download oakink2 --raw /data/OakInk2 \
  --sequences "scene_01__A008++seq__495b60ed26f7b9c5cd2e__2023-04-23-11-41-39"
```

It writes the layout the importer reads:

```text
<raw>/
├── anno_preview/<sequence>.pkl
└── object_repair/align_ds/<object>/model.obj
```

The annotation pickles hold PyTorch tensors, so the import needs `torch` (the
CPU build is enough). File sizes vary substantially; plan worker memory around
the largest annotation files in the release's file listing.

```bash
toporetarget import-hoi --dataset oakink2 --raw /data/OakInk2 --out data/oakink2 --workers 6
```

## How it is read

- Hands: 16 rotations per hand as wxyz quaternions (root first) on the flat
  mean, with per-frame shape coefficients. `tsl` is the position of MANO joint 0.
  Both were checked against the objects: read as wxyz with `tsl` at joint 0,
  a hand comes within 5 mm of the object it holds in about half of a
  sequence's frames; read as xyzw, or with `tsl` as the smplx translation,
  almost never.
- Objects: `obj_transf` per annotation frame, object to world; meshes in metres.
  Objects without a mesh under `object_repair/align_ds` are skipped. Object
  ids look like `O02@0015@00019`; the asset is named `O02_0015_00019`.
- World: Y-up, turned Z-up.
- A unit is the whole sequence. OakInk2 sequences chain several tasks, and
  the importer cuts them into the stretches in which the hands interact with
  objects, so one sequence yields several clips.
- OakInk2 tracks the parts of an articulated object as objects of their own:
  the two arms of a pair of tongs (`O02@0032@00001`, `@00002`), a laptop's lid
  and base, an alcohol burner and its cap. A hand that holds both arms of the
  tongs touches two objects at once, and so does a hand that holds a knife to
  bread, so a clip keeps every object a hand touches in it together; solved
  against one part alone, the fingers on the other would close on nothing.

## What the release yields

The whole preview release imports without a failed sequence: 1,215 clips,
every one with both hands, 6.9 h at 50 Hz, on 106 objects. With
`--collision hull` that takes 31 minutes on 8 cores, most of it reading the
annotation pickles.
