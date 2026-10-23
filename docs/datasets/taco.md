# TACO

[TACO](https://taco2024.github.io) records bimanual tool use -- a tool in one
hand acting on a target object, as `(action, tool, target)` triplets -- with
both hands fitted with MANO at 30 Hz. Download the hand poses, object poses and
object models of the full release (the shared folder the
[TACO repository](https://github.com/Satorinal/TACO) links; the videos are not
needed) and accept its terms:

```text
<raw>/
├── Hand_Poses/(<action>, <tool>, <target>)/<sequence>/{right,left}_hand.pkl
├── Hand_Poses/(<action>, <tool>, <target>)/<sequence>/{right,left}_hand_shape.pkl
├── Object_Poses/(<action>, <tool>, <target>)/<sequence>/{tool,target}_<id>.npy
└── object_models_released/<id>_cm.obj
```

The hand pickles hold PyTorch tensors, so the import needs `torch` (the CPU
build is enough).

```bash
toporetarget import-hoi --dataset taco --raw /data/TACO --out data/taco
toporetarget import-hoi --dataset taco --raw /data/TACO --list | head
toporetarget import-hoi --dataset taco --raw /data/TACO --out data/taco \
  --units "(brush, brush, bowl)/20230919_036" "(cut, knife, plate)/20230926_036"
```

## How it is read

- Hands: 48 axis-angle values (root first) on the flat mean, with the
  sequence's shape coefficients. `hand_trans` is the position of MANO joint 0
  (the wrist), not the smplx translation: placed by joint 0, the hands touch
  the objects they hold (median closest hand-object distance 0.3 mm over the
  frames in contact); read as the smplx translation, they float 2-5 cm away.
- Objects: `(T, 4, 4)` object-to-world poses; the meshes are in centimetres
  and are scaled to metres. Object ids are TACO's numeric ids.
- World: Z-up already.
- A recording is named `<action>__<tool>__<target>__<sequence>`, so a clip is
  e.g. `brush__brush__bowl__20230919_036__00`.

One hand usually holds the tool and the other the target, so a recording
becomes one clip with both hands and both objects.

The tool and the target are often bowls, cups and plates -- strongly concave
(a TACO bowl fills 6 % of its convex hull) -- so keep the default convex
decomposition.

## What the release yields

The whole release imports without a failed recording: 2,317 clips, every one
with both hands, 3.2 h at 50 Hz, on 162 objects. With `--collision hull` that
takes under two minutes on 28 cores; the default convex decomposition adds
roughly half a minute of CoACD per object.
