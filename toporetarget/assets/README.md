# Assets

- `wuji-description/`: the [wuji-description](https://github.com/wuji-technology/wuji-description)
  submodule at `v2026.10.12` (MIT license, Copyright (c) 2025 Wuji Technology).
  The Wuji Hand 2 models are not copied here: the robot configs in
  `toporetarget/robots/` and the scenes in `scenes/` read
  `hand2/hand2_2/body/mjcf/{right,left}.xml` and their meshes from it as they
  are. Fetch it with `git clone --recurse-submodules`, or with
  `git submodule update --init` in an existing clone.
- `scenes/`: the single-hand retarget scenes, one per side. Each attaches the
  hand MJCF under a `wuji_wrist` body and repeats its `<option>` block, so
  attaching it raises no conflict; the solver adds the 6-DoF virtual wrist and
  the object at runtime (`toporetarget/solver/model.py`).

## How the solver reaches the hand

Only through the description's names, listed in the `hand` block of
`toporetarget/robots/wujihand2_{right,left}.json`, under an `r_` / `l_` side
prefix:

- the root `{side}_mount`, the arm-flange mount: its origin is where the
  wrist keypoint lands, and its z axis runs from the fingers toward the arm;
  the wrist shell `{side}_wrist` is welded onto it;
- four links per finger from the knuckle out, `{finger}_proximal`,
  `{finger}_proximal_abd`, `{finger}_middle` and `{finger}_distal`, with
  `{finger}` one of `thumb`, `index_finger`, `middle_finger`, `ring_finger`
  and `pinky`, each turning on its own hinge (`thumb_cmc_flex` ...
  `pinky_dip`, 20 in all);
- the fingertip sites `{finger}_tip` on the distal links.

A fingertip pad (`{finger}_tip_sensor_frame`) is a body of its own without a
joint and counts as part of the distal link it is welded onto, for the contact
mask, the hand-object constraint and the finger-to-finger overlap the
self-collision pass measures. That measurement keeps the MJCF's `<exclude>`
pairs.

The collision geometry is the meshes of geom group 2 (MuJoCo collides each as
its convex hull), the visual meshes are group 1. The hand-object constraint
leaves out every finger's `proximal` link, which sits at the knuckle line and
produces false-positive contact under any grip: it acts on the other 22 meshes
(see `_enable_hand_collision_geoms` in `toporetarget/solver/model.py`). The
collision geoms keep MuJoCo's default `solimp="0.9 0.95 0.001"`, whose 1 mm
width is what `penetration.tolerance` is aligned with.
