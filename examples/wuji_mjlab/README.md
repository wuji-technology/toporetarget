# wuji-mjlab pen_spin adapter

This standalone example converts results solved from our pen-spin raw
capture package into training clips for **wuji-mjlab's pen_spin task only: one
pen, the right Wuji Hand 2, and 50 Hz**. It is compatible with
[wuji-mjlab v2026.10.22](https://github.com/wuji-technology/wuji-mjlab/releases/tag/v2026.10.22)
(`b2b98d7`; the pen_spin readers are unchanged from `1cb9810`). It is not a general
retargeting exporter. For your own data, please refer to our data format.

## Raw data to training clips

The raw data is in [toporetarget-assets.zip](https://github.com/wuji-technology/toporetarget/releases/latest/download/toporetarget-assets.zip).
Follow [Training in wuji-mjlab](../../README.md#training-in-wuji-mjlab)
for the complete raw-data → import-capture → solve → conversion → recipe →
training sequence. Use a TopoRetarget checkout with its `wuji-description`
submodule initialized and normal project dependencies installed. The script
imports only NumPy, MuJoCo and the Python standard library, not the TopoRetarget
package or unpublished helpers. This directory, including its tests, is excluded
from wheel and source distributions.

From the TopoRetarget checkout, after importing and solving as in the main README:

```bash
python examples/wuji_mjlab/convert_to_wuji_mjlab_pen_spin.py \
  output/wuji_mjlab_pen_spin_20261023/clip_001_toporetarget_wujihand2_right.npz \
  --source-clip "$RAW_DATA/04_stable_grasp_rotation/rawdata/clip_001.npz" \
  --asset-json "$RAW_DATA/04_stable_grasp_rotation/assets/pen250/asset.json" \
  --out "$TASK/data/rawdata/clips/$GROUP"
```

`RAW_DATA` is the absolute path of the extracted `toporetarget-assets/pen_spin/rawdata`
directory; `TASK` is the absolute path to your wuji-mjlab checkout's `src/wuji_mjlab/tasks/pen_spin`
directory, and `GROUP` is a new descriptive group ID. Keep the generated
`index.json` and NPZ files together. The output directory must not already exist.
A `wuji_pen_spin_recipe:1` recipe must select that group before training uses it.
The catalog records a 0.02 s frame interval, clip IDs, filenames and SHA-256 hashes.

Both `--source-clip` and `--asset-json` are **required**. Each result must be next
to its matching `<result_stem>_diagnostics.npz` sidecar. Multiple result paths can
be supplied for one group: repeat `--source-clip` once per result, in result order.
Clip IDs must be unique. One `--asset-json` applies to all results in a command;
convert legacy `pen` and `pen250` captures separately, into different output groups.

## Coordinates and supported inputs

- Left-hand, multi-object, non-pen, and non-50 Hz inputs are rejected before
  output is written. The supported raw clips are already 50 Hz; the converter
  does not resample or rerun the solver.
- The original capture NPZ supplies `mediapipe_landmarks_w (T,21,3)` and `t_s`.
  The importer subtracts the initial object's XY position without recording
  that translation in the result. Matching the original landmarks against the
  diagnostics recovers it; the difference must be a constant translation. This
  preserves the original landmarks and world origin, without consulting an
  already-converted clip. It does not restore a final source frame omitted by
  the importer or eliminate differences from another solve.
- The result's `r_mount` and the palm are different frames. The task needs the
  palm (`r_wrist` in the source model, `right_palm_link` in the task). Its offset
  is read from the repository submodule's `hand2/hand2_2/body/mjcf/right.xml`.
  `--hand-xml` can name the corresponding model used to solve the result.
  Joint/link names are mapped to the task's finger-major order.
- Only object IDs `pen` and `pen250` are supported. Use the original raw package's
  `assets/<name>/asset.json`, not the canonical import's reduced metadata.
  Its `source_to_capture_rotation` maps task-asset coordinates into capture
  coordinates and is postmultiplied into the captured object quaternion.
  In `pen250`, the capture shaft is +Z and the task shaft is +X, requiring
  `Ry(-90°)`; the legacy `pen` convention is identity when its original metadata
  omits the rotation. Missing rotation for `pen250` is an error. No files in an
  asset's `source/` subdirectory are read, and no wuji-mjlab asset files are needed.

NPZ output contains the five float32 arrays expected by the task:
`mediapipe_landmarks_w (T,21,3)`, `reference_qpos (T,20)`,
`wrist_pose_w (T,7)`, `object_pose_w (T,7)`, and `link_pos_w (T,21,3)`.
Poses use metres and wxyz quaternions; joints use radians. Wrong palm/pen frames
can produce valid-looking arrays with incorrect motion, which is why this
adapter is restricted to this task and these inputs.

## Example tests

```bash
ruff check examples/wuji_mjlab
ruff format --check examples/wuji_mjlab
pytest examples/wuji_mjlab
```
