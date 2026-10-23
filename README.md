# TopoRetarget

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![MuJoCo](https://img.shields.io/badge/MuJoCo-3.4+-0a7bbb.svg)](https://mujoco.org)
[![Project page](https://img.shields.io/badge/Project-page-8A2BE2.svg)](https://tsinghua-mars-lab.github.io/toporetarget-web/)
[![Paper](https://img.shields.io/badge/Paper-PDF-b31b1b.svg)](https://tsinghua-mars-lab.github.io/toporetarget-web/static/images/paper.pdf)

> Interaction-preserving retargeting of human hand-object motion to the Wuji Hand: an interaction mesh over hand keypoints and object surface samples, solved frame by frame as a QP under kinematic and non-penetration constraints, with one parameter set across objects, scales and clips. The pen-spin reference motions in [wuji-mjlab](https://github.com/wuji-technology/wuji-mjlab) are generated with TopoRetarget and converted to the task's clip format by the [example adapter](examples/wuji_mjlab/).

Code of *TopoRetarget: Interaction-Preserving Retargeting for Dexterous Manipulation*. [Project page](https://tsinghua-mars-lab.github.io/toporetarget-web/) · [Paper](https://tsinghua-mars-lab.github.io/toporetarget-web/static/images/paper.pdf) · [RL training and deployment](https://github.com/wuji-technology/wuji-mjlab)

## Demos

Rendered by the repository's viewer: the robot hand and the objects, the captured keypoints in blue, the links in contact in orange. Clips of the pen-spin capture that ships in `data/pen_spin`, and of GRAB, TACO, OakInk2 and HOT3D; two-hand clips are solved once per hand.

<p align="center">
  <img src="docs/images/demos.jpg" width="100%" alt="Clips of the pen-spin capture, GRAB, TACO, OakInk2 and HOT3D retargeted to the Wuji Hand 2" />
</p>

The GRAB, TACO, OakInk2 and HOT3D panels render data of those releases (their captured hands and objects) and are not covered by this repository's MIT License: each stays under its release's own terms. HOT3D's hand annotations are under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) and its object models under CC BY-SA 4.0 with a no-sale condition (HOT3D Dataset License Agreement, Meta Platforms Technologies, 2024).

## Method

<p align="center">
  <img src="docs/images/pipeline.png" width="100%" alt="TopoRetarget pipeline: input, bone-direction warm up, interaction-mesh Laplacian optimization, output" />
</p>

From a human demonstration (21 hand keypoints and object poses per frame), the object meshes and the robot hand model, TopoRetarget solves each frame as a small QP: a bone-direction warm up, then an interaction-mesh optimization that preserves the distance-weighted Laplacian over hand keypoints and object surface samples under joint limits and a non-penetration constraint. The robot reproduces which keypoints touch the object and how, not the human's joint angles; one parameter set serves every clip, object and scale, and there is no training stage. A sliding-window pass then reduces finger-to-finger interpenetration, for a bounded number of iterations per window. Details, baselines and the RL results are in the [paper](https://tsinghua-mars-lab.github.io/toporetarget-web/static/images/paper.pdf).

## Requirements

- Linux x86_64, Python 3.10+
- MuJoCo 3.4 or newer (3.3.x is refused: it returns `mj_geomDistance` closest points reversed for mesh pairs)
- No GPU: the solver is CPU-only (daqp)
- [conda](https://docs.conda.io) for the one-line environment, or any Python 3.10+ virtual environment

## Installation

```bash
git clone --recurse-submodules https://github.com/wuji-technology/toporetarget   # with the wuji-description submodule
cd toporetarget
conda env create -f environment.yml
conda activate toporetarget
toporetarget --help
```

Without conda, in any Python 3.10+ virtual environment:

```bash
pip install -e .
pip install torch --index-url https://download.pytorch.org/whl/cpu   # only for TACO and OakInk2
toporetarget --help
```

The environment holds the solver (MuJoCo, daqp), the browser viewer (viser), video rendering, CoACD for concave objects and a CPU build of PyTorch, which only reading TACO and OakInk2 needs. The package carries the scenes, the robot configs and the solver defaults; the Wuji Hand 2 models come from the `wuji-description` submodule. Run the commands from the checkout, where `data/` and `output/` are, or name the dataset with `--data`. `pip install -e ".[dev]"` adds `pytest` and `ruff`; CI runs both.

**MANO** (only for the public datasets, which publish MANO parameters): register at [mano.is.tue.mpg.de](https://mano.is.tue.mpg.de), then

```bash
read -rsp "MANO password: " MANO_PASSWORD && export MANO_PASSWORD
MANO_USERNAME=<email> toporetarget download mano   # -> ~/.cache/toporetarget/mano/
unset MANO_PASSWORD
```

or copy `MANO_RIGHT.pkl` and `MANO_LEFT.pkl` there yourself. `TOPORETARGET_MANO_DIR` names another directory; `download mano --out` writes to one, and `import-hoi --mano-dir` reads from one. The MANO forward pass is reimplemented in NumPy, so neither chumpy nor PyTorch is needed to run it.

## Quick start

A pen-spin clip of our own ships in [`data/pen_spin`](data/README.md), so the solver and the viewer run without downloading anything:

```bash
# retarget the clip (about half a minute on one core) and open the viewer at http://localhost:8080
toporetarget run --data data/pen_spin --clip pen_spin_029
```

With a public dataset, GRAB for example, downloaded to `/data/GRAB` ([guide](docs/datasets/grab.md)):

```bash
# 1. import a few sequences into a canonical 50 Hz dataset (both hands, every object they touch)
toporetarget import-hoi --dataset grab --raw /data/GRAB --out data/grab --units s5/flute_play_1.npz s1/apple_lift.npz

# 2. retarget both hands of one clip and replay them together
toporetarget run --data data/grab --clip s5_flute_play_1__00

# 3. or solve the whole dataset in parallel and render it
toporetarget batch --data data/grab --out output/grab --workers 8
toporetarget view --result-dir output/grab --video output/grab/videos
```

Results land under `output/<dataset>/<robot>_retarget/` as `<clip>_toporetarget_<robot>.npz` -- the playback `qpos` `(T, 27)` on the floating-wrist scene, `fps`, `frame_ids`, joint names and the clip it came from -- plus a `_diagnostics.npz` sidecar with the motion as world-frame arrays (`mediapipe_landmarks_w`, `object_pose_w`, `reference_qpos` `(T, 20)`, `wrist_pose_w`, `link_pos_w` `(T, 21, 3)`, `link_object_contact` `(T, 21)`), the per-frame penetration and fingertip error, and the full effective config. Arrays are Z-up, metres, radians, wxyz.

## Training in wuji-mjlab

The [standalone adapter](examples/wuji_mjlab/) converts results solved from our pen-spin raw capture package into clips for **wuji-mjlab's pen_spin task only: one pen, the right Wuji Hand 2, and 50 Hz**. It is compatible with [wuji-mjlab v2026.10.22](https://github.com/wuji-technology/wuji-mjlab/releases/tag/v2026.10.22) (`b2b98d7`). Other objects, multiple objects and left-hand results are rejected. For your own data, please refer to our data format.

1. Download [toporetarget-assets.zip](https://github.com/wuji-technology/toporetarget/releases/latest/download/toporetarget-assets.zip) and extract it. Set `RAW_DATA` to the absolute path of the extracted `toporetarget-assets/pen_spin/rawdata` directory. From the TopoRetarget checkout and environment, import the raw group and solve one clip:

   ```bash
   toporetarget import-capture --source "$RAW_DATA/04_stable_grasp_rotation" \
     --out output/wuji_mjlab_pen_spin_20261023/capture
   toporetarget run --data output/wuji_mjlab_pen_spin_20261023/capture --clip clip_001 \
     --out output/wuji_mjlab_pen_spin_20261023/clip_001_toporetarget_wujihand2_right.npz --skip-view
   ```

2. Set `WUJI_MJLAB` to the absolute path of your wuji-mjlab checkout, `GROUP=toporetarget_pen_spin_20261023`, and `TASK="$WUJI_MJLAB/src/wuji_mjlab/tasks/pen_spin"`. Run from the TopoRetarget checkout:

   ```bash
   python examples/wuji_mjlab/convert_to_wuji_mjlab_pen_spin.py \
     output/wuji_mjlab_pen_spin_20261023/clip_001_toporetarget_wujihand2_right.npz \
     --source-clip "$RAW_DATA/04_stable_grasp_rotation/rawdata/clip_001.npz" \
     --asset-json "$RAW_DATA/04_stable_grasp_rotation/assets/pen250/asset.json" \
     --out "$TASK/data/rawdata/clips/$GROUP"
   ```

   The required original capture and asset metadata restore the capture's world origin and pen coordinates; the adjacent `_diagnostics.npz` supplies the solved arrays. The hand offset is read from the repository's `wuji-description` submodule. Keep the generated `clip_001.npz` and SHA-256 catalog `index.json` together in this new group; existing output directories are refused. See the [adapter instructions](examples/wuji_mjlab/README.md) for the coordinate conventions. The adapter is a checkout example, not a CLI subcommand, and is excluded from wheel and source distributions.

3. Create `$TASK/data/recipe/$GROUP.json`, creating the `recipe` directory if needed:

   ```json
   {"schema": "wuji_pen_spin_recipe:1", "groups": ["toporetarget_pen_spin_20261023"]}
   ```

4. Set `RECIPE="$TASK/data/recipe/$GROUP.json"`. In the wuji-mjlab checkout, after its environment setup, use the [documented pen_spin training command](https://github.com/wuji-technology/wuji-mjlab/blob/v2026.10.22/src/wuji_mjlab/tasks/pen_spin/docs/train.md#train), pointing its recipe override to this file:

   ```bash
   pixi run train --task WujiHand2_PenSpin \
     --agent.logger tensorboard --agent.upload-model False \
     env.commands.motion.recipe_path="$RECIPE" \
     env.scene.num_envs=4096 agent.max_iterations=10000 \
     agent.run_name="PenSpin_$GROUP"
   ```

   The published checkpoints were trained on their corresponding reference groups, not on this single-clip recipe.

## Motion sources

Keypoint data with 21 hand landmarks, an object pose per frame and object meshes can be used when it conforms to the [canonical dataset](docs/datasets/README.md#the-canonical-dataset) or [keypoint bundle schema](docs/datasets/README.md#keypoint-capture-bundles). The importers produce the canonical dataset layout; the public releases are read from your own download, under their licenses, and split into stretches in which the hands interact with objects.

| Source | Data | Guide |
| --- | --- | --- |
| [GRAB](https://grab.is.tue.mpg.de) | MANO, 120 Hz, 51 objects | [grab.md](docs/datasets/grab.md) |
| [TACO](https://taco2024.github.io) | MANO, 30 Hz, tool-object pairs | [taco.md](docs/datasets/taco.md) |
| [OakInk2](https://oakink.net/v2) | MANO, 120 Hz, multi-step tasks | [oakink2.md](docs/datasets/oakink2.md) |
| [HOT3D](https://facebookresearch.github.io/hot3d) | MANO, 30 Hz, Aria and Quest 3 | [hot3d.md](docs/datasets/hot3d.md) |
| Keypoint capture bundle | 21 MediaPipe-ordered keypoints + object pose (e.g. the pen-spin data in `toporetarget-assets.zip`) | [README.md](docs/datasets/README.md#keypoint-capture-bundles) |

## Layout

```text
toporetarget/            the package and the `toporetarget` command
  solver/                the solver: landmarks, hand (HandSpec), settings, clip, laplacian, model, qp,
                         sampling, retargeter, session (FrameSolver), selfcol, materialize, run
  datasets/              importers (import_hoi + readers/, import_capture, merge), MANO in NumPy, download
  pipeline.py, batch.py, viewer.py, objects.py, config.py, dataset.py, cli.py
  robots/                robot configs (Wuji Hand 2 right, left)
  assets/                retarget scenes; wuji-description/ is the Wuji Hand 2 model (git submodule)
docs/datasets/           the canonical dataset, capture bundles, one guide per release
data/                    datasets, one directory each; the pen-spin example ships, imports are git-ignored
output/                  results, output/<dataset>/<robot>_retarget/ (git-ignored)
tests/                   pytest suite on the example clip
```

## Contributors

- [Jielin Wu](https://github.com/AIRJASON50)
- [Shenzhe Yao](https://github.com/LeopoldYao)
- [Guanqi He](https://github.com/GuanqiHe)
- [Xiaohan Liu](https://github.com/Infas12)
- [Zhaoqing Zeng](https://github.com/zzqsgfw)
- [Xiangrui Jiang](https://github.com/XiangruiJiang)
- [Han Yang](https://github.com/yanghan-a)
- [Wentao Zhang](https://github.com/zhangwt20011015)

## Citation

```bibtex
@inproceedings{toporetarget2026,
  title     = {TopoRetarget: Interaction-Preserving Retargeting for Dexterous Manipulation},
  author    = {Jielin Wu and Shenzhe Yao and Guanqi He and Xiaohan Liu and Zhaoqing Zeng and Xiangrui Jiang and Han Yang and Wentao Zhang and Hang Zhao},
  booktitle = {Conference on Robot Learning (CoRL)},
  year      = {2026}
}
```

## License

MIT; see [LICENSE](LICENSE). The hand models come from the `wuji-description` submodule, under its own MIT `LICENSE`; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md), which also covers the other third-party material. MANO and the public datasets are not included.

## Contact

For any questions, please contact [support@wuji.tech](mailto:support@wuji.tech).
