# Third-party notices

TopoRetarget is released under the MIT License (`LICENSE`). It includes or
follows the third-party material below, each under its own terms.

## Included

### dex-retargeting

`OPERATOR2MANO_RIGHT`, `OPERATOR2MANO_LEFT` and
`estimate_frame_from_hand_points` in
`toporetarget/solver/landmarks.py` come from [dex-retargeting](https://github.com/dexsuite/dex-retargeting):
logic unchanged; comments and docstring rewritten.

```text
The MIT License (MIT)

Copyright (c) 2023 Yuzhe Qin

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

### FlashDexRetarget

Parts of the TACO, OakInk2 and HOT3D readers -- `units` and `load_mesh` in
`toporetarget/datasets/readers/taco.py`, and `load` in
`toporetarget/datasets/readers/oakink2.py` and
`toporetarget/datasets/readers/hot3d.py` -- are adapted from the readers of the
same releases in [FlashDexRetarget](https://github.com/Holiday-Robot/FlashDexRetarget)
(`src/retarget/human_demo/sources/`), whose release layouts and conventions
they follow.

```text
MIT License

Copyright (c) 2026 Holiday Robotics

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

### Wuji Hand 2 model

`toporetarget/assets/wuji-description` is the
[wuji-description](https://github.com/wuji-technology/wuji-description)
repository at `v2026.10.12`, a git submodule used unmodified, under the MIT
License (Copyright (c) 2025 Wuji Technology) in its own `LICENSE`.

### Example clip

`data/pen_spin` is a keypoint capture of our own, released with the code
under the MIT License.
No clip or object of the public datasets is redistributed.

## Not included

- **MANO** ([mano.is.tue.mpg.de](https://mano.is.tue.mpg.de)) may not be
  redistributed; you download `MANO_RIGHT.pkl` and `MANO_LEFT.pkl` yourself.
  `toporetarget/datasets/mano_model.py` is an independent NumPy implementation of its
  forward pass that reads those files.
- **GRAB, TACO, OakInk2 and HOT3D** recordings are read from your own download
  of each release, under its license.
