# Changelog

All notable changes to this project will be documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project uses calendar versioning (YYYY.M.D). The Python package
version in `pyproject.toml` is numbered separately.

## [Unreleased]

## [2026.10.23]

### Added

- Released the initial public version (package version 0.1.0): the
  TopoRetarget solver and the `toporetarget` command (`run`, `batch`, `view`,
  `import-hoi`, `import-capture`, `merge`, `download`, `prepare-object`);
  importers for GRAB, TACO, OakInk2 and HOT3D and for keypoint capture
  bundles; the Wuji Hand 2 right and left configs on the `wuji-description`
  submodule (`v2026.10.12`); and the `pen_spin_029` example clip in
  `data/pen_spin`.
- `examples/wuji_mjlab/`: a standalone converter from TopoRetarget results to
  wuji-mjlab v2026.10.22 `pen_spin` clips (one pen, right Wuji Hand 2, 50 Hz),
  with the steps in the README section "Training in wuji-mjlab".
- `toporetarget-assets.zip` release attachment: the pen-spin raw data for the
  wuji-mjlab training path, under the MIT License.

[Unreleased]: https://github.com/wuji-technology/toporetarget/compare/v2026.10.23...HEAD
[2026.10.23]: https://github.com/wuji-technology/toporetarget/releases/tag/v2026.10.23
