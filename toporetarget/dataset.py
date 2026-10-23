"""A canonical dataset on disk: the layout the importers write and every command reads."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MOTIONS_DIR = "motions"
ASSETS_DIR = "assets"
META_SUFFIX = ".meta.json"
IMPORT_HINT = "import one with `toporetarget import-hoi` or `toporetarget import-capture`, or point --data at one"


@dataclass(frozen=True)
class CanonicalDataset:
    """One canonical dataset, by its root directory."""

    root: Path

    @classmethod
    def at(cls, root: str | Path) -> CanonicalDataset:
        """The dataset rooted at ``root``; raises unless it has a ``motions/`` directory."""
        root = Path(root).expanduser().resolve()
        if not (root / MOTIONS_DIR).is_dir():
            raise FileNotFoundError(f"{root} is not a canonical dataset (no {MOTIONS_DIR}/ directory); {IMPORT_HINT}")
        return cls(root)

    @classmethod
    def locate(cls, data: str | Path, clip: str | None = None) -> tuple[CanonicalDataset, str | None]:
        """A dataset from its root or from one of its ``motions/<clip>.npz`` files."""
        path = Path(data).expanduser().resolve()
        if path.suffix == ".npz":
            if clip is not None and clip != path.stem:
                raise ValueError(f"{path} names clip {path.stem!r} but clip is {clip!r}")
            return cls.at(path.parent.parent), path.stem
        return cls.at(path), None

    @property
    def name(self) -> str:
        """The dataset's name: its directory name, which results record as ``source_dataset``."""
        return self.root.name

    @property
    def motions(self) -> Path:
        """The ``motions/`` directory."""
        return self.root / MOTIONS_DIR

    @property
    def assets(self) -> Path:
        """The ``assets/`` directory."""
        return self.root / ASSETS_DIR

    def clips(self) -> list[str]:
        """Sorted clip ids."""
        return sorted(p.stem for p in self.motions.glob("*.npz"))

    def resolve_clip(self, clip: str | None) -> str:
        """``clip`` itself, checked to exist, or the dataset's first clip."""
        if clip is None:
            available = self.clips()
            if not available:
                raise FileNotFoundError(f"no clips under {self.motions}")
            return available[0]
        if not self.clip_npz(clip).is_file():
            raise FileNotFoundError(f"clip not found: {self.clip_npz(clip)}")
        return clip

    def clip_npz(self, clip_id: str) -> Path:
        """The clip's motion file."""
        return self.motions / f"{clip_id}.npz"

    def clip_meta_path(self, clip_id: str) -> Path:
        """The clip's ``.meta.json`` sidecar."""
        return self.motions / f"{clip_id}{META_SUFFIX}"

    def meta(self, clip_id: str) -> dict[str, Any]:
        """The clip's metadata: hands, objects, frame count and source."""
        path = self.clip_meta_path(clip_id)
        if not path.is_file():
            raise FileNotFoundError(f"clip meta not found: {path} (expected next to {self.clip_npz(clip_id).name})")
        return json.loads(path.read_text())

    def hands(self, clip_id: str) -> list[str]:
        """The clip's hands (``left`` / ``right``), in the order of its landmark array."""
        return [str(hand) for hand in self.meta(clip_id)["hands"]]

    def objects(self, clip_id: str) -> list[str]:
        """The clip's object asset names, in the order of its object pose array."""
        return [str(name) for name in self.meta(clip_id)["objects"]]

    def frames(self, clip_id: str) -> int:
        """The clip's frame count."""
        return int(self.meta(clip_id)["frames"])

    def asset_dir(self, name: str) -> Path:
        """The directory of one object asset."""
        return self.assets / name

    def clip_inputs(self, clip_id: str) -> list[Path]:
        """Every file a solve of the clip reads: its npz, its meta and its objects' asset files."""
        asset_files = [p for name in self.objects(clip_id) for p in self.asset_dir(name).rglob("*") if p.is_file()]
        return [self.clip_npz(clip_id), self.clip_meta_path(clip_id), *asset_files]
