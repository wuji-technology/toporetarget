#!/usr/bin/env python3
r"""Download the parts of the public releases that can be fetched without a browser.

Examples:
    toporetarget download oakink2 --raw /data/OakInk2
    toporetarget download oakink2 --raw /data/OakInk2 \
        --sequences "scene_01__A008++seq__495b60ed26f7b9c5cd2e__2023-04-23-11-41-39"
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tarfile
import tempfile
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

from toporetarget.datasets.mano_model import DEFAULT_MANO_DIR

LOG = "Download"
CHUNK = 1 << 20

MANO_URL = "https://download.is.tue.mpg.de/download.php?domain=mano&resume=1&sfile=mano_v1_2.zip"
MANO_MODELS = ("MANO_RIGHT.pkl", "MANO_LEFT.pkl")

OAKINK2_REPO = "kelvin34501/OakInk-v2"
OAKINK2_ANNOTATIONS = "anno_preview"
OAKINK2_OBJECTS = "object_repair.tar"


def fetch(request: str | urllib.request.Request, destination: Path, *, sha256: str | None = None) -> None:
    """Stream ``request`` to ``destination`` through a ``.part`` file, checking its SHA-256 when given."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    part = destination.with_name(destination.name + ".part")
    digest = hashlib.sha256()
    with urllib.request.urlopen(request) as response, part.open("wb") as out:
        while chunk := response.read(CHUNK):
            out.write(chunk)
            digest.update(chunk)
    if sha256 is not None and digest.hexdigest() != sha256:
        part.unlink()
        raise SystemExit(f"[{LOG}] {destination.name}: SHA-256 {digest.hexdigest()} != published {sha256}")
    part.replace(destination)


def download_mano(out: Path) -> None:
    username, password = os.environ.get("MANO_USERNAME"), os.environ.get("MANO_PASSWORD")
    if not username or not password:
        raise SystemExit(f"[{LOG}] set MANO_USERNAME and MANO_PASSWORD to your mano.is.tue.mpg.de account")
    form = urllib.parse.urlencode({"username": username, "password": password}).encode()
    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / "mano_v1_2.zip"
        print(f"[{LOG}] mano_v1_2.zip from download.is.tue.mpg.de")
        fetch(urllib.request.Request(MANO_URL, data=form), archive)
        if not zipfile.is_zipfile(archive):
            raise SystemExit(
                f"[{LOG}] the server sent no archive: check MANO_USERNAME and MANO_PASSWORD, and that the "
                "account has accepted the MANO license"
            )
        with zipfile.ZipFile(archive) as bundle:
            members = {Path(name).name: name for name in bundle.namelist() if Path(name).parent.name == "models"}
            out.mkdir(parents=True, exist_ok=True)
            for model in MANO_MODELS:
                if model not in members:
                    raise SystemExit(f"[{LOG}] mano_v1_2.zip holds no models/{model}")
                (out / model).write_bytes(bundle.read(members[model]))
                print(f"[{LOG}] wrote {out / model}")


def hf_files(repo: str, folder: str = "") -> list[dict]:
    """The files of one folder of a Hugging Face dataset repository: path, size and LFS SHA-256."""
    url = f"https://huggingface.co/api/datasets/{repo}/tree/main/{folder}".rstrip("/")
    with urllib.request.urlopen(url) as response:
        entries = json.load(response)
    return [entry for entry in entries if entry["type"] == "file"]


def hf_fetch(repo: str, entry: dict, root: Path) -> Path:
    """Download one file of a Hugging Face dataset into ``root/<path>``, unless it is there already."""
    destination = root / entry["path"]
    if destination.is_file() and destination.stat().st_size == entry["size"]:
        return destination
    print(f"[{LOG}] {entry['path']} ({entry['size'] / 1e6:.0f} MB)")
    sha256 = entry.get("lfs", {}).get("oid")
    fetch(f"https://huggingface.co/datasets/{repo}/resolve/main/{entry['path']}", destination, sha256=sha256)
    return destination


def extract_tar(archive: Path, root: Path, top: str) -> None:
    """Unpack a tarball whose members all lie under ``top/`` into ``root``."""
    with tarfile.open(archive) as bundle:
        members = bundle.getmembers()
        for member in members:
            parts = Path(member.name).parts
            if not parts or parts[0] != top or ".." in parts or not (member.isfile() or member.isdir()):
                raise SystemExit(f"[{LOG}] {archive.name}: unexpected member {member.name!r}")
        safe = {"filter": "data"} if hasattr(tarfile, "data_filter") else {}
        bundle.extractall(root, members=members, **safe)


def download_oakink2(raw: Path, sequences: list[str] | None) -> None:
    annotations = hf_files(OAKINK2_REPO, OAKINK2_ANNOTATIONS)
    if sequences:
        wanted = {Path(name).stem for name in sequences}
        annotations = [entry for entry in annotations if Path(entry["path"]).stem in wanted]
        missing = wanted - {Path(entry["path"]).stem for entry in annotations}
        if missing:
            raise SystemExit(f"[{LOG}] no OakInk2 sequence {sorted(missing)}")
    total = sum(entry["size"] for entry in annotations)
    print(f"[{LOG}] {len(annotations)} OakInk2 sequences, {total / 1e6:,.0f} MB of annotations, into {raw}")
    for entry in annotations:
        hf_fetch(OAKINK2_REPO, entry, raw)

    objects = next(entry for entry in hf_files(OAKINK2_REPO) if entry["path"] == OAKINK2_OBJECTS)
    if not (raw / "object_repair" / "align_ds").is_dir():
        archive = hf_fetch(OAKINK2_REPO, objects, raw)
        extract_tar(archive, raw, "object_repair")
        archive.unlink()
    print(f"[{LOG}] done: toporetarget import-hoi --dataset oakink2 --raw {raw} --out <dataset>")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="source", required=True)
    mano = commands.add_parser("mano", help="the MANO hand models (needs a MANO account)")
    mano.add_argument("--out", type=Path, default=DEFAULT_MANO_DIR, help="Where the two models go.")
    oakink2 = commands.add_parser("oakink2", help="OakInk2 preview annotations and repaired object models")
    oakink2.add_argument("--raw", type=Path, required=True, help="Release root to download into.")
    oakink2.add_argument("--sequences", nargs="+", default=None, help="Only these sequences (default: all).")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.source == "mano":
        download_mano(args.out.expanduser().resolve())
    else:
        download_oakink2(args.raw.expanduser().resolve(), args.sequences)


if __name__ == "__main__":
    main()
