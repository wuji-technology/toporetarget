"""The ``toporetarget`` command: one entry point for the package's modules."""

from __future__ import annotations

import importlib
import os
import sys

COMMANDS: dict[str, tuple[str, str]] = {
    "run": ("pipeline", "retarget one clip and open the viewer"),
    "batch": ("batch", "solve every hand of every clip of a dataset"),
    "view": ("viewer", "browser viewer and offscreen video of results"),
    "import-hoi": ("datasets.import_hoi", "GRAB / TACO / OakInk2 / HOT3D release -> canonical dataset"),
    "import-capture": ("datasets.import_capture", "keypoint capture bundle -> canonical dataset"),
    "merge": ("datasets.merge", "join canonical datasets into one"),
    "download": ("datasets.download", "fetch the MANO models or the OakInk2 release"),
    "prepare-object": ("objects", "compound collision sidecar for an object asset"),
}


def usage() -> str:
    """The command table, as ``toporetarget --help`` prints it."""
    width = max(len(name) for name in COMMANDS)
    lines = ["usage: toporetarget <command> [args...]", "", "commands:"]
    lines += [f"  {name:<{width}}  {summary}" for name, (_, summary) in COMMANDS.items()]
    lines += ["", "`toporetarget <command> --help` shows the options of one command."]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    """Run the command named by ``argv`` (default ``sys.argv[1:]``) through its module."""
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print(usage())
        return
    command, rest = argv[0], argv[1:]
    if command not in COMMANDS:
        sys.exit(f"toporetarget: unknown command {command!r}\n\n{usage()}")
    sys.argv = [f"toporetarget {command}", *rest]
    module = importlib.import_module(f"toporetarget.{COMMANDS[command][0]}")
    if os.environ.get("TOPORETARGET_DEBUG") == "1":
        module.main()
        return
    from toporetarget.datasets.canonical import DatasetError
    from toporetarget.datasets.merge import MergeError

    try:
        module.main()
    except (FileNotFoundError, DatasetError, MergeError) as exc:
        sys.exit(f"toporetarget {command}: error: {exc}\n(TOPORETARGET_DEBUG=1 shows the traceback)")


if __name__ == "__main__":
    main()
