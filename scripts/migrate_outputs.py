#!/usr/bin/env python3
"""One-shot migration: move the legacy flat `outputs/` layout to the new
nested `outputs/{prompt}/{model}/{ts}/` layout introduced in v0.2.0.

Source layout (v0.1.0):
    outputs/
        results.jsonl
        results.json
        run_summary.json
        validation_report.json
        logs/<id>.json        (504 files)
        results.jsonl.bak_before_retry{2,3,4,5,6}     (legacy backups, left in place)

Target layout (v0.2.0):
    outputs/naive_v1/intern-s1/20260603_150000/
        results.jsonl
        results.json
        run_summary.json
        validation_report.json
        logs/<id>.json

Run from the project root. Idempotent: a second run prints "nothing to do"
and exits 0.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import List

# Fixed destination: the 504-problem run was made with naive_v1 / intern-s1
# on 2026-06-03 15:00 UTC. Updating these constants lets the script be reused
# for future migrations of the same shape.
PROMPT = "naive_v1"
MODEL = "intern-s1"
TS = "20260603_150000"


FILENAMES = (
    "results.jsonl",
    "results.json",
    "run_summary.json",
    "validation_report.json",
)
DIRS = ("logs",)


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1] if __doc__ else "")
    parser.add_argument("--src", type=Path, default=Path("outputs"),
                        help="Source flat output directory (default: outputs).")
    parser.add_argument("--dst-base", type=Path, default=Path("outputs"),
                        help="Root for the nested tree (default: outputs).")
    parser.add_argument("--yes", action="store_true",
                        help="Skip the interactive confirmation prompt.")
    args = parser.parse_args(argv)

    src = args.src
    if not src.is_dir():
        print(f"Source directory {src} does not exist; nothing to migrate.", file=sys.stderr)
        return 0

    dst = args.dst_base / PROMPT / MODEL / TS
    if dst.exists():
        print(f"Destination {dst} already exists; aborting (idempotent re-runs not supported).",
              file=sys.stderr)
        return 1

    present = [name for name in FILENAMES if (src / name).exists()]
    logs_dir = src / "logs"
    has_logs = logs_dir.is_dir() and any(logs_dir.iterdir())

    if not present and not has_logs:
        print(f"No migration candidates found in {src}; nothing to do.", file=sys.stderr)
        return 0

    print("Will migrate:")
    for name in present:
        print(f"  {src / name} → {dst / name}")
    if has_logs:
        print(f"  {logs_dir}/ → {dst / 'logs'}/")
    print(f"  (.bak_before_retry* files in {src} are left in place as legacy.)")

    if not args.yes:
        try:
            answer = input("Proceed? [y/N] ").strip().lower()
        except EOFError:
            answer = "n"
        if answer != "y":
            print("Aborted.", file=sys.stderr)
            return 1

    dst.mkdir(parents=True, exist_ok=True)
    moved_files: List[str] = []
    for name in present:
        shutil.move(str(src / name), str(dst / name))
        moved_files.append(str(dst / name))
    if has_logs:
        shutil.move(str(logs_dir), str(dst / "logs"))
        moved_files.append(str(dst / "logs"))

    print(f"\nMigrated {len(moved_files)} entries to {dst}.")
    print("Next: re-validate with `llm-math-evaluate --results .../results.jsonl --expected ...`")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
