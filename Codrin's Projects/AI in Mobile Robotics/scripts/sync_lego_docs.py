"""Fetch (or refresh) the LEGO(R) Education Python API documentation.

The upstream repo at https://github.com/LEGO/LEGOEducation is the authoritative
reference for every `legoeducation` API call made in this project. This script
vendors a local read-only copy into `reference/LEGOEducation/` so the docs can be
grepped offline while writing robot code.

The copy is gitignored on purpose: upstream stays upstream, and we never edit it.

Usage:
    python scripts/sync_lego_docs.py            # clone if missing, otherwise pull
    python scripts/sync_lego_docs.py --force    # delete and re-clone from scratch
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

REPO_URL = "https://github.com/LEGO/LEGOEducation.git"
REPO_ROOT = Path(__file__).resolve().parent.parent
TARGET = REPO_ROOT / "reference" / "LEGOEducation"


def run(args: list[str], cwd: Path | None = None) -> int:
    """Run a git command, echoing it first so failures are easy to diagnose."""
    print(f"$ {' '.join(args)}")
    return subprocess.call(args, cwd=str(cwd) if cwd else None)


def clone() -> int:
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    return run(["git", "clone", "--depth", "1", REPO_URL, str(TARGET)])


def pull() -> int:
    return run(["git", "pull", "--ff-only"], cwd=TARGET)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force",
        action="store_true",
        help="remove the existing copy and clone it again",
    )
    args = parser.parse_args()

    if shutil.which("git") is None:
        print("error: git is not on PATH; install git or download the docs manually from")
        print(f"       {REPO_URL}")
        return 1

    if args.force and TARGET.exists():
        print(f"removing {TARGET}")
        shutil.rmtree(TARGET, ignore_errors=True)

    if (TARGET / ".git").is_dir():
        code = pull()
    elif TARGET.exists() and any(TARGET.iterdir()):
        print(f"error: {TARGET} exists but is not a git clone; re-run with --force")
        return 1
    else:
        code = clone()

    if code != 0:
        print("error: could not reach GitHub. If you are offline, the LEGO API docs are")
        print("       unavailable -- do not guess at API signatures.")
        return code

    readme = TARGET / "README.md"
    if not readme.is_file():
        print(f"error: sync finished but {readme} is missing; the layout may have changed.")
        return 1

    docs = sorted(p.name for p in TARGET.glob("*.md"))
    print(f"\nLEGO Education docs ready at: {TARGET}")
    print(f"  {len(docs)} doc pages: {', '.join(docs)}")
    print(f"  examples: {TARGET / 'examples'}")
    print("\nFull API reference (grep this before writing any call):")
    print(f"  {TARGET / 'function_description.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
