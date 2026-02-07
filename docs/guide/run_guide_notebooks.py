#!/usr/bin/env python3
"""docs/guide/run_guide_notebooks.py"""

import argparse
import subprocess
import sys
from pathlib import Path


def run(cmd, cwd: Path, dry_run: bool = False):
    """Run a command list in cwd, exiting on failure (or just print with --dry-run)."""
    print(f"> {' '.join(cmd)}  (cwd={cwd.name})")
    if not dry_run:
        subprocess.run(cmd, cwd=str(cwd), check=True)


def discover_notebooks(guide_dir: Path) -> list[Path]:
    """Find all .ipynb under docs/guide (recursively)."""
    return sorted(p for p in guide_dir.rglob("*.ipynb") if p.is_file())


def parse_args():
    parser = argparse.ArgumentParser(
        description="Execute guide notebooks in-place using Papermill.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the commands that would be executed, but don't actually run anything.",
    )
    parser.add_argument(
        "--rewrite",
        action="store_true",
        help="Rewrite notebooks in place (required for execution).",
    )
    return parser.parse_args()


def main() -> int:
    guide_dir = Path(__file__).parent.resolve()
    args = parse_args()

    if not args.rewrite:
        print("Error: --rewrite is required to execute in-place.", file=sys.stderr)
        return 1

    notebooks = discover_notebooks(guide_dir)
    if not notebooks:
        print("No guide notebooks found.", file=sys.stderr)
        return 1

    for nb in notebooks:
        run(
            ["python", "-m", "papermill", str(nb), str(nb)],
            cwd=guide_dir,
            dry_run=args.dry_run,
        )

    print("\n✅ Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
