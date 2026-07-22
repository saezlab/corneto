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
    """Find the indexed guide notebooks, excluding generated output."""
    indexed_sections = {"intro", "networks", "metabolism", "signaling", "interoperability"}
    return sorted(
        p
        for p in guide_dir.rglob("*.ipynb")
        if p.is_file()
        and p.relative_to(guide_dir).parts[0] in indexed_sections
        and not any(part.startswith((".", "_")) for part in p.relative_to(guide_dir).parts)
    )


def parse_args():
    """Parse guide-runner command-line arguments."""
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
        help="Rewrite notebooks in place instead of writing to guide/build/.",
    )
    parser.add_argument(
        "--start-at",
        type=Path,
        help="Resume at this path relative to docs/guide (inclusive).",
    )
    return parser.parse_args()


def main() -> int:
    """Execute the indexed guide notebooks and return a process status."""
    guide_dir = Path(__file__).parent.resolve()
    args = parse_args()

    notebooks = discover_notebooks(guide_dir)
    if args.start_at is not None:
        start_path = (guide_dir / args.start_at).resolve()
        if start_path not in notebooks:
            print(f"Error: indexed guide notebook not found: {args.start_at}", file=sys.stderr)
            return 1
        notebooks = notebooks[notebooks.index(start_path) :]
    if not notebooks:
        print("No guide notebooks found.", file=sys.stderr)
        return 1

    for nb in notebooks:
        if args.rewrite:
            output = nb
        else:
            output = guide_dir / "build" / nb.relative_to(guide_dir)
            output.parent.mkdir(parents=True, exist_ok=True)
        run(
            [sys.executable, "-m", "papermill", str(nb), str(output)],
            cwd=guide_dir,
            dry_run=args.dry_run,
        )

    print("\n✅ Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
