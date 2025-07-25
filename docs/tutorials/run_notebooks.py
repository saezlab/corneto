#!/usr/bin/env python3
"""docs/tutorials/run_notebooks.py"""

import argparse
import subprocess
import sys
from pathlib import Path


def run(cmd, cwd: Path, dry_run: bool = False):
    """Run a command list in cwd, exiting on failure (or just print with --dry-run)."""
    print(f"> {' '.join(cmd)}  (cwd={cwd.name})")
    if not dry_run:
        subprocess.run(cmd, cwd=str(cwd), check=True)


def process_tutorial(proj: Path, dry_run: bool = False):
    """Install env, bootstrap tools, and execute notebooks in proj."""
    print(f"\n=== Processing tutorial: {proj.name} ===")
    # 1) Allow post-link scripts (Graphviz etc.)
    run(["pixi", "config", "set", "--local", "run-post-link-scripts", "insecure"], cwd=proj, dry_run=dry_run)
    # 2) Build/update env from pixi.toml
    run(["pixi", "install"], cwd=proj, dry_run=dry_run)
    # 3) Bootstrap kernel & notebook tools
    run(
        ["pixi", "run", "python", "-m", "pip", "install", "--upgrade", "ipykernel", "papermill", "nbclient"],
        cwd=proj,
        dry_run=dry_run,
    )
    # 4) Execute each notebook via papermill
    build_dir = proj / "build"
    if not dry_run:
        build_dir.mkdir(exist_ok=True)
    for nb in sorted(proj.glob("*.ipynb")):
        out = build_dir / nb.name
        run(["pixi", "run", "python", "-m", "papermill", str(nb), str(out)], cwd=proj, dry_run=dry_run)


def discover_tutorials(tutorials_dir: Path):
    """Find subdirs that look like tutorials (have pixi.toml and at least one .ipynb)."""
    return [
        p.resolve()
        for p in sorted(tutorials_dir.iterdir())
        if p.is_dir() and (p / "pixi.toml").is_file() and any(p.glob("*.ipynb"))
    ]


def parse_args():
    examples = r"""Examples:
  # Run ALL tutorials (default when you don't pass any names/paths)
  ./run_notebooks.py
  ./run_notebooks.py --all

  # Run a single tutorial by folder name (relative to this script's directory)
  ./run_notebooks.py carnival

  # Run by absolute or relative path
  ./run_notebooks.py ../other/path/to/tutorial

  # Run multiple tutorials at once
  ./run_notebooks.py carnival intro-to-foo ../bar/baz

  # Just list the tutorials the script can see (and exit)
  ./run_notebooks.py --list

  # Show commands but don't execute them
  ./run_notebooks.py carnival --dry-run
"""
    parser = argparse.ArgumentParser(
        description="Install and run notebooks inside tutorial directories using their Pixi environments.",
        epilog=examples,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "tutorials",
        nargs="*",
        help="Tutorial folder name(s) under this directory or path(s) to them. "
        "If omitted (and --all not provided), all tutorials are run.",
    )
    parser.add_argument(
        "-a",
        "--all",
        action="store_true",
        help="Run all tutorials (same behavior as omitting positional arguments).",
    )
    parser.add_argument(
        "-l",
        "--list",
        action="store_true",
        help="List discoverable tutorials and exit.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the commands that would be executed, but don't actually run anything.",
    )
    return parser.parse_args()


def main() -> int:
    tutorials_dir = Path(__file__).parent.resolve()
    args = parse_args()

    discovered = discover_tutorials(tutorials_dir)

    if args.list:
        if not discovered:
            print("No tutorials found.", file=sys.stderr)
            return 1
        print("Discovered tutorials:")
        for p in discovered:
            print(f"- {p}")
        return 0

    # Figure out which projects to run
    projects: list[Path] = []
    if args.all or not args.tutorials:
        projects = discovered
    else:
        for t in args.tutorials:
            cand = Path(t)
            if not cand.exists():
                # maybe it's a bare name under tutorials_dir
                cand = tutorials_dir / t
            if not (cand.exists() and (cand / "pixi.toml").is_file()):
                print(f"Error: tutorial '{t}' not found at '{cand}'.", file=sys.stderr)
                return 1
            projects.append(cand.resolve())

    if not projects:
        print("No tutorials found to process.", file=sys.stderr)
        return 1

    for proj in projects:
        process_tutorial(proj, dry_run=args.dry_run)

    print("\n✅ Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
