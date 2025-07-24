#!/usr/bin/env python3
"""docs/tutorials/run_notebooks.py

Usage:
  # Run ALL tutorials
  python run_notebooks.py

  # Run a single tutorial by name (folder under this dir)
  python run_notebooks.py carnival

  # Run a single tutorial by full or relative path
  python run_notebooks.py ../other/path/to/tutorial
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run(cmd, cwd: Path):
    """Run a command list in cwd, exiting on failure."""
    print(f"> {' '.join(cmd)}  (cwd={cwd.name})")
    subprocess.run(cmd, cwd=str(cwd), check=True)


def process_tutorial(proj: Path):
    """Install env, bootstrap tools, and execute notebooks in proj."""
    print(f"\n=== Processing tutorial: {proj.name} ===")
    # 1) Allow post-link scripts (Graphviz etc.)
    run(["pixi", "config", "set", "--local", "run-post-link-scripts", "insecure"], cwd=proj)
    # 2) Build/update env from pixi.toml
    run(["pixi", "install"], cwd=proj)
    # 3) Bootstrap kernel & notebook tools
    run(["pixi", "run", "python", "-m", "pip", "install", "--upgrade", "ipykernel", "papermill", "nbclient"], cwd=proj)
    # 4) Execute each notebook via papermill
    build_dir = proj / "build"
    build_dir.mkdir(exist_ok=True)
    for nb in sorted(proj.glob("*.ipynb")):
        out = build_dir / nb.name
        run(["pixi", "run", "python", "-m", "papermill", str(nb), str(out)], cwd=proj)


def main():
    # Directory containing this script
    tutorials_dir = Path(__file__).parent.resolve()

    # Parse optional tutorial argument
    parser = argparse.ArgumentParser(description="Install and run tutorials via their Pixi envs")
    parser.add_argument("tutorial", nargs="?", help="Tutorial folder name under 'docs/tutorials' or path to it")
    args = parser.parse_args()

    # Build list of projects to process
    if args.tutorial:
        cand = Path(args.tutorial)
        # If user passed a bare name, look under tutorials_dir
        if not cand.exists():
            cand = tutorials_dir / args.tutorial
        if not (cand.exists() and (cand / "pixi.toml").is_file()):
            print(f"Error: tutorial '{args.tutorial}' not found at '{cand}'.", file=sys.stderr)
            sys.exit(1)
        projects = [cand.resolve()]
    else:
        # no arg: process all subdirs with pixi.toml
        projects = [
            p.resolve()
            for p in sorted(tutorials_dir.iterdir())
            if (p / "pixi.toml").is_file() and any(p.glob("*.ipynb"))
        ]

    if not projects:
        print("No tutorials found to process.", file=sys.stderr)
        sys.exit(1)

    for proj in projects:
        process_tutorial(proj)

    print("\n✅ Done.")


if __name__ == "__main__":
    main()
