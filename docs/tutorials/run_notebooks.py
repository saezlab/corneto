#!/usr/bin/env python3
"""docs/tutorials/run_notebooks.py"""

import argparse
import hashlib
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _notebook_execution import (
    check_requirements,
    execute_notebook,
    execution_timeout_for_notebook,
    pixi_environment,
    python_for_pixi,
    requirements_for_notebook,
)


def find_repo_root(start: Path) -> Path:
    """Find repo root by walking up until pyproject.toml is found."""
    for parent in [start, *start.parents]:
        if (parent / "pyproject.toml").is_file():
            return parent
    raise FileNotFoundError(
        "Could not find pyproject.toml when locating repo root. Run from the CORNETO repo or pass an explicit path."
    )


def run(cmd, cwd: Path, dry_run: bool = False):
    """Run a command list in cwd, exiting on failure (or just print with --dry-run)."""
    print(f"> {' '.join(cmd)}  (cwd={cwd.name})")
    if not dry_run:
        subprocess.run(cmd, cwd=str(cwd), check=True)


def process_tutorial(
    proj: Path,
    dry_run: bool = False,
    rewrite: bool = False,
    editable_corneto: bool = False,
    corneto_root: Path | None = None,
):
    """Install the environment and execute notebooks in its Python kernel."""
    print(f"\n=== Processing tutorial: {proj.name} ===")
    # 1) Allow post-link scripts (Graphviz etc.)
    run(["pixi", "config", "set", "--local", "run-post-link-scripts", "insecure"], cwd=proj, dry_run=dry_run)
    # 2) Build/update env from pixi.toml
    run(["pixi", "install"], cwd=proj, dry_run=dry_run)
    # 2b) Optionally install local CORNETO source (editable) into the env
    if editable_corneto:
        root = corneto_root or find_repo_root(Path(__file__).resolve())
        run(
            ["pixi", "run", "python", "-m", "pip", "install", "-e", str(root)],
            cwd=proj,
            dry_run=dry_run,
        )
    notebooks = sorted(proj.glob("*.ipynb"))
    notebook_requirements = {nb: requirements_for_notebook(nb) for nb in notebooks}
    requirements = {item for items in notebook_requirements.values() for item in items}

    if dry_run:
        print(f"> pixi run python -c 'import sys; print(sys.executable)'  (cwd={proj.name})")
        if requirements:
            print(f"> check notebook requirements in {proj.name}: {', '.join(sorted(requirements))}")
        for nb in notebooks:
            out = nb if rewrite else (proj / "build" / nb.name)
            print(f"> execute {nb.name} -> {out.name}  (cwd={nb.parent.name})")
        return

    # Use the project interpreter for dependency checks and as the Jupyter kernel.
    python = python_for_pixi(proj)
    activation_environment = pixi_environment(proj)
    requirement_results = (
        check_requirements(requirements, python, proj, environment=activation_environment) if requirements else {}
    )

    # The runner lives in the docs environment; kernelspecs point directly at this
    # tutorial's Pixi Python, so notebook imports match the tutorial manifest.
    build_dir = proj / "build"
    if not rewrite:
        build_dir.mkdir(exist_ok=True)
    kernel_name = "corneto-" + hashlib.sha256(str(proj).encode()).hexdigest()[:12]
    with tempfile.TemporaryDirectory(prefix="corneto-tutorial-kernels-") as kernels:
        for nb in notebooks:
            unavailable = [
                requirement
                for requirement in notebook_requirements[nb]
                if requirement_results.get(requirement) is not None
            ]
            if unavailable:
                details = "; ".join(f"{name}: {requirement_results[name]}" for name in unavailable)
                print(f"SKIPPED {nb.name}: {details}")
                continue

            out = nb if rewrite else (build_dir / nb.name)
            execute_notebook(
                nb,
                out,
                python=python,
                cwd=nb.parent,
                kernel_root=Path(kernels),
                kernel_name=kernel_name,
                timeout=execution_timeout_for_notebook(nb),
                environment=activation_environment,
            )
            print(f"EXECUTED {nb.name} -> {out.relative_to(proj)}")


def discover_tutorials(tutorials_dir: Path):
    """Find subdirs that look like tutorials (have pixi.toml and at least one .ipynb)."""
    return sorted(
        config.parent.resolve()
        for config in tutorials_dir.rglob("pixi.toml")
        if "build" not in config.parts
        and not any(part.startswith(".") for part in config.relative_to(tutorials_dir).parts)
        and any(config.parent.glob("*.ipynb"))
    )


def parse_args():
    """Parse tutorial-runner command-line arguments."""
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
    parser.add_argument(
        "--rewrite",
        action="store_true",
        help="Rewrite notebooks in place instead of writing to build/.",
    )
    parser.add_argument(
        "--editable-corneto",
        action="store_true",
        help="Install the local CORNETO source (editable) into each tutorial env.",
    )
    parser.add_argument(
        "--corneto-root",
        type=Path,
        help="Path to the CORNETO repo root (used with --editable-corneto).",
    )
    return parser.parse_args()


def main() -> int:
    """Execute selected tutorial projects and return a process status."""
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
        process_tutorial(
            proj,
            dry_run=args.dry_run,
            rewrite=args.rewrite,
            editable_corneto=args.editable_corneto,
            corneto_root=args.corneto_root,
        )

    print("\n✅ Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
