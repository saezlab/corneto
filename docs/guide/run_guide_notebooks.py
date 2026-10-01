#!/usr/bin/env python3
"""docs/guide/run_guide_notebooks.py"""

import argparse
import hashlib
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _notebook_execution import (
    check_requirements,
    execute_notebook,
    execution_timeout_for_notebook,
    requirements_for_notebook,
)


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

    notebook_requirements = {nb: requirements_for_notebook(nb) for nb in notebooks}
    requirements = {item for items in notebook_requirements.values() for item in items}
    if args.dry_run:
        if requirements:
            print(f"> check notebook requirements in the main environment: {', '.join(sorted(requirements))}")
        for nb in notebooks:
            output = nb if args.rewrite else (guide_dir / "build" / nb.relative_to(guide_dir))
            print(f"> execute {nb.relative_to(guide_dir)} -> {output.relative_to(guide_dir)}")
        return 0

    environment = os.environ.copy()
    requirement_results = (
        check_requirements(requirements, sys.executable, guide_dir, environment=environment) if requirements else {}
    )
    kernel_name = "corneto-guide-" + hashlib.sha256(str(guide_dir).encode()).hexdigest()[:10]
    with tempfile.TemporaryDirectory(prefix="corneto-guide-kernels-") as kernels:
        for nb in notebooks:
            unavailable = [
                requirement
                for requirement in notebook_requirements[nb]
                if requirement_results.get(requirement) is not None
            ]
            if unavailable:
                details = "; ".join(f"{name}: {requirement_results[name]}" for name in unavailable)
                print(f"SKIPPED {nb.relative_to(guide_dir)}: {details}")
                continue

            if args.rewrite:
                output = nb
            else:
                output = guide_dir / "build" / nb.relative_to(guide_dir)
                output.parent.mkdir(parents=True, exist_ok=True)
            execute_notebook(
                nb,
                output,
                python=Path(sys.executable),
                cwd=nb.parent,
                kernel_root=Path(kernels),
                kernel_name=kernel_name,
                timeout=execution_timeout_for_notebook(nb),
                environment=environment,
            )
            print(f"EXECUTED {nb.relative_to(guide_dir)}")

    print("\n✅ Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
