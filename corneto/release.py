"""Simple guarded release CLI for tag-based publishing."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from typing import Sequence

VERSION_RE = re.compile(r"^v\d+\.\d+\.\d+(?:-(?:alpha|beta|rc)\.\d+)?$")


class ReleaseError(RuntimeError):
    """Raised when a release precondition is not met."""


def _run(cmd: Sequence[str], *, check: bool = True) -> str:
    result = subprocess.run(
        cmd,
        text=True,
        capture_output=True,
        check=False,
    )
    if check and result.returncode != 0:
        raise ReleaseError(f"Command failed: {' '.join(cmd)}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}")
    return result.stdout.strip()


def _normalize_version(raw: str) -> str:
    version = raw.strip()
    if not version.startswith("v"):
        version = f"v{version}"
    if not VERSION_RE.match(version):
        raise ReleaseError("Invalid version format. Use vX.Y.Z or vX.Y.Z-(alpha|beta|rc).N (example: v1.0.0-beta.4).")
    return version


def _ensure_clean_tree() -> None:
    status = _run(["git", "status", "--porcelain"], check=True)
    if status:
        raise ReleaseError("Working tree is not clean. Commit/stash changes first.")


def _ensure_on_main() -> None:
    branch = _run(["git", "branch", "--show-current"], check=True)
    if branch != "main":
        raise ReleaseError(f"Current branch is '{branch}', expected 'main'.")


def _ensure_up_to_date_with_origin_main() -> None:
    _run(["git", "fetch", "origin", "main", "dev"], check=True)
    head = _run(["git", "rev-parse", "HEAD"], check=True)
    origin_main = _run(["git", "rev-parse", "origin/main"], check=True)
    if head != origin_main:
        raise ReleaseError("HEAD is not at origin/main. Pull main after merging dev -> main.")


def _ensure_dev_is_merged() -> None:
    # `merge-base --is-ancestor A B` succeeds when A is reachable from B.
    result = subprocess.run(
        ["git", "merge-base", "--is-ancestor", "origin/dev", "HEAD"],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise ReleaseError("origin/dev is not merged into current main commit yet. Merge dev -> main before releasing.")


def _ensure_tag_does_not_exist(version: str) -> None:
    local_tags = _run(["git", "tag", "--list", version], check=True)
    if local_tags:
        raise ReleaseError(f"Tag already exists locally: {version}")

    remote_tags = _run(["git", "ls-remote", "--tags", "origin", version], check=True)
    if remote_tags:
        raise ReleaseError(f"Tag already exists on origin: {version}")


def _create_and_push_tag(version: str) -> None:
    _run(["git", "tag", "-a", version, "-m", version], check=True)
    _run(["git", "push", "origin", version], check=True)


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="release",
        description=(
            "Create and push an annotated release tag with safety checks. Example: poetry run release v1.0.0-beta.4"
        ),
    )
    parser.add_argument("version", help="Release version tag (with or without 'v').")
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip confirmation prompt.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate checks and print intended tag without creating/pushing.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        version = _normalize_version(args.version)
        _ensure_clean_tree()
        _ensure_on_main()
        _ensure_up_to_date_with_origin_main()
        _ensure_dev_is_merged()
        _ensure_tag_does_not_exist(version)

        if args.dry_run:
            print(f"[dry-run] All checks passed. Would create and push tag: {version}")
            return 0

        if not args.yes:
            answer = input(f"Create and push release tag {version}? [y/N]: ").strip()
            if answer.lower() not in {"y", "yes"}:
                print("Aborted.")
                return 1

        _create_and_push_tag(version)
        print(f"Release tag pushed: {version}")
        print("GitHub Actions will now build, publish, and create the GitHub release.")
        return 0
    except ReleaseError as exc:
        print(f"Release aborted: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
