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
        raise ReleaseError(
            f"Command failed: {' '.join(cmd)}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
    return result.stdout.strip()


def _normalize_version(raw: str) -> str:
    version = raw.strip()
    if not version.startswith("v"):
        version = f"v{version}"
    if not VERSION_RE.match(version):
        raise ReleaseError(
            "Invalid version format. Use vX.Y.Z or "
            "vX.Y.Z-(alpha|beta|rc).N (example: v1.0.0-beta.4)."
        )
    return version


def _ensure_clean_tree() -> None:
    status = _run(["git", "status", "--porcelain"], check=True)
    if status:
        raise ReleaseError("Working tree is not clean. Commit/stash changes first.")


def _ensure_on_main() -> None:
    branch = _run(["git", "branch", "--show-current"], check=True)
    if branch != "main":
        raise ReleaseError(f"Current branch is '{branch}', expected 'main'.")


def _ensure_remote_exists(remote: str) -> None:
    if remote.startswith("-"):
        raise ReleaseError(f"Invalid remote name: {remote}")
    _run(["git", "remote", "get-url", remote], check=True)


def _ensure_up_to_date_with_remote_main(remote: str) -> None:
    _run(["git", "fetch", remote, "main", "dev"], check=True)
    head = _run(["git", "rev-parse", "HEAD"], check=True)
    remote_main = _run(["git", "rev-parse", f"{remote}/main"], check=True)
    if head != remote_main:
        raise ReleaseError(
            f"HEAD is not at {remote}/main. Pull main after merging dev -> main."
        )


def _ensure_dev_is_merged(remote: str) -> None:
    # `merge-base --is-ancestor A B` succeeds when A is reachable from B.
    result = subprocess.run(
        ["git", "merge-base", "--is-ancestor", f"{remote}/dev", "HEAD"],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise ReleaseError(
            f"{remote}/dev is not merged into current main commit yet. "
            "Merge dev -> main before releasing."
        )


def _ensure_tag_does_not_exist(version: str, remote: str) -> None:
    local_tags = _run(["git", "tag", "--list", version], check=True)
    if local_tags:
        raise ReleaseError(f"Tag already exists locally: {version}")

    remote_tags = _run(["git", "ls-remote", "--tags", remote, version], check=True)
    if remote_tags:
        raise ReleaseError(f"Tag already exists on {remote}: {version}")


def _create_and_push_tag(version: str, remote: str) -> None:
    _run(["git", "tag", "-a", version, "-m", version], check=True)
    _run(["git", "push", remote, version], check=True)


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    """Parse release command-line arguments."""
    parser = argparse.ArgumentParser(
        prog="release",
        description=(
            "Create and push an annotated release tag with safety checks. "
            "Example: poetry run release v1.0.0-beta.4"
        ),
    )
    parser.add_argument("version", help="Release version tag (with or without 'v').")
    parser.add_argument(
        "--remote",
        default="origin",
        help="Git remote containing the release main/dev branches (default: origin).",
    )
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
    """Validate release state, then create and push the requested tag."""
    args = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        version = _normalize_version(args.version)
        _ensure_remote_exists(args.remote)
        _ensure_clean_tree()
        _ensure_on_main()
        _ensure_up_to_date_with_remote_main(args.remote)
        _ensure_dev_is_merged(args.remote)
        _ensure_tag_does_not_exist(version, args.remote)

        if args.dry_run:
            print(f"[dry-run] All checks passed. Would create and push tag: {version}")
            return 0

        if not args.yes:
            answer = input(f"Create and push release tag {version}? [y/N]: ").strip()
            if answer.lower() not in {"y", "yes"}:
                print("Aborted.")
                return 1

        _create_and_push_tag(version, args.remote)
        print(f"Release tag pushed to {args.remote}: {version}")
        print("GitHub Actions will now build, publish, and create the GitHub release.")
        print("Post-release sync reminder:")
        print("  git checkout dev")
        print(f"  git pull --ff-only {args.remote} dev")
        print(f"  git merge --ff-only {args.remote}/main")
        print(f"  git push {args.remote} dev")
        print(
            "This keeps dev aligned with the latest release tag ancestry "
            "for dynamic versioning."
        )
        return 0
    except ReleaseError as exc:
        print(f"Release aborted: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
