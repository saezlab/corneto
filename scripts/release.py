#!/usr/bin/env python3
"""Create and optionally push a semver release tag.

Usage:
  python scripts/release.py major|minor|patch [--no-push] [--remote origin] [--dry-run]
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass
from typing import Iterable

SEMVER_RE = re.compile(r"^v?(\d+)\.(\d+)\.(\d+)(?:[.-].+)?$")


@dataclass(frozen=True)
class Version:
    major: int
    minor: int
    patch: int

    def bump(self, part: str) -> "Version":
        if part == "major":
            return Version(self.major + 1, 0, 0)
        if part == "minor":
            return Version(self.major, self.minor + 1, 0)
        if part == "patch":
            return Version(self.major, self.minor, self.patch + 1)
        raise ValueError(f"Unknown part: {part}")

    def __str__(self) -> str:
        return f"{self.major}.{self.minor}.{self.patch}"


def _run(cmd: list[str], *, capture: bool = False) -> str:
    if capture:
        return subprocess.check_output(cmd, text=True).strip()
    subprocess.check_call(cmd)
    return ""


def _iter_tags() -> Iterable[str]:
    raw = _run(["git", "tag", "--list", "--sort=-v:refname"], capture=True)
    for line in raw.splitlines():
        tag = line.strip()
        if tag:
            yield tag


def _parse_tag(tag: str) -> tuple[Version, bool] | None:
    m = SEMVER_RE.match(tag)
    if not m:
        return None
    version = Version(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    # Stable if tag is exactly vX.Y.Z or X.Y.Z
    stable = re.fullmatch(r"v?\d+\.\d+\.\d+", tag) is not None
    return version, stable


def _latest_version() -> tuple[Version, str | None, bool]:
    best_stable: tuple[Version, str] | None = None
    best_any: tuple[Version, str, bool] | None = None

    for tag in _iter_tags():
        parsed = _parse_tag(tag)
        if parsed is None:
            continue
        version, stable = parsed
        if stable:
            if best_stable is None or version > best_stable[0]:
                best_stable = (version, tag)
        if best_any is None or version > best_any[0]:
            best_any = (version, tag, stable)

    if best_stable is not None:
        version, tag = best_stable
        return version, tag, True
    if best_any is not None:
        version, tag, stable = best_any
        return version, tag, stable
    return Version(0, 0, 0), None, True


def _ensure_clean(allow_dirty: bool) -> None:
    status = _run(["git", "status", "--porcelain"], capture=True)
    if status and not allow_dirty:
        print(
            "Working tree is dirty. Commit or stash changes, or pass --allow-dirty.",
            file=sys.stderr,
        )
        sys.exit(2)


def main() -> int:
    parser = argparse.ArgumentParser(description="Create and optionally push a semver tag.")
    parser.add_argument("part", choices=["major", "minor", "patch"], help="Version component to bump.")
    parser.add_argument("--remote", default="origin", help="Git remote to push to. Default: origin")
    parser.add_argument("--no-push", action="store_true", help="Create the tag but do not push it.")
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="Allow tagging with a dirty working tree.",
    )
    parser.add_argument("--tag-prefix", default="v", help="Tag prefix. Default: v")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would happen without doing it.",
    )
    args = parser.parse_args()

    _ensure_clean(args.allow_dirty)

    latest, latest_tag, latest_is_stable = _latest_version()
    next_version = latest.bump(args.part)
    tag = f"{args.tag_prefix}{next_version}"

    if args.dry_run:
        if latest_tag is None:
            print("Latest tag: <none found>")
        else:
            stability = "stable" if latest_is_stable else "prerelease"
            print(f"Latest tag: {latest_tag} ({stability})")
        print(f"Latest version: v{latest}")
        print(f"Next tag: {tag}")
        print(f"Would run: git tag -a {tag} -m {tag}")
        if not args.no_push:
            print(f"Would run: git push {args.remote} {tag}")
        else:
            print("Would not push (--no-push).")
        return 0

    # Create annotated tag
    _run(["git", "tag", "-a", tag, "-m", tag])
    print(f"Created tag: {tag}")

    if not args.no_push:
        _run(["git", "push", args.remote, tag])
        print(f"Pushed tag to {args.remote}: {tag}")
    else:
        print("Tag not pushed (--no-push).")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
