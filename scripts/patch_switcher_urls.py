#!/usr/bin/env python3
"""Patch switcher.json URL in published HTML files."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

SWITCHER_URL_RE = re.compile(r"(DOCUMENTATION_OPTIONS\.theme_switcher_json_url\s*=\s*')([^']*)(')")
SWITCHER_VERSION_RE = re.compile(r"(DOCUMENTATION_OPTIONS\.theme_switcher_version_match\s*=\s*')([^']*)(')")


def _infer_version_match(path: Path, root: Path) -> str | None:
    rel = path.relative_to(root).as_posix()
    parts = rel.split("/", 1)
    if not parts:
        return None
    top = parts[0]
    if top in {"stable", "latest"}:
        return top
    if top.startswith("v"):
        return top
    return None


def patch_file(path: Path, root: Path, new_url: str) -> bool:
    text = path.read_text(encoding="utf-8")
    if "DOCUMENTATION_OPTIONS.theme_switcher_json_url" not in text:
        return False
    new_text, url_count = SWITCHER_URL_RE.subn(rf"\1{new_url}\3", text)
    version_match = _infer_version_match(path, root)
    version_count = 0
    if version_match:
        new_text, version_count = SWITCHER_VERSION_RE.subn(rf"\1{version_match}\3", new_text)
    if url_count or version_count:
        path.write_text(new_text, encoding="utf-8")
        return True
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="Root folder to patch")
    parser.add_argument("--new-url", required=True, help="New switcher.json URL")
    args = parser.parse_args()

    root = Path(args.root)
    new_url = args.new_url
    changed = 0
    for path in root.rglob("*.html"):
        if patch_file(path, root, new_url):
            changed += 1

    print(f"Updated switcher URL in {changed} HTML file(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
