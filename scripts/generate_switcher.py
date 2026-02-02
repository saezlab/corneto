#!/usr/bin/env python3
"""Generate PyData Sphinx theme switcher.json for deployed docs."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path


def _default_base_url() -> str:
    base_url = os.environ.get("DOCS_BASE_URL")
    if base_url:
        return base_url.rstrip("/")
    repo = os.environ.get("GITHUB_REPOSITORY", "saezlab/corneto")
    owner, name = repo.split("/", 1)
    return f"https://{owner}.github.io/{name}"


def _get_tags() -> list[str]:
    try:
        output = subprocess.check_output(
            ["git", "tag", "--list", "v*", "--sort=-v:refname"],
            text=True,
        )
    except subprocess.CalledProcessError:
        return []
    return [line.strip() for line in output.splitlines() if line.strip()]


def build_entries(base_url: str) -> list[dict[str, object]]:
    entries: list[dict[str, object]] = [
        {
            "name": "stable",
            "version": "stable",
            "url": f"{base_url}/stable/",
            "preferred": True,
        },
        {
            "name": "latest",
            "version": "latest",
            "url": f"{base_url}/latest/",
        },
    ]
    for tag in _get_tags():
        entries.append(
            {
                "name": tag,
                "version": tag,
                "url": f"{base_url}/{tag}/",
            }
        )
    return entries


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="Path to switcher.json")
    parser.add_argument(
        "--base-url",
        default=_default_base_url(),
        help="Docs base URL (e.g., https://org.github.io/repo)",
    )
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    entries = build_entries(args.base_url.rstrip("/"))
    output_path.write_text(json.dumps(entries, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
