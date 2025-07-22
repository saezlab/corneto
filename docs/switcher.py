#!/usr/bin/env python3
"""Simplified switcher.json generator for Sphinx documentation.

This script generates a version switcher configuration for PyData Sphinx Theme
that aligns with the deployment folder structure used in CI/CD.
"""

import json
import os
import subprocess
from pathlib import Path


def get_base_url():
    """Get the base URL for documentation links."""
    url = os.environ.get("PAGE_URL")
    if url:
        return url.rstrip("/")

    repo = os.environ.get("GITHUB_REPOSITORY")
    if repo:
        user, project = repo.split("/", 1)
        return f"https://{user}.github.io/{project}"

    return "https://your-username.github.io/your-repo"


def get_git_tags():
    """Get sorted list of git tags (newest first)."""
    try:
        # Fetch tags from remote
        subprocess.run(["git", "fetch", "origin", "--tags", "--force"], capture_output=True, check=True)

        # Get tags sorted by date
        result = subprocess.run(["git", "tag", "--sort=-committerdate"], capture_output=True, text=True, check=True)
        return [tag.strip() for tag in result.stdout.splitlines() if tag.strip()]
    except subprocess.CalledProcessError:
        return []


def main():
    """Generate switcher.json for documentation version switching."""
    base_url = get_base_url()
    switcher = []

    # Get current deployment context
    is_ci = os.environ.get("GITHUB_ACTIONS") == "true"
    current_ref_name = os.environ.get("GITHUB_REF_NAME", "")

    # Add main/latest entry (always first)
    switcher.append({"name": "latest", "version": "main", "url": f"{base_url}/main/", "preferred": True})

    # Add dev entry if we're in CI or if dev exists
    if is_ci or current_ref_name == "dev":
        switcher.append({"name": "dev", "version": "dev", "url": f"{base_url}/dev/"})

    # Add version tags (only in CI to avoid local complexity)
    if is_ci:
        tags = get_git_tags()
        for tag in tags[:10]:  # Limit to 10 most recent tags
            switcher.append({"name": tag, "version": tag, "url": f"{base_url}/{tag}/"})

    # Write switcher.json
    script_dir = Path(__file__).parent
    output_path = script_dir / "switcher.json"

    with open(output_path, "w") as f:
        json.dump(switcher, f, indent=2)

    print(f"Generated switcher.json with {len(switcher)} entries at {output_path}")
    for entry in switcher:
        print(f"  - {entry['name']} -> {entry['url']}")


if __name__ == "__main__":
    main()
