#!/usr/bin/env python3
"""Sync GitHub releases to Sphinx documentation.

This script fetches releases from the GitHub API and generates corresponding
Markdown files in the docs/releases/ directory for Sphinx documentation.
"""

import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from urllib.error import HTTPError
from urllib.request import Request, urlopen


def get_github_releases(repo: str, token: Optional[str] = None) -> List[Dict]:
    """Fetch releases from GitHub API."""
    url = f"https://api.github.com/repos/{repo}/releases"

    headers = {
        "Accept": "application/vnd.github.v3+json",
        "User-Agent": "corneto-sync-releases/1.0",
    }

    if token:
        headers["Authorization"] = f"Bearer {token}"

    try:
        request = Request(url, headers=headers)
        with urlopen(request) as response:
            return json.loads(response.read().decode())
    except HTTPError as e:
        print(f"Error fetching releases: {e}")
        return []


def clean_release_body(body: str) -> str:
    """Clean and format release body for Sphinx."""
    if not body:
        return ""

    # Remove HTML comments
    body = re.sub(r"<!--.*?-->", "", body, flags=re.DOTALL)

    # Fix relative links to be absolute GitHub links
    # This handles [text](../path) or [text](/path) patterns
    repo_base = "https://github.com/saezlab/corneto"
    body = re.sub(r"\[([^\]]+)\]\(\.\.\/([^)]+)\)", rf"[\1]({repo_base}/\2)", body)
    body = re.sub(r"\[([^\]]+)\]\(\/([^)]+)\)", rf"[\1]({repo_base}/\2)", body)

    # Ensure proper spacing around headers
    body = re.sub(r"\n(#{1,6}\s)", r"\n\n\1", body)
    body = re.sub(r"^(#{1,6}\s)", r"\1", body)  # Don't add space at start

    return body.strip()


def format_release_date(published_at: str) -> str:
    """Format release date for display."""
    try:
        dt = datetime.fromisoformat(published_at.replace("Z", "+00:00"))
        return dt.strftime("%B %d, %Y")
    except (ValueError, AttributeError):
        return published_at


def generate_release_markdown(release: Dict) -> str:
    """Generate Sphinx-compatible markdown for a release."""
    tag_name = release["tag_name"]
    name = release["name"] or tag_name
    published_at = format_release_date(release["published_at"])
    body = clean_release_body(release["body"])
    html_url = release["html_url"]

    # Determine if it's a pre-release
    is_prerelease = release.get("prerelease", False)
    prerelease_badge = " {bdg-warning}`Pre-release`" if is_prerelease else ""

    # Build the markdown content
    content = f"""# Release {tag_name}{prerelease_badge}

**Release Date**: {published_at}

"""

    if body:
        content += f"{body}\n\n"
    else:
        content += "No release notes available for this version.\n\n"

    content += f"""---

## Links

- [📦 PyPI Package](https://pypi.org/project/corneto/{tag_name.lstrip("v")}/)
- [🏷️ GitHub Release]({html_url})
- [📚 Documentation](https://saezlab.github.io/corneto/)

## Installation

```bash
pip install corneto=={tag_name.lstrip("v")}
```
"""

    return content


def update_releases_index(releases: List[Dict], releases_dir: Path) -> None:
    """Update the releases index.md file."""
    # Sort releases by version (newest first)
    sorted_releases = sorted(releases, key=lambda x: x["published_at"], reverse=False)

    content = """# Release Notes

This section contains detailed release notes for CORNETO versions, documenting new features, improvements, bug fixes, and breaking changes.

## Recent Releases

```{toctree}
:maxdepth: 1
:reversed:

"""

    # Add each release to the toctree
    for release in sorted_releases:
        tag_name = release["tag_name"]
        content += f"{tag_name}\n"

    content += """```

## Release Schedule

CORNETO follows [semantic versioning](https://semver.org/) with the following release types:

- **Major releases** (x.0.0): Breaking changes and significant new features
- **Minor releases** (x.y.0): New features and improvements, backward compatible
- **Patch releases** (x.y.z): Bug fixes and small improvements
- **Pre-releases** (x.y.z-alpha/beta/rc): Testing versions before stable releases

## Getting Release Updates

- **GitHub Releases**: Follow releases on the [GitHub repository](https://github.com/saezlab/corneto/releases)
- **PyPI**: Install the latest version with `pip install --upgrade corneto`
- **Development**: Track development progress on the `dev` branch

## Contributing to Releases

For information on contributing to CORNETO and the release process, see our [contributing guidelines](https://github.com/saezlab/corneto/blob/main/CONTRIBUTING.md) and [release documentation](https://github.com/saezlab/corneto/blob/main/RELEASE.md).
"""

    index_file = releases_dir / "index.md"
    with open(index_file, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"Updated {index_file}")


def sync_releases_to_docs(repo: str, docs_dir: str = "docs", token: Optional[str] = None) -> None:
    """Main function to sync GitHub releases to Sphinx docs."""
    print(f"Syncing releases from {repo} to {docs_dir}/releases/")

    # Fetch releases from GitHub
    releases = get_github_releases(repo, token)
    if not releases:
        print("No releases found or error fetching releases")
        return

    print(f"Found {len(releases)} releases")

    # Setup directories
    docs_path = Path(docs_dir)
    releases_dir = docs_path / "releases"
    releases_dir.mkdir(exist_ok=True)

    # Generate markdown files for each release
    for release in releases:
        tag_name = release["tag_name"]
        release_file = releases_dir / f"{tag_name}.md"

        content = generate_release_markdown(release)

        with open(release_file, "w", encoding="utf-8") as f:
            f.write(content)

        print(f"Generated {release_file}")

    # Update the index file
    update_releases_index(releases, releases_dir)

    print("✅ Release sync completed successfully!")


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Sync GitHub releases to Sphinx docs")
    parser.add_argument("--repo", default="saezlab/corneto", help="GitHub repository (owner/name)")
    parser.add_argument("--docs-dir", default="docs", help="Documentation directory path")
    parser.add_argument(
        "--token",
        help="GitHub token for API access (optional, uses GITHUB_TOKEN env var if not provided)",
    )

    args = parser.parse_args()

    # Get token from args or environment
    token = args.token or os.environ.get("GITHUB_TOKEN")

    sync_releases_to_docs(args.repo, args.docs_dir, token)


if __name__ == "__main__":
    main()
