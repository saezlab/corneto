#!/usr/bin/env python3
"""Sync GitHub releases to Sphinx documentation.

This script fetches releases from the GitHub API and generates corresponding
Markdown files in the docs/releases/ directory for Sphinx documentation.

Changes vs. previous version
----------------------------
- **Sorting behavior**: By default, the releases index is sorted by **version number**
  (semantic-style parsing) with newest versions first. An optional CLI flag
  allows sorting by **date** instead.
- **Robust version parsing**: Handles tags like `v1.2.3`, `1.2.3`, and common
  prerelease labels (alpha, beta, rc). If a tag can't be parsed, it falls back
  to a reasonable numeric extraction.
- **Stable > prerelease** ordering.
- Removed reliance on ``:reversed:`` in the toctree; we now output in the
  desired order directly.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.error import HTTPError
from urllib.request import Request, urlopen

# ------------------------------
# GitHub API
# ------------------------------


def get_github_releases(repo: str, token: Optional[str] = None) -> List[Dict]:
    """Fetch releases from GitHub API.

    Notes:
    -----
    - This fetches the first page of releases. If you have more than 30 releases,
      consider adding basic pagination.
    """
    url = f"https://api.github.com/repos/{repo}/releases"

    headers = {
        "Accept": "application/vnd.github.v3+json",
        "User-Agent": "corneto-sync-releases/1.0",
    }

    if token:
        headers["Authorization"] = f"Bearer {token}"

    try:
        request = Request(url, headers=headers)
        with urlopen(request) as response:  # nosec - GitHub API endpoint
            return json.loads(response.read().decode())
    except HTTPError as e:
        print(f"Error fetching releases: {e}")
        return []


# ------------------------------
# Utilities
# ------------------------------

_PRERELEASE_ORDER = {
    "a": -3,
    "alpha": -3,
    "b": -2,
    "beta": -2,
    "rc": -1,
}

_version_re = re.compile(
    r"^\s*v?(?P<major>\d+)\.(?P<minor>\d+)\.(?P<patch>\d+)"
    r"(?:[.-]?(?P<label>[A-Za-z]+)[.-]?(?P<labelnum>\d+)?)?\s*$"
)


def parse_version_key(tag: str) -> Tuple[int, int, int, int, int]:
    """Convert a tag (e.g., 'v1.2.3', '1.2.3-rc1') to a sortable key.

    The returned tuple sorts correctly with **descending** order to get latest first
    when used with ``reverse=True``. Stable releases are considered newer than their
    prerelease counterparts of the same numeric version.

    Structure: ``(major, minor, patch, pre_weight, pre_num)``
    where ``pre_weight`` is 0 for stable, negative for prerelease (alpha < beta < rc).
    """
    s = tag.strip()

    m = _version_re.match(s)
    if m:
        major = int(m.group("major"))
        minor = int(m.group("minor"))
        patch = int(m.group("patch"))
        label = (m.group("label") or "").lower()
        labelnum = int(m.group("labelnum") or 0)
        pre_weight = _PRERELEASE_ORDER.get(label, 0 if not label else -4)
        return (major, minor, patch, pre_weight, labelnum)

    # Fallback: extract up to 3 numeric components; treat as stable
    nums = [int(x) for x in re.findall(r"\d+", s)]
    nums = (nums + [0, 0, 0])[:3]
    return (nums[0], nums[1], nums[2], 0, 0)


def safe_filename_from_tag(tag: str) -> str:
    """Return a safe filename for a given tag name.

    Preserves common version characters while replacing others with underscores.
    """
    return re.sub(r"[^A-Za-z0-9._-]", "_", tag)


def clean_release_body(body: Optional[str]) -> str:
    """Clean and format release body for Sphinx."""
    if not body:
        return ""

    # Remove HTML comments
    body = re.sub(r"<!--.*?-->", "", body, flags=re.DOTALL)

    # Fix relative links to be absolute GitHub links
    # This handles [text](../path) or [text](/path) patterns
    repo_base = "https://github.com/saezlab/corneto"
    body = re.sub(r"\[([^\]]+)\]\(\.\.\/([^)]+)\)", rf"[\\1]({repo_base}/\\2)", body)
    body = re.sub(r"\[([^\]]+)\]\(\/([^)]+)\)", rf"[\\1]({repo_base}/\\2)", body)

    # Ensure proper spacing around headers
    body = re.sub(r"\n(#{1,6}\s)", r"\n\n\\1", body)
    body = re.sub(r"^(#{1,6}\s)", r"\\1", body)  # Don't add space at start

    return body.strip()


def format_release_date(published_at: Optional[str]) -> str:
    """Format release date for display."""
    try:
        dt = datetime.fromisoformat((published_at or "").replace("Z", "+00:00"))
        return dt.strftime("%B %d, %Y")
    except (ValueError, AttributeError):
        return published_at or ""


# ------------------------------
# Rendering
# ------------------------------


def generate_release_markdown(release: Dict) -> str:
    """Generate Sphinx-compatible markdown for a release."""
    tag_name = release["tag_name"]
    name = release.get("name") or tag_name
    published_at = format_release_date(release.get("published_at"))
    body = clean_release_body(release.get("body"))
    html_url = release.get("html_url", "")

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

    # Normalize tag for PyPI and pip install (drop leading 'v' if present)
    pypi_version = tag_name.lstrip("vV")

    content += f"""---

## Links

- [📦 PyPI Package](https://pypi.org/project/corneto/{pypi_version}/)
- [🏷️ GitHub Release]({html_url})
- [📚 Documentation](https://saezlab.github.io/corneto/)

## Installation

```bash
pip install corneto=={pypi_version}
```
"""

    return content


def sort_releases(releases: List[Dict], sort_by: str = "version") -> List[Dict]:
    """Return releases sorted newest-first by version (default) or date.

    Parameters
    ----------
    releases : list of GitHub release dicts
    sort_by : {'version', 'date'}
        - 'version' (default): sort by tag name as a version number.
        - 'date'            : sort by 'published_at' timestamp.
    """
    if sort_by == "date":

        def date_key(r: Dict) -> float:
            ts = r.get("published_at") or ""
            try:
                return datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp()
            except Exception:
                return 0.0

        return sorted(releases, key=date_key, reverse=True)  # newest first

    # Default: version sort (newest version first)
    return sorted(releases, key=lambda r: parse_version_key(r.get("tag_name", "0.0.0")), reverse=True)


def update_releases_index(releases: List[Dict], releases_dir: Path, *, sort_by: str = "version") -> None:
    """Update the releases index.md file.

    The toctree is written in the desired order (newest first). We do not rely on
    the ``:reversed:`` option.
    """
    sorted_releases = sort_releases(releases, sort_by=sort_by)

    content = """# Release Notes

This section contains detailed release notes for CORNETO versions, documenting new features, improvements, bug fixes, and breaking changes.

## Recent Releases

```{toctree}
:maxdepth: 1

"""

    # Add each release to the toctree in already-sorted order
    for release in sorted_releases:
        tag_name = release["tag_name"]
        content += f"{safe_filename_from_tag(tag_name)}\n"

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


# ------------------------------
# Sync process
# ------------------------------


def sync_releases_to_docs(
    repo: str,
    docs_dir: str = "docs",
    token: Optional[str] = None,
    *,
    sort_by: str = "version",
) -> None:
    """Main function to sync GitHub releases to Sphinx docs."""
    print(f"Syncing releases from {repo} to {docs_dir}/releases/ (sort_by={sort_by})")

    # Fetch releases from GitHub
    releases = get_github_releases(repo, token)
    if not releases:
        print("No releases found or error fetching releases")
        return

    print(f"Found {len(releases)} releases")

    # Setup directories
    docs_path = Path(docs_dir)
    releases_dir = docs_path / "releases"
    releases_dir.mkdir(parents=True, exist_ok=True)

    # Generate markdown files for each release (file order doesn't matter)
    for release in releases:
        tag_name = release["tag_name"]
        filename = safe_filename_from_tag(tag_name)
        release_file = releases_dir / f"{filename}.md"

        content = generate_release_markdown(release)

        with open(release_file, "w", encoding="utf-8") as f:
            f.write(content)

        print(f"Generated {release_file}")

    # Update the index file in the requested sort order
    update_releases_index(releases, releases_dir, sort_by=sort_by)

    print("✅ Release sync completed successfully!")


# ------------------------------
# CLI
# ------------------------------


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Sync GitHub releases to Sphinx docs")
    parser.add_argument("--repo", default="saezlab/corneto", help="GitHub repository (owner/name)")
    parser.add_argument("--docs-dir", default="docs", help="Documentation directory path")
    parser.add_argument(
        "--token",
        help="GitHub token for API access (optional, uses GITHUB_TOKEN env var if not provided)",
    )
    parser.add_argument(
        "--sort-by",
        choices=("version", "date"),
        default="version",
        help="Sort releases in the index by 'version' (default) or by 'date' (published_at)",
    )

    args = parser.parse_args()

    # Get token from args or environment
    token = args.token or os.environ.get("GITHUB_TOKEN")

    sync_releases_to_docs(args.repo, args.docs_dir, token, sort_by=args.sort_by)


if __name__ == "__main__":
    main()
