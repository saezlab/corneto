#!/usr/bin/env python3
"""Prepare a checked-out gh-pages tree for one documentation deployment."""

from __future__ import annotations

import argparse
import json
import re
import shutil
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlparse

try:
    from .generate_switcher import build_entries
except ImportError:  # pragma: no cover - direct script execution
    from generate_switcher import build_entries


VERSION_FOLDER_RE = re.compile(r"^(?:stable|v[0-9A-Za-z][0-9A-Za-z._-]*)$")
CSS_URL_RE = re.compile(r"url\(\s*(['\"]?)(.*?)\1\s*\)", re.IGNORECASE)
SWITCHER_URL_RE = re.compile(r"DOCUMENTATION_OPTIONS\.theme_switcher_json_url\s*=\s*(['\"])(?P<value>.*?)\1")
SWITCHER_VERSION_RE = re.compile(r"DOCUMENTATION_OPTIONS\.theme_switcher_version_match\s*=\s*(['\"])(?P<value>.*?)\1")
ASSET_ATTRIBUTES = {"src", "poster"}
LANDING_HOST = "corneto.org"


class _LandingReferenceParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.asset_references: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        if tag == "meta" and attributes.get("property") == "og:image" and attributes.get("content"):
            self.asset_references.append(attributes["content"])
        if tag == "meta" and attributes.get("name") == "twitter:image" and attributes.get("content"):
            self.asset_references.append(attributes["content"])
        for name, value in attrs:
            if not value:
                continue
            if name in ASSET_ATTRIBUTES:
                self.asset_references.append(value)
            elif name == "srcset":
                self.asset_references.extend(item.strip().split()[0] for item in value.split(",") if item.strip())
            elif name == "href" and _is_landing_asset_reference(value):
                self.asset_references.append(value)


def _asset_relative_path(reference: str) -> Path | None:
    parsed = urlparse(reference)
    if parsed.scheme in {"data", "mailto", "javascript"}:
        return None
    if parsed.scheme or parsed.netloc:
        if parsed.scheme not in {"http", "https"} or parsed.netloc != LANDING_HOST:
            return None
        path = unquote(parsed.path).lstrip("/")
    else:
        path = unquote(parsed.path).lstrip("/")
    if not path.startswith("assets/"):
        return None
    return Path(path)


def _is_landing_asset_reference(reference: str) -> bool:
    return _asset_relative_path(reference) is not None


def validate_landing_bundle(landing_dir: Path) -> list[Path]:
    """Validate local asset references and return their bundle-relative paths."""
    landing_dir = landing_dir.resolve()
    index_path = landing_dir / "index.html"
    assets_dir = landing_dir / "assets"
    if not index_path.is_file():
        raise ValueError(f"Landing page is missing: {index_path}")
    if not assets_dir.is_dir():
        raise ValueError(f"Landing assets directory is missing: {assets_dir}")

    html = index_path.read_text(encoding="utf-8")
    parser = _LandingReferenceParser()
    parser.feed(html)
    parser.asset_references.extend(match.group(2) for match in CSS_URL_RE.finditer(html))

    resolved_references: list[Path] = []
    missing: list[str] = []
    invalid: list[str] = []
    for reference in parser.asset_references:
        relative_path = _asset_relative_path(reference)
        if relative_path is None:
            parsed = urlparse(reference)
            if not (parsed.scheme or parsed.netloc) and reference not in {"", "/"}:
                invalid.append(reference)
            continue
        candidate = (landing_dir / relative_path).resolve()
        try:
            candidate.relative_to(assets_dir.resolve())
        except ValueError:
            invalid.append(reference)
            continue
        if not candidate.is_file():
            missing.append(reference)
            continue
        resolved_references.append(relative_path)

    if invalid:
        raise ValueError(f"Landing asset references must use assets/: {sorted(set(invalid))}")
    if missing:
        raise ValueError(f"Landing assets are missing: {sorted(set(missing))}")
    if not resolved_references:
        raise ValueError("Landing page does not reference any local assets")
    return sorted(set(resolved_references))


def _validate_version_folder(version_folder: str) -> None:
    if not VERSION_FOLDER_RE.fullmatch(version_folder):
        raise ValueError(f"Unsafe documentation destination: {version_folder!r}")


def _validate_nonempty_directory(path: Path, label: str) -> None:
    if not path.is_dir() or not any(item.is_file() for item in path.rglob("*")):
        raise ValueError(f"{label} is missing or empty: {path}")


def validate_built_switcher(build_dir: Path, version_folder: str, base_url: str) -> None:
    """Require the generated docs to target the global switcher correctly."""
    index_path = build_dir / "index.html"
    if not index_path.is_file():
        raise ValueError(f"Generated documentation index is missing: {index_path}")
    html = index_path.read_text(encoding="utf-8")
    url_match = SWITCHER_URL_RE.search(html)
    version_match = SWITCHER_VERSION_RE.search(html)
    if not url_match or not version_match:
        raise ValueError("Generated documentation is missing the theme switcher configuration")

    expected_url = f"{base_url.rstrip('/')}/switcher.json"
    actual_url = url_match.group("value")
    if actual_url != expected_url and not actual_url.startswith(f"{expected_url}?"):
        raise ValueError(f"Unexpected switcher URL: expected {expected_url!r}, found {actual_url!r}")
    actual_version = version_match.group("value")
    if actual_version != version_folder:
        raise ValueError(f"Unexpected switcher version match: expected {version_folder!r}, found {actual_version!r}")


def _replace_tree(source: Path, destination: Path) -> None:
    if destination.is_symlink():
        raise ValueError(f"Refusing to replace symlinked destination: {destination}")
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(source, destination)


def prepare_pages_tree(
    *,
    pages_dir: Path,
    build_dir: Path,
    landing_dir: Path,
    version_folder: str,
    base_url: str,
    update_root: bool,
) -> None:
    """Update one version and the global Pages files in an existing checkout."""
    _validate_version_folder(version_folder)
    if update_root and version_folder != "stable":
        raise ValueError("Only the stable deployment may update the root landing page")
    pages_dir = pages_dir.resolve()
    build_dir = build_dir.resolve()
    landing_dir = landing_dir.resolve()
    if not pages_dir.is_dir():
        raise ValueError(f"Pages checkout is missing: {pages_dir}")
    _validate_nonempty_directory(build_dir, "Sphinx build")
    validate_built_switcher(build_dir, version_folder, base_url)
    if update_root:
        validate_landing_bundle(landing_dir)

    _replace_tree(build_dir, pages_dir / version_folder)

    if update_root:
        shutil.copy2(landing_dir / "index.html", pages_dir / "index.html")
        _replace_tree(landing_dir / "assets", pages_dir / "assets")
        (pages_dir / ".nojekyll").touch()

    switcher_path = pages_dir / "switcher.json"
    entries = build_entries(base_url.rstrip("/"))
    switcher_path.write_text(json.dumps(entries, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    """Run the deployment preparation command."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pages-dir", required=True, type=Path)
    parser.add_argument("--build-dir", required=True, type=Path)
    parser.add_argument("--landing-dir", required=True, type=Path)
    parser.add_argument("--version-folder", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--update-root", action="store_true")
    args = parser.parse_args()
    prepare_pages_tree(
        pages_dir=args.pages_dir,
        build_dir=args.build_dir,
        landing_dir=args.landing_dir,
        version_folder=args.version_folder,
        base_url=args.base_url,
        update_root=args.update_root,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
