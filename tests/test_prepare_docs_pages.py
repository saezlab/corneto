"""Tests for assembling the multi-version GitHub Pages tree."""

from pathlib import Path

import pytest

from scripts.prepare_docs_pages import prepare_pages_tree, validate_built_switcher, validate_landing_bundle


def _write(path: Path, content: str = "content") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _landing_bundle(root: Path) -> Path:
    landing = root / "landing"
    _write(
        landing / "index.html",
        """<!doctype html>
<html><head>
<link rel="icon" href="assets/favicon.ico">
<meta property="og:image" content="https://corneto.org/assets/social.jpg">
</head><body><img src="assets/logo.png"><a href="stable/">Docs</a></body></html>
""",
    )
    _write(landing / "assets" / "favicon.ico")
    _write(landing / "assets" / "social.jpg")
    _write(landing / "assets" / "logo.png")
    return landing


def _sphinx_build(root: Path, version: str) -> Path:
    build = root / f"build-{version}"
    _write(
        build / "index.html",
        f"""<script>
DOCUMENTATION_OPTIONS.theme_switcher_json_url = 'https://corneto.org/switcher.json?ts=123';
DOCUMENTATION_OPTIONS.theme_switcher_version_match = '{version}';
</script>
""",
    )
    return build


def test_main_deployment_updates_root_and_exactly_replaces_stable(tmp_path):
    pages = tmp_path / "pages"
    build = _sphinx_build(tmp_path, "stable")
    landing = _landing_bundle(tmp_path)
    _write(pages / "stable" / "stale.html")
    _write(pages / "v0.9.0" / "keep.html")
    _write(pages / "assets" / "stale.png")
    _write(pages / "index.html", "old landing")
    _write(pages / "CNAME", "corneto.org")

    prepare_pages_tree(
        pages_dir=pages,
        build_dir=build,
        landing_dir=landing,
        version_folder="stable",
        base_url="https://corneto.org",
        update_root=True,
    )

    assert (pages / "stable" / "index.html").read_text(encoding="utf-8") == (build / "index.html").read_text(
        encoding="utf-8"
    )
    assert not (pages / "stable" / "stale.html").exists()
    assert (pages / "v0.9.0" / "keep.html").is_file()
    assert (pages / "CNAME").read_text(encoding="utf-8") == "corneto.org"
    assert (pages / "index.html").read_text(encoding="utf-8") == (landing / "index.html").read_text(encoding="utf-8")
    assert sorted(path.name for path in (pages / "assets").iterdir()) == ["favicon.ico", "logo.png", "social.jpg"]
    assert (pages / ".nojekyll").is_file()
    assert (pages / "switcher.json").is_file()


def test_tag_deployment_preserves_root_landing_and_assets(tmp_path):
    pages = tmp_path / "pages"
    build = _sphinx_build(tmp_path, "v1.0.0")
    _write(pages / "index.html", "current landing")
    _write(pages / "assets" / "current.png")
    _write(pages / "v1.0.0" / "stale.html")

    prepare_pages_tree(
        pages_dir=pages,
        build_dir=build,
        landing_dir=tmp_path / "unused-landing",
        version_folder="v1.0.0",
        base_url="https://corneto.org",
        update_root=False,
    )

    assert (pages / "index.html").read_text(encoding="utf-8") == "current landing"
    assert (pages / "assets" / "current.png").is_file()
    assert (pages / "v1.0.0" / "index.html").read_text(encoding="utf-8") == (build / "index.html").read_text(
        encoding="utf-8"
    )
    assert not (pages / "v1.0.0" / "stale.html").exists()
    assert (pages / "switcher.json").is_file()


def test_landing_validation_reports_missing_asset(tmp_path):
    landing = _landing_bundle(tmp_path)
    (landing / "assets" / "logo.png").unlink()

    with pytest.raises(ValueError, match="Landing assets are missing"):
        validate_landing_bundle(landing)


def test_landing_validation_checks_same_site_social_image(tmp_path):
    landing = _landing_bundle(tmp_path)
    (landing / "assets" / "social.jpg").unlink()

    with pytest.raises(ValueError, match=r"https://corneto\.org/assets/social\.jpg"):
        validate_landing_bundle(landing)


def test_built_switcher_rejects_wrong_url(tmp_path):
    build = _sphinx_build(tmp_path, "stable")
    index = build / "index.html"
    index.write_text(index.read_text(encoding="utf-8").replace("corneto.org", "example.org"), encoding="utf-8")

    with pytest.raises(ValueError, match="Unexpected switcher URL"):
        validate_built_switcher(build, "stable", "https://corneto.org")


def test_built_switcher_rejects_wrong_version(tmp_path):
    build = _sphinx_build(tmp_path, "latest")

    with pytest.raises(ValueError, match="Unexpected switcher version match"):
        validate_built_switcher(build, "stable", "https://corneto.org")


def test_built_switcher_rejects_missing_configuration(tmp_path):
    build = tmp_path / "build"
    _write(build / "index.html", "<html></html>")

    with pytest.raises(ValueError, match="missing the theme switcher configuration"):
        validate_built_switcher(build, "stable", "https://corneto.org")


@pytest.mark.parametrize("version_folder", ["latest", "../stable", "v1/escape", "/stable", ""])
def test_unsafe_version_folder_is_rejected(tmp_path, version_folder):
    pages = tmp_path / "pages"
    build = tmp_path / "build"
    pages.mkdir()
    _write(build / "index.html")

    with pytest.raises(ValueError, match="Unsafe documentation destination"):
        prepare_pages_tree(
            pages_dir=pages,
            build_dir=build,
            landing_dir=tmp_path / "unused-landing",
            version_folder=version_folder,
            base_url="https://corneto.org",
            update_root=False,
        )


def test_tag_cannot_update_root_landing(tmp_path):
    pages = tmp_path / "pages"
    build = tmp_path / "build"
    pages.mkdir()
    _write(build / "index.html")

    with pytest.raises(ValueError, match="Only the stable deployment"):
        prepare_pages_tree(
            pages_dir=pages,
            build_dir=build,
            landing_dir=_landing_bundle(tmp_path),
            version_folder="v1.0.0",
            base_url="https://corneto.org",
            update_root=True,
        )
