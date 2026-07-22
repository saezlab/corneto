"""Tests for the documentation release synchronization script."""

from scripts.sync_releases import update_releases_index


def test_release_index_preserves_local_pages(tmp_path):
    """Keep draft release notes and migration guides in the generated index."""
    (tmp_path / "v1.0.0-rc.1.md").touch()
    (tmp_path / "migration-1.0.md").touch()
    releases = [{"tag_name": "v1.0.0-beta.2"}]

    update_releases_index(releases, tmp_path)

    index = (tmp_path / "index.md").read_text(encoding="utf-8")
    assert index.index("v1.0.0-rc.1") < index.index("migration-1.0") < index.index("v1.0.0-beta.2")
