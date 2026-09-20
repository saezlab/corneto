"""Tests for the guarded release helper."""

import corneto.release as release


def test_parse_args_defaults_to_origin():
    """Ordinary clones continue releasing through origin by default."""
    args = release.parse_args(["v1.2.3"])

    assert args.remote == "origin"


def test_dry_run_uses_selected_remote(monkeypatch, capsys):
    """All remote-aware safety checks receive the selected remote."""
    calls = []
    monkeypatch.setattr(release, "_ensure_remote_exists", calls.append)
    monkeypatch.setattr(release, "_ensure_clean_tree", lambda: None)
    monkeypatch.setattr(release, "_ensure_on_main", lambda: None)
    monkeypatch.setattr(release, "_ensure_up_to_date_with_remote_main", calls.append)
    monkeypatch.setattr(
        release,
        "_ensure_tag_does_not_exist",
        lambda version, remote: calls.append((version, remote)),
    )
    monkeypatch.setattr(
        release,
        "_ensure_release_notes",
        lambda version: calls.append(("notes", version)),
    )

    result = release.main(["v1.0.0-beta.8", "--remote", "public", "--dry-run"])

    assert result == 0
    assert calls == [
        "public",
        "public",
        ("v1.0.0-beta.8", "public"),
        ("notes", "v1.0.0-beta.8"),
    ]
    assert "Would create and push tag: v1.0.0-beta.8" in capsys.readouterr().out


def test_create_and_push_tag_uses_selected_remote(monkeypatch):
    """The annotated tag is pushed only to the requested remote."""
    commands = []
    monkeypatch.setattr(
        release,
        "_run",
        lambda command, check=True: commands.append(command) or "",
    )

    release._create_and_push_tag("v1.0.0-beta.8", "public")

    assert commands == [
        ["git", "tag", "-a", "v1.0.0-beta.8", "-m", "v1.0.0-beta.8"],
        ["git", "push", "public", "v1.0.0-beta.8"],
    ]


def test_remote_name_cannot_be_an_option():
    """Remote selection cannot inject a Git command option."""
    try:
        release._ensure_remote_exists("--upload-pack=example")
    except release.ReleaseError as exc:
        assert "Invalid remote name" in str(exc)
    else:
        raise AssertionError("Expected an invalid remote name to be rejected")


def test_release_notes_require_canonical_prerelease_heading_and_highlights(tmp_path):
    """Pre-release notes use the same title and badge as generated documentation."""
    notes_dir = tmp_path / "docs" / "releases"
    notes_dir.mkdir(parents=True)
    notes = notes_dir / "v1.2.3-rc.4.md"
    notes.write_text(
        "# Release v1.2.3-rc.4 {bdg-warning}`Pre-release`\n\n## Highlights\n\n- A release highlight.\n",
        encoding="utf-8",
    )

    release._ensure_release_notes("v1.2.3-rc.4", root=tmp_path)


def test_release_notes_reject_inconsistent_title(tmp_path):
    """A manually reformatted release name cannot pass the release guard."""
    notes_dir = tmp_path / "docs" / "releases"
    notes_dir.mkdir(parents=True)
    (notes_dir / "v1.2.3-rc.4.md").write_text(
        "# CORNETO 1.2.3 RC4\n\n## Highlights\n",
        encoding="utf-8",
    )

    try:
        release._ensure_release_notes("v1.2.3-rc.4", root=tmp_path)
    except release.ReleaseError as exc:
        assert "must start with" in str(exc)
        assert "Pre-release" in str(exc)
    else:
        raise AssertionError("Expected an inconsistent release-note title to be rejected")


def test_release_notes_require_highlights(tmp_path):
    """Every manually prepared release page retains the shared highlights section."""
    notes_dir = tmp_path / "docs" / "releases"
    notes_dir.mkdir(parents=True)
    (notes_dir / "v1.2.3.md").write_text(
        "# Release v1.2.3\n\n## Details\n",
        encoding="utf-8",
    )

    try:
        release._ensure_release_notes("v1.2.3", root=tmp_path)
    except release.ReleaseError as exc:
        assert "## Highlights" in str(exc)
    else:
        raise AssertionError("Expected missing highlights to be rejected")
