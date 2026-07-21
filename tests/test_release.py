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
    monkeypatch.setattr(release, "_ensure_dev_is_merged", calls.append)
    monkeypatch.setattr(
        release,
        "_ensure_tag_does_not_exist",
        lambda version, remote: calls.append((version, remote)),
    )

    result = release.main(["v1.0.0-beta.8", "--remote", "public", "--dry-run"])

    assert result == 0
    assert calls == [
        "public",
        "public",
        "public",
        ("v1.0.0-beta.8", "public"),
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
