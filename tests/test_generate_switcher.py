"""Tests for the public documentation version switcher."""

from scripts.generate_switcher import build_entries


def test_switcher_lists_stable_and_release_tags_without_dev_docs(monkeypatch):
    """The public trunk has no separate latest/development documentation site."""
    monkeypatch.setattr("scripts.generate_switcher._get_tags", lambda: ["v1.0.0-rc.2"])

    entries = build_entries("https://corneto.org")

    assert entries == [
        {
            "name": "stable",
            "version": "stable",
            "url": "https://corneto.org/stable/",
            "preferred": True,
        },
        {
            "name": "v1.0.0-rc.2",
            "version": "v1.0.0-rc.2",
            "url": "https://corneto.org/v1.0.0-rc.2/",
        },
    ]
