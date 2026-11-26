"""The release number is written in several files; they must agree."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _first(pattern: str, path: str) -> str:
    match = re.search(pattern, (ROOT / path).read_text(), flags=re.MULTILINE)
    assert match is not None, f"no version found in {path}"
    return match.group(1)


def test_release_number_agrees_across_files():
    versions = {
        "pyproject.toml": _first(r'^version = "([^"]+)"', "pyproject.toml"),
        "docs/source/conf.py": _first(r'^release = "([^"]+)"', "docs/source/conf.py"),
        "CITATION.cff": _first(r"^version: (\S+)", "CITATION.cff"),
        "CHANGELOG.md": _first(r"^## \[(\d[^\]]*)\]", "CHANGELOG.md"),
        "docs/source/changelog.md": _first(r"^## \[(\d[^\]]*)\]", "docs/source/changelog.md"),
    }
    assert len(set(versions.values())) == 1, versions


def test_changelog_copies_are_identical():
    assert (ROOT / "CHANGELOG.md").read_text() == (ROOT / "docs/source/changelog.md").read_text()
