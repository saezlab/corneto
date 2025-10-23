# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Iterable

import nox

# -----------------------------------------------------------------------------
# Global configuration
# -----------------------------------------------------------------------------

nox.options.reuse_existing_virtualenvs = True
nox.options.error_on_missing_interpreters = False

PYTHON_VERSIONS: tuple[str, ...] = ("3.10", "3.11", "3.12")

# Paths -----------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).parent
DOCS_SRC = PROJECT_ROOT / "docs"
DOCS_HTML = DOCS_SRC / "_build" / "html"
DOCS_LINKCHECK = DOCS_SRC / "_build" / "linkcheck"
JUPYTER_CACHE = DOCS_SRC / "_jupyter-cache"

TUTORIALS_ROOT = DOCS_SRC / "tutorials"


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def _install(session: nox.Session, extras: str | Iterable[str] = "") -> None:
    extras_str = extras if isinstance(extras, str) else ",".join(extras)
    target = f".[{extras_str}]" if extras_str else "."
    session.install("-e", target)


# -----------------------------------------------------------------------------
# Quality assurance
# -----------------------------------------------------------------------------


@nox.session(python=PYTHON_VERSIONS)
def tests(session: nox.Session) -> None:
    """Run pytest across supported interpreters."""
    _install(session, extras=["dev", "docs"])
    session.run("pytest", "tests", *session.posargs)


@nox.session(python=PYTHON_VERSIONS)
def lint(session: nox.Session) -> None:
    """Static analysis with Ruff (no auto‑fix)."""
    _install(session, extras=["dev"])
    session.run("ruff", "check", "corneto", "--exclude", "tests", *session.posargs)


@nox.session(python=PYTHON_VERSIONS)
def format(session: nox.Session) -> None:
    """Apply Ruff auto‑fixes and re‑format code."""
    _install(session, extras=["dev"])
    session.run("ruff", "check", "corneto", "--exclude", "tests", "--fix")
    session.run("ruff", "format", "corneto", "--exclude", "tests")


@nox.session(python=PYTHON_VERSIONS)
def typing(session: nox.Session) -> None:
    """Type‑check with *pyrefly*."""
    _install(session, extras=["dev"])
    session.run("pyrefly", "check", "corneto", *session.posargs)


# -----------------------------------------------------------------------------
# Documentation sessions
# -----------------------------------------------------------------------------


def _sphinx(session: nox.Session, *opts: str) -> None:
    os.makedirs(JUPYTER_CACHE, exist_ok=True)
    session.run("sphinx-build", "-b", "html", *opts, str(DOCS_SRC), str(DOCS_HTML))


@nox.session(python=PYTHON_VERSIONS)
def docs(session: nox.Session) -> None:
    """Build HTML docs with myst‑nb cache (default)."""
    _install(session, extras=["docs"])
    _sphinx(session)


@nox.session(python=PYTHON_VERSIONS)
def docs_force(session: nox.Session) -> None:
    """Force notebook execution, then build HTML docs."""
    _install(session, extras=["docs"])
    _sphinx(session, "-D", "nb_execution_mode=force")


@nox.session(python=PYTHON_VERSIONS)
def docs_clean(session: nox.Session) -> None:
    """Remove previous build and rebuild HTML docs."""
    _install(session, extras=["docs"])
    if DOCS_HTML.exists():
        shutil.rmtree(DOCS_HTML)
    _sphinx(session)


@nox.session(python=PYTHON_VERSIONS)
def docs_werror(session: nox.Session) -> None:
    """Build docs but fail on warnings."""
    _install(session, extras=["docs"])
    _sphinx(session, "-W")


@nox.session(python=PYTHON_VERSIONS)
def docs_all(session: nox.Session) -> None:
    """Clean, force notebook execution, and build docs with warnings as errors."""
    _install(session, extras=["docs"])
    if DOCS_HTML.exists():
        shutil.rmtree(DOCS_HTML)
    _sphinx(session, "-D", "nb_execution_mode=force", "-W")


@nox.session(python=PYTHON_VERSIONS)
def docs_linkcheck(session: nox.Session) -> None:
    """Verify outbound links."""
    _install(session, extras=["docs"])
    session.run("sphinx-build", "-b", "linkcheck", str(DOCS_SRC), str(DOCS_LINKCHECK))


@nox.session(python=PYTHON_VERSIONS)
def docs_full(session: nox.Session) -> None:
    """Full local docs check: build documentation."""
    _install(session, extras=["docs"])
    _sphinx(session)


@nox.session(python=PYTHON_VERSIONS)
def docs_serve(session: nox.Session) -> None:
    """Serve HTML docs at http://localhost:8000."""
    _install(session, extras=["docs"])
    if not DOCS_HTML.exists():
        session.log("Docs not found – building first …")
        _sphinx(session)

    session.log("Serving docs at http://localhost:8000  (CTRL‑C to quit)")
    session.run(
        "python",
        "-m",
        "http.server",
        "8000",
        "--directory",
        str(DOCS_HTML),
        external=True,
    )
