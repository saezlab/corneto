# -*- coding: utf-8 -*-
from __future__ import annotations

import fnmatch
import os
import re
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Optional, Set

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

# Notebook execution ----------------------------------------------------------

SKIP_DIR_KEYWORDS: Set[str] = {".ipynb_checkpoints", "__pycache__", ".git"}
MANIFEST_FILE = "pixi.toml"
KERNEL_FMT = "pixi-{stem}"


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def _install(session: nox.Session, extras: str | Iterable[str] = "") -> None:
    extras_str = extras if isinstance(extras, str) else ",".join(extras)
    target = f".[{extras_str}]" if extras_str else "."
    session.install("-e", target)


def _pixi_available(session: nox.Session) -> bool:
    try:
        session.run("pixi", "--version", external=True, silent=True)
        return True
    except Exception:
        return False


def _gather_notebooks(base_dir: Path) -> list[Path]:
    return sorted(nb for nb in base_dir.rglob("*.ipynb") if not SKIP_DIR_KEYWORDS.intersection(nb.parts))


def _pixi_run(session: nox.Session, manifest: Path, *cmd: str) -> None:
    session.run("pixi", "run", "--manifest-path", str(manifest), *cmd, external=True)


# -----------------------------------------------------------------------------
# Notebook filtering
# -----------------------------------------------------------------------------


def _filter_notebooks(notebooks: list[Path], patterns: list[str]) -> list[Path]:
    if not patterns:
        return notebooks

    regexes: list[re.Pattern[str]] = []
    for p in patterns:
        try:
            regexes.append(re.compile(p))
        except re.error:
            regexes.append(re.compile(fnmatch.translate(p)))

    return [nb for nb in notebooks if any(rx.search(nb.as_posix()) for rx in regexes)]


# -----------------------------------------------------------------------------
# Notebook caching
# -----------------------------------------------------------------------------


@nox.session(name="cache_notebooks_with_pixi")
def cache_notebooks_with_pixi(session: nox.Session) -> None:
    """Execute all (or selected) notebooks under *docs/* with per-directory Pixi
    isolation.

    Pass regex *or* glob patterns after “--”, e.g.:

        nox -s cache_notebooks_with_pixi -- "*metabolic*" "^docs/.*intro.ipynb$"
    """
    _install(session, extras=["dev"])
    session.install("papermill", "ipykernel")

    use_pixi = _pixi_available(session)
    if not use_pixi:
        session.warn("Pixi CLI not found – notebooks will run in the dev environment.")

    # 1 – collect and optionally filter notebooks ---------------------------
    all_nbs = _gather_notebooks(TUTORIALS_ROOT)
    notebooks = _filter_notebooks(all_nbs, session.posargs)

    # 2 – group by manifest path (None → dev env) ---------------------------
    groups: dict[Optional[Path], list[Path]] = defaultdict(list)
    for nb in notebooks:
        manifest = nb.parent / MANIFEST_FILE if use_pixi else None
        print(f"Manifest for {nb}: {manifest}")
        if manifest and manifest.exists():
            groups[manifest].append(nb)
        else:
            print(f"No manifest for {nb} – running in dev env")
            groups[None].append(nb)

    # 3 – run notebooks that *don’t* need Pixi first ------------------------
    for nb in sorted(groups.pop(None, [])):
        session.log(f"📓 {nb.relative_to(PROJECT_ROOT)}  (dev env)")
        with session.chdir(nb.parent):
            session.run("papermill", str(nb.name), str(nb.name), external=True)

    # 4 – run each manifest group exactly once -----------------------------
    for manifest, nbs in groups.items():
        _run_directory_notebooks_with_pixi(session, manifest, sorted(nbs))


def _run_directory_notebooks_with_pixi(
    session: nox.Session,
    manifest: Path,
    notebooks: list[Path],
) -> None:
    dir_id = manifest.parent.relative_to(PROJECT_ROOT).as_posix().replace("/", "-")
    kernel = KERNEL_FMT.format(stem=dir_id)

    # -- resolve env + install tooling --------------------------------------
    session.run("pixi", "install", "--manifest-path", str(manifest), external=True)
    _pixi_run(session, manifest, "python", "-m", "pip", "install", "-e", ".[dev]")
    _pixi_run(session, manifest, "python", "-m", "pip", "install", "ipykernel", "papermill")
    _pixi_run(session, manifest, "dot", "-c")
    _pixi_run(
        session,
        manifest,
        "python",
        "-m",
        "ipykernel",
        "install",
        "--user",
        "--name",
        kernel,
    )

    # -- run each notebook ---------------------------------------------------
    for nb in notebooks:
        session.log(f"📓 {nb.relative_to(PROJECT_ROOT)}")
        with session.chdir(nb.parent):
            session.run(
                "pixi",
                "run",
                "--manifest-path",
                str(manifest),
                "papermill",
                str(nb.name),
                str(nb.name),
                "-k",
                kernel,
                external=True,
            )

    # -- clean once per directory -------------------------------------------
    try:
        session.run(
            "pixi",
            "clean",
            "--manifest-path",
            str(manifest),
            external=True,
            silent=True,
        )
    except Exception:
        session.warn(f"Pixi clean failed for {manifest}")


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
    """Type‑check with *mypy*."""
    _install(session, extras=["dev"])
    session.run("mypy", "corneto", *session.posargs)


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
def generate_switcher(session: nox.Session) -> None:
    """Generate *switcher.json* used by Read‑the‑Docs version switcher."""
    _install(session, extras=["docs"])
    session.run("python", "docs/generate_switcher.py")


@nox.session(python=PYTHON_VERSIONS)
def docs_full(session: nox.Session) -> None:
    """Full local docs check: build + switcher generation."""
    _install(session, extras=["docs"])
    _sphinx(session)
    session.run("python", "docs/generate_switcher.py")


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
