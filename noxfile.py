# -*- coding: utf-8 -*-
import glob
import os
import shutil

import nox

nox.options.reuse_existing_virtualenvs = True
nox.options.error_on_missing_interpreters = False

TUTORIAL_BASE = os.path.join("docs", "tutorials")
TUTORIAL_DIRS = sorted(
    [os.path.basename(path) for path in glob.glob(os.path.join(TUTORIAL_BASE, "*")) if os.path.isdir(path)]
)


@nox.session(python=["3.10", "3.11", "3.12"])
def tests(session):
    """Run the unit tests under multiple Python versions.
    Installs the package in editable mode with dev and docs extras,
    then runs pytest against the tests/ directory.
    """
    session.install("-e", ".[dev,docs]")
    session.run("pytest", "tests")


@nox.session(python="3.10")
def lint(session):
    """Lint code with ruff (checks only, no auto-fix).
    Installs dev extras, then runs:
      ruff check corneto --exclude tests
    """
    session.install("-e", ".[dev]")
    session.run("ruff", "check", "corneto", "--exclude", "tests")


@nox.session(python="3.10")
def format(session):
    """Auto-fix formatting and linting issues with ruff.
    Installs dev extras, then runs:
      ruff check corneto --exclude tests --fix
      ruff format corneto --exclude tests
    """
    session.install("-e", ".[dev]")
    session.run("ruff", "check", "corneto", "--exclude", "tests", "--fix")
    session.run("ruff", "format", "corneto", "--exclude", "tests")


@nox.session(python="3.10")
def typing(session):
    """Run static type checks with mypy.
    Installs dev extras, then runs:
      mypy corneto
    """
    session.install("-e", ".[dev]")
    session.run("mypy", "corneto")


@nox.session(name="cache_tutorial_notebooks", reuse_venv=True)
@nox.parametrize("tutorial_name", TUTORIAL_DIRS)
def cache_tutorial_notebooks(session, tutorial_name):
    """For each .ipynb under docs/tutorials/<tutorial_name>:
    1) install project + dev extras,
    2) install any per-notebook requirements.txt,
    3) register an ipykernel that points to *this* venv,
    4) execute the notebook in-place with Papermill using that kernel.
    """
    tutorial_path = os.path.join("docs", "tutorials", tutorial_name)

    # 1) Install our project with dev dependencies
    session.log(f"[{tutorial_name}] Installing project (dev dependencies)…")
    session.install("-e", ".[dev]")

    # 2) Install Papermill + execution dependencies
    session.log(f"[{tutorial_name}] Installing papermill + supporting libraries…")
    session.install("papermill", "nbformat", "jupyter_client", "ipykernel")

    # 3) Ensure this venv owns the 'python3' kernelspec so papermill uses it
    session.log(f"[{tutorial_name}] Registering ipykernel in this venv…")
    session.run(
        "python",
        "-m",
        "ipykernel",
        "install",
        "--sys-prefix",  # write into <venv>/share/jupyter/kernels
        "--name",
        "python3",  # shadows any user-level 'python3' spec
        "--display-name",
        f"Python (nox-{tutorial_name})",
        external=True,
    )

    # 4) Find all notebooks in this tutorial folder
    notebooks = glob.glob(os.path.join(tutorial_path, "**", "*.ipynb"), recursive=True)
    if not notebooks:
        session.log(f"[{tutorial_name}] No .ipynb files found under {tutorial_path}")
        return

    for nb_path in notebooks:
        nb_dir = os.path.dirname(nb_path)
        req_file = os.path.join(nb_dir, "requirements.txt")

        # 4a) Install notebook-specific requirements if present
        if os.path.isfile(req_file):
            rel_req = os.path.relpath(req_file, tutorial_path)
            session.log(f"[{tutorial_name}] Installing requirements: {rel_req}")
            session.install("-r", req_file)

        # 4b) Execute the notebook in-place with Papermill
        rel_nb = os.path.relpath(nb_path, tutorial_path)
        session.log(f"[{tutorial_name}] Executing (via papermill) {rel_nb} …")
        session.run("papermill", nb_path, nb_path, "--kernel", "python3")


# ────────────────────────────────────────────────────────────────────────
# Documentation sessions (mirroring tox's docs* envs)
#    Each session installs only the docs extras and runs sphinx-build
#    with the appropriate flags. Myst_nb caching must be enabled in docs/conf.py:
#      jupyter_execute_notebooks = "cache"
#      jupyter_cache_path = os.path.join(os.path.dirname(__file__), "_jupyter-cache")
# ────────────────────────────────────────────────────────────────────────
DOCS_SOURCE = "docs"
DOCS_HTML_BUILD = os.path.join(DOCS_SOURCE, "_build", "html")
DOCS_LINKCHECK_BUILD = os.path.join(DOCS_SOURCE, "_build", "linkcheck")
JUPYTER_CACHE_DIR = os.path.join(DOCS_SOURCE, "_jupyter-cache")


@nox.session(python="3.10")
def docs(session):
    """Build documentation (HTML) with myst_nb caching.
    Equivalent to tox: [testenv:docs]
    """
    session.install("-e", ".[docs]")
    os.makedirs(JUPYTER_CACHE_DIR, exist_ok=True)
    session.run("sphinx-build", "-b", "html", DOCS_SOURCE, DOCS_HTML_BUILD)


@nox.session(python="3.10")
def docs_force(session):
    """Build docs forcing notebook re-execution.
    Equivalent to tox: [testenv:docs-force]
    """
    session.install("-e", ".[docs]")
    os.makedirs(JUPYTER_CACHE_DIR, exist_ok=True)
    session.run(
        "sphinx-build",
        "-b",
        "html",
        "-D",
        "nb_execution_mode=force",
        DOCS_SOURCE,
        DOCS_HTML_BUILD,
    )


@nox.session(python="3.10")
def docs_clean(session):
    """Clean the docs build directory and rebuild.
    Equivalent to tox: [testenv:docs-clean]
    """
    session.install("-e", ".[docs]")
    if os.path.isdir(DOCS_HTML_BUILD):
        shutil.rmtree(DOCS_HTML_BUILD)
    os.makedirs(JUPYTER_CACHE_DIR, exist_ok=True)
    session.run("sphinx-build", "-b", "html", DOCS_SOURCE, DOCS_HTML_BUILD)


@nox.session(python="3.10")
def docs_werror(session):
    """Build docs, treat warnings as errors.
    Equivalent to tox: [testenv:docs-werror]
    """
    session.install("-e", ".[docs]")
    os.makedirs(JUPYTER_CACHE_DIR, exist_ok=True)
    session.run("sphinx-build", "-b", "html", "-W", DOCS_SOURCE, DOCS_HTML_BUILD)


@nox.session(python="3.10")
def docs_all(session):
    """Clean + force notebook execution + build + Werror.
    Equivalent to tox: [testenv:docs-all]
    """
    session.install("-e", ".[docs]")
    if os.path.isdir(DOCS_HTML_BUILD):
        shutil.rmtree(DOCS_HTML_BUILD)
    os.makedirs(JUPYTER_CACHE_DIR, exist_ok=True)
    session.run(
        "sphinx-build",
        "-b",
        "html",
        "-D",
        "nb_execution_mode=force",
        "-W",
        DOCS_SOURCE,
        DOCS_HTML_BUILD,
    )


@nox.session(python="3.10")
def docs_linkcheck(session):
    """Check documentation for broken links.
    Equivalent to tox: [testenv:docs-linkcheck]
    """
    session.install("-e", ".[docs]")
    session.run("sphinx-build", "-b", "linkcheck", DOCS_SOURCE, DOCS_LINKCHECK_BUILD)


@nox.session(python="3.10")
def generate_switcher(session):
    """Generate switcher.json for docs version switching.
    Equivalent to tox: [testenv:generate-switcher]
    """
    session.install("-e", ".[docs]")
    session.run("python", "docs/generate_switcher.py")


@nox.session(python="3.10")
def docs_full(session):
    """Full local docs test: build + generate switcher.
    Equivalent to tox: [testenv:docs-full]
    """
    session.install("-e", ".[docs]")
    os.makedirs(JUPYTER_CACHE_DIR, exist_ok=True)
    session.run("sphinx-build", "-b", "html", DOCS_SOURCE, DOCS_HTML_BUILD)
    session.run("python", "docs/generate_switcher.py")


@nox.session(python="3.10")
def docs_serve(session):
    """Serve the built documentation locally at http://localhost:8000.
    Equivalent to tox: [testenv:docs-serve]
    """
    session.install("-e", ".[docs]")
    if not os.path.isdir(DOCS_HTML_BUILD):
        session.log("Docs not found. Building first…")
        os.makedirs(JUPYTER_CACHE_DIR, exist_ok=True)
        session.run("sphinx-build", "-b", "html", DOCS_SOURCE, DOCS_HTML_BUILD)
    session.log("Serving docs at http://localhost:8000 (CTRL-C to quit)")
    session.run(
        "python",
        "-m",
        "http.server",
        "8000",
        "--directory",
        DOCS_HTML_BUILD,
        external=True,
    )
