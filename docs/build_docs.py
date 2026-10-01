#!/usr/bin/env python3
"""Execute eligible CORNETO notebooks and build the documentation site."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import runpy
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from time import monotonic
from typing import Any, Mapping

from _notebook_execution import (
    check_requirements,
    environment_fingerprint,
    execute_notebook,
    execution_timeout_for_notebook,
    load_notebook,
    optional_reason_for_notebook,
    pixi_environment,
    python_for_pixi,
    requirements_for_notebook,
)
from sphinx.util.matching import Matcher

DOCS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = DOCS_DIR.parent
BUILD_DIR = DOCS_DIR / "_build"
STAGE_DIR = BUILD_DIR / "source"
HTML_DIR = BUILD_DIR / "html"
CACHE_DIR = BUILD_DIR / "notebook-cache"
REPORT_PATH = BUILD_DIR / "notebook-report.json"
IGNORED_DIRS = {
    ".pixi",
    ".venv",
    "build",
    "_build",
    "_jupyter-cache",
    "__pycache__",
    ".pytest_cache",
}


def _run(command: list[str], cwd: Path, *, capture: bool = False) -> subprocess.CompletedProcess:
    print(f"> {' '.join(command)}  (cwd={cwd})")
    return subprocess.run(
        command,
        cwd=str(cwd),
        check=True,
        text=True,
        capture_output=capture,
    )


def _copy_docs_tree() -> None:
    if STAGE_DIR.exists():
        shutil.rmtree(STAGE_DIR)

    def ignore(directory: str, names: list[str]) -> set[str]:
        del directory
        return {name for name in names if name in IGNORED_DIRS or name == ".DS_Store"}

    shutil.copytree(DOCS_DIR, STAGE_DIR, ignore=ignore)


def _is_generated_path(relative: Path) -> bool:
    return any(part in IGNORED_DIRS for part in relative.parts)


def _notebook_inventory(docs_dir: Path | None = None) -> list[tuple[Path, str | None]]:
    docs_dir = docs_dir or DOCS_DIR
    configuration = runpy.run_path(str(docs_dir / "conf.py"))
    patterns = configuration.get("exclude_patterns", [])
    if not isinstance(patterns, list) or not all(isinstance(pattern, str) for pattern in patterns):
        raise ValueError(f"{docs_dir / 'conf.py'}: exclude_patterns must be a list of strings")
    pattern_matchers = [(pattern, Matcher([pattern])) for pattern in patterns]
    inventory = []
    for notebook in sorted(docs_dir.rglob("*.ipynb")):
        relative = notebook.relative_to(docs_dir)
        if _is_generated_path(relative):
            continue
        candidates = [Path(*relative.parts[:depth]).as_posix() for depth in range(1, len(relative.parts) + 1)]
        matching = [
            pattern
            for pattern, pattern_matcher in pattern_matchers
            if any(pattern_matcher.match(candidate) for candidate in candidates)
        ]
        if matching:
            reason = "excluded by docs/conf.py: " + ", ".join(matching)
            inventory.append((notebook, reason))
        else:
            inventory.append((notebook, None))
    return inventory


def _nearest_tutorial_project(notebook: Path) -> Path:
    for parent in notebook.parents:
        if parent == DOCS_DIR.parent:
            break
        if (parent / "pixi.toml").is_file():
            return parent
    raise FileNotFoundError(f"No tutorial pixi.toml found for {notebook}")


def _environment_key(project: Path) -> str:
    return "main" if project == PROJECT_ROOT else project.relative_to(DOCS_DIR).as_posix()


def _prepare_environment(project: Path) -> tuple[Path, dict[str, str]]:
    if project == PROJECT_ROOT:
        return Path(sys.executable).resolve(), os.environ.copy()

    _run(["pixi", "config", "set", "--local", "run-post-link-scripts", "insecure"], project)
    _run(["pixi", "install"], project)
    _run(
        [
            "pixi",
            "run",
            "python",
            "-m",
            "pip",
            "install",
            "--no-deps",
            "-e",
            str(PROJECT_ROOT),
        ],
        project,
    )
    python = python_for_pixi(project)
    return python, pixi_environment(project)


def _canonical_notebook(path: Path) -> bytes:
    notebook = load_notebook(path)
    for cell in notebook.get("cells", []):
        cell.pop("outputs", None)
        cell.pop("execution_count", None)
    return json.dumps(notebook, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _hash_document_inputs(docs_dir: Path = DOCS_DIR) -> str:
    digest = hashlib.sha256()
    for path in sorted(docs_dir.rglob("*")):
        relative = path.relative_to(docs_dir)
        if _is_generated_path(relative) or any(part.startswith(".") for part in relative.parts):
            continue
        if not path.is_file() or path.name == ".DS_Store":
            continue
        digest.update(relative.as_posix().encode("utf-8"))
        digest.update(b"\0")
        if path.suffix == ".ipynb":
            digest.update(_canonical_notebook(path))
        else:
            digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _hash_project_source(project_root: Path = PROJECT_ROOT) -> str:
    digest = hashlib.sha256()
    for path in sorted((project_root / "corneto").rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts or path.suffix == ".pyc":
            continue
        digest.update(path.relative_to(project_root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    for name in ("pyproject.toml", "pixi.toml", "pixi.lock"):
        path = project_root / name
        if path.is_file():
            digest.update(name.encode("utf-8"))
            digest.update(path.read_bytes())
    return digest.hexdigest()


def _cache_key(
    notebook: Path,
    project: Path,
    python: Path,
    docs_fingerprint: str,
    code_fingerprint: str,
    package_fingerprint: str,
    activation_environment: Mapping[str, str] | None = None,
) -> str:
    manifest_fingerprint = hashlib.sha256()
    for name in ("pixi.toml", "pixi.lock"):
        manifest = project / name
        if manifest.is_file():
            manifest_fingerprint.update(name.encode("utf-8"))
            manifest_fingerprint.update(manifest.read_bytes())
    payload = {
        "notebook": hashlib.sha256(_canonical_notebook(notebook)).hexdigest(),
        "docs_inputs": docs_fingerprint,
        "corneto_source": code_fingerprint,
        "environment_manifest": manifest_fingerprint.hexdigest(),
        "environment_packages": hashlib.sha256(package_fingerprint.encode()).hexdigest(),
        "activation_environment": hashlib.sha256(
            json.dumps(dict(activation_environment or {}), sort_keys=True).encode()
        ).hexdigest(),
        "execution_timeout": execution_timeout_for_notebook(notebook),
        "python": str(python),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _scan_files(root: Path) -> dict[str, tuple[int, int]]:
    result = {}
    for path in root.rglob("*"):
        if path.is_file():
            stat = path.stat()
            result[path.relative_to(root).as_posix()] = (stat.st_size, stat.st_mtime_ns)
    return result


def _slug(relative: Path) -> str:
    return relative.as_posix().replace("/", "__")


def _read_cache(cache_path: Path, key: str) -> dict[str, Any] | None:
    metadata_path = cache_path / "metadata.json"
    output_path = cache_path / "output.ipynb"
    if not metadata_path.is_file() or not output_path.is_file():
        return None
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if metadata.get("key") != key:
        return None
    return metadata


def _restore_cache(cache_path: Path, metadata: dict[str, Any], stage_notebook: Path, stage: Path) -> None:
    shutil.copy2(cache_path / "output.ipynb", stage_notebook)
    for relative in metadata.get("deleted", []):
        path = stage / relative
        if path.is_file():
            path.unlink()
    artifact_root = cache_path / "artifacts"
    for relative in metadata.get("artifacts", []):
        source = artifact_root / relative
        target = stage / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def _save_cache(
    cache_path: Path,
    key: str,
    output_notebook: Path,
    before: dict[str, tuple[int, int]],
    after: dict[str, tuple[int, int]],
    stage: Path,
    stage_notebook: Path,
) -> None:
    changed = sorted(
        relative
        for relative, stat in after.items()
        if relative != stage_notebook.relative_to(stage).as_posix() and before.get(relative) != stat
    )
    deleted = sorted(set(before) - set(after))
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="notebook-cache-", dir=cache_path.parent) as temp:
        temporary = Path(temp)
        shutil.copy2(output_notebook, temporary / "output.ipynb")
        artifacts = temporary / "artifacts"
        for relative in changed:
            target = artifacts / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(stage / relative, target)
        metadata = {"key": key, "artifacts": changed, "deleted": deleted}
        (temporary / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        if cache_path.exists():
            shutil.rmtree(cache_path)
        shutil.move(str(temporary), cache_path)


def _write_report(report: dict[str, Any]) -> None:
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    report["updated_at"] = datetime.now(timezone.utc).isoformat()
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        prefix=".notebook-report-",
        suffix=".tmp",
        dir=BUILD_DIR,
        delete=False,
    ) as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
        temporary_report = Path(stream.name)
    os.replace(temporary_report, REPORT_PATH)


def build(force: bool = False, include_optional: bool = False) -> int:
    """Execute eligible notebooks and write the Sphinx HTML site."""
    build_started = monotonic()
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "mode": "force-all"
        if force and include_optional
        else ("force" if force else "all" if include_optional else "cache"),
        "include_optional": include_optional,
        "active_notebook": None,
        "notebooks": [],
        "environments": {},
    }
    _write_report(report)
    try:
        inventory = _notebook_inventory()
        eligible: list[tuple[Path, Path, list[str]]] = []
        for notebook, excluded_reason in inventory:
            relative = notebook.relative_to(DOCS_DIR)
            if excluded_reason:
                print(f"EXCLUDED {relative}: {excluded_reason}")
                report["notebooks"].append(
                    {
                        "path": relative.as_posix(),
                        "status": "excluded",
                        "reason": excluded_reason,
                        "duration_seconds": 0.0,
                    }
                )
                _write_report(report)
                continue
            optional_reason = optional_reason_for_notebook(notebook)
            if optional_reason and not include_optional:
                reason = f"{optional_reason} Run `pixi run docs --all` to include it."
                print(f"OPTIONAL {relative}: {reason}")
                report["notebooks"].append(
                    {
                        "path": relative.as_posix(),
                        "status": "optional-skipped",
                        "reason": reason,
                        "duration_seconds": 0.0,
                    }
                )
                _write_report(report)
                continue
            requirements = requirements_for_notebook(notebook)
            unknown = sorted(set(requirements) - {"gurobi"})
            if unknown:
                raise ValueError(f"{relative}: unsupported requirements: {', '.join(unknown)}")
            project = _nearest_tutorial_project(notebook) if relative.parts[0] == "tutorials" else PROJECT_ROOT
            eligible.append((notebook, project, requirements))

        _copy_docs_tree()

        projects = sorted({project for _, project, _ in eligible}, key=_environment_key)
        environments: dict[Path, dict[str, Any]] = {}
        for project in projects:
            python, activation_environment = _prepare_environment(project)
            requirements = {
                requirement
                for _, notebook_project, notebook_requirements in eligible
                if notebook_project == project
                for requirement in notebook_requirements
            }
            check_results = (
                check_requirements(
                    requirements,
                    python,
                    project,
                    environment=activation_environment,
                )
                if requirements
                else {}
            )
            package_fingerprint = environment_fingerprint(
                python,
                project,
                environment=activation_environment,
            )
            environments[project] = {
                "python": python,
                "activation_environment": activation_environment,
                "requirements": check_results,
                "packages": package_fingerprint,
                "kernel_root": BUILD_DIR / "kernels",
            }
            report["environments"][_environment_key(project)] = {
                "python": str(python),
                "requirements": check_results,
            }
            _write_report(report)

        docs_fingerprint = _hash_document_inputs()
        code_fingerprint = _hash_project_source()
        for notebook, project, requirements in eligible:
            relative = notebook.relative_to(DOCS_DIR)
            stage_notebook = STAGE_DIR / relative
            stage_notebook.parent.mkdir(parents=True, exist_ok=True)
            env = environments[project]
            unavailable = [
                requirement for requirement in requirements if env["requirements"].get(requirement) is not None
            ]
            if unavailable:
                reason = "; ".join(f"{requirement}: {env['requirements'][requirement]}" for requirement in unavailable)
                print(f"SKIPPED {relative}: {reason}")
                report["notebooks"].append(
                    {
                        "path": relative.as_posix(),
                        "status": "skipped",
                        "reason": reason,
                        "environment": _environment_key(project),
                        "requirements": requirements,
                        "duration_seconds": 0.0,
                    }
                )
                _write_report(report)
                continue

            timeout = execution_timeout_for_notebook(notebook)
            notebook_started = monotonic()
            entry: dict[str, Any] = {
                "path": relative.as_posix(),
                "status": "running",
                "environment": _environment_key(project),
                "requirements": requirements,
                "timeout": timeout,
                "started_at": datetime.now(timezone.utc).isoformat(),
            }
            report["notebooks"].append(entry)
            report["active_notebook"] = {
                "path": relative.as_posix(),
                "started_at": entry["started_at"],
                "timeout": timeout,
            }
            _write_report(report)
            try:
                key = _cache_key(
                    notebook,
                    project,
                    env["python"],
                    docs_fingerprint,
                    code_fingerprint,
                    env["packages"],
                    env["activation_environment"],
                )
                cache_path = CACHE_DIR / _slug(relative)
                cached = None if force else _read_cache(cache_path, key)
                if cached is not None:
                    _restore_cache(cache_path, cached, stage_notebook, STAGE_DIR)
                    status = "cached"
                else:
                    print(f"EXECUTING {relative} ({env['python']})")
                    before = _scan_files(STAGE_DIR)
                    kernel_name = "corneto-" + hashlib.sha256(_environment_key(project).encode()).hexdigest()[:12]
                    with tempfile.TemporaryDirectory(prefix="corneto-notebook-") as temp:
                        executed_output = Path(temp) / "output.ipynb"
                        execute_notebook(
                            stage_notebook,
                            executed_output,
                            python=env["python"],
                            cwd=stage_notebook.parent,
                            kernel_root=env["kernel_root"],
                            kernel_name=kernel_name,
                            timeout=timeout,
                            environment=env["activation_environment"],
                        )
                        shutil.copy2(executed_output, stage_notebook)
                        after = _scan_files(STAGE_DIR)
                        _save_cache(
                            cache_path,
                            key,
                            executed_output,
                            before,
                            after,
                            STAGE_DIR,
                            stage_notebook,
                        )
                    status = "executed"
            except subprocess.TimeoutExpired as exc:
                entry.update(
                    status="timed-out",
                    error=f"Notebook exceeded its {timeout}-second wall-clock timeout.",
                    finished_at=datetime.now(timezone.utc).isoformat(),
                    duration_seconds=round(monotonic() - notebook_started, 3),
                )
                report["active_notebook"] = None
                _write_report(report)
                raise exc
            except subprocess.CalledProcessError as exc:
                detail = (exc.stderr or "").strip()
                detail = detail[-4000:] if detail else ""
                if exc.returncode != 124:
                    entry.update(
                        status="failed",
                        error=f"Notebook worker exited with code {exc.returncode}." + (f"\n{detail}" if detail else ""),
                        finished_at=datetime.now(timezone.utc).isoformat(),
                        duration_seconds=round(monotonic() - notebook_started, 3),
                    )
                    report["active_notebook"] = None
                    _write_report(report)
                    raise
                entry.update(
                    status="timed-out",
                    error=f"Notebook exceeded its {timeout}-second wall-clock timeout.",
                    finished_at=datetime.now(timezone.utc).isoformat(),
                    duration_seconds=round(monotonic() - notebook_started, 3),
                )
                report["active_notebook"] = None
                _write_report(report)
                raise
            except KeyboardInterrupt:
                entry.update(
                    status="interrupted",
                    error="Notebook execution was interrupted.",
                    finished_at=datetime.now(timezone.utc).isoformat(),
                    duration_seconds=round(monotonic() - notebook_started, 3),
                )
                report["active_notebook"] = None
                _write_report(report)
                raise
            except Exception as exc:
                entry.update(
                    status="failed",
                    error=f"{type(exc).__name__}: {exc}",
                    finished_at=datetime.now(timezone.utc).isoformat(),
                    duration_seconds=round(monotonic() - notebook_started, 3),
                )
                report["active_notebook"] = None
                _write_report(report)
                raise
            entry.update(
                status=status,
                finished_at=datetime.now(timezone.utc).isoformat(),
                duration_seconds=round(monotonic() - notebook_started, 3),
            )
            print(f"{status.upper()} {relative}")
            report["active_notebook"] = None
            _write_report(report)

        sphinx_command = [
            sys.executable,
            "-m",
            "sphinx",
            "-b",
            "html",
            "-E",
            "-a",
            "-D",
            "nb_execution_mode=off",
            str(STAGE_DIR),
            str(HTML_DIR),
        ]
        if HTML_DIR.exists():
            shutil.rmtree(HTML_DIR)
        _run(sphinx_command, PROJECT_ROOT)
        report["build"] = "success"
        return 0
    except Exception as exc:
        report["build"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    except KeyboardInterrupt:
        report["build"] = "interrupted"
        report["error"] = "Build interrupted by user."
        raise
    finally:
        report["active_notebook"] = None
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        report["duration_seconds"] = round(monotonic() - build_started, 3)
        _write_report(report)


def parse_args() -> argparse.Namespace:
    """Parse the docs build command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-execute every eligible notebook instead of using cached executions.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Include notebooks marked optional in their metadata.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    sys.exit(build(force=args.force, include_optional=args.all))
