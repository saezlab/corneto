"""Shared notebook metadata and execution helpers for CORNETO docs."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from time import monotonic
from typing import Any, Mapping


def load_notebook(path: Path) -> dict[str, Any]:
    """Load a notebook as JSON without modifying it."""
    return json.loads(path.read_text(encoding="utf-8"))


def requirements_for_notebook(path: Path) -> list[str]:
    """Return the optional execution requirements declared by a notebook."""
    metadata = load_notebook(path).get("metadata", {})
    requirements = metadata.get("corneto", {}).get("requires", [])
    if isinstance(requirements, str):
        requirements = [requirements]
    if not isinstance(requirements, list) or not all(
        isinstance(requirement, str) and requirement.strip() for requirement in requirements
    ):
        raise ValueError(f"{path}: metadata.corneto.requires must be a string or a list of strings")
    return sorted({requirement.strip().lower() for requirement in requirements})


def optional_reason_for_notebook(path: Path) -> str | None:
    """Return why a notebook is deferred from routine documentation builds."""
    metadata = load_notebook(path).get("metadata", {}).get("corneto", {})
    optional = metadata.get("optional", False)
    if not isinstance(optional, bool):
        raise ValueError(f"{path}: metadata.corneto.optional must be a boolean")
    reason = metadata.get("optional_reason")
    if optional and (not isinstance(reason, str) or not reason.strip()):
        raise ValueError(f"{path}: optional notebooks require metadata.corneto.optional_reason")
    if not optional and reason:
        raise ValueError(f"{path}: metadata.corneto.optional_reason requires optional=true")
    return reason.strip() if optional else None


def execution_timeout_for_notebook(path: Path, default: int = 600) -> int:
    """Get the notebook execution timeout, expressed in seconds."""
    metadata = load_notebook(path).get("metadata", {}).get("corneto", {})
    timeout = metadata.get("execution_timeout", default)
    if type(timeout) is not int or timeout < 1:
        raise ValueError(f"{path}: metadata.corneto.execution_timeout must be a positive integer")
    return timeout


def _gurobi_check_code() -> str:
    return (
        "import sys\n"
        "from corneto.utils import check_gurobi\n"
        "try:\n"
        "    check_gurobi(verbose=False, raise_on_error=True)\n"
        "except Exception as exc:\n"
        "    print(f'{type(exc).__name__}: {exc}', file=sys.stderr)\n"
        "    sys.exit(2)\n"
    )


def check_requirements(
    requirements: set[str] | list[str],
    python: Path | str,
    cwd: Path,
    *,
    environment: Mapping[str, str] | None = None,
    timeout: int = 60,
    runner=subprocess.run,
) -> dict[str, str | None]:
    """Check requirements in the interpreter that will execute the notebooks.

    A failed optional-dependency check is returned as a reason so callers can
    skip dependent notebooks. Unknown requirement names are configuration
    errors and raise immediately.
    """
    normalized = {requirement.strip().lower() for requirement in requirements}
    unknown = normalized - {"gurobi"}
    if unknown:
        raise ValueError(f"Unsupported notebook requirements: {', '.join(sorted(unknown))}")

    results: dict[str, str | None] = {}
    for requirement in sorted(normalized):
        command = [str(python), "-c", _gurobi_check_code()]
        try:
            result = runner(
                command,
                cwd=str(cwd),
                env=environment,
                text=True,
                capture_output=True,
                check=False,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            results[requirement] = f"Gurobi capability preflight timed out after {timeout} seconds"
            print(f"Requirement unavailable: {requirement}: {results[requirement]}")
            continue
        if result.returncode == 0:
            results[requirement] = None
            print(f"Requirement available: {requirement} ({python})")
        elif result.returncode == 2:
            detail = (result.stderr or result.stdout or "Gurobi capability check failed").strip()
            results[requirement] = detail.splitlines()[-1]
            print(f"Requirement unavailable: {requirement}: {results[requirement]}")
        else:
            detail = (result.stderr or result.stdout or "Gurobi capability check failed").strip()
            raise RuntimeError(f"{requirement} preflight worker failed with exit code {result.returncode}: {detail}")
    return results


def pixi_environment(
    project: Path,
    *,
    base_environment: Mapping[str, str] | None = None,
    runner=subprocess.run,
) -> dict[str, str]:
    """Return the project environment after Pixi activation."""
    result = runner(
        ["pixi", "shell-hook", "--json"],
        cwd=str(project),
        check=True,
        text=True,
        capture_output=True,
    )
    try:
        activation = json.loads(result.stdout)["environment_variables"]
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        raise RuntimeError(f"Pixi returned invalid activation environment for {project}") from exc
    if not isinstance(activation, dict) or not all(
        isinstance(key, str) and isinstance(value, str) for key, value in activation.items()
    ):
        raise RuntimeError(f"Pixi returned invalid activation environment for {project}")
    environment = dict(os.environ if base_environment is None else base_environment)
    environment.update(activation)
    return environment


def python_for_pixi(project: Path, *, runner=subprocess.run) -> Path:
    """Return the exact Python executable selected by a Pixi project."""
    command = [
        "pixi",
        "run",
        "python",
        "-c",
        "import sys; print(sys.executable)",
    ]
    result = runner(
        command,
        cwd=str(project),
        check=True,
        text=True,
        capture_output=True,
    )
    return Path(result.stdout.strip().splitlines()[-1]).resolve()


def environment_fingerprint(
    python: Path,
    cwd: Path,
    *,
    environment: Mapping[str, str] | None = None,
    runner=subprocess.run,
) -> str:
    """Capture the installed Python package versions for a notebook environment."""
    result = runner(
        [str(python), "-m", "pip", "freeze", "--all"],
        cwd=str(cwd),
        check=True,
        text=True,
        capture_output=True,
        env=environment,
    )
    return result.stdout


def execute_notebook(
    notebook: Path,
    output: Path,
    *,
    python: Path,
    cwd: Path,
    kernel_root: Path,
    kernel_name: str,
    timeout: int,
    environment: Mapping[str, str] | None = None,
    executor: Path | None = None,
    runner=subprocess.run,
) -> None:
    """Execute a notebook with nbclient and the given environment's Python."""
    kernel_dir = kernel_root / "kernels" / kernel_name
    kernel_dir.mkdir(parents=True, exist_ok=True)
    kernel_spec = {
        "argv": [str(python), "-m", "ipykernel_launcher", "-f", "{connection_file}"],
        "display_name": f"CORNETO ({kernel_name})",
        "language": "python",
    }
    (kernel_dir / "kernel.json").write_text(json.dumps(kernel_spec, indent=2) + "\n", encoding="utf-8")

    output.parent.mkdir(parents=True, exist_ok=True)
    worker = executor or Path(__file__).resolve()
    if worker == Path(__file__).resolve():
        command = [
            sys.executable,
            str(worker),
            "--worker",
            str(notebook),
            str(output),
            str(cwd),
            kernel_name,
            str(timeout),
        ]
    else:
        command = [
            str(executor),
            str(notebook),
            str(output),
            str(cwd),
            kernel_name,
            str(timeout),
        ]

    child_environment = os.environ.copy()
    if environment is not None:
        child_environment.update(environment)
    kernel_paths = [str(kernel_root)]
    if child_environment.get("JUPYTER_PATH"):
        kernel_paths.append(child_environment["JUPYTER_PATH"])
    child_environment["JUPYTER_PATH"] = os.pathsep.join(kernel_paths)
    try:
        runner(
            command,
            cwd=str(cwd),
            env=child_environment,
            check=True,
            timeout=timeout,
            stderr=subprocess.PIPE,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        if exc.stderr:
            print(exc.stderr, file=sys.stderr, end="" if exc.stderr.endswith("\n") else "\n")
        raise


def execute_worker(
    notebook: Path,
    output: Path,
    cwd: Path,
    kernel_name: str,
    timeout: int,
) -> None:
    """Worker process for notebook execution under the docs environment."""
    import nbformat
    from nbclient import NotebookClient
    from nbclient.exceptions import CellTimeoutError

    content = nbformat.read(notebook, as_version=4)
    started = monotonic()

    def remaining_timeout(_cell: Any) -> int:
        return max(1, int(timeout - (monotonic() - started)) - 1)

    client = NotebookClient(
        content,
        timeout=timeout,
        timeout_func=remaining_timeout,
        kernel_name=kernel_name,
        resources={"metadata": {"path": str(cwd)}},
        allow_errors=False,
    )
    try:
        executed = client.execute()
    except CellTimeoutError:
        # Distinguish a notebook's shared wall-clock budget from an execution
        # error so the build report can identify a timeout precisely.
        raise SystemExit(124) from None
    nbformat.write(executed, output)


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "--worker":
    _, _, notebook_arg, output_arg, cwd_arg, kernel_arg, timeout_arg = sys.argv
    execute_worker(
        Path(notebook_arg),
        Path(output_arg),
        Path(cwd_arg),
        kernel_arg,
        int(timeout_arg),
    )
