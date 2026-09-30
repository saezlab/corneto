from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import nbformat
import pytest

DOCS_DIR = Path(__file__).resolve().parents[1] / "docs"
sys.path.insert(0, str(DOCS_DIR))

import build_docs  # noqa: E402
from _notebook_execution import (  # noqa: E402
    check_requirements,
    execute_notebook,
    execution_timeout_for_notebook,
    optional_reason_for_notebook,
    pixi_environment,
    requirements_for_notebook,
)
from build_docs import (  # noqa: E402
    _cache_key,
    _hash_document_inputs,
    _read_cache,
    _restore_cache,
    _save_cache,
    _scan_files,
)


def write_notebook(path: Path, *, source: str = "", metadata: dict | None = None) -> None:
    notebook = nbformat.v4.new_notebook(
        cells=[nbformat.v4.new_code_cell(source)] if source else [],
        metadata=metadata or {},
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(notebook, path)


def test_metadata_declares_requirements_and_timeout(tmp_path: Path) -> None:
    notebook = tmp_path / "requires-gurobi.ipynb"
    write_notebook(
        notebook,
        metadata={
            "corneto": {
                "requires": ["Gurobi", "gurobi"],
                "optional": True,
                "optional_reason": "Slow optimization tutorial.",
                "execution_timeout": 1200,
            }
        },
    )

    assert requirements_for_notebook(notebook) == ["gurobi"]
    assert execution_timeout_for_notebook(notebook) == 1200


def test_optional_metadata_requires_a_reason_and_allows_extended_timeout(
    tmp_path: Path,
) -> None:
    notebook = tmp_path / "slow.ipynb"
    write_notebook(
        notebook,
        metadata={
            "corneto": {
                "optional": True,
                "optional_reason": "Repeated optimization solves.",
                "execution_timeout": 3600,
            }
        },
    )

    assert optional_reason_for_notebook(notebook) == "Repeated optimization solves."
    assert execution_timeout_for_notebook(notebook) == 3600

    write_notebook(notebook, metadata={"corneto": {"optional": True}})
    with pytest.raises(ValueError, match="optional_reason"):
        optional_reason_for_notebook(notebook)


def test_extended_timeout_requires_optional_metadata(tmp_path: Path) -> None:
    notebook = tmp_path / "too-long.ipynb"
    write_notebook(notebook, metadata={"corneto": {"execution_timeout": 601}})

    with pytest.raises(ValueError, match="require optional=true"):
        execution_timeout_for_notebook(notebook)


def test_requirement_check_returns_unavailable_reason_in_selected_environment(
    tmp_path: Path,
) -> None:
    python = tmp_path / "tutorial-python"
    commands = []
    selected_environment = {"CORNETO_ENV_SENTINEL": "tutorial-env", "PATH": "/tutorial/bin"}

    def failed_check(command, **kwargs):
        commands.append((command, kwargs))
        return SimpleNamespace(returncode=2, stderr="Restricted license", stdout="")

    result = check_requirements(
        ["gurobi"],
        python,
        tmp_path,
        environment=selected_environment,
        runner=failed_check,
    )

    assert result == {"gurobi": "Restricted license"}
    assert commands[0][0][0] == str(python)
    assert "check_gurobi" in commands[0][0][2]
    assert commands[0][1]["env"] == selected_environment
    assert commands[0][1]["timeout"] == 60


def test_gurobi_preflight_timeout_is_an_unavailable_reason(tmp_path: Path) -> None:
    def timed_out(command, **kwargs):
        assert kwargs["timeout"] == 60
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])

    result = check_requirements(["gurobi"], sys.executable, tmp_path, runner=timed_out)

    assert "timed out after 60 seconds" in result["gurobi"]


def test_gurobi_preflight_worker_crash_fails_setup(tmp_path: Path) -> None:
    def crashed(command, **kwargs):
        return SimpleNamespace(returncode=1, stderr="ModuleNotFoundError: corneto", stdout="")

    with pytest.raises(RuntimeError, match="worker failed with exit code 1"):
        check_requirements(["gurobi"], sys.executable, tmp_path, runner=crashed)


def test_pixi_environment_merges_shell_hook_activation_variables(tmp_path: Path) -> None:
    base = {"PATH": "/main/bin", "KEEP": "main"}

    def shell_hook(command, **kwargs):
        assert command == ["pixi", "shell-hook", "--json"]
        assert kwargs["cwd"] == str(tmp_path)
        return SimpleNamespace(
            stdout=json.dumps(
                {"environment_variables": {"PATH": "/tutorial/bin:/main/bin", "CONDA_PREFIX": "/pixi/env"}}
            )
        )

    assert pixi_environment(tmp_path, base_environment=base, runner=shell_hook) == {
        "PATH": "/tutorial/bin:/main/bin",
        "KEEP": "main",
        "CONDA_PREFIX": "/pixi/env",
    }


def test_unknown_requirement_is_a_configuration_error(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unsupported notebook requirements"):
        check_requirements(["unknown-solver"], sys.executable, tmp_path)


def test_execution_uses_selected_python_and_notebook_directory(tmp_path: Path) -> None:
    source = tmp_path / "guide" / "tiny.ipynb"
    output = tmp_path / "executed.ipynb"
    write_notebook(
        source,
        source="from pathlib import Path\nPath('relative-result.txt').write_text(str(6 * 7))\n42",
    )

    execute_notebook(
        source,
        output,
        python=Path(sys.executable),
        cwd=source.parent,
        kernel_root=tmp_path / "jupyter",
        kernel_name="corneto-test",
        timeout=60,
    )

    executed = nbformat.read(output, as_version=4)
    assert (source.parent / "relative-result.txt").read_text() == "42"
    assert executed.cells[0].outputs[-1].data["text/plain"] == "42"


def test_kernel_inherits_selected_pixi_environment(tmp_path: Path) -> None:
    selected_bin = tmp_path / "tutorial-bin"
    selected_bin.mkdir()
    executable = selected_bin / "selected-tool"
    executable.write_text("#!/bin/sh\nexit 0\n")
    executable.chmod(0o755)
    source = tmp_path / "tutorial" / "environment.ipynb"
    output = tmp_path / "executed.ipynb"
    write_notebook(
        source,
        source=(
            "import os, shutil\n"
            "assert os.environ['CORNETO_ENV_SENTINEL'] == 'tutorial-env'\n"
            "assert shutil.which('selected-tool') == os.environ['CORNETO_EXPECTED_TOOL']\n"
            "os.environ['CORNETO_ENV_SENTINEL']"
        ),
    )

    execute_notebook(
        source,
        output,
        python=Path(sys.executable),
        cwd=source.parent,
        kernel_root=tmp_path / "jupyter",
        kernel_name="corneto-environment-test",
        timeout=60,
        environment={
            "CORNETO_ENV_SENTINEL": "tutorial-env",
            "CORNETO_EXPECTED_TOOL": str(executable),
            "PATH": str(selected_bin),
        },
    )

    executed = nbformat.read(output, as_version=4)
    assert executed.cells[0].outputs[-1].data["text/plain"] == "'tutorial-env'"


def test_total_execution_timeout_stops_before_the_next_cell_and_writes_no_output(
    tmp_path: Path,
) -> None:
    source = tmp_path / "guide" / "slow.ipynb"
    output = tmp_path / "executed.ipynb"
    continuation = source.parent / "continued.txt"
    pid_file = source.parent / "kernel-pid.txt"
    notebook = nbformat.v4.new_notebook(
        cells=[
            nbformat.v4.new_code_cell(
                "import os, time\n"
                f"from pathlib import Path; Path({str(pid_file)!r}).write_text(str(os.getpid()))\n"
                "time.sleep(1)"
            ),
            nbformat.v4.new_code_cell("import time\ntime.sleep(7)"),
            nbformat.v4.new_code_cell(f"from pathlib import Path; Path({str(continuation)!r}).write_text('continued')"),
        ]
    )
    source.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(notebook, source)

    with pytest.raises((subprocess.TimeoutExpired, subprocess.CalledProcessError)) as exc_info:
        execute_notebook(
            source,
            output,
            python=Path(sys.executable),
            cwd=source.parent,
            kernel_root=tmp_path / "jupyter",
            kernel_name="corneto-timeout-test",
            timeout=4,
        )

    if isinstance(exc_info.value, subprocess.CalledProcessError):
        assert exc_info.value.returncode == 124

    assert not continuation.exists()
    assert not output.exists()
    pid = int(pid_file.read_text())
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


@pytest.mark.skipif(sys.platform == "win32", reason="SIGINT cleanup test requires POSIX signals")
def test_ctrl_c_cleans_up_kernel_without_writing_notebook_output(tmp_path: Path) -> None:
    source = tmp_path / "guide" / "interrupted.ipynb"
    output = tmp_path / "interrupted-output.ipynb"
    continuation = source.parent / "continued.txt"
    pid_file = source.parent / "kernel-pid.txt"
    notebook = nbformat.v4.new_notebook(
        cells=[
            nbformat.v4.new_code_cell(
                "import os, time\n"
                f"from pathlib import Path; Path({str(pid_file)!r}).write_text(str(os.getpid()))\n"
                "time.sleep(30)"
            ),
            nbformat.v4.new_code_cell(f"from pathlib import Path; Path({str(continuation)!r}).write_text('continued')"),
        ]
    )
    source.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(notebook, source)

    def interrupt_worker(command, **kwargs):
        process = subprocess.Popen(command, cwd=kwargs["cwd"], env=kwargs["env"])
        deadline = time.monotonic() + 15
        while not pid_file.exists() and process.poll() is None and time.monotonic() < deadline:
            time.sleep(0.05)
        assert pid_file.exists(), "worker did not start the first notebook cell"
        process.send_signal(signal.SIGINT)
        returncode = process.wait(timeout=10)
        if kwargs.get("check") and returncode:
            raise subprocess.CalledProcessError(returncode, command)
        return subprocess.CompletedProcess(command, returncode)

    with pytest.raises(subprocess.CalledProcessError):
        execute_notebook(
            source,
            output,
            python=Path(sys.executable),
            cwd=source.parent,
            kernel_root=tmp_path / "jupyter",
            kernel_name="corneto-interrupt-test",
            timeout=60,
            runner=interrupt_worker,
        )

    assert not continuation.exists()
    assert not output.exists()
    kernel_pid = int(pid_file.read_text())
    with pytest.raises(ProcessLookupError):
        os.kill(kernel_pid, 0)


def test_notebook_execution_errors_propagate(tmp_path: Path) -> None:
    source = tmp_path / "fails.ipynb"
    write_notebook(source, source="raise RuntimeError('not a dependency skip')")

    with pytest.raises(subprocess.CalledProcessError):
        execute_notebook(
            source,
            tmp_path / "failed-output.ipynb",
            python=Path(sys.executable),
            cwd=tmp_path,
            kernel_root=tmp_path / "jupyter",
            kernel_name="corneto-failure-test",
            timeout=60,
        )


def test_notebook_execution_failure_relays_worker_stderr(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    source = tmp_path / "fails-with-diagnostic.ipynb"
    write_notebook(source, source="raise RuntimeError('diagnostic from notebook cell')")

    with pytest.raises(subprocess.CalledProcessError) as exc_info:
        execute_notebook(
            source,
            tmp_path / "failed-output.ipynb",
            python=Path(sys.executable),
            cwd=tmp_path,
            kernel_root=tmp_path / "jupyter",
            kernel_name="corneto-failure-diagnostic-test",
            timeout=60,
        )

    terminal_error = capsys.readouterr().err
    assert "diagnostic from notebook cell" in terminal_error
    assert "diagnostic from notebook cell" in exc_info.value.stderr


def test_cache_restores_notebook_output_and_generated_artifacts(tmp_path: Path) -> None:
    stage = tmp_path / "stage"
    notebook = stage / "tutorial" / "example.ipynb"
    artifact = stage / "tutorial" / "generated.txt"
    cache_path = tmp_path / "cache" / "tutorial__example.ipynb"
    output = tmp_path / "executed.ipynb"
    write_notebook(notebook, source="1 + 1")
    before = _scan_files(stage)
    write_notebook(notebook, source="1 + 1")
    artifact.write_text("created by notebook")
    write_notebook(output, source="1 + 1")
    after = _scan_files(stage)
    _save_cache(cache_path, "cache-key", output, before, after, stage, notebook)

    notebook.write_text("source version")
    artifact.unlink()
    cached = _read_cache(cache_path, "cache-key")
    assert cached is not None
    _restore_cache(cache_path, cached, notebook, stage)

    assert "cells" in json.loads(notebook.read_text())
    assert artifact.read_text() == "created by notebook"
    assert _read_cache(cache_path, "different-key") is None


def test_local_document_assets_change_the_input_fingerprint(tmp_path: Path) -> None:
    asset = tmp_path / "data.csv"
    asset.write_text("value\n1\n")
    before = _hash_document_inputs(tmp_path)
    asset.write_text("value\n2\n")

    assert _hash_document_inputs(tmp_path) != before


def test_cache_key_changes_with_installed_package_versions(tmp_path: Path) -> None:
    notebook = tmp_path / "example.ipynb"
    write_notebook(notebook, source="1 + 1")
    args = (notebook, tmp_path, Path(sys.executable), "docs", "corneto", "packages-a")

    assert _cache_key(*args) != _cache_key(*args[:-1], "packages-b")
    assert _cache_key(*args, {"PATH": "/first/bin"}) != _cache_key(*args, {"PATH": "/second/bin"})


def configure_build_test(tmp_path: Path, monkeypatch, *, gurobi_available: bool = True):
    docs = tmp_path / "docs"
    build_dir = docs / "_build"
    project = tmp_path / "project"
    regular = docs / "guide" / "regular.ipynb"
    optional = docs / "guide" / "slow-gurobi.ipynb"
    excluded = docs / "custom-excluded" / "must-not-run.ipynb"
    checkpoint = docs / "guide" / ".ipynb_checkpoints" / "must-not-run.ipynb"
    generated = docs / "_build" / "generated.ipynb"
    hidden = docs / "guide" / "_draft.ipynb"
    (docs / "conf.py").parent.mkdir(parents=True, exist_ok=True)
    (docs / "conf.py").write_text(
        "exclude_patterns = ['custom-excluded/**', '**/_*.ipynb', '**.ipynb_checkpoints']\n",
        encoding="utf-8",
    )
    write_notebook(regular, source="1 + 1")
    write_notebook(excluded, source="raise AssertionError('excluded notebook ran')")
    write_notebook(checkpoint, source="raise AssertionError('checkpoint notebook ran')")
    write_notebook(generated, source="raise AssertionError('generated notebook ran')")
    write_notebook(hidden, source="raise AssertionError('Sphinx-excluded notebook ran')")
    write_notebook(
        optional,
        source="2 + 2",
        metadata={
            "corneto": {
                "requires": ["gurobi"],
                "optional": True,
                "optional_reason": "Repeated mixed-integer solves.",
                "execution_timeout": 3600,
            }
        },
    )

    monkeypatch.setattr(build_docs, "DOCS_DIR", docs)
    monkeypatch.setattr(build_docs, "PROJECT_ROOT", project)
    monkeypatch.setattr(build_docs, "BUILD_DIR", build_dir)
    monkeypatch.setattr(build_docs, "STAGE_DIR", build_dir / "source")
    monkeypatch.setattr(build_docs, "HTML_DIR", build_dir / "html")
    monkeypatch.setattr(build_docs, "CACHE_DIR", build_dir / "notebook-cache")
    monkeypatch.setattr(build_docs, "REPORT_PATH", build_dir / "notebook-report.json")

    executed = []
    preflights = []
    execution_environments = []
    progress_snapshots = []

    def execute(source, output, **kwargs):
        executed.append(source.name)
        execution_environments.append(kwargs["environment"])
        progress_snapshots.append(json.loads((build_dir / "notebook-report.json").read_text()))
        shutil.copy2(source, output)

    def check(requirements, python, cwd, environment=None):
        preflights.append((set(requirements), environment))
        return {requirement: None if gurobi_available else "license unavailable" for requirement in requirements}

    monkeypatch.setattr(
        build_docs,
        "_prepare_environment",
        lambda _: (Path(sys.executable), {"PATH": "/tutorial/bin", "CORNETO_TEST_ENV": "selected"}),
    )
    monkeypatch.setattr(build_docs, "check_requirements", check)
    monkeypatch.setattr(build_docs, "environment_fingerprint", lambda *args, **kwargs: "test-packages")
    monkeypatch.setattr(build_docs, "_hash_document_inputs", lambda: "test-docs")
    monkeypatch.setattr(build_docs, "_hash_project_source", lambda: "test-corneto")
    monkeypatch.setattr(build_docs, "execute_notebook", execute)
    monkeypatch.setattr(build_docs, "_run", lambda *args, **kwargs: None)
    return executed, preflights, build_dir / "notebook-report.json", execution_environments, progress_snapshots


def test_docs_force_skips_optional_notebooks_by_default(tmp_path: Path, monkeypatch) -> None:
    executed, preflights, report_path, _, _ = configure_build_test(tmp_path, monkeypatch)

    assert build_docs.build(force=True) == 0

    report = json.loads(report_path.read_text())
    statuses = {item["path"]: item["status"] for item in report["notebooks"]}
    assert executed == ["regular.ipynb"]
    assert preflights == []
    assert statuses["guide/slow-gurobi.ipynb"] == "optional-skipped"
    assert statuses["custom-excluded/must-not-run.ipynb"] == "excluded"
    assert statuses["guide/_draft.ipynb"] == "excluded"
    assert statuses["guide/.ipynb_checkpoints/must-not-run.ipynb"] == "excluded"


def test_docs_all_includes_optional_notebooks_and_keeps_gurobi_preflight(tmp_path: Path, monkeypatch) -> None:
    executed, preflights, report_path, execution_environments, _ = configure_build_test(tmp_path, monkeypatch)

    assert build_docs.build(force=True, include_optional=True) == 0

    report = json.loads(report_path.read_text())
    statuses = {item["path"]: item["status"] for item in report["notebooks"]}
    assert set(executed) == {"regular.ipynb", "slow-gurobi.ipynb"}
    assert preflights == [({"gurobi"}, {"PATH": "/tutorial/bin", "CORNETO_TEST_ENV": "selected"})]
    assert execution_environments == [{"PATH": "/tutorial/bin", "CORNETO_TEST_ENV": "selected"}] * 2
    assert statuses["guide/slow-gurobi.ipynb"] == "executed"


def test_docs_all_still_skips_optional_gurobi_notebooks_without_a_license(tmp_path: Path, monkeypatch) -> None:
    executed, preflights, report_path, _, _ = configure_build_test(tmp_path, monkeypatch, gurobi_available=False)

    assert build_docs.build(force=True, include_optional=True) == 0

    report = json.loads(report_path.read_text())
    statuses = {item["path"]: item["status"] for item in report["notebooks"]}
    assert executed == ["regular.ipynb"]
    assert preflights == [({"gurobi"}, {"PATH": "/tutorial/bin", "CORNETO_TEST_ENV": "selected"})]
    assert statuses["guide/slow-gurobi.ipynb"] == "skipped"


def test_integrated_build_cache_hit_force_bypass_and_source_preservation(tmp_path: Path, monkeypatch) -> None:
    executed, _, report_path, _, progress_snapshots = configure_build_test(tmp_path, monkeypatch)
    source = tmp_path / "docs" / "guide" / "regular.ipynb"
    original_source = source.read_bytes()

    assert build_docs.build() == 0
    first = json.loads(report_path.read_text())
    first_status = {item["path"]: item for item in first["notebooks"]}
    assert first_status["guide/regular.ipynb"]["status"] == "executed"
    assert first_status["guide/regular.ipynb"]["duration_seconds"] >= 0
    assert first_status["guide/regular.ipynb"]["started_at"]
    assert first_status["guide/regular.ipynb"]["finished_at"]
    assert first["duration_seconds"] >= 0
    assert progress_snapshots[0]["active_notebook"]["path"] == "guide/regular.ipynb"
    active_entry = next(item for item in progress_snapshots[0]["notebooks"] if item["path"] == "guide/regular.ipynb")
    assert active_entry["status"] == "running"
    assert executed == ["regular.ipynb"]

    assert build_docs.build() == 0
    second = json.loads(report_path.read_text())
    second_status = {item["path"]: item for item in second["notebooks"]}
    assert second_status["guide/regular.ipynb"]["status"] == "cached"
    assert len(executed) == 1

    assert build_docs.build(force=True) == 0
    forced = json.loads(report_path.read_text())
    forced_status = {item["path"]: item for item in forced["notebooks"]}
    assert forced_status["guide/regular.ipynb"]["status"] == "executed"
    assert len(executed) == 2
    assert source.read_bytes() == original_source


def test_integrated_timeout_is_reported_with_duration_and_cleared_active_state(tmp_path: Path, monkeypatch) -> None:
    _, _, report_path, _, _ = configure_build_test(tmp_path, monkeypatch)

    def timeout(source, output, **kwargs):
        raise subprocess.CalledProcessError(124, ["notebook-worker"])

    monkeypatch.setattr(build_docs, "execute_notebook", timeout)
    with pytest.raises(subprocess.CalledProcessError):
        build_docs.build(force=True)

    report = json.loads(report_path.read_text())
    entry = next(item for item in report["notebooks"] if item["path"] == "guide/regular.ipynb")
    assert entry["status"] == "timed-out"
    assert "600-second" in entry["error"]
    assert entry["duration_seconds"] >= 0
    assert report["active_notebook"] is None


def test_integrated_failure_report_preserves_worker_diagnostic(tmp_path: Path, monkeypatch) -> None:
    _, _, report_path, _, _ = configure_build_test(tmp_path, monkeypatch)

    def failed(source, output, **kwargs):
        raise subprocess.CalledProcessError(
            1,
            ["notebook-worker"],
            stderr="AttributeError: selected tutorial API is unavailable",
        )

    monkeypatch.setattr(build_docs, "execute_notebook", failed)
    with pytest.raises(subprocess.CalledProcessError):
        build_docs.build(force=True)

    report = json.loads(report_path.read_text())
    entry = next(item for item in report["notebooks"] if item["path"] == "guide/regular.ipynb")
    assert entry["status"] == "failed"
    assert "AttributeError: selected tutorial API is unavailable" in entry["error"]
