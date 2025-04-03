import pytest

from corneto.backend import CvxpyBackend, PicosBackend


def pytest_addoption(parser):
    """Add custom command line options for pytest.

    Args:
        parser: pytest argument parser

    Options:
        --solver: Specify solver for optimization backends (e.g., gurobi, SCIPY)
        --backend: Select specific backend to test (cvxpy or picos)
        --run-optional: Flag to enable optional tests
    """
    parser.addoption(
        "--solver",
        action="store",
        default=None,
        help="Specify solver for CvxpyBackend (e.g., gurobi, SCIPY)",
    )
    parser.addoption(
        "--backend",
        action="store",
        default=None,
        choices=["cvxpy", "picos"],
        help="Specify which backend to run tests with (cvxpy or picos)",
    )
    parser.addoption(
        "--run-optional", action="store_true", default=False, help="Run optional tests"
    )


@pytest.fixture
def solver(request):
    """Fixture that provides the solver specified in command line.

    Returns:
        str: Name of the solver or None if not specified
    """
    return request.config.getoption("--solver")


@pytest.fixture(params=["cvxpy", "picos"])
def backend(request, solver):
    """Fixture that provides optimization backend instances for testing.

    Parameterized to test both cvxpy and picos backends unless a specific
    backend is requested via --backend option.

    Args:
        request: pytest request object
        solver: solver fixture

    Returns:
        Backend: Configured CvxpyBackend or PicosBackend instance

    Skips:
        - If specified backend doesn't match current parameter
        - If specified solver is not available for the backend
    """
    selected_backend = request.config.getoption("--backend")
    backend_name = request.param

    # Skip if a specific backend was requested and this is not it
    if selected_backend and backend_name != selected_backend:
        pytest.skip(
            f"Skipping backend '{backend_name}' because --backend={selected_backend} was specified."
        )

    if backend_name == "cvxpy":
        opt = CvxpyBackend()
        solvers = [s.lower() for s in opt.available_solvers()]
        if solver and solver.lower() not in solvers:
            pytest.skip(
                f"Solver '{solver}' is not available in CvxpyBackend. Available solvers: {', '.join(solvers)}"
            )
        opt._default_solver = solver if solver else "SCIPY"
        return opt

    if backend_name == "picos":
        opt = PicosBackend()
        solvers = [s.lower() for s in opt.available_solvers()]
        if solver and solver.lower() not in solvers:
            pytest.skip(
                f"Solver '{solver}' is not available in PicosBackend. Available solvers: {', '.join(solvers)}"
            )
        opt._default_solver = solver if solver else "glpk"
        return opt
