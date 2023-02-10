import urllib.request
import re


def get_latest_version(
    url="https://raw.githubusercontent.com/saezlab/corneto/main/pyproject.toml",
    timeout=5,
):
    try:
        response = urllib.request.urlopen(url, timeout=timeout)
        content = response.read().decode()
        match = re.search(r'version\s*=\s*"(.*)"', content)
        if match:
            version = match.group(1)
            return version
    except Exception as e:
        return None


def info():
    from corneto import __version__
    from corneto.backend import available_backends, DEFAULT_BACKEND, DEFAULT_SOLVER

    latest = get_latest_version()
    if latest == __version__:
        print(f"CORNETO v{__version__} (up to date)")
    else:
        if latest:
            print(f"CORNETO v{__version__} (latest: v{latest})")
        else:
            print(f"CORNETO v{__version__}")
    print(
        "Available backends: ",
        ", ".join([str(e) + f" v{e.version()}" for e in available_backends()]),
    )
    if DEFAULT_BACKEND:
        print("Default backend (corneto.K):", str(DEFAULT_BACKEND))
        print(
            f"Available solvers for {str(DEFAULT_BACKEND)}:",
            ", ".join([s for s in DEFAULT_BACKEND.available_solvers()]),
        )
    else:
        print("No backend detected in the system. Please install Cvxpy or PICOS.")

    try:
        import graphviz

        print(f"Graphviz available: v{graphviz.__version__}.")
    except Exception as e:
        print("Graphviz not installed.")
    print("https://github.com/saezlab/corneto")
