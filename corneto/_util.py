import contextlib
import hashlib
import json
import os
import pickle
import warnings
from collections import OrderedDict
from itertools import filterfalse
from typing import Any, Callable, Dict, Iterable, Iterator, Optional, Set, TypeVar

import numpy as np
from numpy.linalg import svd

T = TypeVar("T")


@contextlib.contextmanager
def suppress_output(
    suppress_stdout: bool = False,
    suppress_stderr: bool = True,
    suppress_warnings: bool = True,
    stdout_redirect_file: Optional[str] = None,
    stderr_redirect_file: Optional[str] = None,
) -> Iterator[None]:
    """A context manager to selectively suppress stdout, stderr, and warnings.

    Args:
        suppress_stdout: If True, suppresses stdout. Defaults to True.
        suppress_stderr: If True, suppresses stderr. Defaults to True.
        suppress_warnings: If True, suppresses warnings. Defaults to True.
        stdout_redirect_file: If provided, redirects stdout to this file path instead of /dev/null.
        stderr_redirect_file: If provided, redirects stderr to this file path instead of /dev/null.
    """
    with contextlib.ExitStack() as stack:
        # Handle stdout redirection/suppression
        if suppress_stdout or stdout_redirect_file:
            stdout_target = open(stdout_redirect_file if stdout_redirect_file else os.devnull, "w")
            stack.enter_context(stdout_target)
            stack.enter_context(contextlib.redirect_stdout(stdout_target))

        # Handle stderr redirection/suppression
        if suppress_stderr or stderr_redirect_file:
            stderr_target = open(stderr_redirect_file if stderr_redirect_file else os.devnull, "w")
            stack.enter_context(stderr_target)
            stack.enter_context(contextlib.redirect_stderr(stderr_target))

        # Suppress warnings if requested
        if suppress_warnings:
            stack.enter_context(warnings.catch_warnings())
            warnings.simplefilter("ignore")

        yield


def obj_content_hash(obj) -> str:
    obj_serialized = pickle.dumps(obj)
    hash_obj = hashlib.sha256()
    hash_obj.update(obj_serialized)
    return hash_obj.hexdigest()


def canonicalize(obj):
    """Recursively convert an object into a JSON-serializable structure
    that is independent of internal ordering.
    """
    if isinstance(obj, dict):
        # Convert dictionary keys to strings and sort the keys
        return {str(key): canonicalize(obj[key]) for key in sorted(obj.keys(), key=lambda x: str(x))}
    elif isinstance(obj, (list, tuple)):
        # Recursively canonicalize each element in the list or tuple
        return [canonicalize(item) for item in obj]
    elif isinstance(obj, set):
        # Convert sets to a sorted list (sorting based on JSON string representation)
        return sorted(
            [canonicalize(item) for item in obj],
            key=lambda x: json.dumps(x, sort_keys=True),
        )
    elif isinstance(obj, (int, float, str, bool)) or obj is None:
        return obj
    else:
        # For non-standard objects, try using the __dict__ attribute if available
        if hasattr(obj, "__dict__"):
            return canonicalize(obj.__dict__)
        else:
            # Fall back to a string representation
            return str(obj)


def obj_canonicalized_hash(obj) -> str:
    # First canonicalize the object
    canonical_obj = canonicalize(obj)
    # Serialize the canonical object to a JSON string.
    # 'sort_keys=True' ensures consistent key order,
    # and separators remove unnecessary whitespace.
    obj_serialized = json.dumps(canonical_obj, sort_keys=True, separators=(",", ":")).encode("utf-8")
    # Compute the SHA256 hash of the serialized bytes
    hash_obj = hashlib.sha256()
    hash_obj.update(obj_serialized)
    return hash_obj.hexdigest()


def unique_iter(iterable: Iterable[T], key: Optional[Callable[[T], Any]] = None) -> Iterable[T]:
    # Based on https://iteration-utilities.readthedocs.io/en/latest/generated/unique_everseen.html
    seen: Set[Any] = set()
    seen_add = seen.add
    if key is None:
        for element in filterfalse(seen.__contains__, iterable):
            seen_add(element)
            yield element
    else:
        for element in iterable:
            k = key(element)
            if k not in seen:
                seen_add(k)
                yield element


def uiter(iterable: Iterable[T], key: Optional[Callable[[T], Any]] = None) -> Iterable[T]:
    seen: Set[Any] = set()
    seen_add = seen.add  # micro-optimization

    for element in iterable:
        val = element if key is None else key(element)
        if val not in seen:
            seen_add(val)
            yield element


def get_latest_version(
    url="https://raw.githubusercontent.com/saezlab/corneto/main/pyproject.toml",
    timeout=5,
):
    import re
    import urllib.request

    try:
        response = urllib.request.urlopen(url, timeout=timeout)
        content = response.read().decode()
        match = re.search(r'version\s*=\s*"(.*)"', content)
        if match:
            version = match.group(1)
            return version
    except Exception:
        return None


class DisplayInspector:
    # From: https://stackoverflow.com/questions/70768390/detecting-if-ipython-notebook-is-outputting-to-a-terminal
    """Objects that display as HTML or text."""

    def __init__(self) -> None:
        self.status = None

    def _repr_html_(self) -> str:
        self.status = "HTML"
        return ""

    def __repr__(self) -> str:
        self.status = "Plain"
        return ""


def supports_html() -> bool:
    # From: https://stackoverflow.com/questions/70768390/detecting-if-ipython-notebook-is-outputting-to-a-terminal
    import sys

    """Test whether current runtime supports HTML."""
    if "IPython" not in sys.modules or "IPython.display" not in sys.modules:
        return False

    from IPython.display import display

    inspector = DisplayInspector()
    display(inspector)
    return inspector.status == "HTML"


def _get_info() -> Dict[str, Dict]:
    from corneto import __version__
    from corneto.backend import DEFAULT_BACKEND, available_backends

    info: Dict[str, Dict] = OrderedDict()

    # latest = get_latest_version()
    # if latest == __version__:
    #    cv = f"v{__version__} (up to date)"
    # else:
    #    if latest:
    #        cv = f"v{__version__} (latest stable: v{latest})"
    #    else:
    cv = f"v{__version__}"
    info["corneto_version"] = {
        "title": "Installed version",
        "message": cv,
        "value": __version__,
    }
    info["backends"] = {
        "title": "Available backends",
        "message": ", ".join([str(e) + f" v{e.version()}" for e in available_backends()]),
        "value": available_backends(),
    }
    info["default_backend"] = {
        "title": "Default backend (corneto.opt)",
        "message": "No backend detected, please install CVXPY or PICOS",
        "value": None,
    }
    info["available_solvers"] = {
        "title": "Installed solvers",
        "message": "No installed solvers",
        "value": [],
    }
    if DEFAULT_BACKEND:
        info["default_backend"]["message"] = str(DEFAULT_BACKEND)
        info["default_backend"]["value"] = DEFAULT_BACKEND
        info["available_solvers"]["message"] = ", ".join([s for s in DEFAULT_BACKEND.available_solvers()])
        info["available_solvers"]["value"] = DEFAULT_BACKEND.available_solvers()
    info["graphviz_version"] = {
        "title": "Graphviz version",
        "message": "Graphviz not installed. To support plotting, please install graphviz with conda",
        "value": None,
    }
    try:
        import graphviz

        info["graphviz_version"]["message"] = f"v{graphviz.__version__}"
        info["graphviz_version"]["value"] = graphviz.__version__
    except Exception:
        pass

    info["installed_path"] = {
        "title": "Installed path",
        "message": os.path.dirname(__file__),
        "value": os.path.dirname(__file__),
    }
    info["repo_url"] = {
        "title": "Repository",
        "message": "https://github.com/saezlab/corneto",
        "value": "https://github.com/saezlab/corneto",
    }
    return info


def info():
    info = _get_info()

    if supports_html():
        import base64
        from importlib.resources import files

        from IPython.display import HTML, display

        # logo_path = pkg_resources.resource_filename(__name__, "resources/logo.png")
        logo_path = files("corneto").joinpath("resources/logo.png")

        with open(logo_path, "rb") as f:
            img_bytes = f.read()
        b64img = base64.b64encode(img_bytes).decode("utf-8")
        html = f"""
        <table style='background-color:rgba(0, 0, 0, 0);'>
        <tr>
            <td style="min-width:85px">
                <img src="data:image/jpeg;base64,{b64img}" style="width: 100%; max-width:100px;" />
            </td>
            <td>
            <table>
                *
            </table>
            </td>
        </tr>
        </table>"""
        html_info = ""
        for k, v in info.items():
            title = v["title"]
            message = v["message"]
            if "_url" in k:
                message = f"<a href={message}>{message}</a>"
            html_info += f"<tr><td>{title}:</td><td style='text-align:left'>{message}</td></tr>"
        display(HTML(html.replace("*", html_info)))

    else:
        for v in info.values():
            title = v["title"]
            message = v["message"]
            print(f"{title}:", f"{message}")


def _info():
    from corneto import __version__
    from corneto.backend import DEFAULT_BACKEND, available_backends

    # latest = get_latest_version()
    # if latest == __version__:
    #    print(f"CORNETO v{__version__} (up to date)")
    # else:
    #    if latest:
    #        print(f"CORNETO v{__version__} (latest: v{latest})")
    #    else:
    print(f"CORNETO v{__version__}")
    print(
        "Available backends: ",
        ", ".join([str(e) + f" v{e.version()}" for e in available_backends()]),
    )
    if DEFAULT_BACKEND:
        print("Default backend (corneto.opt):", str(DEFAULT_BACKEND))
        print(
            f"Available solvers for {DEFAULT_BACKEND!s}:",
            ", ".join([s for s in DEFAULT_BACKEND.available_solvers()]),
        )
    else:
        print("No backend detected in the system. Please install CVXPY or PICOS.")

    try:
        import graphviz

        print(f"Graphviz available: v{graphviz.__version__}.")
    except Exception:
        print("Graphviz not installed.")
    print("https://github.com/saezlab/corneto")


def nullspace(A, atol=1e-13, rtol=0):
    # https://scipy-cookbook.readthedocs.io/items/RankNullspace.html
    A = np.atleast_2d(A)
    _, s, vh = svd(A)
    tol = max(atol, rtol * s[0])
    nnz = (s >= tol).sum()
    ns = vh[nnz:].conj().T
    return ns
