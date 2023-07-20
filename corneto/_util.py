
from collections import OrderedDict
import numpy as np
from typing import Dict, Optional
from numpy.linalg import svd

def get_latest_version(
    url="https://raw.githubusercontent.com/saezlab/corneto/main/pyproject.toml",
    timeout=5,
):
    import urllib.request
    import re
    try:
        response = urllib.request.urlopen(url, timeout=timeout)
        content = response.read().decode()
        match = re.search(r'version\s*=\s*"(.*)"', content)
        if match:
            version = match.group(1)
            return version
    except Exception as e:
        return None
    
def _support_html_output(force_html: bool = False):
    # from https://github.com/tqdm/tqdm/blob/master/tqdm/autonotebook.py
    # and https://github.com/tqdm/tqdm/blob/0bb91857eca0d4aea08f66cf1c8949abe0cd6b7a/tqdm/notebook.py#L38
    try:
        from IPython import get_ipython
        from IPython.display import HTML
        from IPython.core.display import display
        ipy = get_ipython()
        if ipy is None:
            return False
        if 'IPKernelApp' not in get_ipython().config:
            return False
        return True
    except:
        return False

def _get_info() -> Dict[str, Dict]:
    from corneto import __version__
    from corneto.backend import available_backends, DEFAULT_BACKEND
    info: Dict[str, Dict] = OrderedDict()

    latest = get_latest_version()
    if latest == __version__:
        cv = f"v{__version__} (up to date)"
    else:
        if latest:
            cv = f"v{__version__} (latest: v{latest})"
        else:
            cv = f"v{__version__}"
    info['corneto_version'] = {'title': 'Installed version', 'message': cv, 'value': __version__}
    info['backends'] = {
        'title': 'Available backends', 
        'message': ", ".join([str(e) + f" v{e.version()}" for e in available_backends()]), 
        'value': available_backends()
    }
    info['default_backend'] = {
        'title': 'Default backend (corneto.K)', 
        'message': "No backend detected, please install CVXPY or PICOS",
        'value': None
    }
    info['available_solvers'] = {
        'title': 'Installed solvers',
        'message': 'No installed solvers',
        'value': []
    }      
    if DEFAULT_BACKEND:
        info['default_backend']['message'] = str(DEFAULT_BACKEND)
        info['default_backend']['value'] = DEFAULT_BACKEND
        info['available_solvers']['message'] = ", ".join([s for s in DEFAULT_BACKEND.available_solvers()])
        info['available_solvers']['value'] = DEFAULT_BACKEND.available_solvers()
    info['graphviz_version'] = {
        'title': 'Graphviz version',
        'message': 'Graphviz not installed. To support plotting, please install graphviz with conda',
        'value': None
    }
    try:
        import graphviz
        info['graphviz_version']['message'] = f"v{graphviz.__version__}"
        info['graphviz_version']['value'] = graphviz.__version__
    except Exception as e:
        pass
    info['repo_url'] = {
        'title': 'Repository',
        'message': "https://github.com/saezlab/corneto",
        'value': "https://github.com/saezlab/corneto"
    }
    return info

def info():
    info = _get_info()

    if _support_html_output():
        from IPython.display import HTML
        from IPython.core.display import display
        import pkg_resources
        import base64
        logo_path = pkg_resources.resource_filename(__name__, 'rsc/logo/corneto-logo-512px.png')
        with open(logo_path, 'rb') as f:
            img_bytes = f.read()
        b64img = base64.b64encode(img_bytes).decode('utf-8')
        html = f'''
        <table style='background-color:rgba(0, 0, 0, 0);'>
        <tr>
            <td><img src="data:image/jpeg;base64,{b64img}" width="85px" /></td>
            <td>
            <table>
                *
            </table>
            </td>
        </tr>
        </table>'''
        html_info = ''
        for k, v in info.items():
            title = v['title']
            message = v['message']
            if '_url' in k:
                message = f"<a href={message}>{message}</a>"
            html_info += f"<tr><td>{title}:</td><td style='text-align:left'>{message}</td></tr>"
        display(HTML(html.replace('*', html_info)))
        
    else:
        for v in info.values():
            title = v['title']
            message = v['message']
            print(f"{title}:", f"{message}")

def _info():
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


def nullspace(A, atol=1e-13, rtol=0):
    # https://scipy-cookbook.readthedocs.io/items/RankNullspace.html
    A = np.atleast_2d(A)
    u, s, vh = svd(A)
    tol = max(atol, rtol * s[0])
    nnz = (s >= tol).sum()
    ns = vh[nnz:].conj().T
    return ns