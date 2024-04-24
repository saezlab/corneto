# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import sys
import os
from pathlib import Path


# import pydata_sphinx_theme
# from sphinx.application import Sphinx

sys.path.append(str(Path(".").resolve()))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "CORNETO"
copyright = "2023, Saez-Rodriguez group"
author = "Pablo Rodriguez-Mier"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    # "myst_parser",
    "myst_nb",
    "sphinx_design",
    "sphinx.ext.autodoc",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx_autodoc_typehints",
    "sphinx.ext.extlinks",
    "sphinx.ext.autosummary",
    "sphinxcontrib.mermaid",
    "sphinx.ext.intersphinx",
    'sphinx.ext.doctest',
    "sphinx_favicon",
    "hoverxref.extension",
    "sphinx_multiversion",
    "_extension.gallery_directive",
]

myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "html_image",
    "html_admonition",
]

html_context = {
    "display_github": True,
    "github_user": "saezlab",
    "github_repo": "corneto",
    "github_version": "main",
    "conf_py_path": "/docs/",
}

myst_url_schemes = ("http", "https", "mailto")
nb_output_stderr = "remove"
nb_execution_mode = "cache"
nb_execution_timeout = 300
nb_merge_streams = True
typehints_defaults = "braces"

source_suffix = {
    ".rst": "restructuredtext",
    ".ipynb": "myst-nb",
    ".myst": "myst-nb",
}

mermaid_params = [
    "-t",
    "default",
    "-b",
    "transparent",
]

templates_path = ["_templates"]

exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "**.ipynb_checkpoints",
    "**/_*.ipynb",
]


autosummary_generate = True
autodoc_member_order = "bysource"
autodoc_typehints = "description"
bibtex_reference_style = "author_year"
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_use_rtype = True
napoleon_use_param = True
todo_include_todos = False

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_logo = "_static/logo.png"
html_css_files = [
    "css/custom.css",
]
html_show_sphinx = False

html_theme_options = {"primary_sidebar_end": ["sidebar-ethical-ads"]}

intersphinx_mapping = {
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "python": ("https://docs.python.org/3/", None),
    # "corneto": ("./_build/dirhtml", None),
}

html_context = {"default_mode": "light"}

# -- Config for hoverxref -------------------------------------------

hoverx_default_type = "tooltip"
hoverxref_domains = ["py"]
hoverxref_role_types = dict.fromkeys(
    ["ref", "class", "func", "meth", "attr", "exc", "data", "mod"],
    "tooltip",
)
hoverxref_intersphinx = [
    "python",
    "numpy",
    "scipy",
    "pandas",
]

# use proxied API endpoint on rtd to avoid CORS issues
if os.environ.get("READTHEDOCS"):
    hoverxref_api_host = "/_"


def setup(app):
    """App setup hook."""
    app.add_config_value(
        "recommonmark_config",
        {
            # "auto_toc_tree_section": "Contents",
            "enable_auto_toc_tree": True,
            "enable_math": True,
            # "enable_inline_math": False,
            "enable_eval_rst": True,
            "auto_toc_maxdepth": 4,
        },
        True,
    )
