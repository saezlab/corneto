"""Sphinx configuration file for the CORNETO project.

This file contains the configuration settings for building the CORNETO documentation
using Sphinx. For more details, see:
https://www.sphinx-doc.org/en/master/usage/configuration.html
"""

import sys
import os
from datetime import datetime
from pathlib import Path
import inspect

# -- Path setup --------------------------------------------------------------
# Add the project root directory to sys.path to enable autodoc to locate modules.
sys.path.insert(0, str(Path(".").resolve()))

# Import the project module to retrieve version information.
import corneto


# Derive GitHub username (set in GitHub Actions; use a default for local builds)
repo = os.environ.get("GITHUB_REPOSITORY", "username/corneto")
username = repo.split("/")[0]

# -- Project information -----------------------------------------------------
project = "CORNETO"
copyright = (
    f"2023-{datetime.now().year}, Saez-Rodriguez lab. (EMBL-EBI, Heidelberg University)"
)
author = "Pablo Rodriguez-Mier"

# -- General configuration ---------------------------------------------------
# Sphinx extension module names.
extensions = [
    # "myst_parser",  # Alternative parser (currently not in use).
    "myst_nb",  # Support for MyST Notebook (Jupyter notebooks).
    "sphinx_design",  # Provides design elements.
    "sphinx.ext.autodoc",  # Automatic documentation from docstrings.
    "sphinx.ext.mathjax",  # Math rendering.
    "sphinx.ext.napoleon",  # Support for Google and NumPy style docstrings.
    #"sphinx_autodoc_typehints",  # Better integration of type hints.
    "sphinx.ext.extlinks",  # Shortcut for external links.
    "sphinx.ext.autosummary",  # Generate summary tables.
    "sphinx.ext.intersphinx",  # Link to other projects' documentation.
    "sphinx.ext.doctest",  # Test embedded code snippets.
    "sphinx_favicon",  # Favicon support.
    "_extension.gallery_directive",  # Custom gallery directive.
]

# Enable specific MyST extensions.
myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "html_image",
    "html_admonition",
    "substitution",
]

# Substitutions to be used in MyST documents.
myst_substitutions = {
    "version": corneto.__version__,
}

# File types and their corresponding parsers.
source_suffix = {
    ".rst": "restructuredtext",
    ".ipynb": "myst-nb",
    ".myst": "myst-nb",
}

# Notebook execution settings (used by myst_nb).
nb_output_stderr = "remove"
nb_execution_mode = "cache"
nb_execution_timeout = 300
nb_merge_streams = True

# Formatting for typehints in the documentation.
typehints_defaults = "braces"

# Configuration for Mermaid diagrams.
mermaid_params = [
    "-t", "default",
    "-b", "transparent",
]

# Paths that contain templates, relative to this directory.
templates_path = ["_templates"]

# Patterns to ignore when looking for source files.
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "**.ipynb_checkpoints",
    "**/_*.ipynb",
]

# Autosummary and autodoc settings.
autosummary_generate = True
autodoc_member_order = "bysource"
autodoc_typehints = "description"

autodoc_default_options = {
    'members': True,
    'imported-members': True,
    'undoc-members': True,
    'show-inheritance': True,
}

# Napoleon settings (for parsing Google/NumPy style docstrings).
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_use_rtype = True
napoleon_use_param = True

# BibTeX reference style (if using sphinxcontrib-bibtex).
bibtex_reference_style = "author_year"

# Whether to include todo items.
todo_include_todos = False

# -- Options for HTML output --
#html_baseurl = 'https://saezlab.github.io/corneto'
html_baseurl = f"https://{username}.github.io/corneto"
html_favicon = '_static/favicon.ico'
html_show_sourcelink = False
add_function_parentheses = False
html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_logo = "_static/logo.png"
html_css_files = [
    "css/custom.css",
]
html_show_sphinx = False
# Specify sidebars only for index and install pages
html_sidebars = {
    "index": [],
    "install": []
}
# do not show source links
html_show_sourcelink = False

# link to document:section
autosectionlabel_prefix_document = True

# Theme-specific options.
html_theme_options = {
    "primary_sidebar_end": ["sidebar-ethical-ads"],
    "header_links_before_dropdown": 4,
    "show_toc_level": 1,
    "navbar_align": "left",
    "switcher": {
        # The switcher.json file is now available at the root.
        "json_url": f"{html_baseurl}/switcher.json",
        "version_match": corneto.__version__,
    },
    "navbar_start": ["navbar-logo", "version-switcher"],
}

# Additional HTML context for templates.
html_context = {
    "display_github": True,
    "github_user": "saezlab",
    "github_repo": "corneto",
    "github_version": "main",
    "conf_py_path": "/docs/",
    "default_mode": "light",
}

# -- Intersphinx configuration --
# Maps external projects to their documentation.
intersphinx_mapping = {
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "python": ("https://docs.python.org/3/", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable", None),
}


# -- Setup function --
def setup(app):
    """Sphinx setup hook.

    Adds additional configuration values for recommonmark to enable features such as
    automatic table of contents generation and math support in reStructuredText.
    """
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
