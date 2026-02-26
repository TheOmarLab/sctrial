import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.abspath("../../src"))

project = "sctrial"
copyright = f"{datetime.now().year}, Contributors"
author = "Contributors"
release = "0.2.1.dev1"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "nbsphinx",
]

nbsphinx_allow_errors = True
nbsphinx_execute = 'never'  # notebooks are pre-executed locally with outputs saved
nbsphinx_timeout = 300



templates_path = ["_templates"]
exclude_patterns = []

html_theme = "sphinx_rtd_theme"
html_static_path = []

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable/", None),
    "anndata": ("https://anndata.readthedocs.io/en/latest/", None),
}
