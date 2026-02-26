import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.abspath("../../src"))

project = "sctrial"
copyright = f"{datetime.now().year}, Contributors"
author = "Contributors"
release = "0.2.1.dev1"

# -- Extensions ---------------------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "nbsphinx",
    "sphinx_design",
    "sphinx_copybutton",
]

# -- nbsphinx settings --------------------------------------------------------

nbsphinx_allow_errors = True
nbsphinx_execute = "never"  # notebooks are pre-executed locally with outputs saved
nbsphinx_timeout = 300

# -- General configuration -----------------------------------------------------

templates_path = ["_templates"]
exclude_patterns = []

# -- HTML output ---------------------------------------------------------------

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_favicon = "_static/logo_icon.svg"

html_theme_options = {
    "logo": {
        "image_light": "_static/logo.svg",
        "image_dark": "_static/logo_dark.svg",
    },
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/TheOmarLab/sctrial",
            "icon": "fa-brands fa-github",
        },
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/sctrial/",
            "icon": "fa-solid fa-box",
        },
    ],
    "navbar_align": "left",
    "show_nav_level": 2,
    "show_toc_level": 2,
    "navigation_with_keys": True,
    "footer_start": ["copyright"],
    "footer_end": ["last-updated"],
    "secondary_sidebar_items": ["page-toc", "edit-this-page"],
    "use_edit_page_button": True,
}

html_context = {
    "github_user": "TheOmarLab",
    "github_repo": "sctrial",
    "github_version": "main",
    "doc_path": "docs/source",
}

# -- Intersphinx ---------------------------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable/", None),
    "anndata": ("https://anndata.readthedocs.io/en/latest/", None),
}
