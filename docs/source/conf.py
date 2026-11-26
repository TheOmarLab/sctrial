import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.abspath("../../src"))

project = "sctrial"
copyright = f"{datetime.now().year}, Omar Lab, Cedars-Sinai Medical Center"
author = "Omar Lab, Cedars-Sinai Medical Center"
release = "2.0.0"

# -- Extensions ---------------------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "myst_parser",
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
html_baseurl = "https://www.omar-lab.com/sctrial/"
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
    # Navigation: left sidebar with full toctree, no top-bar section links
    "navbar_center": [],
    "show_nav_level": 2,
    "show_toc_level": 2,
    "navigation_with_keys": True,
    "navigation_depth": 3,
    "footer_start": ["copyright"],
    "footer_end": ["last-updated"],
    "secondary_sidebar_items": ["page-toc", "edit-this-page"],
    "use_edit_page_button": True,
}

# Ensure left sidebar appears on ALL pages including the landing page
html_sidebars = {
    "**": ["sidebar-nav-bs.html"],
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
    "anndata": ("https://anndata.scverse.org/en/stable/", None),
}

# -- Feature clips -------------------------------------------------------------


def _copy_feature_clips(app, exception):
    """Copy the player of the study-design clips into the built site.

    The Study Designs page plays the scenes of ``clips/web`` live; they are kept in one
    place and copied at build time.
    """
    if exception is not None or app.builder.format != "html":
        return
    source = Path(__file__).resolve().parents[2] / "clips" / "web"
    if not source.is_dir():
        from sphinx.util import logging

        logging.getLogger(__name__).warning("clips/web was not found: the Study Designs page has no player")
        return
    shutil.copytree(source, Path(app.outdir) / "_static" / "clips" / "web", dirs_exist_ok=True)


def setup(app):
    app.connect("build-finished", _copy_feature_clips)
