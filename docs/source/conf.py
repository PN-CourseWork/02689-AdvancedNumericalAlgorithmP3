"""Sphinx configuration for numutils package documentation."""

import os
import sys

# Add src directory to path for module imports
src_path = os.path.abspath(os.path.join(__file__, "..", "..", "..", "src"))
sys.path.insert(0, src_path)

repo_root = os.path.abspath(os.path.join(__file__, "..", "..", ".."))
sys.path.insert(0, repo_root)

# -- Project information -----------------------------------------------------

project = "Advanced Numerical Algorithms - Project 3"
copyright = "2025, Philip Korsager Nickel, Aske Schrøder Nielsen"
author = "Philip Korsager Nickel, Aske Schrøder Nielsen"

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "numpydoc",
    "sphinx_copybutton",
]

root_doc = "index"
source_suffix = {".rst": "restructuredtext"}

# -- Autodoc configuration ---------------------------------------------------

autosummary_generate = True
autosummary_imported_members = False

autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "inherited-members": True,
    "show-inheritance": True,
}

# Mock heavy runtime dependencies (petsc4py requires PETSc installation)
autodoc_mock_imports = ["petsc4py", "petsc4py.PETSc"]

# -- Numpydoc configuration --------------------------------------------------

numpydoc_show_class_members = False
numpydoc_show_inherited_class_members = False
numpydoc_class_members_toctree = False
numpydoc_xref_param_type = True
numpydoc_xref_ignore = {"optional", "default", "of"}

# Use numpydoc's enhanced autosummary templates
templates_path = ["_templates"]
numpydoc_use_plots = False  # Don't auto-generate plots from Examples

# -- Intersphinx configuration -----------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
}

# -- HTML output options -----------------------------------------------------

html_theme = "pydata_sphinx_theme"
html_title = "Advanced Numerical Algorithms - Project 3"
html_static_path = ["_static"]
html_show_sourcelink = False  # Hide "Show Source" link
html_css_files = ["custom.css"]  # Custom CSS for hiding download buttons

html_theme_options = {
    "github_url": "https://github.com/PN-CourseWork/02689-AdvancedNumericalAlgorithmP3",
    "navbar_align": "left",
    "header_links_before_dropdown": 5,
    "show_toc_level": 2,
    "collapse_navigation": True,
    "navigation_depth": 4,
    "show_nav_level": 2,
    "secondary_sidebar_items": ["page-toc"],
}

# Let pydata theme handle sidebars with defaults
