# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import sys
import os


project = "PythonSI: Python Selective Inference"
copyright = "2025, PythonSI Contributors"
author = "Tran Tuan Kiet, PythonSI Contributors"
release = "First pre-release"

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath(".."))
sys.path.insert(0, os.path.abspath("../.."))


# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
    "sphinx.ext.coverage",
    "sphinx.ext.mathjax",
    "sphinx.ext.ifconfig",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx_gallery.gen_gallery",
    "myst_parser",
    "sphinxcontrib.jquery",
]

sphinx_gallery_conf = {
    "examples_dirs": ["../../examples"],
    "gallery_dirs": "auto_examples",
    "filename_pattern": r"\.py",  # (?!barycenter_fgw)
    "ignore_pattern": r"models",
    "nested_sections": False,
    "backreferences_dir": "gen_modules/backreferences",
    "inspect_global_variables": True,
    "matplotlib_animations": True,
    "reference_url": {"ot": None},
    "copyfile_regex": r"index.rst",
    "write_computation_times": False,
    "min_reported_time": float("inf"),
}

napoleon_numpy_docstring = True

autosummary_generate = True
autosummary_imported_members = True

templates_path = ["_templates"]
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
