# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import datetime
import os
import sys

sys.path.insert(0, os.path.abspath("../.."))

import lightly

# -- Project information -----------------------------------------------------

project = "LightlySSL"
copyright = f"2020-{datetime.date.today().year}, Lightly AG"
author = "Lightly Team"

# The full version, including alpha/beta/rc tags
release = lightly.__version__
master_doc = "index"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",
    "sphinx_gallery.gen_gallery",
    "sphinx_tabs.tabs",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinx_reredirects",
]

sphinx_gallery_conf = {
    "examples_dirs": ["tutorials_source/package", "tutorials_source/platform"],
    "gallery_dirs": [
        "tutorials/package",
        "tutorials/platform",
    ],  # path to where to save gallery generated output
    "filename_pattern": "/tutorial_",
}

napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = False
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = False
napoleon_use_rtype = False
napoleon_type_aliases = None

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "furo"

# Layout follows the LightlyStudio docs so the doc sites read as one product
# family, but the colors are LightlySSL's own, taken from the logo: navy #092643,
# turquoise #2cc2bd and the green-to-blue gradient of the mark. Turquoise is too
# light for text on white, so light-mode links use a darker teal that passes WCAG
# AA. Furo takes the colors as CSS custom properties, so the tokens live here and
# _static/custom.css only handles what they cannot reach.
_BRAND = {
    "color-brand-primary": "#092643",
    "color-brand-content": "#0f7f7b",
    "color-brand-visited": "#0f7f7b",
    # Notes and see-also boxes use the brand teal instead of Furo's material blue.
    "color-admonition-title--note": "#0f7f7b",
    "color-admonition-title--seealso": "#0f7f7b",
    # The banner is the brand navy in both schemes.
    "color-announcement-background": "#092643",
    "color-announcement-text": "#d4e2ec",
}
_BRAND_DARK = {
    "color-brand-primary": "#f7f8fb",
    "color-brand-content": "#2cc2bd",
    "color-brand-visited": "#2cc2bd",
    "color-admonition-title--note": "#2cc2bd",
    "color-admonition-title--seealso": "#2cc2bd",
    "color-announcement-background": "#0d3253",
    "color-announcement-text": "#d4e2ec",
}

html_theme_options = {
    "sidebar_hide_name": True,
    "announcement": (
        "Interested in training your own vision foundation model? Check out "
        '<a href="https://docs.lightly.ai/train/stable/index.html"><b>LightlyTrain</b></a>!'
    ),
    "footer_icons": [
        {
            "name": "GitHub",
            "url": "https://github.com/lightly-ai/lightly",
            # Octicon mark-github, as in the furo docs.
            "html": '<svg stroke="currentColor" fill="currentColor" stroke-width="0" viewBox="0 0 16 16"><path fill-rule="evenodd" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0 0 16 8c0-4.42-3.58-8-8-8z"></path></svg>',
            "class": "",
        },
    ],
    "navigation_with_keys": True,
    # Furo swaps these with the colour scheme; a single html_logo would be
    # invisible against one of the two sidebar backgrounds.
    "light_logo": "lightly_SSL_logo_crop.png",
    "dark_logo": "lightly_SSL_logo_crop_white_text.png",
    "light_css_variables": {
        **_BRAND,
        "font-stack": "'IBM Plex Sans', system-ui, sans-serif",
        "font-stack--monospace": "'IBM Plex Mono', ui-monospace, monospace",
        "color-foreground-primary": "#092643",
        "color-foreground-secondary": "#4c616b",
        "color-foreground-muted": "#5e727c",
        "color-background-primary": "#ffffff",
        "color-background-secondary": "#fafafc",
        "color-background-border": "#e6e6ec",
        "color-api-background": "#fafafc",
        # A gray sidebar sets the navigation apart from the white page.
        "color-sidebar-background": "#f4f6f8",
        "color-sidebar-background-border": "#e3e8ec",
        "color-sidebar-search-background": "#ffffff",
        "color-sidebar-caption-text": "#5e727c",
        # Tint behind the current page in the sidebar.
        "ls-accent-tint": "#e3f6f4",
    },
    "dark_css_variables": {
        **_BRAND_DARK,
        "color-foreground-primary": "#f7f8fb",
        "color-foreground-secondary": "#93a7b0",
        "color-foreground-muted": "#7f949e",
        "color-background-primary": "#061b30",
        "color-background-secondary": "#092643",
        "color-background-border": "#17395a",
        "color-api-background": "#092643",
        "color-sidebar-background": "#092643",
        "color-sidebar-background-border": "#17395a",
        "color-sidebar-search-background": "#061b30",
        "color-sidebar-caption-text": "#7f949e",
        "ls-accent-tint": "#0d3253",
    },
}

# The code panel stays navy in both schemes (see _static/custom.css), so light mode
# needs dark-background token colors too. A light style renders names like `torch`
# near-black on navy.
pygments_style = "native"
pygments_dark_style = "native"

html_css_files = [
    "https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500;600&family=Open+Sans:wght@600;700&display=swap",
    "custom.css",
]
# Method family filter on the start page and the Models page; does nothing on
# pages without filter buttons.
html_js_files = ["method-filter.js"]

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static", "../logos"]

html_favicon = "favicon.png"
