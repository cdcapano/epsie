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
import os
import sys
import glob
import subprocess
import epsie
sys.path.insert(0, os.path.abspath('../epsie'))


# -- Project information -----------------------------------------------------

project = 'EPSIE'
copyright = '2020, Collin D. Capano'
author = 'Collin D. Capano'

# The full version, including alpha/beta/rc tags
release = epsie.__version__
version = epsie.__version__


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx_rtd_theme',
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    ]

# We use numpy style for docs
napoleon_numpy_docstring = True
napoleon_google_docstring = False

# external packages for intersphinx
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/reference', None),
    'h5py': ('https://docs.h5py.org/en/latest/', None),
    }

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

def setup(app):
    app.add_css_file('custom.css')

# see if we're adding versions: we will do this if sphinx-build was run with
# tags set to either latest or versioned
if tags.has('latest'):
    docversion = 'latest'
elif tags.has('versioned'):
    # ignore the patch part
    docversion = '.'.join(epsie.__version__.split('.')[:2])
else:
    docversion = ''

# load previous versions from cache
if docversion:
    # check if there is a cache of previous versions
    if os.path.exists('.docversions'):
        with open('.docversions', 'a+') as fp:
            # skip the comment line
            fp.seek(0)
            cached_versions = [l.rstrip('\n') for l in fp.readlines()
                               if not l.startswith('#')
                               and not l.startswith('latest')]
            if docversion not in cached_versions and docversion != 'latest':
                # add to the file
                print(docversion, file=fp)
    else:
        cached_versions = []
    # add relative paths for index
    versions = [(v, '../{}/index.html'.format(v)) for v in cached_versions
                if v != docversion]
    # current
    versions.append((docversion, ''))
    # add latest if it isn't there
    if docversion != 'latest':
        versions.append(('latest', '../latest/index.html'))
    # reverse order to ensure latest comes first
    versions = versions[::-1]
else:
    versions = []

# set html settings
html_context = {
    'current_version': docversion,
    'version': epsie.__version__,
    'versions': versions,
    'display_lower_left': True,
    'display_github': True,
    'github_user': 'cdcapano',
    'github_repo': 'epsie',
    'github_version': 'master/docs/',
}

# run scripts in _include
print("Running scripts in _include directory:")
cwd = os.getcwd()
os.chdir('_include')
# bash scripts
for fn in glob.glob("*.sh"):
    print("  {}".format(fn))
    subprocess.run(['sh', fn])
# python scripts
for fn in glob.glob("*.py"):
    print("  {}".format(fn))
    subprocess.run(['python', fn])
os.chdir(cwd)
