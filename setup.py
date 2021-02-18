"""Setup file for the glaciersat package.
   Adapted from the Python Packaging Authority template."""

from setuptools import setup, find_packages  # Always prefer setuptools
from os import path
import warnings
import sys
import re

MAJOR = 0
MINOR = 1
MICRO = 0
ISRELEASED = False
VERSION = '%d.%d.%d' % (MAJOR, MINOR, MICRO)
QUALIFIER = ''


DISTNAME = 'glaciersat'
LICENSE = 'MIT'
AUTHOR = 'Johannes Marian Landmann'
AUTHOR_EMAIL = 'landmann@vaw.baug.ethz.ch'
URL = 'https://github.com/jlandmann/glaciersat'
CLASSIFIERS = [
        # How mature is this project? Common values are
        # 3 - Alpha  4 - Beta  5 - Production/Stable
        'Development Status :: 3 - Alpha',
        # Indicate who your project is intended for
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.9'
    ]

DESCRIPTION = 'glaciersat - monitoring glaciers from space'
LONG_DESCRIPTION = """
**Glaciersat is a modular, open-source package for glacier remote sensing**

Glaciersat provides tools to monitor important glacier variables like snow 
distribution on the glaciers, broadband albedo and melt area from different 
platform and sensors. It put a special focus on quantifying uncertainties in 
both input and derived variables.
Links
-----
- HTML documentation: https://glaciersat.readthedocs.org
- Source code: http://github.com/jlandmann/glaciersat
"""


# code to extract and write the version copied from pandas
FULLVERSION = VERSION
write_version = True

if not ISRELEASED:
    import subprocess
    FULLVERSION += '.dev'

    pipe = None
    for cmd in ['git', 'git.cmd']:
        try:
            pipe = subprocess.Popen(
                [cmd, "describe", "--always", "--match", "v[0-9]*"],
                stdout=subprocess.PIPE)
            (so, serr) = pipe.communicate()
            if pipe.returncode == 0:
                break
        except:
            pass

    if pipe is None or pipe.returncode != 0:
        # no git, or not in git dir
        if path.exists('glaciersat/version.py'):
            warnings.warn("WARNING: Couldn't get git revision, using existing "
                          "glaciersat/version.py")
            write_version = False
        else:
            warnings.warn("WARNING: Couldn't get git revision, using generic "
                          "version string")
    else:
        # have git, in git dir, but may have used a shallow clone (travis)
        rev = so.strip()
        # makes distutils blow up on Python 2.7
        if sys.version_info[0] >= 3:
            rev = rev.decode('ascii')

        if not rev.startswith('v') and re.match("[a-zA-Z0-9]{7,9}", rev):
            # partial clone, manually construct version string
            # this is the format before we started using git-describe
            # to get an ordering on dev version strings.
            rev = "v%s.dev-%s" % (VERSION, rev)

        # Strip leading v from tags format "vx.y.z" to get th version string
        FULLVERSION = rev.lstrip('v')
else:
    FULLVERSION += QUALIFIER


def write_version_py(filename=None):
    cnt = """\
version = '%s'
short_version = '%s'
"""
    if not filename:
        filename = path.join(path.dirname(__file__), 'glaciersat',
                             'version.py')

    a = open(filename, 'w')
    try:
        a.write(cnt % (FULLVERSION, VERSION))
    finally:
        a.close()


if write_version:
    write_version_py()

req_packages = ['numpy',
                'scipy',
                'pandas',
                'matplotlib>=3.0.0',
                'shapely',
                'configobj',
                'netcdf4',
                'xarray>=0.16',
                'sentinelsat',
                'salem'
                ]


setup(
    # Project info
    name=DISTNAME,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    # Version info
    setup_requires=['setuptools_scm', 'setuptools_scm_git_archive'],
    use_scm_version=True,
    # The project's main homepage.
    url=URL,
    # Author details
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    # License
    license=LICENSE,
    classifiers=CLASSIFIERS,
    # What does your project relate to?
    keywords=['geosciences', 'glaciers', 'remote sensing', 'gis'],
    # We are a python 3 only shop
    python_requires='>=3.6',
    # Find packages automatically
    packages=find_packages(exclude=['docs']),
    # Include package data
    include_package_data=True,
    # Install dependencies
    install_requires=req_packages,
    # additional groups of dependencies here (e.g. development dependencies).
    extras_require={},
    # Executable scripts
    entry_points={
    },
)
