#!/usr/bin/env python

# Copyright (C) 2019 Collin Capano
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 2 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
# Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.

"""
setup.py file for PyCBC package
"""
import setuptools


install_requires = ["numpy",
                    "scipy>=0.16.0",
                    "randomgen",
                   ]

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="epsie",
    version="0.0.1",
    author="Collin D. Capano",
    author_email="cdcapano@gmail.com",
    description="An Embarrassingly Parallel Sampler for "
                "Inference Estimation.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/cdcapano/epsie",
    install_requires = install_requires,
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 2.7",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
        'Intended Audience :: Science/Research',
        'Natural Language :: English',
    ],
)
