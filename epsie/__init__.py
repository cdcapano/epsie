# Copyright (C) 2019  Collin Capano
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
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

"""EPSIE is a toolkit for doing MCMCs using embarrasingly parallel Markov
chains.
"""

from __future__ import absolute_import

# get the version number
from ._version import __version__

import os
import numpy
from randomgen import PCG64

#
# =============================================================================
#
#                     Random number generation utilities
#
# =============================================================================
#

# The class used for all basic random number generation.
# Users may change this, but it has to be something recognized by randomgen
# and have the ability to accept streams; i.e., it must have calling structure
# class(seed, stream), with default stream set to None.
BRNG = PCG64


def create_seed(seed=None):
    """Creates a seed for a random number generator.

    Parameters
    ----------
    seed : int, optional
        If a seed is given, will just return it. Default is None, in which case
        a seed is created.

    Returns
    -------
    int :
        A seed to use.
    """
    if seed is None:
        # use os.urandom to get a string of random 4 bytes
        bseed = os.urandom(4)
        # convert to int
        seed = sum([ord(c) << (i * 8) for i, c in enumerate(bseed[::-1])])
        # Py3XX: this conversion using a method suggested here:
        # https://stackoverflow.com/questions/444591/how-to-convert-a-string-of-bytes-into-an-int-in-python
        # As stated in the answers there, in python 3.2 and later, can use
        # the following instead:
        # seed = int.from_bytes(bseed, byteorder='big')
    return seed


def create_brng(seed=None, stream=0):
    """Creates a an instance of :py:class:`epsie.BRNG`.

    Parameters
    ----------
    seed : int, optional
        The seed to use. If seed is None (the default), will create a seed
        using ``create_seed``.
    stream : int, optional
        The stream to create the BRNG for. This allows multiple BRNGs to exist
        with the same seed, but that produce different sets of random numbers.
        Default is 0.
    """
    if seed is None:
        seed = create_seed(seed)
    return BRNG(seed, stream)


def create_brngs(seed, nrngs):
    """Creates a collection of basic random number generators (BRNGs).

    The BRNGs are different streams with the same seed. They are all
    statistically independent of each other, while still being reproducable.
    """
    return [BRNG(seed, ii) for ii in range(nrngs)]


#
# =============================================================================
#
#                          Array utilities
#
# =============================================================================
#


def array2dict(array):
    """Converts a structured array into a dictionary."""
    fields = array.dtype.names  # raises an AttributeError if array is None
    if fields is None:
        # not a structred array, just return
        return array
    return {f: _getatomic(array[f]) for f in fields}


def _getatomic(val):
    """Checks if a given value is numpy scalar. If so, it returns
    the value as its native python type.
    """
    if isinstance(val, numpy.ndarray) and val.size == 1 and val.ndim == 0:
        val = val.item(0)
    return val


#
# =============================================================================
#
#                          Parallel tempering utilities
#
# =============================================================================
#


def make_betas_ladder(ntemps, maxtemp):
    """Makes a log spaced ladder of betas."""
    minbeta = 1./maxtemp
    return numpy.geomspace(minbeta, 1., num=ntemps)
