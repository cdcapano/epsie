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

"""Utilities for creating random number generators.

This uses the :py:module:`randomgen` package for creating independent random
number generators, which are then converted to numpy's
:py:class:`<numpy.random.RandomState RandomStates>`. This is done because
``randomgen`` provides a ``jump`` method for the Mersenne-Twister algorithm. At
some point, numpy's random number generator will be replaced by the utilities
more similar to ``randomgen``; see NEP 19:
https://www.numpy.org/neps/nep-0019-rng-policy.html for details. When that
happens, the dependency on ``randomgen`` will be removed, and we'll switch to
just using numpy's random states directly.
"""

import os
import numpy
import randomgen


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


def randgen2numpy(randgenrs):
    """Converts a randomgen.MT19937 generator into a numpy.RandomState

    This works by creating a :py:class:`numpy.random.RandomState`, then
    setting its state to be the same as the given `randomgen.MT19937`'s state.

    Parameters
    ----------
    randgenrs : :py:class:`randomgen.MT19937` instance
        The `randomgen.MT19937` instance to convert.

    Returns
    -------
    numpy.random.RandomState
        A numpy.random.RandomState instance with the same state.
    """
    nprs = numpy.random.RandomState()  # doesn't matter what seed we use
    rsstate = randgenrs.state
    nprs.set_state((rsstate['brng'],
                    rsstate['state']['key'],
                    rsstate['state']['pos']))
    return nprs


def create_randomstates(seed, nstates):
    """Given a seed, creates a number of random independent random states.
    """
    # use randomgen's ability to do large jumps to create the initial states
    rstates = [randomgen.MT19937(seed) for _ in range(nstates)]
    for ii in range(nstates):
        rstates[ii].jump(ii)
    # now convert to numpy
    return map(randgen2numpy, rstates)

