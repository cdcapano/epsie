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

"""Utilities for creating random number generators."""

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
