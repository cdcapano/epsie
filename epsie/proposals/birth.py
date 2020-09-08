# Copyright (C) 2020 Richard Stiskalek, Collin Capano
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

from __future__ import (absolute_import, division)


import numpy
from scipy import stats

try:
    from randomgen import RandomGenerator
except ImportError:
    from randomgen import Generator as RandomGenerator

import epsie
from .base import BaseRandom

class UniformBirth(BaseRandom):
    """Birth distribution object used in nested transdimensional proposals
    to propose birth to parameters which were previously inactive. This
    particular implementation assumes a uniform proposal distribution.

    Parameters
    ----------
    parameters : (list of) str
        The names of the parameters to produce proposals for.
    boundaries : dict
        Dictionary mapping parameters to boundaries. Boundaries must be a
        tuple or iterable of length two.

    Properties
    ----------
    birth : dict
        Returns random variate sample from the uniform distribution for each
        parameter.

    Methods
    ------
    logpdf : py:func
        Evalues the logpdf proposal ratio. Takes dictionary of parameters as
        input.
    """
    name = 'uniform_birth_distribution'

    def __init__(self, parameters, bounds):
        self.parameters = parameters
        self.bounds = bounds
        self._scale = {p: bounds[p][1] - bounds[p][0] for p in parameters}

    @property
    def birth(self):
        return {p: self.random_generator.uniform(
            self.bounds[p][0], self.bounds[p][1]) for p in self.parameters}

    def logpdf(self, xi):
        return sum([stats.uniform.logpdf(
            xi[p], loc=self.bounds[p][0], scale=self._scale[p])
            for p in self.parameters])
