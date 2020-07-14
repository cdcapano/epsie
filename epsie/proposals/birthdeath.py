# Copyright (C) 2020 Richard Stiskalek
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

from abc import ABCMeta
from six import add_metaclass

import numpy as np
from scipy import stats

from .base import BaseProposal
from .bounded_normal import Boundaries


class BirthDeath(BaseProposal):
    """Transdimensional proposal that allows dimension change by +- 1 or  keep
    the same number of dimensions.
    --------------------------
    Parameters:
        parameters: list of str
            Parameter names for this proposal
        boundaries: dict of tuples,  keys -> parameters
            Inclusive lower and upper limits on the parameters
        jump_proposal: dict of funcs, keys -> parameters
            Logpdfs for each parameter
        jump_freq: float (optional)
            Parameter that tunes the proportion of dimension jumps
    """
    name = 'birthdeath'
    symmetric = False

    def __init__(self, parameters, boundaries, jump_proposal,
               jump_freq=0.5):
        self._parameters = None
        self._boundaries = None
        self._jum_proposal = None
        self._jump_freq = None
        self._jump_freq = None

        self.parameters = parameters
        self.boundaries = boundaries
        self.jump_proposal = jump_proposal
        self.jump_freq = jump_freq

    @property
    def parameters(self):
        """Parameters of this proposal"""
        return self._parameters

    @parameters.setter
    def parameters(self, parameters):
        if not isinstance(parameters, list):
            raise ValueError('provide a list')
        if not all([isinstance(p, str) for p in parameters]):
            raise ValueError('all members must be str')
        self._parameters = parameters

    @property
    def boundaries(self):
        """Dictionary of parameter boundaries"""
        return self._boundaries

    @boundaries.setter
    def boundaries(self, boundaries):
        try:
            self._boundaries = {p: Boundaries(boundaries[p])
                                for p in self.parameters}
        except KeyError:
            raise ValueError("must provide a boundary for every parameter")

    @property
    def jump_proposal(self):
        """Dictionary of jump proposals for each parameter"""
        return self._jump_proposal

    @jump_proposal.setter
    def jump_proposal(self, jump_proposal):
        try:
            self._jump_proposal = {p : jump_proposal[p]
                                   for p in self.parameters}
        except KeyError:
            raise ValueError("provide a proposal for each parameter")

    @property
    def jump_freq(self):
        return self._jump_freq

    @jump_freq.setter
    def jump_freq(self, jump_freq):
        if not isinstance(jump_freq, float):
            raise ValueError('must be a float')
        elif not 0.0 <=  jump_freq <= 0.5:
            raise ValueError('jump frequency must be in [0.0, 0.5]')
        self._jump_freq = jump_freq

    @property
    def state(self):
        return {'random_state': self.random_state}

    def set_state(self, state):
        self.random_state = state['random_state']

    def jump(self, fromx):
        for p in self.parameters:
            k = fromx[p]
            current = self.jump_proposal[p](k)
            # Don't forget to check boundaries
            if k == self.boundaries[p].lower:
                death = 0.0
            else:
                death = self.jump_freq * min(1,\
                            np.exp(self.jump_proposal[p](k - 1) - current))

            if k == self.boundaries[p].upper:
                birth = 0.0
            else:
                birth = self.jump_freq * min(1,\
                            np.exp(self.jump_proposal[p](k + 1) - current))

            # Remove a signal with 'death' prob, add with 'birth' prob and
            # update with 'update' prob.
            u = self.random_generator.uniform()
            if u <= birth:
                newpt = {p : k + 1}
            elif u <= birth + death:
                newpt = {p : k - 1}
            else:
                newpt = {p : k}

            return newpt

    def logpdf(self, xi, givenx):
        return sum([self.jump_proposal[p](xi[p]) for p in self.parameters])
