# Copyright (C) 2020 Collin Capano, Richard Stiskalek
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

import numpy
from scipy import stats

from .base import (BaseProposal, BaseAdaptiveSupport)
from .normal import Normal


class DifferentialEvolution(BaseProposal):
    """
    Differential evolution proposal with fixed...

    This proposal may handle one or more parameters.

    Parameters
    ----------
    parameters : (list of) str
        The names of the parameters to produce proposals for.

    jump_interval : int, optional
        The jump interval of the proposal, the proposal only gets called every
        jump interval-th time. After ``jump_interval_duration`` number of
        proposal steps elapses the proposal will again be called on every
        chain iteration. By default ``jump_interval`` = 1.
    jump_interval_duration : int, optional
        The number of proposals steps during which values of ``jump_interval``
        other than 1 are used. After this elapses the proposal is called on
        each iteration.

    Attributes
    ----------

    """

    name = 'differential_evolution'
    symmetric = True

    def __init__(self, parameters, jump_interval=1,
                 jump_interval_duration=None):
        self.parameters = parameters
        self.ndim = len(self.parameters)
        self.set_jump_interval(jump_interval, jump_interval_duration)
        self._scale = numpy.ones(self.ndim) * 2.38 * numpy.sqrt(self.ndim)
        self._jump_stds = numpy.ones(self.ndim)

    @property
    def state(self):
        return {'random_state': self.random_state}

    def set_state(self, state):
        self.random_state = state['random_state']

    def _jump(self, fromx, chain):
        # Randomly pick two points from the chain
        N = chain.iteration
        if N > chain.positions.size:
            raise NotImplementedError("Account for having saved some points.")
        if not N > 1:
            raise NotImplementedError("Need to do a few initial steps with a "
                                      "different proposal")
        i, j = chain.random_generator.choice(range(N), size=2, replace=False)

        self._jump_stds[:] = 1.
        self._jump_stds[:] = self.random_generator.normal(scale=self._scale)

        newpt = [fromx[p]
                 + self._jump_stds[k]
                 * (chain.positions[i][p] - chain.positions[j][p])
                 for k, p in enumerate(self.parameters)]
        
        return dict(zip(self.parameters, newpt))

    def _logpdf(self, xi, givenx):
        return stats.norm.logpdf(x=self._jump_stds, scale=self._scale).sum()
