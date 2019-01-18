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

from __future__ import absolute_import

import numpy
from scipy import stats

from .base import BaseProposal


class Normal(BaseProposal):
    """Uses a normal distribution with a fixed variance for proposals."""

    name = 'normal'
    symmetric = True

    def __init__(self, parameters, cov=None, random_state=None):
        self.parameters = parameters
        # create a frozen distribution to draw from/evaluate
        if cov is None:
            cov = numpy.diag(numpy.ones(len(parameters)))
        elif len(parameters) != cov.ndim:
            raise ValueError("dimension of covariance matrix does not match "
                             "given number of parameters")
        self.cov = cov
        self._dist = stats.multivariate_normal(cov=cov,
                                               seed=random_state)

    def set_random_state(self, random_state):
        self._dist.random_state = random_state

    @property
    def state(self):
        return {'random_state': self.random_state.get_state()}

    def set_state(self, state):
        self.random_state.set_state(state['random_state'])

    @property
    def random_state(self):
        return self._dist.random_state

    def jump(self, size=1):
        rvals = self._dist.rvs(size=size)
        if size == 1:
            rvals = {p: rvals[ii] for ii, p in enumerate(self.parameters)}
        else:
            rvals = {p: rvals[:, ii] for ii, p in enumerate(self.parameters)}
        return rvals 

    def logpdf(self, **vals):
        return self._dist.logpdf([vals[p] for p in self.parameters])
