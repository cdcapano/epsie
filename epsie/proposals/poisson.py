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

from __future__ import (absolute_import, division)

from abc import ABCMeta
from six import add_metaclass

import numpy
from scipy import stats

from .base import BaseProposal


class Poisson(BaseProposal):

    name = 'poisson'
    symmetric = False

    def __init__(self, parameters, mu):
        self.parameters = parameters
        self.ndim = len(self.parameters)
        self._mu = None
        self.mu = mu

    def _ensuredict(self, mu):
        """Function for setting the mean of the Poission
        distributon. This ensures that every parameter has a specified mean.
        """
        if len(mu) is not len(self.parameters):
            raise ValueError("must provide a mean for every parameter")
        return dict(zip(self.parameters, mu))

    @property
    def mu(self):
        """
        The mean of the Poisson distribution.
        """
        return self._mu

    @mu.setter
    def mu(self, mu):
        """
        Sets the mean of the Poisson distribution
        """
        mu = self._ensuredict(mu)
        self._mu = mu

    @property
    def state(self):
        return {'random_state': self.random_state}

    def set_state(self, state):
        self.random_state = state['random_state']

    def jump(self, fromx):
        # Drawing samples from a Poisson distribution does not depend
        # on the current position
        newpt = self.random_generator.poisson(list(self._mu.values()))
        return dict(zip(self.parameters, newpt))

    def logpdf(self, xi, givenx=None):
        xi = [xi[p] for p in self.parameters]
        logp = stats.poisson.logpmf(k=xi, mu=list(self._mu.values())).sum()
        return logp
