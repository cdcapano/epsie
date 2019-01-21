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

    def __init__(self, parameters, cov=None):
        if isinstance(parameters, (str, unicode)):
            parameters = [parameters]
        self.parameters = tuple(parameters)
        self.ndim = len(parameters)
        if cov is None and self.ndim == 1:
            cov = 1.
        elif cov is None:
            cov = numpy.diag(numpy.ones(len(parameters)))
        # check that dimensionality makes sense
        if self.ndim == 1 and isinstance(cov, numpy.ndarray) \
                or self.ndim > 1 and not isinstance(cov, numpy.ndarray) \
                or self.ndim > 1 and self.ndim != cov.ndim:
            raise ValueError("dimension of covariance matrix does not match "
                             "given number of parameters")
        self.cov = cov

    @property
    def state(self):
        return {'random_state': self.random_state}

    def set_state(self, state):
        self.random_state = state['random_state']

    def jump(self, fromx):
        if self.ndim == 1:
            newpt = self.random_generator.normal(fromx[p], self.cov)
            jump = {p: newpt}
        else:
            newpt = self.random_generator.multivariate_normal(
                [fromx[p] for p in self.parameters], self.cov)
            jump = {p: newpt[ii] for ii, p in enumerate(self.parameters)}
        return jump
                    

    def logpdf(self, xi, givenx):
        means = [givenx[p] for p in self.parameters]
        if self.ndim == 1:
            p = self.parameters[0]
            logp = stats.normal.logpdf(xi[p], loc=givenx[p], scale=self.cov)
        else:
            logp = stats.multivariate_normal(
                [xi[p] for p in self.parameters],
                mean=[givenx[p] for p in self.parameters],
                cov=self.cov)
        return logp
