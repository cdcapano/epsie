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

#from .base import BaseProposal
from .normal import Normal


class BirthDeath(Normal):
    """Uses a normal distribution with a fixed variance for proposals.
    """

    name = 'birthdeath'
    symmetric = False

    def __init__(self, model_parameters, model_index, prior_rvs,
                 ncomp_bound=None, logprior_k=None, model_cov=None):
        # Actually in passing model parameters to normal I need to expand the
        # list to account for every possible combination

        super(BirthDeath, self).__init__(model_parameters, cov=model_cov)
        self.model_index = model_index
        # set min and max number of components
        self._ncomp_bound = None
        self._prior_rvs = None
        self.ncomp_bound = ncomp_bound
        self._logprior_k = logprior_k
        self.prior_rvs = prior_rvs

        # Deal with how to store parameters


    @property
    def ncomp_bound(self):
        """A tuple (kmin, kmax)"""
        return self._ncomp_bound

    @ncomp_bound.setter
    def ncomp_bound(self, boundaries):
        """Sets kmin, kmax
        """
        if boundaries is None:
            self._ncomp_bound = (1, 10)
        elif not isinstance(boundaries, tuple):
            raise ValueError("must provide a tuple")
        self._ncomp_bound = boundaries

    @property
    def prior_rvs(self):
        """Returns sample from the prior, does not include the model index
        """
        newpt = self._prior_rvs()
        newpt.pop(self.model_index)
        return newpt

    @prior_rvs.setter
    def prior_rvs(self, prior):
        try:
            tp = prior(size=1)
            for name in (list(self.parameters) + [self.model_index]):
                tp[name]
        except:
            raise ValueError("prior does not return the right parameters")
        self._prior_rvs = prior

    def logprior_k(self, k):
        """Evaluated the model index probability. By default assumes uniform
        """
        if self._logprior_k == None:
            self._logprior_k = stats.randint(self.ncomp_bound[0],\
                                    self.ncomp_bound[1]).logpmf
        return self._logprior_k(k)

    def jump(self, fromx):
        dimchange, birth = self.move(fromx)
        print(dimchange, birth)
        # the normal RVS is much faster than the multivariate one, so use it
        # if we can
        if not dimchange:
            if self.isdiagonal:
                mu = [fromx[p] for p in self.parameters]
                newpt = self.random_generator.normal(mu, self._std)
            else:
                newpt = self.random_generator.multivariate_normal(
                    [fromx[p] for p in self.parameters], self.cov)
            newpt = dict(zip(self.parameters, newpt))
            newpt[self.model_index] = fromx[self.model_index]
            return newpt
        else:
            if birth:
                pass
                # Add a new sinusoid
            else:
                pass
                # Remove a sinusoid
            newpt = fromx.copy()
            newpt[self.model_index] += 1
            return newpt

    # STILL NEEDS EDIDITING
    def logpdf(self, xi, givenx):
        mu = numpy.array([givenx[p] for p in self.parameters])
        xi = numpy.array([xi[p] for p in self.parameters])
        a = (self._lowerbnd - mu)/self._std
        b = (self._upperbnd - mu)/self._std
        return stats.truncnorm.logpdf(xi, a, b, loc=mu, scale=self._std).sum()

    def move(self, fromx):
        """Prior probability of either birth death or update"""
        # move this c up
        c = 0.25
        k = fromx[self.model_index]
        birth = c*min(1, numpy.exp(self.logprior_k(k + 1) - self.logprior_k(k)))
        death = c*min(1, numpy.exp(self.logprior_k(k - 1) - self.logprior_k(k)))
        # Update happens with probability 1 - birth - death
        # Random number U(0, 1)
        u = self.random_generator.uniform()
        if u <= birth:
            return True, True
        elif u <= birth + death:
            return True, False
        else:
            return False, None










