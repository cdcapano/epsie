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

from copy import deepcopy

class BirthDeath(Normal):

    name = 'birthdeath'
    symmetric = False

    # within model pars, all pars, index pars

    def __init__(self, model_pars, all_pars, model_index, prior_dist,
                 ncomp_bound, cov=None):
        self.model_pars = model_pars
        self.all_pars = all_pars
        self._model_index = None
        self._ncomp_bound = None
        self._prior_rvs = None

        super(BirthDeath, self).__init__(self.all_pars, cov=cov)
        self.model_index = model_index
        self.ncomp_bound = ncomp_bound
        self.prior_dist = prior_dist

    @property
    def model_index(self):
        """Parameter name detoning the number of fundamental waves"""
        return self._model_index

    @model_index.setter
    def model_index(self, index):
        if not isinstance(index, str):
            raise ValueError("must provide a str")
        self._model_index = index

    @property
    def ncomp_bound(self):
        """A tuple (kmin, kmax)"""
        return self._ncomp_bound

    @ncomp_bound.setter
    def ncomp_bound(self, ncomp_bound):
        if not isinstance(ncomp_bound, tuple):
            raise ValueError("must provide a tuple")
        self._ncomp_bound = ncomp_bound

    @property
    def prior_dist(self):
        """Prior distributions used to generate new samples, contain rvs and
        logpdf methods"""
        return self._prior_dist

    @prior_dist.setter
    def prior_dist(self, prior_dist):
        try:
            for name in self.model_pars:
                tp = prior_dist[name].rvs()
                prior_dist[name].logpdf(tp)
            tp = prior_dist[self.model_index].rvs()
            prior_dist[self.model_index].logpmf(tp)
        except:
            raise ValueError("prior does not return the right parameters")
        self._prior_dist = prior_dist

    def move(self, fromx):
        """Prior probability of either birth death or update"""
        # move this c up
        c = 0.25
        k = fromx[self.model_index]
        logpk = self.prior_dist[self.model_index].logpmf
        birth = c*min(1, numpy.exp(logpk(k + 1) - logpk(k)))
        death = c*min(1, numpy.exp(logpk(k - 1) - logpk(k)))
        # Update happens with probability 1 - birth - death
        # Random number U(0, 1)
        u = self.random_generator.uniform()
        if u <= birth:
            return True, True
        elif u <= birth + death:
            return True, False
        else:
            return False, None

    def prior_sample(self):
        return {p : float(self.prior_dist[p].rvs(size=1)) for p in self.model_pars}

    def jump(self, fromx):
        """
        A rough draft of the jump method. I will ''de-uglify'' this before
        submitting a pull request.
        """
        dimchange, birth = self.move(fromx)
#        dimchange, birth = False, None
        # the normal RVS is much faster than the multivariate one, so use it
        # if we can
        if not dimchange:
            # This one still updates the wrong parameters
            print('update mode')
            if self.isdiagonal:
                print('is diagonal')
                mu = [fromx[p] for p in self.parameters]
                print(mu)
                newpt = self.random_generator.normal(mu, self._std)
            else:
                newpt = self.random_generator.multivariate_normal(
                    [fromx[p] for p in self.parameters], self.cov)
            print(self.parameters, newpt.shape)
            newpt = dict(zip(self.parameters, newpt))
            newpt[self.model_index] = fromx[self.model_index]
        else:
            newpt = deepcopy(fromx)
            if birth:
                print('giving birth')
                knew = int(fromx[self.model_index]) + 1
                newcomps = self.prior_sample()
                print(newcomps)
                for p in self.model_pars:
                    newpt['{}{}'.format(p, knew)] = newcomps[p]
                newpt[self.model_index] += 1

            else:
                print('removing a component')
                krem = numpy.random.choice(numpy.arange(1, fromx[self.model_index]))
                print('picked {}'.format(krem))
                for p in self.model_pars:
                    newpt['{}{}'.format(p, krem)] = numpy.nan
                # Ok now re-order the frequencies so that nans are at the end
                oldmax = int(fromx['k'])
                if krem != oldmax:
                    for p in self.model_pars:
                        newpt['{}{}'.format(p, krem)] = newpt['{}{}'.format(p, oldmax)]
                        newpt['{}{}'.format(p, oldmax)] = numpy.nan

                newpt[self.model_index] -= 1
        return newpt

    # STILL NEEDS EDIDITING, copied straight from BoundedNormal
    def logpdf(self, xi, givenx):
        mu = numpy.array([givenx[p] for p in self.parameters])
        xi = numpy.array([xi[p] for p in self.parameters])
        a = (self._lowerbnd - mu)/self._std
        b = (self._upperbnd - mu)/self._std
        return stats.truncnorm.logpdf(xi, a, b, loc=mu, scale=self._std).sum()





