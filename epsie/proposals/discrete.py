# Copyright (C) 2020  Collin Capano
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

from .normal import (AdaptiveSupport)
from .bounded_normal import (BoundedNormal)


class BoundedDiscrete(BoundedNormal):
    """A proposal for discrete parameters with bounds.
    """
    name = 'bounded_discrete'
    symmetric = False


    def __init__(self, parameters, boundaries, cov=None):
        super(BoundedDiscrete, self).__init__(parameters, boundaries, cov=cov)
        # ensure boundaries are integers
        self.boundaries = {p: (int(self._floorceil(b.lower)),
                               int(self._floorceil(b.upper)))
                           for p, b in self.boundaries.items()}
        # cache for the cdfs
        self._cdfcache = {}
        self._cachedstd = None

    def _cdf(self, x, a, b, mu, std):
        """Caches CDF for faster call back."""
        if std != self._cachedstd:
            self._cdfcache.clear()
        try:
            return self._cdfcache[x, a, b, mu]
        except KeyError:
            cdf = stats.truncnorm.cdf(x, a/std, b/std, loc=mu, scale=std)
            self._cdfcache[x, a, b, mu] = cdf
            self._cachedstd = std
            return cdf

    def jump(self, fromx):
        # make sure we're in bounds
        if fromx not in self:
            raise ValueError("Given point is not in bounds; I don't know how "
                             "to jump from there.")
        # we'll just use rejection sampling to get a new point. This should
        # be reasonably fast since fromx is always at the peak of the
        # distribution
        to_x = {}
        for ii, p in enumerate(self.parameters):
            inbnds = False
            while not inbnds:
                # draw a delta x
                deltax = self.random_generator.normal(0., self._std[ii])
                # for values < 0, we want the floor; for values > 0, the
                # ceiling
                deltax = int(self._floorceil(deltax))
                newpt = {p: fromx[p]+deltax}
                inbnds = newpt in self
            to_x.update(newpt)
        return to_x

    @staticmethod
    def _floorceil(x):
        """Returns floor (ceil) of values < (>) 0."""
        return numpy.sign(x)*numpy.ceil(abs(x))

    @staticmethod
    def _ceilfloor(x):
        """Returns the ceil (floor) of values < (>) 0."""
        return numpy.sign(x)*numpy.floor(abs(x))

    def logpdf(self, xi, givenx):
        logp = 0
        for ii, p in enumerate(self.parameters):
            mu = int(self._floorceil(givenx[p]))
            x = int(self._ceilfloor(xi[p]))
            # if given point is same as test point, the pdf will just be 0;
            # don't need to evaluate the other parameters
            if x == mu:
                return -numpy.inf
            a = self._lowerbnd[ii] - mu
            b = self._upperbnd[ii] - mu
            # the pdf is the difference in the trunc norm's cdf around x;
            # whether we take the difference between x+1 and x or x and x-1
            # depends on where x is w.r.t. mu
            if x > mu:
                x0 = x - 1
                x1 = x
            else:
                x0 = x
                x1 = x + 1
            p0 = self._cdf(x0, a, b, mu, self._std[ii])
            p1 = self._cdf(x1, a, b, mu, self._std[ii])
            logp += numpy.log(p1 - p0)
        return logp
