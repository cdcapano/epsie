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

from .normal import (Normal, AdaptiveSupport)
from .bounded_normal import (BoundedNormal)


class NormalDiscrete(Normal):
    """A proposal for discrete parameters.
    """
    name = 'discrete'
    symmetric = True

    def __init__(self, parameters, cov=None):
        super(NormalDiscrete, self).__init__(parameters, cov=cov)
        # this only works for diagonal pdfs
        if not self.isdiagonal:
            raise ValueError("Only independent variables are supported "
                             "(all off-diagonal terms in the covariance "
                             "must be zero)")
        # cache for the cdfs
        self._cdfcache = [{}]*len(self.parameters)
        self._cachedstd = [None]*len(self.parameters)

    def _cdf(self, pi, dx, std):
        """Caches CDF for faster call back.

        Parameters
        ----------
        pi : int
            Index of parameter that is being evaluated.
        dx : int
            Value to evaulate. Should be the difference between a point and
            the mean.
        std : float
            Standard deviation of the distribution.
        """
        if std != self._cachedstd[pi]:
            self._cdfcache[pi].clear()
        try:
            return self._cdfcache[pi][dx]
        except KeyError:
            cdf = stats.norm.cdf(dx, scale=std)
            self._cdfcache[pi][dx] = cdf
            self._cachedstd[pi] = std
            return cdf

    def jump(self, fromx):
        to_x = {}
        for ii, p in enumerate(self.parameters):
            dx = self.random_generator.normal(0, self._std[ii])
            # convert to int
            dx = int(_floorceil(dx))
            to_x[p] = int(fromx[p]) + dx
        return to_x

    def logpdf(self, xi, givenx):
        logp = 0
        for ii, p in enumerate(self.parameters):
            dx = int(numpy.floor(xi[p] - givenx[p]))
            # if given point is same as test point, the pdf will just be 0;
            # don't need to evaluate the other parameters
            if dx == 0:
                return -numpy.inf
            # we'll just evaluate positive dx, since the distribution is
            # symmetric about 0
            dx = abs(dx)
            p0 = self._cdf(ii, dx-1, self._std[ii])
            p1 = self._cdf(ii, dx, self._std[ii])
            dp = p1 - p0
            if dp == 0:
                return -numpy.inf
            logp += numpy.log(dp)
        return logp


class BoundedDiscrete(BoundedNormal):
    """A proposal for discrete parameters with bounds.
    """
    name = 'bounded_discrete'
    symmetric = False

    def __init__(self, parameters, boundaries, cov=None):
        super(BoundedDiscrete, self).__init__(parameters, boundaries, cov=cov)
        # ensure boundaries are integers
        self.boundaries = {p: (int(_floorceil(b.lower)),
                               int(_floorceil(b.upper)))
                           for p, b in self.boundaries.items()}
        # cache for the cdfs
        self._cdfcache = [{}]*len(self.parameters)
        self._cachedstd = [None]*len(self.parameters)

    def _cdf(self, pi, x, a, b, mu, std):
        """Caches CDF for faster call back.

        Parameters
        ----------
        pi : int
            Index of parameter that is being evaluated.
        x : int
            Value to evaulate.
        a : int
            Lower bound of the distribution (with respect to mu).
        b : int
            Upper bound of the distribution (with respect to mu).
        mu : int
            Mean of the distribution.
        std : float
            Standard deviation of the distribution.
        """
        if std != self._cachedstd[pi]:
            self._cdfcache[pi].clear()
        try:
            return self._cdfcache[pi][x, a, b, mu]
        except KeyError:
            cdf = stats.truncnorm.cdf(x, a/std, b/std, loc=mu, scale=std)
            self._cdfcache[pi][x, a, b, mu] = cdf
            self._cachedstd[pi] = std
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
                deltax = int(_floorceil(deltax))
                newpt = {p: fromx[p]+deltax}
                inbnds = newpt in self
            to_x.update(newpt)
        return to_x

    def logpdf(self, xi, givenx):
        logp = 0
        for ii, p in enumerate(self.parameters):
            mu = int(_floorceil(givenx[p]))
            x = int(_ceilfloor(xi[p]))
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
            p0 = self._cdf(ii, x0, a, b, mu, self._std[ii])
            p1 = self._cdf(ii, x1, a, b, mu, self._std[ii])
            logp += numpy.log(p1 - p0)
        return logp


def _floorceil(x):
    """Returns floor (ceil) of values < (>) 0."""
    return numpy.sign(x)*numpy.ceil(abs(x))


def _ceilfloor(x):
    """Returns the ceil (floor) of values < (>) 0."""
    return numpy.sign(x)*numpy.floor(abs(x))
