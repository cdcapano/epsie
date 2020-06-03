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
        self.boundaries = {p: (int(self._floorceil(b.lower)),
                               int(self._floorceil(b.upper)))
                           for p, b in self.boundaries.items()}

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
        mu = self._floorceil(
            numpy.array([givenx[p] for p in self.parameters])).astype(int)
        xi = self._ceilfloor(
            numpy.array([xi[p] for p in self.parameters])).astype(int)
        # check if any xi == mu; if so, just return -inf, since this proposal
        # will be 0 for that
        if (xi == mu).any():
            return -numpy.inf
        a = (self._lowerbnd - mu)/self._std
        b = (self._upperbnd - mu)/self._std
        # the pdf is the difference in the trunc norm's cdf around xi; whether
        # we take the difference between xi+1 and xi or xi and xi-1 depends
        # on where xi is w.r.t. mu
        x0 = numpy.empty(len(xi), dtype=int)
        x0[:] = xi[:]
        mask = xi > mu
        if mask.any():
            x0[mask] -= 1
        x1 = x0 + 1
        p = (stats.truncnorm.cdf(x1, a, b, loc=mu, scale=self._std)
             - stats.truncnorm.cdf(x0, a, b, loc=mu, scale=self._std))
        return numpy.log(p).sum()
