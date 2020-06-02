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
        self.boundaries = {p: (int(b.lower), int(b.upper))
                           for p, b in self.boundaries.items()}

    def jump(self, fromx):
        # use BoundedNormal for the jump
        to_x = super(BoundedDiscrete, self).jump(fromx)
        # cast to integer
        return {p: int(x) for p, x in to_x.items()}

    def logpdf(self, xi, givenx):
        # the pdf is the difference in the trunc norm's cdf between the xi and
        # xi+1
        mu = numpy.array([givenx[p] for p in self.parameters])
        xi = numpy.array([xi[p] for p in self.parameters]).astype(int)
        a = (self._lowerbnd - mu)/self._std
        b = (self._upperbnd - mu)/self._std
        p = (stats.truncnorm.cdf(xi+1, a, b, loc=mu, scale=self._std)
             - stats.truncnorm.cdf(xi, a, b, loc=mu, scale=self._std)).sum()
        return numpy.log(p)
