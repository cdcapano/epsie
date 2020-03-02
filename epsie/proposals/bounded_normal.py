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

from .normal import (Normal)


class BoundedNormal(Normal):
    """Uses a normal distribution with fixed bounds on a parameter.

    This uses a truncated normal distribution for jump proposals, with the
    truncation dependent on where a given point is with respect to the
    parameter's boundaries. As a result, the proposal changes depending on
    where it is in the parameter space.

    This proposal can handle more than one parameter; however, parameters must
    all be independent of each other.

    Parameters
    ----------
    parameters : (list of) str
        The names of the parameters to produce proposals for.
    boundaries : dict
        Dictionary mapping parameters to boundaries. Boundaries must be a
        tuple or iterable of length two.
    cov : array, optional
        The covariance matrix of the parameters. May provide either a single
        float, a 1D array with length ``ndim``, or an ``ndim x ndim`` array,
        where ``ndim`` = the number of parameters given. If a single float or
        a 1D array is given, will use a diagonal covariance matrix (i.e., all
        parameters are independent of each other). Default (None) is to use
        unit variance for all parameters.
    """
    name = 'bounded_normal'
    symmetric = False

    def __init__(self, parameters, boundaries, cov=None):
        super(BoundedNormal, self).__init__(parameters, cov=cov)
        # check that a diagonal covariance was provided
        if not self.isdiagonal:
            raise ValueError("Only independent variables are supported "
                             "(all off-diagonal terms in the covariance "
                             "must be zero)")
        # set the boundaries
        self._boundaries = None
        self._lowerbnd = None
        self._upperbnd = None
        self.boundaries = boundaries

    @property
    def boundaries(self):
        """Dictionary of parameter -> boundaries."""
        return self._boundaries

    @boundaries.setter
    def boundaries(self, boundaries):
        """Sets the boundaries, making sure that widths are provided for
        each parameter in ``parameters``.
        """
        try:
            self._boundaries = {p: boundaries[p]
                                for p in self.parameters}
        except KeyError:
            raise ValueError("must provide a boundary for every parameter")
        # set lower and upper bound arrays for speed
        self._lowerbnd = numpy.array([self._boundaries[p][0]
                                      for p in self.parameters])
        self._upperbnd = numpy.array([self._boundaries[p][1]
                                      for p in self.parameters])

    def __contains__(self, testpt):
        # checks if the given parameters are in the bounds
        testpt = testpt.copy()
        isin = None
        for p in list(testpt.keys()):
            val = testpt.pop(p)
            bnds = self.boundaries[p]
            if isinstance(val, numpy.ndarray):
                thisisin = ((val >= bnds[0]) & (val <= bnds[1]))
            else:
                thisisin = bnds[0] <= val <= bnds[1]
            if isin is None:
                isin = thisisin
            else:
                isin &= thisisin
        if testpt:
            raise ValueError("unrecognized parameter(s) {}"
                             .format(', '.join(testpt.keys())))
        return isin

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
            mu = fromx[p]
            newpt = {p: self.random_generator.normal(mu, self._std[ii])}
            while newpt not in self:
                newpt = {p: self.random_generator.normal(mu, self._std[ii])}
            to_x.update(newpt)
        return to_x

    def logpdf(self, xi, givenx):
        mu = numpy.array([givenx[p] for p in self.parameters])
        xi = numpy.array([xi[p] for p in self.parameters])
        scale = 1./self._std
        return stats.truncnorm.logpdf(xi-mu, self._lowerbnd-mu,
                                      self._upperbnd-mu,
                                      scale=self._std).sum()
