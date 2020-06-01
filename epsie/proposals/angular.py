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
from .bounded_normal import Boundaries


class Angular(Normal):
    r"""A propsal distribution for parameters that are cyclic over
    :math:`[0, 2\pi)`.

    To generate a jump proposal from a given point, a truncated normal
    distribution centered on the point and with bounds at :math:`\pm \pi`
    around the point. Cyclic bounds are then applied, resulting in a proposed
    point that is in :math:`[0, 2\pi)`. For example, if a the given point is
    at :math:`1.75\pi` and the jump is :math:`+0.5\pi`, then proposed point
    will be at :math:`0.25\pi`. Since the same proposal distribution is used
    regardless of the location, this is a symmetric proposal.

    This proposal can handle more than one parameter; however, parameters must
    all be independent of each other.

    Parameters
    ----------
    parameters : (list of) str
        The names of the parameters to create proposals for.
    cov : (array of) float, optional
        The covariance to use for the underlying truncated normal
        distribution, in squared radians. May provide either a single float, a
        1D array with length ``ndim``, or an ``ndim x ndim`` array, where
        ``ndim`` = the number of parameters given. If 2D array is given, the
        off-diagonal terms must be zero. Default is 1 sq. radian for all
        parameters.
    """
    name = 'angular'
    symmetric = True

    def __init__(self, parameters, cov=None):
        super(Angular, self).__init__(parameters, cov=cov)
        # _halfwidth is half the width of the interval, in factors of pi
        self._halfwidth = 1.
        self._factor = numpy.pi
        # for speed up
        self._invfactor = 1./numpy.pi
        self._logfactor = numpy.log(self._factor)

    def _apply_cyclic(self, value):
        r"""Applies cyclic boundaries to the given value.

        This causes the value to lie within 0 and ``2*_halfwidth``. With the
        default ``_halfwidth`` = 1, this means the value will be within 0
        and 2.

        Parameters
        ----------
        value : float
            The value to apply the boundaries to.
        offset : float, optional
            Offset to apply. If none provided, will use the ``_offset``
            attribute.

        Returns
        -------
        float :
            The value remapped to be within the boundaries.
        """
        return value % (2*self._halfwidth)

    def jump(self, fromx):
        # we'll do something similar as the bounded normal here: draw points
        # using a bounded normal centered on 0 with bounds at -pi and pi.
        # we'll just use rejection sampling for this
        to_x = {}
        for ii, p in enumerate(self.parameters):
            # remove the common factor from the std
            std = self._std[ii] * self._invfactor
            newpt = self.random_generator.normal(scale=std)
            while abs(newpt) > self._halfwidth:
                newpt = self.random_generator.normal(scale=std)
            # add to fromx, then put back within bounds
            newpt += fromx[p] * self._invfactor
            newpt = self._apply_cyclic(newpt)
            # now convert back to radians
            to_x[p] = newpt * self._factor
        return to_x

    def logpdf(self, xi, givenx):
        # convert to arrays
        xi = numpy.array([xi[p]*self._invfactor for p in self.parameters])
        givenx = numpy.array([givenx[p]*self._invfactor
                              for p in self.parameters])
        # make sure givenx is in bounds and figure out how far it is from the
        # middle of the interval
        givenx = self._apply_cyclic(givenx)
        dx = self._halfwidth - givenx
        # apply cyclic boundaries to xi
        xi = self._apply_cyclic(xi)
        # shift xi and apply cyclic boundaries again
        xi = self._apply_cyclic(xi + dx)
        # now evaluate using a truncated normal centered on the middle of
        # the interval
        # remove the common factor from the std
        std = self._std * self._invfactor
        b = self._halfwidth/std
        a = -b
        return stats.truncnorm.logpdf(xi, a, b, loc=self._halfwidth,
                                      scale=std).sum() - self._logfactor


class AdaptiveAngular(AdaptiveSupport, Angular):
    r"""An angular proposoal with adaptive variance.

    See :py:class:`AdaptiveSupport` for details on the adaptation algorithm.

    Parameters
    ----------
    parameters : (list of) str
        The names of the parameters.
    boundaries : dict
        Dictionary mapping parameters to boundaries. Boundaries must be a
        tuple or iterable of length two. The boundaries will be used for the
        prior widths in the adaptation algorithm.
    adaptation_duration : int
        The number of iterations over which to apply the adaptation. No more
        adaptation will be done once a chain exceeds this value.
    \**kwargs :
        All other keyword arguments are passed to
        :py:func:`AdaptiveSupport.setup_adaptation`. See that function for
        details.
    """
    name = 'adaptive_angular'
    symmetric = True

    def __init__(self, parameters, adaptation_duration,
                 **kwargs):
        # set the parameters, initialize the covariance matrix
        super(AdaptiveAngular, self).__init__(parameters)
        # all parameters have the same (cyclic) boundaries
        boundaries = {p: Boundaries((0, 2*self._halfwidth*self._factor))
                      for p in self.parameters}
        # set up the adaptation parameters
        self.setup_adaptation(boundaries, adaptation_duration, **kwargs)
