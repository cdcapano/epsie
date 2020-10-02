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

from .normal import (Normal, AdaptiveSupport, SSAdaptiveSupport,
                     ATAdaptiveSupport)
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
    jump_interval : int, optional
        The jump interval of the proposal, the proposal only gets called every
        jump interval-th time. After ``jump_interval_duration`` number of
        proposal steps elapses the proposal will again be called on every
        chain iteration. By default ``jump_interval`` = 1.
    jump_interval_duration : int, optional
        The number of proposals steps during which values of ``jump_interval``
        other than 1 are used. After this elapses the proposal is called on
        each iteration.

    """
    name = 'angular'
    symmetric = True

    def __init__(self, parameters, cov=None, jump_interval=1,
                 jump_interval_duration=None):
        super(Angular, self).__init__(
            parameters, cov=cov, jump_interval=jump_interval,
            jump_interval_duration=jump_interval_duration)
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

    def _jump(self, fromx):
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

    def _logpdf(self, xi, givenx):
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
    r"""An angular proposoal with adaptive variance, using the algorithm from
    Veitch et al.

    See :py:class:`AdaptiveSupport` for details on the adaptation algorithm.

    Parameters
    ----------
    parameters : (list of) str
        The names of the parameters.
    boundaries : dict
        Dictionary mapping parameters to boundaries. Boundaries must be a
        tuple or iterable of length two. The boundaries will be used for the
        prior widths in the adaptation algorithm.
    adaptation_duration: int
        The number of proposal steps over which to apply the adaptation. No
        more adaptation will be done once a proposal exceeds this value.
    jump_interval : int, optional
        The jump interval of the proposal, the proposal only gets called every
        jump interval-th time. After ``adaptation_duration`` number of
        proposal steps elapses the proposal will again be called on every
        chain iteration. By default ``jump_interval`` = 1.
    start_step : int, optional
        The proposal step to start doing the adaptation (:math:`k_0+1` in the
        equation below). Must be greater than zero. Default is 1.
    \**kwargs :
        All other keyword arguments are passed to
        :py:func:`AdaptiveSupport.setup_adaptation`. See that function for
        details.
    """
    name = 'adaptive_angular'
    symmetric = True

    def __init__(self, parameters, adaptation_duration, jump_interval=1,
                 start_step=1, **kwargs):
        # set the parameters, initialize the covariance matrix
        super(AdaptiveAngular, self).__init__(
            parameters, jump_interval=jump_interval,
            jump_interval_duration=adaptation_duration)
        # all parameters have the same (cyclic) boundaries
        boundaries = {p: Boundaries((0, 2*self._halfwidth*self._factor))
                      for p in self.parameters}
        # set up the adaptation parameters
        self.setup_adaptation(boundaries, adaptation_duration,
                              start_step=start_step, **kwargs)


class SSAdaptiveAngular(SSAdaptiveSupport, Angular):
    r"""An angular proposoal with adaptive variance, using the algorithm from
    Sivia and Skilling.

    See :py:class:`SSAdaptiveSupport` for details on the adaptation algorithm.

    Parameters
    ----------
    parameters : (list of) str
        The names of the parameters.
    boundaries : dict
        Dictionary mapping parameters to boundaries. Boundaries must be a
        tuple or iterable of length two. The boundaries will be used for the
        prior widths in the adaptation algorithm.
    jump_interval : int, optional
        The jump interval of the proposal, the proposal only gets called every
        jump interval-th time. After ``jump_interval_duration`` number of
        proposal steps elapses the proposal will again be called on every
        chain iteration. By default ``jump_interval`` = 1.
    jump_interval_duration : int, optional
        The number of proposals steps during which values of ``jump_interval``
        other than 1 are used. After this elapses the proposal is called on
        each iteration.
    \**kwargs :
        All other keyword arguments are passed to
        :py:func:`SSAdaptiveSupport.setup_adaptation`. See that function for
        details.
    """
    name = 'ss_adaptive_angular'
    symmetric = True

    def __init__(self, parameters, cov=None, jump_interval=1,
                 jump_interval_duration=None, **kwargs):
        # set the parameters, initialize the covariance matrix
        super(SSAdaptiveAngular, self).__init__(
            parameters, cov=cov, jump_interval=jump_interval,
            jump_interval_duration=jump_interval_duration)
        # set up the adaptation parameters
        if 'max_cov' not in kwargs:
            # set the max std to be (1.49*2*pi)
            kwargs['max_cov'] = (1.49 * 2 * numpy.pi)**2
        self.setup_adaptation(**kwargs)


class ATAdaptiveAngular(ATAdaptiveSupport, Angular):
    r"""An angular proposal with adaptive variance, using the algorithm from
    Andrieu & Thoms.

    See :py:class:`ATAdaptiveSupport` for details on the adaptation algorithm.

    Parameters
    ----------
    parameters : (list of) str
        The names of the parameters.
    adaptation_duration: int
        The number of proposal steps over which to apply the adaptation. No
        more adaptation will be done once a proposal exceeds this value.
    start_step: int, optional
        The proposal step index when adaptation phase begins.
    target_rate: float, optional
        Target acceptance ratio. By default 0.234 and 0.48 for componentwise
        scaling.
    jump_interval : int, optional
        The jump interval of the proposal, the proposal only gets called every
        jump interval-th time. After ``adaptation_duration`` number of
        proposal steps elapses the proposal will again be called on every
        chain iteration. By default ``jump_interval`` = 1.
    """
    name = 'at_adaptive_angular'
    symmetric = True

    def __init__(self, parameters, adaptation_duration, componentwise=False,
                 start_step=1, target_rate=None, jump_interval=1):
        # set the parameters, initialize the covariance matrix
        super(ATAdaptiveAngular, self).__init__(
            parameters, jump_interval=jump_interval,
            jump_interval_duration=adaptation_duration)
        # set up the adaptation parameters
        self.setup_adaptation(adaptation_duration=adaptation_duration,
                              diagonal=True, componentwise=componentwise,
                              start_step=start_step, target_rate=target_rate)
