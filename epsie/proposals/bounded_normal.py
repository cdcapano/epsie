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
        where ``ndim`` = the number of parameters given. If 2D array is given,
        the off-diagonal terms must be zero. Default is 1 for all parameters.
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
    name = 'bounded_normal'
    symmetric = False

    def __init__(self, parameters, boundaries, cov=None, jump_interval=1,
                 jump_interval_duration=None):
        super(BoundedNormal, self).__init__(
            parameters, cov=cov, jump_interval=jump_interval,
            jump_interval_duration=jump_interval_duration)
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
            self._boundaries = {p: Boundaries(boundaries[p])
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
        for p in self.parameters:
            try:
                val = testpt.pop(p)
            except KeyError:
                # only testing a subset of the parameters, which is allowed
                continue
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

    def _jump(self, fromx):
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

    def _logpdf(self, xi, givenx):
        mu = numpy.array([givenx[p] for p in self.parameters])
        xi = numpy.array([xi[p] for p in self.parameters])
        a = (self._lowerbnd - mu)/self._std
        b = (self._upperbnd - mu)/self._std
        return stats.truncnorm.logpdf(xi, a, b, loc=mu, scale=self._std).sum()


class AdaptiveBoundedNormal(AdaptiveSupport, BoundedNormal):
    r"""A bounded normal proposoal with adaptive variance, using the algorithm
    from Veitch et al.

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
    start_step : int, optional
        The proposal step to start doing the adaptation (:math:`k_0+1` in the
        equation below). Must be greater than zero. Default is 1.
    jump_interval : int, optional
        The jump interval of the proposal, the proposal only gets called every
        jump interval-th time. After ``adaptation_duration`` number of
        proposal steps elapses the proposal will again be called on every
        chain iteration. By default ``jump_interval`` = 1.
    \**kwargs :
        All other keyword arguments are passed to
        :py:func:`AdaptiveSupport.setup_adaptation`. See that function for
        details.
    """
    name = 'adaptive_bounded_normal'
    symmetric = False

    def __init__(self, parameters, boundaries, adaptation_duration,
                 start_step=1, jump_interval=1, **kwargs):
        # set the parameters, initialize the covariance matrix
        super(AdaptiveBoundedNormal, self).__init__(
            parameters, boundaries, jump_interval=jump_interval,
            jump_interval_duration=adaptation_duration)
        # set up the adaptation parameters
        self.setup_adaptation(self.boundaries, adaptation_duration,
                              start_step=start_step, **kwargs)


class SSAdaptiveBoundedNormal(SSAdaptiveSupport, BoundedNormal):
    r"""A bounded normal proposoal using the Sivia and Skilling algorithm for
    adjusting the variance.

    By default, the maximum possible standard deviation is set to be 1.49 times
    largest boundary width (via the ``max_cov`` argument). At this value, there
    is a 50% chance that a jump will be larger than the boundary width. For
    more details on the adaptation algorithm, see
    :py:class:`SSAdaptiveSupport`.

    Parameters
    ----------
    parameters : (list of) str
        The names of the parameters.
    boundaries : dict
        Dictionary mapping parameters to boundaries. Boundaries must be a
        tuple or iterable of length two. The boundaries will be used for the
        prior widths in the adaptation algorithm.
    cov : array, optional
        The covariance matrix of the parameters. May provide either a single
        float, a 1D array with length ``ndim``, or an ``ndim x ndim`` array,
        where ``ndim`` = the number of parameters given. If 2D array is given,
        the off-diagonal terms must be zero. Default is 1 for all parameters.
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
    name = 'ss_adaptive_bounded_normal'
    symmetric = False

    def __init__(self, parameters, boundaries, cov=None, jump_interval=1,
                 jump_interval_duration=None, **kwargs):
        # set the parameters, initialize the covariance matrix
        super(SSAdaptiveBoundedNormal, self).__init__(
              parameters, boundaries, cov=cov, jump_interval=jump_interval,
              jump_interval_duration=jump_interval_duration)
        # set up the adaptation parameters
        if 'max_cov' not in kwargs:
            # set the max std to be (1.49*abs(bounds)
            maxwidth = max(map(abs, self.boundaries.values()))
            kwargs['max_cov'] = (1.49*maxwidth)**2
        self.setup_adaptation(**kwargs)


class ATAdaptiveBoundedNormal(ATAdaptiveSupport, BoundedNormal):
    r"""A bounded adaptive proposal, using the algorithm from Andrieu & Thoms.

    See :py:class:`ATAdaptiveSupport` for details on the adaptation algorithm.

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
    componentwise : bool, optional
        Whether to include a componentwise scaling of the parameters
        (algorithm 6 in [1]). By default set to False (algorithm 4 in [1]).
        Componentwise scaling `ndim` times more expensive than global
        scaling.
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
    name = 'at_adaptive_bounded_normal'
    symmetric = False

    def __init__(self, parameters, boundaries, adaptation_duration,
                 componentwise=False, start_step=1, target_rate=None,
                 jump_interval=1):
        # set the parameters, initialize the covariance matrix
        super(ATAdaptiveBoundedNormal, self).__init__(
            parameters, boundaries, jump_interval=jump_interval,
            jump_interval_duration=adaptation_duration)
        # set up the adaptation parameters
        self.setup_adaptation(adaptation_duration=adaptation_duration,
                              diagonal=True, componentwise=componentwise,
                              start_step=start_step, target_rate=target_rate)


class Boundaries(tuple):
    """Support class for bounded proposals.

    This is basically a length-2 tuple with an abs function. Also has lower
    and upper attributes to get the first and second item.

    Example
    --------
    >>> b = Boundaries((-1, 1))
    >>> b[0], b[1]
    (-1, 1)
    >>> b.lower, b.upper
    (-1, 1)
    >>> abs(b)
    2 """
    def __new__(cls, args):
        self = tuple.__new__(cls, args)
        if len(args) != 2:
            raise ValueError("must provide only a lower and upper bound")
        return self

    def __abs__(self):
        """Returns the absolute value of the difference in the boundaries."""
        return abs(self[1] - self[0])

    @property
    def lower(self):
        """The lower bound."""
        return self[0]

    @property
    def upper(self):
        """The upper bound."""
        return self[1]
