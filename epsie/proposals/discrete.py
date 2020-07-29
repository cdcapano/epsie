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

from .normal import (Normal, AdaptiveSupport, SSAdaptiveSupport)
from .bounded_normal import (BoundedNormal)


class NormalDiscrete(Normal):
    r"""A proposal for discrete parameters.

    The jump proposals produced by this are always integer values. For a given
    point :math:`x \in \mathbb{Z}`, a new point :math:`x' = x + \bar{\Delta x}`
    is generated by using

    .. math::

        \bar{\Delta x} = \begin{cases}
            \lfloor \Delta x \rfloor \textrm{ if } \Delta x < 0, \\
            \lceil \Delta x \rceil \textrm{ if } \Delta x > 0,
            \end{cases}

    where :math:`\Delta x \sim \mathcal{N}(0, \sigma)` if `zero_jump` is False.
    This results in a stepped probability density that is zero
    :math:`\in [0, 1)`. In other words, the proposal will never produce the
    same integer on successive jumps, and it will most often draw integers
    closest to the current point.

    If `zero_jump` is True, then for a given point :math:`x \in \mathbb{Z}`,
    a new point :math:`x' = x + \bar{\Delta x}` is generated by using

    .. math:: \bar{\Delta x} = Round\left(\mathcal{N}(0, \sigma)\right),

    meaning that the proposal can now propose the same integer on successive
    jumps.

    The variance used for drawing :math:`\Delta x` need not be an integer,
    and can be set. Multiple parameters are supported, however, they all must
    be independent of each other.

    Parameters
    ----------
    parameters : (list of) str
        The names of the parameters to produce proposals for.
    cov : array, optional
        The covariance matrix of the parameters. May provide either a single
        float, a 1D array with length ``ndim``, or an ``ndim x ndim`` array,
        where ``ndim`` = the number of parameters given. If 2D array is given,
        the off-diagonal terms must be zero. Default is 1 for all parameters.
    successive: dict, optional
        Dictionary of bools, keys must be parameters and items bools. If False
        then the proposal never produces the same integer on successive jumps.
        Default is False for all parameters
    """
    name = 'discrete'
    symmetric = True
    _successive = None

    def __init__(self, parameters, cov=None, successive=None):
        super(NormalDiscrete, self).__init__(parameters, cov=cov)
        # this only works for diagonal pdfs
        if not self.isdiagonal:
            raise ValueError("Only independent variables are supported "
                             "(all off-diagonal terms in the covariance "
                             "must be zero)")
        self.successive = successive
        # cache for the cdfs
        self._cdfcache = [{}]*len(self.parameters)
        self._cachedstd = [None]*len(self.parameters)

    @property
    def successive(self):
        """Dictionary of `successive` toggles for each parameter. If True
        allows two equal integers on successive jumps.
        """
        return self._successive

    @successive.setter
    def successive(self, successive):
        if successive is None:
            self._successive = {p: False for p in self.parameters}
            return
        if not all([isinstance(successive[p], bool)
                    for p in list(successive.keys())]):
            raise ValueError('all dictionary values must be bools')
        try:
            self._successive = {p: successive[p] for p in self.parameters}
        except KeyError:
            raise ValueError('must provide zero_jump for each parameter')


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
            if self.successive[p]:
                dx = int(round(dx, 0))
            else:
                dx = int(_floorceil(dx))
            to_x[p] = int(fromx[p]) + dx
        return to_x

    def logpdf(self, xi, givenx):
        logp = 0
        for ii, p in enumerate(self.parameters):
            if self.successive[p]:
                dx = int(numpy.round(xi[p] - givenx[p], decimals=0))
                dx = abs(dx)
                p0 = self._cdf(ii, dx-0.5, self._std[ii])
                p1 = self._cdf(ii, dx+0.5, self._std[ii])
                dp = p1 - p0
            else:
                dx = int(numpy.floor(xi[p] - givenx[p]))
                # if given point is same as test point, the pdf will just be 0
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
    r"""A proposal for discrete parameters with bounds.

    This is a discretized form of :py:class:`BoundedNormal`. Jump proposals
    are produced in the same manner as :py:class:`NormalDiscrete`, except
    that the distribution used to draw :math:`\Delta x` (before applying the
    floor/ceil or round) is a truncated normal. As such, this is not a
    symmetric distribution.

    The variance used for drawing :math:`\Delta x` need not be an integer,
    and can be set. Multiple parameters are supported, however, they all must
    be independent of each other.

    Parameters
    ----------
    parameters : (list of) str
        The names of the parameters to produce proposals for.
    boundaries : dict
        Dictionary mapping parameters to boundaries. Boundaries must be a
        tuple or iterable of length two. If floats are provided, the floor
        (ceil) of the lower (upper) bound will be used.
    cov : array, optional
        The covariance matrix of the parameters. May provide either a single
        float, a 1D array with length ``ndim``, or an ``ndim x ndim`` array,
        where ``ndim`` = the number of parameters given. If 2D array is given,
        the off-diagonal terms must be zero. Default is 1 for all parameters.
    successive: dict, optional
        Dictionary of bools, keys must be parameters and items bools. If False
        then the proposal never produces the same integer on successive jumps.
        Default is False for all parameters
    """
    name = 'bounded_discrete'
    symmetric = False
    _successive = None

    def __init__(self, parameters, boundaries, cov=None, successive=None):
        super(BoundedDiscrete, self).__init__(parameters, boundaries, cov=cov)
        self.successive = successive
        # ensure boundaries are integers
        self.boundaries = {p: (int(numpy.floor(b.lower)),
                               int(numpy.ceil(b.upper)))
                           for p, b in self.boundaries.items()}
        # cache for the cdfs
        self._cdfcache = [{}]*len(self.parameters)
        self._cachedstd = [None]*len(self.parameters)

    @property
    def successive(self):
        """Dictionary of `successive` toggles for each parameter. If True
        allows two equal integers on successive jumps.
        """
        return self._successive

    @successive.setter
    def successive(self, successive):
        if successive is None:
            self._successive = {p: False for p in self.parameters}
            return
        if not all([isinstance(successive[p], bool)
                    for p in list(successive.keys())]):
            raise ValueError('all dictionary values must be bools')
        try:
            self._successive = {p: successive[p] for p in self.parameters}
        except KeyError:
            raise ValueError('must provide zero_jump for each parameter')

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
                if self.successive[p]:
                    deltax = int(round(deltax, 0))
                else:
                    # for values < 0, we want the floor; for values > 0, the
                    # ceiling
                    deltax = int(_floorceil(deltax))
                newpt = {p: int(fromx[p])+deltax}
                inbnds = newpt in self
            to_x.update(newpt)
        return to_x

    def logpdf(self, xi, givenx):
        logp = 0
        for ii, p in enumerate(self.parameters):
            if self.successive[p]:
                mu = int(round(givenx[p], 0))
                x = int(round(xi[p], 0))
                a = self._lowerbnd[ii] - mu
                b = self._upperbnd[ii] - mu
                p0 = self._cdf(ii, x-0.5, a, b, mu, self._std[ii])
                p1 = self._cdf(ii, x+0.5, a, b, mu, self._std[ii])
                logp += numpy.log(p1 - p0)
            else:
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


#
# =============================================================================
#
#                           Adaptive versions
#
# =============================================================================
#


class SSAdaptiveNormalDiscrete(SSAdaptiveSupport, NormalDiscrete):
    r"""A discrete proposoal with adaptive variance, using the algorithm from
    Sivia and Skilling.

    See :py:class:`SSAdaptiveSupport` for details on the adaptation algorithm.

    Parameters
    ----------
    parameters : (list of) str
        The names of the parameters to produce proposals for.
    cov : array, optional
        The covariance matrix of the parameters. May provide either a single
        float, a 1D array with length ``ndim``, or an ``ndim x ndim`` array,
        where ``ndim`` = the number of parameters given. If 2D array is given,
        the off-diagonal terms must be zero. Default is 1 for all parameters.
    \**kwargs :
        All other keyword arguments are passed to
        :py:func:`SSAdaptiveSupport.setup_adaptation`. See that function for
        details.
    """
    name = 'ss_adaptive_discrete'
    symmetric = True

    def __init__(self, parameters, cov=None, successive=None, **kwargs):
        # set the parameters, initialize the covariance matrix
        super(SSAdaptiveNormalDiscrete, self).__init__(parameters, cov=cov,
                                                       successive=successive)
        # set up the adaptation parameters
        self.setup_adaptation(**kwargs)


class SSAdaptiveBoundedDiscrete(SSAdaptiveSupport, BoundedDiscrete):
    r"""A bounded discrete proposoal with adaptive variance, using the
    algorithm from Sivia and Skilling.

    See :py:class:`AdaptiveSupport` for details on the adaptation algorithm.

    Parameters
    ----------
    parameters : (list of) str
        The names of the parameters to produce proposals for.
    boundaries : dict
        Dictionary mapping parameters to boundaries. Boundaries must be a
        tuple or iterable of length two. If floats are provided, the floor
        (ceil) of the lower (upper) bound will be used.
    cov : array, optional
        The covariance matrix of the parameters. May provide either a single
        float, a 1D array with length ``ndim``, or an ``ndim x ndim`` array,
        where ``ndim`` = the number of parameters given. If 2D array is given,
        the off-diagonal terms must be zero. Default is 1 for all parameters.
    \**kwargs :
        All other keyword arguments are passed to
        :py:func:`AdaptiveSupport.setup_adaptation`. See that function for
        details.
    """
    name = 'ss_adaptive_bounded_discrete'
    symmetric = False

    def __init__(self, parameters, boundaries,  cov=None, successive=None,
                 **kwargs):
        # set the parameters, initialize the covariance matrix
        super(SSAdaptiveBoundedDiscrete, self).__init__(
            parameters, boundaries, cov=cov, successive=successive)
        # set up the adaptation parameters
        if 'max_cov' not in kwargs:
            # set the max std to be (1.49*abs(bounds)
            maxwidth = max(map(abs, self.boundaries.values()))
            kwargs['max_cov'] = (1.49*maxwidth)**2
        self.setup_adaptation(**kwargs)


class AdaptiveNormalDiscrete(AdaptiveSupport, NormalDiscrete):
    r"""A discrete proposoal with adaptive variance, using the algorithm from
    Veitch et al.

    See :py:class:`AdaptiveSupport` for details on the adaptation algorithm.

    Parameters
    ----------
    parameters : (list of) str
        The names of the parameters to produce proposals for.
    prior_widths : dict
        Dictionary mapping parameter names to values giving the width of each
        parameter's prior. The values may be floats, or any object that has
        an ``__abs__`` method that will return a float.
    adaptation_duration : int
        The number of iterations over which to apply the adaptation. No more
        adaptation will be done once a chain exceeds this value.
    \**kwargs :
        All other keyword arguments are passed to
        :py:func:`AdaptiveSupport.setup_adaptation`. See that function for
        details.
    """
    name = 'adaptive_discrete'
    symmetric = True

    def __init__(self, parameters, prior_widths, adaptation_duration,
                 successive=None, **kwargs):
        # set the parameters, initialize the covariance matrix
        super(AdaptiveNormalDiscrete, self).__init__(parameters,
                                                     successive=successive)
        # set up the adaptation parameters
        self.setup_adaptation(prior_widths, adaptation_duration, **kwargs)


class AdaptiveBoundedDiscrete(AdaptiveSupport, BoundedDiscrete):
    r"""A bounded discrete proposoal with adaptive variance.

    See :py:class:`AdaptiveSupport` for details on the adaptation algorithm.

    Parameters
    ----------
    parameters : (list of) str
        The names of the parameters to produce proposals for.
    boundaries : dict
        Dictionary mapping parameters to boundaries. Boundaries must be a
        tuple or iterable of length two. If floats are provided, the floor
        (ceil) of the lower (upper) bound will be used.
    adaptation_duration : int
        The number of iterations over which to apply the adaptation. No more
        adaptation will be done once a chain exceeds this value.
    \**kwargs :
        All other keyword arguments are passed to
        :py:func:`AdaptiveSupport.setup_adaptation`. See that function for
        details.
    """
    name = 'adaptive_bounded_discrete'
    symmetric = False

    def __init__(self, parameters, boundaries, adaptation_duration,
                 successive=None, **kwargs):
        # set the parameters, initialize the covariance matrix
        super(AdaptiveBoundedDiscrete, self).__init__(
            parameters, boundaries, successive=successive)
        # set up the adaptation parameters
        self.setup_adaptation(self.boundaries, adaptation_duration, **kwargs)


#
# =============================================================================
#
#                           Helper functions
#
# =============================================================================
#


def _floorceil(x):
    """Returns floor (ceil) of values < (>) 0."""
    return numpy.sign(x)*numpy.ceil(abs(x))


def _ceilfloor(x):
    """Returns the ceil (floor) of values < (>) 0."""
    return numpy.sign(x)*numpy.floor(abs(x))
