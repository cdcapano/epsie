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

from __future__ import absolute_import

import numpy
from scipy import stats

from .base import BaseProposal


class Normal(BaseProposal):
    """Uses a normal distribution with a fixed variance for proposals.

    This proposal may handle one or more parameters.

    Parameters
    ----------
    parameters : (list of) str
        The names of the parameters to produce proposals for.
    cov : array, optional
        The covariance matrix of the parameters. May provide either a single
        float, a 1D array with length ``ndim``, or an ``ndim x ndim`` array,
        where ``ndim`` = the number of parameters given. If a single float or
        a 1D array is given, will use a diagonal covariance matrix (i.e., all
        parameters are independent of each other). Default (None) is to use
        unit variance for all parameters.

    Attributes
    ----------
    cov : numpy.ndarray
        The covariance matrix being used.
    """

    name = 'normal'
    symmetric = True

    def __init__(self, parameters, cov=None):
        self.parameters = parameters
        self.ndim = len(parameters)
        self.isdiagonal = False
        self._cov = None
        self._std = None
        self.cov = cov

    @property
    def cov(self):
        """The covariance matrix used.
        """
        if self.isdiagonal:
            cov = numpy.diag(self._std**2)
        else:
            cov = self._cov
        return cov

    @property
    def std(self):
        """The standard deviation.

        Raises a ``ValueError`` if the covariance matrix is not diagonal.
        """
        if not self.isdiagonal:
            raise ValueError("standard deviation is undefined for "
                             "multivariate normal distribution with a "
                             "non-diagonal covariance matrix")
        return self._std

    @cov.setter
    def cov(self, cov):
        """Sets the covariance matrix and the isdiagonal attribute.

        If a diagonal covariance, the standard deviation (`std`) attribute is
        also set.

        If a single float or a 1D array is given, will use a diagonal
        covariance matrix (i.e., all parameters are independent of each other).
        Default (None) is to use unit variance for all parameters.

        Raises a ``ValueError`` if the dimensionality of the given array
        isn't ndim x ndim.
        """
        if cov is None:
            cov = 1.
        if not isinstance(cov, numpy.ndarray):
            cov = numpy.array(cov)
        # make sure cov is atleast 1D array
        if cov.ndim < 1 or cov.size == 1:
            cov = numpy.repeat(cov.item(), len(self.parameters))
        # if its a 1D array, means diagonal covariance
        if cov.ndim == 1:
            self.isdiagonal = True
        else:
            # check that dimensionality makes sense
            if cov.shape != (self.ndim, self.ndim):
                raise ValueError("dimension of covariance matrix does not "
                                 "match given number of parameters")
            # check if off-diagonal terms are all zero, if so, just store
            # the diagonals
            self.isdiagonal = (
                cov[~numpy.eye(cov.shape[0], dtype=bool)] == 0).all()
            if self.isdiagonal:
                cov = cov[numpy.diag_indices(cov.shape[0])]
        if self.isdiagonal:
            # for diagonal, we'll store the std instead
            self._std = cov**0.5
        else:
            self._cov = cov

    @property
    def state(self):
        return {'random_state': self.random_state}

    def set_state(self, state):
        self.random_state = state['random_state']

    def jump(self, fromx):
        # the normal RVS is much faster than the multivariate one, so use it
        # if we can
        if self.isdiagonal:
            mu = [fromx[p] for p in self.parameters]
            newpt = self.random_generator.normal(mu, self._std)
        else:
            newpt = self.random_generator.multivariate_normal(
                [fromx[p] for p in self.parameters], self.cov)
        return dict(zip(self.parameters, newpt))

    def logpdf(self, xi, givenx):
        means = [givenx[p] for p in self.parameters]
        xi = [xi[p] for p in self.parameters]
        if self.isdiagonal:
            logp = stats.normal.logpdf(xi, loc=means, scale=self._std)
        else:
            logp = stats.multivariate_normal(xi, mean=means, cov=self._cov)
        return logp


class AdaptiveNormal(Normal):
    r"""Uses a Gaussian distribution with adaptive variance.

    The adaptation algorithm is based on Eqs. 35 and 36 of [1]_.
    The size of the variance at each step is based on the width of the prior
    and whether or not the previous proposal was accepted or not. See Notes
    for more details.

    Multiple parameters may be given. However, currently, only independent
    variables are supported (i.e., the covariance matrix must be diagonal).

    Parameters
    ----------
    parameters : (list of) str
        The names of the parameters.
    prior_widths : dict
        Dictionary mapping parameter names to values giving the width of each
        parameter's prior. The values may be floats, or any object that has
        an ``__abs__`` method that will return a float.
    adaptation_duration : int
        The number of iterations over which to apply the adaptation. No more
        adaptation will be done once a chain exceeds this value.
    adaptation_decay : int, optional
        The decay rate to use for the adaptation size (:math:`beta` in the
        equation below). If not provided, will use :math:`1/\log_10(T)`, where
        :math:`T` is the adaptation duration.
    start_iteration : int, optional
        The iteration to start doing the adaptation (:math:`k_0+1` in the
        equation below). Must be greater than zero. Default is 1.
    target_rate : float, optional
        The target acceptance rate. Default is 0.234.
    initial_std : array, optional
        The initial standard deviation to use. Default is to use
        `(1 - target_rate)*0.09*prior_widths`.

    Notes
    -----
    For a given parameter, the standard deviation of the :math:`k`th iteration
    is given by [1]_:

    .. math::

        \sigma_k = \sigma_{k-1} + \alpha_{k-1}
            \left[\left(k - k_0\right)^{-\beta} - 0.1\right]\frac{\Delta}{10},

    where :math:`\alpha_{k-1} = 1 - \xi` if the previous iteration was
    accpeted and :math:`\alpha_{k-1} = -\xi` if the previous iteration was
    rejected. Here, :math:`\xi` is the target acceptance rate, :math:`\Delta`
    is the prior width, :math:`\beta` is the adaptation decay, and :math:`k_0`
    gives the iteration after which the adaptation begins. The initial standard
    deviation :math:`\sigma_0` to use is a free parameter. The default in this
    function is to use :math:`\sigma_0 = (1-\xi)0.09\Delta`.


    References
    ----------
    .. [1] J. Veitch et al., "Parameter estimation for compact binaries with
        ground-based gravitational-wave observations using the LALInference
        software library, " Phys. Rev. D91 042003 (2015),
        arXiv:1409.7215 [gr-qc].
    """
    name = 'adaptive_normal'
    symmetric = True

    def __init__(self, parameters, prior_widths, adaptation_duration,
                 adaptation_decay=None, start_iteration=1, target_rate=0.234,
                 initial_std=None):
        # set the parameters, initialize the covariance matrix
        super(AdaptiveNormal, self).__init__(parameters)
        # check that a diagonal covariance was provided
        if not self.isdiagonal:
            raise ValueError("Only independent variables are supported "
                             "(all off-diagonal terms in the covariance "
                             "must be zero)")
        # figure out initial variance to use
        self._deltas = None
        self.prior_widths = prior_widths
        self._adaptation_duration = None
        self.adaptation_duration = adaptation_duration
        if adaptation_decay is None:
            adaptation_decay = 1./numpy.log10(self.adaptation_duration)
        self.adaptation_decay = adaptation_decay
        self._start_iteration = None
        self.start_iteration = start_iteration
        self.target_rate = target_rate
        if initial_std is None:
            initial_std = (1 - self.target_rate)*0.09*self.deltas
        # set the covariance to the initial
        self.cov = initial_std**2

    @property
    def prior_widths(self):
        """The width of the prior used for each parameter."""
        return self._prior_widths

    @prior_widths.setter
    def prior_widths(self, prior_widths):
        """Sets the prior widths, making sure that widths are provided for
        each parameter in ``parameters``.

        Also sets the deltas attribute.
        """
        try:
            self._prior_widths = {p: abs(prior_widths[p])
                                  for p in self.parameters}
        except KeyError:
            raise ValueError("must provide prior widths for every parameter")
        self._deltas = numpy.array([self.prior_widths[p]
                                    for p in self.parameters])

    @property
    def deltas(self):
        """The prior widths, as a numpy array."""
        return self._deltas

    @property
    def start_iteration(self):
        """The iteration that the adaption begins."""
        return self._start_iteration

    @start_iteration.setter
    def start_iteration(self, start_iteration):
        """Sets the start iteration, making sure it is >= 1."""
        if start_iteration < 1:
            raise ValueError("start_iteration must be >= 1")
        self._start_iteration = start_iteration

    @property
    def adaptation_duration(self):
        """The adaptation duration used."""
        return self._adaptation_duration

    @adaptation_duration.setter
    def adaptation_duration(self, adaptation_duration):
        """Sets the adaptation duration to the given value, making sure it is
        larger than 1.
        """
        if adaptation_duration < 1:
            raise ValueError("adaptation duration must be >= 1")
        self._adaptation_duration = adaptation_duration

    def update(self, chain):
        """Updates the adaptation based on whether the last jump was accepted.

        This prepares the proposal for the next jump.
        """
        # subtact 1 from the start iteration, since the update happens after
        # the jump
        dk = chain.iteration - (self.start_iteration - 1)
        if 1 <= dk < self.adaptation_duration:
            dk = dk**(-self.adaptation_decay) - 0.1
            if chain.acceptance[-1]['accepted']:
                alpha = 1 - self.target_rate
            else:
                alpha = -self.target_rate
            dsigmas = alpha * dk * self.deltas/10.
            # ensure we don't go negative
            sigmas = self._std
            newsigmas = sigmas + dsigmas
            lzidx = newsigmas < 0
            newsigmas[lzidx] = sigmas[lzidx]
            self._std = newsigmas

    @property
    def state(self):
        return {'random_state': self.random_state,
                'std': self._std}

    def set_state(self, state):
        self.random_state = state['random_state']
        self._std = state['std']
