# Copyright (C) 2020 Collin Capano, Richard Stiskalek
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

from __future__ import (absolute_import, division)

from abc import ABCMeta
from six import add_metaclass

import numpy
from scipy import stats

from .base import (BaseProposal, BaseAdaptiveSupport)


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
    jump_interval : int, optional
        The jump interval of the proposal, the proposal only gets called every
        jump interval-th time. After ``jump_interval_duration`` number of
        proposal steps elapses the proposal will again be called on every
        chain iteration. By default ``jump_interval`` = 1.
    jump_interval_duration : int, optional
        The number of proposals steps during which values of ``jump_interval``
        other than 1 are used. After this elapses the proposal is called on
        each iteration.

    Attributes
    ----------
    cov : numpy.ndarray
        The covariance matrix being used.
    """

    name = 'normal'
    symmetric = True

    def __init__(self, parameters, cov=None, jump_interval=1,
                 jump_interval_duration=None):
        self.parameters = parameters
        self.ndim = len(self.parameters)
        self.set_jump_interval(jump_interval, jump_interval_duration)
        self._isdiagonal = False
        self._cov = None
        self._std = None
        self.cov = cov

    @property
    def isdiagonal(self):
        """Whether or not the parameters are independent of each other.

        If True (the parameters are independent), then a normal distribution
        will be used for proposals. Otherwise, a multivariate normal is used.
        """
        return self._isdiagonal

    @property
    def cov(self):
        """The covariance matrix used.
        """
        if self.isdiagonal:
            cov = numpy.diag(self._std**2)
        else:
            cov = self._cov
        return cov

    def _ensurearray(self, val, default=1):
        """Boiler-plate function for setting covariance or standard deviation.

        This ensures that the given value is an array with the same length as
        the number of parameters. If ``None`` is passed for the value, the
        default value will be used.
        """
        if val is None:
            val = default
        if not isinstance(val, numpy.ndarray):
            val = numpy.array(val)
        # make sure val is atleast 1D array
        if val.ndim < 1 or val.size == 1:
            val = numpy.repeat(val.item(), len(self.parameters))
        # make sure the dimensionality makes sense
        if ((val.ndim == 1 and val.size != self.ndim) or (
             val.ndim != 1 and val.shape != (self.ndim, self.ndim))):
            raise ValueError("must provide a value for every parameter")
        return val

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
        # make sure we have an array, filling in default as necessary
        cov = self._ensurearray(cov)
        # if cov is a 1D array, means diagonal covariance
        if cov.ndim == 1:
            self._isdiagonal = True
        else:
            # check if off-diagonal terms are all zero, if so, just store
            # the diagonals
            self._isdiagonal = (
                cov[~numpy.eye(cov.shape[0], dtype=bool)] == 0).all()
            if self._isdiagonal:
                cov = cov[numpy.diag_indices(cov.shape[0])]
        if self._isdiagonal:
            # for diagonal, we'll store the std instead
            self._std = cov**0.5
        else:
            self._cov = cov

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

    @std.setter
    def std(self, std):
        """Sets the standard deviation of each parameter.

        Setting this means that all parameters are independent.
        """
        # make sure we have an array, filling in default as necessary
        std = self._ensurearray(std)
        self._isdiagonal = True
        self._std = std

    @property
    def state(self):
        return {'random_state': self.random_state}

    def set_state(self, state):
        self.random_state = state['random_state']

    def _jump(self, fromx):
        # the normal RVS is much faster than the multivariate one, so use it
        # if we can
        if self.isdiagonal:
            newpt = self.random_generator.normal(
                [fromx[p] for p in self.parameters], self._std)
        else:
            newpt = self.random_generator.multivariate_normal(
                [fromx[p] for p in self.parameters], self.cov)
        return dict(zip(self.parameters, newpt))

    def _logpdf(self, xi, givenx):
        mu = [givenx[p] for p in self.parameters]
        xi = [xi[p] for p in self.parameters]
        if self.isdiagonal:
            logp = stats.norm.logpdf(xi, loc=mu, scale=self._std).sum()
        else:
            logp = stats.multivariate_normal.logpdf(xi, mean=mu, cov=self._cov,
                                                    allow_singular=True)
        return logp


#
# =============================================================================
#
#                     Veitch et al. adaptive proposal
#
# =============================================================================
#


@add_metaclass(ABCMeta)
class AdaptiveSupport(BaseAdaptiveSupport):
    r"""Utility class for adding adaptive variance support to a proposal.

    The adaptation algorithm is based on Eqs. 35 and 36 of Veitch et al. [1]_.
    The with of the proposal at each step is based on the width of the prior
    and whether or not the previous proposal was accepted or not. See Notes
    for more details.

    This is only meant to be used in conjuction with proposals that have
    independent parameters, each with their own standard deviation.
    Specifically, the propsal must have an ``isdiagonal`` attribute and an
    ``_std`` attribute.

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
    _prior_widths = None
    _deltas = None

    def setup_adaptation(self, prior_widths, adaptation_duration,
                         adaptation_decay=None, start_step=1,
                         target_rate=0.234, initial_std=None):
        r"""Sets up the adaptation parameters.

        Parameters
        ----------
        prior_widths : dict
            Dictionary mapping parameter names to values giving the width of
            each parameter's prior. The values may be floats, or any object
            that has an ``__abs__`` method that will return a float.
        adaptation_duration : int
            The number of proposal steps over which to apply the adaptation. No
            more adaptation will be done once a chain exceeds this value.
        adaptation_decay : int, optional
            The decay rate to use for the adaptation size (:math:`beta` in the
            equation in the notes). If not provided, will use
            :math:`1/\log_10(T)`, where :math:`T` is the adaptation duration.
        start_step : int, optional
            The proposal step to start doing the adaptation (:math:`k_0+1` in
            the equation below). Must be greater than zero. Default is 1.
        target_rate : float, optional
            The target acceptance rate. Default is 0.234.
        initial_std : array, optional
            The initial standard deviation to use. Default is to use
            `(1 - target_rate)*0.09*prior_widths`.
        """
        # check that a diagonal covariance was provided
        if not self.isdiagonal:
            raise ValueError("Only independent variables are supported "
                             "(all off-diagonal terms in the covariance "
                             "must be zero)")
        # figure out initial variance to use
        self.prior_widths = prior_widths
        self.adaptation_duration = adaptation_duration
        self.start_step = start_step
        if adaptation_decay is None:
            adaptation_decay = 1./numpy.log10(self.adaptation_duration)
        self.adaptation_decay = adaptation_decay
        self.target_rate = target_rate
        if initial_std is None:
            initial_std = (1 - self.target_rate)*0.09*self.deltas
        # set the covariance to the initial
        self._std = initial_std

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

    def _update(self, chain):
        """Updates the adaptation based on whether the last jump was accepted.

        This prepares the proposal for the next jump.
        """
        dk = self.nsteps - self.start_step + 1
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
                'std': self._std,
                'nsteps': self._nsteps}

    def set_state(self, state):
        self.random_state = state['random_state']
        self._std = state['std']
        self._nsteps = state['nsteps']


class AdaptiveNormal(AdaptiveSupport, Normal):
    r"""Uses a normal distribution with adaptive variance for proposals.

    See :py:class:`AdaptiveSupport` for details on the adaptation algorithm.

    Parameters
    ----------
    parameters : (list of) str
        The names of the parameters.
    prior_widths : dict
        Dictionary mapping parameter names to values giving the width of each
        parameter's prior. The values may be floats, or any object that has
        an ``__abs__`` method that will return a float.
    adaptation_duration: int
        The number of proposal steps over which to apply the adaptation. No
        more adaptation will be done once a proposal exceeds this value.
    start_step: int, optional
        The proposal step index when adaptation phase begins.
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
    name = 'adaptive_normal'
    symmetric = True

    def __init__(self, parameters, prior_widths, adaptation_duration,
                 start_step=1, jump_interval=1, **kwargs):
        # set the parameters, initialize the covariance matrix
        super(AdaptiveNormal, self).__init__(
            parameters, jump_interval=jump_interval,
            jump_interval_duration=adaptation_duration)
        # set up the adaptation parameters
        self.setup_adaptation(prior_widths, adaptation_duration,
                              start_step=start_step, **kwargs)


#
# =============================================================================
#
#                   Sivia & Skilling adaptive algorithm
#
# =============================================================================
#


@add_metaclass(ABCMeta)
class SSAdaptiveSupport(BaseAdaptiveSupport):
    r"""Utility class for adding adaptive variance support to a proposal.

    The adaptation algorithm is based on a method in Sivia and Skilling [1]_.

    Notes
    -----
    At each iteration :math:`k`, the variance is scaled by a factor
    :math:`\gamma(k)`

    .. math::

        \gamma(k) = \alpha(k) \gamma(k-1),

    where

    .. math::

        \alpha(k) = \begin{cases}
            e^{1/N_a(k)} &\textrm{ if } \frac{N_a(k)}{k} > \xi,\\
            e^{-1/(k-N_a(k))} &\textrm{ if } \frac{N_a(k)}{k} < \xi,\\
            1 &\textrm{ if } \frac{N_a(k)}{k} = \xi.
            \end{cases}

    Here, :math:`N_a(k)` is the number of iterations that have been accepted to
    this point (so that :math:`k-N_a(k)` is the number of rejections), and
    :math:`\xi` is the target acceptance rate. The initial :math:`gamma` is
    set to 1.

    References
    ----------
    .. [1] Sivia D., Skilling J., "Data Analysis: A Bayesian Tutorial,"
        Oxford Univ. Press, Oxford (2006)
    """
    n_accepted = None
    max_std = None

    def setup_adaptation(self, target_rate=0.234, max_cov=None):
        r"""Sets up the adaptation parameters.

        Parameters
        ----------
        target_rate : float, optional
            The target acceptance rate. Default is 0.234.
        max_cov : float, optional
            The maximum value any element in the covariance matrix is allowed
            to obtain. If an adapatation step would cause any element to exceed
            this value, the covariance matrix will not be changed. Default
            (None) means that no cap will be applied.
        """
        self.target_rate = target_rate
        self.n_accepted = 0
        self.start_step = 1
        if max_cov is not None:
            max_std = max_cov**0.5
        else:
            max_std = numpy.inf
        self.max_std = max_std

    def _update(self, chain):
        """Updates the adaptation based on whether the last jump was accepted.

        This prepares the proposal for the next jump.
        """
        self.n_accepted += int(chain.acceptance[-1]['accepted'])
        n_iter = self.nsteps + 1
        rate = self.n_accepted / n_iter
        if rate > self.target_rate:
            alpha = numpy.exp(1/self.n_accepted)
        elif rate < self.target_rate:
            n_rejected = n_iter - self.n_accepted
            alpha = numpy.exp(-1/n_rejected)
        else:
            alpha = 1.
        if self.isdiagonal:
            alpha = alpha**0.5
            # check that we won't go beyond the max specified
            max_std = alpha * self._std.max()
            if max_std <= self.max_std:
                self._std *= alpha
        else:
            # check that we won't go beyond the max specified
            max_cov = alpha * self._cov.max()
            if max_cov <= self.max_std**2:
                self._cov *= alpha

    @property
    def state(self):
        state = {'random_state': self.random_state,
                 'n_accepted': self.n_accepted,
                 'nsteps': self._nsteps}
        if self.isdiagonal:
            state.update({'std': self._std})
        else:
            state.update({'cov': self._cov})
        return state

    def set_state(self, state):
        self.random_state = state['random_state']
        self.n_accepted = state['n_accepted']
        self._nsteps = state['nsteps']
        if self.isdiagonal:
            self._std = state['std']
        else:
            self._cov = state['cov']


class SSAdaptiveNormal(SSAdaptiveSupport, Normal):
    r"""Uses a normal distribution with adaptive variance for proposals.

    See :py:class:`SSAdaptiveSupport` for details on the adaptation algorithm.

    Parameters
    ----------
    parameters : (list of) str
        The names of the parameters.
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
    name = 'ss_adaptive_normal'
    symmetric = True

    def __init__(self, parameters, cov=None, jump_interval=1,
                 jump_interval_duration=None, **kwargs):
        # set the parameters, initialize the covariance matrix
        super(SSAdaptiveNormal, self).__init__(
            parameters, cov=cov, jump_interval=jump_interval,
            jump_interval_duration=jump_interval_duration)
        # set up the adaptation parameters
        self.setup_adaptation(**kwargs)


#
# =============================================================================
#
#                   Andrieu & Thoms  adaptive algorithm
#
# =============================================================================
#


@add_metaclass(ABCMeta)
class ATAdaptiveSupport(BaseAdaptiveSupport):
    r"""Utility class for adding ATAdaptiveNormal proposal support.

    The adaptation algorithm is based on Algorithm 4 and 6 in [1].

    See [1] for details.

    Notes
    ----------
    For the vanishing decay we use

    .. math::
        \gamma_{g+1} = \left(g - g_{0}\right)^{-0.6} - C,

    where :math: `g_{0}` is the iteration at which adaptation starts,
    by default :math: `g_{0}=1` and :math: `C` is a positive constant
    ensuring that when the adaptation phase ends the vanishing decay tends to
    zero.

    References
    ----------
    [1] Andrieu, Christophe & Thoms, Johannes. (2008).
    A tutorial on adaptive MCMC. Statistics and Computing.
    18. 10.1007/s11222-008-9110-y.
    """
    _target_rate = None
    _start_step = None
    _decay_const = None
    _adaptation_duration = None

    def setup_adaptation(self, adaptation_duration, diagonal=False,
                         componentwise=False, start_step=1, target_rate=None):
        r"""Sets up the adaptation parameters.

        Parameters
        ----------
        adaptation_duration: int
            The number of proposal steps over which to apply the adaptation. No
            more adaptation will be done once a proposal exceeds this value.
        diagonal : bool, optional
            Whether to train off-diagonal elements of the proposal covariance.
            By default set to False; off-diagonal elements are being trained.
        componentwise : bool, optional
            Whether to include a componentwise scaling of the parameters
            (algorithm 6 in [1]). By default set to False (algorithm 4 in [1]).
            Componentwise scaling `ndim` times more expensive than global
            scaling.
        start_step : int, optional
            The proposal step when the adaptation phase begins.
        target_rate: float, optional
            Target acceptance ratio. By default 0.234 and 0.48 for
            componentwise scaling.
        """
        self.start_step = start_step
        self.adaptation_duration = adaptation_duration
        self._iscomponentwise = componentwise
        self._isdiagonal = diagonal
        self._decay_const = (adaptation_duration)**(-0.6)

        self._mean = numpy.zeros(self.ndim)  # initial mean
        if self.isdiagonal:
            self._unit_cov = numpy.ones(self.ndim)
            self._std = self._unit_cov**0.5
        else:
            self._unit_cov = numpy.eye(self.ndim)  # inital covariance
            self._cov = self._unit_cov
        if not self._iscomponentwise:
            self._log_lambda = 0
        else:
            self._log_lambda = numpy.zeros(self.ndim)
        # set target rate (componentwise target rate scales differently)
        if target_rate is None:
            if not self._iscomponentwise:
                self.target_rate = 0.234
            else:
                self.target_rate = 0.48
        else:
            self.target_rate = target_rate

    def _componentwise_scaling(self, chain, dk):
        """Componentwise scaling. Does virtual moves along each axis
        one at a time."""
        current_logl = chain.current_stats['logl']
        current_logp = chain.current_stats['logp']
        current_position = chain.current_position
        proposed_position = chain.proposed_position
        dlog_lambda = numpy.zeros_like(self._log_lambda)

        # Make a virtual jump along every componentwise axis
        for i, p in enumerate(self.parameters):
            virtual_move = current_position.copy()
            virtual_move[p] = proposed_position[p]
            # evaluate this move
            r = chain.model(**virtual_move)
            if chain._hasblobs:
                logl, logp, __ = r
            else:
                logl, logp = r
            # if proposed translation goes out of bounds force eject
            if logp == -numpy.inf:
                ar = 0.
            else:
                __, ar = chain._acceptance_ratio(logp, logl, virtual_move,
                                                 current_logp, current_logl,
                                                 current_position)
            # update the component
            dlog_lambda[i] = dk * (ar - self.target_rate)
        return dlog_lambda

    def _update(self, chain):
        """Updates the adaptation based on whether the last jump was accepted.
        This prepares the proposal for the next jump.
        """
        dk = self.nsteps - self.start_step + 1
        if 1 < dk < self.adaptation_duration:
            dk = dk**(-0.6) - self._decay_const
            newpt = numpy.array([chain.current_position[p]
                                 for p in self.parameters])
            # Update of the global scaling
            if not self._iscomponentwise:
                ar = chain.acceptance['acceptance_ratio'][-1]
                self._log_lambda += dk * (ar - self.target_rate)
            else:
                self._log_lambda += self._componentwise_scaling(chain, dk)
            # Update the first moment
            df = newpt - self._mean
            self._mean += dk * df
            # Update the second moment
            if self.isdiagonal:
                self._unit_cov += dk * (df**2 - self._unit_cov)
                self._std = numpy.sqrt(numpy.exp(self._log_lambda)
                                       * self._unit_cov)
            else:
                df = df.reshape(-1, 1)
                self._unit_cov += dk * (numpy.matmul(df, df.T)
                                        - self._unit_cov)
                if not self._iscomponentwise:
                    self._cov = numpy.exp(self._log_lambda) * self._unit_cov
                else:
                    Lambda = numpy.diag(numpy.exp(self._log_lambda))**0.5
                    self._cov = numpy.matmul(
                        numpy.matmul(Lambda, self._unit_cov), Lambda)

    @property
    def state(self):
        state = {'random_state': self.random_state,
                 'mean': self._mean,
                 'log_lambda': self._log_lambda,
                 'unit_cov': self._unit_cov,
                 'nsteps': self._nsteps}
        if self.isdiagonal:
            state.update({'std': self._std})
        else:
            state.update({'cov': self._cov})
        return state

    def set_state(self, state):
        self.random_state = state['random_state']
        self._mean = state['mean']
        self._log_lambda = state['log_lambda']
        self._unit_cov = state['unit_cov']
        self._nsteps = state['nsteps']
        if self.isdiagonal:
            self._std = state['std']
        else:
            self._cov = state['cov']


class ATAdaptiveNormal(ATAdaptiveSupport, Normal):
    r"""Uses a normal distribution with adaptive covariance for proposals.

    See :py:class:`ATAdaptiveSupport` for details on the adaptation algorithm.

    Parameters
    ----------
    parameters: (list of) str
        The names of the parameters.
    adaptation_duration : int
        The number of proposal steps over which to apply the adaptation. No
        more adaptation will be done once a proposal exceeds this value.
    diagonal : bool, optional
        Whether to train off-diagonal elements of the proposal covariance.
        By default set to False; off-diagonal elements are being trained.
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
    \**kwargs:
        All other keyword arguments are passed to
        :py:func:`AdaptiveSupport.setup_adaptation`. See that function for
        details.
    """
    name = 'at_adaptive_normal'
    symmetric = True

    def __init__(self, parameters, adaptation_duration, diagonal=False,
                 componentwise=False, start_step=1, target_rate=None,
                 jump_interval=1):
        # set the parameters, initialize the covariance matrix
        super(ATAdaptiveNormal, self).__init__(
            parameters, jump_interval=jump_interval,
            jump_interval_duration=adaptation_duration)
        # set up the adaptation parameters
        self.setup_adaptation(adaptation_duration=adaptation_duration,
                              diagonal=diagonal, componentwise=componentwise,
                              start_step=start_step, target_rate=target_rate)
