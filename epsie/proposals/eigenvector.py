# Copyright (C) 2020 Richard Stiskalek, Collin Capano
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

import numpy
from scipy.stats import norm

from .base import (BaseProposal, BaseAdaptiveSupport)


class Eigenvector(BaseProposal):
    """Uses a eigenvector jump with a fixed scale that is determined by the
    given covariance matrix.

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
    shuffle_rate : float, optional
        Probability of shuffling the eigenvector jump probabilities. By
        default 0.33.
    minfac : float, optional
        If any eigenvalues are zero, they will be replaced with
        ``minfac`` times the smallest non-zero eigenvalue. Setting a non-zero
        value will force the sampler to at least occasionally jump in the
        zero direction(s). Default is 0.
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

    name = 'eigenvector'
    symmetric = True
    _shuffle_rate = None

    def __init__(self, parameters, cov=None, shuffle_rate=0.33, minfac=0.,
                 jump_interval=1, jump_interval_duration=None):
        self.parameters = parameters
        self.ndim = len(self.parameters)
        self.shuffle_rate = shuffle_rate
        self.set_jump_interval(jump_interval, jump_interval_duration)
        # set up needed parameters
        self._cov = None
        self._eigvals = None
        self._eigvects = None
        self._ind = None  # for caching the eigenvector index
        self._dx = None  # for caching the last jump std
        self._zeroeigvals = numpy.array([])
        self._minfac = minfac
        # calculate and store the eigenvectors/values
        self.cov = cov
        self.eigvals, self.eigvects = numpy.linalg.eigh(self.cov)
        # used for picking which direction to hop along
        self._dims = numpy.arange(self.ndim)

    @BaseProposal.bit_generator.setter
    def bit_generator(self, bit_generator):
        """Sets the random bit generator.

        Also sets the bit generator of the ``initial_proposal``.

        See :py:class:`epsie.proposals.base.BaseProposal` for more details.
        """
        # this borrowed from: https://stackoverflow.com/a/31909212
        BaseProposal.bit_generator.fset(self, bit_generator)

    @property
    def cov(self):
        """The covariance matrix used."""
        return self._cov

    def _ensurearray(self, val, default=1):
        """Boiler-plate function for setting covariance.

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
            val = numpy.eye(len(self.parameters)) * val.item()
        # make sure the dimensionality makes sense
        if val.shape != (self.ndim, self.ndim):
            raise ValueError("must provide a value for every parameter")
        return val

    @cov.setter
    def cov(self, cov):
        """Sets the covariance matrix. Checks that it is symmetric.

        Raises a ``ValueError`` if the dimensionality of the given array
        isn't ndim x ndim.
        """
        # make sure we have an array, filling in default as necessary
        cov = self._ensurearray(cov)
        if not (cov == cov.T).all():
            raise ValueError("must provide a symmetric covariance matrix")
        self._cov = cov

    @property
    def eigvals(self):
        """Returns the eigenvalues"""
        return self._eigvals

    @eigvals.setter
    def eigvals(self, eigvals):
        if eigvals.shape != (self.ndim,):
            raise ValueError("Invalid eigenvalue shape")
        # check that we have no negatives; if we do, and the value is
        # sufficiently close to 0, set it to 0
        negvals = eigvals < 0
        if negvals.any():
            vals = eigvals[negvals]
            relsize = vals / eigvals.max()
            if (abs(relsize) < 1e-12).all():
                # replace with 0
                eigvals[negvals] = 0.
            else:
                # one or more values are < 0 beyond the tolerance; raise error
                raise ValueError("one or more of the given eigenvalues ({}) "
                                 "are negative".format(eigvals))
        self._eigvals = eigvals
        # store and replace any zeroes
        zerovals = eigvals == 0.
        self._zeroeigvals = zerovals
        self._replacezeroeig()

    def _replacezeroeig(self):
        if self._zeroeigvals.any():
            # replace with the min
            minnz = self._eigvals[~self._zeroeigvals].min()
            self._eigvals[self._zeroeigvals] = self.minfac * minnz 

    @property
    def minfac(self):
        """Factor of the smallest non-zero eigenvalue to use for zero values.
        """
        return self._minfac

    @minfac.setter
    def minfac(self, minfac):
        self._minfac = minfac
        # update the zeroeigen values
        self._replacezeroeig()

    @property
    def eigvects(self):
        """Returns the eigenvectors"""
        return self._eigvects

    @eigvects.setter
    def eigvects(self, eigvects):
        if eigvects.shape != (self.ndim, self.ndim):
            raise ValueError("Invalid eigenvector shape.")
        self._eigvects = eigvects

    @property
    def shuffle_rate(self):
        return self._shuffle_rate

    @shuffle_rate.setter
    def shuffle_rate(self, rate):
        if not 0.0 < rate < 1.0:
            raise ValueError("Shuffle rate  must be in range (0, 1).")
        self._shuffle_rate = rate

    @property
    def state(self):
        return {'nsteps': self._nsteps,
                'random_state': self.random_state,
                'cov': self._cov,
                'ind': self._ind}

    def set_state(self, state):
        self.random_state = state['random_state']
        self._nsteps = state['nsteps']
        self._cov = state['cov']
        if self._cov is not None:
            self.eigvals, self.eigvects = numpy.linalg.eigh(self._cov)
        self._ind = state['ind']

    @property
    def _jump_eigenvector(self):
        """Picks along which eigenvector to jump."""
        # replace any zeros with the minimum
        probs = self.eigvals / self.eigvals.sum()
        dims = self._dims
        # with shuffle_rate probability randomly shuffle the probabilities
        if self.random_generator.uniform() < self.shuffle_rate:
            # make sure we don't shuffle any 0 probabilities
            isz = probs == 0.
            if isz.any():
                mask = ~isz
                probs = probs[mask]
                dims = dims[mask]
            self.random_generator.shuffle(probs)
        return self.random_generator.choice(dims, p=probs)

    def _jump(self, fromx):
        self._ind = self._jump_eigenvector
        # scale of the 1D jump
        self._dx = self.random_generator.normal(scale=self.eigvals[self._ind])
        return {p: fromx[p] + self._dx * self.eigvects[i, self._ind]
                for i, p in enumerate(self.parameters)}

    def _logpdf(self, xi, givenx):
        return norm.logpdf(self._dx, loc=0, scale=self.eigvals[self._ind])

    def _update(self, chain):
        pass


class AdaptiveEigenvectorSupport(BaseAdaptiveSupport):
    r"""Utility class for adding AdaptiveEigenvector proposal support.

    The adaptation algorithm is based on Algorithm 8 in [1].

    See [1] for details.

    Notes
    -----
    For the vanishing decay we use

    .. math::
        \gamma_{g+1} = \left(g - g_{0}\right)^{-0.6} - C,

    where :math:`g_{0}` is the iteration at which adaptation starts,
    by default :math:`g_{0}=1` and :math:`C` is a positive constant
    ensuring that when the adaptation phase ends the vanishing decay tends to
    zero. By default assumes that the adaptation phase never ends (only
    decays with time)

    References
    ----------
    [1] Andrieu, Christophe & Thoms, Johannes. (2008).
    A tutorial on adaptive MCMC. Statistics and Computing.
    18. 10.1007/s11222-008-9110-y.
    """

    def setup_adaptation(self, adaptation_duration, start_step=1,
                         target_rate=0.234):
        r"""Sets up the adaptation parameters.

        Parameters
        ----------
        adaptation_duration : int
            The number of adaptation steps.
        start_step : int, optional
            The proposal step when the adaptation phase begins.
        target_rate : float, optional
            Target acceptance rate. By default 0.234
        """
        self.target_rate = target_rate
        self.adaptation_duration = adaptation_duration
        self._decay_const = adaptation_duration**(-0.6)
        self.start_step = start_step
        self._log_lambda = 0.0
        # initialize mu to be zero
        self._mu = numpy.zeros(self.ndim)
        self._unique_steps = 1
        self._lastx = None
        self._lastind = None

    def recursive_covariance(self, chain):
        """Recursively updates the covariance given the latest observation.
        Weights all sampled points uniformly.
        """
        x = numpy.array([chain.current_position[p]
                         for p in self.parameters])
        # only update if we have a new point and the last ind was different
        if (x != self._lastx).any() and self._lastind != self._ind:
            self._unique_steps += 1
            N = self._unique_steps
            dx = (x - self._mu).reshape(-1, 1)
            self._cov = (N - 1) / N \
                * (self._cov + N / (N**2 - 1) * numpy.matmul(dx, dx.T))
            self._mu = (N * self._mu + x) / (N + 1)
            self._lastx = x
            self._lastind = self._ind
            # update the minfac
            self.minfac = (1./N)**0.5

    def _update(self, chain):
        """Updates the adaptation based on whether the last jump was accepted.

        This prepares the proposal for the next jump.
        """
        dk = self.nsteps - self.start_step + 1
        if 1 < dk < self.adaptation_duration:
            # recursively update the covariance
            self.recursive_covariance(chain)
            # update eigenvalues and eigenvectors
            self.eigvals, self.eigvects = numpy.linalg.eigh(self._cov)
            # update the scaling factor
            dk = dk**(-0.6) - self._decay_const
            # Update of the global scaling
            ar = chain.acceptance['acceptance_ratio'][-1]
            self._log_lambda += dk * (ar - self.target_rate)
            # Rescale the eigenvalues
            self.eigvals *= numpy.exp(self._log_lambda)

    @property
    def state(self):
        return {'nsteps': self._nsteps,
                'random_state': self.random_state,
                'mu': self._mu,
                'cov': self._cov,
                'ind': self._ind,
                'unique_steps': self._unique_steps,
                'log_lambda': self._log_lambda}

    def set_state(self, state):
        self._nsteps = state['nsteps']
        self.random_state = state['random_state']
        self._mu = state['mu']
        self._cov = state['cov']
        self._log_lambda = state['log_lambda']
        self._unique_steps = state['unique_steps']
        if self._cov is not None:
            self.eigvals, self.eigvects = numpy.linalg.eigh(self._cov)
            self.eigvals *= numpy.exp(self._log_lambda)
        self._ind = state['ind']


class AdaptiveEigenvector(AdaptiveEigenvectorSupport, Eigenvector):
    r""" Uses jumps along eigenvectors with adaptive scales.

    See :py:class:`AdaptiveEigenvectorSupport` for details on the adaptation
    algorithm.

    Parameters
    ----------
    parameters: (list of) str
        The names of the parameters.
    adaptation_duration: int
        The number of steps after which adaptation of the eigenvectors ends.
        Post-adaptation phase the eigenvectors and eigenvalues are kept
        constant.
    cov0 : array, optional
        The initial covariance matrix of the parameters used to calculate the
        initial eigenvectors. May provide either a single float, a 1D array
        with length ``ndim``, or an ``ndim x ndim`` array,
        where ``ndim`` = the number of parameters given. If a single float or
        a 1D array is given, will use a diagonal covariance matrix (i.e., all
        parameters are independent of each other). Default (None) is to use
        unit variance for all parameters.
    target_rate: float (optional)
        Target acceptance ratio. By default 0.234
    shuffle_rate : float, optional
        Probability of shuffling the eigenvector jump probabilities. By
        default 0.33.
    start_step: int, optional
        The proposal step index when adaptation phase begins.
    jump_interval : int, optional
        The jump interval of the proposal, the proposal only gets called every
        jump interval-th time. After ``adaptation_duration`` number of
        proposal steps elapses the proposal will again be called on every
        chain iteration. By default ``jump_interval`` = 1.
    """
    name = 'adaptive_eigenvector'
    symmetric = True

    def __init__(self, parameters, adaptation_duration, cov0=None,
                 target_rate=0.234, shuffle_rate=0.33, minfac=0., start_step=1,
                 jump_interval=1):
        # set the parameters, initialize the covariance matrix
        super().__init__(
            parameters=parameters, cov=cov0, shuffle_rate=shuffle_rate,
            minfac=minfac,
            jump_interval=jump_interval,
            jump_interval_duration=adaptation_duration)
        # set up the adaptation parameters
        self.setup_adaptation(adaptation_duration, start_step, target_rate)
