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

from __future__ import (absolute_import, division)

from abc import ABCMeta
from six import add_metaclass

import numpy
from scipy.stats import norm

from .base import (BaseProposal, BaseAdaptiveSupport)
from .normal import (Normal, ATAdaptiveNormal)
from .bounded_normal import (BoundedNormal, ATAdaptiveBoundedNormal)


class Eigenvector(BaseProposal):
    """Uses a eigenvector jump with a fixed scale.

    This proposal may handle one or more parameters.

    Parameters
    ----------
    parameters : (list of) str
        The names of the parameters to produce proposals for.
    stability_duration : int
        Number of initial steps done with a initial proposal specified by name
        in ``initial_proposal''. After this eigenvalues and eigenvectors are
        evaluated (and never again) and jumps proposed along those.
    initial_proposal : str (optional)
        Name of the initial proposal that is called before the number of
        proposal seps exceeds ``stability_duration''. By default se to the
        'epsie.proposals.ATAdaptiveProposal'. Supported options
        include: 'normal', 'at_adaptive_proposal'
    shuffle_rate : float (optional)
        Probability of shuffling the eigenvector jump probabilities. By
        default 0.33.
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
    _cov = None
    _mu = None
    _eigvals = None
    _eigvects = None
    _initial_proposal = None
    _stability_duration = None

    def __init__(self, parameters, stability_duration, shuffle_rate=0.33,
                 jump_interval=1, jump_interval_duration=None,
                 initial_proposal='at_adaptive_normal'):
        self.parameters = parameters
        self.ndim = len(self.parameters)
        self.shuffle_rate = shuffle_rate
        self.start_step = 1  # Add support for this
        # store the jump interval information
        self.set_jump_interval(jump_interval, jump_interval_duration)
        self.set_initial_proposal(initial_proposal, stability_duration)
        # cache the eigenvector index used to produce a jump
        self._ind = None
        # cache the last jump scale
        self._dx = None

    @property
    def initial_proposal(self):
        """Initial proposal used during the stabilit phase, before estimated
        eigenvectors become reliable."""
        return self._initial_proposal

    @property
    def stability_duration(self):
        """Number of proposal steps during which ``self.initial_proposal'' is
        used instead of eigenvector jumps"""
        return self._stability_duration

    def set_initial_proposal(self, proposal_name, duration, boundaries=None):
        """Sets up the initial proposal used before able to get stable
        eigenvector estimates."""
        if proposal_name == 'normal':
            self._initial_proposal = Normal(self.parameters)
        elif proposal_name == 'at_adaptive_normal':
            self._initial_proposal = ATAdaptiveNormal(
                self.parameters, adaptation_duration=duration)
        elif proposal_name == 'bounded_normal':
            self._initial_proposal = BoundedNormal(self.parameters, boundaries)
        elif proposal_name == 'at_adaptive_bounded_normal':
            self._initial_proposal = ATAdaptiveBoundedNormal(
                self.parameters, boundaries, adaptation_duration=duration)
        else:
            raise ValueError("Proposal '{}' not implemented for eigenvector"
                             .format(proposal_name))
        self._initial_proposal.bit_generator = self.bit_generator
        self._stability_duration = int(duration)

    @property
    def eigvals(self):
        """Returns the eigenvalues"""
        return self._eigvals

    @eigvals.setter
    def eigvals(self, eigvals):
        if eigvals.shape != (self.ndim,):
            raise ValueError("Invalid eigenvalue shape")
        self._eigvals = eigvals

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
        state = {'nsteps': self._nsteps,
                 'random_state': self.random_state,
                 'mu': self._mu,
                 'cov': self._cov}
        return state

    def set_state(self, state):
        self.random_state = state['random_state']
        self._nsteps = state['nsteps']
        self._mu = state['mu']
        self._cov = state['cov']
        if self._cov is not None:
            self._eigvals, self._eigvects = numpy.linalg.eigh(self._cov)

    @property
    def _call_initial_proposal(self):
        """Decides whether to call the initial proposal."""
        if self.nsteps <= self.stability_duration:
            return True
        return False

    @property
    def _jump_eigenvector(self):
        """Picks along which eigenvector to jump."""
        probs = self.eigvals / numpy.sum(self.eigvals)
        # with shuffle_rate probability randomly shuffle the probabilities
        if self.random_generator.uniform() < self.shuffle_rate:
            self.random_generator.shuffle(probs)
        return self.random_generator.choice(self.ndim, p=probs)

    def _jump(self, fromx):
        if self._call_initial_proposal:
            return self.initial_proposal.jump(fromx)
        self._ind = self._jump_eigenvector
        # scale of the 1D jump
        self._dx = self.random_generator.normal(scale=self.eigvals[self._ind])
        return {p: fromx[p] + self._dx * self.eigvects[i, self._ind]
                for i, p in enumerate(self.parameters)}

    def _logpdf(self, xi, givenx):
        if self._call_initial_proposal:
            return self.initial_proposal.logpdf(xi, givenx)
        return norm.logpdf(self._dx, loc=0, scale=self.eigvals[self._ind])

    def _recursive_mean_cov(self, chain):
        """Recursive updates of the mean and covariance given the new data"""
        n = self.nsteps - self.stability_duration // 2
        newpt = numpy.array([chain.current_position[p]
                             for p in self.parameters])
        df = newpt - self._mu
        self._mu += df / n
        df = df.reshape(-1, 1)
        self._cov += 1 / n * numpy.matmul(df, df.T) - 1 / (n - 1) * self._cov

    def _stability_update(self, chain):
        """Updates the covariance matrix during the stabilisation period"""
        if self.nsteps <= self.stability_duration // 2 + 5:
            self.initial_proposal.update(chain)
        elif self.nsteps < self.stability_duration:
            self.initial_proposal.update(chain)
            # estimate covariance and mean and start recursively updating
            if self._mu is None and self._cov is None:
                X = numpy.array([chain.positions[p][
                    self.stability_duration//2:] for p in self.parameters]).T
                # calculate mean and cov
                self._mu = numpy.mean(X, axis=0)
                self._cov = numpy.cov(X, rowvar=False)
                if self.ndim == 1:
                    self._cov = self._cov.reshape(1, 1)
            else:
                self._recursive_mean_cov(chain)

        if self.nsteps == self.stability_duration:
            self.eigvals, self.eigvects = numpy.linalg.eigh(self._cov)

    def _update(self, chain):
        if self._call_initial_proposal:
            self._stability_update(chain)


@add_metaclass(ABCMeta)
class AdaptiveEigenvectorSupport(BaseAdaptiveSupport):
    r"""Utility class for adding AdaptiveEigenvector proposal support.

    The adaptation algorithm is based on Algorithm 8 in [1].

    See [1] for details.

    Notes
    ----------
    For the vanishing decay we use

    .. math::
        \gamma_{g+1} = \left(g - g_{0}\right)^{-0.6} - C,

    where :math: `g_{0}` is the iteration at which adaptation starts,
    by default :math: `g_{0}=1` and :math: `C` is a positive constant
    ensuring that when the adaptation phase ends the vanishing decay tends to
    zero. By default assumes that the adaptation phase never ends (only
    decays with time)

    References
    ----------
    [1] Andrieu, Christophe & Thoms, Johannes. (2008).
    A tutorial on adaptive MCMC. Statistics and Computing.
    18. 10.1007/s11222-008-9110-y.
    """

    _decay_const = None

    def setup_adaptation(self, adaptation_duration, target_rate=0.234):
        r"""Sets up the adaptation parameters.
        adaptation_duration : int
            The number of adaptation steps.
        target_rate : float, optional
            Target acceptance rate. By default 0.234
        """
        self.target_rate = target_rate
        self.adaptation_duration = adaptation_duration
        self._decay_const = adaptation_duration**(-0.6)
        self._log_lambda = 0.0

    def _update(self, chain):
        """Updates the adaptation based on whether the last jump was accepted.
        This prepares the proposal for the next jump.
        """
        dk = self.nsteps - self.stability_duration + 1
        if self._call_initial_proposal:
            self._stability_update(chain)
        elif dk < self.adaptation_duration:
            self._recursive_mean_cov(chain)
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
                'log_lambda': self._log_lambda}

    def set_state(self, state):
        self.random_state = state['random_state']
        self._nsteps = state['nsteps']
        self._mu = state['mu']
        self._cov = state['cov']
        self._log_lambda = state['log_lambda']
        if self._cov is not None:
            self.eigvals, self.eigvects = numpy.linalg.eigh(self._cov)
            self.eigvals *= numpy.exp(self._log_lambda)


class AdaptiveEigenvector(AdaptiveEigenvectorSupport, Eigenvector):
    r""" Uses jumps along eigenvectors with adaptive scales

    See :py:class:`AdaptiveEigenvectorSupport` for details on the adaptation
    algorithm.

    Parameters
    ----------
    parameters: (list of) str
        The names of the parameters.
    stability_duration : int
        Number of initial steps done with a initial proposal specified by name
        in ``initial_proposal''. After this eigenvalues and eigenvectors are
        evaluated and jumps proposed along those.
    adaptation_duration: int
        The number of steps after which adaptation of the eigenvectors ends.
        This is defined such that while the number of proposal steps :math:`N`
        satisfies :math:`N <= \mathrm{stability_duration}` the
        ``initial_proposal'' is called and while
        :math:`N + \mathrm{stability_duration} < \mathrm{adaptation_duration}`
        the eigenvectors are being adapted. Post-adaptation phase the
        eigenvectors and eigenvalues are kept constant.
    initial_proposal : str (optional)
        Name of the initial proposal that is called before the number of
        proposal seps exceeds ``stability_duration''. By default se to the
        'epsie.proposals.ATAdaptiveProposal'. Supported options
        include: 'normal', 'at_adaptive_normal'.
    target_rate: float (optional)
        Target acceptance ratio. By default 0.234
    shuffle_rate : float, optional
        Probability of shuffling the eigenvector jump probabilities. By
        default 0.33.
    jump_interval : int, optional
        The jump interval of the proposal, the proposal only gets called every
        jump interval-th time. After ``adaptation_duration`` number of
        proposal steps elapses the proposal will again be called on every
        chain iteration. By default ``jump_interval`` = 1.
    """
    name = 'adaptive_eigenvector'
    symmetric = True

    def __init__(self, parameters, stability_duration, adaptation_duration,
                 target_rate=0.234, shuffle_rate=0.33, jump_interval=1,
                 initial_proposal='at_adaptive_normal'):
        # set the parameters, initialize the covariance matrix
        super(AdaptiveEigenvector, self).__init__(
            parameters=parameters, stability_duration=stability_duration,
            shuffle_rate=shuffle_rate, jump_interval=jump_interval,
            jump_interval_duration=adaptation_duration+stability_duration,
            initial_proposal=initial_proposal)
        # set up the adaptation parameters
        self.setup_adaptation(adaptation_duration, target_rate)
