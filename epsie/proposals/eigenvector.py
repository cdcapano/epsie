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
from scipy import stats

from .base import (BaseProposal, BaseAdaptiveSupport)
from .normal import ATAdaptiveNormal


class Eigenvector(BaseProposal):
    """Uses a eigenvector jump with a fixed scale.

    This proposal may handle one or more parameters.

    Parameters
    ----------
    parameters : (list of) str
        The names of the parameters to produce proposals for.
    stability_duration : int
        Number of steps done with a normal proposal. After this eigenvalues
        and eigenvectors are calculated (and never again) and jumps proposed
        along those.
    shuffle_rate : float, optional
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

    def __init__(self, parameters, stability_duration, shuffle_rate=0.33,
                 jump_interval=1, jump_interval_duration=None):
        self.parameters = parameters
        self.ndim = len(self.parameters)
        self.shuffle_rate = shuffle_rate
        self.set_jump_interval(jump_interval, jump_interval_duration)
        # later add this
        self.start_step = 1

        self._cov = None
        self._mu = None
        self._eigvals = None
        self._eigvects = None
        self._jump_std = None

        self.initial_prop = ATAdaptiveNormal(
            self.parameters, adaptation_duration=stability_duration,
            jump_interval=jump_interval)
        self.initial_prop.bit_generator = self.bit_generator
        self.stability_duration = stability_duration

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

    def _jump(self, fromx):
        if self.nsteps <= self.stability_duration:
            return self.initial_prop.jump(fromx)
        else:
            probs = self._eigvals / numpy.sum(self._eigvals)
            if self.random_generator.uniform() < self.shuffle_rate:
                self.random_generator.shuffle(probs)
            choice = self.random_generator.choice(self.ndim, p=probs)
            self._jump_std = self._eigvals[choice]
            move_vector = self._eigvects[:, choice]

            delta = self.random_generator.normal(scale=self._jump_std)

            out = {p: fromx[p] + delta * move_vector[i]
                   for i, p in enumerate(self.parameters)}
            return out

    def _logpdf(self, xi, givenx):
        if self.nsteps <= self.stability_duration:
            return self.initial_prop.logpdf(xi, givenx)
        else:
            delta = (sum((givenx[p] - xi[p])**2 for p in self.parameters))**0.5
            return stats.norm.logpdf(delta, loc=0, scale=self._jump_std)

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
        if self.nsteps <= self.stability_duration // 2 + 4:
            self.initial_prop.update(chain)
        elif self.nsteps <= self.stability_duration:
            self.initial_prop.update(chain)
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
                self._eigvals, self._eigvects = numpy.linalg.eigh(self._cov)

    def _update(self, chain):
        if self.nsteps <= self.stability_duration:
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
        adaptation_duration : int, optional
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
        if dk <= 0:
            self._stability_update(chain)
        elif dk < self.adaptation_duration:
            self._recursive_mean_cov(chain)
            # update eigenvalues and eigenvectors
            self._eigvals, self._eigvects = numpy.linalg.eigh(self._cov)
            # update the scaling factor
            dk = dk**(-0.6) - self._decay_const
            # Update of the global scaling
            ar = chain.acceptance['acceptance_ratio'][-1]
            self._log_lambda += dk * (ar - self.target_rate)
            # Rescale the eigenvalues
            self._eigvals *= numpy.exp(self._log_lambda)

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
            self._eigvals, self._eigvects = numpy.linalg.eigh(self._cov)
            self._eigvals *= numpy.exp(self._log_lambda)


class AdaptiveEigenvector(AdaptiveEigenvectorSupport, Eigenvector):
    r""" Uses jumps along eigenvectors with adaptive scales

    See :py:class:`AdaptiveEigenvectorSupport` for details on the adaptation
    algorithm.

    Parameters
    ----------
    parameters: (list of) str
        The names of the parameters.
    stability_duration : int
        Number of steps done with a normal proposal. After this eigenvalues
        and eigenvectors are calculated for the first time and jumps proposed
        along those.
    adaptation_duration: int
        The number of proposal steps over which to apply the adaptation. No
        more adaptation will be done once a proposal has adapted over this
        duration.
    target_rate: float, optional
        Target acceptance ratio. By default 0.234
    shuffle_rate : float, optional
        Probability of shuffling the eigenvector jump probabilities. By
        default 0.33.
    jump_interval : int, optional
        The jump interval of the proposal, the proposal only gets called every
        jump interval-th time. After ``adaptation_duration`` number of
        proposal steps elapses the proposal will again be called on every
        chain iteration. By default ``jump_interval`` = 1.
    jump_interval_duration : int, optional
        The number of proposals steps during which values of ``jump_interval``
        other than 1 are used. After this elapses the proposal is called on
        each iteration.
    """
    name = 'adaptive_eigenvector'
    symmetric = True

    def __init__(self, parameters, stability_duration, adaptation_duration,
                 target_rate=0.234, shuffle_rate=0.33, jump_interval=1,
                 **kwargs):
        # set the parameters, initialize the covariance matrix
        super(AdaptiveEigenvector, self).__init__(
              parameters, stability_duration, shuffle_rate, jump_interval,
              jump_interval_duration=adaptation_duration)
        # set up the adaptation parameters
        self.setup_adaptation(adaptation_duration, target_rate, **kwargs)
