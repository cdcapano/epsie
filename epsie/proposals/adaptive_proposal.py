# Copyright (C) 2020, Richard Stiskalek, Collin Capano
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

from .normal import Normal


@add_metaclass(ABCMeta)
class AdaptiveProposalSupport(object):
    r""" Utility class for adding adaptive covariance proposal support.

    The adaptation algorithm is based on Algorithm 4 in [1].

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

    _target_acceptance = None
    _start_iter = None
    _decay_const = None
    _adaptation_duration = None

    def setup_adaptation(self, start_iter=1, adaptation_duration=None,
                         target_acceptance=0.234):
        r"""Sets up the adaptation parameters.
        start_iter: int (optional)
            The iteration index when adaptation phase begins.
        adaptation_duration: int (optional)
            The number of adaptation steps. By default assumes the adaptation
            never ends but decays.
        target_acceptance: float (optional)
            Target acceptance ratio. By default 0.234

        """
        self.start_iter = start_iter
        self.target_acc = target_acceptance
        ndim = len(self.parameters)
        self.adaptation_duration = adaptation_duration

        self._mean = numpy.zeros(ndim)  # initial mean
        self._unit_cov = numpy.eye(ndim)  # inital covariance
        self._cov = self._unit_cov
        self._log_lambda = 0

    @property
    def start_iter(self):
        """The iteration that the adaption begins."""
        return self._start_iter

    @start_iter.setter
    def start_iter(self, start_iter):
        """Sets the start iteration, making sure it is >= 1."""
        if start_iter < 1:
            raise ValueError("start_iter must be >= 1")
        self._start_iter = start_iter

    @property
    def target_acceptance(self):
        """Target acceptance ratio."""
        return self._target_acceptance

    @target_acceptance.setter
    def target_acceptance(self, target_acceptance):
        if not 0.0 < target_acceptance < 1.0:
            raise ValueError("Target acceptance must be in range (0, 1)")
        self._target_acceptance = target_acceptance

    @property
    def adaptation_duration(self):
        return self._adaptation_duration

    @adaptation_duration.setter
    def adaptation_duration(self, adaptation_duration):
        if adaptation_duration is None:
            self._decay_const = 0.
            self._adaptation_duration = numpy.infty
            return
        self._decay_const = (adaptation_duration)**(-0.6)
        self._adaptation_duration = adaptation_duration

    def decay(self, iteration):
        """Adaptive decay to ensure vanishing adaptation."""
        return (iteration - self.start_iter + 1)**(-0.6) - self._decay_const

    def update(self, chain):
        """Updates the adaptation based on whether the last jump was accepted.
        This prepares the proposal for the next jump.
        """
        # if start iteration is not 0 then take a weighted average to
        # get an estimate of the initial mean and covariance.
        # do this one iteration before start adaptation
        if chain.iteration == self.start_iter - 1:
            weights = numpy.arange(chain.iteration, 0, -1)**(0.6)
            positions = numpy.vstack([chain.positions[p]
                                     for p in self.parameters]).T
            self._mean = numpy.average(positions, weights=weights, axis=0)
            self.cov = numpy.cov(positions, rowvar=False, aweights=weights)

        if 0 < chain.iteration - self.start_iter < (self.adaptation_duration):
            decay = self.decay(chain.iteration)
            newpt = numpy.array([chain.current_position[p]
                                 for p in self.parameters])
            # Update the first moment
            df = newpt - self._mean
            self._mean = self._mean + decay * df
            # Update the second moment
            df = df.reshape(-1, 1)
            self._unit_cov += decay * (numpy.matmul(df, df.T) - self._unit_cov)
            # Update of the global scaling
            ar = min(1, chain.acceptance['acceptance_ratio'][-1])
            self._log_lambda += decay * (ar - self.target_acc)

            self.cov = numpy.exp(self._log_lambda) * self._unit_cov

    @property
    def state(self):
        return {'random_state': self.random_state,
                'mean': self._mean,
                'cov': self._cov,
                'unit_cov': self._unit_cov,
                'log_lambda': self._log_lambda}

    def set_state(self, state):
        self.random_state = state['random_state']
        self._mean = state['mean']
        self._cov = state['cov']
        self._unit_cov = state['unit_cov']
        self._log_lambda = state['log_lambda']


class AdaptiveProposal(AdaptiveProposalSupport, Normal):
    r"""Uses a normal distribution with adaptive covariance for proposals.

    See :py:class:`AdaptiveProposalSupport` for details on the adaptation
    algorithm.

    Parameters
    ----------
    parameters: (list of) str
        The names of the parameters.
    start_iter: int (optional)
        The iteration index when adaptation phase begins.
    adaptation_duration: int (optional)
        The iteration index when adaptation phase ends. By default never ends.
    target_acceptance: float (optional)
        Target acceptance ratio. By default 0.234
    \**kwargs:
        All other keyword arguments are passed to
        :py:func:`AdaptiveSupport.setup_adaptation`. See that function for
        details.
    """
    name = 'adaptive_proposal'
    symmetric = False

    def __init__(self, parameters, start_iter=1, adaptation_duration=None,
                 target_acceptance=0.234, **kwargs):
        # set the parameters, initialize the covariance matrix
        super(AdaptiveProposal, self).__init__(parameters)
        # set up the adaptation parameters
        self.setup_adaptation(start_iter, adaptation_duration,
                              target_acceptance, **kwargs)
        self._isdiagonal = False
