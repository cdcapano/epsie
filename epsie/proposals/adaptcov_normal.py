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
from scipy import stats

from .base import BaseProposal
from .normal import Normal

@add_metaclass(ABCMeta)
class AdaptiveCovarianceSupport(object):
    r""" add description
    Notes
    -----
    add maths

    References
    ----------
    add reference
    """

    _prior_widths = None
    _deltas = None
    _adaptation_duration = None
    _start_iteration = None
    target_rate = None

    def setup_adaptation(self, prior_widths, adaptation_duration,
                         start_iteration=1, target_rate=0.234,
                         initial_mean = None, initial_cov=None):
        r"""Sets up the adaptation parameters.

        Parameters
        ----------
        """
        self.prior_widths = prior_widths
        self.adaptation_duration = adaptation_duration
        # Let's just keep this adaptation decay for now
#        if adaptation_decay is None:
#            adaptation_decay = 1./numpy.log10(self.adaptation_duration)
#        self.adaptation_decay = adaptation_decay

        self.start_iteration = start_iteration

        self.target_rate = target_rate
        # Set up the initial mean
        if initial_mean is None:
#            initial_mean = 0.5 * numpy.array(list(prior_widths.values()))
            initial_mean = numpy.array([5, 3])
        # Set the mean to the initial
        self._mean = initial_mean
        # Set up the initial covariance
        if initial_cov is None:
            initial_cov = numpy.eye(len(self.prior_widths))\
                          * (1 - self.target_rate) * 0.09 * self.deltas

        # set the covariance to the initial
        self._cov = initial_cov


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

    def update(self, chain):
        """Updates the adaptation based on whether the last jump was accepted.

        This prepares the proposal for the next jump.
        """
        # subtact 1 from the start iteration, since the update happens after
        # the jump
        dk = chain.iteration - (self.start_iteration - 1)
        if 1 <= dk < self.adaptation_duration:
            decay = 0.8
#            decay = 10 * (dk - self.start_iteration)**(-0.2) - 1

            newpos = numpy.array([chain.positions[-1][p]\
                        for p in chain.positions.dtype.names]).reshape(-1, 1)
            oldmean = self._mean.reshape(-1, 1)
            diff = newpos - oldmean
#            newposition = np.array([float(sampler.positions[p][:, -1]) for p in sampler.positions.dtype.names])
            newmean = oldmean + decay * diff
            newcov = self._cov\
                     + decay * (numpy.matmul(diff, diff.T) - self._cov)

            print('newmean', newmean)
            self._mean = newmean.reshape(-1,)
            self._cov = newcov
            print('cov', self._cov)

    @property
    def state(self):
        return {'random_state': self.random_state,
                'mean': self._mean,
                'cov': self._cov,}

    def set_state(self, state):
        self.random_state = state['random_state']
        self._mean = state['mean']
        self._cov = state['cov']

class AdaptiveCovarianceNormal(AdaptiveCovarianceSupport, Normal):
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
    adaptation_duration : int
        The number of iterations over which to apply the adaptation. No more
        adaptation will be done once a chain exceeds this value.
    \**kwargs :
        All other keyword arguments are passed to
        :py:func:`AdaptiveSupport.setup_adaptation`. See that function for
        details.
    """
    name = 'adaptive_covariance_normal'
    symmetric = False

    def __init__(self, parameters, prior_widths, adaptation_duration,
                 **kwargs):
        # set the parameters, initialize the covariance matrix
        super(AdaptiveCovarianceNormal, self).__init__(parameters)
        # set up the adaptation parameters
        self.setup_adaptation(prior_widths, adaptation_duration, **kwargs)

