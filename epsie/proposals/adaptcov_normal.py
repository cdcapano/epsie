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

import numpy as np
from scipy import stats

from .normal import Normal


@add_metaclass(ABCMeta)
class AdaptiveCovarianceSupport(object):
    r""" Utility class for adding adaptive covariance support to a proposal.

    The adaptation algorithm is based on  Algorithm 1 in [1]. The proposal
    covariance matrix is being adjusted at each step. See notes and [1] for
    more details

    Notes
    -----
    add maths overview

    References
    ----------
    [1] Christian L. Müller, Exploring the common concepts of adaptive MCMC
        and Covariance Matrix Adaptation schemes,
        https://mosaic.mpi-cbg.de/docs/Muller2010b.pdf
    """

    _prior_widths = None
    _deltas = None
    _adaptation_duration = None
    _start_iteration = None
    target_rate = None

    def setup_adaptation(self, prior_widths, adaptation_duration,
                         start_iteration=1, target_rate=0.234,
                         initial_mean=None, initial_cov=None):
        r"""Sets up the adaptation parameters.

        Parameters
        ----------
        """
        self.prior_widths = prior_widths
        self.adaptation_duration = adaptation_duration
        self.start_iteration = start_iteration

        self.target_rate = target_rate
        # Set up the initial mean
        if initial_mean is None:
            # update this to be in the middle of the bounds
            initial_mean = np.array([5, 3])
        # Set the mean to the initial
        self._mean = initial_mean
        # Set up the initial covariance
        if initial_cov is None:
            deltas = np.array([self.prior_widths[p] for p in self.parameters])
            initial_cov = (np.eye(len(self.prior_widths))
                           * (1 - self.target_rate) * 0.09 * deltas)
        # set the covariance to the initial
        self._cov = initial_cov
        # this will later be used to achieve target acceptance fraction
        # for now keep unity
        self._r = 1

        self._adaptation_end = self.start_iteration + self.adaptation_duration

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

        if self.start_iteration < chain.iteration < self._adaptation_end:
            # change this decay later. For now ok
            decay = (chain.iteration - self.start_iteration)**(-0.2) - 0.1

            # how to better unpack the last position?
            names = chain.positions.dtype.names
            lpos = chain.positions[-1]
            newpt = np.array([lpos[p] for p in names])
            # Update the first and second moments
            new_mean = self._mean + decay * (newpt - self._mean)
            d = (newpt - self._mean).reshape(-1, 1)
            new_cov = self._cov + decay * (np.matmul(d, d.T) - self._cov)

            # this is to ensure that the logpdf method does not throw an error
            # sometimes it returns SIngularMatrix even if it can sample a rvs
            # but the logpdf method fails. ANyway there has to be some check
            # for singular matrices
            try:
                __ = stats.multivariate_normal.logpdf(newpt, mean=new_mean,
                                                      cov=new_cov)
                self._mean = new_mean
                self._cov = self._r**2 * new_cov
            except np.linalg.LinAlgError:
                pass
            # diagnostics
#        if chain.iteration % 100 == 0:
#            if chain.iteration > 10000:
#                decay = (chain.iteration - self.start_iteration)**(-0.2) - 0.1
#                print(chain.iteration, decay)
#                print('mean', self._mean)
#                print('cov', self._cov)

    @property
    def state(self):
        return {'random_state': self.random_state,
                'mean': self._mean,
                'cov': self._cov,
                'r': self._r,
               }

    def set_state(self, state):
        self.random_state = state['random_state']
        self._mean = state['mean']
        self._cov = state['cov']
        self._rv = state['r']


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
    start_iteration: int (optional)
        The iteration index when adaptation phase begins.
    \**kwargs :
        All other keyword arguments are passed to
        :py:func:`AdaptiveSupport.setup_adaptation`. See that function for
        details.
    """
    name = 'adaptive_covariance_normal'
    symmetric = False

    def __init__(self, parameters, prior_widths, adaptation_duration,
                 start_iteration=1, **kwargs):
        # set the parameters, initialize the covariance matrix
        super(AdaptiveCovarianceNormal, self).__init__(parameters)
        # set up the adaptation parameters
        self.setup_adaptation(prior_widths, adaptation_duration,
                              start_iteration, **kwargs)
        self._isdiagonal = False
