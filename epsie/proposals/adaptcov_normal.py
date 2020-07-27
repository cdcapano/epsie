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

from scipy import stats
import numpy

from .normal import Normal


@add_metaclass(ABCMeta)
class AdaptiveCovarianceSupport(object):
    r""" Utility class for adding adaptive covariance support to a proposal.

    The adaptation algorithm is based on  Algorithm 1 in [1]. The proposal
    covariance matrix is being adjusted at each step. See notes and [1] for
    more details

    Notes
    -----
    Denoting :math: `g` to be the current samples, the new samples
    :math: `x_{g+1}` are sampled from
    :math: `\mathcal{N}\left(x_{g}, r^{2} \times C_{g}`. Where :math: `r` is
    a scalar multipliying the covariance. For now :math: `r=1`. The covariance
    is updates so that:

    .. math::

        C_{g+1} = C_{g} +  \gamma_{g+1}\left[\left(x_{g+1} - m_{g}\right)
                  \left(x_{g+1} - m_{g}\right)^{T} - C_{g}\right],

    where :math:`\gamma_{g+1}` is the vanishing decay and :math: `m_{g}` is a
    supporting vector that is also being updated at each turn:

    .. math::

        m_{g+1} = m_{g} + \gamma_{g+1}\left(x_{g+1} - m_{g}\right).

    By default the following relation is used for the vanishing decay:

    .. math::
        \gamma_{g+1} = \left(g - g_{0}\right)^{-0.2} - C,

    where :math: `g_{0}` is the iteration at which adaptation starts,
    by default :math: `g_{0}=1` and :math: `C` is a positive constant
    ensuring that when the adaptation phase ends the vanishing decay tends to
    zero.

    References
    ----------
    [1] Christian L. Muller, Exploring the common concepts of adaptive MCMC
        and Covariance Matrix Adaptation schemes,
        https://mosaic.mpi-cbg.de/docs/Muller2010b.pdf
    """

    _adapt_dur = None
    _start_iter = None

    def setup_adaptation(self, adapt_dur, start_iter=1):
        r"""Sets up the adaptation parameters.
        adapt_dur : int
            The number of iterations over which to apply the adaptation.
            No more adaptation will be done once a chain exceeds this value.
        start_iter: int (optional)
            The iteration index when adaptation phase begins.

        Parameters
        ----------
        """
        self.adapt_dur = adapt_dur
        self.start_iter = start_iter
        ndim = len(self.parameters)
        # Set the mean to the initial
        self._mean = numpy.zeros(ndim)
        # Set up the initial covariance
        self._cov = numpy.eye(ndim)
        # this will later be used to achieve target acceptance fraction
        self._r = 1
        self._decay_const = (self.adapt_dur - self.start_iter - 1)**(-0.2)

    @property
    def adapt_dur(self):
        """The adaptation duration used."""
        return self._adapt_dur

    @adapt_dur.setter
    def adapt_dur(self, adapt_dur):
        """Sets the adaptation duration to the given value, making sure it is
        larger than 1.
        """
        if adapt_dur < 1:
            raise ValueError("adaptation duration must be >= 1")
        self._adapt_dur = adapt_dur

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

    def decay(self, iteration):
        """Adaptive decay to ensure vanishing adaptation. Later make this the
        default choice and let user specify the functional form"""
        return iteration**(-0.2) - self._decay_const

    def update(self, chain):
        """Updates the adaptation based on whether the last jump was accepted.
        This prepares the proposal for the next jump.
        """

        if self.start_iter < chain.iteration < (self.start_iter
                                                + self.adapt_dur) - 1:
            decay = self.decay(chain.iteration)
            newpt = numpy.array([chain.positions[-1][p]
                                 for p in self.parameters])
            # Update the first and second moments
            d = newpt - self._mean
            new_mean = self._mean + decay * d
            d = d.reshape(-1, 1)
            new_cov = (self.cov + decay*(numpy.matmul(d, d.T) - self.cov))

            try:
                __ = stats.multivariate_normal(new_mean, new_cov)
                self._mean = new_mean
                self.cov = self._r**2* new_cov
            except numpy.linalg.LinAlgError:
                # keep the old covariancec matrix
                pass
        else:
            # after the adaptive phase the proposal becomes symmetric
            self.symmetric = True

    @property
    def state(self):
        return {'random_state': self.random_state,
                'mean': self._mean,
                'cov': self._cov,
                'r': self._r,}

    def set_state(self, state):
        self.random_state = state['random_state']
        self._mean = state['mean']
        self._cov = state['cov']
        self._r = state['r']


class AdaptiveCovarianceNormal(AdaptiveCovarianceSupport, Normal):
    r"""Uses a normal distribution with adaptive variance for proposals.

    See :py:class:`AdaptiveSupport` for details on the adaptation algorithm.

    Parameters
    ----------
    parameters : (list of) str
        The names of the parameters.
    adapt_dur : int
        The number of iterations over which to apply the adaptation. No more
        adaptation will be done once a chain exceeds this value.
    start_iter: int (optional)
        The iteration index when adaptation phase begins.
    \**kwargs :
        All other keyword arguments are passed to
        :py:func:`AdaptiveSupport.setup_adaptation`. See that function for
        details.
    """
    name = 'adaptive_covariance_normal'
    symmetric = False

    def __init__(self, parameters, adapt_dur,
                 start_iter=1, **kwargs):
        # set the parameters, initialize the covariance matrix
        super(AdaptiveCovarianceNormal, self).__init__(parameters)
        # set up the adaptation parameters
        self.setup_adaptation(adapt_dur, start_iter, **kwargs)
        self._isdiagonal = False
