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

    The adaptation algorithm is based on  Algorithm 2 in [1]. The proposal
    covariance matrix is being adjusted at each step. See notes and [1] for
    more details

    Notes
    -----
    Denoting :math: `g` to be the current samples, the new samples
    :math: `x_{g+1}` are sampled from
    :math: `\mathcal{N}\left(x_{g}, r^{2} \times C_{g}`. Where :math: `r` is
    a scalar multipliying the covariance.

    .. math::

        C_{g+1} = C_{g} +  \gamma_{g+1}\left[\left(x_{g+1} - m_{g}\right)
                  \left(x_{g+1} - m_{g}\right)^{T} - C_{g}\right],

    where :math:`\gamma_{g+1}` is the vanishing decay and :math: `m_{g}` is a
    supporting vector that is also being updated at each turn:

    .. math::

        m_{g+1} = m_{g} + \gamma_{g+1}\left(x_{g+1} - m_{g}\right).

    The global scaling factor is updated to achieve a target acceptance ratio,
    :math: `alpha^*`:

    .. math::
        \log\left(r_{g+1}\right) = \log\left(r_{g}\right)
            + \gamma_{g+1}\left[\alpha\left(x_{g}, y_{g+1}\left)
                                            - alpha^*\right)\right],

    where :math `\alpha` is the Metropolis-Hastings ratio and :math `y_{g+1}`
    is the next proposed step. By default the following relation is used
    for the vanishing decay:

    .. math::
        \gamma_{g+1} = \left(g - g_{0}\right)^{-0.5} - C,

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

    _target_acceptance = None
    _start_iter = None

    def setup_adaptation(self, start_iter=1, target_acceptance=0.234):
        r"""Sets up the adaptation parameters.
        start_iter: int (optional)
            The iteration index when adaptation phase begins.
        target_acceptance: float (optional)
            Target acceptance ratio. By default 0.234

        Parameters
        ----------
        """
        self.start_iter = start_iter
        self.target_acceptance = target_acceptance
        ndim = len(self.parameters)
        self._mean = numpy.zeros(ndim) # initial mean
#        self._logr = 0.0 # initial global scaling
        self._cov = numpy.eye(ndim) # inital covariance
        self._log_component_scaling = np.ones(ndim)

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

    def decay(self, iteration):
        """Adaptive decay to ensure vanishing adaptation. Later make this the
        default choice and let user specify the functional form"""
        return (iteration - self.start_iter + 1)**(-0.5)

    def update(self, chain):
        """Updates the adaptation based on whether the last jump was accepted.
        This prepares the proposal for the next jump.
        """
        if self.start_iter < chain.iteration:
            decay = self.decay(chain.iteration)
            newpt = numpy.array([chain.positions[-1][p]
                                 for p in self.parameters])
            # Update the first and second moments
            d = newpt - self._mean
            new_mean = self._mean + decay * d
            d = d.reshape(-1, 1)
            new_cov = (self._cov + decay*(numpy.matmul(d, d.T) - self._cov))

            alpha_mh = min(1, chain.acceptance['acceptance_ratio'][-1])
            current_logl, current_logp = chain.stats[-1]
            current_pos = chain.positions[-1]
            proposed_pos = chain.proposed_positions[-1]

# Not finished. Pushed just as an example

            for i, p in enumerate(self.parameters):
                newp = current_pos
                newp[p] += proposed_pos[p]

                r = chain.model(**newp)
                if self._hasblobs:
                    logl, logp, blob = r
                else:
                    logl, logp = r
                    blob = None

                if logp == -numpy.inf:
                    ar = 0.
                else:
                    logar = logp + logl * chain.beta \
                            - currrent_logp - current_logl * chain.beta
                    if not chain.proposal_dist.symmetric:
                        logar += self.proposal_dist.logpdf(current_pos,
                                                           proposal)\
                                 - self.proposal_dist.logpdf(proposal,
                                                             current_pos)



#                logar = 

#                self._log_component_scaling[i] += decay * 

            new_logr = self._logr\
                       + decay * (alpha_mh - self.target_acceptance)
            try:
                r = numpy.exp(new_logr)
                __ = stats.multivariate_normal(new_mean,
                                               r**2 * new_cov).logpdf(newpt)
                self._mean = new_mean
                self.cov = r**2 * new_cov
                self._logr = new_logr

            except numpy.linalg.LinAlgError:
                pass

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
    start_iter: int (optional)
        The iteration index when adaptation phase begins.
    target_acceptance: float (optional)
        Target acceptance ratio. By default 0.234
    \**kwargs :
        All other keyword arguments are passed to
        :py:func:`AdaptiveSupport.setup_adaptation`. See that function for
        details.
    """
    name = 'adaptive_covariance_normal'
    symmetric = False

    def __init__(self, parameters, start_iter=1, target_acceptance=0.234,
                 **kwargs):
        # set the parameters, initialize the covariance matrix
        super(AdaptiveCovarianceNormal, self).__init__(parameters)
        # set up the adaptation parameters
        self.setup_adaptation(start_iter, target_acceptance, **kwargs)
        self._isdiagonal = False
