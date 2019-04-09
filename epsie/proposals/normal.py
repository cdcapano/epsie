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
        The covariance matrix of the parameters. If provided, must have shape
        `ndim x ndim` where `ndim` = the number of parameters given. Default
        (None) is to use the identity matrix; i.e., unit variance, with all
        parameters independent of each other.

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
        if cov is None and self.ndim == 1:
            cov = 1.
        elif cov is None:
            cov = numpy.diag(numpy.ones(len(parameters)))
        # check that dimensionality makes sense
        if self.ndim == 1 and isinstance(cov, numpy.ndarray) \
                or self.ndim > 1 and not isinstance(cov, numpy.ndarray) \
                or self.ndim > 1 and self.ndim != cov.ndim:
            raise ValueError("dimension of covariance matrix does not match "
                             "given number of parameters")
        self.cov = cov

    @property
    def state(self):
        return {'random_state': self.random_state}

    def set_state(self, state):
        self.random_state = state['random_state']

    def jump(self, fromx):
        if self.ndim == 1:
            newpt = self.random_generator.normal(fromx[p], self.cov)
            jump = {p: newpt}
        else:
            newpt = self.random_generator.multivariate_normal(
                [fromx[p] for p in self.parameters], self.cov)
            jump = {p: newpt[ii] for ii, p in enumerate(self.parameters)}
        return jump
                    

    def logpdf(self, xi, givenx):
        means = [givenx[p] for p in self.parameters]
        if self.ndim == 1:
            p = self.parameters[0]
            logp = stats.normal.logpdf(xi[p], loc=givenx[p], scale=self.cov)
        else:
            logp = stats.multivariate_normal(
                [xi[p] for p in self.parameters],
                mean=[givenx[p] for p in self.parameters],
                cov=self.cov)
        return logp


class AdaptiveNormal(Normal):
    r"""Uses a Gaussian distribution with adaptive variance.

    The adaptation algorithm is based on Eqs. 35 and 36 of [1]_.
    The size of the variance at each step is based on the width of the prior
    and whether or not the previous proposal was accepted or not. See Notes
    for more details.

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
        equation below). If not provided, will use :math:`1/log_10(T)`, where
        :math:`T` is the adaptation duration.
    start_iteration : int, optional
        The iteration to start doing the adaptation (:math:`k_0+1` in the
        equation below). Must be greater than zero. Default is 1.
    target_rate : float, optional
        The target acceptance rate. Default is 0.234.
    initial_var : array, optional
        The initial variance to use. Default is to use
        `(1 - target_rate)*0.09*prior_widths`.

    Notes
    -----
    For a given parameter, the variance of the :math:`k`th iteration is given
    by [1]_:

    .. math::

        \sigma_k = \sigma_{k-1} + \alpha_{k-1}
            \left[\left(k - k_0\right)^{-\beta} - 0.1\right]\frac{\Delta}{10},

    where :math:`\alpha_{k-1} = 1 - \xi` if the previous iteration was
    accpeted and :math:`\alpha_{k-1} = -\xi` if the previous iteration was
    rejected. Here, :math:`\xi` is the target acceptance rate, :math:`\Delta`
    is the prior width, :math:`\beta` is the adaptation decay, and :math:`k_0`
    gives the iteration after which the adaptation begins. The initial variance
    :math:`\sigma_0` to use is a free parameter. The default in this function
    is to use :math:`\sigma_0 = (1-\xi)0.09\Delta`.


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
                 initial_var=None):
        # set the parameters, initialize the covariance matrix
        super(AdaptiveNormal, self).__init__(parameters)


    @property
    def prior_widths(self):
        """The width of the prior used for each parameter."""
        return self._prior_widths

    @prior_widths.setter
    def prior_widths(self, prior_widths):
        """Sets the prior widths, making sure that widths are provided for
        each parameter in ``parameters``.
        """
        try:
            self._prior_widths = {p: abs(prior_widths[p])
                                  for p in self.parameters}
        except KeyError:
            raise ValueError("must provide prior widths for every parameter")

    def update(self, chain):
        """Updates the adaptation based on whether the last iteration was
        accepted or not.
        """
        if chain[-1].acceptance['accepted']:
            pass
