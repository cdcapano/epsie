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

from .base import BaseProposal
from .normal import ATAdaptiveNormal


class Eigenvector(BaseProposal):
    """Uses a eigenvector jump with a fixed scale.

    This proposal may handle one or more parameters.

    Parameters
    ----------
    parameters : (list of) str
        The names of the parameters to produce proposals for.

    Attributes
    ----------
    """

    name = 'eigenvector'
    symmetric = False

    def __init__(self, parameters, burnin_duration):
        self.parameters = parameters
        self.ndim = len(self.parameters)

        self._cov = None
        self._mu = None
        self._eigvals = None
        self._eigvects = None


        self.burnin_duration = burnin_duration
        self.burnin_proposal = ATAdaptiveNormal(self.parameters,
                                           adaptation_duration=burnin_duration)

    @property
    def state(self):
        return {'nsteps': self._nsteps,
                'random_state': self.random_state}

    def set_state(self, state):
        self.random_state = state['random_state']
        self._nsteps = state['nsteps']

    def jump(self, fromx):
        if self.nsteps <= self.burnin_duration:
            return self.burnin_proposal.jump(fromx)
        else:
            choice = self.random_generator.choice(
                self.ndim, p=self._eigvals/numpy.sum(self._eigvals))
            move_vector = self._eigvects[:, choice]

            delta = self.random_generator.normal(scale=self._eigvals[choice])

            out = {p: fromx[p] + delta * move_vector[i]
                   for i, p in enumerate(self.parameters)}
            return out

    def logpdf(self, xi, givenx):
        if self.nsteps <= self.burnin_duration:
            return self.burnin_proposal.logpdf(xi, givenx)
        else:
            return 0

    def update(self, chain):
        if self.nsteps == self.burnin_duration:
            X = numpy.array([chain.positions[p][self.nsteps//2:]
                             for p in self.parameters]).T
            # calculate mean and cov
            self._mu = numpy.mean(X, axis=0)
            self._cov = numpy.cov(X, rowvar=False)
            # calculate the eigenvalues and eigenvectors
            self._eigvals, self._eigvects = numpy.linalg.eigh(self._cov)
        if self.nsteps > self.burnin_duration:
            # recursively update the mean and the covariance
            alpha = 1 / self.nsteps
            df = numpy.array(list(chain.current_position.values())) - self._mu
            self._mu +=  alpha * df
            df = df.reshape(-1, 1)
            self._cov += alpha * (numpy.matmul(df, df.T) - self._cov)
            # update eigenvalues and eigenvectors
            # possibly can be done recursively later
            self._eigvals, self._eigvects = numpy.linalg.eigh(self._cov)

        self.nsteps += 1
