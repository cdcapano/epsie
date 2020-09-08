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


import numpy
from scipy import stats


from .base import BaseBirth
from .bounded_normal import Boundaries


class UniformBirth(BaseBirth):
    """Uniform birth distribution object used in nested transdimensional
    proposals to propose birth to parameters which were previously inactive.

    Parameters
    ----------
    parameters : (list of) str
        The names of the parameters to produce proposals for.
    boundaries : dict
        Dictionary mapping parameters to boundaries.
    Properties
    ----------
    birth : dict
        Returns random variate sample from the uniform distribution for each
        parameter  (dictionary).

    Methods
    ------
    logpdf : py:func
        Evalues the logpdf of the proposal. Takes dictionary of parameters as
        input.
    """
    name = 'uniform_birth'

    def __init__(self, parameters, boundaries):
        self.parameters = parameters
        self._boundaries = None
        self.boundaries = boundaries

    @property
    def boundaries(self):
        """Dictionary of parameter -> boundaries."""
        return self._boundaries

    @boundaries.setter
    def boundaries(self, boundaries):
        """Sets the boundaries, making sure that widths are provided for
        each parameter in ``parameters``.
        """
        try:
            self._boundaries = {p: Boundaries(boundaries[p])
                                for p in self.parameters}
        except KeyError:
            raise ValueError("must provide a boundary for every parameter")

    @property
    def birth(self):
        return {p: self.random_generator.uniform(
            self.boundaries[p].lower, self.boundaries[p].upper)
            for p in self.parameters}

    def logpdf(self, xi):
        return sum([stats.uniform.logpdf(
            xi[p], loc=self.boundaries[p].lower, scale=abs(self.boundaries[p]))
            for p in self.parameters])

    @property
    def state(self):
        return {'random_state': self.random_state}

    def set_state(self, state):
        self.random_state = state['random_state']


class NormalBirth(BaseBirth):
    """Normal birth distribution object used in nested transdimensional
    proposals to propose birth to parameters which were previously inactive.

    Parameters
    ----------
    parameters : (list of) str
        The names of the parameters to produce proposals for.
    mean : dict
        Dictionary mapping parameters to their means
    std : dict
        Dictionary mapping parameters to their stds
    Properties
    ----------
    birth : dict
        Returns random variate sample from the uniform distribution for each
        parameter  (dictionary).

    Methods
    ------
    logpdf : py:func
        Evalues the logpdf of the proposal. Takes dictionary of parameters as
        input.
    """
    name = 'normal_birth'

    def __init__(self, parameters, mean, std):
        self.parameters = parameters
        self._mu = None
        self._std = None
        self.setup_mu_std(mean, std)

    @property
    def std(self):
        """Dictionary of parameter -> std."""
        return self._std

    @property
    def mu(self):
        """Dictionary of parameter -> mu."""
        return self._mu

    def setup_mu_std(self, mu, std):
        """Sets the mean and standard deviation for each parameter
        in ``parameters``."""
        try:
            self._mu = {p: mu[p] for p in self.parameters}
            self._std = {p: std[p] for p in self.parameters}
        except KeyError:
            raise ValueError("must provide a mu and std for every parameter")

    @property
    def birth(self):
        return {p: self.random_generator.normal(
            loc=self.mu[p], scale=self.std[p]) for p in self.parameters}

    def logpdf(self, xi):
        return sum([stats.norm.logpdf(xi[p], loc=self.mu[p], scale=self.std[p])
                    for p in self.parameters])

    @property
    def state(self):
        return {'random_state': self.random_state}

    def set_state(self, state):
        self.random_state = state['random_state']


class LogNormalBirth(BaseBirth):
    """Normal birth distribution object used in nested transdimensional
    proposals to propose birth to parameters which were previously inactive.

    Parameters
    ----------
    parameters : (list of) str
        The names of the parameters to produce proposals for.
    mean : dict
        Dictionary mapping parameters to their means
    std : dict
        Dictionary mapping parameters to their stds
    Properties
    ----------
    birth : dict
        Returns random variate sample from the uniform distribution for each
        parameter  (dictionary).

    Methods
    ------
    logpdf : py:func
        Evalues the logpdf of the proposal. Takes dictionary of parameters as
        input.
    """
    name = 'log_normal_birth'

    def __init__(self, parameters, mean, std):
        self.parameters = parameters
        self._mu = None
        self._std = None
        self.setup_mu_std(mean, std)

    @property
    def std(self):
        """Dictionary of parameter -> std."""
        return self._std

    @property
    def mu(self):
        """Dictionary of parameter -> mu."""
        return self._mu

    def setup_mu_std(self, mu, std):
        """Sets the mean and standard deviation for each parameter
        in ``parameters``."""
        try:
            self._mu = {}
            self._std = {}
            for p in self.parameters:
                # mu_x, std_x are the linear mean and std
                mu_x = mu[p]
                std_x = std[p]
                # mu_log, std_lgo are the mean and std of the log dist.
                mu_log = numpy.log(mu_x**2 / numpy.sqrt(mu_x**2 + std_x**2))
                std_log = numpy.sqrt(numpy.log(1 + (std_x/mu_x)**2))
                self._mu.update({p: mu_log})
                self._std.update({p: std_log})
        except KeyError:
            raise ValueError("must provide a mu and std for every parameter")

    @property
    def birth(self):
        return {p: self.random_generator.lognormal(
            mean=self.mu[p], sigma=self.std[p]) for p in self.parameters}

    def logpdf(self, xi):
        return sum([stats.lognorm.logpdf(
            xi[p], s=self.std[p], scale=numpy.exp(self.mu[p]))
            for p in self.parameters])

    @property
    def state(self):
        return {'random_state': self.random_state}

    def set_state(self, state):
        self.random_state = state['random_state']
