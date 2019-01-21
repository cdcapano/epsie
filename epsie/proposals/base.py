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

# Py3XX: delete abstractproperty
from abc import ABCMeta, abstractmethod, abstractproperty

import numpy
from randomgen import RandomGenerator
from scipy import stats

import epsie


class BaseProposal(object):
    """Abstract base class for all proposal classes."""
    __metaclass__ = ABCMeta
    name = None

    @property
    def brng(self):
        """The basic random number generator (BRNG) instance being used.

        A BRNG will be created if it doesn't exist yet.
        """
        try:
            return self._brng
        except AttributeError:
            self._brng = self._create_brng()
            return self._brng

    @brng.setter
    def brng(self, brng):
        """Sets the basic random number generator (BRNG) to use.

        Parameters
        ----------
        brng : :py:class:`epsie.BRNG`, int, or None
            Either the BRNG to use or an integer/None. If the latter, a
            BRNG will be created by passing ``brng`` as the ``seed`` argument
            to :py:func:`epsie.create_brng`.
        """
        if not isinstance(brng, epsie.BRNG):
            brng = epsie.create_brng(brng)
        self._brng = brng

    @property
    def random_generator(self):
        """The random number generator.

        This is an instance of :py:class:`randgen.RandomGenerator` that is
        derived from the BRNG. It provides has methods to create random
        draws from various distributions.
        """
        return self.brng.generator

    @property
    def random_state(self):
        """The current state of the basic random number generator (BRNG).
        """
        return self.brng.state

    @random_state.setter
    def random_state(self, state):
        """Sets the state of brng.
        
        Parameters
        ----------
        state : dict
            Dictionary giving the state to set.
        """
        self.brng.state = state

    # Py3XX: uncomment the next two lines 
    # @property
    # @abstractmethod
    @abstractproperty  # Py3XX: delete line
    def symmetric(self):
        """Boolean indicating whether the proposal distribution is symmetric
        from jump to jump."""
        pass

    # Py3XX: uncomment the next two lines 
    # @property
    # @abstractmethod
    @abstractproperty  # Py3XX: delete line
    def state(self):
        """Returns all information needed to produce a deterministic jump.

        The information returned by this property should be everything needed
        such that if you pass it to set_state, you will get the same proposal
        on the next call of jump.

        The information should be returned as a dictionary. At the very least,
        this should include the current state of the proposal's
        ``random_state``. For adaptive proposals, this may also include the
        buffer used to adjust the proposal distribution.
        """
        pass

    @abstractmethod
    def set_state(self, state):
        """Set all information needed to produce a deterministic jump.
        """
        pass

    @abstractmethod
    def jump(self, fromx):
        """This should provide random samples from the proposal distribution.

        Samples should be returned as a dictionary mapping parameters to
        the proposed jump.
        """
        pass

    @abstractmethod
    def logpdf(self, xi, givenx):
        """The log pdf of the proposal distribution at a point.

        Parameters
        ----------
        xi : dict
            Dictionary mapping parameter names to values to evaluate.
        givenx : dict, optional
            Dictionary mapping parameter names to values of the point from
            which ``xi`` is evaluated.
        """
        pass

    def pdf(self, xi, givenx):
        """The pdf of proposal at the given values.

        This just expoentiates ``logpdf``.

        Parameters
        ----------
        xi : dict
            Dictionary mapping parameter names to values to evaluate.
        givenx : dict, optional
            Dictionary mapping parameter names to values of the point from
            which ``xi`` is evaluated.
        """
        return numpy.exp(self.logpdf(xi, givenx))

    def update(self, chain):
        """Update the state of the proposal distribution using the given chain.

        This method may optionally be implemented by a proposal. It is called
        by the Markov chains just prior to calling jump. It can be used by,
        e.g., adaptive jump proposals that change their state depending on
        the history of the chain.
        """
        pass
