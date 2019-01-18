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
from scipy import stats


class BaseProposal(object):
    """Abstract base class for all proposal classes."""
    __metaclass__ = ABCMeta
    name = None

    # Py3XX: uncomment the next two lines 
    # @property
    # @abstractmethod
    @abstractproperty  # Py3XX: delete line
    def random_state(self):
        """The ``RandomState`` instance the proposal uses.
        
        This should return the ``RandomState`` **class instance** the proposal
        uses, not the random state of that instance. 
        """
        pass

    @abstractmethod
    def set_random_state(self, random_state):
        """Sets the random state class used by the sampler to the given.
        
        Parameters
        ----------
        random_state : :py:class:numpy.random.RandomState
            A numpy RandomState class instance.
        """
        pass

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
    def jump(self, size=1):
        """This should provide random samples from the proposal distribution.

        Samples should be returned as a dictionary mapping parameters to
        the proposed jump.
        """
        pass

    @abstractmethod
    def logpdf(self, **vals):
        """The log of the pdf of the proposal distribution at the given values.

        Parameters
        ----------
        \**vals :
            The values are passed as keyword arguments mapping parameter
            names to test points.
        """
        pass

    def pdf(self, **vals):
        """The pdf of proposal at the given values.

        This just expoentiates ``logpdf``.

        Parameters
        ----------
        \**vals :
            The values are passed as keyword arguments mapping parameter
            names to test points.
        """
        return numpy.exp(self.logpdf(**vals))

    def update(self, chain):
        """Update the state of the proposal distribution using the given chain.

        This method may optionally be implemented by a proposal. It is called
        by the Markov chains just prior to calling jump. It can be used by,
        e.g., adaptive jump proposals that change their state depending on
        the history of the chain.
        """
        pass
