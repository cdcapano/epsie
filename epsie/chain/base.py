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

"""Base class for Markov chains."""

from __future__ import absolute_import

from abc import (ABCMeta, abstractmethod, abstractproperty)
import six
from six import add_metaclass
import numpy


@add_metaclass(ABCMeta)
class BaseChain(object):
    """Abstract base class for Markov chains.

    Provides standard functions for Chain and ParallelTemperedChain.

    Attributes
    ----------
    parameters
    iteration
    lastclear
    scratchlen
    positions
    stats
    acceptance
    blobs
    start_position
    stats0
    blob0
    current_position
    current_stats
    current_blob
    bit_generator
    random_state
    state
    hasblobs
    chain_id : int
        Integer identifying the chain. Default is 0.
    """
    chain_id = 0

    @property
    def parameters(self):
        """The sampled parameters, as a tuple.
        """
        return self._parameters

    @parameters.setter
    def parameters(self, parameters):
        """Stores the parameters as a tuple."""
        if isinstance(parameters, six.string_types):
            parameters = [parameters]
        self._parameters = tuple(parameters)

    def __len__(self):
        return self.iteration - self.lastclear

    # Py3XX: uncomment the next two lines
    # @property
    # @abstractmethod
    @abstractproperty  # Py3XX: delete line
    def hasblobs(self):
        """Whether the model returns blobs."""
        pass

    # Py3XX: uncomment the next two lines
    # @property
    # @abstractmethod
    @abstractproperty  # Py3XX: delete line
    def iteration(self):
        """The number of times the chain has been stepped."""
        pass

    # Py3XX: uncomment the next two lines
    # @property
    # @abstractmethod
    @abstractproperty  # Py3XX: delete line
    def lastclear(self):
        """Returns the iteration of the last time the chain memory was cleared.
        """
        pass

    # Py3XX: uncomment the next two lines
    # @property
    # @abstractmethod
    @abstractproperty  # Py3XX: delete line
    def scratchlen(self):
        """The length of the scratch space used."""
        pass

    # Py3XX: uncomment the next two lines
    # @property
    # @abstractmethod
    @abstractproperty  # Py3XX: delete line
    def start_position(self):
        """Dictionary mapping parameters to their start position."""
        pass

    # Py3XX: uncomment the next two lines
    # @property
    # @abstractmethod
    @abstractproperty  # Py3XX: delete line
    def stats0(self):
        """Dictionary of the log likelihood and prior at the start position."""
        pass

    # Py3XX: uncomment the next two lines
    # @property
    # @abstractmethod
    @abstractproperty  # Py3XX: delete line
    def blob0(self):
        """Dictionary of the blob data at the start position."""
        pass

    # Py3XX: uncomment the next two lines
    # @property
    # @abstractmethod
    @abstractproperty  # Py3XX: delete line
    def positions(self):
        """The history of all of the positions, as a structred array."""
        pass

    # Py3XX: uncomment the next two lines
    # @property
    # @abstractmethod
    @abstractproperty  # Py3XX: delete line
    def stats(self):
        """The log likelihoods and log priors of the positions, as a structred
        array.
        """
        pass

    # Py3XX: uncomment the next two lines
    # @property
    # @abstractmethod
    @abstractproperty  # Py3XX: delete line
    def acceptance(self):
        """The history of all of acceptance ratios and accepted booleans, as
        a structred array.
        """
        pass

    # Py3XX: uncomment the next two lines
    # @property
    # @abstractmethod
    @abstractproperty  # Py3XX: delete line
    def blobs(self):
        """The history of all of the blob data, as a structred array.

        If the model does not return blobs, this is just ``None``.
        """
        pass

    # Py3XX: uncomment the next two lines
    # @property
    # @abstractmethod
    @abstractproperty  # Py3XX: delete line
    def current_position(self):
        """Dictionary of the current position of the chain."""
        pass

    # Py3XX: uncomment the next two lines
    # @property
    # @abstractmethod
    @abstractproperty  # Py3XX: delete line
    def current_stats(self):
        """Dictionary giving the log likelihood and log prior of the current
        position.
        """
        pass

    # Py3XX: uncomment the next two lines
    # @property
    # @abstractmethod
    @abstractproperty  # Py3XX: delete line
    def current_blob(self):
        """Dictionary of the blob data of the current position.

        If the model does not return blobs, just returns ``None``.
        """
        pass

    @abstractmethod
    def clear(self):
        """Clears memory of the current chain, and sets start position to the
        current position.
        """
        pass

    @abstractmethod
    def __getitem__(self, index):
        """Returns all of the chain data at the requested index."""
        pass

    # Py3XX: uncomment the next two lines
    # @property
    # @abstractmethod
    @abstractproperty  # Py3XX: delete line
    def bit_generator(self):
        """The random bit generator being used."""
        pass

    # Py3XX: uncomment the next two lines
    # @property
    # @abstractmethod
    @abstractproperty  # Py3XX: delete line
    def random_generator(self):
        """Returns the random number generator."""
        pass

    # Py3XX: uncomment the next two lines
    # @property
    # @abstractmethod
    @abstractproperty  # Py3XX: delete line
    def random_state(self):
        """The current state of the random bit generator."""
        pass

    # Py3XX: uncomment the next two lines
    # @property
    # @abstractmethod
    @abstractproperty  # Py3XX: delete line
    def state(self):
        """Returns the current state of the chain.

        The state consists of everything needed such that setting a chain's
        state using another's state will result in identical results.
        """
        pass

    @abstractmethod
    def set_state(self, state):
        """Sets the state of the chain using the given dict.

        .. warning::
           Running this will clear the chain's current memory, and replace its
           current position with what is saved in the state.

        Parameters
        ----------
        state : dict
            Dictionary of state values.
        """
        pass

    @abstractmethod
    def step(self):
        """Evolves the chain by a single step."""
        pass
