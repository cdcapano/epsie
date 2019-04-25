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


class BaseChain(object):
    """Abstract base class for Markov chains.

    Provides standard functions for Chain and ParallelTemperedChain.

    Attributes
    ----------
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
    brng
    random_state
    state
    hasblobs
    """
    __metaclass__ = ABCMeta

    _positions = None
    _stats = None
    _acceptance = None
    _blobs = None
    _hasblobs = False

    def __len__(self):
        return self.iteration - self.lastclear

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
    def start_position(self, position):
        """The start position."""
        pass

    # Py3XX: uncomment the next two lines 
    # @property
    # @abstractmethod
    @abstractproperty  # Py3XX: delete line
    def stats0(self):
        """The log likelihood and log prior of the starting position."""
        pass

    # Py3XX: uncomment the next two lines 
    # @property
    # @abstractmethod
    @abstractproperty  # Py3XX: delete line
    def blob0(self):
        """The blob data of the starting position."""
        pass

    @property
    def blobs(self):
        """The history of all of the blob data.

        If the model does not return blobs, this is just ``None``.
        """
        blobs = self._blobs
        if blobs is not None:
            blobs = blobs[:len(self)]
        return blobs

    @property
    def hasblobs(self):
        """Whether the model returns blobs."""
        return self._hasblobs

    @property
    def positions(self):
        """The history of all of the positions."""
        return self._positions[:len(self)]

    @property
    def stats(self):
        """The log likelihoods and log priors of the positions."""
        return self._stats[:len(self)]

    @property
    def acceptance(self):
        """The history of all of acceptance ratios and accepted booleans."""
        return self._acceptance[:len(self)]

    @property
    def blobs(self):
        """The history of all of the blob data.

        If the model does not return blobs, this is just ``None``.
        """
        blobs = self._blobs
        if blobs is not None:
            blobs = blobs[:len(self)]
        return blobs

    @property
    def hasblobs(self):
        """Whether the model returns blobs."""
        return self._hasblobs

    @property
    def current_position(self):
        """The current position of the chain."""
        if len(self) == 0:
            pos = self.start_position
        else:
            pos = self._positions[len(self)-1]
        return pos

    @property
    def current_stats(self):
        """The log likelihood and log prior of the current position."""
        if len(self) == 0:
            stats = self.stats0
        else:
            stats = self._stats[len(self)-1]
        return stats

    @property
    def current_blob(self):
        """The blob data of the current position.

        If the model does not return blobs, just returns ``None``.
        """
        if not self._hasblobs:
            blob = None
        elif len(self) == 0:
            blob = self.blob0
        else:
            blob = self._blobs[len(self)-1]
        return blob

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
    def brng(self):
        """Returns basic random number generator (BRNG) being used."""
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
        """Returns the current state of the BRNG."""
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
