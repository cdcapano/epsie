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
from six import add_metaclass

import numpy
try:
    from randomgen import RandomGenerator
except ImportError:
    from randomgen import Generator as RandomGenerator
from scipy import stats

import epsie


@add_metaclass(ABCMeta)
class BaseProposal(object):
    """Abstract base class for all proposal classes.

    All proposals must inherit from this class, as it handles random number
    generation, and lays out all of the functions/attributes that a sampler
    will try to access.

    .. warning ::
        All proposals must use the random number generator provided by this
        class (see the ``random_generator`` attribute) for creating random
        numbers. **Do not attempt to use scipy/numpy's random number
        generator.** Use of any other generator may result in chains not being
        independent of each other when run in a parallel environment.

    In addition to the abstract methods/properties, all samplers must set
    a ``parameters`` attribute. This is a list of the names of the parameters
    the proposal produces jumps for.

    Attributes
    ----------
    bit_generator
    random_generator
    random_state
    parameters
    symmetric
    state
    """
    name = None
    _parameters = None

    @property
    def bit_generator(self):
        """The random bit generator instance being used.

        A bit generator will be created if it doesn't exist yet.
        """
        try:
            return self._bit_generator
        except AttributeError:
            self._bit_generator = epsie.create_bit_generator()
            return self._bit_generator

    @bit_generator.setter
    def bit_generator(self, bit_generator):
        """Sets the random bit generator.

        Parameters
        ----------
        bit_generator : :py:class:`epsie.BIT_GENERATOR`, int, or None
            Either the bit generator to use or an integer/None. If the latter,
            a generator will be created by passing ``bit_generator`` as the
            ``seed`` argument to :py:func:`epsie.create_bit_generator`.
        """
        if not isinstance(bit_generator, epsie.BIT_GENERATOR):
            bit_generator = epsie.create_bit_generator(bit_generator)
        self._bit_generator = bit_generator

    @property
    def random_generator(self):
        """The random number generator.

        This is an instance of :py:class:`randgen.RandomGenerator` that is
        derived from the bit generator. It provides methods to create random
        draws from various distributions.
        """
        return RandomGenerator(self.bit_generator)

    @property
    def random_state(self):
        """The current state of the random bit generator.
        """
        return self.bit_generator.state

    @random_state.setter
    def random_state(self, state):
        """Sets the state of bit_generator.
        
        Parameters
        ----------
        state : dict
            Dictionary giving the state to set.
        """
        self.bit_generator.state = state

    @property
    def parameters(self):
        """Sorted tuple of the parameters that proposals are produced for."""
        if self._parameters is None:
            raise AttributeError("no parameters set")
        return self._parameters

    @parameters.setter
    def parameters(self, parameters):
        """Sets the parameters.

        The parameters are stored as a sorted tuple.

        Parameters
        ----------
        parameters : (list of) str
            The names of the parameters. This may either be a list of strings,
            or (for a single parameter), a string.
        """
        if not isinstance(parameters, (list, tuple, numpy.ndarray)):
            parameters = [parameters]
        self._parameters = tuple(sorted(parameters))

    # Py3XX: uncomment the next two lines
    # @property
    # @abstractmethod
    @abstractproperty  # Py3XX: delete line
    def symmetric(self):
        """Boolean indicating whether the proposal distribution is symmetric.

        A jump proposal is symmetric if the proposal probability
        density has the property that :math:`p(x'|x) = p(x|x')`,
        where :math:`x` is the current position, :math:`x'` is the proposed
        position. In this, case, the Metropolis-Hastings ratio simplfies to
        just the ratio of posterior values at the two points.

        Note that an adaptive proposal may still be symmetric. Symmetry only
        means that the current state of the proposal satisfies the above
        condition; it does not necessarily mean that transitions between states
        be symmetric.
        """
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
        """Update the state of the proposal distribution after a jump.

        This method may optionally be implemented by a proposal. It is called
        by the Markov chains just after a jump is evaluated. It can be used by,
        e.g., adaptive jump proposals that change their state depending on
        the history of the chain.
        """
        pass
