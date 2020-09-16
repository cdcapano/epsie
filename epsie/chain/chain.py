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

"""Classes for individual Markov chains."""

from __future__ import absolute_import

import numpy

from epsie.proposals import JointProposal

from .base import BaseChain
from .chaindata import (ChainData, detect_dtypes)


class Chain(BaseChain):
    """A Markov chain.

    The chain requires a ``model`` to  be provided. This can be any object that
    can be called with keyword arguments that map parameter names to values.
    When called, the object must return a tuple of two or three elements. The
    first element must be a float representing the log likelihood at that
    point; the second must be a float representing the log prior at that point.
    Additionally, the model may return a dictionary as a third element in the
    tuple that maps strings to arbitrary data. This additional data will be
    stored as ``blob`` data.

    Parameters
    ----------
    parameters : list or tuple
        List of the parameter names to sample.
    model : object
        Any object that can be called with keyword arguments that map parameter
        names to values. When called, the object must a tuple of ``logl, logp``
        where ``logp`` is the log prior and ``logl`` is the log likelihood at
        the given point. The model may optionally return a dictionary in
        addition that maps strings to any arbitrary data.
    proposals : list of epsie.proposals instances
        List of proposal classes to use for the parameters. There must be one
        and only one proposal for every parameter. A single proposal may cover
        multiple parameters. Proposals must be instances of classes that
        inherit from :py:class:`epsie.proposals.BaseProposal`.
    bit_generator : :py:class:`epsie.BIT_GENERATOR` instance, optional
        Use the given random bit generator for generating random variates. If
        an int or None is provided, a generator will be created instead using
        ``bit_generator`` as a seed.
    chain_id : int, optional
        An interger identifying which chain this is. Default is 0.
    beta : float, optional
        The inverse temperature of the chain. Default is 1.

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
    transdimensional
    chain_id : int or None
        Integer identifying the chain.
    proposal_dist : JointProposal
        The joint proposal used for all parameters.
    """
    def __init__(self, parameters, model, proposals, bit_generator=None,
                 chain_id=0, beta=1.):
        self.parameters = parameters
        self.model = model

        self.transdimensional = None
        self.proposal_dist = self._store_proposals(*proposals,
                                                   bit_generator=bit_generator)
        # store the temp
        self.beta = beta
        self.chain_id = chain_id
        self._iteration = 0
        self._lastclear = 0
        self._scratchlen = 0
        self._positions = ChainData(parameters)
        self._stats = ChainData(['logl', 'logp'])
        self._acceptance = ChainData(['acceptance_ratio', 'accepted'],
                                     dtypes={'accepted': bool})
        self._blobs = None
        self._hasblobs = False
        self._start = None
        self._proposed_position = None
        self._logp0 = None
        self._logl0 = None
        self._blob0 = None

    def _store_proposals(self, *proposals, **kwargs):
        """Store either a ``JointProposal`` or the transdimensional proposal
        """
        bit_generator = kwargs.pop('bit_generator', None)  # Py3XX: delete line
        count = 0
        for prop in proposals:
            try:
                if prop.transdimensional:
                    self.transdimensional = True
                    count += 1
            except AttributeError:
                prop.transdimensional = False

        if self.transdimensional:
            if count > 1:
                raise ValueError("Can only provide a single transdimensinal "
                                 "proposals that includes the constituent "
                                 "proposals")
        return JointProposal(*proposals, bit_generator=bit_generator)

    @property
    def hasblobs(self):
        """Whether the model returns blobs."""
        return self._hasblobs

    @property
    def iteration(self):
        """The number of times the chain has been stepped."""
        return self._iteration

    @property
    def lastclear(self):
        """Returns the iteration of the last time the chain memory was cleared.
        """
        return self._lastclear

    @property
    def scratchlen(self):
        """The length of the scratch space used."""
        return self._scratchlen

    @scratchlen.setter
    def scratchlen(self, n):
        """Set the scratch length to the given value.

        This will immediately increase the scratch length to ``n``. If the
        chain is already longer than ``n``, this will have no immediate impact.
        However, the next time ``clear`` is called, the scratch length will
        be reset to ``n``.

        Parameters
        ----------
        n : int
            The length to set.
        """
        self._scratchlen = n
        try:
            self._positions.set_len(n)
        except ValueError:
            pass
        try:
            self._stats.set_len(n)
        except ValueError:
            pass
        try:
            self._acceptance.set_len(n)
        except ValueError:
            pass
        if self.hasblobs:
            try:
                self._blobs.set_len(n)
            except ValueError:
                pass

    @property
    def start_position(self):
        """Dictionary mapping parameters to their start position.

        If the start position hasn't been set, raises a ``ValueError``.
        """
        if self._start is None:
            raise ValueError("Starting position not set!")
        return self._start

    @start_position.setter
    def start_position(self, position):
        """Sets the starting position.

        This also evaulates the log likelihood and log prior at the starting
        position, as well as determine if the model returns blob data.

        Parameters
        ----------
        position : dict
            Dictionary mapping parameters to values.
        """
        self._start = position.copy()
        # use start position to determine the dtype for positions
        self._positions.dtypes = detect_dtypes(position)
        # evaluate logl, p at this point
        r = self.model(**position)
        # if transdimensional decide which are active and store
        if self.transdimensional:
            self._activate_proposals()
        try:
            logl, logp, blob = r
            self._hasblobs = True
        except ValueError:
            logl, logp = r
            blob = None
            self._hasblobs = False
        if self._hasblobs:
            if not isinstance(blob, dict):
                raise TypeError("model must return blob data as a dictionary")
            self._blobs = ChainData(blob.keys(), dtypes=detect_dtypes(blob))
        self.stats0 = {'logl': logl, 'logp': logp}
        self.blob0 = blob

    def _activate_proposals(self):
        """Decide which proposals are initially active"""
        # find the transdimensional proposal
        for prop in self.proposal_dist.proposals:
            if prop.transdimensional:
                break
        # cast this to int as the discrete jump cannot accept ndarray
        self._start[prop._index] = int(self._start[prop._index])
        # activate proposals that do not have nans
        for ptd in prop.proposals:
            if all(numpy.isnan([self._start[p] for p in ptd.parameters])):
                ptd.active = False
            else:
                ptd.active = True
            if ptd.birth_distribution is None:
                raise ValueError("must provide `birth distribution` "
                                 "for transdimensional proposals")

        self._active_props = numpy.array([p.active for p in prop.proposals])

    @property
    def stats0(self):
        """Dictionary of the log likelihood and log prior at the start
        position.

        Raises a ``ValueError`` if the start position has not been set yet.
        """
        # check if the start position is set
        _ = self.start_position  # raises a ValueError if not set
        return self._stats0

    @stats0.setter
    def stats0(self, stats):
        """Sets the starting stats.

        Parameters
        ----------
        dict, numpy structred array of len 1, or numpy.void
            Dictionary or numpy array/void object mapping parameter names to
            the starting statistics.
        """
        # check that we're not starting outside of the prior
        if stats['logp'] == -numpy.inf:
            raise ValueError("starting position is outside of the prior!")
        self._stats0 = stats.copy()

    @property
    def blob0(self):
        """The blob data of the starting position, as a dictionary.

        Raises a ``ValueError`` if ``set_start`` has not been run yet.
        """
        # check if the start position is set
        _ = self.start_position  # raises a ValueError if not set
        if self._blob0 is None:
            blob = None
        else:
            blob = self._blob0
        return blob

    @blob0.setter
    def blob0(self, blob):
        """Sets the starting blob.

        Parameters
        ----------
        blob: dict
            Dictionary mapping blob parameters to their values.
        """
        if blob is not None:
            blob = blob.copy()
            # create scratch for blobs
            if self._blobs is None:
                self._blobs = ChainData(blob.keys(),
                                        dtypes=detect_dtypes(blob))
        self._blob0 = blob

    @property
    def positions(self):
        """The history of all of the positions, as a structred array."""
        if self.iteration == 0:
            raise ValueError("No positions as chain hasn't been stepped "
                             "yet; run step() at least once")
        return self._positions[:len(self)]

    @property
    def stats(self):
        """The log likelihoods and log priors of the positions, as a structured
        array.
        """
        if self.iteration == 0:
            raise ValueError("No stats as chain hasn't been stepped "
                             "yet; run step() at least once")
        return self._stats[:len(self)]

    @property
    def acceptance(self):
        """The history of all of acceptance ratios and accepted booleans,
        as a structred array.
        """
        if self.iteration == 0:
            raise ValueError("No acceptance as chain hasn't been stepped "
                             "yet; run step() at least once")
        return self._acceptance[:len(self)]

    @property
    def blobs(self):
        """The history of all of the blob data, as a structured array.

        If the model does not return blobs, this is just ``None``.
        """
        if self.iteration == 0:
            raise ValueError("No blobs as chain hasn't been stepped "
                             "yet; run step() at least once")
        if self.hasblobs:
            blobs = self._blobs[:len(self)]
        else:
            blobs = None
        return blobs

    @property
    def current_position(self):
        """Dictionary of the current position of the chain."""
        if len(self) == 0:
            pos = self.start_position
        else:
            pos = self._positions.asdict(len(self)-1)
        return pos

    @property
    def proposed_position(self):
        """Dictionary of the current proposed position of the chain."""
        if self.iteration == 0:
            raise ValueError("No proposed position as chain hasn't been "
                             "stepped yet; run step() at least once")
        return self._proposed_position

    @proposed_position.setter
    def proposed_position(self, proposed_position):
        """Sets the current proposed position of the chain."""
        self._proposed_position = proposed_position

    @property
    def current_stats(self):
        """Dictionary giving the log likelihood and log prior of the current
        position.
        """
        if len(self) == 0:
            stats = self.stats0
        else:
            stats = self._stats.asdict(len(self)-1)
        return stats

    @property
    def current_blob(self):
        """Dictionary of the blob data of the current position.

        If the model does not return blobs, just returns ``None``.
        """
        if not self.hasblobs:
            blob = None
        elif len(self) == 0:
            blob = self.blob0
        else:
            blob = self._blobs.asdict(len(self)-1)
        return blob

    def clear(self):
        """Clears memory of the current chain, and sets start position to the
        current position.

        New scratch space will be created with length equal to ``scratch_len``.
        """
        if self._iteration > 0:
            # save position then clear
            self._start = self.current_position
            self._positions.clear(self.scratchlen)
            # save stats then clear
            self._stats0 = self.current_stats
            self._stats.clear(self.scratchlen)
            # clear acceptance info
            self._acceptance.clear(self.scratchlen)
            # save blobs then clear
            if self.hasblobs:
                self._blob0 = self.current_blob
                self._blobs.clear(self.scratchlen)
        self._lastclear = self._iteration
        return self

    def __getitem__(self, index):
        """Returns all of the chain data at the requested index."""
        index = (-1)**(index < 0) * (index % len(self))
        out = {'positions': self._positions[index],
               'stats': self._stats[index],
               'acceptance': self._acceptance[index]
               }
        if self._hasblobs:
            out['blobs'] = self._blobs[index]
        return out

    @property
    def bit_generator(self):
        """The random bit generator being used."""
        return self.proposal_dist.bit_generator

    @property
    def random_generator(self):
        """Returns the random number generator."""
        return self.proposal_dist.random_generator

    @property
    def random_state(self):
        """Returns the current state of the bit generator."""
        return self.proposal_dist.random_state

    @property
    def state(self):
        """Returns the current state of the chain.

        The state consists of everything needed such that setting a chain's
        state using another's state will result in identical results.
        """
        state = {}
        state['chain_id'] = self.chain_id
        state['proposal_dist'] = self.proposal_dist.state
        state['iteration'] = self.iteration
        state['current_position'] = self.current_position
        state['proposed_position'] = self.proposed_position
        state['current_stats'] = self.current_stats
        state['hasblobs'] = self.hasblobs
        if self.hasblobs:
            blob = self.current_blob
        else:
            blob = None
        state['current_blob'] = blob
        return state

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
        self.chain_id = state['chain_id']
        # set the chain position
        self.clear()
        self._iteration = state['iteration']
        self._lastclear = state['iteration']
        self._start = state['current_position'].copy()
        self.stats0 = state['current_stats']
        self._hasblobs = state['hasblobs']
        self.blob0 = state['current_blob']
        self.proposed_position = state['proposed_position']
        # set the proposals' states
        self.proposal_dist.set_state(state['proposal_dist'])
        # set the positions` dtypes to match the starting point
        self._positions.dtypes = detect_dtypes(self._start)
        if self.transdimensional:
            self._activate_proposals()

    def _acceptance_ratio(self, logp, logl, proposal,
                          current_logp, current_logl, current_pos):
        """Calculates the acceptance ratio and evaluates acceptance"""
        logar = (logp + logl * self.beta
                 - current_logp - current_logl * self.beta)
        if not self.proposal_dist.symmetric:
            logar += self.proposal_dist.logpdf(current_pos, proposal) - \
                     self.proposal_dist.logpdf(proposal, current_pos)
        if logar > 0:
            ar = 1.
            accept = True
        else:
            ar = numpy.exp(logar)
            u = self.random_generator.uniform()
            accept = u <= ar
        return accept, ar


    def step(self):
        """Evolves the chain by a single step."""
        # get the current position; if this is the first step and set_start
        # hasn't been run, this will raise a ValueError
        current_pos = self.current_position
        current_stats = self.current_stats
        current_blob = self.current_blob
        # transdimensional proposals need to know which proposals are active
        if self.transdimensional:
            current_pos.update({'_state': self._active_props})
        # create a proposal and test it
        proposal = self.proposal_dist.jump(current_pos)
        self.proposed_position = proposal.copy()

        r = self.model(**proposal)
        if self._hasblobs:
            logl, logp, blob = r
        else:
            logl, logp = r
            blob = None
        # evaluate
        current_logl = current_stats['logl']
        current_logp = current_stats['logp']
        if logp == -numpy.inf:
            # force a reject
            accept = False
            ar = 0.
        else:
            accept, ar = self._acceptance_ratio(logp, logl, proposal,
                                                current_logp, current_logl,
                                                current_pos)
        if accept:
            if self.transdimensional:
                self._active_props = proposal.pop('_state')
            pos = proposal
            stats = {'logl': logl, 'logp': logp}
        else:
            if self.transdimensional:
                __ = current_pos.pop('_state')
            # reject
            pos = current_pos
            stats = current_stats
            blob = current_blob
        # save
        index = len(self)
        self._positions[index] = pos
        self._stats[index] = stats
        self._acceptance[index] = (ar, accept)
        if self._hasblobs:
            self._blobs[index] = blob
        self._iteration += 1
        # in case the any of the proposals need information about the history
        # of the chain:
        self.proposal_dist.update(self)
        return self
