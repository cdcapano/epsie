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
from scipy.stats import uniform as randuniform

from epsie import array2dict
from epsie.proposals import JointProposal

from .chaindata import (ChainData, detect_dtypes)


class Chain(object):
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
    betas : array of floats, optional
        Array of inverse temperatures. Each beta must be in range 0 (= infinite
        temperature; i.e., only sample the prior) <= beta <= 1 (= coldest
        temperate; i.e., sample the standard posterior). Default is a single
        beta = 1.
    swap_interval : int, optional
        For a parallel tempered chain, how often to calculate temperature
        swaps. Default is 1 (= swap on every iteration).
    brng : :py:class:`randomgen.PGC64` instance, optional
        Use the given basic random number generator (BRNG) for generating
        random variates. If an int or None is provided, a BRNG will be
        created instead using ``brng`` as a seed.
    chain_id : int, optional
        An interger identifying which chain this is. Optional; if not provided,
        the ``chain_id`` attribute will just be set to None.
    """
    def __init__(self, parameters, model, proposals, betas=1., swap_interval=1,
                 brng=None, chain_id=None):
        self.parameters = parameters
        self.model = model
        # combine the proposals into a joint proposal
        self.proposal_dist = JointProposal(*proposals, brng=brng)
        # store the temp
        self._betas = None
        self.betas = betas
        self.chain_id = chain_id
        self._iteration = 0
        self._lastclear = 0
        self._scratchlen = 0
        self._positions = ChainData(parameters, ntemps=self.ntemps)
        self._stats = ChainData(['logl', 'logp'], ntemps=self.ntemps)
        self._acceptance = ChainData(['acceptance_ratio', 'accepted'],
                                     ntemps=self.ntemps,
                                     dtypes={'accepted': bool})
        # for parallel tempering, store an array giving the acceptance ratio
        # between temps and whether or not they swapped
        self.swap_interval = swap_interval
        self._temperature_swaps = None
        if self.ntemps > 1:
            # we pass ntemps=ntemps-1 here because there will be ntemps-1
            # acceptance ratios for ntemp levels
            self._temperature_swaps = ChainData(
                ['acceptance_ratio', 'swap_index'],
                dtypes={'acceptance_ratio': float, 'swap_index': int},
                ntemps=self.ntemps-1)
        self._blobs = None
        self._hasblobs = False
        self._start = None
        self._logp0 = None
        self._logl0 = None
        self._blob0 = None

    def __len__(self):
        return self._iteration - self._lastclear

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
    def betas(self):
        """Returns the betas (=1 / temperatures) used by the chain."""
        return self._betas

    @betas.setter
    def betas(self, betas):
        """Checks that the betas are in the allowed range before setting."""
        if isinstance(betas, (float, int)):
            # only single temperature; turn into list so things below won't
            # break
            betas = [betas]
        if not isinstance(betas, numpy.ndarray):
            # betas is probably a list or tuple; convert to array so we can use
            # numpy functions
            betas = numpy.array(betas)
        if not (betas == 1.).any():
            logging.warn("No betas = 1 found. This means that the normal "
                         "posterior (i.e., likelihood * prior) will not be "
                         "sampled by the chain.")
        # check that all betas are in [0, 1]
        if not ((0 <= betas) & (betas <= 1)).all():
            raise ValueError("all betas must be in range [0, 1]")
        # sort from coldest to hottest and store
        self._betas = numpy.sort(betas)[::-1]  # note: this copies the betas

    @property
    def temperatures(self):
        """Returns the temperatures (= 1 / betas) used by the chain."""
        return 1./self._betas

    @property
    def ntemps(self):
        """Returns the number of temperatures used by the chain."""
        return len(self.betas)

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
        if self.ntemps > 1:
            try:
                self._temperature_swaps.set_len(n)
            except ValueError:
                pass
        if self.hasblobs:
            try:
                self._blobs.set_len(n)
            except ValueError:
                pass

    def set_start(self, position):
        """Sets the starting position.

        This also evaulates the log likelihood and log prior at the starting
        position, as well as determine if the model returns blob data.

        Parameters
        ----------
        position : dict
            Dictionary mapping parameters to values. If ntemps > 1, values
            should be numpy arrays with length = ntemps. Otherwise, these
            should be atomic data typees..
        """
        self.start_position = position
        # Use the coldest temp to determine if have blobs
        if self.ntemps > 1:
            index = (0, 0)
        else:
            index = 0
        r = self.model(**array2dict(self._start[index]))
        try:
            logl, logp, blob = r
            self._hasblobs = True
        except ValueError:
            logl, logp = r
            blob = None
            self._hasblobs = False
        # create dict to store the stats/blobs at each temperature
        stats = {'logl': numpy.zeros(self.ntemps),
                 'logp': numpy.zeros(self.ntemps)}
        stats['logl'][0] = logl
        stats['logp'][0] = logp
        if self._hasblobs:
            if not isinstance(blob, dict):
                raise TypeError("model must return blob data as a dictionary")
            blob0 = ChainData(blob.keys(), dtypes=self._blobs.dtypes,
                              ntemps=self.ntemps)
            blob0.extend(1)
            blob0[..., 0] = blob
        # Evaluate the rest of the temperatures
        for tk in range(1, self.ntemps):
            r = self.model(**array2dict(self._start[0, tk]))
            if self.hasblobs:
                logl, logp, blob = r
                self._blob0[0, tk] = blob
            else:
                logl, logp = r
            stats['logl'][tk] = logl
            stats['logp'][tk] = logp
        # store
        self.stats0 = stats
        if self._hasblobs:
            self.blob0 = blob0.asdict(0)

    @property
    def start_position(self):
        """The starting position.

        Raises a ``ValueError`` if ``set_start`` has not been run yet.
        """
        if self._start is None:
            raise ValueError("Starting position not set! Run set_start.")
        return self._start[0]

    @start_position.setter
    def start_position(self, position):
        """Sets the start position.
        """
        # we'll store current positions as 1-iteration ChainData, since this
        # makes it easy to deal with 1 or more temps
        self._start = ChainData(self.parameters,
                                dtypes=detect_dtypes(position),
                                ntemps=self.ntemps)
        # use detected dtypes to set the dtype of the full chain
        self._positions.dtypes = self._start.dtypes
        # note: if only a single value is given for each parameter, this will
        # expand the values to the number of temps
        self._start[0] = position

    @property
    def stats0(self):
        """The log likelihood and log prior of the starting position.
        
        Raises a ``ValueError`` if ``set_start`` has not been run yet.
        """
        if self._start is None:
            raise ValueError("Starting position not set! Run set_start.")
        return self._stats0[0]

    @stats0.setter
    def stats0(self, stats):
        """Sets the starting stats.

        Parameters
        ----------
        dict, numpy structred array of len 1, or numpy.void
            Dictionary or numpy array/void object mapping parameter names to
            the starting statistics.
        """
        self._stats0 = ChainData(['logl', 'logp'], ntemps=self.ntemps)
        self._stats0[0] = stats
        # check that we're not starting outside of the prior
        if numpy.array(self._stats0[0]['logp'] == -numpy.inf).any():
            raise ValueError("starting position is outside of the prior!")

    @property
    def blob0(self):
        """The blob data of the starting position.
        
        Raises a ``ValueError`` if ``set_start`` has not been run yet.
        """
        if self._start is None:
            raise ValueError("Starting position not set! Run set_start.")
        if self._blob0 is None:
            blob = None
        else:
            blob = self._blob0[0]
        return blob

    @blob0.setter
    def blob0(self, blob):
        """Sets the starting blob.

        Parameters
        ----------
        dict, numpy structred array of len 1, or numpy.void
            Dictionary or numpy array/void object mapping parameter names to
            the starting blob parameters.
        """
        if blob is None:
            self._blob0 = blob
        else:
            self._blob0 = ChainData(blob.keys(), dtypes=detect_dtypes(blob),
                                    ntemps=self.ntemps)
            # create scratch for blobs
            if self._blobs is None:
                self._blobs = ChainData(blob.keys(),
                                        dtypes=detect_dtypes(blob),
                                        ntemps=self.ntemps)
            self._blob0[0] = blob

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
    def temperature_swaps(self):
        """The history of all of the temperature swaps."""
        return self._temperature_swaps[:len(self)]

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

    def clear(self):
        """Clears memory of the current chain, and sets start position to the
        current position.
        
        New scratch space will be created with length equal to ``scratch_len``.
        """
        if self._iteration > 0:
            # save position then clear
            self._start[0] = self.current_position
            self._positions.clear(self.scratchlen)
            # save stats then clear
            self._stats0[0] = self.current_stats
            self._stats.clear(self.scratchlen)
            # clear acceptance info
            self._acceptance.clear(self.scratchlen)
            # clear temperature swaps
            if self.ntemps > 1:
                self._temperature_swaps.clear(self.scratchlen)
            # save blobs then clear
            if self._hasblobs:
                self._blob0[0] = self.current_blob
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
        if self.ntemps > 1:
            out['temperature_swaps'] = self._temperature_swaps[index]
        if self._hasblobs:
            out['blobs'] = self._blobs[index]
        return out

    @property
    def brng(self):
        """Returns basic random number generator (BRNG) being used."""
        return self.proposal_dist.brng

    @property
    def random_generator(self):
        """Returns the random number generator."""
        return self.brng.generator

    @property
    def random_state(self):
        """Returns the current state of the BRNG."""
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
        state['current_position'] = array2dict(self.current_position)
        state['current_stats'] = array2dict(self.current_stats)
        state['hasblobs'] = self._hasblobs
        if self._hasblobs:
            blob = array2dict(self.current_blob)
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
        self.start_position = state['current_position']
        self.stats0 = state['current_stats']
        self._hasblobs = state['hasblobs']
        self.blob0 = state['current_blob']
        # set the proposals' states
        self.proposal_dist.set_state(state['proposal_dist'])


    def step(self):
        """Evolves the chain by a single step."""
        # in case the any of the proposals need information about the history
        # of the chain:
        self.proposal_dist.update(self)
        # get the current position; if this is the first step and set_start
        # hasn't been run, this will raise a ValueError
        # We'll ensure that the returned valued is an array of length 1,
        # so tehf loop below will work
        current_pos = self.current_position.reshape(self.ntemps)
        current_stats = self.current_stats.reshape(self.ntemps)
        if self.hasblobs:
            current_blob = self.current_blob.reshape(self.ntemps)
        else:
            current_bob = None
        ii = len(self)
        for tk in range(self.ntemps):
            if self.ntemps > 1:
                index = (self._iteration, tk)
            else:
                index = self._iteration
            pos, stats, blob, ar, accepted = self._singletemp_step(
                array2dict(current_pos[tk]),
                array2dict(current_stats[tk]),
                array2dict(current_blob[tk]) if self._hasblobs else None,
                self.betas[tk])
            # save
            if self.ntemps > 1:
                index = (ii, tk)
            else:
                index = ii
            self._positions[index] = pos
            self._stats[index] = stats
            self._acceptance[index] = (ar, accepted)
            if self._hasblobs:
                self._blobs[index] = blob
        # do temperature swaps
        if self.ntemps > 1 and ii % self.swap_interval == 0:
            self.swap_temperatures(ii)
        self._iteration += 1
        return self

    def _singletemp_step(self, current_pos, current_stats, current_blob, beta):
        """Steps a single temperature."""
        proposal = self.proposal_dist.jump(current_pos)
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
            ar = 0.
        else:
            logar = logp + logl * beta \
                    - current_logl * beta - current_logp
            if not self.proposal_dist.symmetric:
                logar += self.proposal_dist.logpdf(current_pos, proposal) - \
                         self.proposal_dist.logpdf(proposal, current_pos)
            ar = numpy.exp(logar)
        u = self.random_generator.uniform()
        accept = u <= ar
        if accept:
            pos = proposal
            stats = {'logl': logl, 'logp': logp}
        else:
            # reject
            pos = current_pos
            stats = current_stats
            blob = current_blob
        return pos, stats, blob, ar, accept

    def swap_temperatures(self, ii):
        """Computes acceptance ratio between temperatures and swaps them.

        Parameters
        ----------
        ii : int
            The iteration to do the swaps on.
        """
        # get values of all temps at current step
        position = self._positions[ii, :]
        stats = self._stats[ii, :]
        if self._hasblobs:
            blob = self._blobs[ii, :]
        # we'll create an array of indices to keep track of where things
        # will go after all of the swaps have been done
        swap_index = numpy.arange(self.ntemps, dtype=int)
        # swaps are determined by finding acceptance ratio:
        # A_jk = min( (L_j/L_k)^(beta_k - beta_j), 1)
        # We start with the hottest chain:
        loglk = stats['logl'][-1]
        # to store acceptance ratios and swaps
        ars = numpy.zeros(self.ntemps-1)
        # since stored coldest to hottest, the folling = beta_k - beta_j
        dbetas = numpy.diff(self.betas)
        # now cycle down through the temps, comparing the current one
        # to the one below it
        for tk in range(self.ntemps-1, 0, -1):
            swk = swap_index[tk]
            tj = tk - 1
            loglj = stats['logl'][tj]
            swj = swap_index[tj]
            ar = numpy.exp(dbetas[tj]*(loglj - loglk))
            u = self.random_generator.uniform()
            swap = u <= ar
            if swap:
                # move the colder index into the current slot...
                swap_index[tk] = swj
                # ...and the hotter index into the colder slot
                swap_index[tj] = swk
                # we won't change loglk so that it will get compared to
                # next coldest temperature on the next loop
            else:
                # don't swap, so drop the current loglk here, and pick up
                # the colder logl to compare on the next loop
                loglk = loglj
            # store the acceptance ratio
            ars[tj] = ar
        # now do the swaps and store
        self._positions[ii] = position[swap_index]
        self._stats[ii] = stats[swap_index]
        if self._hasblobs:
            self._blobs[ii] = blob[swap_index]
        # since we have ntemps-1 acceptance ratios, we won't store the
        # hottest swap index, since it can be inferred from the other
        # swap indices
        self._temperature_swaps[ii] = {'acceptance_ratio': ars,
                                       'swap_index': swap_index[:-1]}


