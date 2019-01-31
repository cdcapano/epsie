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

"""Classes for parallel tempered Markov chains."""

from __future__ import absolute_import

import numpy
import logging
import copy

import epsie
from epsie import create_seed, create_brngs
from epsie.chain import Chain
from epsie.proposals import Normal


class ParallelTemperedSampler(object):
    """Evolves a collection of parallel tempered Markov chains.

    Parameters
    ----------
    parameters : tuple or list
        Names of the parameters to sample.
    model : object
        Model object.
    nchains : int
        The number of chains to create per temperature. Must be greater than
        zero.
    betas : numpy.ndarray of floats
        The betas (= 1 / temperatures) to use. All betas must be between [0,1].
        Must provide at least one. If one is not included in the betas, a
        warning will be printed.
    proposals : dict, optional
        Dictionary mapping parameter names to proposal classes. Any parameters
        not listed will use the ``default_propsal``.
    default_proposal : an epsie.Proposal class, optional
        The default proposal to use for parameters not in ``proposals``.
        Default is :py:class:`epsie.proposals.Normal`.
    seed : int, optional
        Seed for the random number generator. If None provided, will create
        one.
    pool : Pool object, optional
        Specify a process pool to use for parallelization. Default is to use a
        single core.
    """
    def __init__(self, parameters, model, nchains, betas, proposals=None,
                 default_proposal=None, seed=None, pool=None):
        self.parameters = tuple(parameters)
        self.model = model
        if nchains < 1:
            raise ValueError("nchains must be >= 1")
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
                         "sampled by any chain.")
        if proposals is None:
            proposals = {}
        # create default proposal instances for the other parameters
        if default_proposal is None:
            default_proposal = Normal
        missing_props =  tuple(set(parameters) - set(proposals.keys()))
        proposals[missing_props] = default_proposal(missing_props)
        self.proposals = proposals
        # create the random number states
        if seed is None:
            seed = create_seed(seed)
        self.seed = seed
        # create the chains
        self.chains = [Chain(self.parameters, self.model,
                             [copy.copy(p) for p in self.proposals.values()],
                             betas=betas,
                             brng=epsie.create_brng(self.seed, stream=cid),
                             chain_id=cid)
                        for cid in range(nchains)]
        # set the mapping function
        if pool is None:
            self.map = map
        else:
            self.map = pool.map

    @property
    def nchains(self):
        """The number of chains being used."""
        return len(self.chains)

    @property
    def betas(self):
        """The betas used."""
        return self.chains[0].betas

    @property
    def ntemps(self):
        """Returns the number of temperatures used."""
        return len(self.betas)

    @property
    def temperatures(self):
        """The temperatures used."""
        return 1./betas

    @property
    def niterations(self):
        """The number of iterations the chains have been run for.
        """
        # all of the chains should be at the same iteration, so just use the
        # first one
        return self.chains[0].iteration

    @property
    def swap_interval(self):
        """Returns the swap interval used for parallel tempering."""
        return self.chains[0].swap_interval

    @swap_interval.setter
    def swap_interval(self, interval):
        """Sets the swap interval to use."""
        for c in self.chains:
            c.swap_interval = interval

    def set_start(self, positions):
        """Sets the starting position of all of the chains.

        Parameters
        ----------
        positions : dict
            Dictionary mapping parameter names to arrays of values. The chains
            must have length equal to the number of chains.
        """
        for (ii, chain) in enumerate(self.chains):
            chain.set_start({p: positions[p][ii, ...] for p in positions})

    @property
    def start_position(self):
        """The start position of the chains.
        
        Will raise a ``ValueError`` is ``set_start`` hasn't been run.

        Returns
        -------
        dict :
            Dictionary mapping parameters to arrays with shape ``nchains``
            giving the starting position of each chain.
        """
        return {p: self.concatenate_chains('start_position', p)
                for p in self.parameters}

    def run(self, niterations):
        """Evolves all of the chains by niterations.

        Parameters
        ----------
        niterations : int
            The number of iterations to evolve the chains for.
        """
        # extend the scratch space for the chains
        for c in self.chains:
            c.scratchlen += max(niterations - (c.scratchlen - len(c)), 0)
        # construct arguments for passing to the pool
        args = zip([niterations]*len(self.chains), self.chains)
        self.chains = self.map(_evolve_chain, args)

    def clear(self):
        """Clears all of the chains."""
        for chain in self.chains:
            chain.clear()

    @property
    def lastclear(self):
        """The iteration that the last clear occurred at."""
        # all of the chains should of been cleared at the same time, so just
        # use the first one
        return self.chains[0].lastclear

    def concatenate_chains(self, attr, item=None):
        """Concatenates the given attribute over all of the chains.
        
        Returned array has shape ``[ntemps x] nchains x niterations``.
        """
        # we'll transpose each chain so that niters x ntemps -> ntemps x niters
        # then stack the chains along the first axis, giving
        # ntemps x nchains x niters
        if item is None:
            getter = lambda x: getattr(x, attr).T
        else:
            getter = lambda x: getattr(x, attr).T[item]
        if self.ntemps == 1:
            axis = 0
        else:
            axis = 1  # will give ntemps x nchains x niterations
        return numpy.stack(map(getter, self.chains), axis=axis)

    @property
    def positions(self):
        """The history of positions from all of the chains."""
        return self.concatenate_chains('positions')

    @property
    def current_positions(self):
        """The current position of the chains.

        This will default to the start position if the chains haven't been
        run yet.
        """
        return self.concatenate_chains('current_position')

    @property
    def stats(self):
        """The history of stats from all of the chains."""
        return self.concatenate_chains('stats')

    @property
    def current_stats(self):
        """The current stats of the chains.

        This will default to the stats of the start positions if the chains
        haven't been run yet.
        """
        return self.concatenate_chains('current_stats')

    @property
    def blobs(self):
        """The history of all of the blobs from all of the chains."""
        if self.chains[0].hasblobs:
            blobs = self.concatenate_chains('blobs')
        else:
            blobs = None
        return blobs

    @property
    def current_blobs(self):
        """The current blob data.

        This will default to the blob data of the start positions if the
        chains haven't been run yet.
        """
        if self.chains[0].hasblobs:
            blobs = self.concatenate_chains('current_blob')
        else:
            blobs = None
        return blobs

    @property
    def acceptance_ratios(self):
        """The history of all acceptance ratios from all of the chains."""
        return self.concatenate_chains('acceptance_ratios')

    @property
    def temperature_swaps(self):
        """The history of all the temperature swaps from all of the chains.
        
        If ntemps is 1, just returns None.
        """
        if self.ntemps == 1:
            return None
        return self.concatenate_chains('temperature_swaps')

    @property
    def state(self):
        """The state of all of the chains.

        Returns a dictionary mapping chain ids to their states.
        """
        return {chain.chain_id: chain.state for chain in self.chains}

    def set_state(self, state):
        """Sets the state of the all of the chains.

        Parameters
        ----------
        state : dict
            Dictionary mapping chain id to the state to set the chain to. See
            :py:func:`chain.Chain.set_state` for details.
        """
        for ii, chain in enumerate(self.chains):
            chain.set_state(state[ii])


def _evolve_chain(niterations_chain):
    """Evolves a chain for some number of iterations.

    This is used by ``Sampler.run`` to evolve a collection of chains in a
    parallel environment. This is not a staticmethod of ``Sampler`` because
    such functions need to be picklable when using python's multiprocessing
    pool.

    Parameters
    ----------
    niterations_chain : tuple of int, chain
        The number of iterations to run on the given chain.
    """
    niterations, chain = niterations_chain
    for _ in range(niterations):
        chain.step()
    return chain
