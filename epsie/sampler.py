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


from epsie import create_seed, create_brngs
from .chain import Chain


global _chains = None

def _create_chain(parameters, model, proposals, seed, chain_id):
    """Creates a chain and stores it to the global ``_chains`` dict.

    Parameters
    ----------
    parameters : list or tuple of strings
        The names of the parameters the chains should sample.
    model : object
        Model object for the chain to use.
    proposals : list of epsie.Proposals
        Proposal classes the chain should use.
    seed : int
        The seed the chains should use for initializing the BRNG.
    chain_ids : list of int
        Chain ids that should be created. The length of the list determines
        the number of chains to create; the ids are used to set the stream
        value for the BRNG, as well as to uniquely identify the chains.
    """
    if _chains is None:
        _chains = {}
    _chains[chain_id] = Chain(parameters, model, proposals,
                              brng=epsie.create_brng(seed, stream=chain_id),
                              chain_id=chain_id)

def _get_chain(chain_id):
    """Returns the chain with the given ``chain_id`` from ``_chains``.

    This is used to send a child process's chain to the parent process.
    """
    return _chains[chain_id]

def _set_start(positions, chain_id):
    """Sets the start position of the ``chain_id`` chain in ``_chains``.
    """
    _chains[chain_id].set_start({p: positions[p][chain_id] for p in positions})

def _run(self, chains, niterations):
    """Private method for evolving the chains."""
    for _ in range(niterations):
        for chain in chains:
            chain.step()
        self.niterations += 1
    return chains


class Sampler(object):
    """
    Parameters
    ----------
    parameters : tuple or list
        Names of the parameters to sample.
    model : object
        Model object.
    nchains : int
        The number of chains to create.
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
        

    def __init__(self, parameters, model, nchains, proposals=None,
                 default_proposal=None, seed=None, pool=None):
        self.parameters = tuple(parameters)
        self.model = model
        self.nchains = nchains
        if proposals is None:
            proposals = {}
        # create default proposal instances for the other parameters
        if default_proposal is None:
            default_proposal = Normal
        missing_props =  tuple(set(parameters) - set(proposals.keys()))
        proposals[missing_props] = default_proposal(missing_props)
        self.proposals = proposals
        self.niterations = 0
        # create the random number states
        if seed is None:
            seed = create_seed(seed)
        self.seed = seed
        # set the mapping function
        if pool is None:
            self.map = map
        else:
            self.map = pool.map
        # Create the chains: if we are using multiple processes, this will
        # cause each child process to only have the subset of chains it will
        # be running
        self.chains = self.map(self._setup_chain, range(self.nchains))

    def _setup_chain(self, chain_id):
        """Creates a chain."""
        _create_chain(self.parameters, self.model, self.proposals,
                      self.seed, chain_id)

    def set_start(self, positions):
        """Sets the starting position of all of the chains.

        Parameters
        ----------
        positions : dict
            Dictionary mapping parameter names to arrays of values. The chains
            must have length equal to the number of chains.
        """
        self.map(lambda x: _set_start(positions, x), range(self.nchains))

    def _collect_chains(self):
        """Collects all of the chains from the children processes."""
        self.chains = map(lambda x: x, self.chains)

    def clear(self):
        """Clears all of the chains."""
        self.map(lambda x: x.clear(), self.chains) 
        self._collect_chains()

    def run(self, niterations):
        """Evolves all of the chains by niterations.

        All chains are cycled over and incremented one step on each iteration;
        this ensures that the collection of chains evolve at approximately
        the same rate.

        Parameters
        ----------
        niterations : int
            The number of iterations to evolve the chains for.
        """
        self.map(lambda x: self._run(x, niterations), self.chains)
        self._collect_chains()
    def concatenate_chains(self, attr, item=None):
        """Concatenates the given attribute over all of the chains."""
        if item is None:
            getter = lambda x: getattr(x, attr)
        else:
            getter = lambda x: getattr(x, attr)[item]
        return numpy.stack(map(getter, chains), axis=0)

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
        #return {p: numpy.array([c.start_position[p] for c in self.chains])
        #        for p in self.parameters}
        return {p: self.concatenate_chains('start_position', p)
                for p in self.parameters}

    @property
    def positions(self):
        """The history of positions from all of the chains."""
        return {p: self.concatenate_chains('positions', p)
                for p in self.parameters}

    @property
    def stats(self):
        """The history of stats from all of the chains."""
        return {s: self.concatenate_chains('stats', s)
                for p in ['logl', 'logp']}

    @property
    def blobs(self):
        """The history of all of the blobs from all of the chains."""
        if self.chains[0].hasblobs:
            blobs = {b: self.concatenate_chains('blobs', b)
                     for b in self.chains[0].blob0}
        else:
            blobs = None
        return blobs

    @property
    def acceptance_ratios(self):
        """The history of all acceptance ratios from all of the chains."""
        return self.concatenate_chains('acceptance_ratio')

