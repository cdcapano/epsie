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

"""Base class and functions for samplers."""

from __future__ import absolute_import

import itertools
from abc import (ABCMeta, abstractmethod)
import six
from six import add_metaclass
import numpy

from epsie import (create_seed, dump_state, load_state)
from epsie.proposals import Normal


@add_metaclass(ABCMeta)
class BaseSampler(object):
    """Base class for samplers.
    """
    _parameters = None
    _proposals = None
    _model = None
    _chains = None
    _seed = None
    _pool = None
    _map = None

    @property
    def parameters(self):
        """The sampled parameters as a tuple."""
        return self._parameters

    @parameters.setter
    def parameters(self, parameters):
        if isinstance(parameters, six.string_types):
            parameters = [parameters]
        self._parameters = tuple(parameters)

    @property
    def proposals(self):
        """List of the proposals used for the sampled parameters."""
        return self._proposals

    def set_proposals(self, proposals=None, default_proposal=None,
                      default_proposal_args=None):
        """Sets the proposals for all parameters.

        If no proposals are given for one or more parameters, the default
        proposal will be used for them.

        Parameters
        ----------
        proposals : list, optional
            A list of :py:class:`proposals.BaseProposal` instances. If none
            provided, will use the ``default_proposal`` for all parameters.
        default_proposal : proposals.BaseProposal type class, optional
            The default proposal class to use for parameters that are not
            specified in ``proposals``. Default (None) is to use
            :py:class:`proposals.Normal`.
        default_proposal_args : dict, optional
            Dictionary of keyword arguments to pass to the default proposal
            when initializing.
        """
        if proposals is None:
            proposals = []
        else:
            # make a copy of the so we don't modify what was given
            # Py3XX: uncomment this and drop the next line when drop 2.7
            #proposals = proposals.copy()
            proposals = [p for p in proposals]
        # create default proposal instances for the other parameters
        if default_proposal is None:
            default_proposal = Normal
        if default_proposal_args is None:
            default_proposal_args = {}
        if proposals:
            given_params = set.union(*[set(p.parameters) for p in proposals])
        else:
            given_params = set()
        missing_params = set(self.parameters) - given_params
        if missing_params:
            proposals.append(default_proposal(missing_params,
                                              **default_proposal_args))
        self._proposals = proposals

    @property
    def model(self):
        """The model used."""
        return self._model

    @model.setter
    def model(self, model):
        self._model = model

    @property
    def chains(self):
        """List of the chains used."""
        return self._chains

    @chains.setter
    def chains(self, chains):
        self._chains = chains

    @property
    def seed(self):
        """The seed used for the random bit generator."""
        return self._seed

    @seed.setter
    def seed(self, seed):
        """Sets the seed. If the given seed is None, will create one."""
        if seed is None:
            seed = create_seed(seed)
        self._seed = seed

    @property
    def pool(self):
        """The pool being used for parallelization."""
        return self._pool

    @pool.setter
    def pool(self, pool):
        self._pool = pool
        # set the map function
        if pool is None:
            map_func = map
        else:
            map_func = pool.map
        self._map = map_func

    @property
    def nchains(self):
        """The number of chains being used."""
        return len(self.chains)

    @property
    def niterations(self):
        """The number of iterations the chains have been run for.
        """
        # all of the chains should be at the same iteration, so just use the
        # first one
        return self.chains[0].iteration

    @abstractmethod
    def _concatenate_dicts(self, attr):
        """Concatenates dictionary attributes over all of the chains.

        This is a convenience function used by properties such as
        ``current_positions`` to gather all of the dictionary attributes from
        the chains.

        Parameters
        ----------
        attr : str
            The name of the attribute to get from the chains. The attribute
            is assumed to return a dictionary.

        Returns
        -------
        dict :
            Dictionary mapping parameters to arrays.
        """
        pass

    @abstractmethod
    def _concatenate_arrays(self, attr, item=None):
        """Concatenates the given attribute over all of the chains.

        This is a convenience function used by properties such as ``positions``
        to gather all of the array attributes from the chains.

        Parameters
        ----------
        attr : str
            The name of the attribute to get from the chains. The attribute
            is assumed to return a (structred) array.
        item : str or array index, optional
            Get a particular item from the (structred) array from each chain
            before concatenating.

        Returns
        -------
        array :
            The returned array sould have ``nchains x niterations`` as the last
            two dimensions.
        """
        pass

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
        return self._concatenate_dicts('start_position')

    @start_position.setter
    def start_position(self, positions):
        """Sets the starting position of all of the chains.

        Parameters
        ----------
        positions : dict
            Dictionary mapping parameter names to arrays of values. The arrays
            must have shape ``[ntemps x] nchains``.
        """
        for (ii, chain) in enumerate(self.chains):
            chain.start_position = {p: positions[p][..., ii]
                                    for p in positions}

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
        # pylint: disable=locally-disabled, not-callable
        self.chains = list(self._map(_evolve_chain, args))

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

    @property
    def positions(self):
        """The history of positions from all of the chains."""
        return self._concatenate_arrays('positions')

    @property
    def current_positions(self):
        """The current position of the chains.

        This will default to the start position if the chains haven't been
        run yet.
        """
        return self._concatenate_dicts('current_position')

    @property
    def stats(self):
        """The history of stats from all of the chains."""
        return self._concatenate_arrays('stats')

    @property
    def current_stats(self):
        """The current stats of the chains.

        This will default to the stats of the start positions if the chains
        haven't been run yet.
        """
        return self._concatenate_dicts('current_stats')

    @property
    def blobs(self):
        """The history of all of the blobs from all of the chains."""
        if self.chains[0].hasblobs:
            blobs = self._concatenate_arrays('blobs')
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
            blobs = self._concatenate_dicts('current_blob')
        else:
            blobs = None
        return blobs

    @property
    def acceptance(self):
        """The history of all acceptance stats from all of the chains."""
        return self._concatenate_arrays('acceptance')

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

    def checkpoint(self, fp, path=None, dsetname='sampler_state'):
        """Save the sampler's state to an HDF file.

        Parameters
        ----------
        fp : :py:class:h5py.File
            Open file handler to an hdf5 file. The file handler must have
            write permission.
        path : str, optional
            What group to write the state to in the hdf file. Default is the
            top-level.
        dsetname : str, optional
            The name of dataset to store the state to. Default is
            "sampler_state".
        """
        dump_state(self.state, fp, path=path, dsetname=dsetname)

    def set_state_from_checkpoint(self, fp, path=None):
        """Loads a state from an HDF file.

        Parameters
        ----------
        fp : :py:class:h5py.File
            Open file handler to an hdf5 file.
        path : str, optional
            What group to store the file to in the hdf file. Default is the
            top-level ('/').
        """
        self.set_state(load_state(fp, path=path))


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
