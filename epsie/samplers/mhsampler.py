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

from __future__ import absolute_import

import numpy
import copy

from epsie import create_bit_generator
from epsie.chain import Chain
from epsie.chain.chaindata import (ChainData, detect_dtypes)

from .base import BaseSampler


class MetropolisHastingsSampler(BaseSampler):
    """A standard Metropolis-Hastings sampler.

    Parameters
    ----------
    parameters : tuple or list
        Names of the parameters to sample.
    model : object
        Model object.
    nchains : int
        The number of chains to create. Must be greater than zero.
    proposals : list, optional
        List of proposals to use. Any parameters that do not have a proposal
        provided will use the ``default_propsal``.
    default_proposal : an epsie.Proposal class, optional
        The default proposal to use for parameters not in ``proposals``.
        Default is :py:class:`epsie.proposals.Normal`.
    default_proposal_args : dict, optional
        Dictionary of arguments to pass to the default proposal.
    seed : int, optional
        Seed for the random number generator. If None provided, will create
        one.
    pool : Pool object, optional
        Specify a process pool to use for parallelization. Default is to use a
        single core.
    """

    def __init__(self, parameters, model, nchains, proposals=None,
                 default_proposal=None, default_proposal_args=None, seed=None,
                 pool=None):
        self.parameters = parameters
        self.model = model
        self.set_proposals(proposals, default_proposal, default_proposal_args)
        self.seed = seed
        self.pool = pool
        self.create_chains(nchains)

    def create_chains(self, nchains):
        """Creates a list of :py:class:`chain.Chain`.

        Parameters
        ----------
        nchains : int
            The number of Markov chains to create.
        """
        if nchains < 1:
            raise ValueError("nchains must be >= 1")
        self._chains = [Chain(
            self.parameters, self.model,
            [copy.deepcopy(p) for p in self.proposals],
            bit_generator=create_bit_generator(self.seed, stream=cid),
            chain_id=cid)
            for cid in range(nchains)]

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
            Dictionary mapping parameters to arrays. The arrays have shape
            ``nchains``.
        """
        # we'll create a chain data instance to stack the dictionaries
        d = getattr(self.chains[0], attr)
        out = ChainData(list(d.keys()), dtypes=detect_dtypes(d))
        out.extend(self.nchains)
        for ii, chain in enumerate(self.chains):
            out[ii] = getattr(chain, attr)
        return out.asdict()

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
            The returned array has shape ``nchains x niterations``.
        """
        if item is None:
            arrs = list(map(lambda x: getattr(x, attr), self.chains))
        else:
            arrs = list(map(lambda x: getattr(x, attr)[item], self.chains))
        return numpy.stack(arrs)
