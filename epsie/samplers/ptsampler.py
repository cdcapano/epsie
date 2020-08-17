# coding: utf-8

# Copyright (C) 2020 Collin Capano, Richard Stiskalek
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
import copy

from epsie import (create_bit_generator, array2dict)
from epsie.chain import ParallelTemperedChain
from epsie.chain.chaindata import (ChainData, detect_dtypes)

from .base import BaseSampler


class ParallelTemperedSampler(BaseSampler):
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
    swap_interval : int, optional
        How often to calculate temperature swaps. Default is 1 (= swap on every
        iteration).
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
    def __init__(self, parameters, model, nchains, betas, swap_interval=1,
                 proposals=None, adaptive_annealer=None, default_proposal=None,
                 default_proposal_args=None, seed=None, pool=None):
        self.parameters = parameters
        self.model = model
        self.set_proposals(proposals, default_proposal, default_proposal_args)
        self.seed = seed
        self.pool = pool
        if isinstance(betas, (float, int)):
            # only single temperature; turn into list so things below won't
            # break
            betas = [betas]
        if not isinstance(betas, numpy.ndarray):
            # betas is probably a list or tuple; convert to array so we can use
            # numpy functions
            betas = numpy.array(betas)
        self.create_chains(nchains, betas, swap_interval, adaptive_annealer)

    def create_chains(self, nchains, betas, swap_interval=1,
                      adaptive_annealer=None):
        """Creates a list of :py:class:`chain.ParallelTemperedChain`.

        Parameters
        ----------
        nchains : int
            The number of parallel tempered chains to create.
        betas : array
            Array of inverse temperatures to use for each parallel tempered
            chain.
        swap_interval : int, optional
            How often to calculate temperature swaps. Default is 1 (= swap on
            every iteration).
        adaptive_annealer : object, optional
            Adaptive anneler adjusting temperatures on the go.
            Default is `None`.
        """
        if nchains < 1:
            raise ValueError("nchains must be >= 1")
        self._chains = [ParallelTemperedChain(
            self.parameters, self.model,
            [copy.deepcopy(p) for p in self.proposals],
            betas=betas, swap_interval=swap_interval,
            adaptive_annealer=adaptive_annealer,
            bit_generator=create_bit_generator(self.seed, stream=cid),
            chain_id=cid)
            for cid in range(nchains)]

    @property
    def betas(self):
        """The betas used for each chain. Shape is (nchains, ntemps).
        If temperatures are being dynamically adjusted each chain will
        have a different set of temperatures.
        """
        return numpy.array([self.chains[i].betas for i in range(self.nchains)])

    @property
    def ntemps(self):
        """Returns the number of temperatures used."""
        return self.betas.shape[1]

    @property
    def temperatures(self):
        """The temperatures used. Shape is (nchains, ntemps)"""
        return 1./self.betas

    @property
    def swap_interval(self):
        """Returns the swap interval used for parallel tempering."""
        return self.chains[0].swap_interval

    @swap_interval.setter
    def swap_interval(self, interval):
        """Sets the swap interval to use."""
        for c in self.chains:
            c.swap_interval = interval

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
            ``ntemps x nchains``.
        """
        # we'll create a chain data instance to stack the dictionaries
        d = getattr(self.chains[0], attr)
        out = ChainData(list(d.keys()), dtypes=detect_dtypes(d),
                        ntemps=self.ntemps)
        out.extend(self.nchains)
        for ii, chain in enumerate(self.chains):
            out[ii] = getattr(chain, attr)
        return array2dict(out.data.T)

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
            The returned array has shape ``ntemps x nchains x niterations``.
        """
        if item is None:
            arrs = list(map(lambda x: getattr(x, attr), self.chains))
        else:
            arrs = list(map(lambda x: getattr(x, attr)[item], self.chains))
        return numpy.stack(arrs, axis=1)

    @property
    def temperature_acceptance(self):
        """The history of the temperature acceptance from all of the chains.

        If ntemps is 1, just returns None.

        .. note::
           This does not return a structured array, since there is only
           one field.
        """
        if self.ntemps == 1:
            return None
        return self._concatenate_arrays('temperature_acceptance')

    @property
    def temperature_swaps(self):
        """The history of all the temperature swaps from all of the chains.

        If ntemps is 1, just returns None.

        .. note::
           This does not return a structured array, since there is only
           one field.
        """
        if self.ntemps == 1:
            return None
        return self._concatenate_arrays('temperature_swaps')
