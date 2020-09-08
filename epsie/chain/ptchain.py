# coding: utf-8
# Copyright (C) 2020  Collin Capano, Richard Stiskalek
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

"""Markov chains with parallel tempering."""

from __future__ import absolute_import

import logging
import numpy
import copy

import epsie

from .base import BaseChain
from .chain import Chain
from .chaindata import (ChainData, detect_dtypes)


class ParallelTemperedChain(BaseChain):
    """A collection of parallel tempered Markov chains.

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
    adaptive_annealer : object, optional
        Adaptive annealing that adjusts the temperature levels during runtime.
        By default `None`, meaning no annealing.
    bit_generator : :py:class:`epsie.BIT_GENERATOR` instance, optional
        Use the given random bit generator for generating random variates. If
        an int or None is provided, a generator will be created instead using
        ``bit_generator`` as a seed.
    chain_id : int, optional
        An interger identifying which chain this is. Default is 0.

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
    bit_generator
    random_state
    state
    hasblobs
    chain_id : int or None
        Integer identifying the chain.
    """

    def __init__(self, parameters, model, proposals, betas=1., swap_interval=1,
                 adaptive_annealer=None, bit_generator=None, chain_id=0):
        self.parameters = parameters
        self.model = model
        # store the temp
        self._betas = None
        self.betas = betas
        self.swap_interval = swap_interval
        self._temperature_acceptance = None
        self._temperature_swaps = None
        self.adaptive_annealer = adaptive_annealer
        if adaptive_annealer is not None:
            # note that pass by reference is required here if setting
            # Tmax=infty
            adaptive_annealer.setup_annealing(self.betas)

        if self.ntemps > 1:
            # we pass ntemps=ntemps-1 here because there will be ntemps-1
            # acceptance ratios for ntemp levels
            self._temperature_acceptance = ChainData(
                ['acceptance_ratio'], dtypes={'acceptance_ratio': float},
                ntemps=self.ntemps-1)
            self._temperature_swaps = ChainData(
                ['swap_index'], dtypes={'swap_index': int},
                ntemps=self.ntemps)
        self.chain_id = chain_id
        # make sure all parallel tempered chains use the same bit_generator
        self._bit_generator = None
        self._random_generator = None
        self.bit_generator = bit_generator
        # create a chain for each temperature
        self.chains = [
            Chain(parameters, model,
                  [copy.deepcopy(p) for p in proposals],
                  bit_generator=self.bit_generator, chain_id=chain_id,
                  beta=beta)
            for beta in self.betas]
        self.transdimensional = any(chain.transdimensional
                                    for chain in self.chains)

    @property
    def bit_generator(self):
        """The random bit generator being used."""
        return self._bit_generator

    @bit_generator.setter
    def bit_generator(self, bit_generator=None):
        """Sets the random bit generator

        Parameters
        ----------
        bit_generator : :py:class:`epsie.BIT_GENERATOR` instance, optional
            Use the given random bit generator for generating random variates.
            If an int or None is provided, a generator will be created instead
            using ``bit_generator`` as a seed.
        """
        if bit_generator is None:
            bit_generator = epsie.create_bit_generator(None,
                                                       stream=self.chain_id)
        self._bit_generator = bit_generator

    @property
    def random_generator(self):
        """Returns the random number generator."""
        return self.chains[0].random_generator

    @property
    def random_state(self):
        """The current state of the random bit generator."""
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
    def state(self):
        """Returns the current state of the chain.

        The state consists of everything needed such that setting a chain's
        state using another's state will result in identical results.

        Returns
        -------
        dict :
            Dictionary of ``tk -> chains[tk].state``, where ``tk`` is the
            index of each temperature chain.
        """
        return {tk: chain.state for tk, chain in enumerate(self.chains)}

    def set_state(self, state):
        """Sets the state of the chain using the given dict.

        .. warning::
           Running this will clear the chain's current memory, and replace its
           current position with what is saved in the state.

        Parameters
        ----------
        state : dict
            Dictionary of ``tk -> dict`` mapping indices of the temperature
            chains to the state they should be set to.
        """
        for tk in state:
            self.chains[tk].set_state(state[tk])

    @property
    def hasblobs(self):
        """Whether the model returns blobs."""
        return self.chains[0].hasblobs

    @property
    def iteration(self):
        """The number of times the chain has been stepped."""
        return self.chains[0].iteration

    @property
    def lastclear(self):
        """Returns the iteration of the last time the chain memory was cleared.
        """
        return self.chains[0].lastclear

    @property
    def scratchlen(self):
        """The length of the scratch space used."""
        return self.chains[0].scratchlen

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
        for chain in self.chains:
            chain.scratchlen = n
        if self.ntemps > 1:
            try:
                self._temperature_swaps.set_len(n//self.swap_interval)
                self._temperature_acceptance.set_len(n//self.swap_interval)
            except ValueError:
                pass

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
            logging.warning("No betas = 1 found. This means that the normal "
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

    def _concatenate_dicts(self, attr):
        """Concatenates dictionary attributes over all of the temperatures.

        Parameters
        ----------
        attr : str
            The name of the attribute to get from the chains. The attribute
            is assumed to return a dictionary.
        """
        # we'll create a chain data instance to stack the dictionaries
        d = getattr(self.chains[0], attr)
        out = ChainData(list(d.keys()), dtypes=detect_dtypes(d))
        out.extend(self.ntemps)
        for tk, chain in enumerate(self.chains):
            out[tk] = getattr(chain, attr)
        return out.asdict()

    def _concatenate_arrays(self, attr, item=None):
        """Concatenates array attributes over all of the temperatures.

        Returned array has shape ``[ntemps x] niterations``.
        """
        if item is None:
            arrs = list(map(lambda x: getattr(x, attr), self.chains))
        else:
            arrs = list(map(lambda x: getattr(x, attr)[item], self.chains))
        return numpy.stack(arrs)

    @property
    def start_position(self):
        """Dictionary mapping parameters to their start position.

        If the start position hasn't been set, raises a ``ValueError``.
        """
        return self._concatenate_dicts('start_position')

    @start_position.setter
    def start_position(self, position):
        """Sets the starting position.

        This also evaulates the log likelihood and log prior at the starting
        position, as well as determine if the model returns blob data.

        Parameters
        ----------
        position : dict
            Dictionary mapping parameters to values. Values
            should be numpy arrays with length = ntemps.
        """
        self._start = position.copy()
        for tk, chain in enumerate(self.chains):
            posk = {param: position[param][tk] for param in position}
            chain.start_position = posk

    @property
    def stats0(self):
        """Dictionary of the log likelihood and log prior at the start
        position.

        The values of the returned dictionary are arrays of length ``ntemps``,
        ordered by increasing temperature.

        Raises a ``ValueError`` if the start position has not been set yet.
        """
        return self._concatenate_dicts('stats0')

    @property
    def blob0(self):
        """The blob data of the starting position, as a dictionary.

        If ``hasblobs`` is False, just returns None. Otherwise,  the values of
        the returned dictionary are arrays of length ``ntemps``, ordered by
        increasing temperature.

        Raises a ``ValueError`` if ``set_start`` has not been run yet.
        """
        if self.hasblobs:
            blob = self._concatenate_dicts('blob0')
        else:
            blob = None
        return blob

    @property
    def positions(self):
        """The history of all of the positions, as a structred array.

        If ``ntemps > 1``, the returned array has shape
        ``ntemps x niterations``. Otherwise, the returned array has shape
        ``niterations``.
        """
        return self._concatenate_arrays('positions')

    @property
    def stats(self):
        """The history of all of the stats, as a structred array.

        If ``ntemps > 1``, the returned array has shape
        ``ntemps x niterations``. Otherwise, the returned array has shape
        ``niterations``.
        """
        return self._concatenate_arrays('stats')

    @property
    def acceptance(self):
        """The history of all of acceptance ratios and accepted booleans,
        as a structred array.

        If ``ntemps > 1``, the returned array has shape
        ``ntemps x niterations``. Otherwise, the returned array has shape
        ``niterations``.
        """
        return self._concatenate_arrays('acceptance')

    @property
    def temperature_acceptance(self):
        """The history of the acceptance ratios between temperatures.

        The returned array has shape
        ``ntemps-1 x (niterations/swap_interval)`` if ``ntemps > 1``.
        Otherwise, returns None.

        .. note::
           This does not return a structured array, since there is only
           one field.
        """
        if self._temperature_acceptance is None:
            return None
        out = self._temperature_acceptance[:(len(self)//self.swap_interval)]
        return out['acceptance_ratio'].T

    @property
    def temperature_swaps(self):
        """The history of all of the temperature swaps.

        The returned array has shape ``ntemps x (niterations/swap_interval)``
        if ``ntemps > 1``. Otherwise, returns None.

        .. note::
           This does not return a structured array, since there is only
           one field.
        """
        if self._temperature_swaps is None:
            return None
        out = self._temperature_swaps[:(len(self)//self.swap_interval)]
        return out['swap_index'].T

    @property
    def blobs(self):
        """The history of all of the blob data, as a structured array.

        If the model does not return blobs, this is just ``None``.

        If ``ntemps > 1``, the returned array has shape
        ``ntemps x niterations``. Otherwise, the returned array has shape
        ``niterations``.
        """
        if self.hasblobs:
            blobs = self._concatenate_arrays('blobs')
        else:
            blobs = None
        return blobs

    @property
    def current_position(self):
        """Dictionary of the current position of the chain."""
        return self._concatenate_dicts('current_position')

    @property
    def current_stats(self):
        """Dictionary giving the log likelihood and log prior of the current
        position.
        """
        return self._concatenate_dicts('current_stats')

    @property
    def current_blob(self):
        """Dictionary of the blob data of the current position.

        If the model does not return blobs, just returns ``None``.
        """
        if not self.hasblobs:
            blob = None
        else:
            blob = self._concatenate_dicts('current_blob')
        return blob

    def clear(self):
        """Clears memory of the current chain, and sets start position to the
        current position.

        New scratch space will be created with length equal to ``scratch_len``.
        """
        for chain in self.chains:
            chain.clear()
        # clear temperature swaps
        if self.ntemps > 1:
            tlen = self.scratchlen//self.swap_interval
            self._temperature_acceptance.clear(tlen)
            self._temperature_swaps.clear(tlen)

    def __getitem__(self, index):
        """Returns all of the chain data at the requested index."""
        out = {'positions': self.positions[index],
               'stats': self.stats[index],
               'acceptance': self.acceptance[index]
               }
        if self.ntemps > 1:
            out['temperature_swaps'] = \
                self.temperature_swaps[index//self.swap_interval]
            out['temperature_acceptance'] = \
                self.temperature_acceptance[index//self.swap_interval]
        if self._hasblobs:
            out['blobs'] = self.blobs[index]
        return out

    def step(self):
        """Evolves all of the temperatures by one iteration.
        """
        for chain in self.chains:
            chain.step()
        # do temperature swaps
        if self.ntemps > 1 and self.iteration % self.swap_interval == 0:
            self.swap_temperatures()

    def swap_temperatures(self):
        """Computes acceptance ratio between temperatures and swaps them.

        The positions, stats, and (if they exist) blobs are swapped. The
        acceptance is not swapped, however.
        """
        # get values of all temps at current step
        stats = self.current_stats
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
            logar = dbetas[tj]*(loglj - loglk)
            if logar > 0:
                ar = 1.
                swap = True
            else:
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
        new_positions = [self.chains[swk].current_position
                         for swk in swap_index]
        new_stats = [self.chains[swk].current_stats
                     for swk in swap_index]
        if self.transdimensional:
            new_active = [self.chains[swk]._active_props
                          for swk in swap_index]
        if self.hasblobs:
            new_blobs = [self.chains[swk].current_blob
                         for swk in swap_index]
        # note: we're not swapping the acceptance ratios
        ii = self.iteration - self.lastclear - 1
        for (tk, chain) in enumerate(self.chains):
            chain._positions[ii] = new_positions[tk]
            chain._stats[ii] = new_stats[tk]
            if self.transdimensional:
                chain._active_props = new_active[tk]
            if self.hasblobs:
                chain._blobs[ii] = new_blobs[tk]
        self._temperature_acceptance[ii//self.swap_interval] = {
            'acceptance_ratio': ars}
        self._temperature_swaps[ii//self.swap_interval] = {
            'swap_index': swap_index}
        # for adaptive PT adjust the temperature leves
        if self.adaptive_annealer is not None:
            self.adaptive_annealer(self)


class DynamicalAnnealer(object):
    """
    Class for dynamical parallel tempering based on algorithm described in [1].

    Parameters
    ----------
    tau : int, optional
        Defines the swap iteration at which adjustments have been reduced to
        half their initial amplitude ([1]). Default value is 1000.
    nu : int, optional
        Defines the initial amplitude of adjustments ([1]).
        Default values is 10
    Tmax_prior: bool, optional
        Whether to set the hottest chain temperature to infinity. This only
        rewrites the hottest chain temperature to be infty and keeps the other
        chains as they were. By default sets it to infinity.


    References
    ----------
    [1] W. D. Vousden, W. M. Farr, I. Mandel, Dynamic temperature selection
    for parallel tempering in Markov chain Monte Carlo simulations,
    Monthly Notices of the Royal Astronomical Society, Volume 455,
    Issue 2, 11 January 2016, Pages 1919–1937,
    https://doi.org/10.1093/mnras/stv2422
    """
    _S = None
    _tau = None
    _nu = None

    def __init__(self, tau=1000, nu=10, Tmax_prior=True):
        self.setup_decay(tau, nu)
        self._Tmax_prior = Tmax_prior

    def setup_decay(self, tau, nu):
        """Set up constants for the vanishing decay"""
        if not tau > nu:
            return ValueError('`tau` must be at least larger than `nu`')
        self._tau = tau
        self._nu = nu

    def setup_annealing(self, betas):
        """Calculates the initial log diffs between temperature levels"""
        self._S = numpy.log(numpy.diff(1.0/betas[:-1]))
        if self._Tmax_prior:
            betas[-1] = 0.0

    def _decay(self, iteration):
        """ Vanishign decay to ensure detailed balance at later stages. Is set
        by `tau` and `nu`"""
        return 1./self._nu * 1./(1 + iteration/self._tau)

    def __call__(self, chain):
        iteration = chain.iteration // chain.swap_interval  # - 1 here ?
        ars = chain.temperature_acceptance[:, -1]
        ars[ars > 1] = 1.
        self._S += numpy.array([self._decay(iteration) * (ars[i] - ars[i+1])
                                for i in range(chain.ntemps - 2)])
        # recursively update the temperature levels
        # note: the coldest and hottest temperatures are kept fixed
        for i in range(1, chain.ntemps - 1):
            chain.betas[i] = 1./(1./chain.betas[i-1] + numpy.exp(self._S[i-1]))
