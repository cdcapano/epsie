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

"""Markov chains with parallel tempering."""

from __future__ import absolute_import

import numpy
from scipy.stats import uniform as randuniform

import epsie
from epsie import array2dict

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
        # store the temp
        self._betas = None
        self.betas = betas
        self.swap_interval = swap_interval
        self._temperature_swaps = None
        if self.ntemps > 1:
            # we pass ntemps=ntemps-1 here because there will be ntemps-1
            # acceptance ratios for ntemp levels
            self._temperature_swaps = ChainData(
                ['acceptance_ratio', 'swap_index'],
                dtypes={'acceptance_ratio': float, 'swap_index': int},
                ntemps=self.ntemps-1)
        # make sure all parallel tempered chains use the same brng
        if brng is None:
            brng = epsie.create_brng(None, stream=chain_id)
        # create a chain for each temperature
        self.chains = [Chain(parameters, model,
                             [copy.copy(p) for p in proposals],
                             brng=brng, chain_id=chain_id,
                             beta=beta)
                        for beta in betas]

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
                self._temperature_swaps.set_len(n)
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
    def temperature_swaps(self):
        """The history of all of the temperature swaps."""
        return self._temperature_swaps[:len(self)]

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

    @start_position.setter
    def start_position(self, position):
        """Sets the starting position.

        This also evaulates the log likelihood and log prior at the starting
        position, as well as determine if the model returns blob data.

        Parameters
        ----------
        position : dict
            Dictionary mapping parameters to values. If ntemps > 1, values
            should be numpy arrays with length = ntemps. Otherwise, these
            should be atomic data types.
        """
        for tk, chain in enumerate(self.chains):
            if self.ntemps == 1:
                posk = {param: position[param][tk] for param in position}
            else:
                posk = position
            chain.start_position = posk

    def clear(self):
        """Clears memory of the current chain, and sets start position to the
        current position.
        
        New scratch space will be created with length equal to ``scratch_len``.
        """
        for chain in self.chains:
            chain.clear()
        # clear temperature swaps
        if self.ntemps > 1:
            self._temperature_swaps.clear(self.scratchlen)

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

    def step(self):
        """Evolves all of the temperatures by one iteration.
        """
        for chain in self.chains:
            chain.step()
        # do temperature swaps
        if self.ntemps > 1 and ii % self.swap_interval == 0:
            self.swap_temperatures(ii)

    def concatenate_temps(self, attr, item=None):
        """Concatenates the given attribute over all of the temperatures.

        Returned array has shape ``[ntemps x] niterations``.
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
            axis = 1  # will give ntemps x niterations
        return numpy.stack(map(getter, self.chains), axis=axis)