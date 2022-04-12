# Copyright (C) 2022  Richard Stiskalek, Collin Capano
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

from copy import copy

import numpy

# from epsie.proposals import JointProposal

# from .base import BaseChain
# from .chaindata import (ChainData, detect_dtypes)
from .chain import Chain
from .chaindata import ChainData


class FastChain(Chain):
    """
    A fast parameters Markov chain.

    Clear up the chain at the end!

    Warn the user that blobs are not supported ?

    """
    _is_fast = True

    def __init__(self, parameters, model, proposals, slow_parameters, ndrags,
                 bit_generator=None):
        self._current_slow = None
        self._proposed_slow = None
        self._ndrags = None

        self.slow_parameters = slow_parameters
        self.ndrags = ndrags

        # Must be renamed because we will pass ``self.model`` to the parent
        # chain instead
        self.original_model = model

        # To store the likelihoods when evaluated with fixed slow and changing
        # fast parameters. Necessary because by default the chain is storing
        # only the interpolated probabilities.
        self._dragged_stats = ChainData(["lpost0", "lpostf"])
        self._dragged_stats.set_len(ndrags)

        # Initialise the parent chain
        super().__init__(parameters, self.model, proposals, bit_generator,
                         chain_id=0, beta=1.)

    @property
    def current_slow(self):
        """Current slow parameters."""
        if self._current_slow is None:
            raise ValueError("``current_slow`` not set!")
        return self._current_slow

    @current_slow.setter
    def current_slow(self, current_slow):
        """Sets the current fast parameters."""
        self._current_slow = self._set_slow_params(current_slow)

    @property
    def proposed_slow(self):
        """Proposed slow parameters."""
        if self._proposed_slow is None:
            raise ValueError("``proposed_slow`` not set!")
        return self._proposed_slow

    @proposed_slow.setter
    def proposed_slow(self, proposed_slow):
        """Sets the proposed slow parameters."""
        self._proposed_slow = self._set_slow_params(proposed_slow)

    def _set_slow_params(self, positions):
        """
        Check that slow parameters in ``positions`` match the expected slow
        parameters and that no extra parameters are given.
        """
        if positions is None:
            return None
        positions = copy(positions)
        checked_positions = {}
        for par in self.slow_parameters:
            val = positions.pop(par, None)
            if val is None:
                raise KeyError("Slow parameter ``{}`` missing position"
                               .format(par))

            checked_positions.update({par: val})
        if len(positions) > 0:
            raise ValueError("Unrecognised slow parameters: ``{}``"
                             .format(list(positions.keys())))

        return checked_positions

    @property
    def ndrags(self):
        """Number of fast parameter dragging step."""
        return self._ndrags

    @ndrags.setter
    def ndrags(self, ndrags):
        """Sets the number of dragging steps."""
        if not ndrags > 1:
            raise ValueError("`ndrags` must be larger than 1.")
        self._ndrags = ndrags

    def model(self, **proposed):
        """
        Interpolated posterior between the previous and proposed slow
        parameters. Sums the logarithms of the two PDFs weighted by the number
        of completed fast steps.
        """
        # Fraction of completed fast steps, + 1 to account for the start pos
        prog = (self.iteration + 1) / self.ndrags
        if prog > 1:
            raise ValueError("Chain should have been cleared.")
        # Call the models with the slow and fast parameters
        # the caching of the slow functions must happen on the user side
        r0 = self.original_model(**{**proposed, **self.current_slow})
        r1 = self.original_model(**{**proposed, **self.proposed_slow})

        # What do do about blobs?
        if self._hasblobs:
            raise NotImplementedError("Blobs not supported")
        else:
            ll0, lp0 = r0
            ll1, lp1 = r1

        # The 0th index is reserved for the start positions, so bump up
        # if start position already set
        index = len(self) + int(~numpy.isnan(self._dragged_stats[0]["lpost0"]))
        self._dragged_stats[index] = sum(r0), sum(r1)

        # Interpolation -- sum of log pdfs weighted by how many fast steps done
        def f(x, y):
            return (1 - prog) * x + prog * y

        return f(ll0, ll1), f(lp0, lp1)

    @property
    def dragging_logar(self):
        """
        Partial acceptance, will still have to be multiplied by the slow param
        proposal.

        """
        # TODO Return error if fast dragging not completed yet
        stats = self._dragged_stats

        logar = numpy.mean(stats["lpost0"] + stats["lpostf"])
        if numpy.isnan(logar):
            raise RuntimeError("Fast dragging not completed yet.")
        return logar

    def dragging_clear(self):
        self.clear()  # clear the parent chain
        self._dragged_stats.clear(self.ndrags)  # clear the dragged stats
        self._lastclear = 0  # we don't want to use this
        # Reset the iterations
        self._iteration = 0
        # Reset the current and proposed slow parameters
        self.current_slow = None
        self.proposed_slow = None

    def fast_stepping(self):
        """
        Util function to test the chain, will later remove.
        """
        for __ in range(1, self.ndrags):
            self.step()
