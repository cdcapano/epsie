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

import itertools

import numpy

from .base import BaseProposal


class JointProposal(BaseProposal):
    """A collection of jump proposals for multiple parameters.

    Parameters
    ----------
    \*proposals :
        The arguments should provide the constituent proposals to use.
    bit_generator : :py:class:`epsie.BIT_GENERATOR` instance or int, optional
        The random bit generator to use, or an integer/None. If the latter, a
        bit generator will be created using
        :py:func:`epsie.create_bit_generator`.
    """
    name = 'joint'

    # Py3XX: change kwargs to explicit random_state=None
    def __init__(self, *proposals, **kwargs):
        bit_generator = kwargs.pop('bit_generator', None)  # Py3XX: delete line
        self.parameters = itertools.chain(*[prop.parameters
                                            for prop in proposals])
        # check that we don't have multiple proposals for the same parameter
        if len(set(self.parameters)) != len(self.parameters):
            # get the repeated parameters
            repeated = [p for p in self.parameters
                        if self.parameters.count(p) > 1]
            raise ValueError("multiple proposals provided for parameter(s) {}"
                             .format(', '.join(repeated)))
        # the joint proposal is symmetric only if all of the constitutent
        # proposals are also
        self._symmetric = all(prop.symmetric for prop in proposals)
        # set the bit generator
        self.bit_generator = bit_generator
        # have all of the proposals use the same random state
        for prop in proposals:
            prop.bit_generator = self.bit_generator
        # store the proposals as a dict of parameters -> proposal; we can do
        # this since the mapping of parameters to proposals is one/many:one
        self.proposals = {prop.parameters: prop for prop in proposals}

    @property
    def symmetric(self):
        return self._symmetric

    def logpdf(self, xi, givenx):
        return sum(p.logpdf(xi, givenx) for p in self.proposals.values())

    def update(self, chain):
        # update each of the proposals
        for prop in self.proposals.values():
            prop.update(chain)

    def jump(self, fromx):
        out = {}
        for prop in self.proposals.values():
            out.update(prop.jump(fromx))
        return out

    @property
    def state(self):
        # get all of the proposals state
        state = {params: prop.state for params, prop in self.proposals.items()}
        # add the global random state
        state['random_state'] = self.random_state
        return state

    def set_state(self, state):
        # set each proposals' state
        for params, prop in self.proposals.items():
            prop.set_state(state[params])
        # set the state of the random number generator
        self.random_state = state['random_state']
