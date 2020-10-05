# Copyright (C) 2020 Richard Stiskalek, Collin Capano
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.  #
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
# Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.

from __future__ import (absolute_import, division)


import numpy
from scipy import stats

import epsie
from .base import BaseProposal


class NestedTransdimensional(BaseProposal):
    """Nested transdimensional proposal.

    Parameters
    ----------
    parameters : (list of) str
        The names of the parameters to produce proposals for.
    model_proposal : py:class `epsie.proposals`
        The model hopping proposals. This must be a discrete, bounded proposal.
    proposals : list of py:class `epsie.proposals`
        The transdimensional proposals that are being turned on/off.
    birth_distributions: list of objects
        Objects that match transdimensional proposals and are used to birth
        new samples and evaluate their proposal probability. Must use structure
        as given in the example.
    bit_generator : :py:class:`epsie.BIT_GENERATOR` instance or int, optional
        The random bit generator to use, or an integer/None. If the latter, a
        bit generator will be created using
        :py:func:`epsie.create_bit_generator`.

    Attributes
    ----------
    proposals: list of py:class `epsie.proposals`
        The constituent in-model proposals.
    model_proposal: py:class `epsie.proposals.DiscreteBounded
        The model hopping proposal
    """
    name = 'nested_transdimensional'
    transdimensional = True

    # Py3XX: change kwargs to explicit random_state=None
    def __init__(self, parameters, model_proposal, proposals,
                 birth_distributions, **kwargs):
        self._model_proposal = None
        self.parameters = parameters
        bit_generator = kwargs.pop('bit_generator', None)  # Py3XX: delete line
        self.bit_generator = bit_generator
        # store the proposals
        self.setup_proposals(model_proposal, proposals)
        # store the birth distributions
        self.setup_births(birth_distributions)
        self._index = self.model_proposal.parameters[0]
        self.set_jump_interval(1)

    @property
    def proposals(self):
        return self._proposals

    @property
    def model_proposal(self):
        return self._model_proposal

    def setup_proposals(self, model_proposal, proposals):
        # let all proposals share the same random generator
        for prop in proposals:
            prop.bit_generator = self.bit_generator
            # check the proposal parameters
            if not all(par in self.parameters for par in prop.parameters):
                raise ValueError("Proposal parameters {} not found "
                                 " in `parameters`.".format(prop.parameters))
        # check the model proposal
        model_proposal.bit_generator = self.bit_generator
        if len(model_proposal.parameters) > 1:
            raise ValueError("Model jump proposal should have single param")
        elif model_proposal.parameters[0] not in self.parameters:
            raise ValueError("Model jump proposal parameter {} not found in "
                             "`parameters`.".format(
                                 model_proposal.parameters[0]))

        self._proposals = numpy.array(proposals)
        self._model_proposal = model_proposal
        self._symmetric = (all(prop.symmetric for prop in proposals) and
                           model_proposal.symmetric)

    def setup_births(self, birth_distributions):
        """Matches birth distributions to proposals. Note that order of
        parameters matters"""
        # check all transdimensional proposals have their birth dists
        # and match the birth distribution to the given proposal. Also ensure
        # that no transdimensional proposal has more than a single birth dist
        for prop in self.proposals:
            matched = 0
            for dist in birth_distributions:
                if prop.parameters == tuple(dist.parameters):
                    dist.bit_generator = prop.bit_generator
                    prop.birth_distribution = dist
                    matched += 1
            if matched == 0:
                raise ValueError("Parameters {} miss `birth dist`. Note "
                                 "that order matters".format(prop.parameters))
            elif matched > 1:
                raise ValueError("Parameters {} have multiple"
                                 "`birth dists`".format(prop.parameters))

    @property
    def symmetric(self):
        return self._symmetric

    def _logpdf(self, xi, givenx):
        lp = 0.0
        # logpdf on the model jump
        lp += self.model_proposal.logpdf({self._index: xi[self._index]},
                                         {self._index: givenx[self._index]})
        # logpdf on the transdimensional moves
        current_state = givenx['_state']
        proposed_state = xi['_state']
        dk = xi[self._index] - givenx[self._index]
        if dk > 0:
            mask = numpy.logical_and(numpy.logical_not(current_state),
                                     proposed_state)
            for prop in self.proposals[mask]:
                lp += prop.birth_distribution.logpdf(
                    {p: xi[p] for p in prop.parameters})
        # logpdf on transdimensional moves that were only updated
        for prop in self.proposals[numpy.logical_and(current_state,
                                                     proposed_state)]:
            lp += prop.logpdf({p: xi[p] for p in prop.parameters},
                              {p: givenx[p] for p in prop.parameters})
        return lp

    def _update(self, chain):
        # check that proposal has been stepped in at least twice in a row
        if chain.iteration > 1:
            for prop in self.proposals:
                current = chain.positions[-1]
                if len(chain) == 1:
                    previous = chain.start_position
                else:
                    previous = chain.positions[-2]

                pars = prop.parameters
                condition1 = not all(numpy.isnan(previous[p]) for p in pars)
                condition2 = not all(numpy.isnan(current[p]) for p in pars)
                if condition1 and condition2:
                    prop.update(chain)

    def _jump(self, fromx):
        current_state = fromx['_state']
        out = fromx.copy()
        out.update(self.model_proposal.jump({self._index: fromx[self._index]}))

        dk = out[self._index] - fromx[self._index]
        if dk != 0:
            if dk > 0:
                indx = numpy.where(numpy.logical_not(current_state))[0]
            elif dk < 0:
                indx = numpy.where(current_state)[0]
            # randomly pick which proposals will be turned on/off
            proposed_state = current_state.copy()
            mask = self.random_generator.choice(indx, size=abs(dk),
                                                replace=False).reshape(-1,)
            proposed_state[mask] = numpy.logical_not(proposed_state[mask])
            # create the boolean mask for which proposals are being flipped
            if dk > 0:
                bd_mask = numpy.logical_and(numpy.logical_not(current_state),
                                            proposed_state)
            else:
                bd_mask = numpy.logical_and(current_state,
                                            numpy.logical_not(proposed_state))
            # update the out dictionary
            for prop in self.proposals[bd_mask]:
                if dk > 0:
                    out.update(prop.birth_distribution.birth)
                elif dk < 0:
                    out.update({p: numpy.nan for p in prop.parameters})
        else:
            proposed_state = current_state
        # do an update move on all proposals that are not nans/just activated
        update_mask = numpy.logical_and(current_state, proposed_state)
        for prop in self.proposals[update_mask]:
            out.update(prop.jump({p: fromx[p] for p in prop.parameters}))
        out.update({'_state': proposed_state})
        return out

    @property
    def state(self):
        # get all of the proposals state
        state = {}
        birth_state = {}
        for prop in self.proposals:
            state.update({frozenset(prop.parameters): prop.state})
            # Get the random states of birth dists
            birth_state.update(
                {frozenset(prop.parameters): prop.birth_distribution.state})

        state.update({'_births': birth_state})
        state.update({frozenset(self.model_proposal.parameters):
                      self.model_proposal.state})
        # add the global random state
        state['random_state'] = self.random_state
        return state

    def set_state(self, state):
        # set each proposals' state, birth dist's state and proposal counters
        for prop in self.proposals:
            prop.set_state(state[frozenset(prop.parameters)])
            prop.birth_distribution.set_state(
                state['_births'][frozenset(prop.parameters)])

        self.model_proposal.set_state(state[frozenset(
            self.model_proposal.parameters)])
        # set the state of the random number generator
        self.random_state = state['random_state']
