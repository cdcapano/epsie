# Copyright (C) 2020 Richard Stiskalek, Collin Capano
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

from __future__ import (absolute_import, division)

from copy import deepcopy

import numpy


from .base import BaseProposal
from .discrete import BoundedDiscrete


class NestedTransdimensional(BaseProposal):
    """Nested transdimensional proposal.

    Parameters
    ----------
    parameters : (list of) str
        The names of the parameters to produce proposals for.
    model_jump_proposal : py:class `epsie.proposals`
        The model hopping proposals. This must be a discrete, bounded proposal.
    transdimensional_proposals : list of py:class `epsie.proposals`
        The transdimensional proposals that are being turned on/off.
    birth_distributions: list of objects
        Objects that match transdimensional proposals and are used to birth
        new samples and evaluate their proposal probability. Must use structure
        as given in the example.
    global_proposals : list of py:class `epsie.proposals` (optional)
        The global proposals that are shared by all transdimensional models.
        By default `None`, meaning no global proposal.
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
    def __init__(self, parameters, model_jump_proposal,
                 transdimensional_proposals, birth_distributions,
                 global_proposals=None, **kwargs):
        self.parameters = parameters
        bit_generator = kwargs.pop('bit_generator', None)  # Py3XX: delete line
        self.bit_generator = bit_generator
        # store the proposals
        self.setup_proposals(model_jump_proposal, global_proposals,
                             transdimensional_proposals)
        # store the birth distributions
        self.setup_births(birth_distributions)

    @property
    def proposals(self):
        return self._proposals

    @property
    def birth_distributions(self):
        return self._birth_distributions

    def setup_proposals(self, model_jump_proposal, global_proposals,
                        transdimensional_proposals):
        if global_proposals is not None:
            global_proposals = numpy.array(global_proposals)
        transdimensional_proposals = numpy.array(transdimensional_proposals)
        proposals = {'model': model_jump_proposal,
                     'global': global_proposals,
                     'transdimensional': transdimensional_proposals}
        # let all proposals share the same random generator
        symmetric = list()
        for key, props in proposals.items():
            if key == 'model':
                props.bit_generator = self.bit_generator
                symmetric.append(props.symmetric)
                if len(props.parameters) > 1:
                    raise ValueError("Model jump proposal should have a "
                                     "single parameter")
                elif props.parameters[0] not in self.parameters:
                    raise ValueError("Model jump proposal parameter not found "
                                     " in `parameters`.")
            else:
                for prop in props:
                    prop.bit_generator = self.bit_generator
                    symmetric.append(prop.symmetric)
                    if not all([par in self.parameters
                                for par in prop.parameters]):
                        raise ValueError("Some proposal parameters not found "
                                         " in `parameters`.")
        self._proposals = proposals
        self._symmetric = all(symmetric)

    def setup_births(self, birth_distributions):
        """Matches birth distributions to proposals. Note that order of
        parameters matters"""
        # check all transdimensional proposals have their birth dists
        # and match the birth distribution to the given proposal. Also ensure
        # that no transdimensional proposal has more than a single birth dist
        for prop in self.proposals['transdimensional']:
            matched = 0
            for dist in birth_distributions:
                if prop.parameters == tuple(dist.parameters):
                    dist.set_bit_generator(self.bit_generator)
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

    def logpdf(self, xi, givenx):
        lp = 0.0
        # logpdf on the model jump 
        k = self.proposals['model'].parameters[0]
        lp += self.proposals['model'].logpdf({k: xi[k]}, {k: givenx[k]})

        # logpdf on the transdimensional moves
        current = givenx['_state']
        proposed = xi['_state']
        dk = xi[k] - givenx[k]
        if dk > 0:
            props = self.proposals['transdimensional'][numpy.logical_and(
                numpy.logical_not(current), proposed)]

            for prop in props:
                lp += prop.birth_distribution.logpdf(
                    {p: xi[p] for p in prop.parameters})
        # logpdf on transdimensional moves that were only updated
        # and on the global proposals
        update_props = self.proposals['transdimensional'][numpy.logical_and(
            current, proposed)]
        for prop in numpy.hstack([update_props, self.proposals['global']]):
            lp += prop.logpdf({p: xi[p] for p in prop.parameters},
                              {p: givenx[p] for p in prop.parameters})
        return lp

    def update(self, chain):
        pass

    def jump(self, fromx):
        current_state = fromx['_state']
        out = fromx.copy()
        # update the global proposals if any
        if self.proposals['global'] is not None:
            for prop in self.proposals['global']:
                out.update(prop.jump({p: fromx[p] for p in prop.parameters}))

        # update the transdimensional proposals
        mod_prop = self.proposals['model']
        td_props = self.proposals['transdimensional']
        out.update(mod_prop.jump({mod_prop.parameters[0]:
                                  fromx[mod_prop.parameters[0]]}))
        dk = out[mod_prop.parameters[0]] - fromx[mod_prop.parameters[0]]
        if dk != 0:
            if dk > 0:
                indx = numpy.where(numpy.logical_not(current_state))[0]
            elif dk < 0:
                indx = numpy.where(current_state)[0]

            # randomly pick which proposals will be turned on/off
            proposed_state = current_state.copy()
            m = self.random_generator.choice(indx, size=abs(dk),
                                             replace=False).reshape(-1,)
            proposed_state[m] = numpy.logical_not(proposed_state[m])
            update_proposals = td_props[numpy.logical_and(current_state,
                                                          proposed_state)]
        else:
            update_proposals = td_props[current_state]
            proposed_state = current_state

        # update the out dictionary
        if dk > 0:
            birth_proposals = td_props[numpy.logical_and(numpy.logical_not(
                current_state), proposed_state)]
            for prop in birth_proposals:
                out.update(prop.birth_distribution.birth)
        elif dk < 0:
            death_proposals = td_props[numpy.logical_and(
                current_state, numpy.logical_not(proposed_state))]

            for prop in death_proposals:
                out.update({p: numpy.nan for p in prop.parameters})

        # do an update move on all proposals that are not nans/just activated
        for prop in update_proposals:
            out.update(prop.jump({p: fromx[p] for p in prop.parameters}))
        out.update({'_state': proposed_state})
        return out

    @property
    def state(self):
        # get all of the proposals state
        state = {frozenset(prop.parameters): prop.state
                 for prop in self._all_proposals}
        # add the global random state
        state['random_state'] = self.random_state
        return state

    def set_state(self, state):
        # set each proposals' state
        for prop in (self.proposals + [self.model_proposal]):
            prop.set_state(state[frozenset(prop.parameters)])
        # set the state of the random number generator
        self.random_state = state['random_state']
