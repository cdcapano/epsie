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
    """Nested transdimensional proposal. Assummes there is an unknown number
    of signals and each signal shares the same parameters.

    Parameters
    ----------
    proposal : py:class `epsie.proposals`
        The within model proposal for each signal. This is deep copied for
        potential signals, thus in the current implementation all consitutent
        signals must share the same functional form.
    prior_dist : dict
        Prior distributions for each parameter. Must be a class that contains
        a `rvs` method.
    bounds: dict
        Bounds on the model index. It is sufficient to only provide the bound
        on the model index k.
    model_indx: str (optinal)
        String denoting the model index. By default `k`.
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
    def __init__(self, parameters, proposals, prior_dist,
                 bounds, model_index='k', **kwargs):
        self.parameters = parameters + [model_index]



        k0, kf = bounds[model_indx]
        self.kmax = kf
        # store the model index
        self.k = model_indx
        bit_generator = kwargs.pop('bit_generator', None)  # Py3XX: delete line
        self.bit_generator = bit_generator
        # proposal suggesting transdimensional jumps
        self.td_proposal = BoundedDiscrete([model_indx],
                                              {model_indx: bounds[model_indx]},
                                              successive={model_indx: True})

        self.proposals = numpy.array(proposals)
        # let all proposals share the same random generator
        for prop in self.proposals + [self.td_proposal]:
            prop.bit_generator = self.bit_generator


        # store the prior distribution to sample new components on the go
        self._prior_dist = None
        self.prior_dist = prior_dist
        # the proposal is not symmetric since we use discrete bounded normal
        # for model jumps
        self._symmetric = False

    @property
    def symmetric(self):
        return self._symmetric

    @property
    def prior_dist(self):
        return self._prior_dist

    @prior_dist.setter
    def prior_dist(self, prior_dist):
        try:
            self._prior_dist = {}
            bound = self.model_proposal.boundaries[self.k]
            for k in range(bound[0], bound[1] + 1):
                self._prior_dist.update({'{}{}'.format(p, k): prior_dist[p]
                                        for p in self.unique_pars})
        except KeyError:
            raise ValueError('provide a prior for each parameter')

    def logpdf(self, xi, givenx):
        # logpdf on the model index
        current = givenx['_state']
        proposed = xi['_state']
        lp = 0.0

        dk = xi[self.k] - givenx[self.k]
        if dk > 0:
            # ones that are inactive in current but active in proposed
            # and simply consider the prior probability on these
            props = self.proposals[numpy.logical_and(
                numpy.logical_not(current), proposed)]
            pars = [par for pars in [prop.parameters for prop in props]
                    for par in pars]
            # evaluate the prior since thats how new ones are proposed
            lp += sum([self.prior_dist[p].logpdf(xi[p]) for p in pars])
        # proposal probability on parameters that were just updated
        props = self.proposals[numpy.logical_and(current, proposed)]
        for prop in props:
            lp += prop.logpdf({p: xi[p] for p in prop.parameters},
                              {p: givenx[p] for p in prop.parameters})
        # the model jumping probability
        lp += self.model_proposal.logpdf({self.k: xi[self.k]},
                                         {self.k: givenx[self.k]})
        return lp

    def update(self, chain):
        pass

    def jump(self, fromx):
        current_state = fromx['_state']
        props = fromx.pop('_proposals')
        out = fromx.copy()
        out.update(self.model_proposal.jump({self.k: fromx[self.k]}))
        dk = out[self.k] - fromx[self.k]
        # flip some proposals according to what dk is
        if dk != 0:
            if dk > 0:
                indx = numpy.where(numpy.logical_not(current_state))[0]
            elif dk < 0:
                indx = numpy.where(current_state)[0]
            # randomly pick which proposals will be turned on/off
            switch = self.random_generator.choice(indx, size=abs(dk),
                                                  replace=False).reshape(-1,)
            proposed_state = current_state.copy()
            proposed_state[switch] = numpy.logical_not(proposed_state[switch])
            # parameters to be turned on/off
            switch_pars = [par for pars in [
                prop.parameters for prop in props.proposals[switch]]
                for par in pars]
            update_proposals = props.proposals[
                numpy.logical_and(current_state, proposed_state)]
        else:
            update_proposals = props.proposals[current_state]
            proposed_state = current_state
        # update the out object
        if dk > 0:
            # sample the prior
            out.update({p: self.prior_dist[p].rvs() for p in switch_pars})
        elif dk < 0:
            # set to nans
            out.update({p: numpy.nan for p in switch_pars})
        # do a MCMC move on all proposals that are not nans/just activated
        for prop in update_proposals:
            p = {p: fromx[p] for p in prop.parameters}
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
