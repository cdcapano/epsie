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

    # Py3XX: change kwargs to explicit random_state=None
    def __init__(self, parameters, proposal, prior_dist,
                 bounds, model_indx='k', **kwargs):
        # store parameters
        self.unique_pars = parameters
        pars = list()
        k0, kf = bounds[model_indx]
        self.kmax = kf
        for p in self.unique_pars:
            for k in range(k0, kf + 1):
                pars.append('{}{}'.format(p, k))
        self.parameters = pars + [model_indx]
        # store the model index
        self.k = model_indx
        bit_generator = kwargs.pop('bit_generator', None) #Py3XX: delete line
        self.bit_generator = bit_generator
        # model jumping proposal
        self.model_proposal = BoundedDiscrete([model_indx],
                                              {model_indx: bounds[model_indx]},
                                              successive={model_indx: True})
        # copy update proposal for each within model proposal
        self.proposals = [deepcopy(proposal) for i in range(k0, kf+1)]
        # rename the within model proposal parameters to reflect the index
        # and have all proposals use the same random state
        for k, prop in enumerate(self.proposals):
            prop.bit_generator = self.bit_generator
            prop.parameters = ['{}{}'.format(p, k+1) for p in prop.parameters]
            try:
                prop.boundaries = {'{}{}'.format(key, k+1): item for key, item
                                   in zip(prop.boundaries.keys(),
                                          prop.boundaries.values())}
            except AttributeError:
                # most proposals do not have .boundaries attribute
                pass
        self.model_proposal.bit_generator = self.bit_generator
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
        # Check whether this is called before update and what happens to
        # recently actived/deactivated proposals
        # this is a huge bottleneck
        lp = 0.0
        for prop in self.proposals:
            if prop.active:
                x0 = {p: givenx[p] for p in prop.parameters}
                x = {p: xi[p] for p in prop.parameters}
                # for a now quick hack
                lp_here = prop.logpdf(x, x0)
                if numpy.isfinite(lp_here):
                    lp += lp_here
        return lp

    def update(self, chain):
        pass
#        for prop in self._all_proposals:
#            prop.update(chain)

    def jump(self, fromx, proposals_list):
        mcmc_move = proposals_list.active_mask.copy()
        switch_indx = None
        # copy the last point so inactive proposals do not have to be
        # explicitly copied
        out = fromx.copy()
        newk = self.model_proposal.jump({self.k: fromx[self.k]})
        out.update(newk)

        dk = out[self.k] - fromx[self.k]
        if dk != 0:
            act_indx = numpy.where(proposals_list.active_mask)
            inact_indx = numpy.where(numpy.logical_not(
                proposals_list.active_mask))

        if dk > 0:
            switch_indx = self.random_generator.choice(inact_indx[0], size=dk,
                                                       replace=False)
            switch_indx = switch_indx.reshape(-1, )
            switch_props = proposals_list.proposals[switch_indx]
            birth_pars = [item for t in [prop.parameters for prop
                                         in switch_props] for item in t]
            # sample the prior
            out.update({p: self.prior_dist[p].rvs() for p in birth_pars})
        elif dk < 0:
            switch_indx = self.random_generator.choice(act_indx[0], size=-dk,
                                                       replace=False)
            switch_indx = switch_indx.reshape(-1, )
            switch_props = proposals_list.proposals[switch_indx]
            death_pars = [item for t in [prop.parameters for prop
                                         in switch_props] for item in t]
            # set deaths to nans
            out.update({p: numpy.nan for p in death_pars})
            mcmc_move[switch_indx] = numpy.logical_not(mcmc_move[switch_indx])

        # do a MCMC move on all proposals that are not nans/just activated
        for prop in proposals_list.proposals[mcmc_move]:
            out.update(prop.jump({p: fromx[p] for p in prop.parameters}))
        return out, switch_indx

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
