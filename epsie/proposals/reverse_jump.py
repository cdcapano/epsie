# Copyright (C) 2020 Richard Stiskalek, Collin Capano

# This program is free software; you can redistribute it and/or modify it # under the terms of the GNU General Public License as published by the
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
from scipy import stats

from .base import BaseProposal
from .discrete import BoundedDiscrete


class NestedTransdimensional(BaseProposal):
    """A collection of jump proposals for multiple parameters.
        # UPDATE THIS DESCRIPTION
    Parameters
    ----------
    *update_props :
        The arguments should provide the constituent proposals to use.
    bit_generator : :py:class:`epsie.BIT_GENERATOR` instance or int, optional
        The random bit generator to use, or an integer/None. If the latter, a
        bit generator will be created using
        :py:func:`epsie.create_bit_generator`.

    Attributes
    ----------
    update_props : list
        The constituent proposals.
    """
    name = 'transdimensional'

    # Py3XX: change kwargs to explicit random_state=None
    def __init__(self, parameters, update_proposal, prior_dist,
                 bounds, model_index='k', **kwargs):
        self.parameters = parameters
        # store the model index
        self.k = model_index
        bit_generator = kwargs.pop('bit_generator', None)  #Py3XX: delete line
        self.bit_generator = bit_generator
        # model jumping proposal
        self.td_proposal = BoundedDiscrete([model_index],
                                           {model_index: bounds[model_index]},
                                           successive={model_index: True})
        # copy update proposal for each within model proposal
        self.inmodel_props = [deepcopy(update_proposal)
                              for i in range(bounds[model_index][0],
                                             bounds[model_index][1] + 1)]
        # rename the within model proposal parameters to reflect the index
        # and have all proposals use the same random state
        for k, prop in enumerate(self.inmodel_props):
            prop.parameters = ['{}{}'.format(p, k+1) for p in prop.parameters]
            prop.bit_generator = self.bit_generator
        self.td_proposal.bit_generator = self.bit_generator
        # store the prior distribution to sample new components on the go
        self._prior_dist = None
        self.prior_dist = prior_dist
        # the proposal is not symmetric since we use discrete bounded normal
        # for model jumps
        self._symmetric = False

        self._initialised = False
        self._props0 = None

    @property
    def symmetric(self):
        return self._symmetric

    @property
    def prior_dist(self):
        return self._prior_dist

    @prior_dist.setter
    def prior_dist(self, prior_dist):
        print(self.parameters)
        try:
            self._prior_dist = {}
            bound = self.td_proposal.boundaries[self.k]
            for k in range(bound[0], bound[1] + 1):
                self._prior_dist.update({'{}{}'.format(p, k): prior_dist[p]
                                        for p in self.parameters})
        except KeyError:
            raise ValueError('provide a prior for each parameter')

    def logpdf(self, xi, givenx):
        # Check whether this is called before update and what happens to
        # recently actived/deactivated proposals
        lp = 0.0
        for prop in self.inmodel_props:
            if prop.active:
                x0 = {p: givenx[p] for p in prop.parameters}
                x = {p: xi[p] for p in prop.parameters}
                # for a now quick hack
                lp_here = prop.logpdf(x, x0)
                if np.isfinite(lp_here):
                    lp += lp_here
        return lp

    def update(self, chain):
        # update each of the proposals
        if chain.acceptance[-1]['accepted'] and not (self._props0 is None):
            for prop in self._props0:
                prop = not prop.active
        self._props0 = None # back to None just to be sure

        # no adaptation for now
#        for prop in self._all_proposals:
#            prop.update(chain)


    def jump(self, fromx):
        newk = self.td_proposal.jump({self.k: fromx[self.k]})
        # if not initialised pick which proposals are active
        if not self._initialised:
            for prop in self.inmodel_props:
                if numpy.alltrue(numpy.isnan([fromx[p]
                                              for p in prop.parameters])):
                    prop.active = False
                else:
                    prop.active = True

        act_props = [prop for prop in self.inmodel_props if prop.active]
        inact_props = [prop for prop in self.inmodel_props if not prop.active]

        out = fromx.copy()
        out.update(newk)
        dk = out[self.k] - fromx[self.k]
        # birth
        if dk > 0:
            self._props0 = self.random_generator.choice(inact_props, size=dk,
                                                        replace=False)
            # sample the prior
            pars = [item for t in [prop.parameters for prop in self._props0]
                    for item in t]
            out.update({p: self.prior_dist[p].rvs() for p in pars})
        # death
        elif dk < 0:
            self._props0 = self.random_generator.choice(act_props, size=-dk,
                                                        replace=False)
            pars = [item for t in [prop.parameters for prop in self._props0]
                    for item in t]
            # set the removed proposal parameters to nans
            out.update({p: numpy.nan for p in pars})
        # keep the same dimensionality
        else:
            for prop in act_props:
                out.update(prop.jump({p: fromx[p] for p in prop.parameters}))

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
        for prop in (self.inmodel_props + [self.td_proposal]):
            prop.set_state(state[frozenset(prop.parameters)])
        # set the state of the random number generator
        self.random_state = state['random_state']
