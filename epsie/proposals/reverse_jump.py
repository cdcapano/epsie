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
    inmodel_prop: py:class `epsie.proposals`
        The within model proposal for each signal. This is deep copied for
        potential signal. For now we assume a given maximum number of signals.
    prior_dist: dict
        Prior distributions for each parameter. Must be a class that contains
        a `rvs` method.
    bounds: dict
        Bounds on the model index. It is sufficient to only provide the bound
        on the model index k.
    model_index: str (optinal)
        String denoting the model index. By default `k`.
    bit_generator : :py:class:`epsie.BIT_GENERATOR` instance or int, optional
        The random bit generator to use, or an integer/None. If the latter, a
        bit generator will be created using
        :py:func:`epsie.create_bit_generator`.

    Attributes
    ----------
    inmodel_probs: list
        The constituent in proposals.
    td_proposal: py:class `epsie.proposals.DiscreteBounded
        The model hopping proposal
    """
    name = 'transdimensional'

    # Py3XX: change kwargs to explicit random_state=None
    def __init__(self, parameters, inmodel_prop, prior_dist,
                 bounds, model_index='k', **kwargs):

        # store parameters
        self.unique_pars = parameters
        pars = list()
        k0, kf = bounds[model_index]
        self.kmax = kf
        for p in self.unique_pars:
            for k in range(k0, kf + 1):
                pars.append('{}{}'.format(p, k))
        self.parameters = pars + [model_index]
        # store the model index
        self.k = model_index
        bit_generator = kwargs.pop('bit_generator', None)  #Py3XX: delete line
        self.bit_generator = bit_generator
        # model jumping proposal
        self.td_proposal = BoundedDiscrete([model_index],
                                           {model_index: bounds[model_index]},
                                           successive={model_index: True})
        # copy update proposal for each within model proposal
        self.inmodel_props = [deepcopy(inmodel_prop) for i in range(k0, kf+1)]
        # rename the within model proposal parameters to reflect the index
        # and have all proposals use the same random state
        for k, prop in enumerate(self.inmodel_props):
            prop.parameters = ['{}{}'.format(p, k+1) for p in prop.parameters]
            # rename the update proposal parameters. Latex make sure
            # this is only triggered for bounded proposals
            prop.boundaries = {'{}{}'.format(key, k+1): item
                               for key, item in zip(prop.boundaries.keys(),
                                                    prop.boundaries.values())}
            prop.bit_generator = self.bit_generator
        self.td_proposal.bit_generator = self.bit_generator
        # store the prior distribution to sample new components on the go
        self._prior_dist = None
        self.prior_dist = prior_dist
        # the proposal is not symmetric since we use discrete bounded normal
        # for model jumps
        self._symmetric = False

        self._initialised = False
        self._act = None
        self._inact = None
        self._choices = None

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
            bound = self.td_proposal.boundaries[self.k]
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
        for prop in self.inmodel_props:
            if prop.active:
                x0 = {p: givenx[p] for p in prop.parameters}
                x = {p: xi[p] for p in prop.parameters}
                # for a now quick hack
                lp_here = prop.logpdf(x, x0)
                if numpy.isfinite(lp_here):
                    lp += lp_here
        return lp

    def update(self, chain):
        # update each of the proposals
        if chain.acceptance[-1]['accepted'] and not (self._choices is None):
            # start popping from the back
            choices = sorted(self._choices['choices'], reverse=True)
            if self._choices['birth']:
                mv = [self._inact.pop(i) for i in choices]
                self._act += mv
            else:
                mv = [self._act.pop(i) for i in choices]
                self._inact += mv
            for prop in mv:
                prop.active = not prop.active
        self._choices = None # back to None just to be sure

        # no adaptation for now
#        for prop in self._all_proposals:
#            prop.update(chain)


    def jump(self, fromx):
        newk = self.td_proposal.jump({self.k: fromx[self.k]})
#        newk = {self.k: fromx[self.k] - 1}
        # if not initialised pick which proposals are active
        # this is entered only once so can afford some inefficiency
        if not self._initialised:
            print('initialising')
            for prop in self.inmodel_props:
                if numpy.alltrue(numpy.isnan([fromx[p]
                                              for p in prop.parameters])):
                    prop.active = False
                else:
                    prop.active = True
                self._act = [prop for prop in self.inmodel_props
                             if prop.active]
                self._inact = [prop for prop in self.inmodel_props
                               if not prop.active]
            self._initialised = True

        out = fromx.copy()
        out.update(newk)
        dk = out[self.k] - fromx[self.k]

        n_inact = len(self._inact)
        n_act = self.kmax - n_inact
        props = []
        if dk > 0: # birth
            choices = self.random_generator.choice(range(n_inact), size=dk,
                                                   replace=False)
            props = [self._inact[i] for i in choices]
            pars = [item for t in [prop.parameters for prop in props]
                    for item in t]
            out.update({p: self.prior_dist[p].rvs() for p in pars})
            self._choices = {'choices': choices, 'birth': True}
        elif dk < 0: # death
            choices = self.random_generator.choice(range(n_act), size=-dk,
                                                   replace=False)
            props = [self._act[i] for i in choices]
            pars = [item for t in [prop.parameters for prop in props]
                    for item in t]
            # set the removed proposal parameters to nans
            out.update({p: numpy.nan for p in pars})
            self._choices = {'choices': choices, 'birth': False}
        for prop in self._act:
            if not prop in props:
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
