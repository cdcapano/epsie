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

import itertools
import math

import random

import numpy as np
from scipy import stats

from .base import BaseProposal
from .bounded_normal import Boundaries


class BirthDeath(BaseProposal):
    """Transdimensional proposal that allows dimension change by +- 1 or  keep
    the same number of dimensions. Accepts only a single parameter.
    --------------------------
    Parameters:
        parameter: str
            Parameter name for this proposal
        boundary: tuple
            Inclusive lower and upper limits on the model parameter
        jump_proposal: func
            1D log of the probability density function
        jump_freq: float (optional)
            Parameter that tunes the proportion of dimension jumps
    """
    name = 'birthdeath'
    symmetric = False

    def __init__(self, parameter, boundary, jump_proposal, jump_freq=0.5):
        self._parameter = None
        self._boundary = None
        self._jum_proposal = None
        self._jump_freq = None

        self.parameter = parameter
        self.boundary = boundary
        self.jump_proposal = jump_proposal
        self.jump_freq = jump_freq

    @property
    def parameter(self):
        """Parameter of this proposal"""
        return self._parameter

    @parameter.setter
    def parameter(self, parameter):
        if not isinstance(parameter, str):
            raise ValueError('provide a str')
        self._parameter = parameter

    @property
    def parameters(self):
        """For compatibility with other proposals returns the model
        parameter wrapped in a list"""
        return [self.parameter]

    @property
    def boundary(self):
        """Dictionary of parameter boundary"""
        return self._boundary

    @boundary.setter
    def boundary(self, boundary):
        self._boundary = Boundaries(boundary)

    @property
    def jump_proposal(self):
        """Dictionary of jump proposals for each parameter"""
        return self._jump_proposal

    @jump_proposal.setter
    def jump_proposal(self, jump_proposal):
        self._jump_proposal = jump_proposal

    @property
    def jump_freq(self):
        return self._jump_freq

    @jump_freq.setter
    def jump_freq(self, jump_freq):
        if not isinstance(jump_freq, float):
            raise ValueError('must be a float')
        elif not 0.0 <= jump_freq <= 0.5:
            raise ValueError('jump frequency must be in [0.0, 0.5]')
        self._jump_freq = jump_freq

    @property
    def state(self):
        return {'random_state': self.random_state}

    def set_state(self, state):
        self.random_state = state['random_state']

    def jump(self, fromx):
        k = fromx[self.parameter]
        current = self.jump_proposal(k)
        # Don't forget to check boundaries
        if k == self.boundary.lower:
            death = 0.0
        else:
            death = min(1, np.exp(self.jump_proposal(k - 1) - current))
            death *= self.jump_freq

        if k == self.boundary.upper:
            birth = 0.0
        else:
            birth = min(1, np.exp(self.jump_proposal(k + 1) - current))
            birth *= self.jump_freq

        # Remove a signal with 'death' prob, add with 'birth' prob and
        # update with 'update' prob.
        u = self.random_generator.uniform()
        if u <= birth:
            newpt = {self.parameter: k + 1}
        elif u <= birth + death:
            newpt = {self.parameter: k - 1}
        else:
            newpt = {self.parameter: k}

        return newpt

    def logpdf(self, xi, givenx):
        return self.jump_proposal(xi[self.parameter])


class TransDimensional(BaseProposal):
    """A collection of jump proposals for multiple parameters.
        # UPDATE THIS DESCRIPTION
    Parameters
    ----------
    *update_proposals :
        The arguments should provide the constituent proposals to use.
    bit_generator : :py:class:`epsie.BIT_GENERATOR` instance or int, optional
        The random bit generator to use, or an integer/None. If the latter, a
        bit generator will be created using
        :py:func:`epsie.create_bit_generator`.

    Attributes
    ----------
    update_proposals : list
        The constituent proposals.
    """
    name = 'transdimensional'

    # Py3XX: change kwargs to explicit random_state=None
    def __init__(self, bd_proposal, prior_dist, *update_proposals, **kwargs):
        bit_generator = kwargs.pop('bit_generator', None)  #Py3XX: delete line
        update_parameters = list(itertools.chain(*[prop.parameters\
                                            for prop in update_proposals]))
        #print(update_parameters)
        # check that we don't have multiple update_proposals
        # for the same parameter
        repeated = [p for p in set(update_parameters)
                        if update_parameters.count(p) > 1]
        if repeated:
            raise ValueError("multiple update_proposals provided "
                             "for parameter(s) {}".format(', '.join(repeated)))
        # store parameters
        self.update_parameters = update_parameters
        self.model_parameter = bd_proposal.parameter
        self.parameters = update_parameters + [self.model_parameter]
        # store proposals
        self.update_proposals = list(update_proposals)
        self.bd_proposal = bd_proposal
        self.all_proposals = self.update_proposals + [self.bd_proposal]
        # the proposal is symmetric only if all of the constitutent
        # proposals are also
        self._symmetric = all([prop.symmetric
                              for prop in self.all_proposals])
        # store the prior distribution to sample new components on the go
        self._prior_dist = None
        self.prior_dist = prior_dist
        # set the bit generator
        self.bit_generator = bit_generator
        # have all of the proposals use the same random state
        for prop in self.all_proposals:
            prop.bit_generator = self.bit_generator


    @property
    def symmetric(self):
        return self._symmetric

    def logpdf(self, xi, givenx):
        return sum(p.logpdf(xi, givenx) for p in self.all_proposals)

    def update(self, chain):
        # update each of the proposals
        for prop in self.all_proposals:
            prop.update(chain)

    @property
    def prior_dist(self):
        return self._prior_dist

    @prior_dist.setter
    def prior_dist(self, prior_dist):
        try:
            self._prior_dist = {p: prior_dist[p]
                                for p in self.update_parameters}
        except KeyError:
            raise ValueError('provide a value for each parameter')

    def jump(self, fromx):
        givenk = fromx[self.model_parameter]
        newk = self.bd_proposal.jump({self.model_parameter: givenk})
        # unpack it
        newk = newk[self.model_parameter]
#        print('newk', newk)
#        print('inititla len updates props', len(self.update_proposals))
#        for prop in self.update_proposals:
#            print(prop.parameters, prop.active)

        inactive_props = list()
        active_props = list()
        for prop in self.update_proposals:
            if prop.active:
                active_props.append(prop)
            else:
                inactive_props.append(prop)


        out = {}
        if newk != givenk:
            if newk > givenk:
                # pick which proposal to turn on
                i = random.choice(range(len(inactive_props)))
                birth_prop = inactive_props.pop(i)
                # pick up the birth signal from the prior
                out.update({p: self.prior_dist[p].rvs()
                            for p in birth_prop.parameters})
                birth_prop.active = True

            else:
                # pick which proposal to turn off
                i = random.choice(range(len(active_props)))
                death_prop = active_props.pop(i)
                # set the the death signal's components to nans
                out.update({p: math.nan for p in death_prop.parameters})
                death_prop.active = False

            # for all other proposals copy their last position
#            print('lens', len(active_props), len(inactive_props))
            for prop in (active_props + inactive_props):
                out.update({p: fromx[p] for p in prop.parameters})
        else:
            # do a within-model MCMC move for active proposals
            for prop in active_props:
                out.update(prop.jump({p: fromx[p] for p in prop.parameters}))
            # for inactive proposals carry on their last position (nan)
            for prop in inactive_props:
                out.update(prop.jump({p: fromx[p] for p in prop.parameters}))
        # update the model index
        out.update({self.model_parameter: newk})
#        print('final len updates props', len(self.update_proposals))
#        for prop in self.update_proposals:
#            print(prop.parameters, prop.active)
        return out

    @property
    def state(self):
        # get all of the proposals state
        state = {frozenset(prop.parameters): prop.state
                 for prop in self.all_proposals}
        # add the global random state
        state['random_state'] = self.random_state
        return state

    def set_state(self, state):
        # set each proposals' state
        for prop in self.all_proposals:
            prop.set_state(state[frozenset(prop.parameters)])
        # set the state of the random number generator
        self.random_state = state['random_state']
