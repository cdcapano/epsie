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

import itertools

import numpy
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
    symmetric = True

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
            death = min(1, numpy.exp(self.jump_proposal(k - 1) - current))
            death *= self.jump_freq

        if k == self.boundary.upper:
            birth = 0.0
        else:
            birth = min(1, numpy.exp(self.jump_proposal(k + 1) - current))
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
    def __init__(self, bd_proposal, prior_dist, *update_props, **kwargs):
        bit_generator = kwargs.pop('bit_generator', None)  #Py3XX: delete line
        update_parameters = list(itertools.chain(*[prop.parameters\
                                            for prop in update_props]))
        # check that we don't have multiple proposals for the same parameter
        repeated = [p for p in set(update_parameters)
                    if update_parameters.count(p) > 1]
        if repeated:
            raise ValueError("multiple update_props provided "
                             "for parameter(s) {}".format(
                                 ', '.join(repeated)))
        # store parameters
        self.update_parameters = update_parameters
        self.model_parameter = bd_proposal.parameter
        self.parameters = update_parameters + [self.model_parameter]
        # store proposals
        self.bd_proposal = bd_proposal
        self.model_proposals = update_props
        self._all_proposals = list(update_props) + [self.bd_proposal]
        # the proposal is symmetric only if all of the constitutent
        # proposals are also
#        self._symmetric = all([prop.symmetric
#                              for prop in self._all_proposals])
        self._symmetric = False
        # store the prior distribution to sample new components on the go
        self._prior_dist = None
        self.prior_dist = prior_dist
        # set the bit generator
        self.bit_generator = bit_generator
        # have all of the proposals use the same random state
        for prop in self._all_proposals:
            prop.bit_generator = self.bit_generator


        self._inact_props = [prop for prop in update_props if not prop.active]
        self._act_props = [prop for prop in update_props if prop.active]

        self._last_prop = None

    @property
    def symmetric(self):
        return self._symmetric

    def logpdf(self, xi, givenx):
        lp = 0.0
        for prop in self._all_proposals:
                lp_here = prop.logpdf({p: xi[p] for p in prop.parameters},
                                  {p: givenx[p] for p in prop.parameters})
                if not numpy.isnan(lp_here):
                    lp += lp_here
        return lp

    def update(self, chain):
        pass
        # update each of the proposals
#        if chain.acceptance[-1]['accepted'] and not (self._last_prop is None):
#            if self._last_prop.active:
#                self._inact_props.append(self._last_prop)
#                self._act_props.remove(self._last_prop)
#            else:
#                self._act_props.append(self._last_prop)
#                self._inact_props.remove(self._last_prop)
#
#            self._last_prop.active = not self._last_prop.active
#        self._last_prop = None # last_prop back to None just to be sure

        # turn this back on later but need to check what happens to adaptation
        # if proposals are getting turned off and on. Do adaptation only on
        # proposals that just did a within-model MCMC step?
        # Also what we could do is introduce an adaption of the mean of the
        # poisson distribution that decides where to jump next.

#        for prop in self._all_proposals:
#            prop.update(chain)

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
        newk = self.bd_proposal.jump(
               {self.model_parameter: fromx[self.model_parameter]})
        out = fromx.copy()
        act_props = [prop for prop in self.model_proposals
                     if not numpy.alltrue(numpy.isnan([fromx[p] for p in prop.parameters]))]
        inact_props = [prop for prop in self.model_proposals
                     if numpy.alltrue(numpy.isnan([fromx[p] for p in prop.parameters]))]


        # decide whether a dimension change will occur
        if newk[self.model_parameter] != fromx[self.model_parameter]:
            # decide whether birth
            if newk[self.model_parameter] > fromx[self.model_parameter]:
                last_prop = self.random_generator.choice(inact_props)
                # sample the prior
                out.update({p: self.prior_dist[p].rvs()
                            for p in last_prop.parameters})
            # else death
            else:
                last_prop = self.random_generator.choice(act_props)
                # set the removed proposal parameters to nans
                out.update({p: numpy.nan for p in last_prop.parameters})
            # update the model index
            out.update({self.model_parameter: newk[self.model_parameter]})
        else:
            # do a within-model MCMC move for active proposals
            for prop in act_props:
                out.update(prop.jump({p: fromx[p] for p in prop.parameters}))
            self.last_prop = None
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
        for prop in self._all_proposals:
            prop.set_state(state[frozenset(prop.parameters)])
        # set the state of the random number generator
        self.random_state = state['random_state']