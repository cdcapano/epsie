#!/usr/bin/env python

# Copyright (C) 2020  Richard Stiskalek, Collin Capano
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
"""Performs unit tests on the NestedTransdimensional proposal."""

from __future__ import (print_function, absolute_import)

import itertools
from scipy import stats
import pytest
import numpy

try:
    from randomgen import RandomGenerator
except ImportError:
    from randomgen import Generator as RandomGenerator

from epsie.proposals import (NestedTransdimensional, Normal, BoundedDiscrete)
from _utils import PolynomialRegressionModel
from epsie import make_betas_ladder

from test_ptsampler import _create_sampler
# from test_ptsampler import test_chains as _test_chains
# from test_ptsampler import test_checkpointing as _test_checkpointing
# from test_ptsampler import test_seed as _test_seed
# from test_ptsampler import test_clear_memory as _test_clear_memory


ITERINT = 64
ADAPTATION_DURATION = ITERINT//2
SWAP_INTERVAL = 1


# Move this elsewhere later
class UniformBirthDistribution(object):
    _random_generator = None

    def __init__(self, parameters, bounds):
        self.parameters = parameters
        self.bounds = bounds

    def set_bit_generator(self, bit_generator):
        self._random_generator = RandomGenerator(bit_generator)

    @property
    def birth(self):
        if self._random_generator is None:
            raise ValueError('must set the random generator first')
        return {p: self._random_generator.uniform(self.bounds[p][0],
                                                  self.bounds[p][1])
                for p in self.parameters}

    def logpdf(self, xi):
        return sum([stats.uniform.logpdf(xi[p], loc=self.bounds[p][0],
                                         scale=(self.bounds[p][1]
                                                - self.bounds[p][0]))
                    for p in self.parameters])


# ADD ADAPTATION
def _setup_proposal(model, cov=None):
    # Initialise the birth-objects for transdimensional parameters
    birth_dists = [UniformBirthDistribution(['a{}'.format(i)],
                                            {'a{}'.format(i): (-2., 2.)})
                   for i in range(1, 5+1)]
    # Initialies transdimensional proposals
    td_proposals = [Normal(['a{}'.format(i)], cov=cov) for i in range(1, 5+1)]
    # Initialise the global proposal that is always on
    global_proposals = [Normal(['a0'], cov=cov)]
    # Model hopping proposal
    model_proposal = BoundedDiscrete(['k'],
                                     boundaries={'k': (0, len(td_proposals))},
                                     successive={'k': True})
    pars = ['a{}'.format(i) for i in range(1, 5+1)] + ['k']
    # Initialise the bundle transdimemnsional proposal
    return NestedTransdimensional(pars, model_proposal, td_proposals,
                                  birth_dists)


@pytest.mark.parametrize('nprocs', [1, 4])
@pytest.mark.parametrize('nchains', [1, 4])
@pytest.mark.parametrize('ntemps', [2, 3])
def test_something(nprocs, nchains, ntemps, model=None):
    """Tests that the standard deviation changes after a few jumps for the type
    of proposal specified by ``proposal_name``.
    """
    # use the test model
    if model is None:
        model = PolynomialRegressionModel()
    proposal = _setup_proposal(model)
    # we'll just use the PTSampler default setup from the ptsampler tests
    betas = make_betas_ladder(ntemps, 1e3)
    sampler = _create_sampler(model, nprocs, nchains=nchains, betas=betas,
                              proposals=[proposal], set_start=False)
    # set the starting point
    sampler.start_position = model.prior_rvs(nchains, ntemps)
    sampler.run(ITERINT)
