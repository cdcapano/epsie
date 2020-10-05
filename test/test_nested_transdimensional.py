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

import numpy
import pytest


from epsie.proposals import (NestedTransdimensional, BoundedDiscrete, Normal,
                             SSAdaptiveNormal, AdaptiveNormal,
                             ATAdaptiveNormal, UniformBirth, NormalBirth,
                             LogNormalBirth)
from _utils import PolynomialRegressionModel
from epsie import make_betas_ladder

from test_ptsampler import _create_sampler
from test_ptsampler import test_checkpointing as _test_checkpointing
from test_ptsampler import test_seed as _test_seed
from test_ptsampler import test_clear_memory as _test_clear_memory


ITERINT = 64
ADAPTATION_DURATION = ITERINT//2
SWAP_INTERVAL = 1


def _setup_proposal(model, td_proposal, birth_dist, cov=None):
    # Initialise the birth-objects for transdimensional parameters
    if birth_dist == 'uniform':
        birth_dists = [UniformBirth(
            ['a{}'.format(i)], {'a{}'.format(i): (0., 4.)})
            for i in range(1, 5+1)]
    elif birth_dist == 'normal':
        birth_dists = [NormalBirth(
            ['a{}'.format(i)], {'a{}'.format(i): 1.0}, {'a{}'.format(i): 1.0})
            for i in range(1, 5+1)]
    elif birth_dist == 'lognormal':
        birth_dists = [LogNormalBirth(
            ['a{}'.format(i)], {'a{}'.format(i): 1.0}, {'a{}'.format(i): 0.7})
            for i in range(1, 5+1)]

    # Initialies transdimensional proposals
    if td_proposal == 'normal':
        td_proposals = [Normal(['a{}'.format(i)], cov=cov)
                        for i in range(1, 5+1)]
    elif td_proposal == 'adaptive_normal':
        widths = {'a{}'.format(i): 4 for i in range(1, 6)}
        td_proposals = [
            AdaptiveNormal(['a{}'.format(i)], widths, ADAPTATION_DURATION)
            for i in range(1, 5+1)]
    elif td_proposal == 'ss_adaptive_normal':
        td_proposals = [SSAdaptiveNormal(['a{}'.format(i)])
                        for i in range(1, 5+1)]
    elif td_proposal == 'adaptive_proposal':
        td_proposals = [ATAdaptiveNormal(
            ['a{}'.format(i)], adaptation_duration=ADAPTATION_DURATION)
            for i in range(1, 5+1)]

    # Model hopping proposal
    model_proposal = BoundedDiscrete(['k'],
                                     boundaries={'k': (0, len(td_proposals))},
                                     successive={'k': True})
    pars = ['a{}'.format(i) for i in range(1, 5+1)] + ['k']
    # Initialise the bundle transdimemnsional proposal
    return NestedTransdimensional(pars, model_proposal, td_proposals,
                                  birth_dists)


@pytest.mark.parametrize('td_proposal', ['normal', 'adaptive_normal',
                                         'ss_adaptive_normal',
                                         'adaptive_proposal'])
@pytest.mark.parametrize('birth_dist', ['uniform', 'normal', 'lognormal'])
@pytest.mark.parametrize('nprocs', [1, 4])
@pytest.mark.parametrize('nchains', [1, 4])
@pytest.mark.parametrize('ntemps', [2, 3])
def test_active_parameters(td_proposal, birth_dist, nprocs, nchains, ntemps,
                           model=None):
    """Tests that the standard deviation changes after a few jumps for the type
    of proposal specified by ``proposal_name``.
    """
    # use the test model
    if model is None:
        model = PolynomialRegressionModel()
    proposal = _setup_proposal(model, td_proposal, birth_dist)
    # we'll just use the PTSampler default setup from the ptsampler tests
    betas = make_betas_ladder(ntemps, 1e3)
    sampler = _create_sampler(model, nprocs, nchains=nchains, betas=betas,
                              proposals=[proposal])
    sampler.run(ITERINT)
    # check that the model index matched the number of finite parameters
    # at each turn
    for n in range(sampler.niterations):
        for i in range(nchains):
            for j in range(ntemps):
                coeffs = numpy.array([
                    sampler.positions[j, i, n]['a{}'.format(m)]
                    for m in range(1, 5 + 1)])
                k = sampler.positions[j, i, n]['k']
                assert numpy.isfinite(coeffs).sum() == k


@pytest.mark.parametrize('td_proposal', ['normal', 'adaptive_normal',
                                         'ss_adaptive_normal',
                                         'adaptive_proposal'])
@pytest.mark.parametrize('birth_dist', ['uniform', 'normal', 'lognormal'])
@pytest.mark.parametrize('nprocs', [1, 4])
def test_checkpointing(td_proposal, birth_dist, nprocs):
    """Performs the same checkpointing test as for the PTSampler, but using
    the adaptive normal proposal.
    """
    model = PolynomialRegressionModel()
    proposal = _setup_proposal(model, td_proposal, birth_dist)
    _test_checkpointing(PolynomialRegressionModel, nprocs,
                        proposals=[proposal])


@pytest.mark.parametrize('td_proposal', ['normal', 'adaptive_normal',
                                         'ss_adaptive_normal',
                                         'adaptive_proposal'])
@pytest.mark.parametrize('birth_dist', ['uniform', 'normal', 'lognormal'])
@pytest.mark.parametrize('nprocs', [1, 4])
def test_seed(td_proposal, birth_dist, nprocs):
    """Runs the PTSampler ``test_seed`` using the adaptive normal proposal.
    """
    model = PolynomialRegressionModel()
    proposal = _setup_proposal(model, td_proposal, birth_dist)
    _test_seed(PolynomialRegressionModel, nprocs, proposals=[proposal])


@pytest.mark.parametrize('td_proposal', ['normal', 'adaptive_normal',
                                         'ss_adaptive_normal',
                                         'adaptive_proposal'])
@pytest.mark.parametrize('birth_dist', ['uniform', 'normal', 'lognormal'])
@pytest.mark.parametrize('nprocs', [1, 4])
def test_clear_memory(td_proposal, birth_dist, nprocs):
    """Runs the PTSampler ``test_clear_memoory`` using the adaptive normal
    proposal.
    """
    model = PolynomialRegressionModel()
    proposal = _setup_proposal(model, td_proposal, birth_dist)
    _test_clear_memory(PolynomialRegressionModel, nprocs, SWAP_INTERVAL,
                       proposals=[proposal])
