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
"""Performs unit tests on AdaptiveProposal"""

from __future__ import (print_function, absolute_import)

import pytest
import numpy

from epsie.proposals import (ATAdaptiveNormal, ATAdaptiveBoundedNormal,
                             ATAdaptiveAngular)
from _utils import Model

from test_ptsampler import _create_sampler
from test_ptsampler import test_chains as _test_chains
from test_ptsampler import test_checkpointing as _test_checkpointing
from test_ptsampler import test_seed as _test_seed
from test_ptsampler import test_clear_memory as _test_clear_memory


ITERINT = 32
ADAPTATION_DURATION = ITERINT//2
SWAP_INTERVAL = 1


def _setup_proposal(model, name, params=None, diagonal=False,
                    componentwise=False, start_iteration=1,
                    adaptation_duration=ADAPTATION_DURATION):
    if params is None:
        params = model.params
    if name == 'at_adaptive_normal':
        return ATAdaptiveNormal(params,
                                adaptation_duration=adaptation_duration,
                                diagonal=diagonal, componentwise=componentwise)
    elif name == 'at_adaptive_bounded_normal':
        boundaries = {'x0': (-20., 20.), 'x1': (-40., 40.)}
        return ATAdaptiveBoundedNormal(params, boundaries=boundaries,
                                       componentwise=componentwise,
                                       adaptation_duration=adaptation_duration)
    elif name == 'at_adaptive_angular':
        return ATAdaptiveAngular(params, componentwise=componentwise,
                                 adaptation_duration=adaptation_duration)
    else:
        raise ValueError('invalid proposal')


@pytest.mark.parametrize('name', ['at_adaptive_normal',
                                  'at_adaptive_bounded_normal',
                                  'at_adaptive_angular'])
@pytest.mark.parametrize('nprocs', [1, 4])
@pytest.mark.parametrize('diagonal', [False, True])
@pytest.mark.parametrize('componentwise', [False, True])
def test_cov_changes(name, nprocs, diagonal, componentwise, model=None):
    """Tests that the covariance changes after a few jumps."""
    # use the test model
    if not diagonal and name in ['at_adaptive_bounded_normal',
                                 'at_adaptive_angular']:
        return
    if model is None:
        model = Model()
    proposal = _setup_proposal(model, name, diagonal=diagonal,
                               componentwise=componentwise)
    if proposal.isdiagonal:
        _test_std_changes(nprocs, proposal, model)
    else:
        _test_cov_changes(nprocs, proposal, model)


def _test_cov_changes(nprocs, proposal, model):
    """Tests that the covariance changes after a few jumps."""
    # we'll just use the PTSampler default setup from the ptsampler tests
    sampler = _create_sampler(model, nprocs, proposals=[proposal])
    # check that all temperatures and all chains have the same initial cov
    N = len(proposal.parameters)
    initial_cov = numpy.zeros((sampler.nchains, sampler.ntemps, N, N))
    for ii, ptchain in enumerate(sampler.chains):
        for jj, subchain in enumerate(ptchain.chains):
            thisprop = subchain.proposal_dist.proposals[0]
            assert(thisprop.cov == proposal.cov).all()
            initial_cov[ii, jj, :, :] = thisprop.cov

    # run the sampler for the adaptation duration, and check that the
    # covariance  of all chains and temperatures has changed
    sampler.run(ADAPTATION_DURATION)
    current_cov = numpy.zeros((sampler.nchains, sampler.ntemps, N, N))
    for ii, ptchain in enumerate(sampler.chains):
        for jj, subchain in enumerate(ptchain.chains):
            thisprop = subchain.proposal_dist.proposals[0]
            current_cov[ii, jj, :, :] = thisprop.cov
    assert (initial_cov != current_cov).all()

    # the adaptive proposal supports shutting down the adaptation after
    # the adaptation duratio. Now run past the adaptation duration
    sampler.run(ITERINT//2)
    previous_cov = current_cov
    current_cov = numpy.zeros((sampler.nchains, sampler.ntemps, N, N))
    for ii, ptchain in enumerate(sampler.chains):
        for jj, subchain in enumerate(ptchain.chains):
            thisprop = subchain.proposal_dist.proposals[0]
            current_cov[ii, jj, :, :] = thisprop.cov

    assert (previous_cov == current_cov).all()
    # close the multiprocessing pool
    if sampler.pool is not None:
        sampler.pool.close()


def _test_std_changes(nprocs, proposal, model):
    """Tests that the covariance changes after a few jumps."""
    # we'll just use the PTSampler default setup from the ptsampler tests
    sampler = _create_sampler(model, nprocs, proposals=[proposal])
    # check that all temperatures and all chains have the same initial cov
    N = len(proposal.parameters)
    initial_std = numpy.zeros((sampler.nchains, sampler.ntemps, N))
    for ii, ptchain in enumerate(sampler.chains):
        for jj, subchain in enumerate(ptchain.chains):
            thisprop = subchain.proposal_dist.proposals[0]
            assert(thisprop.std == proposal.std).all()
            initial_std[ii, jj, :] = thisprop.std

    # run the sampler for the adaptation duration, and check that the
    # covariance  of all chains and temperatures has changed
    sampler.run(ADAPTATION_DURATION)
    current_std = numpy.zeros((sampler.nchains, sampler.ntemps, N))
    for ii, ptchain in enumerate(sampler.chains):
        for jj, subchain in enumerate(ptchain.chains):
            thisprop = subchain.proposal_dist.proposals[0]
            current_std[ii, jj, :] = thisprop.std
    assert (initial_std != current_std).all()

    # the adaptive proposal supports shutting down the adaptation after
    # the adaptation duratio. Now run past the adaptation duration
    sampler.run(ITERINT//2)
    previous_std = current_std
    current_std = numpy.zeros((sampler.nchains, sampler.ntemps, N))
    for ii, ptchain in enumerate(sampler.chains):
        for jj, subchain in enumerate(ptchain.chains):
            thisprop = subchain.proposal_dist.proposals[0]
            current_std[ii, jj, ] = thisprop.std

    assert (previous_std == current_std).all()
    # close the multiprocessing pool
    if sampler.pool is not None:
        sampler.pool.close()


@pytest.mark.parametrize('name', ['at_adaptive_normal',
                                  'at_adaptive_bounded_normal',
                                  'at_adaptive_angular'])
@pytest.mark.parametrize('nprocs', [1, 4])
def test_chains(name, nprocs):
    """Runs the PTSampler ``test_chains`` test using the adaptive proposal.
    """
    model = Model()
    # we'll just use the adaptive normal for one of the params, to test
    # that using mixed proposals works
    proposal = _setup_proposal(model, name,  params=[list(model.params)[0]])
    _test_chains(Model, nprocs, SWAP_INTERVAL, proposals=[proposal])


@pytest.mark.parametrize('name', ['at_adaptive_normal',
                                  'at_adaptive_bounded_normal',
                                  'at_adaptive_angular'])
@pytest.mark.parametrize('nprocs', [1, 4])
def test_checkpointing(name, nprocs):
    """Performs the same checkpointing test as for the PTSampler, but using
    the adaptive proposal.
    """
    model = Model()
    proposal = _setup_proposal(model, name)
    _test_checkpointing(Model, nprocs, proposals=[proposal])


@pytest.mark.parametrize('name', ['at_adaptive_normal',
                                  'at_adaptive_bounded_normal',
                                  'at_adaptive_angular'])
@pytest.mark.parametrize('nprocs', [1, 4])
@pytest.mark.parametrize('componentwise', [False, True])
def test_seed(name, nprocs, componentwise):
    """Runs the PTSampler ``test_seed`` using the adaptive proposal.
    """
    model = Model()
    proposal = _setup_proposal(model, name, componentwise=componentwise)
    _test_seed(Model, nprocs, proposals=[proposal])


@pytest.mark.parametrize('name', ['at_adaptive_normal',
                                  'at_adaptive_bounded_normal',
                                  'at_adaptive_angular'])
@pytest.mark.parametrize('nprocs', [1, 4])
def test_clear_memory(name, nprocs):
    """Runs the PTSampler ``test_clear_memoory`` using the adaptive proposal.
    """
    model = Model()
    proposal = _setup_proposal(model, name)
    _test_clear_memory(Model, nprocs, SWAP_INTERVAL, proposals=[proposal])
