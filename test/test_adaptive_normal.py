#!/usr/bin/env python

# Copyright (C) 2020  Collin Capano
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
"""Performs unit tests on the AdaptiveNormal proposal."""

from __future__ import (print_function, absolute_import)

import itertools
import pytest
import numpy

from epsie.proposals import AdaptiveNormal
from _utils import Model

from test_ptsampler import _create_sampler
from test_ptsampler import test_chains as _test_chains
from test_ptsampler import test_checkpointing as _test_checkpointing
from test_ptsampler import test_seed as _test_seed
from test_ptsampler import test_clear_memory as _test_clear_memory


ITERINT = 32
ADAPTATION_DURATION = ITERINT//2
SWAP_INTERVAL = 1


def _setup_proposal(model, adaptation_duration=None):
    if adaptation_duration is None:
        adaptation_duration = ADAPTATION_DURATION
    prior_widths = {p: abs(bnds[1] - bnds[0])
                    for p, bnds in model.prior_bounds.items()}
    return AdaptiveNormal(model.params, prior_widths,
                          adaptation_duration=adaptation_duration)


@pytest.mark.parametrize('nprocs', [1, 4])
def test_std_changes(nprocs, proposal=None, model=None):
    """Tests that the standard deviation of the proposal changes after a few
    jumps.
    """
    # use the test model
    if model is None:
        model = Model()
    if proposal is None:
        # create an adaptive normal instance
        proposal = _setup_proposal(model)
    # check the proposal's parameters option matches the model's
    assert sorted(proposal.parameters) == sorted(model.params)
    # we'll just use the PTSampler default setup from the ptsampler tests
    sampler = _create_sampler(model, nprocs,
                              proposals={tuple(model.params): proposal})
    # check that all temperatures and all chains have the same initial
    # standard deviation as the proposal
    initial_std = numpy.zeros((sampler.nchains, sampler.ntemps,
                               len(proposal.parameters)))
    for ii, chain in enumerate(sampler.chains):
        for jj, subchain in enumerate(chain.chains):
            thisprop = subchain.proposal_dist.proposals[tuple(model.params)]
            assert (thisprop.std == proposal.std).all()
            initial_std[ii, jj, :] = thisprop.std
    # run the sampler for the adaptation duration, and check that the standard
    # deviation of all chains and temperatures has changed
    sampler.run(ADAPTATION_DURATION)
    current_std = numpy.zeros(initial_std.shape)
    for ii, chain in enumerate(sampler.chains):
        for jj, subchain in enumerate(chain.chains):
            thisprop = subchain.proposal_dist.proposals[tuple(model.params)]
            current_std[ii, jj, :] = thisprop.std
    assert (initial_std != current_std).all()
    # now run past the adaptation duration; since we have gone past it, the
    # standard deviations should no longer change
    sampler.run(ITERINT//2)
    previous_std = current_std
    current_std = numpy.zeros(previous_std.shape)
    for ii, chain in enumerate(sampler.chains):
        for jj, subchain in enumerate(chain.chains):
            thisprop = subchain.proposal_dist.proposals[tuple(model.params)]
            current_std[ii, jj, :] = thisprop.std
    assert (previous_std == current_std).all()
    # close the multiprocessing pool
    if sampler.pool is not None:
        sampler.pool.close()


@pytest.mark.parametrize('nprocs', [1, 4])
def test_chains(nprocs):
    """Runs the PTSampler ``test_chains`` test using the adaptive normal
    proposal.
    """
    model = Model()
    proposal = _setup_proposal(model)
    _test_chains(Model, nprocs, SWAP_INTERVAL,
                 proposals={tuple(model.params): proposal})


@pytest.mark.parametrize('nprocs', [1, 4])
def test_checkpointing(nprocs):
    """Performs the same checkpointing test as for the PTSampler, but using
    the adaptive normal proposal.
    """
    model = Model()
    proposal = _setup_proposal(model)
    _test_checkpointing(Model, nprocs,
                        proposals={tuple(model.params): proposal})


@pytest.mark.parametrize('nprocs', [1, 4])
def test_seed(nprocs):
    """Runs the PTSampler ``test_seed`` using the adaptive normal proposal.
    """
    model = Model()
    proposal = _setup_proposal(model)
    _test_seed(Model, nprocs,
               proposals={tuple(model.params): proposal})


@pytest.mark.parametrize('nprocs', [1, 4])
def test_clear_memory(nprocs):
    """Runs the PTSampler ``test_clear_memoory`` using the adaptive normal
    proposal.
    """
    model = Model()
    proposal = _setup_proposal(model)
    _test_clear_memory(Model, nprocs, SWAP_INTERVAL,
                       proposals={tuple(model.params): proposal})
