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
"""Performs unit tests on the eigenvector proposals."""

from __future__ import (print_function, absolute_import)

import pytest
import numpy

from epsie.proposals import (Eigenvector, AdaptiveEigenvector)
from _utils import Model

from test_ptsampler import _create_sampler
from test_ptsampler import test_chains as _test_chains
from test_ptsampler import test_checkpointing as _test_checkpointing
from test_ptsampler import test_seed as _test_seed
from test_ptsampler import test_clear_memory as _test_clear_memory


STABILITY_DURATION = 70
ADAPTATION_DURATION = 32
SWAP_INTERVAL = 1


def _setup_proposal(model, proposal_name, params=None):
    if params is None:
        params = model.params
    if proposal_name == 'eigenvector':
        return Eigenvector(params, STABILITY_DURATION)
    elif proposal_name == 'adaptive_eigenvector':
        return AdaptiveEigenvector(params, STABILITY_DURATION,
                                   ADAPTATION_DURATION)
    else:
        return -1


@pytest.mark.parametrize('nprocs', [1, 4])
@pytest.mark.parametrize('proposal_name', ['adaptive_eigenvector'])
def test_scale_changes(nprocs, proposal_name, model=None):
    """Tests that the eigenvalues change after a few jumps for the type
    of proposal specified by ``proposal_name``.
    """
    # use the test model
    if model is None:
        model = Model()
    proposal = _setup_proposal(model, proposal_name)
    _test_scale_changes(nprocs, proposal, model)


def _test_scale_changes(nprocs, proposal, model):
    """Tests that the eigenvalues of the proposal changes after a few
    jumps.
    """
    # we'll just use the PTSampler default setup from the ptsampler tests
    sampler = _create_sampler(model, nprocs, proposals=[proposal])
    # run the sampler through its stabilisation phase
    sampler.run(STABILITY_DURATION)

    initial_eigvals = numpy.zeros((sampler.nchains, sampler.ntemps,
                                  len(proposal.parameters)))
    for ii, chain in enumerate(sampler.chains):
        for jj, subchain in enumerate(chain.chains):
            thisprop = subchain.proposal_dist.proposals[0]
            initial_eigvals[ii, jj, :] = thisprop._eigvals
    # run the sampler for the adaptation duration, and check that the standard
    # deviation of all chains and temperatures has changed
    sampler.run(ADAPTATION_DURATION)
    current_eigvals = numpy.zeros(initial_eigvals.shape)
    for ii, chain in enumerate(sampler.chains):
        for jj, subchain in enumerate(chain.chains):
            thisprop = subchain.proposal_dist.proposals[0]
            current_eigvals[ii, jj, :] = thisprop._eigvals
    assert (initial_eigvals != current_eigvals).all()
    # vea adaptive proposals shut the adaptation after the adaption duration
    if proposal.name.startswith('adaptive'):
        # now run past the adaptation duration; since we have gone past it, the
        # standard deviations should no longer change
        sampler.run(ADAPTATION_DURATION)
        previous_eigvals = current_eigvals
        current_eigvals = numpy.zeros(previous_eigvals.shape)
        for ii, chain in enumerate(sampler.chains):
            for jj, subchain in enumerate(chain.chains):
                thisprop = subchain.proposal_dist.proposals[0]
                current_eigvals[ii, jj, :] = thisprop._eigvals
        assert (previous_eigvals == current_eigvals).all()
    # close the multiprocessing pool
    if sampler.pool is not None:
        sampler.pool.close()


@pytest.mark.parametrize('nprocs', [1, 4])
@pytest.mark.parametrize('proposal_name', ['eigenvector',
                                           'adaptive_eigenvector'])
def test_chains(nprocs, proposal_name):
    """Runs the PTSampler ``test_chains`` test using the adaptive normal
    proposal.
    """
    model = Model()
    # we'll just use the adaptive normal for one of the params, to test
    # that using mixed proposals works
    proposal = _setup_proposal(model, proposal_name,
                               params=[list(model.params)[0]])
    _test_chains(Model, nprocs, SWAP_INTERVAL, proposals=[proposal],
                 init_iters=STABILITY_DURATION-2)


@pytest.mark.parametrize('nprocs', [1, 4])
@pytest.mark.parametrize('proposal_name', ['eigenvector',
                                           'adaptive_eigenvector'])
def test_checkpointing(nprocs, proposal_name):
    """Performs the same checkpointing test as for the PTSampler.
    """
    model = Model()
    proposal = _setup_proposal(model, proposal_name)
    _test_checkpointing(Model, nprocs, proposals=[proposal],
                        init_iters=STABILITY_DURATION-2)


@pytest.mark.parametrize('nprocs', [1, 4])
@pytest.mark.parametrize('proposal_name', ['eigenvector',
                                           'adaptive_eigenvector'])
def test_seed(nprocs, proposal_name):
    """Runs the PTSampler ``test_seed`` using the adaptive normal proposal.
    """
    model = Model()
    proposal = _setup_proposal(model, proposal_name)
    _test_seed(Model, nprocs, proposals=[proposal],
               init_iters=STABILITY_DURATION-2)


@pytest.mark.parametrize('nprocs', [1, 4])
@pytest.mark.parametrize('proposal_name', ['eigenvector',
                                           'adaptive_eigenvector'])
def test_clear_memory(nprocs, proposal_name):
    """Runs the PTSampler ``test_clear_memoory`` using the adaptive normal
    proposal.
    """
    model = Model()
    proposal = _setup_proposal(model, proposal_name)
    _test_clear_memory(Model, nprocs, SWAP_INTERVAL, proposals=[proposal],
                       init_iters=STABILITY_DURATION-2)
