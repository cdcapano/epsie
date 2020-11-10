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
"""Performs unit tests on the bounded eigenvector proposals."""

from __future__ import (absolute_import, division)

import pytest

from scipy import stats

from epsie.proposals import (BoundedEigenvector, AdaptiveBoundedEigenvector)
from epsie.proposals.bounded_normal import Boundaries
from _utils import Model

from test_ptsampler import _create_sampler
from test_ptsampler import test_checkpointing as _test_checkpointing
from test_ptsampler import test_seed as _test_seed
from test_ptsampler import test_clear_memory as _test_clear_memory


STABILITY_DURATION = 64
ADAPTATION_DURATION = 16
SWAP_INTERVAL = 1


def _setup_proposal(model, proposal_name, boundaries, params=None):
    if params is None:
        params = model.params
    if proposal_name == 'bounded_eigenvector':
        return BoundedEigenvector(params, boundaries, STABILITY_DURATION)
    elif proposal_name == 'adaptive_bounded_eigenvector':
        return AdaptiveBoundedEigenvector(
            params, boundaries, STABILITY_DURATION, ADAPTATION_DURATION)
    else:
        raise KeyError("unrecognized proposal name {}".format(proposal_name))


@pytest.mark.parametrize('proposal_name', ['bounded_eigenvector',
                                           'adaptive_bounded_eigenvector'])
@pytest.mark.parametrize('xmin,xmax', [(-1, 1), (1.2, 2.8)])
def test_jumps_in_bounds(proposal_name, xmin, xmax):
    """Runs the proposal for a couple of steps and checks that suggested
    eigenvector jumps are in bounds.
    """
    model = Model()
    print(xmin)
    bnd = Boundaries((xmin, xmax))
    model.prior_bounds.update({'x0': bnd})
    model.prior_dist.update({'x0': stats.uniform(bnd.lower, abs(bnd))})

    bounds = {'x0': (xmin, xmax), 'x1': (-40., 40.0)}
    # create the proposal and the sampler
    proposal = _setup_proposal(model, proposal_name, bounds)
    sampler = _create_sampler(model, nprocs=1, proposals=[proposal])
    # Run the sampler through its stability and adaptation phase
    # This will throw an exception in case any suggested points out of bounds
    sampler.run(STABILITY_DURATION + ADAPTATION_DURATION)


@pytest.mark.parametrize('proposal_name', ['bounded_eigenvector',
                                           'adaptive_bounded_eigenvector'])
@pytest.mark.parametrize('nprocs', [1, 4])
def test_checkpointing(proposal_name, nprocs):
    """Performs the same checkpointing test as for the PTSampler, but using
    the adaptive normal proposal.
    """
    model = Model()
    proposal = _setup_proposal(model, proposal_name, model.prior_bounds)
    _test_checkpointing(Model, nprocs, proposals=[proposal],
                        init_iters=STABILITY_DURATION)


@pytest.mark.parametrize('proposal_name', ['bounded_eigenvector',
                                           'adaptive_bounded_eigenvector'])
@pytest.mark.parametrize('nprocs', [1, 4])
def test_seed(proposal_name, nprocs):
    """Runs the PTSampler ``test_seed`` using the adaptive normal proposal.
    """
    model = Model()
    proposal = _setup_proposal(model, proposal_name, model.prior_bounds)
    _test_seed(Model, nprocs, proposals=[proposal],
               init_iters=STABILITY_DURATION)


@pytest.mark.parametrize('proposal_name', ['bounded_eigenvector',
                                           'adaptive_bounded_eigenvector'])
@pytest.mark.parametrize('nprocs', [1, 4])
def test_clear_memory(proposal_name, nprocs):
    """Runs the PTSampler ``test_clear_memoory`` using the adaptive normal
    proposal.
    """
    model = Model()
    proposal = _setup_proposal(model, proposal_name, model.prior_bounds)
    _test_clear_memory(Model, nprocs, SWAP_INTERVAL, proposals=[proposal],
                       init_iters=STABILITY_DURATION)
