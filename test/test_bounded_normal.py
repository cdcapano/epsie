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
"""Performs unit tests on the BoundedNormal and AdaptiveBoundedNormal
proposal.
"""

from __future__ import (absolute_import, division)

import itertools
import pytest
import numpy

from epsie.proposals import proposals
from _utils import Model

from test_ptsampler import _create_sampler
from test_ptsampler import test_chains as _test_chains
from test_ptsampler import test_checkpointing as _test_checkpointing
from test_ptsampler import test_seed as _test_seed
from test_ptsampler import test_clear_memory as _test_clear_memory
from test_adaptive_normal import _test_std_changes


ITERINT = 32
ADAPTATION_DURATION = ITERINT//2
SWAP_INTERVAL = 1


def _setup_proposal(name, parameters, boundaries, cov=None,
                    adaptation_duration=None):
    if name == 'bounded_normal' or name == 'ss_adaptive_bounded_normal':
        return proposals[name](parameters, boundaries, cov=cov)
    elif name == 'adaptive_bounded_normal':
        if adaptation_duration is None:
            adaptation_duration = ADAPTATION_DURATION
        return proposals[name](parameters, boundaries,
                               adaptation_duration)
    else:
        raise KeyError("unrecognized proposal name {}".format(name))


@pytest.mark.parametrize('proposal_name,cov',
                         [('bounded_normal', None),
                          ('bounded_normal', 4),
                          ('ss_adaptive_bounded_normal', None),
                          ('adaptive_bounded_normal', None)])
@pytest.mark.parametrize('xmin,xmax',
                         [(-1, 1), (1.2, 2.8), (-42, -23)])
def test_jumps_in_bounds(proposal_name, cov, xmin, xmax):
    """Tests that all jumps are in the given bounds."""
    proposal = _setup_proposal(proposal_name, ['x'], {'x': (xmin, xmax)}, cov)
    # create 3 points to test, one on the edges and one in the middle
    xmin, xmax = proposal.boundaries['x']
    test_points = numpy.array([xmin, (xmin+xmax)/2, xmax])
    njumps = 1000
    jumps = numpy.zeros((len(test_points), njumps))
    for ii, xi in enumerate(test_points):
        jumps[ii, :] = numpy.array([proposal.jump({'x': xi})['x']
                                   for _ in range(njumps)])
    assert ((jumps >= xmin) & (jumps <= xmax)).all()


@pytest.mark.parametrize('nprocs', [1, 4])
@pytest.mark.parametrize('proposal_name', ['ss_adaptive_bounded_normal',
                                           'adaptive_bounded_normal'])
def test_std_changes(nprocs, proposal_name, model=None):
    """Tests that the standard deviation changes after a few jumps for the type
    of proposal specified by ``proposal_name``.
    """
    # use the test model
    if model is None:
        model = Model()
    proposal = _setup_proposal(proposal_name, model.params, model.prior_bounds)
    _test_std_changes(nprocs, proposal, model)


@pytest.mark.parametrize('proposal_name', ['bounded_normal',
                                           'ss_adaptive_bounded_normal',
                                           'adaptive_bounded_normal'])
@pytest.mark.parametrize('nprocs', [1, 4])
def test_chains(proposal_name, nprocs):
    """Runs the PTSampler ``test_chains`` test using the bounded normal
    proposal.
    """
    model = Model()
    # we'll just the bounded normal proposal for one of the parameters,
    # to test that using mixed proposals works
    param = list(model.params)[0]
    proposal = _setup_proposal(proposal_name, [param],
                               {param: model.prior_bounds[param]})
    _test_chains(Model, nprocs, SWAP_INTERVAL, proposals=[proposal])


@pytest.mark.parametrize('proposal_name', ['bounded_normal',
                                           'ss_adaptive_bounded_normal',
                                           'adaptive_bounded_normal'])
@pytest.mark.parametrize('nprocs', [1, 4])
def test_checkpointing(proposal_name, nprocs):
    """Performs the same checkpointing test as for the PTSampler, but using
    the adaptive normal proposal.
    """
    model = Model()
    proposal = _setup_proposal(proposal_name, model.params, model.prior_bounds)
    _test_checkpointing(Model, nprocs, proposals=[proposal])


@pytest.mark.parametrize('proposal_name', ['bounded_normal',
                                           'ss_adaptive_bounded_normal',
                                           'adaptive_bounded_normal'])
@pytest.mark.parametrize('nprocs', [1, 4])
def test_seed(proposal_name, nprocs):
    """Runs the PTSampler ``test_seed`` using the adaptive normal proposal.
    """
    model = Model()
    proposal = _setup_proposal(proposal_name, model.params, model.prior_bounds)
    _test_seed(Model, nprocs, proposals=[proposal])


@pytest.mark.parametrize('proposal_name', ['bounded_normal',
                                           'ss_adaptive_bounded_normal',
                                           'adaptive_bounded_normal'])
@pytest.mark.parametrize('nprocs', [1, 4])
def test_clear_memory(proposal_name, nprocs):
    """Runs the PTSampler ``test_clear_memoory`` using the adaptive normal
    proposal.
    """
    model = Model()
    proposal = _setup_proposal(proposal_name, model.params, model.prior_bounds)
    _test_clear_memory(Model, nprocs, SWAP_INTERVAL, proposals=[proposal])
