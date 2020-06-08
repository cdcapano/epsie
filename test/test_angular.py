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

import pytest
import numpy

from epsie.proposals import proposals
from _utils import AngularModel

from test_ptsampler import test_chains as _test_chains
from test_ptsampler import test_checkpointing as _test_checkpointing
from test_ptsampler import test_seed as _test_seed
from test_ptsampler import test_clear_memory as _test_clear_memory
from test_adaptive_normal import _test_std_changes


ITERINT = 32
ADAPTATION_DURATION = ITERINT//2
SWAP_INTERVAL = 1


def _setup_proposal(name, parameters, cov=None,
                    adaptation_duration=None):
    if name == 'ss_adaptive_angular' or name == 'angular':
        return proposals[name](parameters, cov=cov)
    elif name == 'adaptive_angular':
        if adaptation_duration is None:
            adaptation_duration = ADAPTATION_DURATION
        return proposals[name](parameters, adaptation_duration)
    else:
        raise KeyError("unrecognized proposal name {}".format(name))


@pytest.mark.parametrize('proposal_name,cov',
                         [('angular', None),
                          ('angular', 2.1),
                          ('ss_adaptive_angular', None),
                          ('adaptive_angular', None)])
def test_jumps_in_bounds(proposal_name, cov):
    """Tests that all jumps are in 0, 2pi."""
    proposal = _setup_proposal(proposal_name, ['phi'], cov)
    # create 3 points to test, one on the edges and one in the middle
    test_points = numpy.array([0., numpy.pi, 2*numpy.pi])
    njumps = 1000
    jumps = numpy.zeros((len(test_points), njumps))
    for ii, xi in enumerate(test_points):
        jumps[ii, :] = numpy.array([proposal.jump({'phi': xi})['phi']
                                   for _ in range(njumps)])
    assert ((jumps >= 0) & (jumps <= 2*numpy.pi)).all()


@pytest.mark.parametrize('params,cov',
                         [(['phi'], None),
                          (['phi'], 0.5),
                          (['phi', 'theta'], None),
                          (['phi', 'theta'], [0.27, 3.14])])
def test_logpdf(params, cov):
    """Tests that the angular logpdf is indeed symmetric."""
    # 1D
    proposal = _setup_proposal('angular', params, cov)
    x0 = {p: numpy.random.uniform(0, 2*numpy.pi) for p in params}
    # get a point
    x1 = proposal.jump(x0)
    p1 = proposal.logpdf(x0, x1)
    # assert that we get a float
    assert isinstance(p1, float)
    # check that we get the same going the other way
    assert proposal.logpdf(x1, x0) == pytest.approx(p1)


@pytest.mark.parametrize('nprocs', [1, 4])
@pytest.mark.parametrize('proposal_name', ['ss_adaptive_angular',
                                           'adaptive_angular'])
def test_std_changes(nprocs, proposal_name, model=None):
    """Tests that the standard deviation changes after a few jumps for the type
    of proposal specified by ``proposal_name``.
    """
    # use the test model
    if model is None:
        model = AngularModel()
    proposal = _setup_proposal(proposal_name, model.params)
    _test_std_changes(nprocs, proposal, model)


@pytest.mark.parametrize('proposal_name', ['angular',
                                           'ss_adaptive_angular',
                                           'adaptive_angular'])
@pytest.mark.parametrize('nprocs', [1, 4])
def test_chains(proposal_name, nprocs):
    """Runs the PTSampler ``test_chains`` test using the angular proposal.
    """
    model = AngularModel()
    proposal = _setup_proposal(proposal_name, model.params)
    _test_chains(AngularModel, nprocs, SWAP_INTERVAL, proposals=[proposal])


@pytest.mark.parametrize('proposal_name', ['angular',
                                           'ss_adaptive_angular',
                                           'adaptive_angular'])
@pytest.mark.parametrize('nprocs', [1, 4])
def test_checkpointing(proposal_name, nprocs):
    """Performs the same checkpointing test as for the PTSampler, but using
    the angular proposal.
    """
    model = AngularModel()
    proposal = _setup_proposal(proposal_name, model.params)
    _test_checkpointing(AngularModel, nprocs, proposals=[proposal])


@pytest.mark.parametrize('proposal_name', ['angular',
                                           'ss_adaptive_angular',
                                           'adaptive_angular'])
@pytest.mark.parametrize('nprocs', [1, 4])
def test_seed(proposal_name, nprocs):
    """Runs the PTSampler ``test_seed`` using the angular proposal.
    """
    model = AngularModel()
    proposal = _setup_proposal(proposal_name, model.params)
    _test_seed(AngularModel, nprocs, proposals=[proposal])


@pytest.mark.parametrize('proposal_name', ['angular',
                                           'ss_adaptive_angular',
                                           'adaptive_angular'])
@pytest.mark.parametrize('nprocs', [1, 4])
def test_clear_memory(proposal_name, nprocs):
    """Runs the PTSampler ``test_clear_memoory`` using the angular proposal.
    """
    model = AngularModel()
    proposal = _setup_proposal(proposal_name, model.params)
    _test_clear_memory(AngularModel, nprocs, SWAP_INTERVAL,
                       proposals=[proposal])
