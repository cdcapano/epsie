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
"""Performs unit tests on the NormalDiscrete, BoundedDiscrete, proposals along
with their adaptive versions.
"""

from __future__ import (absolute_import, division)

import pytest
import numpy

from epsie.proposals import proposals
from _utils import PoissonModel

from test_ptsampler import test_chains as _test_chains
from test_ptsampler import test_checkpointing as _test_checkpointing
from test_ptsampler import test_seed as _test_seed
from test_ptsampler import test_clear_memory as _test_clear_memory
from test_adaptive_normal import test_std_changes as _test_std_changes


ITERINT = 32
ADAPTATION_DURATION = ITERINT//2
SWAP_INTERVAL = 1


def _setup_proposal(name, parameters, boundaries=None, cov=None,
                    adaptation_duration=None):
    if name.startswith('adaptive'):
        if adaptation_duration is None:
            adaptation_duration = ADAPTATION_DURATION
        return proposals[name](parameters, boundaries,
                               adaptation_duration)
    if name == 'bounded_discrete':
        return proposals[name](parameters, boundaries, cov=cov)
    elif name == 'discrete':
        return proposals[name](parameters, cov=cov)
    else:
        raise KeyError("unrecognized proposal name {}".format(name))


@pytest.mark.parametrize('proposal_name,cov',
                         [('bounded_discrete', None),
                          ('bounded_discrete', 4),
                          ('adaptive_bounded_discrete', None)])
@pytest.mark.parametrize('kmin,kmax',
                         [(0, 16), (1, 11)])
def test_jumps_in_bounds(proposal_name, cov, kmin, kmax):
    """Tests that all jumps are integers and in the given bounds.

    Also tests that the proposals can handle parameter names longer than 1.
    """
    proposal = _setup_proposal(proposal_name, ['kappa'],
                               {'kappa': (kmin, kmax)}, cov)
    # check the the number of dimensions is 1
    assert len(proposal.parameters) == 1
    assert proposal.parameters == ('kappa',)
    # create 3 points to test, one on the edges and one in the middle
    kmin, kmax = proposal.boundaries['kappa']
    test_points = numpy.array([kmin, (kmin+kmax)//2, kmax])
    njumps = 1000
    jumps = numpy.zeros((len(test_points), njumps), dtype=int)
    for ii, ki in enumerate(test_points):
        js = numpy.array([proposal.jump({'kappa': ki})['kappa']
                          for _ in range(njumps)])
        # check that all jumps are integers
        assert js.dtype == numpy.int
        jumps[ii, :] = js
    assert ((jumps >= kmin) & (jumps <= kmax)).all()


@pytest.mark.parametrize('proposal_name', ['discrete',
                                           'adaptive_discrete',
                                           'bounded_discrete',
                                           'adaptive_bounded_discrete'])
def test_logpdf(proposal_name):
    """Tests that the logpdf function is constant for two values with the same
    integer value.
    """
    model = PoissonModel()
    proposal = _setup_proposal(proposal_name, model.params,
                               model.prior_bounds)
    givenpt = 3
    for testpt in [1.2, 4.3]:
        logp0 = proposal.logpdf({'k': testpt}, {'k': givenpt})
        logp1 = proposal.logpdf({'k': testpt+0.5}, {'k': givenpt})
        assert isinstance(logp0, float)
        assert numpy.isfinite(logp0)
        assert isinstance(logp1, float)
        assert numpy.isfinite(logp1)
        assert logp0 == logp1
    # test that a jump less than 1 from the given point has 0 probability
    assert proposal.logpdf({'k': givenpt+0.4}, {'k': givenpt}) == -numpy.inf


@pytest.mark.parametrize('nprocs', [1, 4])
@pytest.mark.parametrize('proposal_name', ['adaptive_discrete',
                                           'adaptive_bounded_discrete'])
def test_std_changes(nprocs, proposal_name):
    """Tests that the standard deviation of the adaptive proposals change
    after a few jumps.
    """
    model = PoissonModel()
    proposal = _setup_proposal(proposal_name, model.params,
                               model.prior_bounds)
    _test_std_changes(nprocs, proposal=proposal, model=model)


@pytest.mark.parametrize('proposal_name', ['discrete',
                                           'adaptive_discrete',
                                           'bounded_discrete',
                                           'adaptive_bounded_discrete'])
@pytest.mark.parametrize('nprocs', [1, 4])
def test_chains(proposal_name, nprocs):
    """Runs the PTSampler ``test_chains`` test using the bounded normal
    proposal.
    """
    model = PoissonModel()
    proposal = _setup_proposal(proposal_name, model.params, model.prior_bounds)
    _test_chains(PoissonModel, nprocs, SWAP_INTERVAL, proposals=[proposal])


@pytest.mark.parametrize('proposal_name', ['discrete',
                                           'adaptive_discrete',
                                           'bounded_discrete',
                                           'adaptive_bounded_discrete'])
@pytest.mark.parametrize('nprocs', [1, 4])
def test_checkpointing(proposal_name, nprocs):
    """Performs the same checkpointing test as for the PTSampler, but using
    the adaptive normal proposal.
    """
    model = PoissonModel()
    proposal = _setup_proposal(proposal_name, model.params, model.prior_bounds)
    _test_checkpointing(PoissonModel, nprocs, proposals=[proposal])


@pytest.mark.parametrize('proposal_name', ['discrete',
                                           'adaptive_discrete',
                                           'bounded_discrete',
                                           'adaptive_bounded_discrete'])
@pytest.mark.parametrize('nprocs', [1, 4])
def test_seed(proposal_name, nprocs):
    """Runs the PTSampler ``test_seed`` using the adaptive normal proposal.
    """
    model = PoissonModel()
    proposal = _setup_proposal(proposal_name, model.params, model.prior_bounds)
    _test_seed(PoissonModel, nprocs, proposals=[proposal])


@pytest.mark.parametrize('proposal_name', ['discrete',
                                           'adaptive_discrete',
                                           'bounded_discrete',
                                           'adaptive_bounded_discrete'])
@pytest.mark.parametrize('nprocs', [1, 4])
def test_clear_memory(proposal_name, nprocs):
    """Runs the PTSampler ``test_clear_memoory`` using the adaptive normal
    proposal.
    """
    model = PoissonModel()
    proposal = _setup_proposal(proposal_name, model.params, model.prior_bounds)
    _test_clear_memory(PoissonModel, nprocs, SWAP_INTERVAL,
                       proposals=[proposal])
