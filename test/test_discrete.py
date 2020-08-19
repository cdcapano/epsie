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
from test_adaptive_normal import _test_std_changes


ITERINT = 32
ADAPTATION_DURATION = ITERINT//2
SWAP_INTERVAL = 1


def _setup_proposal(name, parameters, boundaries=None, cov=None,
                    successive=None, adaptation_duration=None):
    if name.startswith('adaptive'):
        if adaptation_duration is None:
            adaptation_duration = ADAPTATION_DURATION
        return proposals[name](parameters, boundaries,
                               successive=successive,
                               adaptation_duration=adaptation_duration)
    if 'bounded' in name:
        return proposals[name](parameters, boundaries, cov=cov,
                               successive=successive)
    else:
        return proposals[name](parameters, cov=cov, successive=successive)


@pytest.mark.parametrize('proposal_name', [('discrete'),
                                           ('bounded_discrete')])
def test_multiple_pars(proposal_name):
    with pytest.raises(ValueError, match=r".* `successive` .*"):
        _setup_proposal(proposal_name, ['foo', 'bar'],
                        {'foo': (0, 10), 'bar': (0, 10)}, None,
                        successive={'foo': True})


@pytest.mark.parametrize('proposal_name,cov',
                         [('bounded_discrete', None),
                          ('bounded_discrete', 4),
                          ('ss_adaptive_bounded_discrete', None),
                          ('adaptive_bounded_discrete', None)])
@pytest.mark.parametrize('kmin,kmax',
                         [(0, 16), (1, 11)])
@pytest.mark.parametrize('successive',
                         [(None),
                          ({'kappa': 1}),
                          ({'kappa': True}),
                          ({'kappa': False})])
def test_jumps_in_bounds(proposal_name, cov, kmin, kmax, successive):
    """Tests that all jumps are integers and in the given bounds.

    Also tests that the proposals can handle parameter names longer than 1.
    """
    if successive is not None:
        if not isinstance(successive['kappa'], bool):
            with pytest.raises(ValueError, match=r".* must .*"):
                proposal = _setup_proposal(proposal_name, ['kappa'],
                                           {'kappa': (kmin, kmax)}, cov,
                                           successive)
            return
    if successive == {}:
        with pytest.raises(ValueError, match=r".* must .*"):
            proposal = _setup_proposal(proposal_name, ['kappa'],
                                       {'kappa': (kmin, kmax)}, cov,
                                       successive)
            return

    proposal = _setup_proposal(proposal_name, ['kappa'],
                               {'kappa': (kmin, kmax)}, cov, successive)

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
    # check successive jumps
    if successive is not None:
        test_point = {'kappa': (kmin + kmax) // 2}
        jumps = numpy.array([proposal.jump(test_point)['kappa']
                             for __ in range(njumps)])
        assert numpy.any(jumps == test_point['kappa']) == successive['kappa']
    # check jump from outside bounds
    test_point = {'kappa': kmin - kmax}
    with pytest.raises(ValueError, match=r"Given point .*"):
        proposal.jump(test_point)


@pytest.mark.parametrize('proposal_name', ['discrete',
                                           'ss_adaptive_discrete',
                                           'adaptive_discrete',
                                           'bounded_discrete',
                                           'ss_adaptive_bounded_discrete',
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


@pytest.mark.parametrize('proposal_name', ['discrete',
                                           'ss_adaptive_discrete',
                                           'adaptive_discrete',
                                           'bounded_discrete',
                                           'ss_adaptive_bounded_discrete',
                                           'adaptive_bounded_discrete'])
def test_logpdf_successive(proposal_name):
    """Tests that the logpdf function is constant for two values with the same
    integer value for successive jumps
    """
    model = PoissonModel()
    proposal = _setup_proposal(proposal_name, model.params,
                               model.prior_bounds,
                               successive={model.params[0]: True})
    givenpt = 3
    for testpt in [1.2, 4.3, 3.8, 2.7, 3.2]:
        logp0 = proposal.logpdf({'k': testpt}, {'k': givenpt})
        logp1 = proposal.logpdf({'k': testpt+0.1}, {'k': givenpt})
        assert isinstance(logp0, float)
        assert numpy.isfinite(logp0)
        assert isinstance(logp1, float)
        assert numpy.isfinite(logp1)
        assert logp0 == logp1
    # test that a jump less than 1 from the given point has non-zero prob
    assert numpy.isfinite(proposal.logpdf({'k': givenpt+0.4}, {'k': givenpt}))

@pytest.mark.parametrize('nprocs', [1, 4])
@pytest.mark.parametrize('proposal_name', ['ss_adaptive_discrete',
                                           'ss_adaptive_bounded_discrete',
                                           'adaptive_discrete',
                                           'adaptive_bounded_discrete'])
def test_std_changes(nprocs, proposal_name):
    """Tests that the standard deviation of the adaptive proposals change
    after a few jumps.
    """
    model = PoissonModel()
    proposal = _setup_proposal(proposal_name, model.params,
                               model.prior_bounds)
    _test_std_changes(nprocs, proposal, model)


@pytest.mark.parametrize('proposal_name', ['discrete',
                                           'ss_adaptive_discrete',
                                           'adaptive_discrete',
                                           'bounded_discrete',
                                           'ss_adaptive_bounded_discrete',
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
                                           'ss_adaptive_discrete',
                                           'adaptive_discrete',
                                           'bounded_discrete',
                                           'ss_adaptive_bounded_discrete',
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
                                           'ss_adaptive_discrete',
                                           'adaptive_discrete',
                                           'bounded_discrete',
                                           'ss_adaptive_bounded_discrete',
                                           'adaptive_bounded_discrete'])
@pytest.mark.parametrize('nprocs', [1, 4])
def test_seed(proposal_name, nprocs):
    """Runs the PTSampler ``test_seed`` using the adaptive normal proposal.
    """
    model = PoissonModel()
    proposal = _setup_proposal(proposal_name, model.params, model.prior_bounds)
    _test_seed(PoissonModel, nprocs, proposals=[proposal])


@pytest.mark.parametrize('proposal_name', ['discrete',
                                           'ss_adaptive_discrete',
                                           'adaptive_discrete',
                                           'bounded_discrete',
                                           'ss_adaptive_bounded_discrete',
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
