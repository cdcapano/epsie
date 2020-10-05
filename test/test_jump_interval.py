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
"""Performs unit tests on the fast parameters support."""

from __future__ import (absolute_import, division)

import pytest
import numpy

from epsie.proposals import (Normal, Eigenvector, BoundedNormal, Angular)

from test_ptsampler import _create_sampler

from _utils import Model

STABILITY_DURATION = 48
DURATION = 16


def _setup_proposal(proposal_name, jump_interval, params=None):
    duration = STABILITY_DURATION + DURATION
    if params is None:
        params = model.params
    if proposal_name == 'normal':
        return Normal(params, jump_interval=jump_interval,
                      jump_interval_duration=duration)
    elif proposal_name == 'eigenvector':
        return Eigenvector(params, stability_duration=STABILITY_DURATION,
                           jump_interval=jump_interval,
                           jump_interval_duration=duration)
    elif proposal_name == 'bounded_normal':
        bounds = {'x0': (-20, 20), 'x1': (-40, 40)}
        return BoundedNormal(params, bounds, jump_interval=jump_interval,
                             jump_interval_duration=duration)
    elif proposal_name == 'angular':
        return Angular(params, jump_interval=jump_interval,
                       jump_interval_duration=duration)
    else:
        return -1


def _extract_positions(chains, kind='current'):
    out = numpy.zeros((len(chains), len(chains[0].chains), 2))
    for i, chain in enumerate(chains):
        for j, subchain in enumerate(chain.chains):
            if kind == 'proposed':
                out[i, j, :] = list(subchain.proposed_position.values())
            else:
                out[i, j, :] = list(subchain.current_position.values())
    return out


@pytest.mark.parametrize('nprocs', [1, 4])
@pytest.mark.parametrize('proposal_name', ['normal', 'eigenvector',
                                           'bounded_normal', 'angular'])
@pytest.mark.parametrize('jump_interval', [1, 2, 5])
def test_jump_proposal_interval(nprocs, proposal_name, jump_interval):
    model = Model()
    # let x0 be the slow parameter and x1 the fast one
    proposal = _setup_proposal(proposal_name, jump_interval, params=['x0'])
    sampler = _create_sampler(model, nprocs, proposals=[proposal])
    # Run the sampler for some number of initial iterations
    sampler.run((STABILITY_DURATION + 1) * jump_interval)

    for _ in range((DURATION - 1) * jump_interval):
        current_pos = _extract_positions(sampler.chains, 'current')
        sampler.run(1)
        proposed_pos = _extract_positions(sampler.chains, 'proposed')

        # check that x0 are different if proposing a move, else the same
        if (sampler.niterations - 1) % jump_interval != 0:
            numpy.testing.assert_equal(current_pos[:, :, 0],
                                       proposed_pos[:, :, 0])
        else:
            assert numpy.all(current_pos[:, :, 0] != proposed_pos[:, :, 0])
        # check that x1 proposed position is always different
        assert numpy.all(current_pos[:, :, 1] != proposed_pos[:, :, 1])

    # Now that the burnin phase is over check both x0 and x1 are proposed at
    # each turn
    for i in range(DURATION):
        current_pos = _extract_positions(sampler.chains, 'current')
        sampler.run(1)
        proposed_pos = _extract_positions(sampler.chains, 'proposed')

        assert numpy.all(current_pos != proposed_pos)
