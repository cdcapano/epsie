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
"""Performs unit tests on the solid angle proposals."""

from __future__ import (print_function, absolute_import)

import pytest

from epsie.proposals import (IsotropicSolidAngle, AdaptiveIsotropicSolidAngle)
from _utils import SolidAngleModel

from test_ptsampler import test_checkpointing as _test_checkpointing
from test_ptsampler import test_seed as _test_seed
from test_ptsampler import test_clear_memory as _test_clear_memory


ADAPTATION_DURATION = 32
SWAP_INTERVAL = 1


def _setup_proposal(model, proposal_name, params=None):
    if params is None:
        params = model.params
    if proposal_name == 'isotropic_solid_angle':
        return IsotropicSolidAngle(params[0], params[1])
    elif proposal_name == 'adaptive_isotropic_solid_angle':
        return AdaptiveIsotropicSolidAngle(params[0], params[1],
                                           ADAPTATION_DURATION)
    else:
        return -1


@pytest.mark.parametrize('nprocs', [1, 4])
@pytest.mark.parametrize('proposal_name', ['isotropic_solid_angle',
                                           'adaptive_isotropic_solid_angle'])
def test_checkpointing(nprocs, proposal_name):
    """Performs the same checkpointing test as for the PTSampler.
    """
    model = SolidAngleModel()
    proposal = _setup_proposal(model, proposal_name)
    _test_checkpointing(SolidAngleModel, nprocs, proposals=[proposal])


@pytest.mark.parametrize('nprocs', [1, 4])
@pytest.mark.parametrize('proposal_name', ['isotropic_solid_angle',
                                           'adaptive_isotropic_solid_angle'])
def test_seed(nprocs, proposal_name):
    """Runs the PTSampler ``test_seed`` using the adaptive normal proposal.
    """
    model = SolidAngleModel()
    proposal = _setup_proposal(model, proposal_name)
    _test_seed(SolidAngleModel, nprocs, proposals=[proposal])


@pytest.mark.parametrize('nprocs', [1, 4])
@pytest.mark.parametrize('proposal_name', ['isotropic_solid_angle',
                                           'adaptive_isotropic_solid_angle'])
def test_clear_memory(nprocs, proposal_name):
    """Runs the PTSampler ``test_clear_memoory`` using the adaptive normal
    proposal.
    """
    model = SolidAngleModel()
    proposal = _setup_proposal(model, proposal_name)
    _test_clear_memory(SolidAngleModel, nprocs, SWAP_INTERVAL,
                       proposals=[proposal])
