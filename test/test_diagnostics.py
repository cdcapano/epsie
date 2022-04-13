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
"""Performs unit tests on the ParallelTempered sampler."""

import pytest
import numpy
from epsie import make_betas_ladder
from epsie import diagnostic
from _utils import Model

from test_ptsampler import _create_sampler as _create_pt_sampler
from test_mhsampler import _create_sampler as _create_mh_sampler


NCHAINS = 2
NTEMPS = 3
BETAS = make_betas_ladder(NTEMPS, 1e5)
ITERINT = 64
SEED = 2020


@pytest.mark.parametrize("sampler_cls", ["mh", "pt"])
def test_thinning(sampler_cls):
    if sampler_cls == "mh":
        _test_mh_thinning()
    elif sampler_cls == "pt":
        _test_pt_thinning("coldest")
        _test_pt_thinning("max")
    else:
        raise ValueError("Unknown sampler class.")


def _test_mh_thinning():
    model = Model()
    sampler = _create_mh_sampler(model, nprocs=1, nchains=NCHAINS,
                                 seed=SEED)
    sampler.start_position = model.prior_rvs(size=NCHAINS)
    # run both for a few iterations
    sampler.run(ITERINT)

    # Given a random seed test a comparison to a known value
    acl = diagnostic.acl_chain(sampler.chains[0], full=True)
    assert numpy.all(acl == numpy.array([8, 9]))
    # Check that the thinnd arrays have the right shape
    thinned = diagnostic.thinned_samples(sampler, burnin_iter=int(ITERINT/2))
    shape = None
    for i, param in enumerate(sampler.parameters):
        px = thinned[param]
        assert px.ndim == 1
        if i == 0:
            shape = px.shape
        else:
            assert px.shape == shape

    # Test the GR calculation
    Rs = diagnostic.gelman_rubin_test(
        sampler, burnin_iter=int(ITERINT/2), full=True)
    assert isinstance(Rs, numpy.ndarray)
    assert Rs.ndim == 1
    assert ~numpy.any(Rs < 1)


def _test_pt_thinning(temp_acls_method):
    model = Model()
    sampler = _create_pt_sampler(model, 1, nchains=NCHAINS, betas=BETAS,
                                 seed=SEED, swap_interval=1, proposals=None,
                                 set_start=False)
    sampler.start_position = model.prior_rvs(size=(NTEMPS, NCHAINS))

    sampler.run(ITERINT)

    thinned = diagnostic.thinned_samples(
        sampler, burnin_iter=int(ITERINT/2), temp_acls_method=temp_acls_method)

    shape = None
    for i, param in enumerate(sampler.parameters):
        px = thinned[param]
        assert px.ndim == 2
        if i == 0:
            shape = px.shape
        else:
            assert px.shape == shape

    # Test the GR calculation
    Rs = diagnostic.gelman_rubin_test(
        sampler, burnin_iter=int(ITERINT/2), full=True)
    assert isinstance(Rs, numpy.ndarray)
    assert Rs.ndim == 2
    assert ~numpy.any(Rs < 1)
