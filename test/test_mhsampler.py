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
"""Performs unit tests on the MetropolisHastings sampler."""

from __future__ import (print_function, absolute_import)

import itertools
import multiprocessing
import pytest
import numpy
import epsie
from epsie.samplers import MetropolisHastingsSampler
from _utils import (Model, ModelWithBlobs, _check_array, _compare_dict_array,
                    _anticompare_dict_array, _check_chains_are_different)


NCHAINS = 8
ITERINT = 32
SEED = 11001001


def _create_sampler(model, nprocs, nchains=None, seed=None, set_start=True):
    """Creates a sampler."""
    if nchains is None:
        nchains = NCHAINS
    if nprocs == 1:
        pool = None
    else:
        pool = multiprocessing.Pool(nprocs)
    sampler = MetropolisHastingsSampler(model.params, model,
                                        nchains, pool=pool,
                                        seed=seed)
    if set_start:
        sampler.start_position = model.prior_rvs(size=nchains)
    return sampler


@pytest.mark.parametrize('model_cls', [Model, ModelWithBlobs])
@pytest.mark.parametrize('nprocs', [1, 4])
def test_chains(model_cls, nprocs):
    """Sets up and runs a sampler for a few iterations, then performs
    the following checks:

    * That the positions, stats, acceptance ratios, and (if the model
      returns blobs) blobs all have the expected parameters, shape
      nchains x niterations, and can be converted to dictionaries of
      arrays.
    * That the ``current_(positions|stats|blobs)`` (if the model returns
      blobs) are the same as the last item in the positions/stats/blobs.
    * If the model does not return blobs, that the ``blobs`` and
      ``current_blobs`` are all None.
    * That the chains all have different random states after the
      iterations, and different positions/stats/blobs.
    """
    model = model_cls()
    sampler = _create_sampler(model, nprocs, nchains=NCHAINS, seed=SEED)
    # check that the number of parameters that we have proposals for
    # matches the number of model parameters
    joint_dist = sampler.chains[0].proposal_dist
    prop_params = set.union(*[set(p.parameters) for p in joint_dist.proposals])
    assert set(joint_dist.parameters) == prop_params
    assert prop_params == set(model.params)
    # run for some iterations
    sampler.run(ITERINT)
    # check that the number of recorded iterations matches how long we
    # actually ran for
    assert sampler.niterations == ITERINT
    # check that we get the positions back in the expected format
    positions = sampler.positions
    expected_shape = (NCHAINS, ITERINT)
    _check_array(positions, model.params, expected_shape)
    # check that the current position is the same as the last in the array
    _compare_dict_array(epsie.array2dict(positions[..., -1]),
                        sampler.current_positions)
    # check that the stats have the expected fields and shape
    stats = sampler.stats
    _check_array(stats, ['logl', 'logp'], expected_shape)
    # check that the current position is the same as the last in the array
    _compare_dict_array(epsie.array2dict(stats[..., -1]),
                        sampler.current_stats)
    # check that the acceptance ratios have the expected fields and shape
    acceptance = sampler.acceptance
    _check_array(acceptance, ['acceptance_ratio', 'accepted'], expected_shape)
    # check the individual chains
    for ii, chain in enumerate(sampler.chains):
        # check that the length matches the number of iterations
        assert len(chain) == ITERINT
        # check that hasblobs is None if the model doesn't return any
        assert chain.hasblobs == bool(model.blob_params)
    # check the blobs
    blobs = sampler.blobs
    current_blobs = sampler.current_blobs
    if model.blob_params:
        _check_array(blobs, model.blob_params, expected_shape)
        _compare_dict_array(epsie.array2dict(blobs[..., -1]),
                            current_blobs)
    else:
        # check that blobs are None since this model doesn't have blobs
        assert blobs is None
        assert current_blobs is None
    # check that each chain's random state and current values are different
    combos = itertools.combinations(range(len(sampler.chains)), 2)
    for ii, jj in combos:
        _check_chains_are_different(sampler.chains[ii], sampler.chains[jj],
                                    test_blobs=bool(model.blob_params))
    # terminate the multiprocessing pool so we don't end up with too many
    # open processes
    if sampler.pool is not None:
        sampler.pool.close()


@pytest.mark.parametrize('model_cls', [Model, ModelWithBlobs])
@pytest.mark.parametrize('nprocs', [1, 4])
def test_checkpointing(model_cls, nprocs):
    """Tests that resuming from checkpoint yields the same result as if
    no checkpoint happened.

    This test requires h5py to be installed.
    """
    try:
        import h5py
    except ImportError:
        raise ImportError("h5py must be installed to run this test")
    model = model_cls()
    sampler = _create_sampler(model, nprocs, nchains=NCHAINS, seed=SEED)
    # create a second sampler for comparison; we won't bother setting
    # a seed or start position, since that shouldn't matter when loading
    # from a checkpoint
    sampler2 = _create_sampler(model, nprocs, nchains=NCHAINS, seed=None,
                               set_start=False)
    sampler.run(ITERINT)
    # checkpoint to an h5py file in memory
    fp = h5py.File('test.hdf', 'w', driver='core', backing_store=False)
    sampler.checkpoint(fp)
    # run for another set of iterations
    sampler.run(ITERINT)
    # set the other sampler's state using the checkpoint
    sampler2.set_state_from_checkpoint(fp)
    fp.close()
    # run again
    sampler2.run(ITERINT)
    # compare the two
    _compare_dict_array(sampler.current_positions, sampler2.current_positions)
    _compare_dict_array(sampler.current_stats, sampler2.current_stats)
    if model.blob_params:
        _compare_dict_array(sampler.current_blobs, sampler2.current_blobs)
    # terminate the multiprocessing pools so we don't end up with too many
    # open processes
    if sampler.pool is not None:
        sampler.pool.close()
        sampler2.pool.close()


@pytest.mark.parametrize('model_cls', [Model, ModelWithBlobs])
@pytest.mark.parametrize('nprocs', [1, 4])
def test_seed(model_cls, nprocs):
    """Tests that running with the same seed yields the same results,
    while running with a different seed yields different results.
    """
    model = model_cls()
    sampler = _create_sampler(model, nprocs, nchains=NCHAINS, seed=SEED)
    # now create another sampler with the same seed and starting position
    same_seed = _create_sampler(model, nprocs, nchains=NCHAINS, seed=SEED,
                                set_start=False)
    same_seed.start_position = sampler.start_position
    # we'll start the different seed from the same start position; this
    # should still yield different positions after several iterations
    diff_seed = _create_sampler(model, nprocs, nchains=NCHAINS, seed=None,
                                set_start=False)
    diff_seed.start_position = sampler.start_position
    # not passing a seed should result in a different seed; check that
    assert sampler.seed != diff_seed.seed
    sampler.run(ITERINT)
    same_seed.run(ITERINT)
    diff_seed.run(ITERINT)
    # check that the same seed gives the same result
    _compare_dict_array(sampler.current_positions, same_seed.current_positions)
    _compare_dict_array(sampler.current_stats, same_seed.current_stats)
    if model.blob_params:
        _compare_dict_array(sampler.current_blobs, same_seed.current_blobs)
    # check that different seeds give different results
    _anticompare_dict_array(sampler.current_positions,
                            diff_seed.current_positions)
    _anticompare_dict_array(sampler.current_stats, diff_seed.current_stats)
    if model.blob_params:
        _anticompare_dict_array(sampler.current_blobs,
                                diff_seed.current_blobs)
    if sampler.pool is not None:
        sampler.pool.close()
        same_seed.pool.close()
        diff_seed.pool.close()

@pytest.mark.parametrize('model_cls', [Model, ModelWithBlobs])
@pytest.mark.parametrize('nprocs', [1, 4])
def test_clear_memory(model_cls, nprocs):
    """Tests that clearing memory and running yields the same result as if
    the memory had not been cleared.
    """
    model = model_cls()
    sampler = _create_sampler(model, nprocs, nchains=NCHAINS, seed=SEED)
    sampler2 = _create_sampler(model, nprocs, nchains=NCHAINS, seed=SEED,
                               set_start=False)
    sampler2.start_position = sampler.start_position
    # run both for a few iterations
    sampler.run(ITERINT)
    sampler2.run(ITERINT)
    # clear one, but don't clear the other
    sampler.clear()
    # now run both for a few more iterations
    sampler.run(ITERINT)
    sampler2.run(ITERINT)
    # check that the number of recorded iterations matches how long we
    # actually ran for
    assert sampler.niterations == 2*ITERINT
    assert sampler2.niterations == 2*ITERINT
    # but that the lengths of the stored arrays differ
    expected_shape = (NCHAINS, ITERINT)
    expected_shape2 = (NCHAINS, 2*ITERINT)
    _check_array(sampler.positions, model.params, expected_shape)
    _check_array(sampler2.positions, model.params, expected_shape2)
    _check_array(sampler.stats, ['logl', 'logp'], expected_shape)
    _check_array(sampler2.stats, ['logl', 'logp'], expected_shape2)
    _check_array(sampler.acceptance, ['acceptance_ratio', 'accepted'],
                 expected_shape)
    _check_array(sampler2.acceptance, ['acceptance_ratio', 'accepted'],
                 expected_shape2)
    if model.blob_params:
        _check_array(sampler.blobs, model.blob_params, expected_shape)
        _check_array(sampler2.blobs, model.blob_params, expected_shape2)
    # they should be in the same place
    _compare_dict_array(sampler.current_positions, sampler2.current_positions)
    _compare_dict_array(sampler.current_stats, sampler2.current_stats)
    if model.blob_params:
        _compare_dict_array(sampler.current_blobs, sampler2.current_blobs)
    if sampler.pool is not None:
        sampler.pool.close()
        sampler2.pool.close()
