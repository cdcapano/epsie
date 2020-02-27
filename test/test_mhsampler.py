#!/usr/bin/env python

from __future__ import (print_function, absolute_import)

import unittest
import itertools
import multiprocessing
import numpy
import randomgen
import epsie
from epsie.samplers import MetropolisHastingsSampler
from _utils import (Model, ModelWithBlobs)


def _compare_dict_array(a, b):
    """Helper function to test if two dictionaries of arrays are the
    same.
    """
    # first check that keys are the same
    if a.keys() != b.keys():
        return False
    return all([(a[p] == b[p]).all() for p in a])

class TestMHSampler(unittest.TestCase):
    """Runs the MHSampler and runs various tests on it."""
    nchains = 8
    nprocs = 1
    iterint = 32  # standard number of iterations we'll run for
    seed = 11001001
    model_cls = Model

    def setUp(self):
        self.model = self.model_cls()
        if self.nprocs == 1:
            self.pool = None
        else:
            self.pool = multiprocessing.Pool(self.nprocs)
        self.start_position = self.model.prior_rvs(size=self.nchains)

    def _create_sampler(self, seed=None, set_start=True):
        """Creates a sampler."""
        sampler = MetropolisHastingsSampler(self.model.params, self.model,
                                            self.nchains, pool=self.pool,
                                            seed=seed)
        if set_start:
            sampler.start_position = self.start_position
        return sampler

    def _check_array(self, array, expected_params, expected_shape):
        """Helper function to test arrays returned by the sampler."""
        # check that the fields are the same as the model's
        self.assertEqual(sorted(array.dtype.names),
                         sorted(expected_params),
                         "Sampler's returned parameters differ from the "
                         "expected")
        # check that the shape is what's expected
        self.assertEqual(array.shape, expected_shape,
                         "Array does not have the expected shape "
                         "(expected {},  got {})".format(expected_shape,
                                                         array.shape))
        # check that we can turn this into a dictionary
        adict = epsie.array2dict(array)
        self.assertEqual(sorted(adict.keys()), sorted(expected_params),
                         "Not all parameters converted to dictionary")
        for param, val in adict.items():
            self.assertEqual(val.shape, expected_shape,
                             "Dictionary values do not have expected shape "
                             "(expected {}; got {})".format(expected_shape,
                                                            val.shape))

    def test_chains(self):
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
        sampler = self._create_sampler(self.seed)
        sampler.run(self.iterint)
        # check that we get the positions back in the expected format
        positions = sampler.positions
        expected_shape = (self.nchains, self.iterint)
        self._check_array(positions, self.model.params,
                          expected_shape)
        # check that the current position is the same as the last in the array
        comp = _compare_dict_array(epsie.array2dict(positions[..., -1]),
                                   sampler.current_positions)
        self.assertTrue(comp,
                        "Last values in position array are not the same "
                        "as the current_positions attribute")
        # check that the stats have the expected fields and shape
        stats = sampler.stats
        self._check_array(stats, ['logl', 'logp'],
                          expected_shape)
        # check that the current position is the same as the last in the array
        comp = _compare_dict_array(epsie.array2dict(stats[..., -1]),
                                   sampler.current_stats)
        self.assertTrue(comp,
                        "Last values in stats array are not the same "
                        "as the current_stats attribute")
        # check that the acceptance ratios have the expected fields and shape
        acceptance = sampler.acceptance
        self._check_array(acceptance, ['acceptance_ratio', 'accepted'],
                          expected_shape)
        # check the individual chains
        for ii, chain in enumerate(sampler.chains):
            # check that the length matches the number of iterations
            self.assertEqual(len(chain), self.iterint,
                             "Length of chain {} is different than the number "
                             "of iterations".format(ii))
            # check that hasblobs is None if the model doesn't return any
            if not self.model.blob_params:
                self.assertFalse(chain.hasblobs,
                                 "chain.hasblobs should be false if the model "
                                 "does not return blobs")
            else:
                self.assertTrue(chain.hasblobs,
                                "chain.hasblobs should be true if the model "
                                "returns blobs")
        # check the blobs
        blobs = sampler.blobs
        current_blobs = sampler.current_blobs
        if self.model.blob_params:
            self._check_array(blobs, self.model.blob_params,
                              expected_shape)
            comp = _compare_dict_array(epsie.array2dict(blobs[..., -1]),
                                       current_blobs)
            self.assertTrue(comp,
                            "Last values in the blobs array are not the "
                            "same as the current_blobs attribute")
        else:
            # check that blobs are None since this model doesn't have blobs
            self.assertTrue(blobs is None,
                            "Sampler should return None as the blob attribute "
                            "when the model does not return blobs")
            self.assertTrue(current_blobs is None,
                            "current_blobs should be None since the model "
                            "does not return blobs")
        # check that each chain's random state and current values are different
        combos = itertools.combinations(range(len(sampler.chains)), 2)
        for ii, jj in combos:
            chain = sampler.chains[ii]
            other = sampler.chains[jj]
            rstate = chain.state['proposal_dist']['random_state']
            ostate = other.state['proposal_dist']['random_state']
            self.assertNotEqual(rstate, ostate,
                                "Chains should all have different "
                                "random states")
            comp = _compare_dict_array(chain.current_position,
                                       other.current_position)
            self.assertFalse(comp,
                             "Chains should all have different current "
                             "positions")
            comp = _compare_dict_array(chain.current_stats,
                                       other.current_stats)
            self.assertFalse(comp,
                             "Chains should all have different current "
                             "positions")
            if self.model.blob_params:
                # note: we're checking that the blobs aren't the same, but
                # it might happen for a model that they would be the same
                # across chains, depending on the data. The testing models
                # in utils.py return the value of the log likelihood in
                # each parameter for the blobs, so we expect them to be
                # different in this case
                comp = _compare_dict_array(chain.current_blob,
                                           other.current_blob)
                self.assertFalse(comp,
                                 "Chains should all have different blobs "
                                 "since the tested model returns blob "
                                 "values that depend on the position")


    def test_checkpointing(self):
        """Tests that resuming from checkpoint yields the same result as if
        no checkpoint happened.
        """
        import h5py
        sampler = self._create_sampler(self.seed)
        # create a second sampler for comparison; we won't bother setting
        # a seed or start position, since that shouldn't matter when loading
        # from a checkpoint
        sampler2 = self._create_sampler(set_start=False)
        sampler.run(self.iterint)
        # checkpoint to an h5py file in memory
        fp = h5py.File('test.hdf', 'w', driver='core', backing_store=False)
        sampler.checkpoint(fp)
        # run for another set of iterations
        sampler.run(self.iterint)
        # set the other sampler's state using the checkpiont
        sampler2.set_state_from_checkpoint(fp)
        fp.close()
        # run again
        sampler2.run(self.iterint)
        # compare the two
        comp = _compare_dict_array(sampler.current_positions,
                                   sampler2.current_positions)
        self.assertTrue(comp,
                        "Resuming from checkpoint did not yield the same "
                        "positions as running continously")
        comp = _compare_dict_array(sampler.current_stats,
                                   sampler2.current_stats)
        self.assertTrue(comp,
                        "Resuming from checkpoint did not yield the same "
                        "stats as running continously")
        if self.model.blob_params:
            comp = _compare_dict_array(sampler.current_blobs,
                                       sampler2.current_blobs)
            self.assertTrue(comp,
                            "Resuming from checkpoint did not yield the same "
                            "blobs as running continously")

    def test_seed(self):
        """Tests that running with the same seed yields the same results,
        while running with a different seed yields different results.
        """
        sampler = self._create_sampler(self.seed)
        same_seed = self._create_sampler(self.seed)
        # we'll start the different seed from the same start position; this
        # should still yield different positions after several iterations
        diff_seed = self._create_sampler()
        # not passing a seed should result in a different seed; check that
        self.assertNotEqual(sampler.seed, diff_seed.seed,
                            "Creating a random seed gave the same result as "
                            "a {} (the odds of this happening should be very "
                            "small)".format(self.seed))
        sampler.run(self.iterint)
        same_seed.run(self.iterint)
        diff_seed.run(self.iterint)
        # check that the same seed gives the same result
        comp = _compare_dict_array(sampler.current_positions,
                                   same_seed.current_positions)
        self.assertTrue(comp,
                        "Using the same seed and start did not give the "
                        "same positions")
        comp = _compare_dict_array(sampler.current_stats,
                                   same_seed.current_stats)
        self.assertTrue(comp,
                        "Using the same seed and start did not give the "
                        "same stats")
        if self.model.blob_params:
            comp = _compare_dict_array(sampler.current_blobs,
                                       same_seed.current_blobs)
            self.assertTrue(comp,
                            "Using the same seed and start did not give the "
                            "same blobs")
        # check that different seeds give different results
        comp = _compare_dict_array(sampler.current_positions,
                                   diff_seed.current_positions)
        self.assertFalse(comp,
                         "Using the same seed and start did not give the "
                         "same positions")
        comp = _compare_dict_array(sampler.current_stats,
                                   diff_seed.current_stats)
        self.assertFalse(comp,
                         "Using the same seed and start did not give the "
                         "same stats")
        if self.model.blob_params:
            comp = _compare_dict_array(sampler.current_blobs,
                                       diff_seed.current_blobs)
            self.assertFalse(comp,
                             "Using the same seed and start did not give "
                             "the same blobs")

    def test_clear_memory(self):
        """Tests that clearing memory and running yields the same result as if
        the memory had not been cleared.
        """
        sampler = self._create_sampler(self.seed)
        sampler2 = self._create_sampler(self.seed)
        # run both for a few iterations
        sampler.run(self.iterint)
        sampler2.run(self.iterint)
        # clear one, but don't clear the other
        sampler.clear()
        # now run both for a few more iterations
        sampler.run(self.iterint)
        sampler2.run(self.iterint)
        # they should be in the same place
        comp = _compare_dict_array(sampler.current_positions,
                                   sampler2.current_positions)
        self.assertTrue(comp,
                        "Clearing memory did not yield the same positions "
                        "as if the memory had not been cleared")
        comp = _compare_dict_array(sampler.current_stats,
                                   sampler2.current_stats)
        self.assertTrue(comp,
                        "Clearing memory did not yield the same stats "
                        "as if the memory had not been cleared")
        if self.model.blob_params:
            comp = _compare_dict_array(sampler.current_blobs,
                                       sampler2.current_blobs)
            self.assertTrue(comp,
                            "Clearing memory did not yield the same blobs "
                            "as if the memory had not been cleared")


class TestMHSamplerMultiProc(TestMHSampler):
    """Repeats the MH sampler tests in a multiprocessing environment."""
    nprocs = 4


class TestMHSamplerWithBlobs(TestMHSampler):
    """Repeats the MH sampler tests with a model that returns blobs."""
    model_cls = ModelWithBlobs


class TestMHSamplerWithBlobsMultiProc(TestMHSampler):
    """Repeats the MH sampler tests with a model that returns blobs
    in a multiprocessing environment."""
    model_cls = ModelWithBlobs
    nprocs = 4


if __name__ == '__main__':
    unittest.main()
