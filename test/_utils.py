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
"""Utilities for carrying out tests."""


import numpy
from scipy import stats
import epsie
from epsie.proposals import Boundaries


class Model(object):
    """A simple model for testing the samplers.

    The likelihood function is a 2D gaussian with parameters "x" and "y", which
    have mean 2, 5 and standard deviation 1 and 2, respectively. The prior is
    uniform in x, between -20 and 20, and uniform in y, between -40 and 40.
    """
    blob_params = None

    def __init__(self):
        # we'll use a 2D Gaussian for the likelihood distribution
        self.params = ['x', 'y']
        self.mean = numpy.array([2., 5.])
        self.std = numpy.array([1., 2.])
        self.likelihood_dist = stats.norm(loc=self.mean, scale=self.std)
        # we'll just use a uniform prior
        xbnds = Boundaries((-20., 20.))
        ybnds = Boundaries((-40., 40.))
        self.prior_bounds = {'x': xbnds,
                             'y': ybnds}
        self.prior_dist = {'x': stats.uniform(xbnds.lower, abs(xbnds)),
                           'y': stats.uniform(ybnds.lower, abs(ybnds))}

    def prior_rvs(self, size=None, shape=None):
        return {p: self.prior_dist[p].rvs(size=size).reshape(shape)
                for p in self.params}

    def logprior(self, **kwargs):
        return sum([self.prior_dist[p].logpdf(kwargs[p]) for p in self.params])

    def loglikelihood(self, **kwargs):
        return self.likelihood_dist.logpdf([kwargs[p]
                                            for p in self.params]).sum()

    def __call__(self, **kwargs):
        logp = self.logprior(**kwargs)
        if logp == -numpy.inf:
            logl = None
        else:
            logl = self.loglikelihood(**kwargs)
        return logl, logp


class ModelWithBlobs(Model):
    """Adds blobs to ``Model``.

    The added blobs are the values of the marginal log likelihood of each
    parameter.
    """
    blob_params = ['xlogl', 'ylogl']

    def loglikelihood(self, **kwargs):
        xlogl, ylogl = self.likelihood_dist.logpdf([kwargs['x'], kwargs['y']])
        return xlogl+ylogl, {'xlogl': xlogl, 'ylogl': ylogl}

    def __call__(self, **kwargs):
        logp = self.logprior(**kwargs)
        if logp == -numpy.inf:
            logl = blob = None
        else:
            logl, blob = self.loglikelihood(**kwargs)
        return logl, logp, blob


def _check_array(array, expected_params, expected_shape):
    """Helper function to test arrays returned by the sampler."""
    # check that the fields are the same as the model's
    assert sorted(array.dtype.names) == sorted(expected_params)
    # check that the shape is what's expected
    assert array.shape == expected_shape
    # check that we can turn this into a dictionary
    adict = epsie.array2dict(array)
    assert sorted(adict.keys()) == sorted(expected_params)
    for param, val in adict.items():
        assert val.shape == expected_shape


def _compare_dict_array(a, b):
    """Helper function to test if two dictionaries of arrays are the
    same.
    """
    # first check that keys are the same
    assert list(a.keys()) == list(b.keys())
    # now check the values
    assert all([(a[p] == b[p]).all() for p in a])


def _anticompare_dict_array(a, b):
    """Helper function to test if two dictionaries of arrays are the
    not the same.
    """
    # first check that keys are the same
    assert list(a.keys()) == list(b.keys())
    # now check the values
    assert not all([(a[p] == b[p]).all() for p in a])


def _check_chains_are_different(chain, other, test_blobs,
                                test_state=True):
    """Checks that two chains' random states and current positions/stats/blobs
    are different.
    """
    if test_state:
        rstate = chain.state['proposal_dist']['random_state']
        ostate = other.state['proposal_dist']['random_state']
        assert rstate != ostate
    _anticompare_dict_array(chain.current_position,
                            other.current_position)
    _anticompare_dict_array(chain.current_stats,
                            other.current_stats)
    if test_blobs:
        # note: we're checking that the blobs aren't the same, but
        # it might happen for a model that they would be the same
        # across chains, depending on the data. The testing models
        # in utils.py return the value of the log likelihood in
        # each parameter for the blobs, so we expect them to be
        # different in this case
        _anticompare_dict_array(chain.current_blob,
                                other.current_blob)
