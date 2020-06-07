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
"""Performs unit tests on the Normal proposal."""

from __future__ import (absolute_import, division)

import itertools
import pytest
import numpy
from scipy import stats

from epsie.proposals import Normal


SEED = 8341
NUMPY_SEED = 2340

# set the scipy seed for comparison
numpy.random.seed(NUMPY_SEED)


@pytest.mark.parametrize("params,cov",
                         [(['x'], 1),
                          (['x', 'y'], 3.14),
                          (['x', 'y'], [1, 2]),
                          (['foo', 'bar'], [[2., 1.414], [1.414, 2.]])])
def test_logpdf(params, cov):
    """Tests that the logpdf function returns expected values at various test
    points.
    """
    proposal = Normal(params, cov=cov)
    assert len(proposal.parameters) == len(params)
    # test that the logpdf returns what scipy stats does
    test_points = {}
    ntestpts = 1000
    test_points = stats.multivariate_normal.rvs(cov=proposal.cov,
                                                size=ntestpts)
    givenpts = [{p: val for p in params} for val in [-1., 0., 1.]]
    for ii in range(ntestpts):
        if len(params) == 1:
            testpt = {params[0]: test_points[ii]}
        else:
            testpt = dict(zip(params, test_points[ii, :]))
        for givenpt in givenpts:
            logpdf = proposal.logpdf(testpt, givenpt)
            # check that we got a float back
            assert isinstance(logpdf, float)
            # compare to scipy's multivariate pdf
            expected = stats.multivariate_normal.pdf(
                [testpt[p] for p in params], mean=[givenpt[p] for p in params],
                cov=proposal.cov)
            assert numpy.exp(logpdf) == pytest.approx(expected)


@pytest.mark.parametrize('cov', [0.5, 1, 3])
def test_jump(cov):
    """Tests that the distribution of jumps follows a normal distribution."""
    proposal = Normal('x', cov)
    proposal.bit_generator = SEED
    # check that the standard deviation is the sqrt of the covariance
    assert proposal.std == pytest.approx(cov**0.5)
    njumps = 10000
    given_points = [-1., 0., 1.]
    for ii, given_pt in enumerate(given_points):
        jumps = numpy.array([proposal.jump({'x': given_pt})['x']
                             for _ in range(njumps)])
        # check that the mean is around the given pt; this should have
        # ~ a 1% error
        assert jumps.mean() == pytest.approx(given_pt, abs=0.03)
        assert jumps.std() == pytest.approx(proposal.std, rel=0.03)
        # check that expected % are in N sigma  (error should be ~1%)
        for (nsig, expected) in [(1, 0.6827), (2, 0.9545)]:
            pcnt = ((jumps >= given_pt - nsig*proposal.std) &
                    (jumps < given_pt + nsig*proposal.std)).sum()/njumps
            assert pcnt == pytest.approx(expected, rel=0.03)
