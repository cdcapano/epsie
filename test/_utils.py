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
        self.prior_bounds = {'x': (-20., 20.),
                             'y': (-40., 40.)}
        xmin = self.prior_bounds['x'][0]
        dx = self.prior_bounds['x'][1] - xmin
        ymin = self.prior_bounds['y'][0]
        dy = self.prior_bounds['y'][1] - ymin
        self.prior_dist = {'x': stats.uniform(xmin, dx),
                           'y': stats.uniform(ymin, dy)}

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

