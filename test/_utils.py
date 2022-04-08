# Copyright (C) 2020  Collin Capano
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.  #
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
from epsie.proposals import (Boundaries, IsotropicSolidAngle)

MODEL_SEED = 1983

#
# =============================================================================
#
#                              Test Models
#
# =============================================================================
#


class Model:
    """A simple model for testing the samplers.

    The likelihood function is a 2D gaussian with parameters "x0" and "x1",
    which have mean 2, 5 and standard deviation 1 and 2, respectively. The
    prior is uniform in x0, between -20 and 20, and uniform in x1, between -40
    and 40.
    """
    blob_params = None

    def __init__(self, seed=None):
        # we'll use a 2D Gaussian for the likelihood distribution
        self.params = ['x0', 'x1']
        self.mean = numpy.array([2., 5.])
        self.std = numpy.array([1., 2.])
        self.likelihood_dist = stats.norm(loc=self.mean, scale=self.std)
        # we'll just use a uniform prior
        xbnds = Boundaries((-20., 20.))
        ybnds = Boundaries((-40., 40.))
        self.prior_bounds = {'x0': xbnds,
                             'x1': ybnds}
        self.prior_dist = {'x0': stats.uniform(xbnds.lower, abs(xbnds)),
                           'x1': stats.uniform(ybnds.lower, abs(ybnds))}
        # create an rng for drawing prior samples
        if seed is None:
            seed = MODEL_SEED
        self.rng = numpy.random.default_rng(seed)

    def prior_rvs(self, size=None, shape=None):
        return {p: self.prior_dist[p].rvs(size=size,
                                          random_state=self.rng).reshape(shape)
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
        xlogl, ylogl = self.likelihood_dist.logpdf([kwargs['x0'],
                                                    kwargs['x1']])
        return xlogl+ylogl, {'xlogl': xlogl, 'ylogl': ylogl}

    def __call__(self, **kwargs):
        logp = self.logprior(**kwargs)
        if logp == -numpy.inf:
            logl = blob = None
        else:
            logl, blob = self.loglikelihood(**kwargs)
        return logl, logp, blob


class AngularModel:
    r"""A simple angular model.

    The likelihood is a truncated normal centered on :math:`\phi_0`, with
    cyclic bounds at :math:`0` and :math:`2\pi`. The prior is uniform.
    """
    blob_params = None

    def __init__(self, phi0=0., std=1., seed=None):
        self.phi0 = phi0
        self.params = ['phi']
        self.std = numpy.array([std])
        # we'll just use a uniform prior
        self.prior_bounds = {'phi': Boundaries((0., 2*numpy.pi))}
        pmin = self.prior_bounds['phi'].lower
        dp = abs(self.prior_bounds['phi'])
        self.prior_dist = {'phi': stats.uniform(pmin, dp)}
        # create an rng for drawing prior samples
        if seed is None:
            seed = MODEL_SEED
        self.rng = numpy.random.default_rng(seed)

    def prior_rvs(self, size=None, shape=None):
        return {p: self.prior_dist[p].rvs(size=size,
                                          random_state=self.rng).reshape(shape)
                for p in self.params}

    def logprior(self, **kwargs):
        return sum([self.prior_dist[p].logpdf(kwargs[p]) for p in self.params])

    def loglikelihood(self, **kwargs):
        # apply cyclic bounds to xi to put in [0, 2\pi]
        xi = numpy.array([kwargs[p] for p in self.params]) % (2*numpy.pi)
        # shift xi by the amounted needed to move phi0 to the cetner of
        # [0, 2\pi), and apply bounds again
        dphi = numpy.pi - self.phi0
        xi = (xi + dphi) % (2*numpy.pi)
        # now use a truncated normal centered on pi
        b = numpy.pi/self.std
        a = -b
        return stats.truncnorm.logpdf(xi, a, b, loc=numpy.pi,
                                      scale=self.std).sum()

    def __call__(self, **kwargs):
        logp = self.logprior(**kwargs)
        if logp == -numpy.inf:
            logl = None
        else:
            logl = self.loglikelihood(**kwargs)
        return logl, logp


class PoissonModel:
    r"""A poisson model.

    The free parameter is the number of foreground counts, which is
    an integer. The intrinsic rate can be set using the ``lmbda`` argument
    on initialization.
    """
    blob_params = None

    def __init__(self, lmbda=3, seed=None):
        # we'll use a Poission distribution for the likelihood
        self.params = ['k']
        self.likelihood_dist = stats.poisson(lmbda)
        # we'll just use a uniform prior
        kmin = 0
        kmax = 10
        self.prior_bounds = {'k': Boundaries((kmin, kmax))}
        self.prior_dist = {'k': stats.randint(kmin, kmax)}
        # create an rng for drawing prior samples
        if seed is None:
            seed = MODEL_SEED
        self.rng = numpy.random.default_rng(seed)

    def prior_rvs(self, size=None, shape=None):
        return {p: self.prior_dist[p].rvs(size=size,
                                          random_state=self.rng).reshape(shape)
                for p in self.params}

    def logprior(self, **kwargs):
        return sum([self.prior_dist[p].logpmf(kwargs[p]) for p in self.params])

    def loglikelihood(self, **kwargs):
        return self.likelihood_dist.logpmf([kwargs[p]
                                            for p in self.params]).sum()

    def __call__(self, **kwargs):
        logp = self.logprior(**kwargs)
        if logp == -numpy.inf:
            logl = None
        else:
            logl = self.loglikelihood(**kwargs)
        return logl, logp


class PolynomialRegressionModel:
    r""" A polynomial regression model.

    The free parameters are the polynomial coefficients:

        .. math::
            y\left( x \right) = \sum_{n=0}^{5} a_n * x^{n}.
    The coefficients :math `a_n` can be either turned on or off, thus
    increasing or decreasing the dimensionality of the model.

    This simple model is used to test the nested transdimensional proposal.

    The initial inactive parameters must be set to `numpy.nan`.
    """
    blob_params = None

    def __init__(self, seed=None):
        self.inmodel_pars = ['a']
        self.params = ['a0', 'a1', 'a2', 'a3', 'a4', 'a5', 'k']
        self.k = 'k'
        self.kmax = 5
        self.prior_dist = {'a': stats.uniform(0., 4.),
                           'k': stats.randint(0, self.kmax + 1)}
        # Define the injected signal
        self.true_signal = {'a0': 0.0,
                            'a1': 2.5,
                            'a2': 0.5,
                            'a3': numpy.nan,
                            'a4': numpy.nan,
                            'a5': numpy.nan,
                            'k': 2}

        numpy.random.seed(seed)
        self.npoints = 51
        self.t = numpy.linspace(0.0, 5, self.npoints)
        self.ysignal = self.reconstruct(**self.true_signal)
        # create an rng for drawing prior samples
        if seed is None:
            seed = MODEL_SEED
        self.rng = numpy.random.default_rng(seed)

    def prior_rvs(self, size=None, shape=None):
        ntemps, nchains = shape
        out = {}
        if ntemps is None:
            coeffs = self.prior_dist['a'].rvs(
                size=nchains * (self.kmax + 1),
                random_state=self.rng).reshape(nchains, self.kmax + 1)
            ks = numpy.full(nchains, numpy.nan, dtype=int)
            for i in range(nchains):
                rands = self.rng.choice(range(self.kmax),
                                        self.rng.integers(self.kmax),
                                        replace=False)
                coeffs[i, 1:][rands] = numpy.nan
                ks[i] = int(self.kmax - len(rands))

            for i in range(self.kmax + 1):
                out.update({'a{}'.format(i): coeffs[:, i].reshape(nchains, )})
        else:
            coeffs = self.prior_dist['a'].rvs(
                size=nchains * ntemps * (self.kmax + 1),
                random_state=self.rng).reshape(ntemps, nchains, self.kmax + 1)
            ks = numpy.full((ntemps, nchains), numpy.nan, dtype=int)
            for i in range(nchains):
                for j in range(ntemps):
                    rands = self.rng.choice(range(self.kmax),
                                            self.rng.integers(self.kmax),
                                            replace=False)
                    coeffs[j, i, 1:][rands] = numpy.nan
                    ks[j, i] = int(self.kmax - len(rands))

            for i in range(self.kmax + 1):
                out.update({'a{}'.format(i): coeffs[:, :, i].reshape(
                    ntemps, nchains)})

        out.update({'k': ks})
        return out

    def logprior(self, **kwargs):
        lp = 0.0
        # Prior on the model index
        lp += sum(self.prior_dist[self.k].logpmf([kwargs[self.k]]))
        coeffs = numpy.array([kwargs['a{}'.format(i)]
                             for i in range(self.kmax + 1)])
        # take only ones that are active
        coeffs = coeffs[numpy.isfinite(coeffs)]
        lp += self.prior_dist['a'].logpdf(coeffs).sum()
        return lp

    def reconstruct(self, **kwargs):
        coeffs = numpy.array([kwargs['a{}'.format(i)]
                             for i in range(self.kmax, -1, -1)])
        coeffs[numpy.isnan(coeffs)] = 0.0
        return numpy.polyval(coeffs, self.t)

    def loglikelihood(self, **kwargs):
        df = self.ysignal - self.reconstruct(**kwargs)
        return - numpy.dot(df.T, df)

    def __call__(self, **kwargs):
        logp = self.logprior(**kwargs)
        if logp == -numpy.inf:
            logl = None
        else:
            logl = self.loglikelihood(**kwargs)
        return logl, logp


class SolidAngleModel:
    r"""A solid angle isotropic model centered at some point on a 2-sphere.
    """
    blob_params = None

    def __init__(self, radec=False, degs=False, seed=None):
        self.params = ['phi', 'theta']
        x = 1. / numpy.sqrt(3)
        self.mu = numpy.array([x, x, x])
        self.kappa = 23.
        self.solid_angle_prop = IsotropicSolidAngle(
            self.params[0], self.params[1], self.kappa, radec, degs)
        phi, theta = self.solid_angle_prop._cartesian2spherical(
            *self.mu, convert=True)
        self.mu_spherical = {'phi': phi, 'theta': theta}
        # create an rng for drawing prior samples
        if seed is None:
            seed = MODEL_SEED
        self.rng = numpy.random.default_rng(seed)

    def prior_rvs(self, size=None, shape=None):
        phi = self.rng.uniform(0, 2*numpy.pi, size).reshape(shape)
        theta = numpy.arccos(1 - 2 * self.rng.uniform(0, 1, size))
        theta = theta.reshape(shape)
        return {'phi': phi, 'theta': theta}

    def loglikelihood(self, **kwargs):
        return self.solid_angle_prop.logpdf(kwargs, self.mu_spherical)

    def __call__(self, **kwargs):
        return self.loglikelihood(**kwargs), -numpy.log(4 * numpy.pi)

#
# =============================================================================
#
#                        Utility functions for tests
#
# =============================================================================
#


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
    # now check the values. Here need to check value by value because
    # comparisons of ``numpy.nan == numpy.nan`` returns False
    for p in a:
        numpy.testing.assert_equal(a[p], b[p])


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
    """Checks that two chains' random states and positions/stats/blobs are
    different.
    """
    if test_state:
        rstate = chain.state['proposal_dist']['random_state']
        ostate = other.state['proposal_dist']['random_state']
        assert rstate != ostate
    _anticompare_dict_array(epsie.array2dict(chain.positions),
                            epsie.array2dict(other.positions))
    _anticompare_dict_array(epsie.array2dict(chain.stats),
                            epsie.array2dict(other.stats))
    if test_blobs:
        # note: we're checking that the blobs aren't the same, but
        # it might happen for a model that they would be the same
        # across chains, depending on the data. The testing models
        # in utils.py return the value of the log likelihood in
        # each parameter for the blobs, so we expect them to be
        # different in this case
        _anticompare_dict_array(epsie.array2dict(chain.blobs),
                                epsie.array2dict(other.blobs))

def _closepool(sampler):
    """Terminates a sampler's pool, if the sampler has one."""
    if sampler.pool is not None:
        # the pool processes aren't always being released in
        # python >= 3.8 on github, causing test functions to
        # hang at this point. I found that making a command-line
        # call via subprocess unsticks them... I have nod idea why
        import subprocess
        subprocess.run("echo")
        sampler.pool.terminate()
        sampler.pool.join()
        subprocess.run("echo")
    return
