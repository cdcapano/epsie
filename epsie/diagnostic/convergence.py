# Copyright (C) 2022 Richard Stiskalek, Collin Capano
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

"""MCMC convergence tests."""

import numpy
from epsie.samplers import MetropolisHastingsSampler, ParallelTemperedSampler


def gelman_rubin_test(sampler, burnin_iter):
    """
    Calculate the Gelman-Rubin (GR) convergence test outlined in [1] and
    summarised in [2] for parallel, independent MCMC chains.

    For a PT sampler the GR statistic is calculated for each temperature level.

    Typically values of less than 1.1 or 1.2 are recommended.

    Arguments
    ---------
    sampler : {:py:class:`epsie.sampler.MetropolisHastingsSampler`,
               :py:class:`epsie.sampler.ParallelTemperedSampler`}
        Epsie sampler whose samples to extract.
    burnin_iter : int, optional
        Number of burnin iterations to be thrown away.

    Returns
    -------
    Rs : array
        GR statistic of each parameter. In case of a MH sampler 1-dimensional
        array of length `len(sampler.paramaters)` and in case of a PT sampler
        2-dimensional array of shape `len(sampler.ntemps, len(sampler.params)`.

    References
    ----------
    .. [1] Gelman, A. and Rubin, D.B. (1992). "Inference from Iterative
        Simulation using Multiple Sequences". Statistical Science, 7,
        p. 457–511.
    .. [2] (https://mc-stan.org/docs/2_18/reference-manual/
        notation-for-samples-chains-and-draws.html)
    """
    params = sampler.parameters
    # Cut off burnin iterations and for PT take coldest chains
    samples = sampler.positions[..., burnin_iter:]
    if isinstance(sampler, MetropolisHastingsSampler):
        return _gelman_rubin_at_temp(samples, params)
    elif isinstance(sampler, ParallelTemperedSampler):
        Rs = numpy.zeros((sampler.ntemps, len(params)))
        for tk in range(sampler.ntemps):
            Rs[tk, :] = _gelman_rubin_at_temp(samples[tk, ...], params)
        return Rs
    else:
        raise ValueError("Unknown sampler type ``{}``".format(type(sampler)))


def _gelman_rubin_at_temp(samples, params):
    """Calculate the Gelman Rubin statistic at a given temperature level."""
    N = samples.shape[1]
    # Calculate the GH statistic for each parameter independently
    Rs = numpy.zeros(len(params))
    for i, param in enumerate(params):
        # Between chains variance
        B = N * numpy.var(numpy.mean(samples[param], axis=1), ddof=1)
        # Within chains variance
        W = numpy.mean(numpy.var(samples[param], axis=1, ddof=1))
        Rs[i] = ((N - 1) / N + 1 / N * B / W)**0.5

    return Rs


