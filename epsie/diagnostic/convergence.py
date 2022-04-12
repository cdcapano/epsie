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


def gelman_rubin_test(sampler, burnin_iter, full=False):
    """
    Calculate the Gelman-Rubin (GR) convergence test outlined in [1] for
    parallel, independent MCMC chains. The final statistic is averaged over all
    parameters unless `full=True`, in which case the GR statistic is returned
    for each parameter.

    For a PT sampler the GR statistic is calculated using the coldest chains.

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
    R : float (or array)
        GR statistics.

    References
    ----------
    .. [1] Gelman, A. and Rubin, D.B. (1992). "Inference from Iterative
        Simulation using Multiple Sequences". Statistical Science, 7,
        p. 457–511.
    """
    params = sampler.parameters
    # Cut off burnin iterations and for PT take coldest chains
    if isinstance(sampler, MetropolisHastingsSampler):
        samples = sampler.positions[:, burnin_iter:]
    elif isinstance(sampler, ParallelTemperedSampler):
        samples = sampler.positions[0, :, burnin_iter:]
    else:
        raise ValueError("Unknown sampler type ``{}``".format(type(sampler)))
    # Number of iterations post-burnin
    N = sampler.niterations - burnin_iter
    J = samples.shape[0]
    # Calculate the GH statistic for each parameter independently
    Rs = numpy.zeros(len(params))
    for i, param in enumerate(params):
        # Mean of each chain
        chain_means = numpy.mean(samples[param], axis=1)
        # Mean of means of each chain
        grand_mean = numpy.mean(chain_means)
        B = N / (J - 1) * numpy.sum((chain_means - grand_mean)**2)
        # Within chain variance
        s2 = numpy.var(samples[param], axis=1)

        W = numpy.mean(s2)
        Rs[i] = (1 - (1 / N) + B / W / N)**0.5

    if full:
        return Rs

    return numpy.mean(Rs)
