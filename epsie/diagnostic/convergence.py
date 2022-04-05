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


def gelman_rubin_test(sampler, burnin_iter):
    if sampler.name == "mh_sampler":
        return _mh_gelman_rubin_test(sampler, burnin_iter)
    else:
        raise NotImplementedError("Other sampler not implemented yet.")



def _mh_gelman_rubin_test(sampler, burnin_iter, full=False):
    params = sampler.parameters

    samples = sampler.positions[:, burnin_iter:]

    N = sampler.niterations - burnin_iter

    Rs = numpy.zeros(len(params))

    for i, param in enumerate(params):
        chains_variance = numpy.var(samples[param], axis=1)
        chains_mean = numpy.mean(samples[param], axis=1)

        W = numpy.mean(chains_variance)
        B = numpy.var(chains_mean)

        V = (1 - 1 / N) * W + B
        Rs[i] = numpy.sqrt(V / W) 

    if full:
        return Rs 
    
    return numpy.mean(Rs)
    