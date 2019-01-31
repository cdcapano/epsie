# Copyright (C) 2019  Collin Capano
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

from __future__ import absolute_import

from .ptsampler import ParallelTemperedSampler


class MetropolisHastingsSampler(ParallelTemperedSampler):
    """A standard Metropolis-Hastings sampler.

    Parameters
    ----------
    parameters : tuple or list
        Names of the parameters to sample.
    model : object
        Model object.
    nchains : int
        The number of chains to create. Must be greater than zero.
    proposals : dict, optional
        Dictionary mapping parameter names to proposal classes. Any parameters
        not listed will use the ``default_propsal``.
    default_proposal : an epsie.Proposal class, optional
        The default proposal to use for parameters not in ``proposals``.
        Default is :py:class:`epsie.proposals.Normal`.
    seed : int, optional
        Seed for the random number generator. If None provided, will create
        one.
    pool : Pool object, optional
        Specify a process pool to use for parallelization. Default is to use a
        single core.
    """

    def __init__(self, parameters, model, nchains, proposals=None,
                 default_proposal=None, seed=None, pool=None):
        # just call the parallel tempered sampler with 1 temperature
        super(MetropolisHastingsSampler, self).__init__(
            parameters, model, nchains, betas=1, proposals=proposals,
            default_proposal=default_proposal, seed=seed, pool=pool)
