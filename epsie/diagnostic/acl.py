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

"""Autocorrelation time calculation."""
import numpy


def acl_1d(x, c=5.0):
    """
    Calculate the autocorrelation length (ACL) of some series.
    
    Algorithm used is from:
    N. Madras and A.D. Sokal, J. Stat. Phys. 50, 109 (1988).

    Implementation from:
    https://emcee.readthedocs.io/en/stable/tutorials/autocorr/#autocorr

    Arguments
    ---------
    x : 1-dimensional array
        Samples whose autocorrelation length is to be calculated.
    c : float, optional
        ACL hyperparameter. As per Sokal by default set to 5.
    
    Returns
    -------
    ACL: int
        Autocorrelation length.
    """
    if x.ndim != 1:
        raise TypeError("``x`` must be a 1D array. Currently {}".format(x.ndim))
    # n is the next power of 2 for length of x
    n = (2**numpy.ceil(numpy.log2(x.size))).astype(int)
    # Compute the FFT and then (from that) the auto-correlation function
    f = numpy.fft.fft(x - numpy.mean(x), n=2 * n)
    acf =  numpy.fft.ifft(f * numpy.conjugate(f))[: len(x)].real / (4 * n)

    acf /= acf[0]
    
    taus = 2.0 * numpy.cumsum(acf) - 1.0
    window = auto_window(taus, c)
    return numpy.ceil(taus[window]).astype(int)

def auto_window(taus, c):
    """
    ACL automated windowing procedure following Sokal (1989).

    Implementation from:
    https://emcee.readthedocs.io/en/stable/tutorials/autocorr/#autocorr
    """
    m = numpy.arange(len(taus)) < c * taus
    if numpy.any(m):
        return numpy.argmin(m)
    return len(taus) - 1


def acl_chain(chain, burnin_iter=0, c=5.0, full=False):
    """
    Calculate the ACL of :py:class:`epsie.chain.Chain` chain.

    Arguments
    ---------
    chain : :py:class:`epsie.chain.Chain`
        Epsie chain.
    burnin_iter : int, optional
        Number of burnin iterations to be thrown away.
    c : float, optional
        ACL calculation hyperparameter. By default 5.
    full : bool, optionall
        Whether to return the ACL of every parameter. By default False, thus
        returning only the maximum ACL.
    
    Returns
    -------
    ACL : int or array of integers
        Chain's ACL(s).
    """
    acls = [acl_1d(chain.positions[p][burnin_iter:]) for p in chain.parameters]
    if full:
        return acls
    return max(acls)


def thinned_samples(sampler, burnin_iter=0, c=5.0, temp_acl_func=None):
    """
    Parse a sampler and return its samples thinned by the ACL.

    Arguments
    ---------
    sampler : {:py:class:`epsie.sampler.MetropolisHastingsSampler`,
               :py:class:`epsie.sampler.ParallelTemperedSampler`}
        Epsie sampler whose samples to extract.
    burnin_iter : int, optional
        Number of burnin iterations to be thrown away.
    c : float, optional
        ACL calculation hyperparameter. By default 5.
    temp_acl_func : :py:func, optional.
        A function that given a list of ACLs of a single chain on different
        temperature levels returns the chain global ACL. By default 
        :py:func`numpy.max`. Must act on a 1-dimensional array and return an
        integer.

    Returns
    -------
    thinned samples : structured array
        Array whose `dtype.names` are the sampler parameters. Each parameter's
        data is 1-dimensional for the MH sampler and 2-dimensional for the PT
        sampler. In the latter case the first and second axis represent the
        temperature and sample index.
    """
    if sampler.name == "mh_sampler":
        return _thinned_mh_samples(sampler, burnin_iter, c)
    elif sampler.name == "pt_sampler":
        return _thinned_pt_samples(sampler, burnin_iter, c, temp_acl_func)
    else:
        return ValueError("Invalid sampler kind.")


def _thinned_mh_samples(sampler, burnin_iter=0, c=5.0):
    # Calculate the ACL for each chain 
    acls = [acl_chain(chain, burnin_iter, c) for chain in sampler.chains]

    params = sampler.parameters
    _thinned = {p: [] for p in params}
    # Explicitly cut off the burnin iterations
    samples = sampler.positions[:, burnin_iter:]

    # Cycle over the chains and thin them
    for ii, chain in enumerate(sampler.chains):
        for p in params:
            _thinned[p].append(samples[p][ii, :][::-1][::acls[ii]][::-1])
    # Put the thinned samples into a structured array
    N = sum(x.size for x in _thinned[params[0]])
    thinned = numpy.zeros(N, dtype=samples.dtype)
    for p in params:
        thinned[p] = numpy.concatenate(_thinned[p])
    return thinned


def _thinned_pt_samples(sampler, burnin_iter=0, c=5.0, temp_acl_func=None):
    # Calculate the ACL across temperatures for each chain    
    temp_acls = numpy.zeros((sampler.nchains, sampler.ntemps), dtype=int)
    for ii in range(sampler.nchains):
        for jj in range(sampler.ntemps):
            temp_acls[ii, jj] = acl_chain(sampler.chains[ii].chains[jj], burnin_iter, c=0.5)
    # Grab the ACL for each chain. Typically the maximum ACL among the temps.
    if temp_acl_func is None:
        temp_acl_func = numpy.max
    acls = numpy.apply_along_axis(temp_acl_func, axis=1, arr=temp_acls)
    # Ensure these are integers
    acls = numpy.ceil(acls).astype(int)

    params = sampler.parameters
    samples = sampler.positions[:, :, burnin_iter:]

    thinned = {p: [] for p in params}
    # Cycle over the chains, thinning them
    for tk in range(sampler.ntemps):
        _thinned = {p: [] for p in params}

        for ii in range(sampler.nchains):
            for param in params:
                sp = samples[param][tk, ii, :]
                _thinned[param].append(sp[::-1][::acls[ii]][::-1])
        
        for param in params:
            thinned[param].append(numpy.concatenate(_thinned[param]))
    
    # Cast the thinned samples from a list of arrays into a 2D array for each
    # parameter and return as a structured array
    thinned_samples = numpy.zeros((sampler.ntemps, thinned[params[0]][0].size),
                                  dtype=samples.dtype)
    for param in params:
        thinned_samples[param] = numpy.asarray(thinned[param])
    return thinned_samples
