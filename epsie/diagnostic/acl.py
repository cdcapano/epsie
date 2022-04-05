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


"""
Autocorrelation time calculation. 1D autocorrelation code inspired from
https://emcee.readthedocs.io/en/stable/tutorials/autocorr/#autocorr.
"""

import numpy


def acl_1d(x, c=5.0):
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
#    print(taus[window])
    return numpy.round(taus[window]).astype(int)

def auto_window(taus, c):
    """
    Automated windowing procedure following Sokal (1989)
    """
    m = numpy.arange(len(taus)) < c * taus
    if numpy.any(m):
        return numpy.argmin(m)
    return len(taus) - 1


def acl_chain(chain, burnin_iter=0, c=5.0, full=False):
    acls = [acl_1d(chain.positions[p][burnin_iter:]) for p in chain.parameters]
    if full:
        return acls
    return max(acls)


def thinned_samples(sampler, burnin_iter=0, c=5.0):
    if sampler.name == "mh_sampler":
        return _thinned_mh_samples(sampler)
    elif sampler.name == "pt_sampler":
        raise NotImplementedError("PT thinning not implemented yet")
    else:
        return ValueError("Invalid sampler kind.")

def _thinned_mh_samples(sampler, burnin_iter=0, c=5.0):
    params = sampler.chains[0].parameters
    _thinned = {p: [] for p in params}
    positions = sampler.positions[:, burnin_iter:]

    # cycle over the chains, calculating the ACLs and thinning them
    for i, chain in enumerate(sampler.chains):
        acl = acl_chain(chain, burnin_iter, c)
        for p in params:
            _thinned[p].append(positions[p][i, :][::-1][::acl][::-1])
    
    N = sum(x.size for x in _thinned[params[0]])
    thinned = numpy.zeros(N, dtype=positions.dtype)
    for p in params:
        thinned[p] = numpy.concatenate(_thinned[p])
    return thinned