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

"""EPSIE is a toolkit for doing MCMCs using embarrasingly parallel Markov
chains.
"""

from __future__ import absolute_import

# get the version number
from ._version import __version__

import os
import sys
import pickle
from io import BytesIO
import numpy
from randomgen import PCG64

#
# =============================================================================
#
#                     Random number generation utilities
#
# =============================================================================
#

# The bit generator used for all random number generation.
# Users may change this, but it has to be something recognized by randomgen
# and have the ability to accept streams; i.e., it must have calling structure
# class(seed, stream, mode), with default stream set to None.
BIT_GENERATOR = PCG64


def create_seed(seed=None):
    """Creates a seed for a random number generator.

    Parameters
    ----------
    seed : int, optional
        If a seed is given, will just return it. Default is None, in which case
        a seed is created.

    Returns
    -------
    int :
        A seed to use.
    """
    if seed is None:
        # use os.urandom to get a string of random 4 bytes
        bseed = os.urandom(4)
        # convert to int
        if sys.version_info < (3,):
            # py27
            seed = sum([ord(c) << (i * 8) for i, c in enumerate(bseed[::-1])])
        else:
            # Py3XX
            seed = int.from_bytes(bseed, byteorder='big')
    return seed


def create_bit_generator(seed=None, stream=0):
    """Creates a an instance of :py:class:`epsie.BIT_GENERATOR`.

    Parameters
    ----------
    seed : int, optional
        The seed to use. If seed is None (the default), will create a seed
        using ``create_seed``.
    stream : int, optional
        The stream to create the bit generator for. This allows multiple
        generators to exist with the same seed, but that produce different sets
        of random numbers. Default is 0.
    """
    if seed is None:
        seed = create_seed(seed)
    try:
        return BIT_GENERATOR(seed, stream, mode="sequence")
    except TypeError as e:
        # eaerlier versions of randomgen (used for py27) did not support a
        # mode argument
        import randomgen
        if float('.'.join(randomgen.__version__.split('.')[:2])) < 1.17:
            return BIT_GENERATOR(seed, stream)
        else:
            raise e


def create_bit_generators(seed, ngenerators):
    """Creates a collection of random bit generators.

    The bit generators are different streams with the same seed. They are all
    statistically independent of each other, while still being reproducable.
    """
    return [BIT_GENERATOR(seed, ii, mode="sequence")
            for ii in range(ngenerators)]


#
# =============================================================================
#
#                          Array utilities
#
# =============================================================================
#


def array2dict(array):
    """Converts a structured array into a dictionary."""
    fields = array.dtype.names  # raises an AttributeError if array is None
    if fields is None:
        # not a structred array, just return
        return array
    return {f: _getatomic(array[f]) for f in fields}


def _getatomic(val):
    """Checks if a given value is numpy scalar. If so, it returns
    the value as its native python type.
    """
    if isinstance(val, numpy.ndarray) and val.size == 1 and val.ndim == 0:
        val = val.item(0)
    return val


#
# =============================================================================
#
#                          Parallel tempering utilities
#
# =============================================================================
#


def make_betas_ladder(ntemps, maxtemp):
    """Makes a log spaced ladder of betas."""
    minbeta = 1./maxtemp
    return numpy.geomspace(minbeta, 1., num=ntemps)


#
# =============================================================================
#
#                          Checkpointing utilities
#
# =============================================================================
#


def dump_state(state, fp, path=None, dsetname='sampler_state', protocol=None):
    """Dumps the given state to an hdf5 file handler.

    The state is stored as a raw binary array to ``{path}/{dsetname}`` in the
    given hdf5 file handler. If a dataset with the same name and path is
    already in the file, the dataset will be resized and overwritten with the
    new state data.

    Parameters
    ----------
    state : any picklable object
        The sampler state to dump to file. Can be the object returned by
        any of the samplers' `.state` attribute (a dictionary of dictionaries),
        or any picklable object.
    fp : h5py.File
        An open hdf5 file handler. Must have write capability enabled.
    path : str, optional
        The path (group name) to store the state dataset to. Default (None)
        will result in the array being stored to the top level.
    dsetname : str, optional
        The name of the dataset to store the binary array to. Default is
        ``sampler_state``.
    protocol : int, optional
        The protocol version to use for pickling. See the :py:mod:`pickle`
        module for more details.
    """
    memfp = BytesIO()
    pickle.dump(state, memfp, protocol=protocol)
    dump_pickle_to_hdf(memfp, fp, path=path, dsetname=dsetname)


def dump_pickle_to_hdf(memfp, fp, path=None, dsetname='sampler_state'):
    """Dumps pickled data to an hdf5 file object.

    Parameters
    ----------
    memfp : file object
        Bytes stream of pickled data.
    fp : h5py.File
        An open hdf5 file handler. Must have write capability enabled.
    path : str, optional
        The path (group name) to store the state dataset to. Default (None)
        will result in the array being stored to the top level.
    dsetname : str, optional
        The name of the dataset to store the binary array to. Default is
        ``sampler_state``.
    """
    memfp.seek(0)
    bdata = numpy.frombuffer(memfp.read(), dtype='S1')
    if path is not None:
        fp = fp[path]
    if dsetname not in fp:
        fp.create_dataset(dsetname, shape=bdata.shape, maxshape=(None,),
                          dtype=bdata.dtype)
    elif bdata.size != fp[dsetname].shape[0]:
        fp[dsetname].resize((bdata.size,))
    fp[dsetname][:] = bdata


def load_state(fp, path=None, dsetname='sampler_state'):
    """Loads a sampler state from the given hdf5 file object.

    The sampler state is expected to be stored as a raw bytes array which can
    be loaded by pickle.

    Parameters
    ----------
    fp : h5py.File
        An open hdf5 file handler.
    path : str, optional
        The path (group name) that the state data is stored to. Default (None)
        is to read from the top level.
    dsetname : str, optional
        The name of the dataset that the state data is stored to. Default is
        ``sampler_state``.
    """
    if path is not None:
        fp = fp[path]
    bdata = fp[dsetname][()].tobytes()
    return pickle.load(BytesIO(bdata))
