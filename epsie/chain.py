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

"""Classes for individual Markov chains."""

from __future__ import absolute_import

import numpy
from scipy.stats import uniform as randuniform

from .proposals import JointProposal


class ChainData(object):
    """Provides easy IO for adding and reading data from chains.
    """

    def __init__(self, parameters, dtypes=None, initial_len=None):
        self.parameters = tuple(parameters)
        self._data = None
        if dtypes is None:
            dtypes = {}
        self.set_dtype(**dtypes)
        if initial_len is not None:
            self.extend(initial_len)

    @staticmethod
    def a2d(array):
        """Converts a structured array into a dictionary."""
        fields = array.dtype.names  # raises an AttributeError if array is None
        if fields is None:
            # not a structred array, just return
            return array
        return {f: array[f] for f in fields}

    @property
    def data(self):
        try:
            return self.a2d(self._data)
        except AttributeError as e:
            if self._data is None:
                return None
            else:
                raise AttributeError(e)

    def set_dtype(self, **dtypes):
        """Sets the data types to use for the parameters.

        If ``data`` is not currently None, the data will be cast to the new
        data types.

        Parameters
        ----------
        \**dtypes :
            The keyword arguments should map parameter names to types. Any
            parameters not specified will default to ``float``. A
            ``ValueError`` will be raised if any parameters are given that are
            not in the ``parameters`` attribute.
        """
        # fill in any missing parameters
        dtypes.update({p: float for p in self.parameters if p not in dtypes})
        dtype = numpy.dtype([(p, dtypes.pop(p)) for p in self.parameters])
        # make sure there were no unrecognized parameters
        if dtypes:
            raise ValueError("unrecognized parameter(s) {}"
                             .format(', '.join(dtypes.keys())))
        self.dtype = dtype
        # cast to new dtype if data already exists
        if self._data is not None:
            self._data = self._data.astype(dtype)

    def __len__(self):
        if self._data is None:
            return 0
        else:
            return self._data.size

    def extend(self, n):
        """Extends scratch space by n items.
        """
        new = numpy.zeros(n, dtype=self.dtype)
        if self._data is None:
            self._data = new
        else:
            self._data = numpy.append(self._data, new)

    def clear(self, newlen=None):
        """Clears the memory.
        
        Parameters
        ----------
        newlen : int, optional
            If provided, will create new scratch space with the given length.
        """
        self._data = None
        if newlen is None:
            newlen = 0
        self.extend(newlen)

    def __repr__(self):
        return repr(self.data)

    def __getitem__(self, index):
        return self.a2d(self._data[index])

    def __setitem__(self, index, value):
        # try to get the element to set; if it fails, then try extending the
        # data by the amount needed
        try:
            elem = self._data[index]
        except (IndexError, TypeError):  # get TypeError if _data is None
            self.extend(index + 1 - len(self))
            elem = self._data[index]
        # if value is a dictionary and index is not a string, then we're
        # setting values in the array by dictionary
        if isinstance(value, dict) and not isinstance(index, (str, unicode)):
            for p in value:
                elem[p] = value[p]
        # otherwise, just fall back to using the structred array's setitem
        else:
            self._data[index] = value


def detect_dtypes(data):
    """Convenience function to detect the dtype of a dictionary of data.

    Parameters
    ----------
    data : dict
        Dictionary mapping parameter names to some (arbitrary) values.
        The values may be either arrays or atomic data. If the former, the
        dtype will be taken from the array's dtype.

    Returns
    -------
    dict :
        Dictionary mapping the parameter names to types.
    """
    return {
        p: val.dtype if isinstance(val, numpy.ndarray) else type(val)
        for p,val in data.items()}



class Chain(object):
    """A Markov chain.

    The chain requires a ``model`` to  be provided. This can be any object that
    can be called with keyword arguments that map parameter names to values.
    When called, the object must return a tuple of two or three elements. The
    first element must be a float representing the log likelihood at that
    point; the second must be a float representing the log prior at that point.
    Additionally, the model may return a dictionary as a third element in the
    tuple that maps strings to arbitrary data. This additional data will be
    stored as ``blob`` data.

    Parameters
    ----------
    parameters : list or tuple
        List of the parameter names to sample.
    model : object
        Any object that can be called with keyword arguments that map parameter
        names to values. When called, the object must a tuple of ``logl, logp``
        where ``logp`` is the log prior and ``logl`` is the log likelihood at
        the given point. The model may optionally return a dictionary in
        addition that maps strings to any arbitrary data.
    proposals : list of epsie.proposals instances
        List of proposal classes to use for the parameters. There must be one
        and only one proposal for every parameter. A single proposal may cover
        multiple parameters. Proposals must be instances of classes that
        inherit from :py:class:`epsie.proposals.BaseProposal`.
    brng : :py:class:`randomgen.PGC64` instance, optional
        Use the given basic random number generator (BRNG) for generating
        random variates. If an int or None is provided, a BRNG will be
        created instead using ``brng`` as a seed.
    scratch_len : int, optional
        Set the length of memory to use for scratch space.
    chain_id : int, optional
        An interger identifying which chain this is. Optional; if not provided,
        the ``chain_id`` attribute will just be set to None.
    """
    def __init__(self, parameters, model, proposals, brng=None,
                 scratch_len=None, chain_id=None):
        self.parameters = parameters
        self.model = model
        # combine the proposals into a joint proposal
        self.proposal_dist = JointProposal(*proposals, brng=brng)
        self.chain_id = chain_id
        self._iteration = 0
        self._lastclear = 0
        self.scratch_len = scratch_len
        self._positions = ChainData(parameters, initial_len=scratch_len)
        self._stats = ChainData(['logl', 'logp'], initial_len=scratch_len)
        self._acceptance_ratios = ChainData(['ar'],
                                            initial_len=scratch_len)
        self._blobs = None
        self._hasblobs = False
        self._start = None
        self._logp0 = None
        self._logl0 = None
        self._blob0 = None

    def set_start(self, position):
        """Sets the starting position.

        This also evaulates the log likelihood and log prior at the starting
        position, as well as determine if the model returns blob data.

        Parameters
        ----------
        position : dict
            Dictionary mapping parameters to single values.
        """
        self._start = position.copy()
        # use start position to determine the dtype for positions
        self._positions.set_dtype(**detect_dtypes(position))
        # evaluate logl, p at this point
        r = self.model(**position)
        try:
            logl, logp, blob = r
            self._hasblobs = True
        except ValueError:
            logl, logp = r
            blob = None
            self._hasblobs = False
        if self._hasblobs:
            if not isinstance(blob, dict):
                raise TypeError("model must return blob data as a dictionary")
            self._blobs = ChainData(blob.keys(), dtypes=detect_dtypes(blob),
                                    initial_len=self.scratch_len)
        # check that we're not starting outside of the prior
        if logp == -numpy.inf:
            raise ValueError("starting position is outside of the prior!")
        self._logp0 = logp
        self._logl0 = logl
        self._blob0 = blob

    def __len__(self):
        return self._iteration - self._lastclear

    @property
    def iteration(self):
        return self._iteration

    @property
    def lastclear(self):
        """Returns the iteration of the last time the chain memory was cleared.
        """
        return self._lastclear

    @property
    def start_position(self):
        if self._start is None:
            raise ValueError("Starting position not set! Run set_start.")
        return self._start

    @property
    def logl0(self):
        """The log likelihood of the starting position."""
        return self._logl0

    @property
    def logp0(self):
        """The log prior of the starting position."""
        return self._logp0

    @property
    def positions(self):
        return self._positions[:len(self)]

    @property
    def stats(self):
        return self._stats[:len(self)]

    @property
    def acceptance_ratios(self):
        return self._acceptance_ratios[:len(self)]['ar']

    @property
    def blobs(self):
        blobs = self._blobs
        if blobs is not None:
            blobs = blobs[:len(self)]
        return blobs

    @property
    def hasblobs(self):
        return self._hasblobs

    def __getitem__(self, index):
        """Returns all of the chain data at the requested index."""
        index = (-1)**(index < 0) * (index % len(self))
        out = {'positions': self._positions[index],
               'stats': self._stats[index],
               'acceptance_ratios': self._acceptance_ratios[index]['ar']
               }
        if self._hasblobs:
            out['blobs'] = self._blobs[index]
        return out

    @property
    def current_position(self):
        if len(self) == 0:
            pos = self.start_position
        else:
            pos = self._positions[len(self)-1]
        return pos

    @property
    def current_stats(self):
        if len(self) == 0:
            stats = {'logl': self._logl0, 'logp': self._logp0}
        else:
            stats = self._stats[len(self)-1]
        return stats

    @property
    def current_blob(self):
        if not self._hasblobs:
            blob = None
        elif len(self) == 0:
            blob = self._blob0
        else:
            blob = self._blobs[len(self)-1]
        return blob

    def clear(self):
        """Clears memory of the current chain, and sets start position to the
        current position.
        
        New scratch space will be created with length equal to ``scratch_len``.
        """
        if self._iteration > 0:
            # save position then clear
            self._start = self.current_position.copy()
            self._positions.clear(self.scratch_len)
            # save stats then clear
            self._logp0 = self.current_stats['logp']
            self._logl0 = self.current_stats['logl']
            self._stats.clear(self.scratch_len)
            # clear acceptance ratios
            self._acceptance_ratios.clear(self.scratch_len)
            # save blobs then clear
            if self._hasblobs:
                self._blob0 = self.current_blob.copy(self.scratch_len)
                self._blobs.clear(self.scratch_len)
        self._lastclear = self._iteration
        return self

    @property
    def brng(self):
        """Returns basic random number generator (BRNG) being used."""
        return self.proposal_dist.brng

    @property
    def random_generator(self):
        """Returns the random number generator."""
        return self.brng.generator

    @property
    def random_state(self):
        """Returns the current state of the BRNG."""
        return self.proposal_dist.random_state

    @property
    def state(self):
        """Returns the current state of the chain.

        The state consists of everything needed such that setting a chain's
        state using another's state will result in identical results.
        """
        state = {}
        state['chain_id'] = self.chain_id
        state['proposal_dist'] = self.proposal_dist.state
        state['iteration'] = self.iteration
        state['current_position'] = self.current_position
        state['current_stats'] = self.current_stats
        state['current_blob'] = self.current_blob
        state['hasblobs'] = self._hasblobs
        return state

    def set_state(self, state):
        """Sets the state of the chain using the given dict.

        .. warning::
           Running this will clear the chain's current memory, and replace its
           current position with what is saved in the state.

        Parameters
        ----------
        state : dict
            Dictionary of state values.
        """
        self.chain_id = state['chain_id']
        # set the chain position
        self.clear()
        self._iteration = state['iteration']
        self._lastclear = state['iteration']
        self._start = state['current_position']
        self._positions.set_dtype(**detect_dtypes(self._start))
        self._logl0 = state['current_stats']['logl']
        self._logp0 = state['current_stats']['logp']
        self._blob0 = state['current_blob']
        self._hasblobs = state['hasblobs']
        # set the proposals' states
        self.proposal_dist.set_state(state['proposal_dist'])


    def step(self):
        """Evolves the chain by a single step."""
        # get the current position; if this is the first step and set_start
        # hasn't been run, this will raise a ValueError
        current_pos = self.current_position
        # in case the any of the proposals need information about the history
        # of the chain:
        self.proposal_dist.update(self)
        # now call a proposal
        proposal = self.proposal_dist.jump(current_pos)
        r = self.model(**proposal)
        if self._hasblobs:
            logl, logp, blob = r
        else:
            logl, logp = r
        # evaluate
        current_stats = self.current_stats
        current_logl = current_stats['logl']
        current_logp = current_stats['logp']
        if logp == -numpy.inf:
            # force a reject
            ar = 0.
        else:
            logar = logp + logl - current_logl - current_logp
            if not self.proposal_dist.symmetric:
                logar += self.proposal_dist.logpdf(current_pos, proposal) - \
                         self.proposal_dist.logpdf(proposal, current_pos)
            ar = numpy.exp(logar)
        u = self.random_generator.uniform()
        if u <= ar:
            # accept
            pos = proposal
            stats = {'logl': logl, 'logp': logp}
        else:
            # reject
            pos = current_pos
            stats = current_stats
            blob = self.current_blob
        # save
        ii = len(self)
        self._positions[ii] = pos
        self._stats[ii] = stats
        self._acceptance_ratios[ii] = ar
        if self._hasblobs:
            self._blobs[ii] = blob
        self._iteration += 1
        return self
