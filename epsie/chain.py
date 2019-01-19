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

class _ChainMem(object):
    """Provides easy IO for adding and reading data from chains.
    """

    def __init__(self, parameters, dtype=None):
        self.parameters = parameters
        self.dtype = None
        if dtype is not None:
            self.set_dtype(**dtype)
        self.mem = None

    def set_dtype(self, **dtypes):
        """Sets the dtype to use for storing values.

        A type for every parameter must be provided.

        Parameters
        ----------
        \**dtypes :
            Parameter types should be specified as keyword arguments. Every
            parameter must be provided.
        """
        self.dtype = [(p, dtypes.pop(p)) for p in self.parameters]
        if dtypes:
            raise ValueError("unrecognized parameters {}"
                             .format(', '.join(dtypes.keys())))

    def detect_dtype(self, params):
        """Detects the dtype to use given some parameter values.

        Parameters
        ----------
        params : dict
            Dictionary mapping parameter names to some (arbitrary) values.
        """
        self.set_dtype(**{p: type(val) for p, val in params.items()})

    def extend(self, n):
        """Extends scratch space by n samples.
        """
        if self.dtype is None:
            raise ValueError("dtype not set! Run set_dtype")
        extra = numpy.zeros(n, dtype=self.dtype)
        if self.mem is None:
            self.mem = extra
        else:
            self.mem = numpy.append(self.mem, extra)

    def clear(self):
        """Clears the memory."""
        self.mem = None

    def __repr__(self):
        return repr(self.mem)

    def __getitem__(self, ii):
        return self.mem[ii]

    def __setitem__(self, ii, params):
        if self.dtype is None:
            self.detect_dtype(params)
        vals = tuple(params.pop(p) for p in self.parameters)
        # check for unknown params
        if params:
            raise ValueError("unrecognized parameters {}"
                             .format(', '.join(params.keys())))
        try:
            self.mem[ii] = vals
        except (TypeError, IndexError):
            self.extend(ii+1)
            self.mem[ii] = vals


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
    random_state : :py:class:`numpy.random.RandomState` or int, optional
        An instance of :py:class:`numpy.random.RandomState` to use for
        generating random variates. If an int or None is provided, a
        :py:class:`numpy.random.RandomState` will be created instead, with
        ``random_state`` used as a seed.
    chain_id : int, optional
        An interger identifying which chain this is. Optional; if not provided,
        the ``chain_id`` attribute will just be set to None.
    """
    def __init__(self, parameters, model, proposals, random_state=None,
                 chain_id=None):
        self.parameters = parameters
        self.model = model
        # combine the proposals into a joint proposal
        self.proposal_dist = JointProposal(*proposals,
                                           random_state=random_state)
        self.chain_id = chain_id
        self._iteration = 0
        self._lastclear = 0
        self.positions = _ChainMem(parameters)
        self.stats = _ChainMem(['logp', 'logl'],
                               {'logp': float, 'logl': float})
        self.acceptance_ratios = _ChainMem(['acceptance_ratio'],
                                           {'acceptance_ratio': float})
        self.blobs = None
        self._hasblobs = False
        self._p0 = None
        self._logp0 = None
        self._logl0 = None
        self._blob0 = None

    def set_p0(self, p0):
        """Sets the starting position.

        This also evaulates the log likelihood and log prior at the starting
        position, as well as determine if the model returns blob data.

        Parameters
        ----------
        p0 : dict
            Dictionary mapping parameters to single values.
        """
        self._p0 = p0.copy()
        # evaluate logl, p at this point
        r = self.model(**p0)
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
            self.blobs = _ChainMem(blob.keys())
            self.blobs.detect_dtype(blob)
        self._logp0 = logp
        self._logl0 = logl
        self._blob0 = blob

    @property
    def random_state(self):
        """Returns the ``RandomState`` class."""
        return self.proposal_dist.random_state

    @property
    def p0(self):
        if self._p0 is None:
            raise ValueError("p0 not set! Run set_p0")
        return self._p0

    @property
    def iteration(self):
        return self._iteration

    @property
    def lastclear(self):
        """Returns the iteration of the last time the chain memory was cleared.
        """
        return self._lastclear

    @property
    def _index(self):
        """The index in memory of the current iteration."""
        return self._iteration - 1 - self._lastclear

    def clear(self):
        """Clears memory of the current chain, and sets p0 to the current
        position."""
        if self._iteration > 0:
            # save position then clear
            self._p0 = self.current_position.copy()
            self.positions.clear()
            # save stats then clear
            self._logp0 = self.current_stats['logp']
            self._logl0 = self.current_stats['logl']
            self.stats.clear()
            # clear acceptance ratios
            self.acceptance_ratios.clear()
            # save blobs then clear
            if self._hasblobs:
                self._blob0 = self.current_blob.copy()
                self.blobs.clear()
        self._lastclear = self._iteration

    @property
    def current_position(self):
        try:
            return self.positions[self._index]
        except (TypeError, IndexError):
            return self.p0

    @property
    def current_stats(self):
        try:
            return self.stats[self._index]
        except (TypeError, IndexError):
            return {'logp': self._logp0, 'logl': self._logl0}

    @property
    def current_blob(self):
        try:
            return self.blobs[self._index]
        except (TypeError, IndexError):
            return self._blob0

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
        self._p0 = state['current_position']
        self._logl0 = state['current_stats']['logl']
        self._logp0 = state['current_stats']['logp']
        self._blob0 = state['current_blob']
        # set the proposals' states
        self.proposal_dist.set_state(state['proposal_dist'])


    def step(self):
        """Evolves the chain by a single step."""
        # in case the proposal needs information about the history of the
        # chain
        self.proposal_dist.update(self)
        # now call a proposal
        current_pos = self.current_position
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
        logar = logp + logl - current_logl - current_logp
        if not self.proposal_dist.symmetric:
            logar += self.proposal_dist.logpdf(current_pos, proposal) - \
                     self.proposal_dist.logpdf(proposal, current_pos)
        ar = numpy.exp(logar)
        u = randuniform.rvs(random_state=self.random_state)
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
        self._iteration += 1
        self.positions[self._index] = pos
        self.stats[self._index] = stats
        self.acceptance_ratios[self._index] = {'acceptance_ratio': ar}
        if self._hasblobs:
            self.blobs[self._index] = blob
