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


class _ChainMem(object):
    """Provides easy IO for adding and reading data from chains.
    """

    def __init__(parameters):
        self.parameters = parameters
        self.dtype = None
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
        except TypeError, IndexError:
            self.extend(ii)
            self.mem[ii] = vals


class Chain(object):
    def __init__(self, parameters, model, propose):
        self.parameters = parameters
        self.model = model
        self.propose = propose
        self._iteration = 0
        self.positions = _ChainMem(parameters)
        self.stats = _ChainMem(['logp', 'logl'])
        self.acceptance_ratios = _ChainMem(['acceptance_ratio'])
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
            logp, logl, blob = r
            self._hasblobs = True
        except ValueError:
            logp, logl = r
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
    def p0(self):
        if self._p0 is None:
            raise ValueError("p0 not set! Run set_p0")
        return self._p0

    @property
    def iteration(self):
        return self._iteration

    @property
    def current_position(self):
        if self.positions.mem is None:
            return self.p0
        else:
            return self.positions[self._iteration]

    @property
    def current_stats(self):
        if self.stats.mem is None:
            return {'logp': self._logp0, 'logl': self._logl0}
        else:
            return self.stats[self._iteration]

    @property
    def current_blob(self):
        if not self._hasblobs or self.blobs.mem is None:
            return self._blob0
        else:
            return self.blobs[self._iteration]

    def step(self):
        """Evolves the chain by a single step."""
        # in case the proposal needs information about the history of the
        # chain
        self.propose.update(self)
        # now call a proposal
        proposal = self.propose.jump()
        r = self.model(**proposal)
        if self._hasblobs:
            logp, logl, blob = r
        else:
            logp, logl = r
        # evaluate
        current_stats = self.current_stats
        current_logl = current_stats['logl']
        current_logp = current_stats['logp']
        ar = numpy.exp(logp + logl - current_logl - current_logp)
        u = numpy.unifom()
        if u <= ar:
            # keep
            pos = proposal
            stats = {'logl': logl, 'logp': logp}
        else:
            # reject
            pos = self.current_position
            stats = current_stats
            blob = self.current_blob
        self._iteration += 1
        self.positions[self._iteration] = pos
        self.stats[self._iteration] = stats
        self.acceptance_ratios[self._iteration] = {'acceptance_ratio:' ar}
        if self._hasblobs:
            self.blobs[self._iteration] = blob
