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

"""Utilities for handling chain data."""

from __future__ import absolute_import

import numpy

from epsie import array2dict


class ChainData(object):
    """Handles reading and adding data to a chain.

    When initialized, a list of parameter names must be provided for which the
    data will be stored. Items may then be added by giving a dictionary mapping
    parameter names to their values. See the Examples below. If the index given
    where the new data should be added is larger then the length of the
    instance's current memory, it will automatically be extended by the amount
    needed. Scratch space may be allocated ahead of time by using the
    ``extend`` method.

    Data can be retrieved using the ``.data`` attribute, which will return the
    data as a numpy structred array.

    Space for multiple temperatures may be specified by providing the
    ``ntemps`` argument. In this case, the array will have shape
    ``niterations x ntemps``.

    .. note::

        Note that the temperatures are the last index. This is because numpy is
        row major. When stepping a chain, the collection of temperatures at a
        given iteration are often accessed, to write data, and to do
        temperature swaps. However, once the chain is complete, it is more
        common to access all the iterations at once for a given temperature;
        e.g., to calculate autocorrelation length. For this reason, it is
        recommended that the chain data be transposed before doing other
        operations and writing to disk.

    Parameters
    ----------
    parameters : list or tuple of str
        The names of the parameters to store data for.
    dtypes : dict, optional
        Dictionary mapping parameter names to types. Will default to using
        ``float`` for any parameters that are not provided.
    ntemps : int, optional
        The number of temperatures used by the chain. Default is 1.

    Attributes
    ----------
    parameters : tuple
        The names of the parameters that were given.
    dtypes : dict
        The data type used for each of the parameters.
    data

    Examples
    --------
    Create an scratch space for two parameters, "x" and "y". Note that,
    initially, the data attribute returns None, and the length is zero:

    >>> from epsie.chain import ChainData
    >>> chaindata = ChainData(['x', 'y'])
    >>> print(len(chaindata))
    0
    >>> print(chaindata.data)
    None

    Now add some data by passing a dictionary of values. Note that the length
    is automatically extended to accomodate the given index, with zeroes filled
    in up to that point:

    >>> chaindata[1] = {'x': 2.5, 'y': 1.}
    >>> chaindata.data
    array([(0. , 0.), (2.5, 1.)], dtype=[('x', '<f8'), ('y', '<f8')])
    >>> len(chaindata)
    2

    Manually extend the scratch space, and fill it. Note that we can set
    multiple values at once using standard slicing syntax:

    >>> chaindata.extend(4)
    >>> chaindata[2:] = {'x': [3.5, 4.5, 5.5, 6.5], 'y': [2, 3, 4, 5]}
    >>> chaindata.data
    array([(0. , 0.), (2.5, 1.), (3.5, 2.), (4.5, 3.), (5.5, 4.), (6.5, 5.)],
      dtype=[('x', '<f8'), ('y', '<f8')])

    Since we did not specify dtypes, the data types have all defaulted to
    floats. Change 'y' to be ints instead:

    >>> chaindata.dtypes = {'y': int}
    >>> chaindata.data
    array([(0. , 0), (2.5, 1), (3.5, 2), (4.5, 3), (5.5, 4), (6.5, 5)],
      dtype=[('x', '<f8'), ('y', '<i8')])

    Clear the memory, and set the new length to be 3:

    >>> chaindata.clear(3)
    >>> chaindata.data
    array([(0., 0), (0., 0), (0., 0)], dtype=[('x', '<f8'), ('y', '<i8')])

    """

    def __init__(self, parameters, dtypes=None, ntemps=1):
        self.parameters = tuple(parameters)
        self._data = None
        self._dtypes = {}
        if dtypes is None:
            dtypes = {}
        self.dtypes = dtypes  # will call the dtypes.setter, below
        self.ntemps = ntemps

    @property
    def data(self):
        """Returns the saved data as a numpy structered array.

        If no data has been added yet, and an initial length was not specified,
        returns ``None``.
        """
        try:
            return self._data
        except AttributeError as e:
            if self._data is None:
                return None
            else:
                raise AttributeError(e)

    @property
    def dtypes(self):
        """Dictionary mapping parameter names to their data types."""
        return self._dtypes

    @dtypes.setter
    def dtypes(self, dtypes):
        """Sets/updates the dtypes to the given dictionary.

        If data has already been saved for a parameter, an attempt will be
        made to cast the data to the new data type.

        A ``ValueError`` will be raised if a parameter name is in the
        dictionary that was not provided upon initialization.

        Parameters
        ----------
        dtypes : dict
            Dictionary mapping parameter names to data types. The data type
            of any parameters not provided will remain their current/default
            (float) type.
        """
        unrecognized = [p for p in dtypes if p not in self.parameters]
        if any(unrecognized):
            raise ValueError("unrecognized parameter(s) {}"
                             .format(', '.join(unrecognized)))
        # store it
        self._dtypes.update(dtypes)
        # fill in any missing parameters
        self._dtypes.update({p: float for p in self.parameters
                             if p not in self._dtypes})
        # create the numpy verion
        self._npdtype = numpy.dtype([(p, self.dtypes[p])
                                     for p in self.parameters])
        # cast to new dtype if data already exists
        if self._data is not None:
            self._data = self._data.astype(self._npdtype)

    def __len__(self):
        if self._data is None:
            return 0
        else:
            return self._data.shape[0]

    def extend(self, n):
        """Extends scratch space by n items.
        """
        if self.ntemps == 1:
            newshape = n
        else:
            newshape = (n, self.ntemps)
        new = numpy.full(newshape, numpy.nan, dtype=self._npdtype)

        if self._data is None:
            self._data = new
        else:
            self._data = numpy.append(self._data, new, axis=0)

    def set_len(self, n):
        """Sets the data length to ``n``.

        If the data length is already > ``n``, a ``ValueError`` is raised.
        """
        lself = len(self)
        if lself < n:
            self.extend(n - lself)
        else:
            raise ValueError("current length ({}) is already greater than {}"
                             .format(lself, n))

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
        if self._data is None:
            raise ValueError("no data has been set yet")
        return self._data[index]

    def asdict(self, index=None):
        """Returns the data as a dictionary.

        Parameters
        ----------
        index : slice, optional
            Only get the elements indicated by the given slice before
            converting to a dictionary.
        """
        if index is None:
            return array2dict(self.data)
        else:
            return array2dict(self[index])

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
        if isinstance(value, dict) and not isinstance(index, str):
            for p in value:
                elem[p] = value[p]
        # otherwise, just fall back to using the structred array's setitem
        else:
            self._data[index] = value


def detect_dtypes(data):
    """Convenience function to detect the dtype of some data.

    Parameters
    ----------
    data : dict or numpy.ndarray
        Either a numpy structred array/void or a dictionary mapping parameter
        names to some (arbitrary) values. The values may be either arrays or
        atomic data. If the former, the dtype will be taken from the array's
        dtype.

    Returns
    -------
    dict :
        Dictionary mapping the parameter names to types.
    """
    if not isinstance(data, dict):  # assume it's a numpy.void or numpy.ndarray
        data = array2dict(data)
    return {p: val.dtype if isinstance(val, numpy.ndarray) else type(val)
            for (p, val) in data.items()}
