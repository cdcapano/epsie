# Copyright (C) 2020  Richard Stiskalek, Collin Capano
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

from itertools import product

import numpy
from scipy import stats
from scipy.spatial import ConvexHull

from .eigenvector import (Eigenvector, AdaptiveEigenvectorSupport)
from .bounded_normal import Boundaries


class BoundedEigenvector(Eigenvector):
    """
    Bounded Eigenvector
    """
    name = 'bounded_eigenvector'
    symmetric = False
    _boundaries = None
    _lowerbnd = None
    _upperbnd = None
    _hyperplanes = None

    def __init__(self, parameters, boundaries, stability_duration,
                 initial_proposal='at_adaptive_bounded_normal',
                 shuffle_rate=0.33):
        super(BoundedEigenvector, self).__init__(parameters, stability_duration,
                                                 shuffle_rate=shuffle_rate)
        # set the boundaries
        self.boundaries = boundaries
        # set the initial phase settings
        self.set_initial_proposal(initial_proposal, stability_duration,
                                  boundaries)
        # cache the boundaries for repeated calls
        self._old_hash = None
        self._cache= {}

    @property
    def boundaries(self):
        """Dictionary of parameter -> boundaries."""
        return self._boundaries

    @property
    def hyperplanes(self):
        """Returns an array of hyperplanes defining the boundaries"""
        return self._hyperplanes

    @boundaries.setter
    def boundaries(self, boundaries):
        """Sets the boundaries, making sure that widths are provided for
        each parameter in ``parameters``.
        """
        try:
            self._boundaries = {p: Boundaries(boundaries[p])
                                for p in self.parameters}
        except KeyError:
            raise ValueError("must provide a boundary for every parameter")
        # set lower and upper bound arrays for speed
        self._lowerbnd = numpy.array([self._boundaries[p][0]
                                      for p in self.parameters])
        self._upperbnd = numpy.array([self._boundaries[p][1]
                                      for p in self.parameters])
        # calculate the hyperplanes
        X = numpy.vstack([i for i in product(*(boundaries[p]
                                               for p in self.parameters))])
        hull = ConvexHull(X)
        self._hyperplanes = numpy.unique(hull.equations, axis=0)

    def __contains__(self, testpt):
        # checks if the given parameters are in the bounds
        testpt = testpt.copy()
        isin = None
        for p in self.parameters:
            try:
                val = testpt.pop(p)
            except KeyError:
                # only testing a subset of the parameters, which is allowed
                continue
            bnds = self.boundaries[p]
            # check if ``val`` is within tolerance to bnds
            if numpy.isclose(val, bnds.lower):
                val = bnds.lower
            elif numpy.isclose(val, bnds.upper):
                val = bnds.upper
            # check if within the boundaries
            if isinstance(val, numpy.ndarray):
                thisisin = ((val >= bnds[0]) & (val <= bnds[1]))
            else:
                thisisin = bnds[0] <= val <= bnds[1]
            if isin is None:
                isin = thisisin
            else:
                isin &= thisisin
        if testpt:
            raise ValueError("unrecognized parameter(s) {}"
                             .format(', '.join(testpt.keys())))
        return isin

    def _intersects(self, x, eigvec):
        """Given a vector (the eigenvector) and an initial point, finds
        where does this line intersects the boundaries
        """
        x = numpy.array([x[p] for p in self.parameters])
        intersections = list()

        for eq in self.hyperplanes:
            norm = eq[:-1]
            dist = -eq[-1]

            eigdot = numpy.dot(norm, eigvec)
            # check if eigenvector parallel to this plane
            if eigdot == 0:
                continue
            intersect = x + eigvec * (dist - numpy.dot(norm, x)) / eigdot
            intersect = {p: intersect[i]
                         for i, p in enumerate(self.parameters)}

            # check if this intersection falls within boundaries
            if intersect in self:
                intersections.append(intersect)

        # check for duplicates
        if len(intersections) == 2:
            return intersections
        else:
            hashes = [hash(frozenset(p.values())) for p in intersections]
            intersections = [intersections[hashes.index(h)]
                             for h in list(set(hashes))]
        # if even after removing duplicates do not have len 2 raise error
        if len(intersections) != 2:
            raise ValueError("Unexpected behaviour. Expected to have always "
                             "two unique intersections.")
        return intersections

    def jump(self, fromx):
        # make sure we're in bounds
        if fromx not in self:
            raise ValueError("Given point is not in bounds; I don't know how "
                             "to jump from there.")
        if self._call_initial_proposal():
            return self.initial_proposal.jump(fromx)

        self._ind = self._pick_jump_eigenvector()
        # rejection sampling to find a point within bounds
        while True:
            self._dx = self.random_generator.normal(
                scale=self.eigvals[self._ind])
            out = {p: fromx[p] + self._dx * self.eigvects[i, self._ind]
                   for i, p in enumerate(self.parameters)}
            if out in self:
                return out

    def logpdf(self, xi, givenx):
        if self._call_initial_proposal():
            return self.initial_proposal.logpdf(xi, givenx)

        # cache the intersects and width for repeated calls with givenx and xi
        new_hash = set((frozenset(xi[p] for p in self.parameters),
                        frozenset(givenx[p] for p in self.parameters)))
        if new_hash == self._old_hash:
            in1, in2 = self._cache['intersects']
            width = self._cache['width']
        else:
            # calculate the intersections and width
            in1, in2 = self._intersects(givenx, self.eigvects[:, self._ind])
            width = numpy.linalg.norm([in2[p] - in1[p]
                                       for p in self.parameters])
            # update the cached data
            self._old_hash = new_hash
            self._cache.update({'intersects': (in1, in2), 'width': width})

        mu = numpy.linalg.norm([givenx[p] - in1[p] for p in self.parameters])
        xi = numpy.linalg.norm([xi[p] - in1[p] for p in self.parameters])

        a = - mu / self.eigvals[self._ind]
        b = (width - mu) / self.eigvals[self._ind]
        return stats.truncnorm.logpdf(xi, a, b, loc=mu,
                                      scale=self.eigvals[self._ind])
