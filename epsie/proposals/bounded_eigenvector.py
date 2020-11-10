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

from itertools import (product, combinations)

import numpy
from scipy.stats import truncnorm
from scipy.spatial import ConvexHull

from .eigenvector import (Eigenvector, AdaptiveEigenvectorSupport)
from .bounded_normal import Boundaries


class BoundedEigenvector(Eigenvector):
    r"""Uses a  bounded eigenvector jump with a fixed scale.

    This proposal calculates the eigenvectors from the covariance matrix and
    always proposes a jump along a *single* eigenvector from a univariate
    normal distribution. Uses a truncated normal distribution whose scale
    corresponds to the width of the boundaries surface along the specific
    eigenvector.

    This proposal may handle one or more parameters.

    Parameters
    ----------
    parameters : (list of) str
        The names of the parameters to produce proposals for.
    boundaries : dict
        Dictionary mapping parameters to boundaries. Boundaries must be a
        tuple or iterable of length two.
    stability_duration : int
        Number of initial steps done with a initial proposal specified by name
        in ``initial_proposal''. After this eigenvalues and eigenvectors are
        evaluated (and never again) and jumps proposed along those.
    initial_proposal : str (optional)
        Name of the initial proposal that is called before the number of
        proposal seps exceeds ``stability_duration''. By default se to the
        'epsie.proposals.ATAdaptiveProposal'. Supported options
        include: 'bounded_normal', 'at_adaptive_bounded_normal'.
    jump_interval : int, optional
        The jump interval of the proposal, the proposal only gets called every
        jump interval-th time. After ``jump_interval_duration`` number of
        proposal steps elapses the proposal will again be called on every
        chain iteration. By default ``jump_interval`` = 1.
    jump_interval_duration : int, optional
        The number of proposals steps during which values of ``jump_interval``
        other than 1 are used. After this elapses the proposal is called on
        each iteration.
    shuffle_rate : float (optional)
        Probability of shuffling the eigenvector jump probabilities. By
        default 0.33.
    """

    name = 'bounded_eigenvector'
    symmetric = False
    _boundaries = None
    _lowerbnd = None
    _upperbnd = None
    _hyperplanes = None

    def __init__(self, parameters, boundaries, stability_duration,
                 jump_interval=1, jump_interval_duration=None,
                 shuffle_rate=0.33,
                 initial_proposal='at_adaptive_bounded_normal'):
        super(BoundedEigenvector, self).__init__(
            parameters, stability_duration, shuffle_rate=shuffle_rate,
            jump_interval=jump_interval,
            jump_interval_duration=jump_interval_duration)
        if not self.ndim > 1:
            raise ValueError("Dimensionality of {} proposal must be at "
                             "least 2".format(self.name))
        # set the boundaries
        self.boundaries = boundaries
        # set the initial phase settings
        self.set_initial_proposal(initial_proposal, stability_duration,
                                  boundaries)
        # cache the boundaries for repeated calls
        self._cache = {'hash': None}

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
            counter = 0
            # check with duplicates with some tolerance
            while len(intersections) != 2:
                counter += 1
                for i, j in combinations(range(len(intersections)), 2):
                    x0 = [intersections[i][p] for p in parameters]
                    x1 = [intersections[j][p] for p in parameters]
                    # the tolerance here might have to be tuned further
                    if np.allclose(x0, x1, rtol=1e-3):
                        intersections.pop(i)
                        break
                # after some number of iterations raise this value Error
                # if even after removing duplicates do not have len 2
                if counter > 100:
                    if len(intersections) != 2:
                        raise ValueError("Unexpected behaviour. Expected to "
                                         "find two unique intersections. "
                                         "Found intersections are {}"
                                         .format(intersections))
        return intersections

    def _jump(self, fromx):
        # make sure we're in bounds
        if fromx not in self:
            raise ValueError("Given point is not in bounds; I don't know how "
                             "to jump from there.")
        if self._call_initial_proposal:
            return self.initial_proposal.jump(fromx)

        self._ind = self._jump_eigenvector
        # rejection sampling to find a point within bounds
        while True:
            self._dx = self.random_generator.normal(
                scale=self.eigvals[self._ind])
            out = {p: fromx[p] + self._dx * self.eigvects[i, self._ind]
                   for i, p in enumerate(self.parameters)}
            if out in self:
                return out

    def _logpdf(self, xi, givenx):
        if self._call_initial_proposal:
            return self.initial_proposal.logpdf(xi, givenx)
        # cache the intersects and width for repeated calls with givenx and xi
        x0 = [xi[p] for p in self.parameters]
        x1 = [givenx[p] for p in self.parameters]

        if x0 + x1 == self._cache['hash']:
            in1, in2 = self._cache['intersects']
            width = self._cache['width']
        else:
            # calculate the intersections and width
            in1, in2 = self._intersects(givenx, self.eigvects[:, self._ind])
            width = numpy.linalg.norm([in2[p] - in1[p]
                                       for p in self.parameters])
            # update the cached data
            self._cache.update({'hash': x1 + x0, 'intersects': (in1, in2),
                                'width': width})
        mu = numpy.linalg.norm([givenx[p] - in1[p] for p in self.parameters])
        xi = numpy.linalg.norm([xi[p] - in1[p] for p in self.parameters])

        a = - mu / self.eigvals[self._ind]
        b = (width - mu) / self.eigvals[self._ind]
        return truncnorm.logpdf(xi, a, b, loc=mu,
                                scale=self.eigvals[self._ind])


class AdaptiveBoundedEigenvector(AdaptiveEigenvectorSupport,
                                 BoundedEigenvector):
    r"""Uses a  bounded eigenvector jump with a adaptive scales

    See :py:class:`AdaptiveEigenvectorSupport` for details on the adaptation
    algorithm.

    Parameters
    ----------
    parameters: (list of) str
        The names of the parameters.
    boundaries : dict
        Dictionary mapping parameters to boundaries. Boundaries must be a
        tuple or iterable of length two.
    stability_duration : int
        Number of initial steps done with a initial proposal specified by name
        in ``initial_proposal''. After this eigenvalues and eigenvectors are
        evaluated and jumps proposed along those.
    adaptation_duration: int
        The number of steps after which adaptation of the eigenvectors ends.
        This is defined such that while the number of proposal steps :math:`N`
        satisfies :math:`N <= \mathrm{stability_duration}` the
        ``initial_proposal'' is called and while
        :math:`N + \mathrm{stability_duration} < \mathrm{adaptation_duration}`
        the eigenvectors are being adapted. Post-adaptation phase the
        eigenvectors and eigenvalues are kept constant.
    jump_interval : int, optional
        The jump interval of the proposal, the proposal only gets called every
        jump interval-th time. After ``jump_interval_duration`` number of
        proposal steps elapses the proposal will again be called on every
        chain iteration. By default ``jump_interval`` = 1.
    initial_proposal : str (optional)
        Name of the initial proposal that is called before the number of
        proposal seps exceeds ``stability_duration''. By default se to the
        'epsie.proposals.ATAdaptiveProposal'. Supported options
        include: 'bounded_normal', 'at_adaptive_bounded_normal'.
    target_rate: float (optional)
        Target acceptance ratio. By default 0.234
    shuffle_rate : float (optional)
        Probability of shuffling the eigenvector jump probabilities. By
        default 0.33.
    """

    name = 'adaptive_bounded_eigenvector'
    symmetric = False

    def __init__(self, parameters, boundaries, stability_duration,
                 adaptation_duration, jump_interval=1,
                 initial_proposal='at_adaptive_bounded_normal',
                 target_rate=0.234, shuffle_rate=0.33):
        # set the parameters, initial proposal
        super(AdaptiveBoundedEigenvector, self).__init__(
            parameters=parameters, boundaries=boundaries,
            stability_duration=stability_duration, jump_interval=jump_interval,
            jump_interval_duration=adaptation_duration + stability_duration,
            shuffle_rate=shuffle_rate, initial_proposal=initial_proposal)
        # set up the adaptation parameters
        self.setup_adaptation(adaptation_duration, target_rate)
