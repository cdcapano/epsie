# Copyright (C) 2020 Richard Stiskalek, Collin Capano
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
"""Sky-dedicated proposals sampling on the surface of a 2-sphere."""
from __future__ import (absolute_import, division)

from abc import ABCMeta
from six import add_metaclass

import numpy
from scipy import stats

from .base import (BaseProposal, BaseAdaptiveSupport)


class IsotropicSkyProposal(BaseProposal):
    """
    L.
    """

    name = 'isotropic_sky'
    symmetric = True

    def __init__(self, parameters, kappa=100):
        self.parameters = parameters
        self.ndim = len(self.parameters)
        # check dimensions
        if self.ndim != 2:
            raise ValueError("{} proposal is defined only on a 2-sphere.")

        self.kappa = kappa

    @property
    def _norm(self):
        """Calculates the normalisation constant for the von Mises-Fisher
        distribution on a 2-sphere.
        """
        return self.kappa / (4 * numpy.pi * numpy.sinh(self.kappa))

    @property
    def _new_point(self):
        """Draws a random point from the von Mises-Fisher distribution
        centered at the north pole.
        """
        phi, cdf = self.random_generator.random(size=2)
        phi *= 2 * numpy.pi
        # feed the randomly generated point into the cdf inverse
        theta = numpy.log(numpy.exp(self.kappa)
                          - self.kappa * cdf / (2 * numpy.pi * self._norm))
        theta = numpy.arccos(theta / self.kappa)
        return phi, theta

    @staticmethod
    def _spherical2cartesian(phi, theta):
        """Transform from spherical to Cartesian coordinates. Theta is defined
        on [0, pi] and phi on [0, 2*pi].
        """
        stheta = numpy.sin(theta)
        x = stheta * numpy.cos(phi)
        y = stheta * numpy.sin(phi)
        z = numpy.cos(theta)
        return numpy.array([x, y, z])

    @staticmethod
    def _cartesian2spherical(x, y, z):
        """Returns the azimuthat and polar angle for normalised unit vector
        with components ``x``, ``y``, and ``z``.
        """
        phi = numpy.arctan2(y, x)
        if phi < 0:
            phi += 2 * numpy.pi
        theta = numpy.arccos(z)
        return (phi, theta)

    @staticmethod
    def _rotmat(mu):
        """Returns a transformation matrix that transforms the north pole (the
        original z-axis) such that it now coincides with the vector ``mu``.
        """
        beta = numpy.arccos(mu[2])
        gamma = numpy.arccos(mu[0] / numpy.sqrt(mu[0]**2 + mu[1]**2))
        # arccos is from 0 to pi but we want the rotation from 0 to 2pi
        if mu[1] < 0:
            gamma = 2 * numpy.pi - gamma
        # pre-calculates the sine and cos of the two angles
        sbeta, sgamma = numpy.sin(beta), numpy.sin(gamma)
        cbeta, cgamma = numpy.cos(beta), numpy.cos(gamma)
        return numpy.array([[cbeta * cgamma, -sgamma, sbeta * cgamma],
                            [cbeta * sgamma, cgamma, sbeta * sgamma],
                            [-sbeta, 0., cbeta]])

    def _jump(self, fromx):
        # unpack the fromx point and convert to cartesian
        mu = self._spherical2cartesian(*[fromx[p] for p in self.parameters])
        # Draw a point that is centered at the north pole
        xi = self._spherical2cartesian(*self._new_point)
        # find a rotation matrix from the north pole to mu
        R = self._rotmat(mu)
        # rotate xi to the new reference frame and convert to sphericals
        xi = self._cartesian2spherical(*numpy.matmul(R, xi))
        out = {p: xi[i] for i, p in enumerate(self.parameters)}
        return out

    def _logpdf(self, xi, givenx):
        # symmetric in theta so just need to write down the angular distance
        # between the two points and feed it to the logpdf

        # ADD THIS
        pass

    @property
    def state(self):
        state = {'nsteps': self._nsteps,
                 'random_state': self.random_state}
        return state

    def set_state(self, state):
        self.random_state = state['random_state']
        self._nsteps = state['nsteps']
