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
"""Proposals for sampling on the surface of a 2-sphere."""
from __future__ import (absolute_import, division)

from abc import ABCMeta
from six import add_metaclass

import numpy

from .base import (BaseProposal, BaseAdaptiveSupport)


class IsotropicSolidAngle(BaseProposal):
    r"""Uses the von Mises-Fisher distribution with a fixed concentration
    parameter.

    The von Mises-Fisher distribution is an isotropic distribution
    on the surface of a 2-sphere and is analogoues to the isotropic normal
    distribution on a plane (where the covariance matrix is proportional to
    unity).

    Parameters
    ----------
    azimuthal_parameter : str
        The name of the azimuthal parameter.
    polar_parameter : str
        The name of the polar parameter.
    kappa : float, optional
        The distribution concentration parameter. The mean angular distance
        of a jump in degrees for several values of ``kappa`` is:

            ``kappa`` = 1   : 69 degrees
            ``kappa`` = 10  : 23 degrees
            ``kappa`` = 100 : 7 degrees

    radec : bool, optional
        Defines the polar angle convention. By default False, i.e. the polar
        angle range is [0, np.pi]. If ``radec`` = True, then the polar angle
        range is [-np.pi/2, np.pi/2].
    degs : bool, optional
        Whether the input parameters are in degrees.
        By default False (radians).
    jump_interval : int, optional
        The jump interval of the proposal, the proposal only gets called every
        jump interval-th time. After ``jump_interval_duration`` number of
        proposal steps elapses the proposal will again be called on every
        chain iteration. By default ``jump_interval`` = 1.
    jump_interval_duration : int, optional
        The number of proposals steps during which values of ``jump_interval``
        other than 1 are used. After this elapses the proposal is called on
        each iteration.
    """

    name = 'isotropic_solid_angle'
    symmetric = True
    _kappa = None
    _norm = None
    _radec = None
    _degs = None

    def __init__(self, azimuthal_parameter, polar_parameter, kappa=10,
                 radec=False, degs=False, jump_interval=1,
                 jump_interval_duration=None):
        self.parameters = [azimuthal_parameter, polar_parameter]
        self.isdegs = degs
        self.isradec = radec
        # store the concentration parameter
        self.kappa = kappa
        # calculate the normalisation constant
        self.norm = self._normalisation(kappa)
        self.set_jump_interval(jump_interval, jump_interval_duration)

    @property
    def kappa(self):
        """Returns the concentration parameter of the von Mises-Fisher
        distribution.
        """
        return self._kappa

    @kappa.setter
    def kappa(self, kappa):
        """Sets the concentration parameter."""
        if not kappa > 0:
            raise ValueError("``kappa`` must be > 0.")
        self._kappa = kappa

    @property
    def isradec(self):
        """Whether the polar angle assumes the right-ascension declination
        convention and runs in range [-np.pi/2, np.pi/2].
        """
        return self._radec

    @isradec.setter
    def isradec(self, radec):
        """Sets ``isradec``."""
        if not isinstance(radec, bool):
            raise ValueError("``radec`` must be a boolean.")
        self._radec = radec

    @property
    def isdegs(self):
        """Whether input and ouput values are in degrees."""
        return self._degs

    @isdegs.setter
    def isdegs(self, degs):
        """Sets ``isdegs``."""
        if not isinstance(degs, bool):
            raise ValueError("``degs`` must be a boolean.")
        self._degs = degs

    @property
    def norm(self):
        """Returns the normalisation constant for the von Mises-Fisher
        distribution.
        """
        return self._norm

    @norm.setter
    def norm(self, norm):
        """Sets the normalisation constants and checks it's positive."""
        if not norm > 0:
            raise ValueError("``normalisation must be > 0.")
        self._norm = norm

    @staticmethod
    def _normalisation(kappa):
        """Calculates the normalisation constant for the von Mises-Fisher
        distribution on a 2-sphere.
        """
        return kappa / (4 * numpy.pi * numpy.sinh(kappa))

    @property
    def _new_point(self):
        """Draws a random point from the von Mises-Fisher distribution
        centered at the north pole.
        """
        # once python2 dropped keep just .random
        try:
            phi, cdf = self.random_generator.random(size=2)
        except AttributeError:
            phi, cdf = self.random_generator.random_sample(size=2)
        phi *= 2 * numpy.pi
        # feed the randomly generated point into the cdf inverse
        theta = numpy.log(numpy.exp(self.kappa)
                          - self.kappa * cdf / (2 * numpy.pi * self.norm))
        theta = numpy.arccos(theta / self.kappa)
        return phi, theta

    def _spherical2cartesian(self, phi, theta, convert=False):
        """Transform from spherical to Cartesian coordinates. Theta is defined
        on [0, pi] and phi on [0, 2*pi].
        """
        # convert from radec/degrees
        if self.isradec and convert:
            if self.isdegs:
                theta += 90.
            else:
                theta += numpy.pi / 2
        if self.isdegs and convert:
            phi *= numpy.pi / 180.
            theta *= numpy.pi / 180.
        stheta = numpy.sin(theta)
        x = stheta * numpy.cos(phi)
        y = stheta * numpy.sin(phi)
        z = numpy.cos(theta)
        return numpy.array([x, y, z])

    def _cartesian2spherical(self, x, y, z, convert=False):
        """Returns the azimuthal and polar angle for normalised unit vector
        with components ``x``, ``y``, and ``z``.
        """
        phi = numpy.arctan2(y, x)
        if phi < 0:
            phi += 2 * numpy.pi
        theta = numpy.arccos(z)
        # convert back to radec/degrees
        if self.isdegs and convert:
            theta *= 180. / numpy.pi
            phi *= 180. / numpy.pi
        if self.isradec and convert:
            if self.isdegs:
                theta -= 90.
            else:
                theta -= numpy.pi / 2
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
        mu = self._spherical2cartesian(*[fromx[p] for p in self.parameters],
                                       convert=True)
        # Draw a point that is centered at the north pole
        xi = self._spherical2cartesian(*self._new_point)
        # find a rotation matrix from the north pole to mu
        R = self._rotmat(mu)
        # rotate xi to the new reference frame and convert to sphericals
        xi = self._cartesian2spherical(*numpy.matmul(R, xi), convert=True)
        out = {p: xi[i] for i, p in enumerate(self.parameters)}
        return out

    def _logpdf(self, xi, givenx):
        mu = self._spherical2cartesian(*[givenx[p] for p in self.parameters],
                                       convert=True)
        x = self._spherical2cartesian(*[xi[p] for p in self.parameters],
                                      convert=True)
        return numpy.log(self.norm) + self.kappa * numpy.dot(mu, x)

    @property
    def state(self):
        state = {'nsteps': self._nsteps,
                 'random_state': self.random_state}
        return state

    def set_state(self, state):
        self.random_state = state['random_state']
        self._nsteps = state['nsteps']


@add_metaclass(ABCMeta)
class AdaptiveIsotropicSolidAngleSupport(BaseAdaptiveSupport):
    r"""A utility class for adding adaptation support for the
    ``IsotropicSolidAngle`` proposal.

    The adaptation adapts the concentration parameter to achieve a specific
    acceptance ration.

    Notes
    ----------
    For the vanishing decay we use

    .. math::
        \gamma_{g+1} = \left(g - g_{0}\right)^{-0.6} - C,

    where :math: `g_{0}` is the iteration at which adaptation starts,
    by default :math: `g_{0}=1` and :math: `C` is a positive constant
    ensuring that when the adaptation phase ends the vanishing decay tends to
    zero.
    """

    def setup_adaptation(self, adaptation_duration, start_step, target_rate):
        """Sets up the adaptation parameters.

        Parameters
        ----------
        adaptation_duration: int
            The number of proposal steps over which to apply the adaptation. No
            more adaptation will be done once a proposal exceeds this value.
        start_step : int, optional
            The proposal step when the adaptation phase begins.
        target_rate: float, optional
            Target acceptance ratio. By default 0.234 and 0.48 for
            componentwise scaling.
        """
        self.adaptation_duration = adaptation_duration
        self.start_step = start_step
        self.target_rate = target_rate
        self._decay_const = (adaptation_duration)**(-0.6)
        # this one will be getting updated
        self._log_kappa = numpy.log(self.kappa)

    def _update(self, chain):
        dk = self.nsteps - self.start_step + 1
        if 1 < dk < self.adaptation_duration:
            # our decay is 1/dk**(0.6)
            dk = dk**(-0.6) - self._decay_const
            ar = chain.acceptance['acceptance_ratio'][-1]
            # update log of the concetration parameter
            self._log_kappa += dk * (self.target_rate - ar)
            # update the concentration parameter and the norm constant
            self.kappa = numpy.exp(self._log_kappa)
            self.norm = self._normalisation(self.kappa)

    @property
    def state(self):
        state = {'nsteps': self._nsteps,
                 'random_state': self.random_state,
                 'kappa': self.kappa}
        return state

    def set_state(self, state):
        self.random_state = state['random_state']
        self._nsteps = state['nsteps']
        self.kappa = state['kappa']
        # store the log and the normalisation constant
        self._log_kappa = numpy.log(self.kappa)
        self.norm = self._normalisation(self.kappa)


class AdaptiveIsotropicSolidAngle(AdaptiveIsotropicSolidAngleSupport,
                                  IsotropicSolidAngle):
    r"""An adaptive isotropic solid angle proposal based on the
    von Mises-Fisher distribution.

    See :py:class:`AdaptiveIsotropicSolidAngleSupport` and
    :py:class:`IsotropicSolidAngle` for more details on the adaptation
    algorithm and the proposal distribution.

    Parameters
    ----------
    azimuthal_parameter : str
        The name of the azimuthal parameter.
    polar_parameter : str
        The name of the polar parameter.
    adaptation_duration : int
        The number of proposal steps over which to apply the adaptation. No
        more adaptation will be done once a proposal exceeds this value.
    start_step: int, optional
        The proposal step index when adaptation phase begins.
    target_rate: float, optional
        Target acceptance ratio. By default 0.234 and 0.48 for componentwise
        scaling.
    radec : bool, optional
        Defines the polar angle convention. By default False, i.e. the polar
        angle range is [0, np.pi]. If ``radec`` = True, then the polar angle
        range is [-np.pi/2, np.pi/2].
    degs : bool, optional
        Whether the input parameters are in degrees.
        By default False (radians).
    jump_interval : int, optional
        The jump interval of the proposal, the proposal only gets called every
        jump interval-th time. After ``jump_interval_duration`` number of
        proposal steps elapses the proposal will again be called on every
        chain iteration. By default ``jump_interval`` = 1.
    """
    name = 'adaptive_isotropic_solid_angle'
    symmetric = True

    def __init__(self, azimuthal_parameter, polar_parameter,
                 adaptation_duration, start_step=1, target_rate=0.234,
                 radec=False, degs=False, jump_interval=1):
        # setup the main isotropic solid angle proposal
        super(AdaptiveIsotropicSolidAngle, self).__init__(
            azimuthal_parameter=azimuthal_parameter,
            polar_parameter=polar_parameter, kappa=5., radec=radec, degs=degs,
            jump_interval=jump_interval,
            jump_interval_duration=adaptation_duration)
        # setup the adaptation
        self.setup_adaptation(adaptation_duration, start_step, target_rate)
