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

import inspect
import numpy
from scipy.stats import norm

from .base import (BaseProposal, BaseAdaptiveSupport)
from .normal import ATAdaptiveNormal


class Eigenvector(BaseProposal):
    """Uses a eigenvector jump with a fixed scale that is determined by the
    given covariance matrix.

    This proposal may handle one or more parameters.

    Parameters
    ----------
    parameters : (list of) str
        The names of the parameters to produce proposals for.
    cov : array, optional
        The covariance matrix of the parameters. May provide either a single
        float, a 1D array with length ``ndim``, or an ``ndim x ndim`` array,
        where ``ndim`` = the number of parameters given. If a single float or
        a 1D array is given, will use a diagonal covariance matrix (i.e., all
        parameters are independent of each other). Default (None) is to use
        unit variance for all parameters.
    shuffle_rate : float, optional
        Probability of shuffling the eigenvector jump probabilities. By
        default 0.33.
    minfac : float, optional
        If any eigenvalues are zero, they will be replaced with
        ``minfac`` times the smallest non-zero eigenvalue. Setting a non-zero
        value will force the sampler to at least occasionally jump in the
        zero direction(s). Default is 0.
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

    name = 'eigenvector'
    symmetric = True
    _shuffle_rate = None

    def __init__(self, parameters, cov=None, shuffle_rate=0.33, minfac=0.,
                 jump_interval=1, jump_interval_duration=None):
        self.parameters = parameters
        self.ndim = len(self.parameters)
        self.shuffle_rate = shuffle_rate
        self.set_jump_interval(jump_interval, jump_interval_duration)
        # set up needed parameters
        self._cov = None
        self._eigvals = None
        self._eigvects = None
        self._ind = None  # for caching the eigenvector index
        self._dx = None  # for caching the last jump std
        self._zeroeigvals = numpy.array([])
        self._minfac = minfac
        # calculate and store the eigenvectors/values
        self.cov = cov
        # used for picking which direction to hop along
        self._dims = numpy.arange(self.ndim)

    @BaseProposal.bit_generator.setter
    def bit_generator(self, bit_generator):
        """Sets the random bit generator.

        Also sets the bit generator of the ``initial_proposal``.

        See :py:class:`epsie.proposals.base.BaseProposal` for more details.
        """
        # this borrowed from: https://stackoverflow.com/a/31909212
        BaseProposal.bit_generator.fset(self, bit_generator)

    @property
    def cov(self):
        """The covariance matrix used."""
        return self._cov

    def _ensurearray(self, val, default=1):
        """Boiler-plate function for setting covariance.

        This ensures that the given value is an array with the same length as
        the number of parameters. If ``None`` is passed for the value, the
        default value will be used.
        """
        if val is None:
            val = default
        if not isinstance(val, numpy.ndarray):
            val = numpy.array(val)
        # make sure val is atleast 1D array
        if val.ndim < 1 or val.size == 1:
            val = numpy.eye(len(self.parameters)) * val.item()
        # make sure the dimensionality makes sense
        if val.shape != (self.ndim, self.ndim):
            raise ValueError("must provide a value for every parameter")
        return val

    @cov.setter
    def cov(self, cov):
        """Sets the covariance matrix, and calculates eigenvectors/values from
        it.

        Raises a ``ValueError`` if the dimensionality of the given array
        isn't ndim x ndim, or if the matrix is not symmetric.

        If ``cov`` is None, will just use the identity matrix.
        """
        # make sure we have an array, filling in default as necessary
        cov = self._ensurearray(cov)
        if not numpy.isclose(cov, cov.T).all():
            raise ValueError("must provide a symmetric covariance matrix")
        self._cov = cov
        # calculate and store the eigenvectors/values
        self.eigvals, self.eigvects = numpy.linalg.eigh(self._cov)

    @property
    def eigvals(self):
        """Returns the eigenvalues"""
        return self._eigvals

    @eigvals.setter
    def eigvals(self, eigvals):
        if eigvals.shape != (self.ndim,):
            raise ValueError("Invalid eigenvalue shape")
        # check that we have no negatives; if we do, and the value is
        # sufficiently close to 0, set it to 0
        negvals = eigvals < 0
        if negvals.any():
            vals = eigvals[negvals]
            relsize = vals / eigvals.max()
            if (abs(relsize) < 1e-12).all():
                # replace with 0
                eigvals[negvals] = 0.
            else:
                # one or more values are < 0 beyond the tolerance; raise error
                raise ValueError("one or more of the given eigenvalues ({}) "
                                 "are negative".format(eigvals))
        self._eigvals = eigvals
        # store and replace any zeroes
        zerovals = eigvals == 0.
        self._zeroeigvals = zerovals
        self._replacezeroeig()

    def _replacezeroeig(self):
        if self._zeroeigvals.any():
            # replace with the min
            minnz = self._eigvals[~self._zeroeigvals].min()
            self._eigvals[self._zeroeigvals] = self.minfac * minnz 

    @property
    def minfac(self):
        """Factor of the smallest non-zero eigenvalue to use for zero values.
        """
        return self._minfac

    @minfac.setter
    def minfac(self, minfac):
        self._minfac = minfac
        # update the zeroeigen values
        self._replacezeroeig()

    @property
    def eigvects(self):
        """Returns the eigenvectors"""
        return self._eigvects

    @eigvects.setter
    def eigvects(self, eigvects):
        if eigvects.shape != (self.ndim, self.ndim):
            raise ValueError("Invalid eigenvector shape.")
        self._eigvects = eigvects

    @property
    def shuffle_rate(self):
        return self._shuffle_rate

    @shuffle_rate.setter
    def shuffle_rate(self, rate):
        if not 0.0 <= rate <= 1.0:
            raise ValueError("Shuffle rate  must be in range [0, 1].")
        self._shuffle_rate = rate

    @property
    def state(self):
        return {'nsteps': self._nsteps,
                'random_state': self.random_state,
                'cov': self._cov,
                'ind': self._ind}

    def set_state(self, state):
        self.random_state = state['random_state']
        self._nsteps = state['nsteps']
        self.cov = state['cov']
        self._ind = state['ind']

    @property
    def _jump_eigenvector(self):
        """Picks along which eigenvector to jump."""
        # replace any zeros with the minimum
        probs = self.eigvals / self.eigvals.sum()
        dims = self._dims
        # with shuffle_rate probability randomly shuffle the probabilities
        if self.random_generator.uniform() < self.shuffle_rate:
            # make sure we don't shuffle any 0 probabilities
            isz = probs == 0.
            if isz.any():
                mask = ~isz
                probs = probs[mask]
                dims = dims[mask]
            self.random_generator.shuffle(probs)
        return self.random_generator.choice(dims, p=probs)

    def _jump(self, fromx):
        self._ind = self._jump_eigenvector
        # scale of the 1D jump
        self._dx = self.random_generator.normal(
            scale=self.eigvals[self._ind]**0.5)
        return {p: fromx[p] + self._dx * self.eigvects[i, self._ind]
                for i, p in enumerate(self.parameters)}

    def _logpdf(self, xi, givenx):
        return norm.logpdf(self._dx, loc=0, scale=self.eigvals[self._ind])

    def _update(self, chain):
        pass


class AdaptiveEigenvectorSupport(BaseAdaptiveSupport):
    r"""Utility class for adding AdaptiveEigenvector proposal support.

    The adaptation algorithm is based on Algorithm 8 in [1].

    See [1] for details.

    Notes
    -----
    For the vanishing decay we use

    .. math::
        \gamma_{g+1} = \left(g - g_{0}\right)^{-0.6} - C,

    where :math:`g_{0}` is the iteration at which adaptation starts,
    by default :math:`g_{0}=1` and :math:`C` is a positive constant
    ensuring that when the adaptation phase ends the vanishing decay tends to
    zero. By default assumes that the adaptation phase never ends (only
    decays with time)

    References
    ----------
    [1] Andrieu, Christophe & Thoms, Johannes. (2008).
    A tutorial on adaptive MCMC. Statistics and Computing.
    18. 10.1007/s11222-008-9110-y.
    """

    def setup_adaptation(self, adaptation_duration, start_step=1,
                         target_rate=0.234):
        r"""Sets up the adaptation parameters.

        Parameters
        ----------
        adaptation_duration : int
            The number of adaptation steps.
        start_step : int, optional
            The proposal step when the adaptation phase begins.
        target_rate : float, optional
            Target acceptance rate. By default 0.234
        """
        self.target_rate = target_rate
        self.adaptation_duration = adaptation_duration
        self._decay_const = adaptation_duration**(-0.6)
        self.start_step = start_step
        self._log_lambda = 0.0
        # initialize mu to be zero
        self._mu = numpy.zeros(self.ndim)
        self._unique_steps = 1
        self._lastx = None
        self._lastind = None

    def recursive_covariance(self, chain):
        """Recursively updates the covariance given the latest observation.
        Weights all sampled points uniformly.
        """
        x = numpy.array([chain.current_position[p]
                         for p in self.parameters])
        # only update if we have a new point and the last ind was different
        if (x != self._lastx).any() and self._lastind != self._ind:
            self._unique_steps += 1
            N = self._unique_steps
            dx = (x - self._mu).reshape(-1, 1)
            self._cov = (N - 1) / N \
                * (self._cov + N / (N**2 - 1) * numpy.matmul(dx, dx.T))
            self._mu = (N * self._mu + x) / (N + 1)
            self._lastx = x
            self._lastind = self._ind
            # update the minfac
            self.minfac = (1./N)**0.5

    def _update(self, chain):
        """Updates the adaptation based on whether the last jump was accepted.

        This prepares the proposal for the next jump.
        """
        dk = self.nsteps - self.start_step + 1
        if 1 < dk < self.adaptation_duration:
            # recursively update the covariance
            self.recursive_covariance(chain)
            # update eigenvalues and eigenvectors
            self.eigvals, self.eigvects = numpy.linalg.eigh(self._cov)
            # update the scaling factor
            dk = dk**(-0.6) - self._decay_const
            # Update of the global scaling
            ar = chain.acceptance['acceptance_ratio'][-1]
            self._log_lambda += dk * (ar - self.target_rate)
            # Rescale the eigenvalues
            self.eigvals *= numpy.exp(self._log_lambda)

    @property
    def state(self):
        return {'nsteps': self._nsteps,
                'random_state': self.random_state,
                'mu': self._mu,
                'cov': self._cov,
                'ind': self._ind,
                'unique_steps': self._unique_steps,
                'log_lambda': self._log_lambda}

    def set_state(self, state):
        self._nsteps = state['nsteps']
        self.random_state = state['random_state']
        self._mu = state['mu']
        self._cov = state['cov']
        self._log_lambda = state['log_lambda']
        self._unique_steps = state['unique_steps']
        if self._cov is not None:
            self.eigvals, self.eigvects = numpy.linalg.eigh(self._cov)
            self.eigvals *= numpy.exp(self._log_lambda)
        self._ind = state['ind']


class AdaptiveEigenvector(AdaptiveEigenvectorSupport, Eigenvector):
    r""" Uses jumps along eigenvectors with adaptive scales.

    See :py:class:`AdaptiveEigenvectorSupport` for details on the adaptation
    algorithm.

    Parameters
    ----------
    parameters: (list of) str
        The names of the parameters.
    adaptation_duration: int
        The number of steps after which adaptation of the eigenvectors ends.
        Post-adaptation phase the eigenvectors and eigenvalues are kept
        constant.
    cov0 : array, optional
        The initial covariance matrix of the parameters used to calculate the
        initial eigenvectors. May provide either a single float, a 1D array
        with length ``ndim``, or an ``ndim x ndim`` array,
        where ``ndim`` = the number of parameters given. If a single float or
        a 1D array is given, will use a diagonal covariance matrix (i.e., all
        parameters are independent of each other). Default (None) is to use
        unit variance for all parameters.
    target_rate: float (optional)
        Target acceptance ratio. By default 0.234
    shuffle_rate : float, optional
        Probability of shuffling the eigenvector jump probabilities. By
        default 0.33.
    start_step: int, optional
        The proposal step index when adaptation phase begins.
    jump_interval : int, optional
        The jump interval of the proposal, the proposal only gets called every
        jump interval-th time. After ``adaptation_duration`` number of
        proposal steps elapses the proposal will again be called on every
        chain iteration. By default ``jump_interval`` = 1.
    """
    name = 'adaptive_eigenvector'
    symmetric = True

    def __init__(self, parameters, adaptation_duration, cov0=None,
                 target_rate=0.234, shuffle_rate=0.33, minfac=0., start_step=1,
                 jump_interval=1):
        # set the parameters, initialize the covariance matrix
        super().__init__(
            parameters=parameters, cov=cov0, shuffle_rate=shuffle_rate,
            minfac=minfac,
            jump_interval=jump_interval,
            jump_interval_duration=adaptation_duration)
        # set up the adaptation parameters
        self.setup_adaptation(adaptation_duration, start_step, target_rate)


class ATAdaptiveEigenvector(BaseProposal):
    """Uses the :py:class:`~normal.ATAdativeNormal` with the
    :py:class:`Eigenvector` proposal.
    """
    name = "at_adaptive_eigenvector"
    symmetric = True
    _switch_ratio = None

    def __init__(self, parameters, adaptation_duration, switch_ratio,
                 jump_interval=1, jump_interval_duration=None, **kwargs):
        self.parameters = parameters
        self.adaptation_duration = adaptation_duration
        self.switch_ratio = switch_ratio
        self.set_jump_interval(jump_interval, jump_interval_duration)
        # figure out which kwargs to pass to the ATAdaptiveNormal proposal
        # and which to pass to the Eigenvector
        sig = inspect.signature(ATAdaptiveNormal)
        possible_akwargs = set([arg for arg, spec in sig.parameters.items()
                                if spec.default is not spec.empty])
        sig = inspect.signature(Eigenvector)
        possible_ekwargs = set([arg for arg, spec in sig.parameters.items()
                                if spec.default is not spec.empty])
        akwargs = {arg: kwargs[arg] for arg in possible_akwargs
                   if arg in kwargs}
        ekwargs = {arg: kwargs[arg] for arg in possible_ekwargs
                   if arg in kwargs}
        # make sure there's no leftover
        remaining = set(kwargs.keys()) - possible_akwargs - possible_ekwargs
        if remaining:
            raise ValueError("unrecognized argument(s) {}"
                              .format(', '.join(remaining)))
        # depending on the switch ratio, the ATAdaptiveNormal may be called
        # fewer times, so we'll modify it's adaptaiton duration accordingly
        adaptdur = int(adaptation_duration * self.switch_ratio[0]
                       / sum(self.switch_ratio))
        self._adaptivep = ATAdaptiveNormal(parameters, adaptdur,
                                           **akwargs)
        self._eigenvecp = Eigenvector(parameters, **ekwargs)
        # ensure both proposals are using the same bit generator as this; the
        # following will just create a new bit generator using a random seed;
        # this may be changed later if the bit generator is set
        self._adaptivep._bit_generator = self._eigenvecp._bit_generator = \
            self.bit_generator
        # eigenactive controls whether the adaptive or the eigenvector proposal
        # is called. We always start out with the adaptive
        self._eigenactive = False
        # to keep track of how many jumps the active proposals has taken since
        # the last switch
        self._active_steps = 0

    @BaseProposal.bit_generator.setter
    def bit_generator(self, bit_generator):
        BaseProposal.bit_generator.fset(self, bit_generator)
        # ensure both sub-proposals use the same generator
        self._adaptivep._bit_generator = self._eigenvecp._bit_generator = \
            self._bit_generator

    @property
    def switch_ratio(self):
        """The number of steps the adaptive proposal takes relative to the
        eigenvector proposal duration the adaptation phase.
        """
        return (self._adaptive_interval, self._eigenvec_interval)

    @switch_ratio.setter
    def switch_ratio(self, switch_ratio):
        if isinstance(switch_ratio, tuple):
            a, e = map(int, switch_ratio)
        elif switch_ratio == 0:
            a = 1
            e = 0
        else:
            a = int(switch_ratio)
            e  = 1
        self._adaptive_interval = a
        self._eigenvec_interval = e

    @property
    def adaptive_proposal(self):
        """The proposal used for adapting the covariance."""
        return self._adaptivep

    @property
    def eigenvector_proposal(self):
        """The proposal used for doing eigenvector jumps."""
        return self._eigenvecp

    @property
    def active_proposal(self):
        """The currently active proposal."""
        if self._eigenactive:
            return self._eigenvecp
        return self._adaptivep

    @property
    def inactive_proposal(self):
        """The currently inactive proposal."""
        if self._eigenactive:
            return self._adaptivep
        return self._eigenvecp

    @property
    def active_interval(self):
        """The number of steps the currently active proposal is run for."""
        if self._eigenactive:
            return self._eigenvec_interval
        return self._adaptive_interval

    @property
    def inactive_interval(self):
        """The number of steps the currently inactive proposal is run for."""
        if self._eigenactive:
            return self._adaptive_interval
        return self._eigenvec_interval

    @property
    def eigenactive(self):
        """Whether or not the eigenvector proposal is currently active."""
        return self._eigenactive

    @eigenactive.setter
    def eigenactive(self, eigenactive):
        """Toggles the eigenactive setting.

        If eigenactive was previously False, and this sets it to True, the
        eigenvectors and values used by the eigenvector proposal will be
        updated to use the current covariance of the adaptive proposal.
        """
        if not self._eigenactive and eigenactive:
            # we're turning on th eigenvector proposal, so reset its cov,
            # and calculate new eigenvectors and values
            self._eigenvecp.cov = self._adaptivep.cov
        self._eigenactive = eigenactive

    def _update_active(self):
        """Sets the active proposal based on the number of steps that have
        been taken and the switch ratio.
        """
        # if we are beyond the adaptation duration, just use the eigenvector
        if self.nsteps > self.adaptation_duration:
            self.eigenactive = True
        elif self._active_steps >= self.active_interval and \
                    self.inactive_interval != 0:
            # switch and reset the active counter
            self.eigenactive = not self.eigenactive
            self._active_steps = 0
        return

    def _update(self, chain):
        # call the active proposal's update
        self.active_proposal.update(chain)
        # update the active counter
        self._active_steps += 1
        # switch
        self._update_active()

    def _jump(self, fromx):
        return self.active_proposal._jump(fromx)

    def _logpdf(self, xi, givenx):
        return self.active_proposal._logpdf(xi, givenx)

    @property
    def state(self):
        state = {'random_state': self.random_state,
                 'switch_ratio': self.switch_ratio,
                 'eigenactive': self._eigenactive,
                 'active_steps': self._active_steps,
                 'nsteps': self.nsteps,
                }
        # add the state of the adaptive and eigenvector proposals
        state['adaptive_proposal'] = self._adaptivep.state
        state['eigenvector_proposal'] = self._eigenvecp.state
        return state

    def set_state(state):
        self.random_state = state['random_state']
        self.switch_ratio = switch_ratio
        self._eigenactive = state['eigenactive']
        self._active_steps = state['active_steps']
        self.nsteps = state['nsteps']
        self._adaptivep.set_state(state['adaptive_proposal'])
        self._eigenvecp.set_state(state['eigenvector_proposal'])
