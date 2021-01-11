Proposals
---------

The classical Metropolis-Hastings algorithm uses a normal distribution for
producing jump proposals. This can be quite inefficient for many likelihood
surfaces. To improve convergence, multiple proposal classes are offered that
can be used for different parameters in your model. These are:

.. include:: proposal_list.rst

Proposals with ``Adaptive`` in the name attempt to learn the shape of the
likelihood surface during a burn-in period; see the documentation for each
class for details.

By default, if no proposal class is provided for a parameter the
:py:class:`~epsie.proposals.normal.Normal` proposal is used with a standard
deviation of 1. To use one of the other proposals (or to use a different
standard deviation with the normal proposal), you must first initialize the
proposal class giving the name(s) of the parameter(s) you wish to use it for.
You then provide the list of proposals to use to the sampler's ``proposals``
argument.

For example, say your model consists of two parameters, ``amp`` and ``phase``,
where ``phase`` is cyclic on :math:`[0, 2\pi)` and ``amp`` is in :math:`[0,
1)`. You wish to use the
:py:class:`~epsie.proposals.bounded_normal.BoundedNormal` proposal for ``amp``
and the :py:class:`~epsie.proposals.angular.Angular` proposal for ``phase``.
To do so, you first initialize the appropriate proposal for each parameter:

.. code-block:: python

    from epsie.proposals import (BoundedNormal, Angular)
    amp_prop = BoundedNormal(['amp'], boundaries={'amp': (0, 1)})
    phase_prop = Angular(['phase'])
    proposals = [amp_prop, phase_prop]

You would then initialize the sampler passing the list of proposals to the
sampler's ``proposals`` keyword argument (both the
:py:class:`~epsie.samplers.mhsampler.MetropolisHastingsSampler` and the
:py:class:`~epsie.samplers.ptsampler.ParallelTemperedSampler` use this
argument). For a detailed example, see the
:doc:`angular proposal tutorial <tutorials/test_angular>`.

Creating Custom Proposals
+++++++++++++++++++++++++

You may also define your own proposal classes. The only requirement is that
the proposal class inherits from
:py:class:`~epsie.proposals.base.BaseProposal`. This is an abstract base class
that defines the standard API needed for all proposals. At bare minimum, you
must provide:

 * a ``parameters`` attribute: a list of the parameter names that the
   proposal produces jumps for.
 * a ``_jump`` method: this should take a dictionary (``fromx``) that defines
   the current point in parameter space and returns another dictionary giving
   the new proposed point to jump to.
 * a ``_logpdf`` method: this should take two dictionaries (``xi`` and
   ``givenx``) that define the proposed point and the current position,
   respectively, and returns the log of the pdf of jumping from the current
   position to the new point. This only used for non-symmetric proposals, but
   is good to define regardless.
 * ``symmetric``: a boolean indicating whether the proposal is symmetric or
   not (see
   :py:meth:`~epsie.proposals.base.BaseProposal.symmetric` for details).
 * ``state``: a dictionary of any parameters that need to be set in order to
   produce a deterministic jump; see
   :py:attr:`~epsie.proposals.base.BaseRandom.state` for details.
 * ``set_state``: a method for setting the state when recovering from a
   checkpoint; see :py:meth:`~epsie.proposals.base.BaseRandom.set_state` for
   details.

.. note::
    The :py:class:`~epsie.proposals.base.BaseProposal` has a
    :py:attr:`~epsie.proposals.base.BaseRandom.random_generator` attribute,
    which is inherited from :py:class:`~epsie.proposals.base.BaseRandom`. This
    is a thin wrapper around an instance of :py:class:`numpy.random.Generator`.
    **You must use the**
    :py:attr:`~epsie.proposals.base.BaseRandom.random_generator` **attribute
    for producing random variates for jump proposals, or any other random
    variates you create in the proposal class.** Use of any other psuedo-random
    number generator will cause the sampler to not work correctly in a parallel
    environment.

If you wish to create a proposal that makes use of the chain history --- e.g.,
for creating an adaptive proposal --- an
:py:meth:`~epsie.proposals.base.BaseProposal.update` method can be optionally
implemented. This should expect to take an instance of
:py:class:`~epsie.chain.chain.Chain`, which contains the history of the samples
(up to the last time a :py:meth:`~epsie.chain.chain.Chain.clear` was called),
as well as several other properties, such as the acceptance ratio.  See the
:py:class:`~epsie.chain.chain.Chain` API for more details.
