EPSIE
=====

.. image:: https://github.com/cdcapano/epsie/workflows/build/badge.svg?branch=master
    :target: https://github.com/cdcapano/epsie/actions?query=workflow%3Abuild+branch%3Amaster
.. image:: https://coveralls.io/repos/github/cdcapano/epsie/badge.svg
    :target: https://coveralls.io/github/cdcapano/epsie

.. docs-start-marker-do-not-remove

EPSIE is a parallelized Markov chain Monte Carlo (MCMC) sampler for Bayesian
inference. It is meant for problems with complicated likelihood surfaces,
including multimodal distributions.  It has support both for parallel tempering
and nested transdimensional problems. It was originally developed for
gravitational-wave parameter estimation, but can be used for any Bayesian
inference problem requring a stochastic sampler.

EPSIE is in many ways similar to `emcee
<https://emcee.readthedocs.io/en/stable/>`_ and other bring your own
likelihood Python-based samplers. The primary difference from emcee is EPSIE
is not an ensemble sampler; i.e., the Markov chains used by EPSIE do not
attempt to share information between each other. Instead, to speed convergence,
multiple jump proposal classes are offered that can be customized to the
problem at hand.  These include adaptive proposals that attempt to learn the
shape of the distribution during a burn-in period. The user can also easily
create their own jump proposals.

.. docs-end-marker-do-not-remove

For more information, see the documentation at:
https://cdcapano.github.io/epsie
