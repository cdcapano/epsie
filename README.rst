EPSIE
=====

.. image:: https://github.com/cdcapano/epsie/workflows/build/badge.svg?branch=master
    :target: https://github.com/cdcapano/epsie/actions?query=workflow%3Abuild+branch%3Amaster
.. image:: https://coveralls.io/repos/github/cdcapano/epsie/badge.svg
    :target: https://coveralls.io/github/cdcapano/epsie

EPSIE is a parallelized Markov chain Monte Carlo (MCMC) sampler for Bayesian
inference. It is meant for problems with complicated likelihood surfaces,
including multimodal distributions.  It has support both for parallel tempering
and nested transdimensional problems. It was originally developed to tackle
problems in gravitational-wave astronomy, but can be used for any general
purpose problem.

EPSIE is in many ways similar to `emcee
<https://emcee.readthedocs.io/en/stable/>`_ and other "bring your own
likelihood" Python-based samplers. The primary difference from emcee is EPSIE
is not an ensemble sampler; i.e., the Markov chains used by EPSIE do not
attempt to share information between each other. Instead, to speed convergence,
multiple jump proposal classes are offered that can be customized to the
problem at hand.  These include adaptive proposals that attempt to learn the
shape of the distribution during a burn-in period. The user can also easily
create their own jump proposals.


Installation
------------

The easiest way to install EPSIE is via pip:

.. code-block:: bash

   pip install epsie

If you wish to install from the latest source code instead, you can clone a
copy of the repository from GitHub, at https://github.com/cdcapano/epsie.
