Quick Start
-----------

Two samplers are provided: the
:py:class:`~epsie.samplers.mhsampler.MetropolisHastingsSampler` and the
:py:class:`~epsie.samplers.ptsampler.ParallelTemperedSampler`.
Parallel tempering is useful for multi-modal distributions and for estimating
Bayesian evidence, but is more computationally expensive.

To use either sampler, you provide a function that evaluates the likelihood and
prior at given points. This function must take keyword arguments as input that
map parameter names to values and return a tuple of the log likelihood and log
prior at that point. The function may also return additional "blob" data which
is a dictionary of additional statistics that you would like to keep track of,
but this is optional.

For example, the following sets up the Metropolis-Hastings sampler with 3
chains to sample a 2D normal distribution, with a prior uniform between [-5, 5)
on each parameter:

.. code-block:: python

    # create the function to evaluate
    from scipy import stats
    def model(x, y):
        """Evaluate model at given point."""
        logp = stats.uniform.logpdf([x, y], loc=-5, scale=10).sum()
        logl = stats.norm.logpdf([x, y]).sum()
        return logl, logp

    # set up the sampler
    from epsie.samplers import MetropolisHastingsSampler
    params = ['x', 'y']
    nchains = 3
    sampler = MetropolisHastingsSampler(params, model, nchains) 

    # set the start positions: we'll just draw from the prior
    sampler.start_position = {
        'x': stats.uniform.rvs(size=nchains, loc=-5, scale=10),
        'y': stats.uniform.rvs(size=nchains, loc=-5, scale=10)
        }

    # run for a few iterations
    sampler.run(4)

    # retrieve positions
    positions = sampler.positions
    # positions is numpy structured array with shape nchains x niterations
    print(positions['x'])
    print(positions['y'])


In this simple example we will have used the default
:py:class:`~epsie.proposals.normal.Normal` proposal class for producing
jumps, and we will have used only a single core.

To learn how to use more advanced features and how to interact with
sampler data, see the notebooks provided in the :doc:`tutorials <tutorials>`.
