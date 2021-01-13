Parallelization
---------------

EPSIE samplers can be parallelized over multiple cores and even (via MPI)
multiple CPUs. In order to do so, you create a ``pool`` object and pass that
to the sampler on initialization. The pool object must have a ``map`` method
that has the same API as the standard Python :py:func:`map` function.

Overview
++++++++

Parallelization occurs over multiple chains. After a sampler has been setup
with some number of Markov chains, you evolve the chains for a given number of
iterations by calling the sampler's
:py:meth:`~epsie.samplers.base.BaseSampler.run` method. The run method uses the
``pool`` that was provided to the sampler to split up the chains over child
processes. Each child process gets a subset of chains, and is told to iterate
the chains for the desired number of iterations by the parent process.  When
the children processes have finished iterating their chains, they send the
results (positions, proposal states, etc.) back to the parent process. How many
chains a child process gets is determined by the pool being used (not by
EPSIE), but, roughly, is the number of chains divided by the number of
processes. 

No other communication occurs between the parent process and the child
processes while chains are being iterated. This differs from ensemble samplers,
which typically pass information between children and parent processes on each
iteration of the sampler.

Children processes evolve each chain by the number of iterations passed to the
run command before moving on to the next chain. For example, if a child process
is told to evolve 4 chains for 100 iterations, it will
:py:func:`~epsie.chain.chain.Chain.step` the first chain 100 times, then move
to the second chain, etc. For this reason, samplers should not be interrupted
while the :py:meth:`~epsie.samplers.base.BaseSampler.run` method is being
executed.  Instead, it is best to run samplers for successive short periods of
time, with results checkpointed in between.

For the :py:class:`~epsie.samplers.ptsampler.ParallelTemperedSampler`, all
temperatures of a given chain are grouped together via the
:py:class:`~epsie.chain.ptchain.ParallelTemperedChain`. **Temperatures are not
split up over processes.** Instead, all temperatures for a given chain are
evolved together on each iteration.  For example, if a child process receives
N chains, each with K temperatures, it will
:py:func:`~epsie.chain.chain.Chain.step` each temperature of the first chain,
perform any temperature swaps, then repeat for the same chain by the requested
number of iterations. It will then move on to the next chain.

For a detailed example see the :doc:`Example of creating and running a
Metropolis-Hastings Sampler <tutorials/02-mhsampler>` and the :doc:`Example of
creating and running a Parallel Tempered Sampler <tutorials/03-ptsampler>`
tutorials.

Using Python's multiprocessing
++++++++++++++++++++++++++++++

If you are using shared memory, the easiest way to parallelize is to use
Python's :py:mod:`multiprocessing` module. For example, if you wish to use
``N`` cores:

.. code-block:: python

    import multiprocessing
    pool = multiprocessing.Pool(N)

You then pass the ``pool`` object to the sampler's ``pool`` keyword argument
when initializing it:

.. code-block:: python

    samlper = MetropolisHastingsSampler(params, model, nchains, pool=pool)


Using MPI
+++++++++

To use parallelize over multiple CPUs (not shared memory) you will need to use
some implementation of
`MPI <https://en.wikipedia.org/wiki/Message_Passing_Interface>`_, such as
`Open MPI <https://www.open-mpi.org/>`_. To use within Python, we recommend
using a combination of `mpi4py <https://mpi4py.readthedocs.io/en/stable/>`_ and
`schwimmbad <https://schwimmbad.readthedocs.io/en/latest/>`_, both of which can
be installed via ``pip``. For example, to use ``N`` processes, you would
create the pool by doing:

.. code-block:: python

    import schwimmbad
    pool = schwimmbad.choose_pool(mpi=True, processes=N)

This ``pool`` object can be passed to the sampler. You would then run your
Python script using your installation of MPI, e.g., `mpirun
<https://www.open-mpi.org/doc/v4.0/man1/mpirun.1.php>`_.

For more information, see the documentation for these packages. A more
feature-rich example of setting up an MPI pool can be found in the PyCBC
suite's `pool module
<http://pycbc.org/pycbc/latest/html/pycbc.html#module-pycbc.pool>`_.
