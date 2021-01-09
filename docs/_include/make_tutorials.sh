#!/usr/bin/env sh
tutdir='tutorials'
mkdir -p ../${tutdir}

jupyter nbconvert ../../examples/*ipynb --to rst --output-dir ../${tutdir}

tutfile="../tutorials.rst"
echo 'Tutorials
---------

The following notebooks illustrate features of EPSIE and how to use them:

.. toctree::
    :maxdepth: 2
    :caption: Tutorials:
' > ${tutfile}

for fn in $(ls ../${tutdir}); do
    echo "    ${tutdir}/${fn}" >> ${tutfile}
done
