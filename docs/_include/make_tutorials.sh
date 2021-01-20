#!/usr/bin/env sh
tutdir='tutorials'
mkdir -p ../${tutdir}

jupyter nbconvert ../../tutorials/*ipynb --to rst --output-dir ../${tutdir}

tutfile="../tutorials.rst"
echo '.. toctree::
    :maxdepth: 2
    :caption: Tutorials:
' > ${tutfile}

for fn in $(ls ../${tutdir}); do
    echo "    ${tutdir}/${fn}" >> ${tutfile}
done
