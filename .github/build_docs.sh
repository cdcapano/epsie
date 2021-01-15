#!/usr/bin/env sh

#
# Builds documentation and commits it to the gh-pages branch.
#
# Usage:
#
#   ./build_docs.sh TYPE
#
# where TYPE is either "latest" or "versioned"
#
set -e

# get the type of docs we're building
TYPE=$1

# get cache of the currently documented versions on gh-pages
pushd ..
echo "Getting list of perviously documented versions"
git ls-tree --name-only gh-pages | egrep '^latest|^[0-9]+' > docs/.docversions

# clean and build the docs
echo "Building docs"
make -C docs clean
make -C docs ${TYPE}
# get the name of the build directory
docdir=$(ls docs/_build | egrep '^latest|^[0-9]+') 
echo "Docs to commit: ${docdir}"

# commit to gh-pages
echo "Committing changes to gh-pages"
working_branch=$(git branch --show-current)
git checkout gh-pages
echo "Overwritting previous"
rsync -a --remove-source-files  docs/_build/${docdir} ./
git add ${docdir}
# only generate a commit if there were changes (credit: https://stackoverflow.com/a/8123841)
git diff-index --quiet HEAD && echo "No changes, not committing anything" || git commit -m "Update/Add ${docdir} docs"
git checkout ${working_branch}
popd
