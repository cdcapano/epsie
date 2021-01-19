#!/usr/bin/env bash

#
# Builds documentation and commits it to the gh-pages branch.
#
# Usage:
#
#   ./.github/scripts/build_docs.sh TYPE
#
# where TYPE is either "latest" or "versioned".
#
# Note: This *must* be run from the top level directory of the repository.
#
set -e

# get the type of docs we're building
TYPE=$1

# configure git
target_branch="test-gh-pages"
working_branch=$(git branch --show-current)
if [ -z "${working_branch}" ]; then
    # possibly in detached head state, just use master
    working_branch=master
fi

git config user.name github-actions
git config user.email github-actions@github.com
git fetch origin

# get cache of the currently documented versions on gh-pages
echo "Getting list of perviously documented versions"
git ls-tree --name-only origin/${target_branch} | egrep '^latest|^[0-9]+' > docs/.docversions

# clean and build the docs
echo "Building docs"
make -C docs clean
make -C docs ${TYPE}
# get the name of the build directory
docdir=$(ls docs/_build | egrep '^latest|^[0-9]+') 
echo "Docs to commit: ${docdir}"

# commit to gh-pages
echo "Committing changes to ${target_branch}"

# make a random local branch to stage changes in
tmp=$(mktemp)
# remove the file that was created; we only want the string
rm ${tmp}
tmpbranch=$(basename ${tmp})
echo "Staging changes in branch ${tmpbranch}"
git checkout --track -b ${tmpbranch} origin/${target_branch}

echo "Moving built docs and committing"
# remove the current if it exists
if [ -d ${docdir} ]; then
    rm -r ${docdir}
fi
mv docs/_build/${docdir} .
git add ${docdir}

# only generate a commit if there were changes
changes=true
git commit -m "Update/Add ${docdir} docs" || changes=false
if [ "$changes" = true ]; then
    git push origin ${tmpbranch}:${target_branch}
else
    echo "No changes, not committing anything"
fi

echo "Deleting temp branch"
git checkout ${working_branch}
git branch -D ${tmpbranch}
