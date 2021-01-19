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
# adopted from https://www.innoq.com/en/blog/github-actions-automation/
#repo_uri="https://x-access-token:${GITHUB_TOKEN}@github.com/${GITHUB_REPOSITORY}.git"
target_branch="test-gh-pages"

#git config user.name "$GITHUB_ACTOR"
#git config user.email "${GITHUB_ACTOR}@bots.github.com"
git config user.name github-actions
git config user.email github-actions@github.com
#git remote set-url origin ${repo_uri}
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
working_branch=$(git branch --show-current)
if [ -z "${working_branch}" ]; then
    # possible in detached head state, just use master
    working_branch=master
fi
git checkout --track -b ${tmpbranch} origin/${target_branch}

echo "Moving built docs and committing"
rsync -a --remove-source-files  docs/_build/${docdir} ./
git add ${docdir}

# only generate a commit if there were changes
if [[ $(git commit -m "Update/Add ${docdir} docs") ]]; then
    echo "No changes, not committing anything"
else
    git push origin ${tmpbranch}:${target_branch}
fi

echo "Deleting temp branch"
git checkout ${working_branch}
git branch -D ${tmpbranch}
