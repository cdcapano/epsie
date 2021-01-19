#!/usr/bin/env sh

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
repo_uri="https://x-access-token:${GITHUB_TOKEN}@github.com/${GITHUB_REPOSITORY}.git"
working_branch=$(git branch --show-current)
target_branch="test-gh-pages"

git config user.name "$GITHUB_ACTOR"
git config user.email "${GITHUB_ACTOR}@bots.github.com"
git remote set-url origin ${repo_uri}

# get cache of the currently documented versions on gh-pages
echo "Getting list of perviously documented versions"
git ls-tree --name-only ${target_branch} | egrep '^latest|^[0-9]+' > docs/.docversions

# clean and build the docs
echo "Building docs"
make -C docs clean
make -C docs ${TYPE}
# get the name of the build directory
docdir=$(ls docs/_build | egrep '^latest|^[0-9]+') 
echo "Docs to commit: ${docdir}"

# commit to gh-pages
echo "Committing changes to ${target_branch}"

git checkout ${target_branch}
echo "Overwriting previous"
rsync -a --remove-source-files  docs/_build/${docdir} ./
git add ${docdir}

# only generate a commit if there were changes (credit: https://stackoverflow.com/a/8123841)
if [ $(git diff-index --quiet HEAD) ]; then
    echo "No changes, not committing anything"
else
    git commit -m "Update/Add ${docdir} docs"
fi

git push origin ${target_branch}

git checkout ${working_branch}
