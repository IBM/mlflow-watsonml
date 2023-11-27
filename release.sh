#!/usr/bin/env bash

VERSION=$1
ROOT="$( cd "$(dirname "$0")" ; pwd -P )"

if [ -z $VERSION ]; then
    echo "usage: bash release <version>"
    exit
fi

git checkout mlflow_watsonml/__init__.py
current_branch=$(git symbolic-ref --short HEAD 2>/dev/null)

if [ "$current_branch" != "main" ]; then
    echo "Error: Branch must be 'main'. Current branch is '$current_branch'."
    exit 1
fi

if ! git diff-index --quiet HEAD --; then echo "can't create release, you have uncommitted files"; exit; fi
if git status --porcelain 2>/dev/null | egrep "^ M|??"; then echo "Can't create release, you have uncommitted or untracked files"; exit; fi

# make sure this tag is in our setup.py
if ! grep "__version__ = \"$VERSION\"" mlflow_watsonml/_version.py; then "$VERSION does not match that listed in mlflow_watsonml/_version.py"; exit ; fi

echo creating mlflow_watsonml release $VERSION
echo "# THIS IS AN AUTOMATICALLY GENERATED FILE" > mlflow_watsonml/__init__.py
echo "# DO NOT EDIT IT SINCE YOUR CHANGES WILL"  >> mlflow_watsonml/__init__.py
echo "# BE LOST" >> mlflow_watsonml/__init__.py
echo "__version__ = '$VERSION'" >> mlflow_watsonml/__init__.py

git commit -m"release $VERSION" mlflow_watsonml/__init__.py

cd $ROOT
echo "creating tag and pushing"
git tag -a $VERSION -m"new release" \
 && git push origin $VERSION \
 &&  git push