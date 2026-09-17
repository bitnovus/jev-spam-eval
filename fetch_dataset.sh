#!/usr/bin/env sh
# Fetch realprogrammersusevim/email-dataset at the commit these results were produced with.
set -eu
COMMIT=84209612df4c55831f074a65b7c137a4b980059d
cd "$(dirname "$0")"
if [ ! -d email-dataset/.git ]; then
  git clone --depth 1 https://github.com/realprogrammersusevim/email-dataset.git email-dataset
fi
if [ "$(git -C email-dataset rev-parse HEAD)" != "$COMMIT" ]; then
  git -C email-dataset fetch --depth 1 origin "$COMMIT"
  git -C email-dataset checkout --quiet "$COMMIT"
fi
echo "email-dataset at $(git -C email-dataset rev-parse HEAD)"
