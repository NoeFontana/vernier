#!/usr/bin/env bash
# In a linked worktree, make .cache a symlink to the main checkout's
# .cache, so the val datasets the tools/*_cache packages fetch (about
# 2 GiB) are downloaded once per machine instead of once per worktree.
# The packages write atomically, so concurrent worktrees can share it.
#
# Run by `just bootstrap`. No-op in the main checkout, or when .cache
# already exists.

set -euo pipefail

root=$(git rev-parse --show-toplevel)
main=$(dirname "$(git rev-parse --path-format=absolute --git-common-dir)")
if [ "$root" != "$main" ] && [ ! -e "$root/.cache" ] && [ ! -L "$root/.cache" ]; then
    mkdir -p "$main/.cache"
    ln -s "$main/.cache" "$root/.cache"
fi
