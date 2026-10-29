#!/usr/bin/env bash
# Delete incremental-compilation caches untouched for DAYS (default 3) in
# every worktree's target/. Nothing else ever removes them — one appears
# per (crate, profile, feature set) — and they were 11 GiB of one
# checkout's 18 GiB target/. Cost of a deletion: one non-incremental
# rebuild of that crate.

set -euo pipefail

days=${1:-3}

git worktree list --porcelain | sed -n 's/^worktree //p' | while read -r wt; do
    [ -d "$wt/target" ] || continue
    removed=$(find "$wt/target" -mindepth 3 -maxdepth 3 -path '*/incremental/*' -type d \
        -mtime +"$days" -print -exec rm -rf {} + | wc -l)
    echo "$wt/target: removed $removed stale incremental caches"
done
