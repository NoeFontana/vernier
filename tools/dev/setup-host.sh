#!/usr/bin/env bash
# Write host build speedups into a cargo config: sccache as rustc-wrapper
# (dependencies compiled once, shared by every worktree) and clang + mold
# as linker on x86_64 Linux.
#
#   tools/dev/setup-host.sh [--config PATH] [--yes]
#
# PATH defaults to ~/.cargo/config.toml; <dir>/.cargo/config.toml scopes
# the settings to checkouts under <dir>. The settings sit between marker
# comments, so re-running replaces only that block.

set -euo pipefail

config=${CARGO_HOME:-$HOME/.cargo}/config.toml
assume_yes=0
while [ $# -gt 0 ]; do
    case $1 in
        --config) config=$2; shift 2 ;;
        --yes) assume_yes=1; shift ;;
        *) echo "usage: $0 [--config PATH] [--yes]" >&2; exit 2 ;;
    esac
done

begin='# >>> vernier setup-host >>>'
end='# <<< vernier setup-host <<<'

block=$begin$'\n'
if sccache=$(command -v sccache); then
    block+=$'[build]\n'"rustc-wrapper = \"$sccache\""$'\n'
else
    echo "warning: sccache not found; install it with 'cargo install sccache --locked'" >&2
fi
if [ "$(uname -sm)" = "Linux x86_64" ]; then
    if command -v clang >/dev/null && command -v mold >/dev/null; then
        block+=$'[target.x86_64-unknown-linux-gnu]\nlinker = "clang"\nrustflags = ["-C", "link-arg=-fuse-ld=mold"]\n'
    else
        echo "warning: clang and mold not both found; keeping the default linker" >&2
    fi
fi
block+=$end

current=""
[ -f "$config" ] && current=$(cat "$config")
# Everything outside our block, which must not define the same tables.
outside=$(printf '%s\n' "$current" | sed "/^$begin\$/,/^$end\$/d")
for table in '[build]' '[target.x86_64-unknown-linux-gnu]'; do
    if grep -qxF "$table" <<<"$block" && grep -qxF "$table" <<<"$outside"; then
        echo "error: $config already defines $table outside the vernier block;" >&2
        echo "merge these settings by hand:" >&2
        echo "$block" >&2
        exit 1
    fi
done

if [ -n "$outside" ]; then
    desired=$outside$'\n\n'$block
else
    desired=$block
fi
if [ "$desired" = "$current" ]; then
    echo "$config is up to date."
    exit 0
fi

diff -u --label "$config" --label "$config (new)" <(printf '%s\n' "$current") <(printf '%s\n' "$desired") || true
if [ "$assume_yes" -ne 1 ]; then
    read -r -p "Write $config? [y/N] " answer
    [ "$answer" = y ] || [ "$answer" = Y ] || { echo "Aborted."; exit 1; }
fi
mkdir -p "$(dirname "$config")"
printf '%s\n' "$desired" >"$config"
echo "Wrote $config. Builds that predate it recompile once under the new flags."
