#!/usr/bin/env bash
# Build the pinned detectron2 rotated-IoU harness (ADR-0063 M0 PR-0.2).
#
# The flag set is the claim. `-march=x86-64` holds the build to the
# baseline ISA detectron2's own wheels target, which has no FMA, and
# `-ffp-contract=off` says so a second time in case a future default
# changes. `-fno-fast-math` is explicit for the same reason: nothing here
# may reassociate.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
out="${1:-${here}/harness}"

flags=(-std=c++14 -O3 -fno-fast-math -ffp-contract=off -I "${here}")
case "$(uname -m)" in
  x86_64) flags+=(-march=x86-64) ;;
  *)      echo "warning: the pinned D2 claim is keyed to linux-x86_64; \
this build is $(uname -m) and is a cross-check only, not the oracle." >&2 ;;
esac

"${CXX:-g++}" "${flags[@]}" -o "${out}" "${here}/harness.cpp"
echo "built ${out}"
