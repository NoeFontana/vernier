#!/usr/bin/env bash
# Reserve the `vernier`, `vernier-core`, `vernier-mask` names
# on crates.io and the `vernier` name on PyPI. (`vernier-cli` was promoted
# to a real workspace member at v0.2.0 and is no longer reserved here.)
#
# Default mode is --dry-run: every step runs in validation mode and nothing
# is uploaded. Pass --publish to actually upload to crates.io. PyPI uploads
# are deliberately NOT performed by this script — the PyPI placeholder is
# uploaded by the GitHub Actions workflow that uses a Trusted Publisher
# (see docs/engineering/registry-reservations.md when written).
#
# Usage:
#   ./reserve.sh                         # dry-run everything (default)
#   ./reserve.sh --dry-run               # explicit dry-run
#   ./reserve.sh --publish               # publish all 3 crates to crates.io
#   ./reserve.sh --publish --only NAME   # publish a single crate (e.g. vernier-mask)
#
# Auth for --publish:
#   Either run `cargo login <token>` once before invoking, or set
#   CARGO_REGISTRY_TOKEN in the environment for this invocation only.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CRATES_DIR="${SCRIPT_DIR}/crates"
PYPI_DIR="${SCRIPT_DIR}/pypi/vernier"

# Order matters only for legibility; crates.io has no inter-package
# dependencies among these placeholders.
CRATES=("vernier-core" "vernier-mask" "vernier")

MODE="dry-run"
ONLY=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)  MODE="dry-run"; shift ;;
        --publish)  MODE="publish"; shift ;;
        --only)     ONLY="${2:-}"; shift 2 ;;
        -h|--help)
            sed -n '2,/^set -e/{/^set -e/!p;}' "${BASH_SOURCE[0]}" | sed 's/^# \?//'
            exit 0
            ;;
        *)
            echo "error: unknown argument: $1" >&2
            exit 2
            ;;
    esac
done

if [[ -n "${ONLY}" ]]; then
    found=0
    for c in "${CRATES[@]}"; do
        [[ "${c}" == "${ONLY}" ]] && found=1
    done
    if [[ "${found}" -eq 0 ]]; then
        echo "error: --only must be one of: ${CRATES[*]}" >&2
        exit 2
    fi
    CRATES=("${ONLY}")
fi

log()  { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m==>\033[0m %s\n' "$*" >&2; }
ok()   { printf '\033[1;32m==>\033[0m %s\n' "$*"; }

if [[ "${MODE}" == "publish" ]]; then
    if [[ -z "${CARGO_REGISTRY_TOKEN:-}" ]] && ! [[ -f "${HOME}/.cargo/credentials.toml" || -f "${HOME}/.cargo/credentials" ]]; then
        warn "no cargo credentials found. Run 'cargo login <token>' first or set CARGO_REGISTRY_TOKEN."
        exit 1
    fi
    warn "--publish: this will upload to crates.io and the names cannot be deleted, only yanked."
    warn "press Ctrl-C in the next 5 seconds to abort."
    sleep 5
fi

# crates.io: validate or publish each skeleton.
for crate in "${CRATES[@]}"; do
    crate_dir="${CRATES_DIR}/${crate}"
    if [[ ! -d "${crate_dir}" ]]; then
        echo "error: missing crate skeleton: ${crate_dir}" >&2
        exit 1
    fi
    if [[ "${MODE}" == "publish" ]]; then
        log "publishing ${crate} to crates.io"
        ( cd "${crate_dir}" && cargo publish )
        ok "published ${crate}"
    else
        log "dry-run packaging ${crate}"
        ( cd "${crate_dir}" && cargo publish --dry-run --allow-dirty )
        ok "${crate} packages cleanly"
    fi
done

# PyPI: always build, never upload from this script.
log "building PyPI placeholder (sdist + wheel)"
rm -rf "${PYPI_DIR}/dist"
( cd "${PYPI_DIR}" && uv build >/dev/null )
ok "PyPI artifacts built at ${PYPI_DIR}/dist/"
ls -1 "${PYPI_DIR}/dist/"

if [[ "${MODE}" == "publish" ]]; then
    cat <<'EOF'

next: PyPI upload
-----------------
This script does not upload to PyPI. The PyPI placeholder is uploaded by
the GitHub Actions workflow that authenticates via a Trusted Publisher
(OIDC). To kick that off, push a tag matching the workflow's filter or
trigger it manually from the Actions UI.

If you ever need to upload from a laptop in a pinch:
    uv publish tools/reservations/pypi/vernier/dist/* \
        --token "$PYPI_API_TOKEN"
EOF
else
    cat <<'EOF'

dry-run complete. To actually upload:
    1. cargo login <crates.io-token>      # or set CARGO_REGISTRY_TOKEN
    2. ./reserve.sh --publish
    3. trigger the GHA workflow for the PyPI placeholder.
EOF
fi
