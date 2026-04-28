#!/usr/bin/env bash
# Fetch the COCO val2017 ground-truth annotations into a local cache
# for the whole-dataset parity smoke (`just test-coco-val`).
#
# The GT annotations are required and downloaded here. The detector
# predictions JSON is *not* fetched — choose your own (a Detectron2 or
# MMDetection model-zoo baseline works) and point VERNIER_COCO_DT_PATH
# at it. See docs/engineering/coco-val-parity.md for suggestions.
#
# COCO data is governed by the COCO terms of use; we never commit it.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CACHE_DIR="${VERNIER_COCO_CACHE:-${REPO_ROOT}/.cache/coco-val2017}"
ANNOTATIONS_URL="http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
GT_FILENAME="instances_val2017.json"
EXPECTED_SHA256="e8c7f7908f1d7278341fae127d0da654f102f11bd7b21d8aeefa635b8c810b6f"

mkdir -p "${CACHE_DIR}"
gt_path="${CACHE_DIR}/${GT_FILENAME}"

need_fetch=1
if [[ -f "${gt_path}" ]]; then
    actual="$(sha256sum "${gt_path}" | awk '{print $1}')"
    if [[ "${actual}" == "${EXPECTED_SHA256}" ]]; then
        echo "GT already present and verified at ${gt_path}"
        need_fetch=0
    else
        echo "GT at ${gt_path} has unexpected SHA256 (got ${actual}); re-fetching"
        rm -f "${gt_path}"
    fi
fi

dt_bbox_path="${CACHE_DIR}/perfect_dt.json"
dt_segm_path="${CACHE_DIR}/perfect_dt_segm.json"

if [[ "${need_fetch}" -eq 1 ]]; then
    zip_path="${CACHE_DIR}/annotations_trainval2017.zip"
    echo "Downloading ${ANNOTATIONS_URL} → ${zip_path}"
    curl --fail --location --output "${zip_path}" "${ANNOTATIONS_URL}"

    echo "Extracting ${GT_FILENAME}"
    unzip -j -o "${zip_path}" "annotations/${GT_FILENAME}" -d "${CACHE_DIR}"
    rm -f "${zip_path}"

    actual="$(sha256sum "${gt_path}" | awk '{print $1}')"
    if [[ "${actual}" != "${EXPECTED_SHA256}" ]]; then
        echo "ERROR: SHA256 mismatch for ${gt_path}" >&2
        echo "  expected: ${EXPECTED_SHA256}" >&2
        echo "  actual:   ${actual}" >&2
        exit 1
    fi
    echo "GT ready at ${gt_path}"
    # GT just changed; the cached perfect-DTs are now stale.
    rm -f "${dt_bbox_path}" "${dt_segm_path}"
fi

synthesise_dt() {
    local label="$1" out_path="$2"
    shift 2
    if [[ ! -f "${out_path}" ]]; then
        echo "Synthesising a 'perfect' ${label} DT → ${out_path}"
        python3 "${REPO_ROOT}/tools/make-perfect-dt.py" "$@" "${gt_path}" "${out_path}"
    fi
}

synthesise_dt bbox "${dt_bbox_path}"
synthesise_dt segm "${dt_segm_path}" --segm

cat <<MSG

Cache ready. Export the env vars (or eval the lines below) and run
\`just test-coco-val\`:

  export VERNIER_COCO_GT_PATH=${gt_path}
  export VERNIER_COCO_DT_PATH=${dt_bbox_path}
  export VERNIER_COCO_DT_SEGM_PATH=${dt_segm_path}

The synthesised DTs are smoke tests only (AP[0.5:0.95] is trivially
1.0); for non-trivial parity, point VERNIER_COCO_DT_PATH /
VERNIER_COCO_DT_SEGM_PATH at a real detector's predictions JSON. See
docs/engineering/coco-val-parity.md.
MSG
