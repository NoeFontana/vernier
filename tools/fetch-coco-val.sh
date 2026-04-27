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

if [[ -f "${gt_path}" ]]; then
    actual="$(sha256sum "${gt_path}" | awk '{print $1}')"
    if [[ "${actual}" == "${EXPECTED_SHA256}" ]]; then
        echo "GT already present and verified at ${gt_path}"
        printf '\nexport %s=%s\n' "VERNIER_COCO_GT_PATH" "${gt_path}"
        exit 0
    fi
    echo "GT at ${gt_path} has unexpected SHA256 (got ${actual}); re-fetching"
    rm -f "${gt_path}"
fi

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
printf '\nexport %s=%s\n' "VERNIER_COCO_GT_PATH" "${gt_path}"
echo "Now point VERNIER_COCO_DT_PATH at a detector predictions JSON; see"
echo "docs/engineering/coco-val-parity.md."
