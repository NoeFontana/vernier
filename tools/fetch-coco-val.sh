#!/usr/bin/env bash
# Fetch the COCO val2017 ground-truth annotations into a local cache
# for the whole-dataset parity smoke (`just test-coco-val`). With
# `--with-images`, also fetch the val2017/ image set (~6.2 GB
# extracted) — required by harnesses that run inference, e.g.
# tests/python/integration/real_models/tide/.
#
# The GT annotations are required and downloaded here. The detector
# predictions JSON is *not* fetched — choose your own (a Detectron2 or
# MMDetection model-zoo baseline works) and point VERNIER_COCO_DT_PATH
# at it. See docs/engineering/coco-val-parity.md for suggestions.
#
# COCO data is governed by the COCO terms of use; we never commit it.
set -euo pipefail

usage() {
    cat <<EOF
Usage: $(basename "$0") [--with-images] [-h|--help]

  --with-images   Also download and extract val2017/ images
                  (~778 MB zipped, ~6.2 GB extracted). Required by
                  inference harnesses (real-model TIDE validation);
                  not needed by the parity smoke.
  -h, --help      Show this message.

Cache location: \${VERNIER_COCO_CACHE:-<repo>/.cache/coco-val2017}
EOF
}

WITH_IMAGES=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --with-images) WITH_IMAGES=1; shift;;
        -h|--help) usage; exit 0;;
        *) echo "Unknown argument: $1" >&2; usage >&2; exit 2;;
    esac
done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CACHE_DIR="${VERNIER_COCO_CACHE:-${REPO_ROOT}/.cache/coco-val2017}"
ANNOTATIONS_URL="http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
GT_FILENAME="instances_val2017.json"
GT_EXPECTED_SHA256="e8c7f7908f1d7278341fae127d0da654f102f11bd7b21d8aeefa635b8c810b6f"

# Image set: pinned by file count + a canonical-filename probe, not
# SHA256. The GT JSON's bytes matter for parity (every quirk we
# reproduce is keyed to that exact file); image bytes don't — they're
# inputs to inference, and a bit-flip in a JPEG would manifest as a
# different prediction, not a silent parity false-positive. The probe
# (`000000000139.jpg`, the lowest-numbered val2017 image) plus the
# 5000-file count guard against "this directory has 5000 jpgs from a
# different dataset" without committing to a full SHA pin.
IMAGES_URL="http://images.cocodataset.org/zips/val2017.zip"
IMAGES_DIRNAME="val2017"
IMAGES_EXPECTED_COUNT=5000
IMAGES_PROBE_FILENAME="000000000139.jpg"

mkdir -p "${CACHE_DIR}"
gt_path="${CACHE_DIR}/${GT_FILENAME}"

need_fetch=1
if [[ -f "${gt_path}" ]]; then
    actual="$(sha256sum "${gt_path}" | awk '{print $1}')"
    if [[ "${actual}" == "${GT_EXPECTED_SHA256}" ]]; then
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
    if [[ "${actual}" != "${GT_EXPECTED_SHA256}" ]]; then
        echo "ERROR: SHA256 mismatch for ${gt_path}" >&2
        echo "  expected: ${GT_EXPECTED_SHA256}" >&2
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

if [[ "${WITH_IMAGES}" -eq 1 ]]; then
    images_dir="${CACHE_DIR}/${IMAGES_DIRNAME}"
    images_zip="${CACHE_DIR}/$(basename "${IMAGES_URL}")"

    images_dir_ok() {
        # Probe + count: the canonical lowest-numbered jpg must be
        # present, and the directory must have exactly the canonical
        # count. Cheaper than a full SHA pass and catches both partial
        # extractions and "this is some other dataset's 5000 jpgs."
        local n
        [[ -f "${images_dir}/${IMAGES_PROBE_FILENAME}" ]] || return 1
        n="$(find "${images_dir}" -maxdepth 1 -name '*.jpg' -type f | wc -l)"
        [[ "${n}" -eq "${IMAGES_EXPECTED_COUNT}" ]]
    }

    if images_dir_ok; then
        echo "val2017 images already present at ${images_dir}/"
    else
        if [[ -d "${images_dir}" ]]; then
            echo "val2017/ at ${images_dir} fails the integrity check; re-fetching"
            rm -rf "${images_dir}"
        fi
        # Atomic download via .part + mv: a SIGINT mid-download
        # leaves a stale partial zip the next run wouldn't be able
        # to distinguish from a complete one without redownloading.
        # The .part -> final rename only happens after curl exits 0.
        echo "Downloading ${IMAGES_URL} → ${images_zip} (~778 MB)"
        curl --fail --location --output "${images_zip}.part" "${IMAGES_URL}"
        mv "${images_zip}.part" "${images_zip}"
        echo "Extracting val2017/ to ${CACHE_DIR}/"
        unzip -q -o "${images_zip}" -d "${CACHE_DIR}"
        rm -f "${images_zip}"
        if ! images_dir_ok; then
            n="$(find "${images_dir}" -maxdepth 1 -name '*.jpg' -type f 2>/dev/null | wc -l)"
            echo "ERROR: post-extract integrity check failed for ${images_dir}/" >&2
            echo "  expected: ${IMAGES_EXPECTED_COUNT} jpgs, including ${IMAGES_PROBE_FILENAME}" >&2
            echo "  got:      ${n} jpgs" >&2
            exit 1
        fi
        echo "val2017 images ready at ${images_dir}/ (${IMAGES_EXPECTED_COUNT} jpgs)"
    fi
fi

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

if [[ "${WITH_IMAGES}" -eq 1 ]]; then
    cat <<MSG
Images at ${CACHE_DIR}/${IMAGES_DIRNAME}/ are picked up automatically by
the real-model harness — no extra env var, just run:

  uv run --extra real-models pytest -m real_models -v
MSG
fi
