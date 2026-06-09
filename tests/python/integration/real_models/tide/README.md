# TIDE validation harness — COCO val2017 (rf-detr + DETR-R50)

Real-model validation for `vernier.error_decomposition` (the TIDE
machinery). Two complementary entry points share this directory:

- **pytest harness** (`test_tide_real_models.py`) gates structural
  invariants (coherence + determinism) on rf-detr Nano / SegNano
  and DETR-R50, plus numpy-oracle parity on DETR-R50 (closing the
  ADR-0022 follow-up on `t_b = 0.1` for set-prediction transformer
  detectors). Marked `@pytest.mark.real_models`; skipped by default.
  Run via `uv run --extra real-models pytest -m real_models`.
- **`run.py` CLI** produces a human-readable JSON report (per-bin
  ΔmAP, baseline mAP, wall-clock, peak RSS) for ad-hoc validation
  after a TIDE-related change. rf-detr-only; not a regression gate.

Why rf-detr: small enough to run on a laptop CPU, ships two
flavors (`RFDETRNano` for bbox, `RFDETRSegNano` for instance
segmentation) so the same package covers all three vernier kernels
(boundary IoU is computed from segm masks). Pinned at
`rfdetr==1.6.5.post0`; bumping is an ADR-level operation per
[`docs/engineering/vendoring.md`](../../../../docs/engineering/vendoring.md).

Why DETR-R50: a set-prediction transformer detector with a
fundamentally different score distribution from rf-detr's
anchor-based output — exercises the bbox `t_b = 0.1` default
(ADR-0022) on the prediction shape the empirical-ratification
follow-up was waiting for. The DETR-R50 prediction cache is owned
by the sibling SOTA harness (`tests/python/integration/real_models/sota/`,
PR #265); this TIDE tree reads the cache by reference and skips
cleanly when it's absent — we deliberately do not invoke the SOTA
populator from a TIDE fixture (~12 h on an 8-core CPU).

## Prerequisites

1. **Install the `real-models` extra** (heavy: pulls torch,
   transformers, supervision, ~5 GB):

   ```bash
   uv sync --extra real-models
   ```

2. **Stage COCO val2017** at the cache root the existing parity
   harness uses (`VERNIER_COCO_CACHE` env var, defaulting to
   `<repo>/.cache/coco-val2017/`). The fetcher populates both the
   GT JSON and the image directory in one shot:

   ```bash
   ./tools/fetch-coco-val.sh --with-images
   ```

   The image set is ~778 MB zipped, ~6.2 GB extracted. Same
   canonical CDN as the GT, governed by the
   [COCO terms of use](https://cocodataset.org/#termsofuse).
   Resulting layout:

   ```
   <cache>/instances_val2017.json
   <cache>/val2017/000000000139.jpg
   <cache>/val2017/000000000285.jpg
   ...
   ```

## Running the pytest harness

```bash
uv run --extra real-models pytest -m real_models -v
```

Four coherence cells (parametrized: rf-detr Nano / SegNano +
DETR-R50 bbox), one determinism check, and one numpy-oracle parity
cell on DETR-R50 (closes the ADR-0022 follow-up). Per-cell skip
semantics:

- The rf-detr cells skip when `rfdetr` is not importable.
- The DETR-R50 cells skip when the SOTA harness's prediction cache
  is absent (`real_predictions_cache.detr_resnet50_cache_path`).
  Populate via `./tools/fetch-real-predictions.sh --detr` or the
  sibling SOTA harness (`pytest -m real_models tests/python/integration/real_models/sota/`).
- All cells skip when `VERNIER_COCO_CACHE` doesn't point at a
  populated val2017 layout (GT JSON + `val2017/` images).

First run does inference; subsequent runs read from the on-disk
predictions cache (under `platformdirs.user_cache_dir("vernier") /
"real-models"`). Bumping the `rfdetr` pin invalidates the rf-detr
cache by construction (the version is in the cache filename); the
DETR-R50 cache key embeds the full Hugging Face commit SHA (PR #265).

## Running the report CLI

```bash
uv run --extra real-models python -m \
    tests.python.integration.real_models.tide.run \
    --model both --kernel all --output validation-report.json
```

Flags:

- `--model {nano, segnano, both}` — which model(s) to run. `both`
  runs RFDETRNano and RFDETRSegNano. Default: `both`.
- `--kernel {bbox, segm, boundary, all}` — which vernier kernel(s).
  `all` expands to every kernel compatible with the chosen model(s)
  (RFDETRNano supports `bbox` only). Default: `all`.
- `--output PATH` — write the JSON report to a file instead of
  stdout. A human-readable summary table is always printed to
  stderr.

Output schema:

```json
{
  "rfdetr_version": "1.6.5.post0",
  "dataset": "coco-val2017",
  "cells": [
    {
      "model": "nano",
      "kernel": "bbox",
      "baseline_map": 0.4123,
      "delta": {"cls": 0.0287, "loc": 0.0512, "both": 0.0089,
                "dupe": 0.0021, "bkg": 0.0344, "missed": 0.0631},
      "delta_all_fp_removed": 0.0934,
      "config": {"t_f": 0.5, "t_b": 0.1, "kernel": "bbox"},
      "wall_clock_seconds": 5.7,
      "peak_rss_mb_after_cell": 612.4
    },
    ...
  ]
}
```

## Performance targets

These are *targets for a healthy implementation*, not pass/fail
gates. Hardware varies; reviewers diff the JSON report from before
and after a TIDE-related change to spot regressions:

- **bbox** (RFDETRNano predictions, ~7 dets/image): < 6 s
  single-threaded on a modern x86-64 laptop CPU.
- **segm** (RFDETRSegNano, masks): < 30 s. Mask IoU is the cost
  driver; segm is bound by `vernier-mask`'s RLE-intersection kernel.
- **boundary** (RFDETRSegNano, dilation_ratio=0.02): < 60 s. Adds
  the dilation pass on top of segm.

If a change moves any of these by more than 2× without a
documented reason, that's a regression worth chasing.

## Troubleshooting

- **`SKIPPED [reason: real-model harness needs the real-models extra]`**
  — install the extra: `uv sync --extra real-models`.
- **`SKIPPED [reason: real-model harness needs both <gt> and <images>/]`**
  — run `./tools/fetch-coco-val.sh --with-images` (Prerequisites
  step 2).
- **`FileNotFoundError: image referenced by GT JSON missing on disk`**
  — the GT JSON was downloaded but the val2017 image directory is
  empty or partial. Re-run `./tools/fetch-coco-val.sh --with-images`;
  the integrity check (probe + count) auto-detects partial
  extractions and re-fetches.
- **First run takes ~30 minutes** — that's inference. Subsequent
  runs hit the on-disk predictions cache and complete in seconds
  (TIDE is the only cost).
- **`rfdetr.RFDETRSegNano()` downloads model weights on first
  instantiation** — these go to HuggingFace's default cache
  (`~/.cache/huggingface/`), not the vernier-managed predictions
  cache. The two caches are independent.
