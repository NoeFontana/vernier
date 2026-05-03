# TIDE validation harness — rf-detr × COCO val2017

Real-model validation for `vernier.error_decomposition` (the TIDE
machinery). Two complementary entry points share this directory:

- **pytest harness** (`test_tide_real_models.py`) gates structural
  invariants (coherence + determinism). Marked
  `@pytest.mark.real_models`; skipped by default. Run via
  `uv run --extra real-models pytest -m real_models`.
- **`run.py` CLI** produces a human-readable JSON report (per-bin
  ΔmAP, baseline mAP, wall-clock, peak RSS) for ad-hoc validation
  after a TIDE-related change. Not a regression gate.

Why rf-detr: small enough to run on a laptop CPU, ships two
flavors (`RFDETRNano` for bbox, `RFDETRSegNano` for instance
segmentation) so the same package covers all three vernier kernels
(boundary IoU is computed from segm masks). Pinned at
`rfdetr==1.6.5.post0`; bumping is an ADR-level operation per
[`docs/engineering/vendoring.md`](../../../../docs/engineering/vendoring.md).

## Prerequisites

1. **Install the `real-models` extra** (heavy: pulls torch,
   transformers, supervision, ~5 GB):

   ```bash
   uv sync --extra real-models
   ```

2. **Stage COCO val2017** at the cache root the existing parity
   harness uses (`VERNIER_COCO_CACHE` env var, defaulting to
   `<repo>/.cache/coco-val2017/`). The harness needs both the GT
   JSON *and* the image directory:

   ```
   <cache>/instances_val2017.json
   <cache>/val2017/000000000139.jpg
   <cache>/val2017/000000000285.jpg
   ...
   ```

   `tools/fetch-coco-val.sh` downloads the GT JSON; the val2017
   image set is a separate fetch (license-restricted, not in the
   fetcher script).

## Running the pytest harness

```bash
uv run --extra real-models pytest -m real_models -v
```

Three coherence cells (parametrized) plus one determinism check.
Skips cleanly on a clean machine if either prerequisite is missing
(no `rfdetr` import, no COCO val cache, no `val2017/` images).

First run does inference; subsequent runs read from the on-disk
predictions cache (under `platformdirs.user_cache_dir("vernier") /
"real-models"`). Bumping the `rfdetr` pin invalidates the cache by
construction (the version is in the cache filename).

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
  — populate the COCO val cache (see Prerequisites step 2).
- **`FileNotFoundError: image referenced by GT JSON missing on disk`**
  — the GT JSON was downloaded but the val2017 image directory is
  empty or partial. Verify `<cache>/val2017/` contains all 5000
  images.
- **First run takes ~30 minutes** — that's inference. Subsequent
  runs hit the on-disk predictions cache and complete in seconds
  (TIDE is the only cost).
- **`rfdetr.RFDETRSegNano()` downloads model weights on first
  instantiation** — these go to HuggingFace's default cache
  (`~/.cache/huggingface/`), not the vernier-managed predictions
  cache. The two caches are independent.
