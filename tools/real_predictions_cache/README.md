# vernier-real-predictions-cache

Single source of truth for the real-model predictions cache contract.
Predictions come from three sources, all landing under the same cache
root (`platformdirs.user_cache_dir("vernier") / "real-models"`):

- **Mask R-CNN R50-FPN (Detectron2 model zoo).** Pre-computed on
  COCO val2017, hosted as a Hugging Face dataset, fetched here via
  `ensure_maskrcnn()`. Pinned URL + SHA256.
- **rf-detr (Nano / SegNano).** Generated on-demand by the TIDE
  validation harness (`tests/python/integration/real_models/tide/`),
  which depends on the `real-models` extra (rfdetr, torch, supervision).
  This package only owns the cache *path*; the inference is owned by
  TIDE.
- **DETR-R50 (`facebook/detr-resnet-50`).** Generated on-demand by the
  Hugging Face SOTA validation harness
  (`tests/python/integration/real_models/sota/`), which depends on the
  same `real-models` extra (torch, transformers, huggingface_hub). The
  hub commit SHA is pinned in `DETR_RESNET50_REVISION` and embedded
  in the cache filename, so a weights bump invalidates the cache by
  construction.

Consumers:

- `tools/fetch-real-predictions.sh` — user-facing bash entry point.
- `bench/bench/workloads/real_predictions.py` — bench harness adapter
  that reads predictions from these cache paths to materialize the
  `coco_val2017_maskrcnn_r50fpn_d2_*` and `coco_val2017_rfdetr_*`
  workload cells.

This package is dev-only: not part of the published `vernier` wheel.
It's a `[tool.uv.sources]` path dependency of `bench/pyproject.toml`.

Common entry points:

```bash
./tools/fetch-real-predictions.sh --maskrcnn          # download Mask R-CNN dump
./tools/fetch-real-predictions.sh --rfdetr nano       # run rfdetr inference (~30 min, real-models extra)
./tools/fetch-real-predictions.sh --rfdetr segnano    # run rfdetr-seg inference (~30 min, real-models extra)
```

Bumping the Mask R-CNN URL / SHA256 is an ADR-level decision: the
prediction blob is what the v1.0 perf snapshot quotes, so swapping it
means re-running the release-mode capture and amending the snapshot.
