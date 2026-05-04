# Migrating from `lvis-api` to vernier

vernier reproduces `lvis-api`'s evaluation semantics bit-for-bit in
strict parity mode. ADR-0026 is the design record for the LVIS
support; this guide is the user-facing migration path. Audience:
anyone moving an existing LVIS evaluation pipeline onto vernier.

## TL;DR — what to change

| `lvis-api` | vernier |
|---|---|
| `LVIS(gt_path)` | `Dataset.from_lvis_json(gt_bytes)` |
| `LVISResults(lvis_gt, dt_path, max_dets=300)` | `dt_bytes` (the trim is automatic on federated datasets) |
| `LVISEval(lvis_gt, lvis_dt, iou_type="bbox").run()` | `Evaluator(iou=Bbox(), max_dets=(300,)).evaluate(dataset, dt_bytes)` (bbox / segm / boundary) |
| `lvis_eval.print_results()` | `summary.pretty_lines()` (returns list, no stdout) |
| `lvis_eval.results["AP"]` | `summary.lines[0].value` |

The 13-entry summary plan (`AP, AP50, AP75, APs, APm, APl, APr, APc,
APf, AR@300, ARs@300, ARm@300, ARl@300`) lands on `summary.lines` in
that order when `Dataset.from_lvis_json(...)` flips the dataset into
federated mode. No extra summarize call is needed.

## Sentinels: `-1` vs `0` vs `nan`

LVIS, panoptic, and pycocotools each surface a different value when a
summary entry has no input data. **Take care when comparing entries
across codebases.**

| Codebase | Empty bucket → | Why |
|---|---|---|
| LVIS (this guide) | `-1.0` (quirk **AF6**) | `lvis-api/eval.py:441-442`: `s[s>-1]` keeps the `-1` filled at allocation; an entirely-empty filter never overwrites. |
| pycocotools | `-1.0` (quirk **C5**) | Same shape — pycocotools is the lineage. |
| panopticapi | `0.0` (vernier **corrected**, ADR-0025 **W6**) | upstream `panopticapi/evaluation.py:73` raises `ZeroDivisionError`; vernier returns `0.0` because `pq` over zero classes is a defined zero. |
| Uninitialized read | `nan` | Not a sentinel — it's a bug. Surface it loudly. |

Plotting AP_r/c/f as a bar chart? Filter out `-1` first or you'll
get a misleading "below zero" bar for missing buckets. The
[migration trap fixture in `parity_lvis/test_parity_lvis.py`][q5] pins
this behavior.

[q5]: https://github.com/NoeFontana/vernier/blob/main/tests/python/parity_lvis/test_parity_lvis.py

## Federated metadata is silent

`Dataset.from_json(bytes)` (the COCO loader) silently drops the LVIS
extras (`neg_category_ids`, `not_exhaustive_category_ids`,
`frequency`). The orchestrator then runs COCO semantics — every
unmatched detection is a false positive, regardless of whether the
image's GT was complete. Use `Dataset.from_lvis_json(bytes)` for
LVIS data; the federated check at `evaluate.rs` only fires when
`dataset.is_federated() is True`.

If you find yourself comparing AP between vernier and `lvis-api` and
the numbers disagree, the first thing to check is the loader. A
COCO-loaded LVIS dataset will produce systematically lower AP — the
unmatched detections that LVIS would skip via cell-AA4 land as FPs
under COCO.

## `max_dets=300` is explicit

`lvis-api` hardcodes `Params.max_dets = 300` (`eval.py:528`); the
trim happens at `LVISResults` construction time. vernier's
`Evaluator` constructor takes `max_dets` as an explicit argument
(quirk **F2**) — coupling kernel choice to a per-dataset sentinel
through a hidden lookup is exactly the surprise we wanted to avoid
when porting from pycocotools' `useCats=False` defaults.

Pass `max_dets=(300,)` for LVIS evaluation. The federated grid path
auto-applies the per-image top-K trim; you do not need to call
`LVISResults` or its analogue.

## Boundary IoU on LVIS

`lvis-api` does not implement boundary IoU — it raises at
`eval.py:25-26` for any `iou_type` other than `bbox` or `segm`. The
LVIS-flavored boundary IoU lives in the `bowenc0221/boundary-iou-api`
fork (ADR-0010); vernier exposes it as `Boundary(dilation_ratio=0.008)`.
The `0.008` is the LVIS convention (boundary-iou-quirks Q1, ADR-0026
quirk **AE3**); the `Boundary()` default of `0.02` is the COCO
convention. Pick one explicitly when evaluating LVIS data.

## Vendored oracle pin

`lvis==0.5.3` is pinned in `[dependency-groups].dev` of `pyproject.toml`,
mirrored by `ORACLE_LVIS_VERSION` in
`crates/vernier-core/src/lvis_parity.rs`. The vendored source tree
at `tests/python/parity_lvis/oracle/lvis_api/` is byte-equal to the
PyPI sdist (verified via SHA-256 in
`tests/python/parity_lvis/oracle/VENDORING.md`).

Bumping the pin is an ADR-level decision per ADR-0026 §"Parity
strategy". Strict-mode parity claims are keyed to the SHA recorded
in `VENDORING.md`; if the upstream `lvis` PyPI package ever drifts
off `pycocotools==2.0.11` (vernier's pycocotools pin per ADR-0002),
the resolution path is fork-and-pin per the documented procedure.

## Whole-dataset parity smoke

The `tests/python/parity_lvis/test_lvis_val.py` smoke pins the
13-entry summary plan bit-equal against `LVISEval` on the full LVIS
v1 val. It is env-gated and slow (~3 minutes); run with:

```
$ python -m lvis_val_cache              # populate the cache
$ just test-parity-lvis-val
```

The cache is `.cache/lvis-val/` by default; override via
`VERNIER_LVIS_CACHE`. The GT JSON is downloaded under the LVIS
[terms of use][lvis-terms] and is never committed to the repo.

[lvis-terms]: https://www.lvisdataset.org/terms-of-use
