# Migrating from `panopticapi` to vernier

vernier reproduces `panopticapi`'s panoptic-quality (PQ) evaluation
bit-for-bit in strict parity mode. ADR-0025 is the design record for
the panoptic support; this guide is the user-facing migration path.
Audience: anyone moving an existing PQ evaluation pipeline onto
vernier.

## TL;DR — what to change

The public surface lives under `vernier.panoptic` (per ADR-0029):

```python
from vernier.panoptic import Dataset, Evaluator, Predictions
```

| `panopticapi` | vernier |
|---|---|
| `pq_compute(gt_json_path, dt_json_path, gt_folder, dt_folder)` | `Evaluator().evaluate(gt: Dataset, dt: Predictions)` |
| `Image.open(...)` + `rgb2id(...)` | `np.array(Image.open(...), dtype=np.uint32)` then pass to `Dataset.from_arrays` (which post-decodes via `id = R + 256·G + 256²·B`) |
| Result dict `{"All", "Things", "Stuff", "per_class"}` | `Summary` dataclass with field accessors; `summary.to_dict(strict=True)` for the panopticapi-shape dict |
| Multiprocessing pool default (`pq_compute`) | `vernier.panoptic.Evaluator` runs single-threaded (ADR-0006). Strict-mode parity is against `pq_compute_single_core(proc_id=0)` directly |

`vernier.panoptic.Evaluator` is **not** a variant of
`vernier.instance.Evaluator` — it's a sibling class because PQ does
not share the AP fold (one-to-one matching on `IoU > 0.5`, no score
gradient, no T/R/A/M axes). Choose the submodule that matches your
metric.

> **Status.** Panoptic PQ is bit-equal vs `pq_compute_single_core` in
> strict mode; `boundary=True` is the one open follow-up. See the
> [README §Status & validation](https://github.com/NoeFontana/vernier/#status--validation)
> matrix.

## Single-threaded vs multi-process

`panopticapi.evaluation.pq_compute` parallelizes over
`multiprocessing.Pool(cpu_count())` and has **no `num_proc`
parameter** — the only knob is host CPU count. For *fixed*
`cpu_count` the result is deterministic, but f64 IoU summation is
non-associative: different `cpu_count` produces different last-bit
`sum_iou`, so the same code on different hosts gives ULP-level PQ
differences (quirk **X2**).

vernier evaluates single-threaded by design (ADR-0006). **Strict-mode
parity is bit-equal against `pq_compute_single_core(proc_id=0)`
invoked with the full annotation list**, bypassing the pool entirely.
Multi-process panopticapi traces match vernier under
`ParityMode::Aligned` only, with a tolerance pinned by
`PANOPTIC_PARITY_EPS` (`crates/vernier-panoptic/src/parity.rs`).

If you are migrating from a CI pipeline that ran `pq_compute` with
the implicit pool: expect last-bit PQ drift on the first run (your
prior numbers are host-dependent; vernier's are not). The strict-mode
parity claim against the single-core path is the deterministic anchor.

## Boundary PQ raises `NotImplementedError`

`Evaluator(boundary=True)` raises `NotImplementedError` pointing at
the **Q3 / Z1** follow-up ADR. The
composition rule in the [`bowenc0221/boundary-iou-api`][bowen-fork]
fork's panoptic path is *not* the instance-case
`min(mask_iou, boundary_iou)` and resolving it requires its own pass.
ADR-0025 §"explicitly does not decide" anchors the deferral.

[bowen-fork]: https://github.com/bowenc0221/boundary-iou-api

## Sentinels: `-1` vs `0` vs `nan`

panopticapi, LVIS, and pycocotools each surface a different value
when a summary entry has no input data. **Take care when comparing
entries across codebases.**

| Codebase | Empty bucket → | Why |
|---|---|---|
| panopticapi (this guide) | `0.0` (vernier **corrected**, **W6**) | upstream `panopticapi/evaluation.py:73` raises `ZeroDivisionError` on `pq / n` when `n == 0`; vernier returns `0.0` because `pq` over zero classes is a defined zero. Strict mode replays the `EmptyCategoryFilter` error shape. |
| LVIS | `-1.0` (quirk **AF6**, ADR-0026) | `s[s>-1]` keeps the `-1` filled at allocation. |
| pycocotools | `-1.0` (quirk **C5**) | Same lineage as LVIS. |
| Uninitialized read | `nan` | Not a sentinel — it's a bug. Surface it loudly. |

If you compare a panoptic-stuff-only run against an LVIS-frequent-only
run, the empty buckets land at `0.0` and `-1.0` respectively. Don't
average across the two without filtering first.

## Things / stuff buckets

panopticapi's report is `[("All", None), ("Things", True),
("Stuff", False)]` — three independent unweighted means over their
isthing-filtered subsets (quirk **W4**). vernier surfaces them as
`Summary.{pq, sq, rq}` for All and
`{pq, sq, rq}_{things, stuff}` for the buckets.

`Evaluator(things_stuff_split=False)` skips the bucket computation;
the `_things` / `_stuff` accessors return `None`. Use this when you
don't care about the breakdown — the kernel work is the same.

Note: panopticapi's `cat_isthing = label_info['isthing'] == 1` check
treats `isthing` as int. vernier accepts `True/False` or `1/0`
(boolean coercion is symmetric in Python).

## `iscrowd` is GT-side only

panopticapi consults `iscrowd` only on the GT (quirk **S6**). DT
`iscrowd` flags are silently ignored. vernier mirrors this: the
`SegmentInfo.iscrowd` field on prediction segments is read but
discarded by the kernel. Don't try to suppress FPs by setting
`iscrowd=True` on predictions — it has no effect.

## Categories are GT-side only

`pred['categories']` is silently ignored by panopticapi (quirk **S9**).
vernier's `Predictions.from_arrays(...)` constructor intentionally
takes **no** `categories` argument; predictions reference the GT's
category taxonomy by id alone. A category id in predictions that's
not in GT raises a structured `ValueError` at the FFI boundary —
the Rust-side `PanopticError` enum surfaces as `ValueError` with a
typed message (corrected over panopticapi's bare `KeyError`, ADR-0025
quirk **Y6**).

## Per-class output shape

panopticapi returns
`{"per_class": {category_id: {"pq", "sq", "rq"}}}` — only the three
ratios, no raw counts (quirk **W8**). vernier extends this with
`n_tp` / `n_fp` / `n_fn` so users debugging a long-tailed
distribution can see *why* a category's PQ is what it is.

`summary.to_dict(strict=True)` drops the count fields so the dict
shape exactly matches panopticapi's `pq_compute` return; default
`strict=False` keeps them.

## Multi-process tolerance

If you need to compare against an existing `pq_compute_multi_core`
run (a deployed CI pipeline using the pool default), use
`ParityMode::Aligned` with the host's `cpu_count`. The
`PANOPTIC_PARITY_EPS` constant pins the bound; bumping it is an
ADR-level decision (the source of truth is the val-measured ULP
ceiling on COCO panoptic val2017 — see Q6 closure procedure in
`tests/python/parity_panoptic/panoptic_val_paths.py`).

For a green-field pipeline, prefer single-threaded vernier and the
strict-mode bit-equal claim against `pq_compute_single_core`. The
multi-process tolerance only exists to support migration from
existing host-dependent baselines.

## Cross-references

- ADR-0025 (panoptic-quality evaluation as a sibling crate).
- ADR-0026 (LVIS support — sentinel-vs-zero migration trap).
- ADR-0010 (boundary IoU — same vendored-oracle + three-tier parity
  model that ADR-0025 follows).
- `tests/python/parity_panoptic/oracle/VENDORING.md` — pinned
  panopticapi commit, byte-equality SHA-256s, fork plan.
- `tests/python/parity_panoptic/test_parity_panoptic.py` — Q1-Q5
  fixtures closing the ADR appendix open questions.
- `tools/panoptic_val_cache/` — env-gated COCO panoptic val2017
  cache for the whole-dataset smoke.
