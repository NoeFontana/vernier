# Changelog

All notable changes to this project will be documented in this file. The
format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
The project graduated out of the 0.0.x line with 0.1.0 — the four
evaluation paradigms (instance, panoptic, semantic, LVIS federated)
plus the LRP / oLRP and detection-calibration diagnostic surfaces are
all wired with strict bit-equal parity against their oracles, the
local bench harness is stable, and the public API surface has held its
shape across the last three patches. Pre-1.0 still means the API can
break between minor versions; the bar moves from "every patch is
exploratory" to "minor bumps signal breakage, patch bumps are
additive / perf / docs".

## [Unreleased]

### Fixed

- **`evaluate_bbox_grid_with_dataset` is reachable from `vernier.instance`.**
  It shipped in `_core` but was missing from the wrapper's re-export
  list, while all four `evaluate_*_summary_with_dataset` siblings were
  present. The effect was that ADR-0020's parsed-once dataset handle
  served *summary* evaluation but not *grid* evaluation, so a caller
  reading `accumulate()`'s per-class precision and recall — what a
  training-loop metric does — had no public way to avoid re-parsing its
  ground truth for each IoU type. Found by a downstream integration
  hitting exactly that wall, not by this project's own suites, which
  reach `_core` directly and so never notice. The re-export list is
  hand-maintained, so the fix is pinned by an invariant rather than a
  name: `test_instance_reexports_every_dataset_taking_entry_point`
  asserts every `_with_dataset` entry point in `_core` is reachable
  from `vernier.instance`.

## [0.4.0] - 2026-09-18

### Added

- **Ground truth as columnar arrays** (ADR-0060).
  `CocoDataset.from_arrays(images, annotations, categories)` builds a
  dataset from NumPy columns with no JSON in the middle, converging on
  the same `from_parts` constructor the file route ends at — so D1, D2,
  A4, E1 and referential integrity are inherited rather than
  reimplemented, and no route-specific parity surface is created. GT
  `area` is required and read verbatim (unlike a detection's, which
  quirk **J3** derives), and GT ids are supplied, never assigned — the
  mirror image of quirk **J1**.

  A columnar array has no null, but `ignore` and `num_keypoints` are
  genuinely optional *per annotation* and mean something different
  absent than present-and-zero (under **D1**, absent lets
  `parity_mode="corrected"` fall back to `iscrowd`). So an optional
  integer column may be passed signed, where a **negative entry means
  the field was absent on that annotation**. The rule is scoped to
  those two columns: `iscrowd` is required, so a negative entry there
  is refused rather than read as false, which would hand back a
  different crowd set than was passed.

  Measured against `json.dumps` + parse on real published GT (min of 7,
  every cell verified by `dataset_hash`): COCO val2017 bbox-only
  84.4 ms -> 4.5 ms (**18.8x**), full segm 459.5 ms -> 43.0 ms
  (**10.7x**), LVIS v1 val bbox-only 493.1 ms -> 32.1 ms (**15.4x**),
  full segm 4 467.9 ms -> 355.7 ms (**12.6x**).

  LVIS federated metadata has no columnar spelling on this route and is
  **not** supported: `is_federated` is `False` on every dataset it
  produces, pinned by test. LVIS callers use `from_lvis_json`.
- **`CocoDataset.dataset_hash`** — the 32-byte BLAKE3 fingerprint of the
  dataset's canonical form (ADR-0031) is now readable from Python. Two
  handles hash equal exactly when they carry the same images,
  categories, annotations and federated metadata; each section is
  sorted by id first, so the fingerprint pins *content*, not input
  order. It runs detached, because on an LVIS-scale dataset that walk
  costs about as much as the whole ingest.

### Performance

- **`vernier.COCOeval` stops serializing detections** (ADR-0057). The
  drop-in used to `json.dumps` the `cocoDt` annotations and hand the
  bytes to the JSON parser, which rebuilt the objects Python was already
  holding — and held the text and the parsed values at once. It now
  passes the caller's own list of result dicts straight down the
  ADR-0057 list route. The `params.catIds` filter itself did **not**
  move — it is the same comprehension over the same annotation dicts it
  always was, so ADR-0055's "a requested category the dataset never
  declared evaluates as an empty category" is bit-identical by
  construction; what is gone is the re-serialization of the survivors.
  On 500 000 detections over 5 000 images, `evaluate()` end to end falls
  from 1913 ms to 580 ms (**3.3x**, min of 7 interleaved samples, run-to-run
  spread under 3 %) and the stage's peak memory (VmHWM delta) from
  555 MiB to 468 MiB (**-16 %**). `evalImgs` / `ious` re-evaluate off the
  held list, so widening retention costs no serialization either.
- **Grids skip pycocotools-shaped per-cell metadata unless asked for
  it.** Every populated `(k, a, i)` cell used to carry a boxed
  `EvalImageMeta` (sorted DT / GT ids and matched-id arrays) that only
  `EvalGrid.eval_imgs()` and the per_detection / per_pair / LRP paths
  read; accumulate and summarize never do. At COCO-val bbox scale
  (5000 images x 300 detections, 1.56 M populated cells) that was ~5 M
  extra allocations, freed one by one when the grid dropped. Dropping
  the grid falls from 1.2-1.4 s to 0.45-0.6 s at 1 / 4 / 8 threads;
  core matching allocates a third less. Summary paths
  (`Evaluator.evaluate`) get this for free.
- **`accumulate` sorts each category's detection stream once** (ADR-0052).
  The `(K, A, M)` walk ran `argsort_score_desc` per cell — 12 stable
  sorts per category on the COCO defaults — for 12 streams that are all
  derivable from one. The four area ranges of a `(category, image)` pair
  share a `dt_scores` vector, and each smaller `maxDet` stream is an
  induced subsequence of the largest, so filtering one permutation
  reproduces all twelve exactly. Both derivations assume scores are
  totally ordered, so a grid built through the `pub` Rust API with a
  `NaN` score — the dataset path rejects those — falls back to the
  per-cell sort, and the area-range guard compares bit patterns so
  `-0.0` keeps its sign. Bit-equal output; `accumulate` drops
  16.9 % on a val2017-shaped grid. Long-tail grids (LVIS) barely move —
  their per-category streams are short and the cost is the dense grid
  walk, which this does not touch.
- **Detections that cannot match no longer scan the GT list**
  (ADR-0053). The matching ladder read all `G` GTs per
  `(threshold, detection)` even when the detection's best overlap was
  below the lowest rung. Per-cell column maxima make that skip exact —
  `best` only rises from the threshold seed, so the scan was provably a
  no-op — and the ladder is monomorphized on whether the prefilter is
  active, so cells below the `G · D = 256` gate compile to the previous
  loop. Dense cells (250 GT x 250 DT) drop 84 %; COCO-shaped cells are
  unchanged.
- **JSON ingestion splits across the thread budget** (ADR-0054). One
  structural scan finds every element's byte range; the elements are then
  handed to the same `serde_json` deserializers the serial path uses, in
  input order, so parsed values are bit-identical and `loadRes` positional
  ids (quirk **J1**) are preserved. Anything the splitter does not plainly
  recognize — an unexpected shape, a duplicate key, a parse error — falls
  back to the serial loader, which also owns the canonical error message.
  LVIS v1 val GT parse: 732 ms -> 204 ms at 8 threads (3.6x); end-to-end
  `evaluate_bbox_grid` on that dataset 1142 ms -> 554 ms (-51 %).
  `num_threads=None` stays serial per ADR-0047.

### Benchmarks

Full release-mode round on the host the previous two rounds used
(`59aab88b17f4`), so these are directly comparable — see
`docs/engineering/benchmarking/2026-09-release-0.4.0-round.md`.

| cell | 0.3.0 | 0.4.0 |
| --- | ---: | ---: |
| COCO bbox, 1 CPU | 354 ms | **305 ms** |
| COCO segm, 1 CPU | 968 ms | **931 ms** |
| LVIS v1 val bbox | 2.64 s | **2.50 s** (81.0x over lvis-api) |
| Objects365, 1 CPU | 8.771 s | **6.60 s** (-24.8 %) |
| Objects365, 8 threads | 4.702 s | **3.20 s** (-32 %) |
| thread scaling, bbox 1->8 | 1.56x | **2.34x** |
| thread scaling, segm 1->8 | 3.08x | **3.87x** |
| thread scaling, boundary 1->8 | 4.05x | **4.47x** |

The Objects365 numbers carry their own control: hotcoco (+0.6 %,
+1.9 %) and pycocotools (+1.6 %) land within 2 % of the previous round
on the same host, so the movement is the code (ADR-0050, ADR-0051), not
the machine. The mechanism is ingest — vernier's `load` stage is 0.32 s
against hotcoco's 5.77 s, and at eight threads 6.02 s of hotcoco's
8.17 s total is still serial parsing.

vernier is **bit-equal to `lvis-api`** on both LVIS cells (identical
tensor hashes). `hotcoco_lvis` diverges from the reference at 7 069 and
6 060 positions — a known-open divergence on hotcoco's side, not
vernier's.

### Changed (BREAKING — pre-1.0)

- **`vernier.COCOeval` holds the caller's detection list instead of a
  JSON snapshot of it** (ADR-0057). That reference is where the memory
  above is saved, and it opens an aliasing window the `json.dumps` path
  closed by accident. `eval` and `stats` are computed during
  `evaluate()` / `accumulate()` and frozen, but `evalImgs` and `ious`
  re-ingest the held list on first read (ADR-0055's lazy widening), so a
  detection dict mutated in place *after* `evaluate()` shows up in those
  two while `stats` still reports the pre-mutation numbers, and
  `.clear()`ing the list yields cells with empty `dtIds`. It is
  asymmetric: under a `params.catIds` subset the shim holds a *new* list
  of the same dicts, so list-level mutation no longer propagates while
  element-level mutation still does. Hand each pass its own copy if you
  mutate detections between passes.
- **`vernier.COCOeval` detection scores can differ from 0.3.0's by one
  ULP on real payloads.** The old path serialized scores and re-parsed
  them through vernier's JSON number parser, which can land one ULP off
  the `float` Python already held (the `dtScores` drift reported in
  #265, visible only on real prediction data). The list route hands the
  `float` straight over, so the drift is gone — an improvement, but the
  retained `dtScores` and `eval["scores"]` do change against the
  previous release, and that is worth knowing before you diff them.
- **`summarize_detection` takes a `ParityMode`** and reads the M-axis the
  way pycocotools' `_summarizeDets` does (quirk L9): `AR_1` / `AR_10` /
  `AR_100` read `max_dets[0|1|2]` positionally instead of looking up
  `1` / `10` / `100`, so a ladder such as `[1, 10, 500]` summarizes
  instead of raising `InvalidConfig`. The aggregate `AP` reads
  `maxDets=100` in strict mode (`-1` when absent, bit-exact with
  pycocotools) and the largest cap in corrected mode. Both modes are
  unchanged on the default `[1, 10, 100]` ladder.
  `StatRequest::coco_detection(parity_mode)` and
  `Breakdown::detection_plan(parity_mode)` carry the same plans;
  `MaxDetSelector` gains `Index(i)` and `ValueOrSentinel(n)`.
  `Accumulated.summarize()` in Python uses the parity mode of the grid it
  was accumulated from.
- **`EvaluateParams` gains `retain_meta`** (Rust struct literals need
  it). `EvalGrid::eval_imgs_meta` is filled only when `retain_meta` or
  `retain_iou` is set, and is empty otherwise (`EvalGrid::has_meta`).
  The per_detection / per_pair table builders now raise `InvalidConfig`
  on a grid without metadata instead of returning empty tables.
- **`evaluate_{bbox,segm,boundary,keypoints}_grid` and
  `evaluate_bbox_grid_with_dataset` take `retain_meta=False`**;
  `EvalGrid.eval_imgs()` raises `ValueError` on a grid built without
  `retain_meta=True`. `vernier.COCOeval` leaves it off and re-evaluates
  on first read of `evalImgs` (see Added below).

### Added

- **Two direct detection-ingest routes for in-process callers**
  (ADR-0057). `dt=` now also takes a list of per-annotation COCO
  *result* dicts — the shape `pycocotools.COCO.loadRes` consumes and a
  TorchMetrics-style caller already holds — and an `(N, 7)`
  C-contiguous float64 matrix laid out as
  `image_id, x, y, w, h, score, category_id`. Both skip the
  `json.dumps` / parse round trip entirely. Neither adds semantics:
  both terminate in the same `Vec<DetectionInput>` the file route
  produces, so `loadRes`'s id assignment (**J1**), area derivation
  (**J3**) and forced non-crowd flag (**E2**/**J4**) keep happening in
  exactly one place; `tests/python/test_ingest_route_equivalence.py`
  pins the three routes cell-for-cell on `eval_imgs()`. The list
  route's `segmentation` takes every shape a results file carries
  (polygons, `str` and list-of-ints `counts`) plus ADR-0030's
  in-memory forms; its `area` is carried, so `dt_area="supplied"`
  reads it exactly as it does from a file. `vernier.instance` exports
  the payload types: `ResultAnnotation`, `DetectionMatrix`,
  `SegmentationInput`, `JsonRLE`, `PolygonSegmentation`.
- **`vernier.COCOeval.ious`**, the pycocotools-shaped
  `{(imgId, catId): matrix}` map TorchMetrics reads under
  `extended_summary=True` (ADR-0055). Each matrix is
  `(detections, ground truths)` with detections score-descending and
  truncated to `max(params.maxDets)`; a pair with nothing on one side is
  the bare `[]` upstream returns (quirk **F5**). Backed by a new
  `EvalGrid.ious()` on the FFI, which requires `retain_iou=True`;
  `evaluate_keypoints_grid` gains the `retain_iou` flag the other three
  grids already had, and `RetainedIous::iter` is public.
- **`vernier.COCOeval` supports `params.catIds` subsetting** (ADR-0055),
  which is how `MeanAveragePrecision(class_metrics=True)` drives one
  evaluator around a per-class loop. The shim filters `categories` and
  `annotations` exactly as `COCOeval._prepare` does; a category the
  dataset never declares evaluates to a row of `-1`s, as upstream.
  `params.imgIds` subsetting and `params.areaRng` mutation still raise.
- **`evalImgs` and `ious` are built on first read** and cached, so the
  evaluate / accumulate / summarize cycle keeps both retentions off and
  a caller that reads either pays for it once.
- **`vernier.adapters` publishes the COCO-JSON normalizers**
  (ADR-0055): `with_placeholder_image_sizes`, `with_mask_image_sizes`,
  `detection_image_sizes`, `coco_json_default` and `to_coco_json` — the
  conversions the drop-in applies to a `pycocotools`-shaped dataset, for
  callers that assemble one and drive a vernier grid directly rather
  than through the shim.
- **NumPy 1 is supported again: `numpy>=1.26`** (was `>=2.0`). The abi3
  extension binds to NumPy 1 or 2 at runtime and the Python layer uses no
  NumPy-2-only API; CI's `test (python 3.10, numpy 1.26.4)` leg runs the
  PR test suite against NumPy 1 so the floor stays honest. Unblocks
  installs next to packages that still pin NumPy 1 (e.g. `onnx2tf`).
- `evaluate_*_grid(..., dt_area="supplied")` reads a supplied `area`
  verbatim, falling back to the bbox when absent — pycocotools'
  `COCOeval`, which takes `d['area']` off whatever `cocoDt` it is handed
  (quirk J3). `DetectionInput` gains an optional `area` (Rust struct
  literals need `area: None`), ignored unless the detections are built
  with `DetectionArea::Supplied`.
- `evaluate_{segm,boundary}_grid(..., dt_area="mask")` takes
  `maskUtils.area` of each detection's RLE, as `loadRes` does for segm
  results, for JSON and array-form detections (compressed, uncompressed
  or bitmask RLEs). It raises for a detection without an RLE, and on the
  bbox / keypoints grids (quirk J3). Joint bbox + segm evaluation from
  one array payload can now bucket each pass by its own area.

### Fixed

- **OKS divides by the keypoint count instead of multiplying by its
  reciprocal** (quirk **F7**). The kernel hoisted a `1.0 / count` out of
  the detection loop and multiplied; pycocotools computes
  `np.sum(np.exp(-e)) / e.shape[0]`. For any count that is not a power
  of two the reciprocal rounds once and the product rounds again, so the
  OKS landed 1 ULP off the oracle on 8.4 % of cells at the COCO-person
  count of 17 (29.4 % at 14, 36.6 % at 133). Because the matching ladder
  gates on `iou >= t`, that shift can drop a match that sits exactly on
  a threshold — reachable with the arbitrary-length per-category sigmas
  quirk **F1** ships, and with user-supplied `iou_thresholds` at any
  count. Perfect matches also stop being bit-exactly `1.0` at counts
  such as 49 and 98. The division costs well under 1 % of a cell that
  already runs `count` `exp()` calls.
- **`vernier.COCOeval` matches pycocotools on in-memory COCO objects**
  such as the ones TorchMetrics builds:
  - Mutated `params.iouThrs` / `params.recThrs` are evaluated as given
    (e.g. float32-rounded ladders from `torch.linspace(...).tolist()`)
    instead of raising `NotImplementedError`.
  - Detection `area` is read from `cocoDt`, as `COCOeval` does, instead
    of being re-derived from the bbox; a detection whose mask and box
    fall in different area buckets was mis-bucketed under `segm`.
  - RLE `counts` given as `bytes` (straight from
    `pycocotools.mask.encode`) serialize instead of raising `TypeError`.
  - GT images without `width` / `height` evaluate wherever pycocotools
    never reads the size: always under bbox / keypoints; under segm /
    boundary when no GT annotation is on the image (the size comes from
    the cocoDt image when DT masks are on it). Where pycocotools raises
    `KeyError`, vernier still raises its schema error.
  - `maxDets` ladders without `100` summarize (see L9 above) instead of
    raising from `accumulate()`; plan errors now surface from
    `summarize()`.
  - `evalImgs` is built on first read instead of in `evaluate()`; an
    evaluate / accumulate / summarize cycle no longer pays for the
    per-image dicts it never reads, nor for the per-cell metadata behind
    them (see Added).
  - `params.catIds = [class_id]` evaluates that category instead of
    raising, so `MeanAveragePrecision(class_metrics=True)` works.

- **Array ingest accepts every buffer NumPy and torch call contiguous.**
  A size-1 axis's stride is ignored and a zero-element array is
  contiguous whatever its strides, so one-detection `(1, 4)` boxes and
  empty `(0, 4)` / `(0,)` per-image batches from
  `tensor.contiguous().numpy()` no longer raise "not C-contiguous"; an
  empty torch tensor's null data pointer is accepted too.
- **JSON floats are now correctly rounded** (ADR-0054). `serde_json`'s
  default parser sent some near-tie decimals to the adjacent double,
  which drifted ~16 % of `eval_imgs.dtScores` by 1 ULP against
  pycocotools on real detector output (documented in
  `docs/engineering/real-predictions-parity.md`, where it was held under
  that page's float-tolerance gate).
  Enabling `float_roundtrip` makes vernier's doubles bit-equal to
  CPython's `json`. Costs ~4 % on the sequential parse path.

### Removed

- **`vernier eval --parity-mode aligned` is gone** (ADR-0059). Pass
  `strict` instead: ADR-0002's 2026-05-10 amendment folded the
  `aligned` disposition tier into `strict`, and the CLI value had been
  a silent alias for `strict` ever since — so this changes no number
  any invocation reports, only what you are allowed to type. It was
  also the last place in the project where a user could name a parity
  mode that does not exist: `ParityMode` has shipped `Strict` /
  `Corrected` only, and the Python `parity_mode` argument has rejected
  `"aligned"` since the amendment. Because the JSON formatter emits the
  kernel-resolved mode, `--parity-mode aligned --emit json` used to
  record `"parity_mode": "strict"`, so the flag never round-tripped
  through its own result document. This is a breaking change to a
  committed CLI surface (ADR-0015 §"Output stability"): a pinned
  invocation now fails at argument-parse time with exit code 2, before
  any eval work, and the error names `strict` as the replacement and
  lists the valid values.

## [0.3.0] - 2026-09-16

### Performance

- **The evaluate path stops paying for empty cells** (ADR-0051). Both
  evaluate paths walked the full `K x I` cell grid and discovered
  emptiness inside the per-cell body, and the parallel path then permuted
  an image-major buffer into canonical layout. Both costs scale with
  `K * A * I` however little of the grid holds anything — invisible at
  COCO's 80 categories, dominant at Objects365's 365 (29.2 M cells, 2.1 %
  occupied) and LVIS's 1203. `cell_occupancy` now builds the candidate
  list once; skipping a non-candidate is output-identical, being the
  negation of the emptiness check the per-cell body already performs. The
  grid is filled one category per worker instead of permuted.
- **`accumulate` parallelizes across the category axis** (ADR-0050),
  which ADR-0047 had deferred. Categories own disjoint slices of all
  three output tensors, so the fan-out is bit-identical by construction —
  no float reduction crosses a thread boundary. `num_threads=None` keeps
  the sequential walk and never enters rayon.
- **GT and DT parse concurrently** when the call has a thread budget.
  They are independent payloads that were parsed in sequence.

  Measured on one host, same harness mode and CPU budget before and after
  (`docs/engineering/benchmarking/2026-09-longtail-perf-round.md`):

  | workload | before | after |
  | --- | ---: | ---: |
  | Objects365 val bbox, 8 threads | 8.76 s | **4.70 s** |
  | Objects365 val bbox, 1 CPU | 10.00 s | **8.77 s** |
  | LVIS v1 val bbox (1203 categories) | 3.42 s | **2.64 s** |

  Thread scaling is where most of it lands — `num_threads` cells on
  COCO val2017, before → after:

  | iou | `nt=2` | `nt=4` | `nt=8` |
  | --- | ---: | ---: | ---: |
  | bbox | 350 → 267 ms | 316 → 229 ms | 314 → **226 ms** |
  | segm | 652 → 569 ms | 468 → 375 ms | 408 → **319 ms** |
  | boundary | 1.77 → 1.70 s | 1.02 → 0.94 s | 867 → **790 ms** |

  bbox previously barely scaled (1.14× from 1 to 8 threads, parse-bound);
  it now reaches 1.56×. Keypoints is unchanged — a one-category grid has
  no empty cells to skip.

  COCO single-CPU cells are unchanged (bbox 356 → 354 ms), as are
  panoptic and semantic: the tax this removes grows with the category
  axis. Output stays bit-equal to pycocotools at every thread count, and
  `vernier_lvis` stays bit-equal to the `lvis-api` oracle.


### Changed (BREAKING — pre-1.0)

- **`BackgroundEvaluator.finalize_with_tables(...)` is now keyword-only.**
  Its seven parameters (`per_image`, `per_class`, `per_detection`,
  `per_pair`, `per_pair_iou_floor`, `per_pair_max_rows`,
  `per_detection_with_geometry`) gained a `*` in the Rust
  `#[pyo3(signature = ...)]`, matching what `_core.pyi` has declared
  since 2026-05-09 and matching the sibling `BackgroundEvaluator(...)`
  constructor. Positional calls such as
  `ev.finalize_with_tables(True)` now raise `TypeError`; pass
  `per_image=True` instead. Type-checked callers were already
  constrained by the stub and are unaffected.
- **Context-manager `__exit__` parameters renamed** on
  `BackgroundEvaluator`, `BackgroundPanopticEvaluator` and
  `BackgroundSemanticEvaluator`: `_exc_type` / `_exc` / `_tb` →
  `exc_type` / `exc` / `tb`. The leading underscores were a Rust
  unused-variable artifact leaking into the public Python signature.
  `with` statements are unaffected (CPython calls `__exit__`
  positionally); only explicit keyword calls would break.

### Added

- **hotcoco joins the bench matrix** (ADR-0049) — `hotcoco==1.0.1`
  is benchmarked and parity-checked on instance bbox / segm / keypoints and on
  LVIS bbox, wheel-only so the measured artifact is what `pip install` ships.
  It is the closest competitor: 1.37–1.62x behind vernier per CPU on COCO and
  1.03x on LVIS. Its COCO precision tensor sits 1 ULP from pycocotools (same
  cells as faster-coco-eval); on LVIS it diverges from the `lvis-api`
  reference by up to 8.0e-3 per cell on both GT-as-DT and jittered detections,
  where vernier is bit-equal.
- **Objects365 v2 val scale workload** — `objects365_val_jittered_seed<N>`
  (80k images, 1.24M GT boxes, 365 categories, ~1.06M detections, bbox only).
  Annotations are fetched CC BY 4.0 and SHA-256-pinned by the new
  `tools/objects365_val_cache`; images are never downloaded and no dataset
  bytes are committed. vernier stays bit-exact to pycocotools at that size
  while being 35.8x faster and using 4.5x less memory; faster-coco-eval is
  OOM-killed at ~30 GiB.
- **Per-cell CPU budget in the bench harness** (ADR-0049) — every batch runner
  is pinned to `cpu_budget(num_threads)` logical CPUs (1 for headline cells),
  chosen one per physical core before SMT siblings, with the budget also
  forwarded to `RAYON_NUM_THREADS` / `rle_iou_max_workers` /
  `boundary_cpu_count` / `torch.set_num_threads`. Needed because
  faster-coco-eval >= 1.8, hotcoco and mmsegmentation are multi-threaded by
  default; without it the headline compared 1 thread against 8.
- **Exact per-stage memory** — stages record RSS at entry and the kernel's
  `VmHWM` high-water mark (reset per stage via `/proc/self/clear_refs`), plus
  process CPU time and the observed CPU affinity. `docs/benchmarks.md` gains
  CPU/wall, peak RSS and eval-Δ-RSS columns and generated thread-scaling
  tables.
- **`vernier` is now a real facade crate** (ADR-0048). `crates/vernier/`
  is a seventh publishable workspace member whose entire content is
  whole-crate re-exports plus rustdoc:
  `vernier::{instance, mask, panoptic, semantic, partial}` alias
  `vernier-core` / `vernier-mask` / `vernier-panoptic` /
  `vernier-semantic` / `vernier-partial`. `cargo add vernier` now gets
  the whole library under one dependency, with a module map that
  mirrors the Python namespace from ADR-0029. Because the aliases are
  whole-crate, the facade's public API *is* the union of the leaf APIs
  and cannot drift from them — the flip side being that any breaking
  change in any leaf crate is a breaking change in `vernier`.

  Three additive, default-on, re-export-only features (`panoptic`,
  `semantic`, `partial`) let a narrow consumer trim within the facade
  rather than abandon it; `instance` and `mask` are unconditional, so
  the crate is non-empty in every feature combination. A feature here
  changes what is *nameable*, never what is *computed*: none is
  forwarded to a paradigm crate, so ADR-0047's "one wheel, one
  behavior" is untouched. `cargo hack --feature-powerset check -p
  vernier` gates every combination in CI and in `just lint`.

  `vernier-ffi` and `vernier-cli` are deliberately not re-exported —
  the first ships only inside the wheel, and a library dep on the
  second would drag `clap` into every consumer's tree (ADR-0015).

- **Stub conformance test** (`tests/python/test_core_stub_conformance.py`)
  — asserts `python/vernier/_core.pyi` matches the compiled
  `vernier._core` in symbol coverage, class members, property-vs-method
  kind, and parameter names / order / defaults / keyword-only boundary.
  Runs in the `test-py` job against the built wheel, so it validates the
  stub as shipped. Types are deliberately not compared; they stay
  hand-curated. Rationale and maintenance rules:
  `docs/engineering/python-type-stubs.md`.
- **`pyright --verifytypes vernier`** in `just lint-py` and CI, guarding
  completeness of the public typed surface (baseline 100%).

### Fixed

- **LVIS val cache never verified** — `lvis_val_cache.ensure_gt` compared the
  *extracted* JSON against a SHA-256 that is the *zip's*, so it raised on every
  clean cache; `just test-parity-lvis-val` and the bench LVIS cell were both
  unreachable from scratch. Now verifies the archive before extraction,
  matching the sibling `lvis_v1_val_cache`. The pinned value is unchanged — the
  upstream artifact has not drifted — so ADR-0026's "bumping the pin is an
  ADR-level decision" is not engaged.

- **rkyv bumped to 0.8.18 for RUSTSEC-2026-0233 / -0234 / -0235.** The
  archive validator that ADR-0031 relies on — `bytecheck` running under
  `rkyv::access` — accepted several classes of malformed archive:
  insufficient archive-range validation could reach a use-after-free,
  and incomplete hash-table and `Rc`/`Arc` pointer validation could
  reach out-of-bounds reads. Partials are cross-process input by
  design, so this is the threat model the crate was chosen for. The
  workspace floor moves from `"0.8"` to `"0.8.17"` (the first fixed
  release) so the bound binds for downstream consumers of the
  published crates, not just for our lockfile.

  **No wire-format change.** `FORMAT_VERSION` stays at `2`: a partial
  emitted by an 0.8.16 build decodes under 0.8.18 with bit-identical
  confusion counts, verified across two wheel builds. The advisories
  tighten validation; they do not move bytes.
- **`evaluate_{bbox,segm,boundary,keypoints}_summary_with_dataset`
  first parameter is named `dataset`, not `gt`,** in `_core.pyi`. The
  stub had advertised a keyword that did not exist, so
  `evaluate_bbox_summary_with_dataset(gt=..., ...)` raised `TypeError`
  despite type-checking. Positional callers were unaffected.
- **`Breakdown` is correctly typed as unhashable.** `_core.pyi` declared
  `def __hash__(self) -> int`, but the class is `#[pyclass(eq)]` without
  `hash`, so PyO3 sets `__hash__ = None` and `hash(breakdown)` raises
  `TypeError: unhashable type`. The stub now spells it
  `__hash__: ClassVar[None]`, so putting a `Breakdown` in a `set` or
  using one as a `dict` key is caught at type-check time instead of at
  runtime. Runtime behaviour is unchanged.

### Removed

- **`pyo3-stub-gen` from `[workspace.dependencies]`** — it was declared
  but depended on by no crate, never resolved into `Cargo.lock`, and no
  `#[gen_stub_*]` macro was ever used. `_core.pyi` is hand-written by
  design; see `docs/engineering/python-type-stubs.md`.
- **`tools/reservations/` in full** — the four crate skeletons, the PyPI
  skeleton, and `reserve.sh` (ADR-0048). Every name it held has been
  redeemed by a real release. The practice it encoded produced a
  crates.io anti-squatting report on 2026-08-10 against the empty
  `vernier@0.0.0` placeholder, and is replaced by one rule: *a crate
  name is claimed by its first real release, and never before* — an
  anticipated crate is recorded in the ADR that anticipates it, and
  nothing is published to hold it. `vernier-assign` and any future 3D
  evaluation crate are explicitly not pre-reserved.
  `docs/engineering/registry-reservations.md` is rewritten from a
  reservation register into a published-artifact register.

## [0.2.0] — 2026-06-09

Real-prediction parity follow-up to 0.1.0. The headline work is six
new SOTA-harness cells that drive every kernel — bbox, segm, boundary,
keypoints, panoptic PQ, semantic mIoU, LVIS, and calibration — through
a frozen real-model prediction cache so the parity surface no longer
relies solely on synthetic fixtures. Two strict-mode behavioural fixes
ride along: the TIDE Missed-bin rewrite (previously a no-op under
`parity_mode="strict"`) and the accumulator's `n_d==0` precision/scores
write. The minor bump signals those output-value changes; the kernel
surface is otherwise unchanged from 0.1.0.

### Added

- **Real-prediction SOTA harness — six new cells.** Each cell drives a
  pinned upstream checkpoint through the existing `_harness_common`
  scaffolding (full-SHA cache key, `_ensure_pinned_revision` preflight,
  `torch.set_num_threads(1)`, `int64` target_sizes, loud-fail on
  unmapped class names) and asserts vernier-vs-oracle parity on real
  output distributions:
  - **DETR-R50** (`#265`) — instance bbox / segm against the
    `facebook/detr-resnet-50` checkpoint on COCO val2017. Aligned tier
    loosens `dtScores` to `rtol = 2 * eps` to absorb the documented
    `serde_json` vs Python `strtod` 1-ULP score-parser drift; all
    integer-reduction surfaces (precision, recall, counts, 12-stat AP/AR
    summary) stay bit-equal.
  - **Mask2Former panoptic + ADE-semantic** (`#266`) — panoptic PQ
    against `facebook/mask2former-swin-large-coco-panoptic` on COCO
    panoptic val2017; semantic mIoU against the ADE checkpoint on
    ADE20K val. Both bit-equal to their oracles on integer-reduction
    surfaces.
  - **DETR-R50 calibration** (`#267`) — reuses the `#265` prediction
    cache to validate ADR-0018 ECE / MCE / reliability against the
    NumPy oracle at full distribution scale.
  - **rfdetr-segnano boundary** (`#269`) — boundary IoU against
    bowenc0221's `boundary_iou_api` over the rfdetr-segnano TIDE cache;
    no new inference (boundary IoU is a different metric over the same
    RLE masks).
  - **LVIS detector** (`#270`) — federated LVIS evaluation against the
    LVIS API. Reuses the TIDE cache pattern; gates the K=168/817
    full-val divergence currently tracked in `open_followups.md`.
  - **ViTPose keypoints** (`#271`) — keypoints OKS evaluation against
    the `usyd-community/vitpose-base-coco` checkpoint on COCO val2017.

### Fixed

- **TIDE Missed-bin strict-mode parity** (`#273`) — the rewrite-layer
  Missed fix was setting `ignore_flag = Some(true)` on missed GTs and
  relying on `effective_ignore` to resolve under both parity modes.
  Quirk D1's strict disposition discards `ignore_flag` entirely and
  reads only `is_crowd`, so under `parity_mode="strict"` the rewrite
  was a no-op: the AP denominator stayed unchanged and the per-bin
  delta collapsed to exactly 0.0 (vs the ADR-0021 NumPy oracle's
  spec'd 0.119 on DETR-R50). Fixed by deleting missed GTs from the
  corrected dataset entirely — parity-mode-independent and
  AP-equivalent to ignoring on the oracle's semantics. Validated to
  within 1 ULP against the oracle on COCO val2017 + DETR-R50
  (~150k detections, 8 ULP gate). Closes the ADR-0022 follow-up on
  `t_b = 0.1` for set-prediction transformer detectors.
- **`n_d == 0` precision/scores write** (`#272`) — the accumulator path
  for classes with zero detections now writes `0.0` (not `-1`) into
  the precision and scores tensors. Downstream consumers comparing
  raw tensor values across releases will see this change; the public
  AP / AR summary statistics are unaffected (they already skipped
  `-1` sentinel entries).

## [0.1.0] — 2026-05-19

First release out of the 0.0.x line. Mostly a performance + parallelism
follow-up to 0.0.4 — no new evaluation paradigms, every shipped kernel
keeps strict bit-equal parity with its oracle. The cross-paradigm
benchmark page is refreshed against the post-0.0.4 SHA on the same
machine fingerprint as the 0.0.4 snapshot (`37652a58e939`).

### Added

- **`num_threads` parallelism** ([ADR-0047](docs/adr/0047-threading-model.md))
  (#251, #253, #254, #256) — opt-in `num_threads: int | None = None`
  on every public evaluate surface across all four paradigms: instance
  (bbox / segm / boundary / keypoints), semantic, panoptic, and LVIS,
  on batch + streaming + background entry points (`Evaluator.evaluate`,
  `Evaluator.background`, `submit` / `submit_png`). The sequential
  path (`num_threads=None` or `1`) is byte-for-byte unchanged from
  0.0.4; no rayon symbol is entered. `parity_threads` parity tests
  assert bit-equal results across `num_threads ∈ {None, 1, 2, 4, 8}`
  on every paradigm. CLI gains `vernier eval --threads N`.
- **`bench-timings` Cargo feature** (#256) — atomic `(par_iter,
  serial_post)` split + `build_*_anns` call counter on
  `evaluate_with_parallel`, attributed via the new `BenchCounterSet`
  shared helper (#258). Off by default and stripped from the shipped
  wheel; powers the bbox-scaling attribution at
  `docs/engineering/benchmarking/2026-05-bbox-cdf.md`.
- **`mimalloc-global` Cargo feature** on `vernier-ffi` (#256) —
  allocator A/B knob, off by default; lets users opt into mimalloc
  for hot-allocation workloads without it being a default cost.
- **Semantic divan microbench** (#261) —
  `crates/vernier-semantic/benches/accumulate_confusion.rs`
  exercises three input distributions (`realistic_perfect`,
  `realistic_jittered`, `uniform_random`) at the val2017
  panoptic-semantic geometry; prereq for the chunked-u8 kernel work.

### Changed

- **bbox AP perf** (#256, #258, #259) — KernelScratch per-worker
  annotation pool + direct-write parallel runner (replaces the
  per-image `Vec<CellOutput>` intermediate with `par_chunks_mut`);
  in-place image-major → canonical transpose via cycle-following
  (eliminates a 26 MB intermediate buffer pair on val2017); the
  `eval_imgs` + `eval_imgs_meta` transposes fuse into a single
  cycle walk (halves index arithmetic, drops one of two 1.6 MB
  visited-bitset allocations). Net val2017 nt=4: par_iter region
  42 → 32 ms, serial_post 45 → 19 ms, peak working-set
  −24 MB. The remaining Amdahl floor on `--num-threads` for bbox is
  the ~200 ms single-threaded `dataset_build` (HashMap validation in
  `CocoDataset::from_parts`), attributed via `bench-timings`.
- **Panoptic PQ perf** (#260) — sparse-remap adjacent-pixel cache on
  `build_dense_intersections` and `build_dense_boundary_intersections`.
  COCO panoptic always hits the sparse branch (RGB-packed ids exceed
  the 1 M dense cap) and panoptic segments are spatially contiguous,
  so consecutive `(g, d)` pairs are usually identical; a 4-state
  `(last_g, last_d, last_gi, last_di)` cache skips the `FxHashMap`
  lookup on adjacent-pixel matches. Dense branch is deliberately
  uncached (`Vec::get` is cheap enough that the miss overhead
  regresses synthetic by ~70%). SSSE3 RGB→u32 pack on the panoptic
  PNG decode path. New `coco_like_rgb` microbench arm exercises the
  sparse-RGB path that the existing `coco_like` arms missed
  (their ids 1..=50 took the dense path).
- **Semantic mIoU perf** (#261) — decode buffer pool + chunked u8
  kernel on `accumulate_confusion` for the `T = u8` PNG fused-decode
  path that drives `Semantic — mIoU (val2017)`. The pool reuses the
  per-image decode `Vec<u8>` across submissions; the chunked kernel
  keeps the strict-mode u64-additive fold but processes pixels in
  cache-line-sized batches.
- **Background-evaluator threading wired** (#253, #254) —
  `BackgroundConfig.num_threads` is no longer hardcoded `None` on the
  panoptic and semantic FFI ctors; `BackgroundCapable` gains a
  default-method `apply_update_parallel` that the panoptic and
  semantic streaming impls override. Panoptic `submit_png` defers
  PNG decode into the worker pool (`PyBackedBytes` zero-copy) so
  libpng decode parallelises across submissions; the single-threaded
  path keeps inline decode and is byte-for-byte unchanged.
- **`vernier-pixel-pack` folded into `vernier-panoptic`** — the
  SSSE3 RGB→u32 pack primitive added in #260 lived briefly as a
  standalone workspace crate. With a single consumer
  (`vernier-panoptic::decode`) and 172 LOC, it sat below the
  leaf-crate threshold and the audited-unsafe carveout fits cleanly
  inside the host crate (`#![deny(unsafe_code)]` at root, module-local
  `#[allow(unsafe_code)]` on the SSSE3 `pshufb` fn). Folding it
  back keeps the published crate set at the six 0.0.4 crates and
  avoids the registry-reservations + Trusted-Publisher loop in the
  release runbook for a non-reusable internal SIMD primitive.
- **Bench harness `--num-threads`** (#251, #252) — `bench run
  --num-threads "1,2,4,8"` override overrides the workload's pinned
  `num_threads` tuple; panoptic + semantic spawn helpers now forward
  the flag (previously dropped, so every panoptic / semantic cell
  ran with `args.num_threads = None` regardless of what the CLI
  swept).
- **Bench page** refreshed against `3a509df6c525` on the same
  `37652a58e939` fingerprint as the 0.0.4 snapshot, so the speedup
  deltas are not confounded by host change. Per-cell movements
  (vernier median, 0.0.4 → HEAD):
  - **panoptic** PQ: 12.59 s → **10.53 s** (−16.4%; speedup
    2.73× → **3.30×** vs panopticapi). IQR also narrows from 21.22%
    to 9.78% (still over the 5% gate — PNG decode is chronically
    noisy on this host).
  - **semantic** mIoU val2017: 5.00 s → **2.82 s** (−43.6%;
    speedup 4.12× → **7.40×** vs mmsegmentation).
  - **instance** bbox / segm / boundary / keypoints / synth-semantic /
    LVIS move within VPS noise of their 0.0.4 numbers; speedups
    widen by 0.1×–0.5× as baselines drift slightly slower on this
    run.

### Fixed

- **`bench run --impl all` on non-instance paradigms** — `impls_for_iou`
  raised `KeyError` for the paradigm-specific impls
  (`vernier_panoptic`, `panopticapi`, `mmsegmentation`,
  `vernier_lvis`, `lvis-api`) that #252 widened `ALL_IMPLS` to
  include. Falls back to an empty IoU set for impls that aren't
  registered for the instance paradigm.

## [0.0.4] — 2026-05-16

Robustness follow-up to 0.0.3. No new evaluation paradigms or kernel
changes — this release widens the typed-error surface, adds a fuzz
harness, and pins the platform-compat matrix in CI. The cross-paradigm
benchmark page is refreshed against the post-0.0.3 SHA on a single
fingerprint (no more dual-SHA LVIS caveat).

### Added

- **Typed Python error surface** (#249) — four new `PyValueError`
  subclasses (`InvalidAnnotationError`, `NonFiniteError`,
  `DimensionMismatchError`, `InvalidConfigError`) with the public
  surface pinned by `tests/python/test_error_matrix.py` and documented
  at `docs/reference/errors.md`. Previously these all surfaced as bare
  `ValueError`; existing `except ValueError:` catches still match the
  new subclasses.
- **Fuzz harness** (#249) — `tools/fuzz/` cargo-fuzz targets for the
  COCO / manifest / RLE / segmentation parsers (non-workspace crate so
  the nightly cargo-fuzz toolchain stays out of the publishable
  workspace). The `vernier_core::fuzz_regressions` integration test
  replays minimised crashes on every `cargo nextest run`; CI's
  `slow.yml` carries a 120 s/target smoke that builds once and exits.
- **Platform-compat matrix** (#249) — `slow.yml` adds a
  py3.10 × py3.13 × py3.14 ladder crossed with numpy / torch combos,
  exercising the `BackgroundEvaluator` tutorial end-to-end. Catches
  ABI / DLPack regressions that the single-version `ci.yml` matrix
  doesn't surface.
- **`bench-histogram` Cargo feature** (#249) — opt-in `(G, D, wall_ns)`
  per-call recorder on `match_image`, off by default and stripped from
  the shipped wheel. Powers the 10× val2017 scaling proof at
  `docs/engineering/matching-scaling.md`. Gated on
  `vernier-core` / `vernier-ffi` / `vernier-mask`; no production cost.
- **Stress-matrix workloads** (#249) — 6 named regimes
  (`coco-baseline`, `detr-output`, `lvis-crowded`, `open-images-cats`,
  `satellite-4k`, `pathology-8k`) plus per-axis sweeps in
  `bench/workloads/stress_matrix.py`; runner at
  `bench/runners/stress_runner.py`. Catalogue and expected behaviour
  per axis in `docs/engineering/stress-matrix.md`.
- **Memory-under-training-load runner** (#249) —
  `bench/bench/runners/memory_bench.py` (reuses
  `bench.harness.rss.RSSSampler`); methodology and reading guide at
  `docs/engineering/memory-under-training.md`.
- **Colab smoke notebook** (#249) — free-tier platform-check entry
  point; README badge links to it.

### Changed

- **Tutorial smoke** now ingests via the DLPack array path —
  `fake_model(image_ids) -> list[Detections]` returning numpy arrays
  submitted batch-mode (matches torchvision's detection-API
  convention). Notebook cell-3 stays byte-identical to the `.py` body
  modulo the module docstring and `__main__` guard.
- **Bench page** refreshed against the AMD EPYC-Milan host on a fresh
  machine fingerprint (`37652a58e939`). Speedups hold within VPS
  variance; absolute medians shift by ±3% versus the 0.0.3 snapshot.
  The dual-SHA LVIS caveat retires — every section now lives at the
  same SHA / fingerprint. The panoptic and synthetic-semantic cells
  exceeded the 5% relative-IQR gate (chronically noisy on this host —
  PNG decode dominates panoptic wall time, mmseg synthetic sits at the
  noise floor at 200-image scale); flagged inline with `*` per the
  renderer's existing convention.

## [0.0.3] — 2026-05-15

This is the diagnostic-surfaces and scenario-slicing release: instance
gains an oLRP error decomposition (Oksuz et al.), a detection-family
calibration summarizer (ECE / MCE / reliability), and a manifest-driven
slice-and-aggregate lane that runs one matching pass across N scenario
cells. Panoptic picks up boundary PQ. No paradigm shifts, no
crates.io additions — every kernel slots into the existing
vernier-core / vernier-panoptic / vernier-semantic surface.

### Added

- **LRP / oLRP error decomposition** (ADR-0043, ADR-0044, ADR-0045) —
  Oksuz et al. (ECCV 2018 / TPAMI 2021) Localization Recall Precision
  as an opt-in metric alongside AP. `vernier.instance.optimal_lrp(gt,
  dt, iou=Bbox()|Segm()|Boundary()|Keypoints())` decomposes detection
  performance into `oLRP_Loc + oLRP_FP + oLRP_FN`, minimised over a
  per-class confidence threshold `tau`. CLI gains `--metric {ap,olrp}`
  with `ap` preserving the existing headline-table contract. The Rust
  core lives in `crates/vernier-core/src/lrp/`; the ADR-0005 firewall
  is held (no edits to `matching.rs` / `accumulate.rs` / `evaluate.rs`).
  Pure-NumPy oracle is the correctness contract (ADR-0043);
  `kemaloksuz/LRP-Error` is an opt-in tripwire, not a parity gate.
  `vernier.panoptic.optimal_lrp` is a typed `NotImplementedError` stub
  — panoptic predictions carry no per-segment score so the tau sweep
  has nothing to scan; extension is a follow-up ADR.
- **Boundary Panoptic Quality** (ADR-0025 §Z1/Z2 amendment) —
  `PanopticEvaluator(boundary=True, dilation_ratio=0.02)` now ships
  under both `parity_mode="strict"` (bit-exact reproduction of
  `bowenc0221/boundary-iou-api`'s `coco_panoptic_api/evaluation.py`
  at SHA `37d25586a677`) and `parity_mode="corrected"` (deterministic,
  snapshot-based; segment-id-sorted iteration). Composition is
  `iou = min(mask_iou, boundary_iou)` — identical to the instance
  Boundary case (the prior Q3 row of `boundary-iou-quirks.md` had
  miscalled this; corrected in the same amendment). FN/FP attribution
  is unchanged; U6/U7/V1-V7/W1/W7 stand. The streaming runner threads
  boundary state per image with `BoundaryScratch` reuse, and
  distributed-eval partials hash the `dilation_ratio` into
  `params_hash` so silent boundary/instance partial mixing is rejected
  at envelope-validation time. No `FORMAT_VERSION` bump. Cityscapes
  panoptic (Z3) remains deferred.
- **Detection-family calibration summarizer** (ADR-0018) —
  ECE / MCE / reliability table for bbox / segm / boundary /
  keypoints. Opt-in via `Evaluator.evaluate(..., calibration=True)`;
  the lazy `result.calibration(iou=..., n_bins=15,
  binning="quantile", min_score=0.05, per_class=False, ...)` re-fold
  returns a `vernier.calibration.CalibrationResult` (polars
  `reliability` / `per_class` plus scalar `ece` / `mce`). Re-folding
  with different params does not re-run matching. Streaming pairing:
  `BackgroundEvaluator.finalize_with_cells()` plus the
  `vernier.calibration.StreamingSnapshot` wrapper. Clean-room NumPy
  oracle is the correctness contract; 16/16 parity bit-equal at
  strict mode. Panoptic and semantic calibration are deferred
  (data-model prerequisites per the ADR's per-paradigm shape map).
- **Slice-and-aggregate** (ADR-0046) — manifest-driven scenario
  slicing across all three paradigms plus the `vernier aggregate`
  fan-in verb. Python:
  `Evaluator.evaluate(..., manifest=..., cross_axes=...)` accepts a
  dict, JSON / CSV path, or Arrow PyCapsule manifest and returns
  `EvalResult.slices` as a polars `DataFrame` (one row per
  `(axis, value)` cell). CLI: `vernier eval --manifest weather.json
  [--cross weather,time_of_day] [--label NAME] [--metric {ap,olrp}]`
  emits a v2 envelope; un-partitioned `vernier eval` keeps emitting
  v1 verbatim. `vernier.aggregate(results, manifest, *,
  baseline=None, metric=None)` and `vernier aggregate result1.json
  result2.json --manifest runs.json --baseline clean` fan N runs
  into a comparative table with `<metric>` (mPC) and
  `<metric>__rpc` (rPC) columns when `--baseline` is set. The
  `tables=` + `manifest=` cross product is a deliberate non-feature
  with a client-side recipe at
  [`docs/how-to/per-class-by-slice.md`](docs/how-to/per-class-by-slice.md).
  New reference schemas: [`manifest-schema.md`](docs/reference/manifest-schema.md),
  [`aggregate-schema.md`](docs/reference/aggregate-schema.md).

## [0.0.2] — 2026-05-12

This is the three-paradigm release: instance gains panoptic and semantic
siblings, distributed eval lands across all three, and the bench harness
brings real-model + alternatives numbers to the docs site. Two new
crates ship to crates.io (`vernier-panoptic`, `vernier-semantic`) plus
the `vernier-partial` leaf that holds the shared partial wire envelope.

### Added

- **Distributed-eval entry points on `Evaluator`** (ADR-0035) — each
  paradigm's public `Evaluator` gains
  `evaluate_to_partial(..., *, rank_id) -> bytes` and a
  classmethod `from_partials(...) -> Summary`. Per-paradigm shapes:
  instance takes JSON bytes, semantic takes `Dataset/Predictions`,
  panoptic takes per-image tuples + `categories=` (the one
  asymmetry — `PanopticDataset` doesn't yet expose per-image
  accessors; closing that gap is a follow-up). The streaming
  substrate, the `vernier-partial` wire format, `FORMAT_VERSION`,
  partition-disjointness invariant, and the five paradigm-shared
  `Partial*` exception classes are all unchanged. The same DDP
  recipe works on instance, semantic, and panoptic.
- **Distributed evaluation wire format** (ADR-0031, ADR-0032) — new
  `vernier-partial` workspace crate holds the shared partial-envelope
  (magic + `FORMAT_VERSION` + framing + the five `Partial*` typed
  errors) used by all three paradigms. `FORMAT_VERSION` is a 1→2
  hard break (pre-1.0 policy). Cross-paradigm merge is structurally
  rejected (paradigm tag in the envelope). Determinism contract is
  paradigm-specific: instance preserves bit-exactness, semantic
  preserves it for any partition, panoptic only when the partition
  order matches the original GT order. `BackgroundEvaluator` reuses
  the same substrate via `finalize_to_partial`.

### Changed

- **Public-surface consolidation** (ADR-0035, supersedes the public
  `StreamingEvaluator` portion of ADR-0013; amends ADR-0014, ADR-0031,
  ADR-0032). Each paradigm now exposes two classes: `Evaluator`
  (frozen config dataclass; batch + DDP entry points) and
  `BackgroundEvaluator` (in-training entry point; `submit` /
  `finalize` / `finalize_with_tables` / `finalize_to_partial` /
  context manager). The streaming pyclasses are removed from Python
  entirely; the Rust substrate stays and is reachable via new
  PyO3 functions (`evaluate_*_to_partial`, `merge_*_partials`) and
  via `BackgroundEvaluator`.

### Removed

- `vernier.{instance,panoptic,semantic}.StreamingEvaluator` — the
  three streaming pyclasses are removed from Python entirely. They no
  longer appear on `vernier._core`, on any paradigm namespace, or
  under a `vernier._impl` shim. The Rust streaming substrate
  (`vernier_core::stream::StreamingEvaluator<K>`,
  `StreamingPanopticEvaluator`, `StreamingSemanticEvaluator`) remains
  as the implementation behind the new
  `evaluate_*_to_partial` / `merge_*_partials` PyO3 functions and
  `BackgroundEvaluator`'s worker. No public deprecation shim — pre-1.0
  hard break.
- `Evaluator.stream(...)` factory on `vernier.{panoptic,semantic}` —
  removed alongside the public streaming class. Use
  `BackgroundEvaluator(...)` directly, or `Evaluator.evaluate_to_partial`
  / `Evaluator.from_partials` for DDP.
- `StreamingEvaluator.snapshot(running=True)` and its Rust-side
  `snapshot_running()` method — the biased fast path that ADR-0013
  itself flagged as inappropriate for quality gates.
- `StreamingEvaluator.checkpoint()` / `restore()` — these were
  `NotImplemented` thin wrappers around `snapshot_to_partial` /
  `from_partials`. The persistence story is now exclusively
  `evaluate_to_partial` → store bytes → `from_partials` on resume.
- `BackgroundEvaluator.snapshot()`, `snapshot(peek=True)`,
  `snapshot_with_tables()`, and the non-finalize `to_partial()` on
  all three paradigms. Public surface is the consuming
  `finalize` / `finalize_with_tables` / `finalize_to_partial` only.
- `BackgroundPanopticEvaluator.from_partials` /
  `BackgroundSemanticEvaluator.from_partials` — vestigial (return-
  type bug carried them; no caller used them).

- **Semantic-segmentation user docs** (ADR-0028 PR-B10) — three new
  pages in `docs/`: `migrate/from-mmsegmentation.md` (semantic-side
  migration recipe with preset / streaming / NaN-vs-0.0 /
  binary-mask coverage), and `explanation/three-paradigms.md` (paradigm
  picker — when to reach for instance vs panoptic vs semantic, why
  they're sibling submodules rather than a single evaluator with a
  knob). README updated to
  feature the three-paradigm surface in a top-level section
  alongside the install commands; `mkdocs.yml` nav surfaces both
  new pages plus the previously-orphaned panoptic migration
  guide.
- **Semantic-segmentation streaming evaluator** (ADR-0028 PR-B9
  partial — streaming only; Breakdown / result-tables follow-ups
  scoped to a future PR). New
  `vernier_semantic::StreamingSemanticEvaluator` is a flat
  `O(n_classes²)` accumulator over `ConfusionMatrix`: `update(image_id,
  gt, dt)` folds via the same `accumulate_confusion` kernel the batch
  path uses; `snapshot()` is constant-time relative to image count
  (per ADR-0013, no fast-vs-running mode distinction needed). FFI
  pyclass `vernier._core.StreamingSemanticEvaluator` is registered
  on the module; the Python `Evaluator.stream(n_classes,
  ignore_label=None)` factory returns a fresh streaming evaluator
  carrying the parent's `parity_mode`. **Load-bearing invariant**
  (pinned by `tests/python/test_semantic_streaming.py::test_streaming_finalize_bit_equals_batch_evaluate`):
  streaming `finalize()` is bit-equal to batch `evaluate(...)` over
  the same images on f64 outputs. 10 new Python tests + 7 new Rust
  tests; total workspace 472 Rust + 376 Python tests pass.
- **Semantic-segmentation Python wrapper + per-dataset presets**
  (ADR-0028 PR-B5) — new `vernier.semantic` submodule (per ADR-0029)
  exposing `Dataset` / `Predictions` / `Evaluator` frozen dataclasses
  plus `Summary` / `ClassSemanticStats` / `ConfusionMatrix`
  re-exports of the FFI pyclasses (under their unprefixed names).
  `Dataset.from_arrays` and `Predictions.from_arrays` accept any
  unsigned-integer dtype; the wrapper preserves the input dtype and
  the FFI/kernel walks at native dtype (since ADR-0037).
  `Dataset.from_files` / `Predictions.from_files` decode single-
  channel PNG label maps via lazy-imported Pillow (raises a
  structured `ImportError` if Pillow is missing); RGB-encoded panoptic
  PNGs are rejected with a typed message pointing at
  `vernier.panoptic.Dataset`. `Predictions.from_binary_masks`
  implements the **AN2** per-class binary-mask merge with explicit
  `merge ∈ {"argmax", "first", "highest_class_id"}` selector and
  `unlabeled_class` parameter (quirks **AN3**, **AN4**). Per-dataset
  presets `Dataset.cityscapes` / `ade20k` / `pascal_voc` bake the
  canonical `(n_classes, ignore_label)` constants from
  `vernier_semantic::parity::*`. 23 new Python tests cover the
  wrapper round-trip, dtype handling, ignore-label / label-remap
  propagation, binary-mask merge rules, RGB rejection, and
  end-to-end PNG decode + evaluate.
- **Semantic-segmentation FFI surface** (ADR-0028 PR-B4) —
  `vernier._core.evaluate_semantic_from_arrays(gt_label_maps,
  dt_label_maps, n_classes, parity_mode, *, ignore_label=None,
  label_remap=None)` is the load-bearing pyfunction that drives the
  Rust kernel + summarize pass under `py.detach` (ADR-0006). Inputs
  are dicts mapping image_id (int) → 2-D `numpy.ndarray` of dtype
  `uint32`. New pyclasses `SemanticSummary`, `ClassSemanticStats`,
  `ConfusionMatrix` expose the per-class and global metrics; the
  confusion matrix is materialized as a 2-D `numpy.uint64` array
  via `ConfusionMatrix.counts()` (ADR-0028 §F1 first-class output).
  GT image-id ordering is sorted for deterministic accumulation
  (quirk **AM5** aligned). `label_remap` is pre-applied to DT
  buffers at the FFI boundary (quirk **AK2**) so the hot kernel
  loop avoids per-pixel dict lookups. PNG-decode (`from_files`) and
  binary-mask (`from_binary_masks`) variants land in PR-B5
  alongside the per-dataset preset constructors that drive them.
  14 Python smoke tests pass; full workspace 465 Rust + 343 Python
  green.
- **Semantic-segmentation kernel + summarize** (ADR-0028 PR-B3) —
  `vernier_semantic::kernel::accumulate_confusion` per-image
  histogram fold (one pass over flattened `(H, W)` slices into a
  `u64` `(n_classes, n_classes)` matrix; ignore-label mask before
  the bincount per quirk **AJ2**; out-of-range DT silent-skip per
  **AI4** strict-MS path). `ConfusionMatrix` is a flat-`Vec<u64>`
  row-major shape that doubles as the FFI `(N, N)` numpy-view
  source. `vernier_semantic::summarize::summarize` derives the
  seven headline outputs (mIoU, FWIoU, pixel accuracy, mean
  accuracy, per-class IoU/accuracy/precision, plus the confusion
  matrix as a first-class output per **AL8**). `parity_mode`
  selects NaN vs. 0.0 for zero-support per-class entries (quirk
  **AL2**); means skip zero-support classes regardless of mode
  (**AL3**, mirroring panopticapi **W2** and LVIS **AB3**). 16
  unit tests (kernel + summarize) on hand-computed fixtures, all
  pass in `--release` and debug. No SIMD per ADR-0028 §"Numerical
  layout" — the kernel is integer/memory-bandwidth bound. Dataset
  constructors and FFI surface land in PR-B5 / PR-B4 respectively.
- **Semantic-segmentation crate scaffold** (ADR-0028 PR-B2) — new
  workspace member `crates/vernier-semantic/` with `Cargo.toml` /
  `lib.rs` / `error.rs` / `parity.rs`. Re-exports
  `vernier_core::parity::ParityMode` per ADR-0028 §"Workspace and
  dependency direction" — the first dep-edge asymmetry vs.
  `vernier-panoptic ⊥ vernier-core`, justified by concrete reuse.
  Pins the per-dataset ignore-label conventions
  (`CITYSCAPES_IGNORE_LABEL=255`, `ADE20K_IGNORE_LABEL=0`,
  `PASCAL_VOC_IGNORE_LABEL=255`), class counts, and
  `SEMANTIC_PARITY_EPS` placeholder. `SemanticError` enum surfaces
  the corrected-disposition rows (AI3, AI4, AI6, AM1, AJ5) at the
  dataset-constructor boundary. The kernel, summarize, dataset, and
  FFI surfaces land in subsequent PRs (PR-B3..PR-B5).
- **User-parametrizable evaluation grids** (ADR-0040, ADR-0041,
  ADR-0042) — each paradigm's `Evaluator` accepts a structured
  config object describing the slice/aggregation surface (instance:
  IoU thresholds, area buckets, max-dets, per-class filter; semantic:
  `class_filter` + `class_grouping`; panoptic: things/stuff split
  override + per-class filter). The instance kernel grew the
  `Breakdown` axis abstraction (ADR-0039 Phase 2A/2B): area buckets,
  class groups, and CategoryFilter compose orthogonally; cells whose
  combined `(IoU × area × class)` shape is empty short-circuit. New
  `InvalidEvalParams` exception hierarchy (ADR-0039 Phase 1) replaces
  the prior assert-based validation with structured Python errors
  pointing at the offending param. Defaults reproduce the canonical
  COCO / LVIS / panoptic / mIoU shapes — opt in only when you need
  custom slicing.
- **TIDE error decomposition** (ADR-0022, weeks 1–5) — new
  `vernier.instance.tide` module returning the canonical six-bin
  decomposition (`Cls`, `Loc`, `Both`, `Dupe`, `Bkg`, `Miss`) for
  bbox / segm / boundary IoU kinds. Public Python surface +
  debugging tutorial under `docs/explanation/`; per-image
  confusion-matrix capability; FP-IoU histogram extractor + CLI for
  cross-model `t_b` ratification rounds. Validated against a numpy
  oracle on six synthetic fixtures; rf-detr-anchored real-model
  harness drives the COCO val2017 cross-check. The bbox `t_b` row
  in ADR-0022 is ratified; segm + boundary `t_b` rows remain
  tentative (ratification pending the val2017 cross-model run).
- **Result tables — opt-in Arrow surface** (ADR-0019, ADR-0038) —
  `Evaluator.evaluate(..., tables=[...])` materializes per-detection,
  per-pair, and per-class rows as zero-copy Arrow record batches
  (PyArrow / pandas / polars zero-copy import). Tables stream out of
  `BackgroundEvaluator` and `StreamingEvaluator` as well. Per-class
  tables now extend to panoptic and semantic (ADR-0038): both
  paradigms expose `per_class` schemas via the same
  `RequestedTables` mechanism.
- **Compressed-RLE + 2-D bitmask ingest on `Detections.rles`**
  (ADR-0030) — instance accepts pycocotools-encoded compressed-RLE
  bytes and 2-D `numpy.bool_`/`uint8` bitmasks alongside the
  pre-existing decoded-RLE tuple form. The FFI path is zero-copy
  via DLPack for array inputs, decoded-once for byte inputs, and
  routes through `vernier_mask::Rle::from_*` constructors so all
  three forms produce the same `Rle` representation downstream.
- **Semantic uint8 / uint16 / uint32 ingest + fused PNG decode**
  (ADR-0037) — `Dataset.from_arrays` / `Predictions.from_arrays`
  accept any of the three unsigned-integer dtypes; the kernel walks
  at the input dtype, so uint8 from a torch tensor avoids the 4×
  upcast earlier wheel versions paid. New `Evaluator.evaluate_from_pngs`
  fuses libpng decode + label-map fold in Rust under `py.detach`,
  eliminating the per-image NumPy round-trip; `submit_png` is the
  matching streaming entry point on `BackgroundEvaluator`. The
  semantic kernel was generalized over a `ClassId` trait so the
  fused path is monomorphized per dtype.
- **`BackgroundEvaluator` generalized to semantic + panoptic**
  (ADR-0014 follow-up) — the in-training entry point now exists for
  all three paradigms with the same shape (`submit` / `finalize` /
  `finalize_with_tables` / `finalize_to_partial` / context manager,
  bounded queue + worker-thread offload + soft-warn memory budget).
- **Parsed-once `Dataset` handle** (ADR-0020) — the GT cache
  produced by parsing `Dataset.from_coco_json` is retained as an
  opaque PyO3 handle that subsequent `Evaluator.evaluate(...)`
  calls reuse without re-parsing. `EvalGrid` keeps the `Dataset`
  alive across the `tables=` second pass, removing the prior
  double-parse.
- **`vernier-bench` cross-paradigm benchmark harness** (ADR-0017,
  ADR-0033) — multi-paradigm runner orchestrates vernier vs.
  pycocotools / faster-coco-eval / lvis-api / panopticapi /
  mmsegmentation across a workload ladder (synthetic, COCO val2017
  perfect-DT, mask-space jittered, real-model). `compare` /
  `report` / `bench-sync` subcommands; release-mode pin, machine
  fingerprint, IQR gate; bbox-IoU histogram dump on shutdown for
  Stage-0 instrumentation; `--with-images` cache for inference
  harnesses. Numbers feed `docs/comparison.md` and
  `docs/benchmarks.md`.
- **LVIS bench oracle** (ADR-0026 + ADR-0033) — `lvis-api 0.5.3` is
  wired as the federated-AP strict-tier oracle alongside
  pycocotools, with `Frequency`-aware K-axis cells.
- **Semantic bench oracle** (ADR-0036) — mmsegmentation is wired as
  the semantic strict-tier oracle (vernier-only cells published; the
  ADE20K/mmseg parity gate remains externally blocked).

### Changed (BREAKING — pre-1.0)

- **Per-paradigm namespace restructure** (ADR-0029) — the public
  Python surface splits across submodules: AP-fold types live under
  `vernier.instance` (`Evaluator`, `Bbox`, `Segm`, `Boundary`,
  `Keypoints`, `IouKind`, `Summary`, `EvalResult`, `Dataset`,
  `StreamingEvaluator`, `BackgroundEvaluator`, the TIDE / table /
  confusion-matrix surface, and the FFI exception classes); panoptic
  types live under `vernier.panoptic` (`Evaluator`, `Dataset`,
  `Predictions`, `Summary`, `ClassPanopticStats` — note the dropped
  `Panoptic` prefix on the unqualified type names). The cross-paradigm
  shared types (`ParityMode`, `Frequency`) and the pycocotools
  migration shim (`COCOeval`, `patch_pycocotools`) stay at the root.
  Per ADR-0029 §B1, no flat-root re-exports for moved symbols —
  `from vernier import Evaluator` raises `ImportError`.

### Added

- **Panoptic-quality (PQ) evaluation** (ADR-0025) — new sibling
  workspace crate `vernier-panoptic`, parallel to `vernier-core`,
  for the third leg of COCO evaluation
  (Kirillov et al. 2019, arXiv:1801.00868). Surface:
  `PanopticEvaluator(parity_mode='corrected', things_stuff_split=True)`
  with `.evaluate(gt: PanopticDataset, dt: PanopticPredictions)`
  returning a `PanopticSummary` (global PQ/SQ/RQ + things/stuff
  buckets + per-class rows). `PanopticDataset.from_arrays` and
  `PanopticPredictions.from_arrays` accept dicts of uint32 label
  maps + JSON segments_info; both run S1/S7/S11 validation and
  S3 PNG-marginal area recompute on the DT side. Single-threaded
  per ADR-0006 + X1 corrected disposition (bypasses panopticapi's
  multiprocessing pool entirely). `PanopticEvaluator(boundary=True)`
  raises `NotImplementedError` pointing at the Q3/Z1 follow-up
  ADR. ADR-0005 invariant preserved: zero edits to
  `crates/vernier-core/`; the firewall is structural (the new
  crate has no edge to vernier-core).
- **Panoptic parity oracle** — `cocodataset/panopticapi` vendored at
  SHA `7bb4655548f98f3fedc07bf37e9040a992b054b0` under
  `tests/python/parity_panoptic/oracle/panopticapi/`; pinned
  constants in `crates/vernier-panoptic/src/parity.rs`. Strict-mode
  bit-equality on the All/Things/Stuff + per-class shape is verified
  against `pq_compute_single_core(proc_id=0)` by `just
  test-parity-panoptic`. The multi-process pool is bypassed
  intentionally (X1 corrected; multi-process traces match under
  `Aligned` only, with `PANOPTIC_PARITY_EPS` placeholder
  `1e-9` until Q6 val measurement lands).
- **Migration guide** — `docs/explanation/panoptic-migration.md`
  covers the API mapping (`pq_compute` -> `PanopticEvaluator`),
  things/stuff semantics, sentinel divergence (panoptic `0.0` vs
  LVIS `-1.0`), single-vs-multi-process tolerance gotcha, and the
  boundary-PQ deferral.
- **LVIS federated evaluation** (ADR-0026) — long-tail benchmark
  support landed as modules in `vernier-core`. `Dataset.from_lvis_json`
  loads per-image `pos`/`neg`/`not_exhaustive_category_ids` and
  per-category `frequency`; the orchestrator's federated cell-skip
  (AA4) + `dt_ignore` extension (AA3) flow above the locked spine
  (ADR-0005, `matching.rs` and `accumulate.rs` unchanged).
  `Accumulated.summarize_lvis(dataset)` returns the canonical 13-entry
  plan (`AP`, `AP50`, `AP75`, `APs/m/l`, `APr/c/f`, `AR@300`,
  `ARs/m/l@300`); `CategoryFilter::{All, Frequency, ByIds}` is the
  K-axis subset selector behind it. `CocoDetections::lvis_trim`
  reproduces `LVISResults.limit_dets_per_image` (per-image top-K
  across all categories, AC2). `Frequency` enum (`r`/`c`/`f`) is the
  Python-facing tag.
- **LVIS parity oracle** — `lvis==0.5.3` vendored at
  `tests/python/parity_lvis/oracle/lvis_api/`; pinned constants in
  `crates/vernier-core/src/lvis_parity.rs`. Strict-mode bit-equality
  on the 13-entry summary against `LVISEval` is verified by
  `just test-parity-lvis-val`.
- **Migration guide** — `docs/explanation/lvis-migration.md` covers
  the silent-federated-semantics gotcha, the AF6 sentinel
  cross-reference (LVIS `-1` vs panoptic `0` vs uninitialized `nan`),
  and the explicit `max_dets=300` requirement.

### Performance

- **Bbox-IoU**: `pulp` `Arch` dispatch hoisted out of the inner loop
  + bool-mask prefilter; small-cell fast path bypasses dispatch
  overhead entirely.
- **Boundary**: bbox-cropped erode + bbox-cropped XOR scan skip
  per-mask full-image work; u64-packed row pass eliminates the
  strided gather/scatter; band derivation skips prefilter-empty
  rows/cols; per-image `BoundaryGtCache` + scratch reuse.
  COCO val2017 perfect-DT: 21.4s → 3.1s (-85.5%).
- **Segm**: bbox + area + offsets fused into a single counts walk;
  `SegmGtCache` + scratch reuse; `SegmentTable` + offset-based
  intersect closes the boundary regression; sparse-table AND-fold
  for `min_filter_binary`; single-pass band derivation skips the
  RLE round-trip; XOR fused into the segment scan.
  COCO val2017 segm: -21%.
- **Panoptic**: streaming runner + `FxHash` on per-image hashmaps
  drops val2017 perfect-DT 85.6s / 21.17 GiB → 32.3s / 127 MiB
  (now 1.11× faster than panopticapi); `submit_png` fuses libpng
  decode + RGB→id + S3 area recompute in Rust; row-streamed libpng
  output; thread-local DT-lookup scratch; `SegmentLookup` dispatch
  hoisted out of `decode_dt`; dense intersection matrix in
  `pq_image_with_id`.
- **Semantic**: per-image streaming through the FFI; deduped
  `parity_mode` parser; `evaluate_from_pngs` is the bench runner's
  default path.
- **FFI**: zero-copy GT/DT bytes via `PyBackedBytes` across
  `py.detach`; per-cell results boxed so `EvalGrid` skips the
  prior 268 MB zero-init; scratch buffers for cell-level + per-area
  gathers; `from_inputs` HashMap built off the GIL; `Dataset` /
  `CocoDetections` retained on `PyEvalGrid` so the `tables=` path
  skips a double-parse.

### Fixed

- **LVIS GT area filter** (quirk AG6) — strict mode now mirrors the
  oracle's `area > 0` ground-truth filter, eliminating a 0.06%
  divergence on two categories at the federated K=168 / K=817
  cells.
- **Panoptic `isthing` ingest** — tolerates `int 0/1` and numpy
  `int64` in `segments_info` (previously required `bool`).
- **sdist** — `LICENSE-APACHE` and `LICENSE-MIT` are included in
  the source distribution.

## [0.0.1] — 2026-04-30

First release with code. The placeholder 0.0.0 reservations on
crates.io and PyPI exposed no public API; 0.0.1 is the first wheel and
crate set that ship the evaluator.

### Added

- **Bbox parity** with `pycocotools==2.0.11` — strict-mode byte-equality
  on `evaluate()` / `accumulate()` / `summarize()` over COCO val2017,
  via the `EvalKernel` trait (ADR-0005) and the IoU-type-agnostic
  matching engine (ADR-0004).
- **Segm parity** — COCO RLE codec, polygon rasterizer, and mask ops in
  the leaf `vernier-mask` crate (ADR-0009); RLE bbox-IoU prefilter for
  the typical-pair speedup (quirk I1).
- **Boundary IoU** — bowenc0221 `boundary-iou-api` is the strict-mode
  oracle; `--dilation-ratio` selects band thickness (ADR-0010,
  isolated subsystem).
- **OKS keypoints** — per-category sigmas via `IouKind::Keypoints`
  (ADR-0012); `kpt_oks_sigmas` does not leak across `iouType`s; kp
  10-stat summarizer plan with `maxDets = [20]`.
- **Generalized `Breakdown` axis** (ADR-0016) — `Breakdown { axis,
  buckets }` lifts the hard-coded small/medium/large area buckets;
  closed-on-both-ends `contains` per quirk D6.
- **`StreamingEvaluator`** (ADR-0013) — push-batches surface for
  out-of-core inference workloads; bounded memory, snapshot/finalize.
- **`BackgroundEvaluator`** (ADR-0014) — async wrapper that offloads
  matching/accumulation onto a worker thread with backpressure and a
  soft-warn memory budget.
- **`patch_pycocotools()` shim** — replaces `pycocotools.cocoeval.COCOeval`
  in `sys.modules` so existing user code transparently exercises
  vernier (ADR-0007).
- **`Evaluator` extended-API class** — Rust-native builder surface
  exposing strict / aligned / corrected parity modes (ADR-0002).
- **`COCOeval` drop-in class** — faithful replication of the `Params`
  mutability pattern; `iouType ∈ {bbox, segm, boundary, keypoints}`.
- **`vernier-cli`** — `vernier eval` workspace binary (ADR-0015) with
  text + JSON v1 emit formatters; strict-mode stdout byte-equal to
  `COCOeval(...).summarize()`. Schema version is independent of the
  package version.
- **Stable-Rust SIMD via `pulp`** (ADR-0003) — runtime CPU-feature
  dispatch on bbox / boundary IoU inner loops.
- **Quirk survey** — `docs/engineering/pycocotools-quirks.md` enumerates
  61 quirks (A1–L8) with three-tier dispositions; cited verbatim
  throughout the codebase.
- **Parity fixtures** — minimal per-quirk fixtures plus full COCO
  val2017 perfect-DT smoke for bbox, segm, boundary, and keypoints.

[Unreleased]: https://github.com/NoeFontana/vernier/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/NoeFontana/vernier/compare/v0.0.4...v0.1.0
[0.0.4]: https://github.com/NoeFontana/vernier/compare/v0.0.3...v0.0.4
[0.0.3]: https://github.com/NoeFontana/vernier/compare/v0.0.2...v0.0.3
[0.0.2]: https://github.com/NoeFontana/vernier/compare/v0.0.1...v0.0.2
[0.0.1]: https://github.com/NoeFontana/vernier/releases/tag/v0.0.1
