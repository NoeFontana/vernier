# ADR-0056: Two direct detection-ingest routes for in-process callers

- **Status:** proposed
- **Date:** 2026-09-17
- **Deciders:** @NoeFontana
- **Consulted:** —
- **Informed:** —

## Context and problem statement

vernier has two detection-ingest routes today. ADR-0054 made the JSON
route fast; ADR-0030 added a columnar `Detections` dict of per-image
arrays. Neither serves the caller who already holds a Python list of
COCO *result* dicts — which is every pycocotools user, every
TorchMetrics `MeanAveragePrecision` user, and the `COCOeval` shim
itself.

That caller pays for a round trip through text. `python/vernier/_compat.py:260`
does `json.dumps(dt_anns, default=_json_default).encode()` and hands the
bytes to the JSON parser, which rebuilds the objects Python had already
built. Measured on 500 000 detections (see below), serializing costs
977 ms and parsing the result a further 385 ms, against 163 ms to read
the same dicts directly. The competitors this route is benchmarked
against do not pay it.

RF-DETR shows the other half of the problem. `src/rfdetr/training/coco_map.py:542`
already drives `evaluate_bbox_grid` with arrays to avoid building
per-detection dicts at all — but it has to marshal them into the
ADR-0030 per-image dict shape, which means one Python dict and four
array slices per image. A detector that already has one big
`(N, 7)` tensor of results has nowhere to put it.

## Decision drivers

- The routes must be *indistinguishable downstream*. A second way to
  express the same detections is a second place for parity to drift.
- Extraction cannot release the GIL — it reads Python objects. It is a
  serial floor, so it must be short, and everything after it must be
  detached.
- Validation must refuse, never repair. A route that exists to avoid a
  copy must not hide one, and a route that takes ids as float64 must not
  truncate them.
- ADR-0030 payloads must keep working untouched.

## Considered options

1. **Keep the `json.dumps` round trip.** Zero new surface; the list
   caller keeps paying 8x.
2. **A Python-side pre-pass** building the ADR-0030 columnar dicts from
   the list before calling in. Moves the per-annotation cost into
   Python, where it is more expensive, not less.
3. **Two new routes in the FFI**, both terminating in the existing
   `Vec<DetectionInput>`.

## Decision outcome

Chosen option: **Option 3**, with the convergence point as the load-bearing
detail.

Neither route introduces a new column layout, a new constructor, or a
new semantics. Both produce `Vec<DetectionInput>` — the exact type
`serde_json` produces from a results file — and both hand it to
`CocoDetections::from_inputs`. **There is no downstream to distinguish,
because after `crates/vernier-ffi/src/result_ingest.rs` there is no
route: there is one vector.** The `loadRes` semantics are therefore
inherited, not restated: id assignment (quirk **J1**), area derivation
(**J3**) and the forced non-crowd flag (**E2**/**J4**) all continue to
happen in exactly one place, `crates/vernier-core/src/dataset.rs::from_inputs`.

This is what keeps the parity surface from widening. A route that
re-implemented `ann['id'] = i + 1` would be a second implementation of
J1 to keep in step; a route that produces the input vector and stops
cannot drift from a rule it does not contain.

### The list route

`result_ingest::ann_dicts_to_inputs` reads eight keys — `image_id`,
`category_id`, `bbox`, `score`, `id`, `segmentation`, `keypoints`,
`num_keypoints` — with PyO3's `intern!`, hoisted out of the loop. The
claim that fresh key strings dominate per-annotation cost was checked,
not assumed: an otherwise identical build using plain `&str` keys runs
the same 500 000 annotations in **271.4 ms against 162.9 ms**, a 1.67x
difference, while the matrix route in the same two builds is unchanged
(123.5 vs 124.6 ms). The delta is dict-lookup-specific.

`area` and `iscrowd` are accepted and dropped, per J3 and E2/J4.

### The matrix route

An `(N, 7)` C-contiguous float64 array: `image_id, x, y, w, h, score,
category_id`. dtype, contiguity and column count are enforced by the
existing DLPack helper, which names `np.ascontiguousarray` as the fix
rather than copying silently — a hidden copy is the cost this route
exists to avoid.

`image_id` and `category_id` arrive as float64 and are **round-trip
checked**: a value that is non-finite, has a fractional part, or exceeds
2^53 is rejected. Truncating would turn `category_id=3.5` into category
3 and a 2^60 image id into a neighbouring image's detections — changing
the evaluated dataset with no diagnostic. Beyond 2^53 the caller is told
to use the list or JSON route, which carry exact integers.

### Dispatch

`boxes` (ADR-0030, columnar) is tested before `bbox` (per-annotation),
so an existing payload can never be re-routed. The matrix is probed via
`__dlpack_device__` *before* the sequence branch, because NumPy arrays
satisfy the `Sequence` protocol and would otherwise be iterated row by
row.

### Consequences

- **Positive.** Measured on 500 000 detections with the shipped release
  profile (no `target-cpu=native`), against an empty-detection baseline
  so the constant downstream cost is subtracted: the list route is
  **7.9-8.4x** faster than the `json.dumps` + parse path it replaces for
  in-process callers (166 ms vs 1311 ms), and **1.7-2.4x** faster than
  parsing pre-serialized bytes. The matrix route is **2.2-3.1x** faster
  than pre-serialized bytes and **1.3x** faster than the list route, and
  is the only route that touches no per-detection Python object. The
  ranges are run-to-run spread on the JSON baseline, which is
  memory-bandwidth bound; the two new routes are stable to ~3 %.
  Peak RSS (VmHWM delta) on the list route is **126 MiB** against
  **1040 MiB** for the bytes route, which must hold the serialized text
  and the parsed objects at once.
- **Negative.** Three input shapes to document and keep in parity. This
  is mitigated structurally (one convergence point) and by test, not by
  convention.
- **Neutral.** The matrix route cannot express segmentations or
  keypoints; it is a bbox-only fast path by construction. Callers with
  masks use the list or columnar route.

## Verification

`tests/python/test_ingest_route_equivalence.py` asserts that the list
and matrix routes produce `EvalGrid.eval_imgs()` **cell for cell
identical** to the file route's for the same logical detections, and
identical 12-stat summaries. `eval_imgs` is the assertion target rather
than the summary because it carries `dtIds`, `dtScores`, `dtMatches`
and `dtIgnore` per cell — it pins the columns, including the J1 id
assignment, which is observable only because `gtm[tind, m] = d['id']`
writes detection ids into the match arrays. The detection fixture is
deliberately unsorted by score and interleaved across images, so a route
that reordered — and therefore renumbered under J1 — would fail.
