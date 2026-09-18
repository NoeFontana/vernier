# ADR-0057: Two direct detection-ingest routes for in-process callers

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

`iscrowd` is accepted and dropped, per E2/J4 — `DetectionInput` has
nowhere to put it. `area` is accepted and **carried**, because J3's
disposition is not "derive" but "derive *by default*": the `dt_area`
keyword selects between the derived area (`"bbox"`, the default), the
one the payload supplied (`"supplied"`) and the mask's
(`"mask"`). A results *file* carries `area` through to that switch, so
dropping it on the list route would silently re-bucket every detection
into whatever the box implies under `dt_area="supplied"` — the routes
would disagree on AP-small/medium/large for the same payload. What the
list route must not do is *interpret* it, and it does not: `from_inputs`
still owns the choice.

`segmentation` takes **every shape a results file can carry** — polygons
(`[[x0, y0, x1, y1, …], …]`, unioned under **K2**), an RLE dict whose
`counts` is the compressed `str` a file carries (**K3**) or an
uncompressed list of ints — **plus** the in-memory shapes ADR-0030 added
for `Detections.rles`: `bytes` counts, a `uint32` counts array, and a 2-D
`bool`/`uint8` bitmask. It is a superset of the file route's, because the
caller this route exists for may hold either. The columnar `rles` field
is *not* widened: it stays the array surface ADR-0030 specified, and
still refuses polygons.

The field is **optional on every `iou_type`**, exactly as it is in a
file, where `serde_json` simply leaves it `None`. What an absent
`segmentation` means under `segm`/`boundary` is quirk **J2**, decided
once in core — see below.

### The matrix route

An `(N, 7)` C-contiguous float64 array: `image_id, x, y, w, h, score,
category_id`. dtype, contiguity and column count are enforced by the
existing DLPack helper, which names `np.ascontiguousarray` — and torch's
`.contiguous()`, since not every caller of this route holds a numpy
array — as the fix rather than copying silently. A hidden copy is the
cost this route exists to avoid.

`image_id` and `category_id` arrive as float64 and are **round-trip
checked**: a value that is non-finite, has a fractional part, or exceeds
2^53 is rejected. Truncating would turn `category_id=3.5` into category
3 and a 2^60 image id into a neighbouring image's detections — changing
the evaluated dataset with no diagnostic. Beyond 2^53 the caller is told
to use the list or JSON route, which carry exact integers.

`cast_inputs=True` is honoured here on the same terms as
`Detections.boxes`: default-off is the strict ADR-0004 boundary, and the
flag is the documented opt-in that asks for the promoting copy. The one
route that takes a single array should not be the one route that ignores
the flag.

The matrix has no `id` column, so every row takes the **J1** positional
id. That is the one field the list route preserves and this one cannot
express; it is not a divergence, because a results file without `id`
fields behaves identically. Pinned by
`test_matrix_route_cannot_express_an_id_and_gets_the_j1_assignment`.

### A route carrying no segmentation under `segm` means J2

An `(N, 7)` matrix cannot carry a mask, and a result dict need not. Under
`iou_type="segm"`/`"boundary"` that is **not** a new condition to guard:
it is exactly what a *bbox-only results file* carries, and quirk **J2**
already says what it means. Under the default `strict` mode core
synthesizes pycocotools' `[[x1,y1, x1,y2, x2,y2, x2,y1]]` rectangle
(`coco.py:341`) and evaluates it; under `corrected` it refuses, naming
the detection and its image.

So `evaluate_segm_grid(gt, matrix, "strict", …)` returns mask APs
computed from boxes — deliberately, and identically to what the same
detections in a bbox-only results file have always returned. **No guard
is added at the dispatch site**, for the reason this ADR exists: a guard
there would make the matrix and list routes *stricter* than the file
route, which is precisely the downstream distinction the decision
outcome rules out. A caller who wants the diagnostic instead of the
number asks for it with `parity_mode="corrected"`, on any of the three
routes, and gets the same core error from all three.

The alternative — guarding at the FFI — was considered and rejected:
it moves one quirk's disposition into the binding layer, where the
`strict`/`corrected` switch that owns it does not reach, and it would
diverge the routes to avoid a surprise the parity contract already
documents. Keypoints needs no decision here: core's
`missing_keypoints_err` already refuses a detection without keypoints,
whichever route delivered it.

### Route selection reads the first entry (J6)

`DetectionsArg::extract` classifies a list by `dicts.first()`: whether
entry 0 carries `boxes` (columnar) or `bbox`/`category_id`
(per-annotation) decides the route for the whole list. This is quirk
**J6** — *first-entry-decides* — re-introduced at the FFI layer, and it
is recorded here rather than left implicit, because J6's disposition is
`corrected` and an unremarked re-introduction is how a corrected quirk
quietly becomes a strict one.

Two things bound it. First, the routes are disjoint by *field name*, not
by iou_type: a mixed list does not get "whatever derivation entry 0
chose", it gets a hard error from the route entry 0 picked, naming the
offending index in either direction — `detections[1].category_id:
missing required field` when entry 0 was a result annotation, and
`detections[1]: missing required field 'boxes'` when entry 0 was
columnar. The index is threaded into `extract_inputs_one` for exactly
that reason; a single *bare* dict has no position in a list and keeps
the un-indexed `detections: …` form. Second, the consequence J6
names — heterogeneous *segmentation* handling — does not arise, because
`segmentation` is read per entry, never per list: an entry with a mask
and an entry without are both legal in one list, and each is resolved on
its own under J2. What remains is a route choice, checked once instead
of N times, that cannot silently produce numbers from the wrong reading
of a payload.

Re-checking the predicate on every entry would buy a marginally better
error message for a list no supported caller produces, at a per-entry
dict probe on the hot path this ADR exists to keep short. If that trade
ever stops holding, the fix is to validate the predicate across the list
and reject disagreement — not to route per entry.

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
  The `COCOeval` drop-in is the first caller: it hands the list route
  the `cocoDt` annotations it was given, and keeps no serialized copy —
  `evaluate()` on 500 000 detections falls 3.3x with a 16 % lower peak.
- **Negative.** Three input shapes to document and keep in parity. This
  is mitigated structurally (one convergence point) and by test, not by
  convention.
- **Neutral.** The matrix route cannot express segmentations,
  keypoints or ids; it is a bbox-only fast path by construction, and
  what it *does* mean under a mask iou_type is J2 (above). Callers with
  masks use the list or columnar route. The list route carries every
  shape a results file carries, so "use the list route instead" is
  always an answer.

## Verification

`tests/python/test_ingest_route_equivalence.py` asserts that the list
and matrix routes produce `EvalGrid.eval_imgs()` **cell for cell
identical** to the file route's for the same logical detections, and
identical 12-stat summaries. `eval_imgs` is the assertion target rather
than the summary because it carries `dtIds`, `dtScores`, `dtMatches`
and `dtIgnore` per cell — it pins the columns, including the J1 id
assignment, which is observable only because `gtm[tind, m] = d['id']`
writes detection ids into the match arrays. The detection fixture is
deliberately unsorted by score and interleaved across images, and carries
**tied scores** inside one `(image, category)` cell, so a route that
reordered — and therefore renumbered under J1, or broke the stable sort's
tie differently — would fail.

The same cell-for-cell assertion covers the other three kernels:

- **segm**, file route vs list route, in all six accepted `segmentation`
  spellings — polygons, `counts` as a list of ints / `str` / `bytes` /
  `uint32` array, and a 2-D bitmask — against the file route fed the JSON
  spelling of the same masks.
- **boundary**, file vs list, on the polygon spelling.
- **keypoints**, file vs list, cell for cell and on the 10-stat summary,
  covering `keypoints` and `num_keypoints`.
- **J2**, on all three routes at once: a bbox-only payload under
  `iou_type="segm"` produces identical cells as bytes, as dicts and as an
  `(N, 7)` matrix under `strict`, and the identical core refusal under
  `corrected`.

Three negative tests pin the boundaries the routes must *not* cross: a
polygon is still refused on ADR-0030's columnar `rles`, a bad
`segmentation` names `detections[i].segmentation` (not `rles[i]`, which
is the other route's field), and an `(N, 7)` matrix under
`iou_type="keypoints"` is refused by core.
