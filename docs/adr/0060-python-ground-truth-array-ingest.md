# ADR-0060: Columnar ground-truth ingest for in-process callers

- **Status:** proposed
- **Date:** 2026-09-18
- **Deciders:** @NoeFontana
- **Consulted:** —
- **Informed:** —

## Context and problem statement

The detection side of the ingest boundary has been worked twice.
ADR-0030 added the columnar `Detections` dict; ADR-0057 added the
list-of-result-dicts and `(N, 7)` matrix routes. The ground-truth side
has never been touched: `CocoDataset` has `from_json` and
`from_json_bytes` and nothing else.

That asymmetry has a price, and it is now measured. Benchmarking
RF-DETR-Seg-Nano predictions on COCO val2017 (5000 images, 36 781 GT
annotations, 80 categories) against four COCO-eval backends found
vernier fastest on every cell — while spending **258 ms of a 954 ms
bbox `compute()` inside `json.dumps`**, serializing a 6.3 MiB GT
document that Rust then parses in 22.8 ms. Twenty-seven percent of the
call is Python building a text representation of arrays the caller
already holds, so that the parser can rebuild the objects from the text.

The caller in question is not exotic. It is the shape every training
harness and every evaluation wrapper has: ground truth held as numpy
columns, or as a structure trivially projected to them. `torchmetrics`,
RF-DETR's `coco_map.py`, and any dataset class backed by a parquet or
`.npz` file are all in this position. The only way any of them can hand
vernier ground truth today is to write a JSON document.

This ADR triggers ADR-0001 §"Affect the public API" and
§"Cross the FFI boundary".

### Out of scope

- LVIS federated metadata. Explicitly declined; see below.
- The `evaluate_*_grid(gt_json=...)` entry points, which keep their
  bytes-only signature. The array route produces a `CocoDataset` handle,
  which those entry points' `_with_dataset` siblings already accept.
- Panoptic and semantic ground truth, which are PNG-backed and have
  their own ingest story (ADR-0025, ADR-0028).

## Decision drivers

- **The routes must be indistinguishable downstream.** A second way to
  express ground truth is a second place for the parity contract to
  drift, and ground truth is where the contract is densest: D1, D2, A4
  and E1 all read fields this route would have to carry.
- **Refuse, never repair.** A route that exists to avoid a copy must not
  hide one; a route that takes dimensions as `int64` must not truncate
  them into `u32`.
- **The caller must not build per-annotation Python dicts.** That is the
  cost being removed. ADR-0057 measured the per-annotation dict route at
  ~750 ms against ~20 ms columnar at 500 k, which settles the shape of
  the answer before any of the details.
- **Extraction cannot release the GIL** — it reads Python objects — so
  it is a serial floor and must be short, with everything after it
  detached.

## Considered options

1. **Status quo.** The caller keeps writing JSON. Zero new surface;
   keeps a 258 ms tax on the single most common call.
2. **A `from_dicts` route** taking the `images` / `annotations` /
   `categories` lists a `pycocotools.COCO` object already holds. This is
   the GT analogue of ADR-0057's list route, and it would help the
   pycocotools-shaped caller. But it is not the caller that is paying
   here — RF-DETR holds arrays — and it reinstates exactly the
   per-annotation dict cost this ADR exists to remove. Rejected as the
   *primary* surface; see "Future work".
3. **One flat GT matrix**, the mirror of ADR-0057's `(N, 7)`. Rejected
   outright: a GT document is three sections, not one table. Categories
   carry names, images carry dimensions, and annotations carry
   segmentations. None of the three is expressible as a float64 matrix,
   and the two that are partly expressible would need the other two
   passed alongside anyway.
4. **A columnar dict per section**, converging on
   `CocoDataset::from_parts`.

## Decision outcome

Chosen option: **Option 4**, with the convergence point as the
load-bearing detail — the same argument ADR-0057 made, applied to the
other side of the boundary.

`CocoDataset.from_arrays(images, annotations, categories, *,
cast_inputs=False)` produces the triple
`(Vec<ImageMeta>, Vec<CocoAnnotation>, Vec<CategoryMeta>)` and hands it
to `CocoDataset::from_parts` — the exact constructor `from_json_bytes`
ends at, with the exact types `serde_json` produces. **There is no
downstream to distinguish, because after
`crates/vernier-ffi/src/gt_ingest.rs` there is no route: there is one
`CocoDataset`.**

Every ground-truth rule is therefore *inherited*, not restated:

| Rule | Where it lives | What this route does |
| --- | --- | --- |
| **D1** — `ignore` overwritten by `iscrowd` | `dataset.rs::effective_ignore`, at eval time | fills `is_crowd` and `ignore_flag` verbatim |
| **D2** — zero visible keypoints ⇒ implicit ignore | the OKS kernel | carries `num_keypoints` |
| **A4** — GT sorted ascending by `_ignore` | the matching engine | produces input order; the engine sorts it |
| **E1** — crowd IoA denominator | `similarity/bbox.rs::CrowdDenom` | carries `is_crowd` |
| GT `area` read verbatim (contrast **J3**) | `CocoAnnotation::area` | `area` is a **required** column |
| GT ids supplied, never assigned (contrast **J1**) | nowhere — there is no assignment | `id` is a **required** column |
| Reference integrity | `from_parts` | not re-implemented |

This is what keeps the parity surface from widening. A route that
re-derived GT area from the box would be a second implementation of a
rule that does not exist on the GT side; a route that produces the parts
and stops cannot drift from a rule it does not contain.

### Why `categories` is the one section that is not columnar

`images` and `annotations` are dicts of equal-length arrays.
`categories` is a plain sequence of small dicts (`id`, `name`, optional
`supercategory`).

`name` is a string, which has no array form, so a columnar `categories`
would be an `id` array beside two Python lists — three objects to keep
in step, and a shape no caller naturally holds. And the cost argument
does not apply: the per-annotation dict cost this ADR removes is O(N) in
annotations (36 781 on COCO val2017, 1.2 M on LVIS), while categories
are O(K) with K = 80 or 1203. Paying dict-shaped ingest for the K axis
buys a surface that matches what callers actually have —
`coco.dataset["categories"]`, verbatim — for a cost that does not appear
in any profile.

The asymmetry is deliberate and is recorded here so it is not "fixed"
later by someone matching the other two sections for symmetry's sake.

### Absent versus zero, and why it needed a rule

A columnar array has no null. Two COCO ground-truth fields are genuinely
optional *per annotation* and mean something different absent than
present-and-zero:

- **`ignore`**, because of **D1**. `CocoAnnotation::ignore_flag` is an
  `Option<bool>` precisely so `parity_mode="corrected"` can fall back to
  `iscrowd` when the field is absent while honouring a present `0`.
- **`num_keypoints`**, an `Option<u32>` for the same reason.

One rule covers both:

> An optional integer column may be passed as a **signed** array, in
> which a **negative entry means the field was absent on that
> annotation**. Omitting the column entirely means absent on every
> annotation. A `bool` / `uint8` column has no negative and therefore
> means present on every annotation.

Neither field has a meaningful negative value otherwise — `ignore` is a
flag, `num_keypoints` is a count — so the encoding is unambiguous, and
it lets the array route express *every* GT document the JSON route can,
including one that carries `ignore` on some annotations and not others.

The alternative considered was **per-column all-or-nothing**: the column
is present (all annotations have the field) or omitted (none do). It is
simpler and it is what a first draft would write. It was rejected
because it produces a **silent wrong answer** on a mixed document: the
caller fills the missing entries with `0`, and under `corrected` an
annotation that should have inherited `iscrowd=1` gets `ignore=False`
instead. The route cannot detect that, because a zero is a legal value.
The measured negative control below shows that this specific error moves
five of eleven `eval_imgs` columns under `corrected` and **zero** under
`strict` — which is to say it is exactly the class of bug that ships.

### `keypoints` is the one field this route cannot express per annotation

An `(N, K, 3)` array cannot say "absent on row `i`", and the sentinel
trick does not transfer: every float is a legal coordinate, and using
`NaN` as the marker would re-admit the value this route otherwise
refuses. So `keypoints` is all-or-nothing per document. A GT that
carries keypoints on some annotations and not others must use the JSON
route.

This is stated as a limitation rather than engineered around because the
case is not real: COCO `person_keypoints` carries `keypoints` on every
annotation, and a keypoints dataset that does not is one core already
refuses under `iou_type="keypoints"`. Adding a companion mask column to
cover it would widen the surface for a document nobody has.

### Non-finite values are refused, and that is not a divergence

`bbox`, `area` and `keypoints` are checked for finiteness, and a `NaN`
or infinity is rejected naming the annotation and the element.

This does **not** make the array route stricter than the file route in
the sense ADR-0057 rules out. JSON has no `NaN` or `Infinity` literal,
so a GT *file* cannot carry one: the check refuses input the JSON route
could never have handed downstream, rather than second-guessing input it
would have accepted. It is the same argument that justifies the `(N, 7)`
matrix route's exact-integer check on `image_id`, and it closes the
failure mode where a `NaN` box compares false against every threshold
and is read as a perfect similarity.

Image `width` / `height` arrive as `int64` — the dtype a caller's index
arrays already are — and are range-checked into `ImageMeta`'s `u32`. A
negative or oversized dimension is refused, never wrapped.

### LVIS federated metadata is explicitly not supported

`from_arrays` builds a **COCO-flat** dataset. `neg_category_ids` and
`not_exhaustive_category_ids` are per-image variable-length sets and
per-category `frequency` is a letter tag; none has a natural columnar
spelling, and inventing one (ragged arrays plus offsets) for a section
that is read once at load would be surface without a profile behind it.
LVIS callers use `from_lvis_json`.

The risk this carries is the one `from_lvis_json`'s own docstring
already names: LVIS data loaded under COCO semantics scores
systematically lower, **silently**. The mitigation is that
`is_federated` is `False` on every dataset this route produces, pinned
by test, and that the limitation is documented on the method rather than
only here.

### Dispatch, and what is *not* added

`from_arrays` is a new static constructor, not an overload of
`from_json`. There is no type-sniffing between bytes and dicts, and
therefore no GT analogue of quirk **J6**'s first-entry-decides hazard:
the caller names the route by calling it.

The `evaluate_*_grid` entry points are untouched. They take GT JSON
bytes; their `_with_dataset` siblings take the handle both routes
produce. Widening `gt_json=` to accept columns would have put a
discriminator on the hot path for no gain, since a caller using this
route wants the parsed-once handle (ADR-0020) anyway.

## Verification

`tests/python/test_gt_ingest_route_equivalence.py` asserts that the
array route produces `EvalGrid.eval_imgs()` **cell for cell identical**
to the file route's for the same logical document, **in both parity
modes**, and identical 12-stat summaries.

`eval_imgs` is the assertion target rather than the summary because it
carries `gtIds`, `gtIgnore`, `gtMatches`, `dtMatches` and `dtIgnore` per
cell — it pins the columns, including the supplied-id property and the
**A4** sort order, which are observable only because ground-truth ids
are written into the match arrays.

Two supporting levers:

- **`CocoDataset.dataset_hash`** is exposed to Python by this ADR. It is
  the ADR-0031 canonical-form fingerprint: two handles share it exactly
  when they carry the same images, categories and annotations, every
  field included, down to each segmentation's stored bytes. Because each
  section is sorted by id before hashing it is *order-independent*, so
  it pins **content** while `eval_imgs` pins **order**. That pair is
  what makes the equivalence claim whole, and it is how the segm and
  keypoints comparisons are made at all: only `bbox` has a grid-taking
  `_with_dataset` entry point today, so those two kernels are compared
  on summaries plus the hash rather than on cells.
- **A measured negative control.** Rather than asserting an unverified
  number, the suite builds one deliberately-wrong dataset per bug class
  and records how many of the eleven per-cell columns move:

  | perturbation | `strict` | `corrected` |
  | --- | --- | --- |
  | zero the `iscrowd` column | **5** | — |
  | derive `area` from the bbox | **4** | — |
  | drop the `ignore` column | **0** | **4** |
  | absent `ignore` read as `0` | **0** | **5** |
  | rotate annotation input order | **3** | — |
  | renumber GT ids `1..N` | **2** | — |

  The two zeroes are the finding. **A route that dropped `ignore`
  entirely is invisible under `strict`** — not because the fixture is
  weak, but because **D1** specifies that the field is overwritten
  there. An equivalence suite that ran only the default parity mode
  would have certified that bug. Every equivalence assertion in the
  module is parametrized over both modes as a result, and building the
  table is what surfaced it.

  Two further fixture facts were measured rather than assumed. A GT area
  that disagrees with `w * h` only registers if it disagrees *across a
  bucket edge* (4 of 11 once the fixture was corrected; 0 before). And
  image `width` / `height` move nothing at all on a bbox grid — bbox
  evaluation never reads them — so they are pinned under `segm`, where
  they are the canvas a polygon is rasterized onto.

### Payload-shape matrix

Every `segmentation` spelling the JSON route accepts is accepted here,
plus the in-memory shapes ADR-0030 added, sharing
`result_ingest::extract_segmentation_with_root` with the detection list
route so **K2** and **K3** have one implementation:

| spelling | JSON route | array route | canonical form |
| --- | --- | --- | --- |
| polygons `[[x0,y0,…],…]` | yes | yes | polygon |
| `counts` as `str` | yes | yes | compressed |
| `counts` as list of ints | yes | yes | uncompressed |
| `counts` as `bytes` | no | yes | compressed |
| `counts` as `uint32` array | no | yes | uncompressed |
| 2-D `bool`/`uint8` bitmask | no | yes | uncompressed |
| absent (`None` per entry) | yes | yes | — |

The six spellings collapse to **three** stored canonical forms. The
evaluated numbers are identical across all six; `dataset_hash` agrees
within a canonical group, by design, since it is a wire-format identity
and not a pixel digest. The one divergence from the file route is in the
permissive direction and is the same superset ADR-0057 granted the
detection list route, for the same reason: the caller this route exists
for holds masks in memory, and a file never does.

A stacked `(N, H, W)` array is **refused** for the `segmentation`
column. NumPy arrays satisfy the sequence protocol, so it would
otherwise be silently iterated into `N` planes — a plausible-looking
answer from a payload the column does not accept. This is the GT
instance of the lesson ADR-0057 records about buffer-protocol objects
that merely satisfy a sequence protocol.

## Consequences

- **Positive.** Measured on COCO val2017 GT with the shipped release
  profile (no `target-cpu=native`), min of 7: see
  §"Measured" below. The `json.dumps` stage the 258 ms figure names is
  removed outright, not made faster.
- **Negative.** A second GT input shape to document and keep in parity,
  and a rule (absent-is-negative) that has no precedent on the detection
  side, because the detection side has no field where absent and zero
  differ. Both are mitigated structurally — one convergence point — and
  by the two-mode equivalence suite, not by convention.
- **Neutral.** The route is opt-in and additive; `from_json` is
  untouched. It cannot express LVIS federated metadata or per-annotation
  keypoint absence, and says so in both cases rather than producing a
  quietly different dataset.

## Measured

Shipped release profile (`[profile.release]`, `lto = "thin"`, no
`.cargo/config.toml` and therefore no `target-cpu=native`), min of 7
runs, 8-core box. All four cells are real published ground truth, not
synthetic. Times in ms.

| cell | `json.dumps` | `from_json` | **baseline** | `from_arrays` | ratio | saving |
| --- | --- | --- | --- | --- | --- | --- |
| COCO val2017, bbox-only (4.9 MiB) | 55.7 | 19.8 | 75.5 | **5.7** | **13.3x** | 69.8 |
| COCO val2017, full segm (21.5 MiB) | 345.2 | 95.5 | 440.7 | **35.0** | **12.6x** | 405.7 |
| LVIS v1 val, bbox-only (30.5 MiB) | 367.7 | 148.1 | 515.8 | **37.4** | **13.8x** | 478.4 |
| LVIS v1 val, full segm (191.9 MiB) | 3844.9 | 770.7 | 4615.6 | **294.8** | **15.7x** | 4320.8 |

COCO val2017 is 5000 images / 36 781 annotations / 80 categories; LVIS
v1 val is 19 809 / 244 707 / 1203.

The ratio is stable at **12.6–15.7x** across a 40x span of document
size, and rises with scale because `json.dumps` is superlinear in
practice where the columnar read is not.

Against the 258 ms figure that motivated this ADR: that stage is
**deleted**, not made faster. It is worth being precise that the 258 ms
was not reproduced exactly here — the closest comparable cells on this
box are 55.7 ms for a 4.9 MiB bbox-only GT and 345.2 ms for the 21.5 MiB
full document, bracketing the reported 6.3 MiB / 258 ms. The gap is
document composition and hardware, not method: `json.dumps` with
`_compat.py`'s `default=` callable was measured alongside the plain call
and is within noise of it (55.9 / 345.7 / 345.9 / 3822.3), so the
`default=` hook is not the explanation. The **ratio** is what this ADR
claims, and it holds at every scale measured.

Peak memory, per-stage VmHWM in isolated processes, COCO val2017 full
GT, over a common 186 MiB `json.load` baseline: the bytes route peaks at
**215 MiB** (+29) because it must hold the serialized text and the
parsed dataset at once; the array route peaks at **198 MiB** (+12).

**Whole-dataset canonical-form check.** Both routes' `dataset_hash`
agree byte-for-byte on the full COCO val2017 GT and on the full LVIS v1
val GT — 244 707 annotations, every polygon and every crowd RLE
included. That is an end-to-end field-for-field equality on real data,
not a fixture.

**Confidence.** The box was **contended by other agents** throughout;
1-minute load average ranged 0.69–2.06 across the runs. Min-of-7
absorbs most of that, and the ratio is consistent across four
independent cells taken at different load levels, so the ordering and
the order of magnitude are solid. The individual millisecond figures
should be read as ±10–15 % rather than as a pinned regression baseline;
promoting any of them into `docs/benchmarks.md` needs a quiet box and
the ADR-0049 CPU-budget harness.

## Links and references

- ADR-0001 — record architecture decisions (significance criteria).
- ADR-0002 — parity model (`strict` / `corrected`).
- ADR-0004 — numerical layout policy (the f64 boundary `cast_inputs`
  opts out of).
- ADR-0006 — threading model (GIL drop at every PyO3 entry).
- ADR-0020 — parsed-once `Dataset` handle, which this route constructs.
- ADR-0026 — LVIS federated evaluation (the metadata declined above).
- ADR-0030 — columnar `Detections` dict; the shape this mirrors, and the
  source of the accepted bitmask / `bytes`-counts spellings.
- ADR-0031 — partial wire format, which defines `dataset_hash`'s
  canonical form.
- ADR-0054 — split parallel JSON ingestion (what the route this replaces
  had already been optimized into).
- ADR-0057 — detection ingest routes; the template for the convergence
  argument and the source of the validation lessons applied here.
- `docs/engineering/pycocotools-quirks.md` — A4, D1, D2, E1, J1, J3,
  K2, K3.
