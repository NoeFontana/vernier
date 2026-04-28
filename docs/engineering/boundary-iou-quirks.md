# boundary-iou-quirks survey

A working note (not an ADR) cataloguing the numerical and structural
quirks of `bowenc0221/boundary-iou-api` that vernier must reckon with
when implementing boundary IoU. ADR-0010 is the venue where the
disposition column below is ratified.

This survey is intentionally **independent of**
`docs/engineering/pycocotools-quirks.md`. The two documents share no
quirks, no fixtures, no parity harness, and no oracle. They share only
the `ParityMode` enum (`strict` / `aligned` / `corrected`) defined in
`crates/vernier-core/src/parity.rs`, which is the user-facing semantic
common to all of vernier's parity stories. The boundary-specific
constants live in `crates/vernier-core/src/boundary_parity.rs`:
`BOUNDARY_DILATION_RATIO_DEFAULT = 0.02`,
`BOUNDARY_PARITY_EPS = 1e-9` (IoU equality tolerance under
`ParityMode::Aligned`), `ORACLE_COMMIT_SHA`, `ORACLE_OPENCV_PIN`.

Each row below was discovered by reading the oracle line-by-line. The
disposition column is one of:

- **strict** — vernier reproduces this behavior bit-exactly.
- **aligned** — vernier matches the *semantics* but may differ in
  incidental details (e.g., separable single-pass erosion vs iterative
  3×3, which on integer binary input produce identical output).
  User-visible outputs match within a documented tolerance.
- **corrected** — vernier opts to fix this. Default behavior diverges
  from the oracle and the divergence is documented as an opinionated
  improvement.
- **informational** — pins a property of the upstream that vernier
  *relies on* but does not reproduce (e.g., the pinned OpenCV is
  itself deterministic). No vernier-side behavior follows; the row
  exists as evidence for an ADR commitment.
- A trailing `(deferred)` qualifier marks a row whose disposition
  applies only when the relevant subsystem ships; until then, vernier
  does not implement the behavior.

The disposition column is a **draft proposal** by the author of this
survey — ADR-0010 is the venue where it gets ratified or revised.

## Source reference

All line numbers refer to `bowenc0221/boundary-iou-api` at the commit
SHA pinned in `tests/python/parity_boundary/oracle/VENDORING.md`. The
oracle has had four commits in five years; this pin is unlikely to
move except for a fork.

- `boundary_iou/utils/boundary_utils.py` — `mask_to_boundary`,
  `augment_annotations_with_boundary_*`. The erosion + boundary band
  routine.
- `boundary_iou/coco_instance_api/cocoeval.py` — `COCOeval` subclass
  with `iouType="boundary"` support; the `min(mask_iou, boundary_iou)`
  composition lives here.
- `boundary_iou/lvis_instance_api/eval.py` — `LVISEval` analogue with
  the same composition and a parameterised `dilation_ratio`.
- `boundary_iou/coco_panoptic_api/evaluation.py` — panoptic boundary
  PQ. Out of scope for vernier v0.1; quirks listed here for future
  reference only.

Conventions used below:

- "bu" = `boundary_utils.py`, "ce" = `coco_instance_api/cocoeval.py`,
  "le" = `lvis_instance_api/eval.py`.
- Line citations like `bu:14` mean `boundary_utils.py:14`. Exact line
  numbers are stable as of the pinned commit. A `~` prefix
  (e.g., `ce:~270`) marks an approximate citation pointing to a
  *section* of the file rather than a specific call site — used for
  rows that cover several adjacent lines or a contiguous block.

---

## M. Erosion specification

These rows pin the structuring element, kernel size, and iteration
count of the erosion that defines the boundary band.

| # | Quirk | Source | Disposition |
|---|---|---|---|
| M1 | Erosion uses iterative `cv2.erode` with a 3×3 all-ones kernel applied `dilation` times. Mathematically equivalent to a single erosion by a `(2d+1)×(2d+1)` Chebyshev-ball structuring element. | bu:14, bu:24 | **aligned**. vernier uses single-pass van Herk / Gil-Werman separable erosion by the (2d+1)-square kernel. Bit-equal output on integer binary input; ~30× fewer passes for COCO-typical `d`. |
| M2 | Dilation in pixels: `dilation = int(round(dilation_ratio * sqrt(h² + w²)))`. Python's `round` is banker's rounding (half-to-even). | bu:13 | **strict**. Half-to-even rounding reproduced. The Rust `f64::round_ties_even` (stable since 1.77) is the matching primitive. |
| M3 | Minimum dilation clamp: `if dilation < 1: dilation = 1`. The boundary band is never thinner than 1 pixel even for tiny images. | bu:15-16 | **strict**. Same clamp, same threshold. |
| M4 | Default `dilation_ratio = 0.02` (Cheng et al. 2021 paper choice). Exposed as a constructor parameter on `LVISEval`; not exposed at the COCO call site (the COCO eval hardcodes 0.02). | bu:9, le:34 | **strict** for the value. **corrected** for the API: vernier exposes `dilation_ratio` on every entry point that uses boundary IoU, including the pycocotools-shim. |
| M5 | Structuring element is 3×3 all-ones (square / Chebyshev), **not** 3×3 cross. `kernel = np.ones((3, 3), dtype=np.uint8)`. | bu:21 | **strict**. Chebyshev ball, not Manhattan. |
| M6 | OpenCV's `cv2.erode` with a binary `np.uint8` mask and integer `iterations` is a deterministic operation in the supported OpenCV pin range. | bu:22 | **informational**. vernier does not consume `cv2.erode`; this row pins a property of the *upstream* needed to justify the frozen-commit + pinned-OpenCV oracle (E2 in ADR-0010). Bit-stable across OpenCV minor versions for binary input + 3×3 kernel + integer iterations; `cv2.distanceTransform`, which has known cross-version drift, is not used. |

## N. Padding and edge handling

These rows pin how the mask is bordered before erosion and how
edge pixels are treated.

| # | Quirk | Source | Disposition |
|---|---|---|---|
| N1 | Before erosion, the mask is zero-padded by exactly 1 pixel on all four sides via `cv2.copyMakeBorder(..., 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)`. | bu:19 | **strict**. Border-touching foreground pixels count as boundary because they are eroded against the zero pad. Dropping the pad changes the metric for any mask touching an image edge. |
| N2 | After erosion, the result is sliced back to the original `(h, w)` via `mask_erode = new_mask_erode[1:h+1, 1:w+1]`. The pad is consumed by the erosion, not preserved. | bu:23 | **strict**. Same slice. |
| N3 | Pad value is hard-zero (`value=0`). Not configurable in the reference. | bu:19 | **strict**. Pad value is part of the metric definition. |
| N4 | The pad is applied **once**, before all `dilation` iterations of 3×3 erosion. With `dilation > 1`, an iterative 3×3 erosion can "eat into" the original mask through the pad — but because the pad is zero and the kernel is min-filter, the foreground only shrinks. The single-pass (2d+1)-square decomposition produces the same output. | bu:19, bu:22 | **aligned**. vernier's single-pass impl pads once before the row erosion and the result is bit-equal to the iterative form on integer binary input. |
| N5 | Boundary band is computed as `boundary_mask = mask - mask_erode`. Subtraction on `np.uint8` underflows to wraparound on negative values — but here `mask_erode ≤ mask` everywhere because erosion is a min-filter, so subtraction is safe. | bu:25 | **aligned**. vernier computes `boundary = mask AND NOT mask_erode` (logical XOR equivalent). Output is bit-equal; the formulation avoids the latent uint8-wraparound risk. |

## O. Composition with mask IoU

These rows pin how the boundary IoU matrix is folded with the mask
IoU matrix to produce the final per-pair IoU under
`iouType="boundary"`.

| # | Quirk | Source | Disposition |
|---|---|---|---|
| O1 | `iouType="boundary"` returns the **minimum** of mask IoU and boundary IoU per pair: `ious[:, iscrowd == 0] = np.minimum(mask_ious[:, iscrowd == 0], boundary_ious[:, iscrowd == 0])`. The min is part of the metric definition, not an implementation detail. | ce:~270 | **strict**. Same min, same per-cell semantics. |
| O2 | Crowd-column asymmetry: the `np.minimum` is applied only to columns where `iscrowd == 0`. Crowd ground-truth columns retain the pure mask IoU, with no boundary contribution. | ce:~270 | **strict**. Mirrors the spirit of the pycocotools quirk **E1** (crowd asymmetry on the mask-IoU denominator). vernier reproduces both asymmetries faithfully. |
| O3 | The boundary IoU is computed via `pycocotools.mask.iou` on RLE-encoded boundary masks — the same kernel used for mask IoU. No separate boundary-only IoU formulation. | ce:~265 | **aligned**. vernier's `BoundaryIou::compute` uses the same RLE intersection primitive (`vernier_mask::Rle::intersect_area`) as `SegmIou`, with the bbox-IoU prefilter (quirk **I1**) reused canonically. Different control flow, same arithmetic. |
| O4 | DT-side `iscrowd` is irrelevant: detections never carry crowd flags in valid input. Only GT crowd flags trigger the min/no-min branch. | ce:~270 | **strict**. Mirrors quirks **E2** / **J4** in the pycocotools survey. Enforced at the dataset boundary. |

## P. Edge cases and degenerate inputs

These rows cover inputs the reference never explicitly tested for and
where vernier must define its own behaviour.

| # | Quirk | Source | Disposition |
|---|---|---|---|
| P1 | Mask too small for the structuring element: if `(2d+1) > min(h, w)`, single-pass erosion of a fully-foreground mask yields fully-zero, so the boundary band equals the original mask. The iterative reference produces the same output. | (implicit) | **strict**. Documented in vernier's `boundary_band` docstring; covered by a dedicated fixture. |
| P2 | Empty input mask (`area == 0`): `erode(empty) == empty`, so `boundary_band == empty`. IoU of two empty boundary masks is `0 / 0`. The reference relies on `pycocotools.mask.iou` to return `0.0` for an empty pair, which it does (no division by zero — the kernel short-circuits). | (implicit) | **strict**. vernier returns `0.0` for empty-vs-empty boundary IoU, matching the reference. |
| P3 | Mask filling the image (`area == h * w`): erosion shrinks it from the borders only (because of the 1-pixel zero pad — N1), so the boundary band is a frame of width `d`. Symmetric on both axes. | bu:19, bu:22 | **strict**. Border-as-boundary is intentional and metric-defining. |
| P4 | Single-pixel mask: `dilation ≥ 1` clamp (M3) means erosion of a 1-pixel mask always yields zero, so `boundary == mask`. The pair `(1px, 1px)` matching has `mask_iou == boundary_iou == 1.0`, so `min == 1.0`. | bu:15, bu:22 | **strict**. Covered by a fixture. |
| P5 | Self-intersecting polygon GT (a "bowtie"): the polygon → RLE rasterizer (pycocotools quirk **H3**) determines the input mask, then erosion proceeds normally on whatever mask resulted. Boundary IoU inherits H3's polygon-direction sensitivity transitively. | bu:19 (transitively) | **aligned**. vernier inherits its own H3 disposition for polygon rasterization; the boundary path adds nothing new. The fixture in `tests/python/parity/fixtures/self_intersecting_polygon_segm/` is reused for boundary parity by composing it with `iouType="boundary"`. |

## Q. Variants and dataset specialisations

These rows document non-default behaviour exposed by the reference for
specific datasets. vernier supports them as configuration, not as
separate code paths.

| # | Quirk | Source | Disposition |
|---|---|---|---|
| Q1 | LVIS exposes `dilation_ratio` as an `LVISEval.__init__` parameter (default 0.02). The COCO instance API does not expose it at all — the value is implicit at 0.02. | le:34 | **corrected**. vernier exposes `dilation_ratio` everywhere boundary IoU is invoked, including the pycocotools-shim and the CLI. |
| Q2 | Cityscapes instance / panoptic boundary metrics use the same `mask_to_boundary` routine with default `dilation_ratio=0.02`. No dataset-specific tuning. | (separate API files) | **strict**. Single algorithm, configurable ratio. |
| Q3 | Panoptic boundary PQ (`boundary_iou/coco_panoptic_api/`) has its own composition logic that is **not** the simple `min(mask_iou, boundary_iou)` of the instance case. Out of scope for v0.1. | (panoptic eval) | **corrected (deferred)**. A separate ADR will dispose of panoptic boundary PQ if and when vernier extends to panoptic evaluation; until that ADR lands, vernier does not implement panoptic boundary PQ at all. |

---

## Glossary cross-reference (for ADR-0010 authors and reviewers)

When ADR-0010 is reviewed, every row above must be cited by ID. The
disposition column is the author's *proposal*; the ADR is the venue
where each row is signed off. A short cheat-sheet:

- **Most rows: strict.** The bowenc0221 oracle is what every paper
  cites. Reproducing it is the only way to claim the metric.
- **A handful of aligned:** M1 (single-pass vs iterative erosion —
  bit-equal on integer input), N4 (single-pass pad equivalence), N5
  (XOR vs subtract — bit-equal, safer), O3 (shared RLE intersection
  kernel), P5 (polygon rasterisation transitive).
- **Corrected:** M4 (`dilation_ratio` exposed everywhere), Q1 (LVIS-
  style parameterisation generalised to all entry points), Q3
  (panoptic boundary PQ — deferred).
- **Informational:** M6 (upstream `cv2.erode` determinism within the
  pinned OpenCV range; vernier does not call it).

The ADR-0010 disposition table is the canonical source; this survey
is the per-row evidence base.

## Open questions

These are quirks where the reading is uncertain and we should write a
small reproducer before signing off.

1. **M2 + non-finite arithmetic.** Does `int(round(...))` ever cross
   into the regime where Python's `round` and Rust's
   `f64::round_ties_even` produce different output for the same input?
   For `dilation_ratio · √(h² + w²)` with realistic image sizes the
   answer is "no", but write a parametric fixture exercising
   half-integer cases (e.g., `h = 100`, `w = 100`,
   `dilation_ratio = 0.005` produces `dilation = 0.5√2 = 0.707...`,
   safely non-half — but we should sweep).
2. **N4 single-pass vs iterative equivalence for very large `d`.**
   For `d > min(h, w) / 2`, both approaches yield fully-eroded
   (empty) masks. Confirm that the single-pass implementation does
   not produce numerically different intermediate states under
   property tests. Adding a fixture at `dilation_ratio = 0.5` is
   sufficient.
3. **O3 RLE intersection kernel reuse.** Confirm via property tests
   that vernier's `Rle::intersect_area` produces identical output to
   `pycocotools.mask.iou` numerator (i.e., the intersection in pixel
   units) for boundary RLEs as well as full RLEs. Likely yes — the
   kernel makes no assumptions about mask connectivity — but worth
   asserting.
4. **P2 division-by-zero behaviour.** The pycocotools `mask.iou`
   kernel returns `0.0` for empty-vs-empty pairs. vernier's
   `SegmIou` does the same. Add an explicit fixture asserting
   `BoundaryIoU` returns `0.0` for empty-vs-empty (rather than
   matching `1.0` or `NaN` as some interpretations of "complete
   agreement on emptiness" might suggest). The chosen value is
   pinned by O1 (the min-fold) when both inputs are empty.
5. **Q3 panoptic composition.** When (if) vernier extends to
   panoptic boundary PQ, the composition rule differs from the
   instance case in ways the reference panoptic eval encodes but
   does not document. A future ADR is the right home for that
   investigation.

These open questions are the tail; the head is large and the
disposition table above is high-confidence. ADR-0010 ratifies the
table modulo the open questions above, which are tracked as
follow-up fixtures rather than blockers on the v0.1 ship.
