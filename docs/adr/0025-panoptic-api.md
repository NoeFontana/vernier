# ADR-0025: Add panoptic-quality evaluation as a sibling crate

- **Status:** accepted (amended 2026-05-14: Z1 + Z2 shipped — see §Z1/Z2 amendment)
- **Date:** 2026-05-03
- **Ratified:** 2026-05-03 (PRs #122–#128, ADR-0025 panoptic rollout)
- **Deciders:** @NoeFontana
- **Consulted:** —
- **Informed:** all contributors

## Context and problem statement

Panoptic segmentation is the third leg of COCO evaluation (after instance
detection and keypoints) and a real user need for robotics / AV
perception teams that consume things-and-stuff outputs from one network
head. Through 0.0.1 we deferred it: `THIRD_PARTY_NOTICES.md` carries
`panopticapi` as "out of scope" and `boundary-iou-quirks.md` quirk Q3
deferred boundary PQ "if and when vernier extends to panoptic." This
ADR is that extension; the boundary-PQ track (Q3 / Z1 / Z2) shipped
under the §Z1/Z2 amendment (2026-05-14) below.

Two facts shape the integration. **Panoptic eval does not share the AP
spine.** Matching is constrained one-to-one on `IoU > 0.5` with
categorical agreement (Kirillov et al. 2019, Eq. 1) — no score-ranked
greedy matcher, no T/R/A/M axes, no per-detection score gradient. The
ADR-0005 accumulator, the `Similarity` trait, and the matching engine
all apply to a different problem. **Panoptic GT is a different data
model**: a per-image PNG label map (RGB-encoded ids:
`id = R + 256·G + 256²·B`) plus a `segments_info` JSON of
`{id, category_id, iscrowd}`. Nothing in `vernier-core` reads PNG
today, and `CocoDataset` cannot represent label maps without warping.

This ADR triggers ADR-0001 §"Affect the public API", §"Cross the FFI
boundary", and §"Add or remove a top-level dependency" (`png` crate;
`panopticapi` test-only).

## Decision drivers

- **ADR-0005 invariant.** No edits to `matching.rs`, `accumulate.rs`,
  or the `Similarity` trait. Phase 2 (segm) was the first test the
  lock held across IoU types; this is the first test it holds across
  evaluation paradigms.
- **ADR-0009 leaf direction.** Instance-PQ works on `uint32` label
  maps and reuses no mask primitives; the boundary-PQ track (Q3 / Z2,
  shipped under §Z1/Z2 amendment) consumes the same erosion
  primitives in `vernier-mask` that the instance Boundary AP and
  TIDE rows depend on. The `vernier-panoptic → vernier-mask` edge
  was declared up front for exactly this reason.
- **ADR-0002 parity discipline, applied to a different oracle.**
  panopticapi is to PQ what pycocotools is to AP. The strict /
  aligned / corrected tier model transplants; the disposition table
  and harness do not — same argument ADR-0010 made for boundary IoU.
- **No `IouKind` sprawl.** ADR-0011 made `IouKind` a discriminated
  union over kernels sharing the AP fold. PQ does not share that
  fold; a `Panoptic` arm would force every `match self.iou:` site
  to grow a dummy arm.

## Considered options

| Axis | Options (chosen in **bold**) |
|---|---|
| A. Workspace placement | A1 module in `vernier-core`. **A2 new crate `vernier-panoptic`.** A3 separate package. A4 defer further. |
| B. Public-API integration | B1 `IouKind::Panoptic` arm. **B2 sibling `PanopticEvaluator`.** B3 top-level function. |
| C. Input format | C1 PNG paths only. C2 arrays only. **C3 both via two `from_*` constructors.** |
| D. Parity oracle | **D1 `panopticapi` pinned + vendored.** D2 NumPy reference only. D3 panopticapi + NumPy sidecar. |
| E. PNG decode dep | E1 `image` umbrella. **E2 `png` only.** E3 no native decode. |
| F. Boundary PQ | F1 ship together. **F2 instance PQ now; Q3 follow-up.** |

## Decision outcome

**Chosen: A2 + B2 + C3 + D1 + E2 + F2.**

### Workspace and dependency direction

A new workspace crate `crates/vernier-panoptic/` with edges:

- `vernier-panoptic → vernier-mask` (declared per ADR-0009; load-bearing
  for boundary-PQ, shipped under §Z1/Z2 amendment).
- `vernier-panoptic ⊥ vernier-core` (no edge in either direction).
- `vernier-ffi → vernier-panoptic`.

The two evaluation crates are siblings sharing `vernier-mask` as a leaf.
Crate publishes to crates.io as `vernier-panoptic` with independent
SemVer; the wheel pins a compatible range. Reservation pattern follows
`docs/engineering/registry-reservations.md`.

### Public Python surface

```python
@dataclass(frozen=True, slots=True)
class PanopticEvaluator:
    """Panoptic-quality evaluator (ADR-0025).

    Sibling to :class:`Evaluator`. Per category, ``PQ_c`` is computed
    directly as ``iou_c / (TP_c + 0.5*FP_c + 0.5*FN_c)`` (panopticapi
    form, quirk W1) — algebraically equal to ``SQ_c * RQ_c`` but f64
    non-associative, so the direct form is what holds bit-equality.
    Global ``PQ`` / ``SQ`` / ``RQ`` are unweighted means over the
    present-categories subset (W2, W3, W7); things / stuff buckets
    are independent unweighted means over their subsets (W4).
    """

    parity_mode: ParityMode = "corrected"
    things_stuff_split: bool = True

    def evaluate(self, gt: PanopticDataset, dt: PanopticPredictions) -> PanopticSummary: ...


@dataclass(frozen=True, slots=True)
class PanopticDataset:
    """Panoptic ground truth.

    - :meth:`from_files` — JSON path + PNG dir; Rust decodes via ``png``.
    - :meth:`from_arrays` — pre-decoded ``uint32`` label maps + segments_info.
    """
    @classmethod
    def from_files(cls, json_path: str | Path, png_dir: str | Path) -> Self: ...
    @classmethod
    def from_arrays(
        cls,
        label_maps: Mapping[ImageId, np.ndarray],
        segments_info: Mapping[ImageId, Sequence[SegmentInfo]],
        categories: Sequence[CategoryMeta],
    ) -> Self: ...


@dataclass(frozen=True, slots=True)
class PanopticPredictions:
    """Sibling shape to PanopticDataset; predictions never carry a
    taxonomy (categories are GT-side only, quirk S9), so neither
    constructor accepts ``categories``."""
    @classmethod
    def from_files(cls, json_path: str | Path, png_dir: str | Path) -> Self: ...
    @classmethod
    def from_arrays(
        cls,
        label_maps: Mapping[ImageId, np.ndarray],
        segments_info: Mapping[ImageId, Sequence[SegmentInfo]],
    ) -> Self: ...


@dataclass(frozen=True, slots=True)
class SegmentInfo:
    id: int
    category_id: int
    iscrowd: bool = False


@dataclass(frozen=True, slots=True)
class ClassPanopticStats:
    """Strict superset of panopticapi's per-class shape (which is just
    {pq, sq, rq}, quirk W8). Count fields are vernier-only."""
    pq: float
    sq: float
    rq: float
    n_tp: int
    n_fp: int
    n_fn: int


@dataclass(frozen=True, slots=True)
class PanopticSummary:
    pq: float
    sq: float
    rq: float
    pq_things: float | None  # things/stuff buckets None when split=False
    sq_things: float | None
    rq_things: float | None
    pq_stuff: float | None
    sq_stuff: float | None
    rq_stuff: float | None
    # ParityMode::Strict drops count fields in to_dict() to match the
    # panopticapi per_class shape (Y3).
    per_class: Mapping[int, ClassPanopticStats]
```

`PanopticEvaluator` is **not** added to `IouKind`; the user surface is
"AP" or "PQ" — choice of evaluator class, not kernel arm. `PanopticSummary`
is a sibling type to `Summary`, not a subtype (ADR-0019 precedent for
`EvalResult` vs `Summary`). `patch_pycocotools` (ADR-0007) is **not**
extended; a future `patch_panopticapi` is its own decision.

### FFI surface

```rust
fn evaluate_panoptic_from_files(
    gt_json: &str, gt_png_dir: &str,
    dt_json: &str, dt_png_dir: &str,
    parity_mode: &str, things_stuff_split: bool,
) -> PyResult<PanopticSummaryFfi> { ... }

fn evaluate_panoptic_from_arrays<'py>(
    py: Python<'py>,
    gt_label_maps: Bound<'py, PyDict>,    // ImageId -> uint32 ndarray
    gt_segments_info: &str,                // JSON-serialized
    dt_label_maps: Bound<'py, PyDict>,
    dt_segments_info: &str,
    categories: &str,                      // JSON-serialized; GT-only per S9
    parity_mode: &str, things_stuff_split: bool,
) -> PyResult<PanopticSummaryFfi> { ... }
```

Both drop the GIL via `py.detach` (ADR-0006) and run single-threaded
through Phase 5. The `from_arrays` path is zero-copy via
`numpy::PyArrayLike1<u32>`.

### Parity strategy

- **Vendored, pinned oracle.** `panopticapi` at a frozen commit SHA
  vendored at `tests/python/parity_panoptic/oracle/panopticapi/` with
  `VENDORING.md` recording the SHA, the Pillow version (the
  load-bearing dep — Pillow's PNG decoder determines the decoded byte
  order; NumPy is incidental), and a fork plan.
- **Strict-mode parity claim.** Bit-equal `PQ`, `SQ`, `RQ`, and
  per-class rows on COCO panoptic val2017 against
  `panopticapi.evaluation.pq_compute_single_core` invoked **directly**
  with the full annotation list and `proc_id=0` — bypassing
  `pq_compute`'s multiprocessing pool, which has no `num_proc`
  parameter and would otherwise pin the comparison to the harness
  host's CPU count (X1, X2). Multi-process traces get bounded-ULP
  equality under `ParityMode::Aligned` only.
- **Constants module.** `crates/vernier-panoptic/src/parity.rs`
  pins `PANOPTIC_VOID = 0`, `PANOPTIC_OFFSET = 256³`,
  `PANOPTIC_IOU_THRESHOLD = 0.5` (strict inequality, U7),
  `PANOPTIC_PARITY_EPS` (aligned-mode tolerance, value pinned by
  appendix open question 6), `ORACLE_COMMIT_SHA`, `ORACLE_PILLOW_PIN`.
  Drift between this module and `VENDORING.md` is a build failure
  (mirrors `boundary_parity.rs`).
- **Disposition vocabulary.** `strict` (bit-exact) / `aligned`
  (semantics match within tolerance) / `corrected` (opt-in fix) /
  `informational` (relied-upon upstream property, no vernier behavior
  follows) / `(deferred)` qualifier (applies once the relevant
  subsystem ships). The full row table is in the appendix; the
  global `ParityMode` enum (`crates/vernier-core/src/parity.rs`)
  is reused.

The CI parity harness runs strict against `pq_compute_single_core`;
corrected-mode results are captured separately and asserted to differ
from strict only at flagged rows — a regression test for the
disposition table itself (ADR-0002 pattern).

### Numerical layout

`f64` end-to-end on the IoU and aggregation paths. The kernel is
integer-bound (intersection histogram over `uint32`; SQ is a sum of
`f64` IoUs; RQ is integer counts). Per ADR-0008: when SIMD upside
is small and parity is tight, prefer `f64`. No `pulp` dispatch on
the v1 path; SIMD lands in a follow-up under ADR-0003 if profiling
warrants.

### Algorithm

The PQ kernel is a single pass per image:

1. **Decode + validate.** GT and DT label maps as `uint32` at
   `(h, w)`. Cross-validate `(h, w)` agreement at the FFI boundary
   (corrected over panopticapi's silent broadcast error, R4).
2. **Build segment dicts.** `{segment_id → (category_id, iscrowd)}`,
   rejecting duplicate ids (corrected over panopticapi's silent
   last-wins, S7). DT areas come from PNG marginals (S3); GT areas
   come from JSON, with a corrected-mode recompute that warns on
   disagreement (S4).
3. **Histogram.** GT × DT intersection in one pass:
   `HashMap<(u32, u32), u32>` populated by
   `for (g, d) in zip(gt.flat, dt.flat): hist[(g, d)] += 1`. The
   aligned-mode replacement for panopticapi's `np.unique` over the
   uint64 OFFSET encoding (T1, T2): bit-equal output, no uint64
   materialization.
4. **Match, then attribute.** For each `(g, d)` with non-zero
   intersection: skip if `g ∉ gt_segms` (catches `g == VOID`, U2),
   `d ∉ pred_segms` (U3), `gt_segms[g].iscrowd` (U4 — crowd GT
   cannot be matched as TP), or category mismatch (U5). Compute
   `union = gt_area + dt_area − intersection − hist[(VOID, d)]`
   (panopticapi union, U6) and `iou = intersection / union` in f64
   (U8). If `iou > 0.5` (strict, U7), record a TP (U9 ensures
   at-most-one-match, so iteration order is irrelevant).
   Each unmatched non-crowd GT contributes `+1` to FN (V1);
   unmatched crowd GT is excluded from FN and recorded in
   `crowd_labels_dict[category_id]` (V2). Vernier sums overlaps
   from all same-category crowds (corrected, V3); strict mode
   reproduces last-wins. Each unmatched prediction is excluded
   from FP if `hist[(VOID, d)] + same_cat_crowd_overlap > 0.5 *
   dt_area` (strict greater-than, V4); otherwise it contributes
   `+1` to the **predicted** category's FP (V7).
5. **Aggregate.** Per category: `(sum_iou, n_TP, n_FP, n_FN)`. At
   summarize: `PQ_c = sum_iou_c / (TP_c + 0.5*FP_c + 0.5*FN_c)`
   computed directly (W1; **not** `SQ_c × RQ_c` — f64
   non-associative). Global PQ / SQ / RQ are unweighted means over
   the present-categories subset (W2, W3, W7); empty filters return
   zeros rather than crash with `ZeroDivisionError` (corrected, W6).
   Things / stuff buckets are independent unweighted means (W4).

The histogram is the only non-obvious step. Two fixtures pin its
behavior under fuzzy alignment, including the case where one DT
overlaps two GT segments (one wins by IoU; the other becomes FN).

## Consequences

- **Positive.** PQ ships behind the same parity discipline as AP and
  boundary IoU: vendored oracle, three-tier disposition, strict-mode
  bit-equality on the canonical val workload. ADR-0005's invariant
  survives across paradigms by not being asked to absorb PQ. The
  ADR-0009 edge `vernier-panoptic → vernier-mask` is declared up
  front so boundary-PQ (Z2) can land without a cross-crate refactor.
  Robotics / AV teams get one tool instead of two.
- **Negative.** New workspace crate (lints, CI, release matrix,
  crates.io presence). New top-level Rust dep (`png`, ~150 KB).
  New test-only Python deps (`panopticapi` + `Pillow`). New parity
  surface with its own governance lifecycle. Doubles the "how to
  evaluate" docs surface. The `PanopticEvaluator` / `Evaluator`
  split is a real ergonomic cost — accepted in exchange for not
  conflating two paradigms under one type.
- **Neutral.** `IouKind` stays AP-shaped. `Summary` and
  `PanopticSummary` are siblings, not variants. `patch_pycocotools`
  does not grow. The `f64` end-to-end choice is the same one
  ADR-0008 set for bbox.

## What this ADR explicitly does *not* decide

- **Boundary PQ (Q3 / appendix Z1, Z2).** Disposed under §Z1/Z2
  amendment (2026-05-14): composition rule is `iou = min(mask_iou,
  boundary_iou)` (identical to the instance case), computed on the
  iteratively-constructed boundary panoptic map.
  `PanopticEvaluator(boundary=True)` is functional. **Cityscapes
  panoptic (Z3)** remains deferred — different stuff-class semantics,
  separate follow-up.
- **Cityscapes panoptic, streaming PQ, `vernier panoptic` CLI,
  Arrow result tables for PQ.** Each is a separate follow-up;
  ADR-0015's verb extensibility and ADR-0019's table surface both
  apply cleanly when those land.
- **mIoU / semantic-only.** Different metric family (per-pixel
  class accuracy, not segment matching); does not belong in
  `vernier-panoptic` even though label-map inputs overlap.

## Pros and cons of the non-chosen options

- **A1** muddles `vernier-core`'s "AP eval kernel" identity and forces
  every consumer to compile `png`. **A3** fragments crates.io
  identity and wastes the `vernier-mask` shared leaf. **A4** is a
  growing adoption cost on a decided persona.
- **B1** sprawls every `match self.iou:` site. **B3** has nowhere
  to hang `parity_mode` / `things_stuff_split`; grows kwargs into
  a class anyway.
- **C1** forces every user through the file system. **C2** forces
  every user to install Pillow / OpenCV / imageio.
- **D2** loses the "matches the reference everyone cites" claim.
  **D3** doubles the oracle maintenance — defer until real-model
  validation forces it.
- **E1** adds ~5 MB compiled for formats panoptic does not use.
  **E3** breaks the CLI follow-up before it ships.
- **F1** doubles initial scope and couples to a fork composition
  rule the upstream documents poorly.

## Links and references

- ADR-0001 — Triggers §"Affect the public API", §"Cross the FFI
  boundary", §"Add or remove a top-level dependency".
- ADR-0002 — Three-tier parity model (vocabulary shared; tables not).
- ADR-0005 — `Similarity` trait and matching-engine API lock.
  Validated: PQ requires no edits to `matching.rs` / `accumulate.rs`.
- ADR-0006 — Threading model; `py.detach` at FFI entry, single-
  threaded through Phase 5.
- ADR-0008 — Bbox IoU `f64` end-to-end (precedent for U8, X2).
- ADR-0009 — `vernier-mask` leaf direction.
- ADR-0010 — Boundary IoU as an isolated subsystem (structural
  precedent: separate oracle, separate constants, shared `ParityMode`).
- ADR-0011 — `IouKind` stays AP-shaped.
- ADR-0015 — `vernier-cli` verb extensibility (CLI follow-up).
- ADR-0019 — Result tables (table follow-up).
- ADR-0024 — Defer-with-alternative-track pattern.
- `docs/engineering/boundary-iou-quirks.md` Q3 — resolved by §Z1/Z2
  amendment (2026-05-14).
- `THIRD_PARTY_NOTICES.md` — `panopticapi` moves from "out of scope"
  to "pinned-package env" on merge.
- Kirillov, He, Girshick, Rother, Dollár. *Panoptic Segmentation.*
  CVPR 2019. arXiv:1801.00868.
- `cocodataset/panopticapi` — strict-mode oracle; commit pinned in
  `tests/python/parity_panoptic/oracle/VENDORING.md`.

---

## Appendix: panopticapi quirks survey

Per-row evidence base for the parity strategy above. Each row was
discovered by reading the oracle. Disposition column uses the
vocabulary defined in *Parity strategy*. Citations: `ev` =
`panopticapi/evaluation.py`, `ut` = `panopticapi/utils.py`; ranges
like `ev:121-138` cover a contiguous block. Line numbers pinned
against `master` HEAD as of this ADR's drafting; the SHA in
`VENDORING.md` is authoritative once vendored.

### R. PNG decoding and segment-id encoding

| # | Quirk | Source | Disposition |
|---|---|---|---|
| R1 | Segment-id encoding `id = R + 256·G + 256²·B` (R is the low byte). The `IdGenerator` docstring at ut:30 prints a malformed formula; `rgb2id` itself is correct. | ut:73-78 | **strict**. |
| R2 | PNG load: `np.array(Image.open(path), dtype=np.uint32)` then `rgb2id`. PIL preserves source mode — does **not** auto-convert. Mode behavior: RGB ✓; **RGBA silently drops alpha** (`rgb2id` indexes `[:,:,0..2]` only, ut:77); **P / L crash mid-eval** with `ValueError` (`rgb2id` falls into the scalar branch on a 2-D array, ut:78). | ev:86-89 | **strict** for RGB; **corrected** for non-RGB (FFI rejects with typed error). |
| R3 | `VOID = 0` is the ignore segment id. Pixels with `id == 0` are excluded from FN (GT side) and contribute "free" overlap (V4). Hardcoded; not configurable. | ev:20 | **strict** (`PANOPTIC_VOID`). |
| R4 | No `(h, w)` validation. A size mismatch produces a NumPy broadcast error inside the OFFSET multiplication with no failing-image attribution. | ev:110 | **corrected**. Typed `ShapeMismatch { image_id, gt_shape, dt_shape }` at the FFI. |
| R5 | `pan_gt`, `pan_pred` are `uint32`; the OFFSET-combined buffer is `uint64`. Mirrored: `u32` per id, `u64` for the encoded pair. | ev:86-110 | **strict**. |
| R6 | Missing PNG → `FileNotFoundError`. No "image absent → all FN" branch. | ev:86 | **strict** (predictions-must-cover-GT, Y4). vernier surfaces `MissingPanopticImage`. |

### S. Categories and segments_info

| # | Quirk | Source | Disposition |
|---|---|---|---|
| S1 | Every non-VOID id in the prediction PNG must have a `pred['segments_info']` entry; missing → `KeyError`. (PNG → JSON direction.) | ev:96-101 | **strict**. Typed `UnknownPredSegmentId`. |
| S2 | Every prediction's `category_id` must be in the GT-side `categories`; missing → `KeyError`. | ev:104-105 | **strict**. |
| S3 | **Pred area is always overwritten from PNG** (`pred_segms[label]['area'] = label_cnt`). Any JSON `area` field is silently ignored. | ev:102 | **strict** (load-bearing for the IoU denominator). |
| S4 | **GT area is taken from JSON as-is**, never recomputed from PNG. Asymmetric with S3; no internal cross-check. | ev:132 | **strict** for parity; **corrected (default)** recomputes and emits `GtAreaMismatch` on disagreement. |
| S5 | Crowd GT signaled by `iscrowd == 1` (int, not bool). Mirrors pycocotools L4. | ev:127 | **strict**. vernier accepts `int \| bool`, stores `bool`. |
| S6 | Prediction `iscrowd` is **ignored**. Mirrors pycocotools J4. | ev:127, 152-163 | **strict**. |
| S7 | Duplicate segment ids in `segments_info` silently keep the **last** (`{el['id']: el for el in segments_info}`). | ev:91-92 | **corrected** (default rejects with `DuplicateSegmentId`); strict preserves last-wins. |
| S8 | **GT-only.** GT JSON-extras are silently kept in `gt_segms` but never accessed (matching iterates the histogram, not the dict). No symmetric validation on the GT side; only predictions get S11. | ev:91 | **strict**. |
| S9 | Categories list is taken from **GT JSON only**; `pred_json['categories']` is ignored. Mismatched taxonomy surfaces only via S2. | ev:196 | **strict**. |
| S10 | `isthing` is consulted only at aggregation (W4), not during matching. | ev:53 (read), ev:53-56 (filter) | **strict**. |
| S11 | **Pred-only, sibling to S8.** Pred JSON ids absent from the PNG raise `KeyError` (`pred_labels_set` is decremented as PNG ids are visited; non-empty leftover fails at ev:106-107). The asymmetry with S8 is metric-relevant: a missed pred entry would hide an FP or a TP. | ev:95, 103, 106-107 | **strict**. Typed `MissingPredSegmentInPng`. |

### T. Intersection histogram

| # | Quirk | Source | Disposition |
|---|---|---|---|
| T1 | `OFFSET = 256³ = 16,777,216`; combined label = `pan_gt.uint64 * OFFSET + pan_pred.uint64`. Collision-free since each id is bounded by `256³−1`. | ev:19, 110 | **strict** (`PANOPTIC_OFFSET`). |
| T2 | Per-pair intersection from `np.unique(pan_gt_pred, return_counts=True)`, decoded as `(gt, pred) = (label // OFFSET, label % OFFSET)`. | ev:110-116 | **aligned**. vernier uses a `HashMap<(u32, u32), u32>` filled in one pass over the slices — bit-equal output, no uint64 materialization. |
| T3 | Missing pairs default to zero on lookup (`gt_pred_map.get((g, p), 0)`). Consumed by U6 and V4. | ev:132, 156, 159 | **strict**. |
| T4 | The `(VOID, pred)` entry is read in two places — U6 (subtract from union) and V4 (FP-exclusion overlap). Same source, two consumers. | ev:132, 156 | **strict**. |
| T5 | `(gt, VOID)` entries don't exist (predictions cannot be VOID, S1). `(gt, 0)` rows are caught by S1 before they matter; documented for property-test design. | ev:96-101 | **strict**. |

### U. Matching rule and IoU

| # | Quirk | Source | Disposition |
|---|---|---|---|
| U1 | Matching iterates `gt_pred_map.items()`. Each non-zero-intersection pair is a candidate; U2–U5 reject. | ev:121-122 | **strict** loop shape; **aligned** iteration order (vernier sorts by `(gt, pred)` — same TP set under U9). |
| U2 | Skip if `gt_label not in gt_segms` (catches VOID, which is never in segments_info). | ev:123-124 | **strict**. |
| U3 | Skip if `pred_label not in pred_segms`. | ev:125-126 | **strict**. |
| U4 | Skip if `gt_segms[gt_label]['iscrowd'] == 1`. Crowd GT cannot be a TP. | ev:127-128 | **strict** (crowd's role is FP-exclusion, V4). |
| U5 | Skip if categories disagree. Metric definition, not optimization. | ev:129-130 | **strict**. |
| U6 | **Panoptic union**: `union = gt_area + pred_area − intersection − gt_pred_map.get((VOID, pred), 0)`. Pred pixels on VOID GT subtract from the union. | ev:132 | **strict**. Pinned in `parity.rs::compute_panoptic_union`. |
| U7 | **`iou > 0.5`** strict inequality. Equality is *not* a match. The `>` (vs `≥`) is the pivot guaranteeing at-most-one-match per GT (U9). | ev:134 | **strict** (metric-defining). |
| U8 | IoU in float64 via `intersection / union` (int64 → float64 promotion). | ev:133 | **strict** (ADR-0008 precedent). |
| U9 | Once TP, both labels go to `gt_matched` / `pred_matched`. Combined with U7, matching is order-independent. | ev:137-138 | **strict**. Property-tested by feeding fixtures in two orders. |
| U10 | TP IoU is **summed** per category (averaging happens at summarize, W1). | ev:136 | **strict**. f64 non-associative; see X2. |

### V. FP / FN attribution

| # | Quirk | Source | Disposition |
|---|---|---|---|
| V1 | Each unmatched non-crowd GT → `pq_stat[gt_cat].fn += 1`. | ev:149 | **strict**. |
| V2 | Unmatched **crowd** GT does **not** count as FN. Recorded in `crowd_labels_dict[category_id]` for V3 / V4. | ev:142-148 | **strict**. |
| V3 | Two same-category crowd GT segments on one image: `crowd_labels_dict[cat] = gt_label` overwrites — only the last is consulted. | ev:147 | **corrected**. vernier sums overlaps from all same-category crowds; strict reproduces last-wins. |
| V4 | Unmatched pred is **excluded** from FP if `void_overlap + same_cat_crowd_overlap > 0.5 * pred_area`. Strict greater-than. | ev:156-161 | **strict** (matches U7's strict-inequality direction). |
| V5 | The crowd term in V4 uses `(crowd_gt, pred)` from the same `gt_pred_map`. | ev:159 | **strict**. |
| V6 | A V4-excluded prediction contributes neither TP nor FP — *erased* from per-image accounting. | ev:162 | **strict**. |
| V7 | An unmatched-and-not-excluded pred is FP-counted against the **predicted** category, regardless of cross-category GT overlap. PQ has no cross-class confusion concept. | ev:163 | **strict**. |

### W. Aggregation

| # | Quirk | Source | Disposition |
|---|---|---|---|
| W1 | Per-category formulas as panopticapi computes them: `SQ_c = iou_c / TP_c` (or 0), `RQ_c = TP_c / (TP_c + 0.5*FP_c + 0.5*FN_c)`, **`PQ_c = iou_c / (TP_c + 0.5*FP_c + 0.5*FN_c)` directly** — *not* `SQ_c × RQ_c`. Algebraically equivalent, f64 non-associative; bit-equality requires the direct form. | ev:65-68 | **strict**. |
| W2 | Categories with `TP + FP + FN == 0` are excluded from the average (zero row in `per_class_results`, no `n` increment). | ev:61-63 | **strict**. |
| W3 | Global PQ is the **unweighted mean** of per-category PQs over the present subset. Counterintuitive on long-tailed datasets. | ev:73 | **strict**. |
| W4 | **Things/stuff split**: `pq_average(categories, isthing)` with `None` / `True` / `False`. Three independent unweighted means. Canonical report is `[("All", None), ("Things", True), ("Stuff", False)]`. | ev:49-56, 221-224 | **strict**. |
| W5 | Returned `n` is the count of contributing categories. | ev:73 | **strict**. |
| W6 | Empty filter (no category contributes): `n` stays 0 and `pq / n` **raises `ZeroDivisionError`**. There is no zero-or-NaN guard. | ev:73 | **corrected**. vernier returns `(pq=sq=rq=0.0, n=0)`; strict replays the error shape. |
| W7 | Global SQ is the mean of per-category SQs, **not** `total_iou / total_TP`. Same for RQ. | ev:73 | **strict** (metric-defining; pooling produces a different number). |
| W8 | panopticapi's `per_class_results` is `{category_id: {pq, sq, rq}}` only — *no raw counts*, no per-class pretty-printer. | ev:62, 68 | **aligned**. vernier returns `ClassPanopticStats { pq, sq, rq, n_tp, n_fp, n_fn }` — strict superset; the three shared ratios are byte-equal. `to_dict()` under strict drops the count fields. |

### X. Multiprocessing and determinism

| # | Quirk | Source | Disposition |
|---|---|---|---|
| X1 | `pq_compute_multi_core` parallelizes via `Pool(cpu_count())` over `np.array_split(annotations, cpu_count)`; per-worker `PQStat` accumulated via `__iadd__` in submission order. **`pq_compute` has no `num_proc` parameter** — the only knob is host CPU count. | ev:168-181 | **corrected**. vernier evaluates single-threaded (ADR-0006). Strict-mode parity is against `pq_compute_single_core` invoked **directly** with the full annotation list and `proc_id=0`, bypassing the pool entirely. |
| X2 | f64 IoU summation is non-associative. For *fixed* `cpu_count`, `pq_compute_multi_core` IS deterministic (gather is in submission order, not completion). Different `cpu_count` produces different last-bit `sum_iou` — same code, different host → different ULP-level PQ. | ev:29-34 (`__iadd__`), ev:178-180 (gather) | **strict** vs `pq_compute_single_core`; **aligned** under bounded-ULP tolerance for multi-process traces (tolerance pinned by open question 6). |
| X3 | Single-process fold is deterministic (Python 3.7+ insertion-ordered dicts, sorted `np.unique` output). | ev:121-122 | **strict**. |
| X4 | `pq_stat[cat]` is `defaultdict(PQStatCat)`: unseen and all-zero are indistinguishable; W2's `tp+fp+fn>0` check disambiguates. | ev:38-39, 41-42 | **strict**. |

### Y. CLI and API surface

| # | Quirk | Source | Disposition |
|---|---|---|---|
| Y1 | `pq_compute(gt_json_file, pred_json_file, gt_folder=None, pred_folder=None)`. Folder defaults to `gt_json_file.replace('.json', '')`; silent failure if convention doesn't match. | ev:184, 192-195 | **strict** for the function shape; **corrected** for the implicit folder default (`from_files` requires `png_dir` explicitly). |
| Y2 | `pq_compute` prints All/Things/Stuff via `print`. **No per-category printer.** Per-class data only via the returned dict. | ev:227-237 | **corrected** (vernier returns structured `PanopticSummary`); CLI follow-up reproduces text format under `--format text`. |
| Y3 | `pq_compute` returns `{"All": {pq, sq, rq, n}, "Things": {...}, "Stuff": {...}, "per_class": {category_id: {pq, sq, rq}}}`. **No `iou`/`tp`/`fp`/`fn` in per_class** (corrected against an earlier-draft claim). | ev:62, 68, 222-226, 242 | **aligned**. `to_dict()` under strict produces the exact panopticapi shape. |
| Y4 | Predictions must cover every GT image: `if image_id not in pred_annotations: raise Exception(...)`. No "missing pred → all FN" branch. | ev:215-216 | **strict**. Typed `MissingPredictionsForImage`. |
| Y5 | Pred-only images (in pred but not GT) are silently ignored; matching is GT-driven. Mirrors pycocotools J5. | ev:213-216 | **strict**. |
| Y6 | Errors are bare `Exception` / `KeyError` with f-string messages. Raise sites: ev:101, 105, 107, 207, 209, 216, plus the implicit ZeroDivisionError from W6. | ev:101, 105, 107, 207, 209, 216 | **corrected**. Typed `PanopticError` variants; strict replays via thin wrapper. |
| Y7 | Progress: `print(f'Core: {proc_id}, {idx} from ...')` every 100 images, plus an end-of-core line. Hardcoded format. | ev:82-83, 164 | **corrected**. No stdout from the core; structured `progress=Callable[[int, int], None]` arg on `PanopticEvaluator`. |
| Y8 | `evaluation.py` consumes only `id` (key) and `isthing` from each category. **`name` is *not* required by the eval** (no per-category printer, Y2). `color` is used by `IdGenerator` (a creation helper, not eval). | ev:53-54 | **aligned**. `CategoryMeta` requires `id` + `isthing`; extras preserved on the dataset handle, ignored by the kernel. |

### Z. Boundary PQ

Anchors the disposition and cross-references Q3 in
`boundary-iou-quirks.md`. Z1 and Z2 are shipped under the §Z1/Z2
amendment (2026-05-14); Z3 (Cityscapes panoptic) remains deferred to
a separate follow-up.

| # | Quirk | Source | Disposition |
|---|---|---|---|
| Z1 | Composition rule is `iou = min(mask_iou, boundary_iou)` — **identical** to the instance case (O1); upstream `boundary_iou/coco_panoptic_api/evaluation.py:195` at SHA `37d25586a677b043ed585f10e5c42d4e80176ea9` is literally `iou = min(iou, boundary_iou)`. The non-trivial part is the iterative, JSON-order-dependent construction of the boundary panoptic map: each segment's binary mask is read from the partially-mutated id-map and eroded; interior pixels are wiped to `BOUNDARY_ID = max(category_id) + 1`; the eroded band is painted back as the segment id. Later segments may lose pixels their predecessors' bands stomped. | bowenc0221 fork (`boundary_iou/coco_panoptic_api/evaluation.py:105-127, 195`) | **corrected** (default; snapshot-based, segment-id-sorted, deterministic) **+ strict** (bit-exact upstream in-place JSON-order mutation). See §Z1/Z2 amendment below. |
| Z2 | Boundary PQ inherits M1–M5 (erosion spec) from boundary-IoU survey: 3×3 Chebyshev (M1, M5), half-to-even rounding (M2), `dilation_ratio = 0.02` (M4), `dilation ≥ 1` clamp (M3). | boundary-iou survey | **strict**. Same `vernier-mask` primitives that the instance Boundary AP / TIDE rows consume — `crates/vernier-mask/src/ops/boundary.rs:123` for `dilation_pixels`; PR #184/185 for the u64-packed and bbox-cropped erode kernels. Shipped via §Z1/Z2 amendment. |
| Z3 | Cityscapes panoptic via the same fork. Different categories, ignore semantics, stuff-class handling. Out of scope for both Z1 and the initial ship. | cityscapes fork | **corrected (deferred)**. Separate follow-up. |

### Open questions

Each is a ~30-minute fixture; resolved by follow-up commits before the ADR is **accepted**.

1. **S4 GT-area-from-JSON in real datasets.** Audit COCO panoptic /
   Cityscapes / ADE20K's first 100 images for `area`-vs-PNG-pixel-count
   agreement. If any disagree, S4's corrected disposition becomes
   load-bearing. **Resolved 2026-05-03 (PR-6 §test_panoptic_val).** The
   perfect-DT smoke surfaces any disagreement automatically; the
   corrected disposition stands as documented.
2. **U7 strict inequality at 0.5.** Construct a fixture where
   `intersection / union` evaluates to exactly `0.5` in f64 (e.g.,
   1 / 2). Confirm vernier and panopticapi both reject. **Resolved
   2026-05-03 (PR-5 §test_q2_iou_at_exactly_half_rejected_u7).**
3. **U9 iteration-order independence.** Property-test that
   constructing three viable candidates `(g1,p1)`, `(g1,p2)`,
   `(g2,p1)` all with `iou > 0.5` is impossible (U7 forbids it),
   asserting the at-most-one-match property on random PNG pairs.
   **Resolved 2026-05-03 (PR-5 §test_q3_iter_order_independence_property).**
4. **V3 multiple same-category crowd regions.** Find / construct one
   such image; document panopticapi (last-wins) vs vernier corrected
   (sum-of-overlaps) divergence; pin strict-mode reproduction.
   **Resolved 2026-05-03 (PR-5
   §test_q4_v3_multi_same_category_crowd_strict_vs_corrected).**
5. **W7 global-SQ asymmetry.** Two fixtures: balanced (mean-of-SQ_c ≈
   total_iou / total_TP within 1e-6) and long-tailed (differ by >0.05).
   Pin strict values; the latter is the regression test catching a
   future "innocent" pooling refactor. **Resolved 2026-05-03 (PR-5
   §test_q5_w7_long_tailed_global_sq_bit_equal).**
6. **X2 single-vs-multi-process bit-equality bound and
   `PANOPTIC_PARITY_EPS`.** Run `pq_compute_single_core` and
   `pq_compute_multi_core` for `cpu_count ∈ {2, 4, 8}` on COCO
   panoptic val; measure ULP distance per category; pin
   `PANOPTIC_PARITY_EPS`. Build-time check enforces agreement with
   `VENDORING.md`. **Procedurally resolved 2026-05-03 (PR-6).** The
   measurement procedure is captured in
   `tests/python/parity_panoptic/panoptic_val_paths.py`'s module
   docstring; the placeholder `1e-9` guards aligned mode until the
   first developer provisions the cache and runs the measurement.
   Strict mode demands bit-equality vs `pq_compute_single_core`
   regardless and is unaffected.
7. **Z1 boundary-PQ composition.** *Resolved 2026-05-14 (§Z1/Z2
   amendment below).* The earlier draft's suspicion — "threshold is
   `boundary_iou > 0.5` rather than `min(mask_iou, boundary_iou) >
   0.5`" — was incorrect. Upstream
   `boundary_iou/coco_panoptic_api/evaluation.py:195` is literally
   `iou = min(iou, boundary_iou)`; composition is identical to the
   instance case (O1). The non-triviality lives in the iterative
   boundary panoptic map construction (segment-by-segment
   interior-erase + band-paint, JSON-order dependent — see §Z1/Z2
   amendment). The U6 union applies to both the mask and boundary
   IoU computations.

---

## Z1/Z2 amendment (2026-05-14)

- **Status:** Shipped 2026-05-14.
- **Scope:** Closes Z1 and Z2; flips both from `(deferred)` to live
  dispositions. Z3 (Cityscapes panoptic) remains deferred and is
  unaffected.

### Composition rule (definitive)

For each `(gt_id, pred_id)` candidate pair with category agreement
(U5) and non-zero intersection:

1. `mask_iou = mask_inter / (gt.area + pred.area − mask_inter −
   void_pred_mask)`, the standard panoptic union (U6) over the
   `pan_gt × pan_pred` confusion histogram.
2. `boundary_iou = b_inter / (gt.boundary_area + pred.boundary_area
   − b_inter − void_pred_boundary)`, the same union shape (U6)
   over the `pan_gt_boundary × pan_pred_boundary` confusion
   histogram, with `BOUNDARY_ID` treated as the boundary-map's void.
3. `iou = min(mask_iou, boundary_iou)`; match if `iou > 0.5` (strict
   inequality per U7); on match, `tp += 1` and `sum_iou += iou`
   (U9, U10).

FN / FP attribution is **unchanged** from the mask-only case — it
consumes the *mask* confusion matrix and *mask* areas only. U6 / U7
/ V1-V7 / W1 / W7 all stand verbatim. The `boundary_area` field
participates **only** in step 2.

### Boundary panoptic map construction

The boundary panoptic map is built segment-by-segment from a per-image
id-map. `BOUNDARY_ID = max(category_id) + 1` is the sentinel for
"interior" pixels that get wiped between segments. `dilation_px` is
computed per quirks M2 / M3 / M4 / M5 via `vernier-mask`'s
`dilation_pixels` (single source of truth for both modes).

**strict** (mirrors upstream `evaluation.py:105-127`, bit-exact
JSON-order in-place mutation):

```text
pan_boundary = pan.clone()                # one allocation per image
for el in segments_info (JSON order):
    binary_mask = (pan_boundary == el.id)
    binary_band = mask XOR erode_chebyshev(mask, dilation_px)
    pan_boundary[binary_mask] = BOUNDARY_ID     # wipe interior
    pan_boundary[binary_band] = el.id           # paint band
    el.boundary_area = popcount(binary_band)
```

In-place mutation means a later segment `k`'s
`(pan_boundary == k_id)` excludes pixels that earlier segments'
bands stomped on. Order-dependent by construction.

**corrected** (default; snapshot + sorted-id, deterministic):

```text
snapshot = pan.clone()                    # read-only reference
pan_boundary = filled(BOUNDARY_ID)        # never read for masks
for el in segments_info (sorted by id):
    binary_mask = (snapshot == el.id)
    binary_band = mask XOR erode_chebyshev(mask, dilation_px)
    pan_boundary[binary_band] = el.id
    el.boundary_area = popcount(binary_band)
```

Bands may overlap on output (later sorted-id segments can co-paint
into pixels an earlier segment already painted), but each segment's
`boundary_area` and intersection counts are derived from the
read-only `snapshot` and are independent of segment order. Equal to
strict whenever no two segments' bands overlap.

### Parity modes shipped

- **strict** — bit-exact reproduction of upstream
  `pq_compute_single_core(..., iou_type="boundary",
  dilation_ratio=...)` at the pinned SHA
  `37d25586a677b043ed585f10e5c42d4e80176ea9`.
- **corrected** — `PanopticEvaluator.parity_mode` default;
  deterministic under segment reordering, equal to strict when bands
  do not overlap.
- **aligned** is unused for boundary PQ — no meaningful tolerance
  band sits between strict and corrected; `strict + corrected`
  covers the space.

API surface additions:

- `PanopticEvaluator.parity_mode: ParityMode` — unchanged default
  (`"corrected"`).
- `PanopticEvaluator.dilation_ratio: float = 0.02` — new field,
  consulted only when `boundary=True`. Per quirk M4 / Q1, vernier
  exposes `dilation_ratio` at every entry point that uses boundary
  IoU.

### Citation chain

Every disposition in this amendment is grounded in a row of the two
quirks surveys, not re-litigated here:

- **Q3** (`boundary-iou-quirks.md`, rewritten 2026-05-14) — the
  composition rule and the iterative boundary panoptic map
  construction.
- **M1-M5** (`boundary-iou-quirks.md`) — erosion spec: 3×3
  Chebyshev, half-to-even rounding, `dilation ≥ 1` clamp,
  `dilation_ratio = 0.02` default. Already shipped in `vernier-mask`
  (`crates/vernier-mask/src/ops/boundary.rs:123` for
  `dilation_pixels`; PR #184 / PR #185 for the u64-packed and
  bbox-cropped erode kernels).
- **U6** — panoptic union with void-pred subtraction. Applies to
  both `mask_iou` and `boundary_iou` denominators.
- **U7** — strict-`>` threshold at `iou > 0.5`. Unchanged.
- **V1-V7** — FP / FN attribution from the mask confusion matrix.
  Unchanged — `boundary_area` does not enter V4 / V5 (V4's
  FP-exclusion overlap is mask-area-only by design).
- **W1 / W7** — per-category and global PQ / SQ / RQ aggregation.
  Unchanged — `sum_iou` consumes whatever `iou` step 3 above
  decides.
- **ADR-0010** — the instance Boundary IoU disposition baseline;
  panoptic boundary inherits its erosion primitives wholesale via
  the `vernier-panoptic → vernier-mask` edge declared in this ADR's
  *Workspace and dependency direction* section.

### Why amend instead of supersede

`CLAUDE.md` flags ADRs as immutable once `accepted` and prefers
superseding ADRs over in-place edits. The user explicitly elected
amendment here: Z1 and Z2 were always tagged `(deferred)` in the
appendix, and the ADR's *What this ADR explicitly does not decide*
section already anticipated a follow-up. Flipping a deferral
disposition in place is a status update consistent with the
original ADR's intent, not a re-decision of an accepted call. Z3
remains deferred and is not affected. *(Author's note: this is a
deliberate exception to the convention, taken once and recorded
for posterity rather than absorbed as a pattern.)*
