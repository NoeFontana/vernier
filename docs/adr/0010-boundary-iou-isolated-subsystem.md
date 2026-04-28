# ADR-0010: Boundary IoU as an isolated subsystem with its own oracle, quirks, and performance baseline

- **Status:** proposed
- **Date:** 2026-04-27
- **Deciders:** @NoeFontana
- **Consulted:** —
- **Informed:** all contributors

## Context and problem statement

Boundary IoU (Cheng, Girshick, Dollár, Berg, Kirillov, CVPR 2021) is a
segmentation evaluation metric that complements mask IoU by measuring
overlap on a thin band around each segmentation boundary. It is widely
cited, stable since 2021, and a real user need for perception teams
evaluating thin or fine-grained masks where mask IoU saturates. ADR-0005
explicitly anticipates it as an example `Similarity` impl.

Two facts shape the integration. First, **boundary IoU is not part of
pycocotools.** The canonical reference is `bowenc0221/boundary-iou-api`,
an unmaintained "beta" repository depending on OpenCV. That is a
different oracle from `pycocotools==2.0.11` (ADR-0002's strict-mode
oracle). Second, **the reference implementation is wall-clock
catastrophic.** It performs iterative dense 3×3 erosion repeated
`d ≈ 0.02 · √(h² + w²)` times — order 30 iterations for typical
1280×720 imagery, O(area · d) per mask, ~35 000 masks for one
COCO val2017 run.

The naïve integration extends `pycocotools-quirks.md` with new rows,
places `BoundaryIou` in `vernier-core::similarity::boundary`, and re-uses
`tests/python/parity/` for the new harness. This works but conflates two
distinct reference oracles under one heading, couples their ADR
governance, and pulls boundary-specific noise into the document that
matters most for vernier's primary-product parity story. A future
oracle refresh (when `bowenc0221` finally breaks under an OpenCV bump)
or an LVIS-specific `dilation_ratio = 0.008` variant would force edits to
documents that have nothing to do with pycocotools. The naïve port also
inherits the iterative-erosion performance disaster, which would make
boundary IoU 10–30× slower than mask IoU at v0.1 and silently set a bad
ceiling for every downstream that adopts it.

This ADR ratifies the decisions needed to ship boundary IoU as a
**first-class but isolated subsystem**: its own oracle pin, its own
quirks document, its own parity harness, its own constants module, its
own performance baseline, and a re-implementation of the algorithm in
single-pass O(area) form rather than the reference's O(area · d) loop —
while preserving both *mathematical parity* with the paper and
*implementation parity* (bit-equal output) with the reference.

## Decision drivers

- The Similarity trait extension point (ADR-0005) must absorb boundary
  IoU without modification: no new methods, no new parameters, no new
  lifetime constraints, no edits to `matching.rs` or `accumulate.rs`.
- The pycocotools parity infrastructure (ADR-0002,
  `pycocotools-quirks.md`, `tests/python/parity/`,
  `coco-val-parity.md`, `crates/vernier-core/src/parity.rs`) must not
  acquire boundary-IoU concerns. Boundary IoU is a different oracle,
  governed by a different ADR — coupling them in code or docs is a
  category error.
- The boundary-IoU oracle is unmaintained, OpenCV-dependent, and not
  on PyPI. CI reliance on it requires explicit mitigation (vendoring,
  frozen commit, OpenCV pin, fork plan).
- Both **mathematical parity** (the metric matches Cheng et al. 2021
  Eq. 2) and **implementation parity** (bit-equal output on shared
  fixtures) must be achievable. They are independent: an
  implementation can satisfy one without the other.
- Performance must be competitive with the mask IoU path **from v0.1**,
  not deferred. Shipping a slow-but-correct first cut creates a
  numbers-don't-change-but-perf-does revision later that breaks user
  expectations.
- ADR-0008's commitment to f64 IoU end-to-end is non-negotiable.
- ADR-0009's commitment that vernier-mask is a pure-Rust leaf reusable
  by non-evaluator users stands: boundary kernels go in vernier-mask.

## Considered options

Decided across five independent axes.

### A. Erosion algorithm

- **A1.** Iterative 3×3 erosion in dense space, applied `d` times.
  The reference's approach. O(area · d) per mask.
- **A2.** Single-pass dense erosion by a (2d+1)×(2d+1) all-ones
  structuring element, separable into two 1D passes via the van Herk /
  Gil-Werman family of sliding-window minimum algorithms. O(area) per
  mask, independent of `d`. Mathematically equivalent to A1: iterating
  3×3 erosion `d` times produces erosion by the Chebyshev (L∞) ball of
  radius `d`, which is exactly the (2d+1)-square kernel. On integer
  binary input, the operation is associative and commutative — the
  output is bit-equal regardless of which decomposition is used.
- **A3.** Distance transform (`cv2.distanceTransform` on the inverted
  mask, threshold at `d`). O(area), well-known. **Wrong norm**: produces
  L2-disk erosion, not Chebyshev. Mathematically different from A1 and
  the reference; would not produce parity. Dead on arrival.
- **A4.** RLE-native erosion (faster-coco-eval's approach: operate on
  RLE counts directly). O(area), no decode/encode. Hard to validate;
  the C++ reference is over 100 lines of unsafe pointer arithmetic and
  history of it has been bug-prone.

### B. Parity infrastructure isolation

- **B1.** Extend `pycocotools-quirks.md` with new rows; reuse
  `tests/python/parity/` and the existing `parity.rs` constants.
- **B2.** Stand up an independent `boundary-iou-quirks.md`,
  `tests/python/parity_boundary/`, vendored oracle directory,
  `coco-val-parity-boundary.md`, and `boundary_parity.rs` constants
  module. The `ParityMode` enum stays global; only the disposition
  tables it consults differ by IoU type.

### C. Module layout

- **C1.** `BoundaryIou` impl in `vernier-core::similarity::boundary`,
  kernels (`erode_chebyshev_ball`, `boundary_band`) in
  `vernier-mask::ops`. The standard pattern.
- **C2.** New crate `vernier-boundary` for both. Symmetric with the
  rest of the workspace structure but adds a crate for one impl.

### D. IoU sweep

- **D1.** Compose: `BoundaryIou::compute` calls `SegmIou::compute`
  twice (once on mask RLEs, once on boundary RLEs) and folds with
  `min()`. Reuses tested code; pays for the bbox prefilter, the RLE
  intersection sweep, and the matrix iteration twice.
- **D2.** Bespoke: run the bbox prefilter once via `BboxIou` (canonical
  for both impls), then for each surviving (g, d) pair compute mask
  intersect and boundary intersect in a single fused sweep, applying
  the crowd-aware `min()` inline. ~1.6× the lines, ~2× the throughput.

### E. Oracle dependency

- **E1.** Take an unpinned dependency on `bowenc0221/boundary-iou-api`
  and `opencv-python`. Brittle: zero recourse on upstream silence.
- **E2.** Vendor the oracle at a frozen commit hash; pin
  `opencv-python` in dev-extras; document the fork plan in
  `tests/python/parity_boundary/oracle/VENDORING.md`.
- **E3.** Re-implement the oracle in our test harness (NumPy
  reference). Decouples from upstream; trades upstream risk for
  "did we copy the spec correctly" risk.
- **E2 + E3 sidecar.** Vendored oracle is the strict-mode reference;
  NumPy reference is a sanity oracle, useful for distinguishing
  "vernier diverges from upstream" from "vernier and upstream both
  diverge from the spec".

## Decision outcome

Chosen: **A2 + B2 + C1 + D2 + E2 + E3 (sidecar)**.

### Algorithm specification (A2)

The metric, pinned in this ADR for downstream reference:

- Let `M` be the binary mask, `(h, w)` its dimensions.
- Let `d = round(dilation_ratio · √(h² + w²))`, with `dilation_ratio`
  defaulting to `0.02` (Cheng et al. paper); LVIS variant `0.008`
  selectable but not default. Half-to-even rounding. Clamp `d ≥ 1`.
- Pad `M` with a 1-pixel zero border on all four sides before
  erosion, so border-touching foreground pixels count as boundary.
- Erode the padded mask by a `(2d+1) × (2d+1)` all-ones structuring
  element. Strip the pad. Call the result `M_d`.
- Boundary band `B(M) = M ∧ ¬M_d` (equivalently `M ⊕ M_d`, since
  `M_d ⊆ M`).
- For ground-truth `G` and prediction `P`:
  - `mask_iou(G, P) = |G ∩ P| / |G ∪ P|`.
  - `boundary_iou(G, P) = |B(G) ∩ B(P)| / |B(G) ∪ B(P)|`.
  - Result reported under `iouType = "boundary"` is
    `min(mask_iou, boundary_iou)` for non-crowd `G`,
    `mask_iou` alone for crowd `G` (mirrors quirk **E1** asymmetry).

Implementation: erode by the (2d+1)-square structuring element in a
**single pass**, separable as a (2d+1) row pass followed by a (2d+1)
column pass. Each 1D pass is a sliding-window AND on the binary u8
dense mask — equivalently, a sliding-window minimum on `{0, 1}`. The
fused pipeline is two passes total over the dense mask: row erosion
into a scratch buffer, then column erosion fused with the XOR step
writing `boundary[y][x] = mask[y][x] AND NOT eroded[y][x]` directly
into the output.

This produces the bit-exact same eroded mask as iterative 3×3 erosion
applied `d` times on the same integer binary input — the operation is
the Chebyshev-ball erosion, and decomposition does not affect the
output. Mathematical parity holds against the paper; implementation
parity holds against the reference oracle.

### Parity infrastructure (B2)

New paths, none of which touch existing pycocotools artefacts:

```
docs/engineering/
  boundary-iou-quirks.md                 <- new (this PR)
  coco-val-parity-boundary.md            <- new (Phase 5)

crates/vernier-core/src/
  boundary_parity.rs                     <- new, parallel to parity.rs:
                                            BOUNDARY_DILATION_RATIO_DEFAULT,
                                            BOUNDARY_PARITY_EPS,
                                            ORACLE_COMMIT_SHA,
                                            ORACLE_OPENCV_PIN
  similarity/boundary.rs                 <- new
crates/vernier-mask/src/ops/
  erode.rs                               <- erode_chebyshev_ball(rle, radius)
  boundary.rs                            <- boundary_band(rle, dilation_ratio)

tests/python/parity_boundary/            <- new tree, sibling of parity/
  harness.py                             <- own harness (signature differs)
  test_parity_boundary.py                <- BOUNDARY_FIXTURES corpus
  fixtures/                              <- bnd_*-prefixed, no shared names
  numpy_reference.py                     <- E3 sanity oracle
  oracle/
    boundary_iou_api/                    <- vendored bowenc0221 at SHA
    VENDORING.md                         <- commit hash, OpenCV pin, fork plan
```

The `ParityMode` enum (strict / aligned / corrected) stays global. It
is a *user-facing semantic* — "am I being faithful to references or
applying opinionated fixes" — and switching that intent on a
per-IoU-type basis would be incoherent. What is per-IoU-type is the
disposition table the mode is interpreted against:

- Strict mask IoU consults `pycocotools-quirks.md`.
- Strict boundary IoU consults `boundary-iou-quirks.md`.

Both docs exist; both are loaded by their respective parity tests;
neither references the other in code paths. No shared fixture
filenames. No shared parity-test imports. The only structural coupling
is the `ParityMode` enum, which lives in
`crates/vernier-core/src/parity.rs` and is referenced by both
disposition layers.

### Module layout (C1)

Boundary kernels go in `vernier-mask` per ADR-0009 (any downstream Rust
user gets erosion and boundary extraction, not just the evaluator).
The `BoundaryIou` impl and `EvalKernel` impl go in `vernier-core`. The
boundary-specific constants module (`boundary_parity.rs`) is the
in-source counterpart to the docs-side decoupling: a reviewer searching
for "what tunables govern boundary behaviour" finds them in one place,
not commingled with the pycocotools constants in `parity.rs`.

### IoU sweep (D2)

`BoundaryIou::compute` is bespoke, not a layered call into `SegmIou`:

1. Validate dimensions identically to `SegmIou`.
2. Run `BboxIou::compute` to fill the output matrix with bbox IoU
   (the **I1** prefilter, shared with `SegmIou`).
3. Precompute `B(g)` for every gt and `B(d)` for every dt by calling
   `vernier_mask::ops::boundary_band` once each. Cache in two
   `Vec<Rle>` local to the call. Parallelise with rayon when count
   exceeds a threshold (~16 by feel; tune in Phase 7); each boundary
   computation is independent.
4. For each (g, d) pair where the prefilter wrote a non-zero value:
   compute `inter_mask = g.rle.intersect_area(d.rle)` and
   `inter_boundary = B(g).intersect_area(B(d))` in one (g, d)
   iteration. Compute `mask_iou` and `boundary_iou` from those two
   intersections plus the precomputed areas. For non-crowd gt rows
   write `mask_iou.min(boundary_iou)`; for crowd rows write
   `mask_iou`.
5. IoU divisions are f64 per ADR-0008. Intersections and areas are
   u64.

The bespoke kernel reuses `BboxIou` (the prefilter is canonical) but
does not call into `SegmIou`. `SegmIou` and `BoundaryIou` are
implementation peers, not stacked layers — both read from
`vernier_mask::Rle::intersect_area`, which is the shared kernel.

### Oracle (E2 + E3)

Vendor `bowenc0221/boundary-iou-api` at a specific commit SHA, recorded
in `tests/python/parity_boundary/oracle/VENDORING.md` along with:

- The OpenCV version pinned in dev-extras (the latest known-working
  version at vendoring time, tested in CI).
- A **fork plan**: if the upstream becomes broken (CVE, OpenCV API
  break), we fork to `vernier-fontana/boundary-iou-api-vendored` and
  point the vendored copy at the fork. This commitment is made now,
  in this ADR, so the fork is not a panic decision later.

The NumPy reference (E3 sidecar) at
`tests/python/parity_boundary/numpy_reference.py` is a small
implementation of the spec — not the strict-mode oracle. Its job is
to distinguish "vernier diverges from upstream" from "vernier and
upstream both diverge from the spec". ~50 lines, maintained against
this ADR rather than the upstream.

### Performance baseline (load-bearing, encoded in CI from day one)

Boundary IoU performance is governed by three CI-enforced budgets:

- **Per-mask boundary RLE precomputation:** wall-clock time at the
  median over a benchmark suite of 1000 representative masks must be
  ≤ 2× the per-mask RLE encode/decode roundtrip on the same masks.
  The single-pass van Herk pass makes this realistic; the iterative
  reference would not.
- **End-to-end `evaluate_boundary` on COCO val2017:** wall-clock time
  must be ≤ 3× the wall-clock of `evaluate_segm` on the same data.
- **Allocation discipline:** per-image scratch buffers for the dense
  erosion are arena-allocated and reused across images. Per-call
  heap allocations are forbidden in the hot path; CI runs a
  heap-profile assertion on the benchmark.

These budgets ship in `crates/vernier-core/benches/boundary_iou.rs`
(divan, parallel to `bbox_iou.rs`). Failing a budget is a build break,
not a deferred ticket. Budgets are revisited only via follow-up ADR.

The bitpacking question (storing the dense binary mask as packed bits,
processing 64 columns per word with bitwise AND for erosion) is
explicitly **out of scope** for v0.1. The byte-per-pixel implementation
is simpler, easier to validate, and meets the 3× budget by margin per
back-of-envelope estimates. If the budget tightens, bitpacking is a
follow-up ADR with its own perf and correctness story.

### Public API

Rust:

```rust
// vernier_core::similarity
pub struct BoundaryIou {
    pub dilation_ratio: f64,
}

impl Default for BoundaryIou { /* dilation_ratio: 0.02 */ }
impl Similarity for BoundaryIou { /* type Annotation = SegmAnn */ }
impl EvalKernel for BoundaryIou { /* mirrors SegmIou impl */ }

// vernier_core
pub fn evaluate_boundary(...) -> Result<EvalGrid, EvalError>;

// vernier_mask::ops
pub fn erode_chebyshev_ball(rle: &Rle, radius_pixels: u32)
    -> Result<Rle, MaskError>;
pub fn boundary_band(rle: &Rle, dilation_ratio: f64)
    -> Result<Rle, MaskError>;
```

Python:

```python
# vernier
vernier.BoundaryIoU(dilation_ratio=0.02)
vernier.evaluate(..., iou_type="boundary")

# pycocotools drop-in shim
from vernier import COCOeval
COCOeval(gt, dt, iouType="boundary")  # mirrors bowenc0221 signature
```

The `BoundaryIoU` Python class docstring states explicitly: metric is
Cheng et al. 2021; parity oracle is `bowenc0221/boundary-iou-api`, not
pycocotools; default `dilation_ratio` is 0.02; the `min()` composition
with mask IoU is part of the metric definition, not an implementation
detail.

## Consequences

### Positive

- Boundary IoU ships at v0.1 of the metric without contaminating
  pycocotools parity infrastructure. Future oracle refreshes,
  algorithm variants (LVIS's 0.008), or quirks discoveries are local
  edits to boundary-specific files. No churn in the documents that
  carry vernier's primary parity story.
- Performance is competitive from the start: O(area) erosion, fused
  XOR, arena-allocated scratch buffers, CI-enforced wall-clock
  budget. The eventual user expectation that "boundary IoU costs
  about as much as mask IoU" is set correctly from day one.
- Both mathematical and implementation parity are achievable
  simultaneously. The van Herk decomposition does not perturb the
  integer output relative to iterative 3×3 erosion.
- ADR-0005 and ADR-0009 are validated end-to-end: a non-trivial new
  IoU type slots in via one Similarity impl + one EvalKernel impl +
  two new mask kernels, with no edits to matching, accumulation, or
  summarization, and no new crate.
- The vendored oracle and fork plan remove "upstream went silent" as
  a critical-path risk. The decision to fork is pre-made.

### Negative

- Two parallel parity systems is more surface than one. New
  contributors need a brief orientation: pycocotools quirks live
  here, boundary quirks live there, they share `ParityMode` but not
  disposition tables. We accept this as a one-time onboarding cost
  in exchange for keeping the two oracles' lifecycles independent.
- The vendored oracle is unmaintained. If OpenCV's API breaks or a
  CVE forces a major-version bump, we own the fork. The ADR commits
  to that path in advance.
- The bespoke IoU sweep is more code than D1's compose-by-calling-
  SegmIou-twice approach. Higher review and test burden, justified
  by the ~2× perf delta (which compounds with the erosion speedup).
- Reproducing iterative 3×3 erosion via van Herk requires a small
  but real correctness argument. Property tests against an iterative
  reference implementation (kept in `vernier-mask` test code, not
  shipped) are the mitigation: 10 000 random binary masks at varied
  sizes, asserting bit-equality between van Herk and iterative.

### Neutral

- The `evaluate_boundary` entry point is a peer to `evaluate_bbox` /
  `evaluate_segm`, accessed via `iouType="boundary"` at the Python
  layer. Discoverability is identical to the existing IoU types.
- Keeping `ParityMode` global is revisitable. If a third oracle
  joins (e.g., a panoptic-boundary variant in the future), the
  global mode might need per-IoU-type overrides — that's a future
  ADR, not a v0.1 concern.
- The decision to defer bitpacked-binary erosion to a follow-up ADR
  is itself revisitable. If the byte-per-pixel implementation
  misses the 3× budget on first measurement, we open that ADR
  before shipping rather than after.

## Pros and cons of the options (full)

### Algorithm

- **A1 iterative.** 👍 simplest, matches reference verbatim. 👎
  O(area · d), wall-clock disaster for typical d ≈ 30.
- **A2 van Herk separable** *(chosen)*. 👍 O(area), bit-equal to A1
  on integer input, separable + SIMD-friendly via pulp dispatch
  (ADR-0003). 👎 small correctness argument needed; mitigated by
  iterative-reference property tests.
- **A3 distance transform.** 👍 fast, well-known. 👎 wrong norm
  (L2 vs Chebyshev), would break parity. Dead.
- **A4 RLE-native.** 👍 no decode/encode. 👎 hard to validate,
  structurally fragile, history of bugs in the C++ reference.

### Parity infrastructure

- **B1 shared with pycocotools.** 👍 one parity system. 👎 conflates
  two oracles, governance becomes coupled, future divergences in
  either spill into the other's documents.
- **B2 isolated** *(chosen)*. 👍 boundary changes are local; oracle
  refreshes don't reopen pycocotools rows. 👎 two systems to learn,
  one ParityMode enum split conceptually across two disposition
  tables.

### Module layout

- **C1 split across vernier-mask + vernier-core** *(chosen)*. 👍
  ADR-0009 honored, kernels reusable for non-evaluator users. 👎
  boundary-related code in two crates.
- **C2 own crate.** 👍 single home. 👎 yet another crate for one
  impl; users now depend on `vernier-boundary` too.

### IoU sweep

- **D1 compose.** 👍 reuses tested SegmIou. 👎 two bbox prefilters,
  two RLE intersection sweeps, two matrix iterations.
- **D2 bespoke** *(chosen)*. 👍 ~2× faster, single fused sweep, one
  matrix iteration. 👎 more lines, more tests.

### Oracle

- **E1 unpinned.** 👍 nothing to maintain. 👎 brittle, breaks on
  upstream silence.
- **E2 vendor + pin** *(chosen, primary)*. 👍 reproducible CI, fork
  plan documented. 👎 vendoring discipline.
- **E3 NumPy reference only.** 👍 zero upstream coupling. 👎 lose
  the "matches the reference everyone cites" claim.
- **E2 + E3 sidecar** *(chosen)*. 👍 strict-mode parity vs upstream
  + sanity check vs spec. 👎 two oracles to maintain (one frozen,
  one tracking spec); the second is small.

## Links and references

- ADR-0001 — Record architecture decisions.
- ADR-0002 — Three-tier parity model. This ADR is its boundary-IoU
  analogue, intentionally isolated.
- ADR-0003 — `pulp` for stable-Rust SIMD with runtime dispatch.
  The 1D row/column erosion pass is wrapped in `pulp::Arch::dispatch`.
- ADR-0005 — Similarity trait and matching engine API. Validated by
  this ADR: `BoundaryIou` requires no edits to `matching.rs`.
- ADR-0008 — Bbox IoU f64 end-to-end. Reaffirmed for boundary IoU.
- ADR-0009 — vernier-mask as a pure-Rust leaf crate. Honored: new
  kernels land in vernier-mask.
- `docs/engineering/boundary-iou-quirks.md` — the new quirks survey
  ratified by this ADR.
- `docs/engineering/coco-val-parity-boundary.md` — the boundary
  equivalent of `coco-val-parity.md`, to be created in Phase 5.
- `tests/python/parity_boundary/` — new harness, fixtures, vendored
  oracle.
- Cheng, Girshick, Dollár, Berg, Kirillov. *Boundary IoU: Improving
  Object-Centric Image Segmentation Evaluation.* CVPR 2021.
  arXiv:2103.16562.
- van Herk. *A fast algorithm for local minimum and maximum filters
  on rectangular and octagonal kernels.* Pattern Recognition Letters
  13(7), 1992.
- Gil & Werman. *Computing 2-D min, median, and max filters.* IEEE
  Transactions on Pattern Analysis and Machine Intelligence 15(5),
  1993.
- `bowenc0221/boundary-iou-api` — the strict-mode oracle (commit
  pinned in `VENDORING.md`).
