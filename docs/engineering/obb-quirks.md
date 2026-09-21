# oriented-box quirks survey

A working note (not an ADR) cataloguing the numerical and structural
quirks of the two oriented-box oracles vernier reckons with:

- **`D2`** — detectron2's `RotatedCOCOeval` and the
  `single_box_iou_rotated<float>` kernel behind it, the reference for
  **rotated boxes** (`[cx, cy, w, h, theta]`).
- **`DK`** — DOTA_devkit's `polyiou.iou_poly` composed with
  `dota_evaluation_task1.py`'s horizontal-box gate, the reference for
  **quads** (four vertices).

> **Status:** proposed with [ADR-0063](../adr/0063-oriented-box-evaluation.md).
> The `(quirk_id, oracle) → mode` cells below are the contract that
> `crates/vernier-geom/src/replica/` and `tests/python/parity_obb/`
> implement against. Rows marked **audited** carry a citation into the
> pinned oracle source and were confirmed by running it; rows marked
> *hypothesis* have not been.

This survey is independent of the other five. It shares no quirks, no
fixtures, no harness and no oracle with them — only the `ParityMode`
enum (`strict` / `corrected`, per ADR-0002 as amended 2026-05-10).

## Two tiers, not three

ADR-0002's [2026-05-10 amendment](../adr/0002-three-tier-parity-model.md#amendment-2026-05-10-collapse-aligned-into-strict)
retired the `aligned` tier: every row that carried it was already
asserted bit-equal, so it was a review-time annotation rather than a
tolerance budget. This survey therefore has exactly two dispositions:

- **strict** — vernier's output matches the oracle bit for bit. The
  implementation may be structurally different where that is cleaner or
  faster; the row says so when it is.
- **corrected** — vernier opts to fix this. Default behavior diverges
  and the divergence is documented; `parity_mode="strict"` reproduces
  the original.

ADR-0063's draft used `aligned` for rows **OB11**, **OB14** and
**OB18**. They are `strict` here, with the structural note kept in the
behavior cell.

## Source reference

| Oracle | File | Pinned at | Held how |
| --- | --- | --- | --- |
| `D2` | `detectron2/layers/csrc/box_iou_rotated/box_iou_rotated_utils.h` | `a25898a09d6ee232767647e92c6177fb1c642369` | vendored, `tests/python/parity_obb/oracle/detectron2/` |
| `D2` | `detectron2/evaluation/rotated_coco_evaluation.py` | same | vendored, same directory |
| `DK` | `polyiou.cpp` | `d3f8da45d4091b1dab37d9fbe4d6e6a50928e410` | **not vendored** — no license; SHA-256 pinned, fetched into `.cache/` |
| `DK` | `dota_evaluation_task1.py` | same | same |

Citations below are `file:line` into those exact revisions, which never
move. `biru:NNN` means `box_iou_rotated_utils.h:NNN`, `rce:NNN` means
`rotated_coco_evaluation.py:NNN`, `pi:NNN` means `polyiou.cpp:NNN`, and
`t1:NNN` means `dota_evaluation_task1.py:NNN`.

`tests/python/parity_obb/oracle/VENDORING.md` records the provenance and
the licensing position for each, including why DOTA_devkit's bytes are
not in this repository.

## What the M0 audit settled

Four rows were open questions in ADR-0063's draft. Three are now closed
by reading the pinned sources and running them, and one of the three
reversed the ADR's risk assessment:

1. **OB6 — the hull's sort (D2 go/no-go).** The draft's worst case was
   that `convex_hull_graham` feeds an epsilon-based comparator to
   `std::sort`, which is not a strict weak ordering and would make the
   output depend on libstdc++'s introsort internals. **It does not.**
   The `std::sort` call is *commented out* upstream (`biru:228-240`);
   both the CPU and CUDA paths run the same hand-written `O(n^2)`
   exchange sort (`biru:242-256`), fully specified by its own source.
   There is no standard-library ordering dependency and no libstdc++
   version to pin. **D2 strict is a go.**
2. **OB10 — the threshold comparison dtype.** `computeIoU` returns a
   torch tensor (`rce:95`), and `pycocotools`' `evaluateImg` compares it
   against an `np.float64`. Probed directly on torch 2.11: a 0-d f32
   tensor compared with `np.float64(0.7)` evaluates **in f32** — the
   scalar is weak-typed and rounds to `f32(0.7)`. So D2 matches where a
   faithful f64 comparison does not, and **ADR-0063's T1 ladder ships**.
3. **OB4 — the trigonometry.** `biru:62-65`:
   `double theta = box.a * 0.01745329251; T cosTheta2 = (T)cos(theta) *
   0.5f;`. The literal is a 12-digit truncation of `pi/180`, short by
   about `9.94e-12`; `cos`/`sin` run in **f64** and are then narrowed to
   f32. Both facts are load-bearing and both are pinned in
   `crates/vernier-geom/src/pinned.rs`.

**OB2** is confirmed by construction rather than by audit: the replica
reproduces detectron2's documented `(5, 3, 4, 2, 90)` corner probe, and
the bridge test below covers it at scale.

A fourth row is **new**, found during the port:

4. **OB19 — DOTA_devkit reads uninitialized memory.** `pi:63-64` writes
   `lineCross(a, b, p[i], p[i+1], pp[m++])`; the post-increment fires
   whether or not `lineCross` wrote anything, and `lineCross` returns
   without writing on its parallel (`pi:35`) and fully-degenerate
   (`pi:35`) paths. The slot then holds stack garbage. This is
   undefined behavior in the oracle, so `strict` has nothing to
   reproduce — see the row for what the replica does instead.

## How the claims are checked

| Claim | Where |
| --- | --- |
| D2 replica ≡ detectron2's C++ kernel, bit for bit | `crates/vernier-geom/tests/d2_bridge.rs` — 2 x 10^7 pairs across five adversarial regimes, zero divergences |
| DK replica ≡ DOTA_devkit's `iou_poly`, bit for bit | `crates/vernier-geom/tests/dk_bridge.rs` — 5 x 10^6 pairs, zero divergences |
| Canonical kernel vs an independent f64 implementation | `crates/vernier-geom/tests/canonical_properties.rs` — agreement with DK to `1e-9` |
| `IoU(a, a) = 1.0` exactly | same file, proptest over arbitrary boxes |
| Prefilters are admissible | same file, plus `replica::d2::tests::beyond_reach_the_oracle_is_zero` |
| The T1 ladder changes a real match | `crates/vernier-core/tests/obb_end_to_end.rs` — and reaches the funnel, the parallel pass and the LRP threshold, each asserted |
| DT area tracks the oriented geometry (**OB13**) | same file, plus `tests/python/parity_obb/test_obb_surface.py` — a 60x15 box at 45 deg straddles the small/medium boundary |

Both bridges are built from the pinned sources with release-equivalent,
FMA-free flags; see `tests/python/parity_obb/oracle/detectron2/build.sh`
and `tests/python/parity_obb/oracle/dota_devkit/fetch.py`.

### What actually runs, and where

Neither bridge runs under a bare `cargo test`: building a harness pins a
compiler flag set, and that is a decision rather than a side effect.
Without its `VERNIER_OBB_*_HARNESS` variable each test reports the skip
and passes, which keeps a clean checkout green.

Because a skip and a pass are indistinguishable in a CI summary, the
lanes that *mean* to run a bridge set `VERNIER_OBB_REQUIRE_BRIDGE=1`,
which turns a missing harness into a failure. Two consequences worth
stating plainly:

- **D2 runs in CI.** The `obb parity (detectron2 bridge)` job in
  `.github/workflows/ci.yml` builds the vendored Apache-2.0 header at
  its pinned commit and runs 2 x 10^6 pairs on every push. It is a
  required check through `ci-complete`.
- **DK does not.** DOTA_devkit carries no license, so it is never
  vendored and its harness needs a network fetch of unlicensed source.
  It stays a developer-provisioned check, run by `just test-obb-parity`
  when `.cache/dota-devkit/` is present. The `Quad` strict claim is
  therefore verified on demand rather than continuously, and that
  asymmetry is a known limitation of the claim, not an oversight.

---

## Disposition table

| # | Quirk | `D2` behavior → mode | `DK` behavior → mode |
| --- | --- | --- | --- |
| **OB1** | Angle unit | Degrees, via the constant at `biru:62` → `strict`; vernier requires the unit to be declared and never infers it | n/a (no angles) |
| **OB2** | Rotation direction | Counter-clockwise on screen. `biru:67-70` builds the width axis as `(cos a, -sin a)` in a y-down frame, reproducing the documented `(5, 3, 4, 2, 90)` corner probe → `strict` | n/a |
| **OB3** | Angle parameterization (`le90` / `le135` / `oc`, `w`↔`h` swap) | Corner *bits* depend on `(w, h, theta)` even though the point set does not (`biru:67-74`) → `strict` passes user numbers through untouched; `corrected` is invariant to within its error bound | n/a |
| **OB4** | Trig dtype and the deg→rad constant | **Audited.** `biru:62-65`: f64 `cos`/`sin` on `a * 0.01745329251`, narrowed to f32, then `* 0.5f`. The literal is *not* `pi/180` → `strict`; pinned as `D2_DEG_TO_RAD` | n/a (no trigonometry) |
| **OB5** | Precision frame | f32 throughout, after a pair-midpoint center shift (`biru:366-379`); `RotatedCOCOeval` instantiates at `float` via `torch.FloatTensor` (`rce:41`) → `strict` replica | f64 throughout; triangle fan hinged on the **coordinate origin** (`pi:73-74`), so intermediates are `O(X^2)` in the image coordinate → `strict` replica |
| **OB6** | Intersection construction | **Audited.** Vertex collection with `EPS = 1e-5` relaxations (`biru:96-160`), then a hand-written `O(n^2)` exchange sort and Graham scan (`biru:242-306`). The `std::sort` path is commented out (`biru:228-240`), so there is no libstdc++ ordering dependency → `strict` | Per-edge-pair triangle cuts with the absolute tolerance `sig(d)` at `pi:10` → `strict` |
| **OB7** | Degenerate geometry | `if (area1 < 1e-14 \|\| area2 < 1e-14) return 0.f` (`biru:383`) → `strict` | Zero-area quads reach `inter / union = 0/0` and return `NaN` (`pi:120-121`) → **`corrected` in both modes**: a typed error at ingestion, because ADR-0005's matrix contract forbids `NaN` and there is no oracle value to reproduce |
| **OB8** | Out-of-range output | IoU outside `[0, 1]` is reachable (detectron2#350); the epsilon-relaxed hull can enclose more area than either box → `strict` preserves it, and the matching guard `min(t, 1-1e-10)` handles it; `corrected` clamps | Non-zero residue on envelope-overlapping but polygon-disjoint pairs, from incomplete cancellation in the signed fan → `strict` preserves; `corrected` returns exactly `+0.0` |
| **OB9** | Crowd ground truth | `assert all(c == 0 for c in is_crowd)` (`rce:60`) → `strict` raises a typed error; `corrected` applies the ordinary **E1** asymmetry | n/a — DOTA's `difficult` is not `iscrowd`; `strict` raises for symmetry, `corrected` applies **E1** |
| **OB10** | Threshold comparison dtype | **Audited.** `computeIoU` returns a torch f32 tensor (`rce:95`); torch weak-types the `np.float64` threshold, so the comparison runs in f32 → `strict` via the T1 ladder (`parity::f32_projected_thresholds`) | f64 on both sides → `strict`, no action |
| **OB11** | Matrix orientation | `pairwise_iou_rotated(dt, gt)` returns `[D, G]` (`rce:63`); vernier stores `[G, D]` — structurally different, bit-equal → `strict`. Argument order is *not* interchangeable: the kernel emits line crossings on `pts1`'s edges (`biru:116`) | The composed oracle is assembled as `[G, D]` from `iou_poly(gt, dt)` (`t1:233`) — structurally different, bit-equal → `strict` |
| **OB12** | `maxDets` truncation | `computeIoU` truncates DTs to `p.maxDets[-1]` before computing (`rce:82-83`) → `strict` | Inherited from the composed oracle → `strict` |
| **OB13** | DT area (**J3**) | `loadRes` sets `area = bb[2]*bb[3]`, which on a length-5 `bbox` is `w*h` of the oriented box → `strict`. vernier's native surface keeps `bbox` as the envelope and *derives* the area from the oriented geometry through `DetectionArea::Oriented`, which is the default on every oriented entry point, so the area bucket tracks the kernel rather than the envelope. A 60x15 box at 45 deg makes the difference visible: oriented area 900 is `small`, its envelope 2812 is `medium` | `\|shoelace\|` of the quad, summed cyclically in index order (`pi:23-30`) → `strict` (composed) |
| **OB14** | Signed zero | — | `if (s1*s2 == -1) res = -res` (`pi:88`) emits `-0.0` when `res` is zero → `strict`, normalized to `+0.0` on the way out. Structurally different, bit-equal under ADR-0063's `±0` equivalence |
| **OB15** | Vertex order | n/a | Reversed in place when the signed area is negative (`pi:92-93`), and the caller then observes the reversal because `iou_poly` re-reads both areas afterwards (`pi:120`) → `strict` |
| **OB16** | Non-convex and self-intersecting quads | n/a | The fan handles any simple polygon → `strict`. A self-intersecting quad returns a cancellation-dependent value → `corrected`: typed error |
| **OB17** | Pair prefilter | None in the kernel; every pair is evaluated → `strict` with a derived admissible margin (`replica::d2::reach`), outside which the oracle returns exactly `+0.0` | Envelope overlap with the Pascal VOC `+1` (`t1:201-225`), keeping pairs whose envelopes merely touch → `strict`, and **part of the composed oracle**, not an optimization: SAT is inadmissible here because of **OB8** |
| **OB18** | Synthetic segmentation in `loadRes` | `loadRes` builds an axis-aligned polygon from `bb[0]` as `x`, which is `cx` for a length-5 `bbox`; bbox eval never reads it. vernier never constructs it — structurally different, bit-equal → `strict` | n/a |
| **OB19** | *(new)* Uninitialized scratch slot | n/a | `pi:63-64` advances `pp[m++]` even when `lineCross` returns without writing (`pi:35-36`), so the slot holds stack garbage — undefined behavior, with no value for `strict` to reproduce. The replica models it as faithfully as a defined program can: one `pp` buffer per triangle-pair call, reused across the three cuts exactly as upstream's stack array is, with a first-touch slot reading `(0, 0)`. `FanTrace::stale_slots` counts how often the path fires. Unreachable on integer-coordinate polygons — which is what DOTA annotations are — since a cross-product sign change then forces a gap of at least 1, far above `eps = 1e-8` → **`corrected` in both modes** |
| **OB20** | *(new)* Convention translation under `strict` | The replica speaks one dialect: degrees, screen-CCW. `screen_cw` input reaches it through an exact negation, so nothing is lost. `unit="rad"` needs a `180/pi` scaling that **rounds**, so the pinned bit-equality claim is keyed to `unit="deg"` → `strict` for degrees, decision-equivalence for radians | n/a |

## Wired-by

Citations name **definitions, never line numbers** — vernier's own files
move on every refactor, so a `file:line` into our tree is stale the
moment someone inserts a line above it. `tools/check_quirk_citations.py`
enforces the shape and that every name resolves;
`just lint-citations` runs it.

<!-- citation-root: crates -->

| # | Wired-by |
| --- | --- |
| OB1, OB2, OB3 | `vernier-geom/src/convention.rs::Convention` |
| OB4 | `vernier-geom/src/pinned.rs::D2_DEG_TO_RAD` |
| OB5, OB6 (D2) | `vernier-geom/src/replica/d2.rs::iou` |
| OB5, OB6 (DK) | `vernier-geom/src/replica/dk.rs::iou_poly_raw` |
| OB7 | `vernier-geom/src/error.rs::GeomError` |
| OB8 | `vernier-geom/src/kernel.rs::ratio` |
| OB9 | `vernier-core/src/similarity/obb.rs::crowd_under_strict_err` |
| OB10 | `vernier-core/src/parity.rs::f32_projected_thresholds` |
| OB11, OB12 | `vernier-core/src/similarity/obb.rs::RotatedBoxIou` |
| OB13 | `vernier-core/src/dataset.rs::oriented_area` |
| OB14 | `vernier-geom/src/replica/dk.rs::iou` |
| OB15, OB16 | `vernier-geom/src/quad.rs::PreparedQuad` |
| OB17 (D2) | `vernier-geom/src/replica/d2.rs::ReachBound` |
| OB17 (DK) | `vernier-geom/src/replica/dk.rs::hbb_plus_one_overlap` |
| OB19 | `vernier-geom/src/replica/dk.rs::FanTrace` |
| OB20 | `vernier-core/src/similarity/obb.rs::to_d2_params` |
