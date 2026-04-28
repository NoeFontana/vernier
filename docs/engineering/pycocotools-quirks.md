# pycocotools quirks survey

A working note (not an ADR) cataloguing the numerical and structural quirks of `pycocotools` 2.0.11 that vernier must reckon with. (Survey originally written against 2.0.8; 2.0.11 adds NumPy 2.0 compatibility and one extra line in `loadRes` that preserves the `info` field. No algorithm changes.) Each row is a quirk we discovered by reading the source line-by-line; ADR-0002 (three-tier parity) will dispose of every row by assigning it to one of:

- **strict** — vernier reproduces this behavior bit-exactly.
- **aligned** — vernier matches the *semantics* but may differ in incidental details (e.g., float ordering inside a stable reduction). User-visible outputs match within a documented tolerance.
- **corrected** — vernier opts to fix this. Default behavior diverges from pycocotools and the divergence is documented as an opinionated improvement.

The disposition column below is a **draft proposal** by the author of this survey — ADR-0002 is the venue where it gets ratified or revised.

## Source reference

All line numbers refer to `pycocotools-2.0.11` from PyPI:

- `pycocotools/cocoeval.py` — `COCOeval`, `Params`
- `pycocotools/coco.py` — `COCO.loadRes`, `COCO.annToRLE`
- `pycocotools/mask.py` — thin wrappers around `_mask`
- `pycocotools/_mask.pyx` — Cython glue, type discrimination
- `common/maskApi.c` / `common/maskApi.h` — RLE C kernel

Conventions used below:
- "ce" = `cocoeval.py`, "co" = `coco.py`, "mp" = `_mask.pyx`, "mc" = `maskApi.c`.
- Line citations like `ce:276` mean `cocoeval.py:276`.

---

## A. Sorting and determinism

| # | Quirk | Source | Disposition (proposed) | Wired-by |
|---|---|---|---|---|
| A1 | All argsorts use `kind='mergesort'` (stable) — explicitly chosen "to be consistent as Matlab implementation". Score-tie ordering thus depends on input array order, not on a deterministic tiebreaker like `(score, ann_id)`. | ce:173, ce:197, ce:257, ce:259, ce:366 | **strict** in default mode. Document that user input order is load-bearing. Offer a `corrected` mode that breaks ties by `(-score, ann_id)` for fully order-independent results. | `parity.rs::argsort_score_desc:75`; tests `matching::tests::a1_score_ties_resolve_to_input_order:304`, `accumulate::tests::merged_sort_breaks_ties_by_input_order:517`; fixtures `score_ties` / `score_ties_segm`. ✓ |
| A2 | `p.maxDets = sorted(p.maxDets)` — user input order is silently overwritten. Affects which `AR@1/10/100` slot maps to which threshold. | ce:137 | **aligned**. Same sort, same observable behavior. | *pending: PR `phase2/audit-a2-sort-maxdets`* |
| A3 | `evalImgs` is a flat list indexed as `[catIdx*A0*I0 + areaIdx*I0 + imgIdx]`. `accumulate()` reconstructs the cube by integer arithmetic on that flat layout. Re-running `accumulate()` with a smaller `imgIds`/`catIds`/`maxDets` works *only because* the flat index can be reconstructed; changing `areaRng` between `evaluate()` and `accumulate()` quietly produces wrong results. | ce:154-158, ce:353-358 | **corrected**. We can keep the flat layout for parity tests but expose it only through a typed accessor that panics on shape mismatch. | `evaluate.rs::flat_index:347`, `accumulate.rs::DimensionMismatch:144-156`; test `accumulate::tests::dimension_mismatch_on_grid_size_is_typed_error:557`. ⚠ partial (no different-areaRng test) — *pending: PR `phase2/audit-p2-test-gaps`* |
| A4 | GTs sorted ascending by `_ignore` (so non-ignore first, ignore last). The matching loop then relies on this ordering for its early-termination logic (B3). | ce:257-258 | **strict**. The matching loop's correctness is tied to this ordering. | `matching.rs::sort_by_key:153`; test `a4_gt_sort_puts_ignore_at_tail_regardless_of_input_order:340`. ✓ |

## B. Matching (`evaluateImg`)

| # | Quirk | Source | Disposition | Wired-by |
|---|---|---|---|---|
| B1 | Initial best-IoU is `min(t, 1 - 1e-10)`, *not* `t`. Equivalent to using `>=` with a tolerance of `1e-10` at the threshold boundary. A detection with IoU exactly equal to threshold `t` matches; one with IoU `t - 5e-11` also matches. | ce:276 | **strict**. This affects fixtures designed to land exactly on the threshold. | `parity.rs::IOU_BOUNDARY_EPS:47`, `matching.rs:177`; tests `b1_iou_exactly_at_threshold_matches:272`, `b1_iou_at_one_still_matches:281`; fixture `iou_at_threshold`. ✓ |
| B2 | Greedy match: a DT iterates GTs in order; updates "best" only when `ious[d,g] >= current_best`; non-strict `>=` means later GTs with equal IoU win over earlier ones. Combined with A4, this means an ignore GT can replace a non-ignore best match if IoU is equal — but B3 prevents this. | ce:286-290 | **strict** (load-bearing). | `matching.rs:198`; test `b4_crowd_gt_matches_many_dts:290`; fixture `crowd_overlap_tiebreak`. ✓ |
| B3 | Early termination: if a real (non-ignore) match has been recorded, scanning stops as soon as the next GT is an ignore GT (`if m>-1 and gtIg[m]==0 and gtIg[gind]==1: break`). Relies on A4's sort. | ce:283-284 | **strict**. | `matching.rs:189`; test `b3_ignore_gt_terminates_inner_loop_after_real_match:316`. ✓ |
| B4 | Crowd GTs allow many-to-one: `if gtm[tind,gind]>0 and not iscrowd[gind]: continue` skips already-matched GTs unless they are crowd. So multiple DTs can match the same crowd GT. | ce:280 | **strict**. Documented COCO semantic. | `matching.rs:183`; test `b4_crowd_gt_matches_many_dts:290`; fixtures `crowd_region` / `crowd_region_segm`. ✓ |
| B5 | `dtm` and `gtm` store the *opposite* side's id (DT row stores matched GT id; GT row stores matched DT id). Stored as float64 (zero-init array). IDs become floats and must be cast back. | ce:295-296, ce:268-269 | **aligned**. We can store as int with a sentinel and cast at the boundary. | `matching.rs:83-87` (`Array2<i64>` -1 sentinel); `harness.py:146` (snapshot float64 coercion). ✓ |
| B6 | `dtIg[t,d]` after matching is `gtIg[m]` (the ignore-flag of the matched GT). So a DT matched to an ignored GT inherits "ignore", which means it is *neither TP nor FP* — it disappears from the curve. | ce:294, ce:375-376 | **strict**. | `matching.rs:210`; test `b6_dt_matched_to_ignore_inherits_flag:329`; fixtures `crowd_region`(_segm). ✓ |
| B7 | DT outside area range gets `dtIg=1` *only if unmatched*: `dtIg = logical_or(dtIg, logical_and(dtm==0, repeat(a, T, 0)))`. So a *matched* DT outside area range still counts as TP. The matching itself ignores DT area entirely. | ce:298-299 | **strict** (this is a known asymmetry in COCO eval). | `evaluate.rs:597-600`; test `b7_unmatched_dt_outside_area_range_is_ignored:757`. ✓ |

## C. Recall integration and PR curve

| # | Quirk | Source | Disposition | Wired-by |
|---|---|---|---|---|
| C1 | 101-point recall integration: `recThrs = linspace(0, 1, 101)`. AP is the mean of interpolated precision at exactly these 101 recalls. Uses `np.searchsorted(rc, recThrs, side='left')` to find the first cumulative recall `>= rt`. | ce:507, ce:402 | **strict**. Trapezoid integration would be measurably different on noisy curves. | `accumulate.rs:330`; test `partition_point_matches_numpy_searchsorted_left:505`. ✓ |
| C2 | Precision envelope: descending sweep `for i in range(nd-1, 0, -1): if pr[i] > pr[i-1]: pr[i-1] = pr[i]`. Standard PASCAL "interpolated AP" — precision at recall ≥ r is the *max* precision achievable at any recall ≥ r. | ce:398-400 | **strict**. | `accumulate.rs:310-314`; test `precision_envelope_runs_right_to_left:482`. ✓ |
| C3 | If a recall threshold is unreachable (`searchsorted` returns `nd`), the bare `try/except: pass` swallows the resulting IndexError and leaves remaining `q[ri]`/`ss[ri]` at zero. Means precision at unreached recalls is exactly **0**, not -1. | ce:402-408 | **strict** but **rewrite without `except:`**. We can do `inds = clip(inds, 0, nd-1)` plus an explicit mask of unreached rows; same numerical output, no swallowed exceptions. | `accumulate.rs:331-338`; test `lone_fp_yields_zero_recall_zero_precision:437`. ✓ |
| C4 | `recall[t,k,a,m] = rc[-1] if nd else 0` — "AR" is the **terminal** cumulative recall (i.e., total TPs / total positives), *not* the area under a PR curve's recall axis. Many users expect "AR@N" to be an integral; it's not. | ce:389-392 | **strict**. Document loudly. | `accumulate.rs:307`; test `perfect_match_yields_ap_one_and_ar_one:419`. ✓ |
| C5 | Absent-category sentinel: `precision`, `recall`, `scores` arrays initialized to `-1`. `summarize()` filters with `s[s>-1]` before computing the mean. So a category with no GTs disappears from the average; it does not get counted as zero. | ce:334-336, ce:452-455 | **strict**. | `accumulate.rs:188-190`, `summarize.rs:393-395`; test `cell_with_only_ignore_gts_skips_entirely:410`. ✓ |
| C6 | `pr.tolist()` and `q.tolist()` then iterate as Python lists for the inner monotonic sweep — explicit comment "numpy is slow without cython optimization for accessing elements". A vector op (`np.maximum.accumulate(pr[::-1])[::-1]`) gives the same result much faster. | ce:396 | **aligned**. Same outputs, faster path. | `accumulate.rs:329-338` (vectorized). ⚠ no labeled test — *pending: PR `phase2/audit-p2-test-gaps`* |
| C7 | TP/FP are computed as `logical_and(dtm, logical_not(dtIg))` and `logical_and(logical_not(dtm), logical_not(dtIg))`. A DT that matched an ignore-GT is in *neither* set. | ce:375-376 | **strict**. | `accumulate.rs:295`; test `ignored_dt_does_not_count_as_fp:458`. ✓ |
| C8 | `np.spacing(1)` ≈ 2.22e-16 used as eps in `pr = tp / (fp+tp+np.spacing(1))`. Per-platform/numpy-version reproducible but *not* a clean constant. | ce:385 | **aligned**. We pin the eps to `f64::EPSILON` and document that the resulting precision matches numpy's `np.spacing(1)` to all bits. | `parity.rs::PARITY_EPS:42`; test `parity_eps_matches_numpy_spacing_1:109`. ✓ |

## D. Area range and ignore flags

| # | Quirk | Source | Disposition | Wired-by |
|---|---|---|---|---|
| D1 | `gt['ignore'] = ('iscrowd' in gt) and gt['iscrowd']` **overwrites** any prior `ignore` field set by the dataset. Line ce:108 reads it ("default to 0 if absent"), line ce:109 immediately discards that read. The line ce:108 read is dead code unless `iscrowd` is missing AND ignore is present — but then line 109 still overwrites with `False`. So a user-set `ignore=1` field is silently discarded if `iscrowd` is missing. | ce:107-109 | **corrected**. The dead-code read is almost certainly a pre-existing bug. We should honor an explicit `ignore` field. | `dataset.rs::effective_ignore:181-186`; tests `d1_strict_mode_drops_explicit_ignore_field:768`, `d1_strict_mode_uses_iscrowd_when_ignore_absent:788`, `d1_parity_mode_propagates_to_base_ignore:870`. ✓ |
| D2 | For keypoints, `gt['ignore'] = (num_keypoints == 0) or gt['ignore']` — adds a second OR. So a GT with 0 visible keypoints is implicitly an ignore region, regardless of iscrowd. | ce:110-111 | **strict**. | Ratified by [ADR-0012](../adr/0012-oks-keypoints-surface.md). Implemented by `OksSimilarity::extra_gt_ignore` (separate PR). |
| D3 | `_ignore` is a per-evaluation flag: `True` if `gt['ignore']` is set OR `gt['area']` is outside the current area range. So changing `aRng` rewrites `_ignore` mid-loop. The same GT object is mutated in place. | ce:250-254 | **aligned**. We compute `_ignore` per-call without mutating; observable output identical. | `evaluate.rs:550-556` (no-mutation by `Arc<Vec<…>>` construction). ✓ |
| D4 | Default `areaRng` for det: `[[0, 1e10], [0, 1024], [1024, 9216], [9216, 1e10]]`. The "all" bucket is `[0, 1e10]` — floats; `1e5**2` is exactly `1e10`. | ce:509 | **strict** (these are user-visible defaults; changing them breaks every published score). | `evaluate.rs::AreaRange::coco_default:99-122`. ⚠ no literal-pinning test — *pending: PR `phase2/audit-p2-test-gaps`* |
| D5 | Default `areaRng` for kp: `[[0, 1e10], [1024, 9216], [9216, 1e10]]` — kp drops the small bucket. `summarize()` for kp asks for area='medium'/'large' but never 'small'. | ce:520, ce:478-479 | **strict**. | Ratified by [ADR-0012](../adr/0012-oks-keypoints-surface.md). Caller-side: keypoints clients pass the 3-entry `area_ranges` `[(0, 1e10), (1024, 9216), (9216, 1e10)]`. |
| D6 | Area-range filter on GT uses **strict** inequalities: `g['area'] < aRng[0] or g['area'] > aRng[1]`. So a GT with area exactly `1024` falls in *both* small (`[0, 1024]`) and medium (`[1024, 9216]`). Buckets are not partitions. | ce:251 | **strict**. | *pending: PR `phase2/audit-d6-area-boundary`* (current implementation diverges from pycocotools; that PR flips inclusion to non-strict and adds fixture `boundary_area_segm`) |
| D7 | `getAnnIds(areaRng=...)` filter, by contrast, uses **strict** open interval: `ann['area'] > aRng[0] and ann['area'] < aRng[1]`. Different inequality from the eval filter. So filtering GTs upfront via `getAnnIds(areaRng=[a,b])` yields a different set than the eval-time `_ignore` would mark in. | co:148 | **strict** (different code path; documents the inconsistency). | *n/a (out-of-scope)* — no `getAnnIds(areaRng=...)` API surface |

## E. Crowd semantics

| # | Quirk | Source | Disposition | Wired-by |
|---|---|---|---|---|
| E1 | Crowd IoU semantic: `iou(crowd_gt, dt) = area(intersect) / area(dt)`. So a tiny DT inside a huge crowd region scores 1.0. Asymmetric: not a real IoU. | mc:109 (rle), mc:133 (bb) | **strict**. | `similarity/bbox.rs:96-104`; test `e1_crowd_gt_uses_dt_area_denominator:220`; fixtures `crowd_region`(_segm), `crowd_rle_gt_segm`. ✓ |
| E2 | DT is *never* crowd: `loadRes` sets `iscrowd=0` for every detection regardless of input. Even if the user's result JSON contains `iscrowd=1`, it is overwritten on load. | co:344, co:353 | **strict**. | `dataset.rs:469-484`, `evaluate.rs:206-214`; test `dt_iscrowd_flag_is_ignored:235`. ✓ |
| E3 | The `iou()` API takes `iscrowd` as a length-`n_gt` array. There is no DT-side iscrowd vector. The Cython assertion `crowd_length == n` enforces gt-length only. | mp:218-219 | **strict**. | `similarity/mod.rs::Similarity::compute:49-54` (type signature has no DT-side iscrowd vector). ⚠ no labeled test — *pending: PR `phase2/audit-p2-test-gaps`* |

## F. Keypoints / OKS

| # | Quirk | Source | Disposition | Wired-by |
|---|---|---|---|---|
| F1 | OKS uses 17 hardcoded sigmas for COCO person, scaled by `/10.0`. No mechanism to override per category in the official API; downstream forks (e.g. CrowdPose) monkey-patch `Params.kpt_oks_sigmas`. | ce:523 | **corrected**. We accept sigmas as a per-category parameter; default values match COCO person. | Ratified by [ADR-0012](../adr/0012-oks-keypoints-surface.md). Implemented by `OksSimilarity::new(sigmas)` at `crates/vernier-core/src/similarity/oks.rs`; per-category sigmas mapping `Mapping[int, tuple[float, ...]]` carried on the future `Keypoints` IouKind variant. |
| F2 | OKS area uses `gt['area'] + np.spacing(1)`. The `+spacing` is purely a divide-by-zero guard. | ce:229 | **aligned**. | Ratified by [ADR-0012](../adr/0012-oks-keypoints-surface.md). Implemented by `OksSimilarity::compute` (`area + f64::EPSILON`) at `crates/vernier-core/src/similarity/oks.rs`. |
| F3 | When no keypoints are visible (`k1==0`), the per-keypoint distance falls back to a **completely different metric**: distance from each predicted keypoint to a 2x-expanded bbox (`[bb_x - bb_w, bb_x + 2*bb_w]`). This is not OKS at all; it's a "stay near the GT bbox" surrogate. | ce:215-216, ce:225-228 | **strict**. The fallback only matters when `k1==0`, which is rare in practice and conceptually corresponds to "this GT has no visible keypoints, so reward DTs that at least picked a plausible region". | Ratified by [ADR-0012](../adr/0012-oks-keypoints-surface.md). Implemented by the bbox-surrogate fallback in `OksSimilarity::compute` at `crates/vernier-core/src/similarity/oks.rs` (triggers when GT has zero visible keypoints — `k1 == 0` counted from GT visibility flags `v > 0`; DT visibility is unread, every DT keypoint contributes). |
| F4 | OKS bbox-expansion direction: `[bb[0] - bb[2], bb[0] + 2*bb[2]]` is asymmetric — extends `bb[2]` to the left, `2*bb[2]` to the right. So a DT keypoint at `bb[0] - bb[2] - eps` is "far left" but a DT at `bb[0] + 2*bb[2] + eps` is "far right" by the same amount. The expansion shape is asymmetric around the GT. | ce:215-216 | **strict**. Pre-existing asymmetry in cocoeval; preserve. | Ratified by [ADR-0012](../adr/0012-oks-keypoints-surface.md). Implemented by `OksSimilarity::compute` (`[bb.x - bb.w, bb.x + 2*bb.w]` asymmetry preserved) at `crates/vernier-core/src/similarity/oks.rs`. |
| F5 | `computeOks` returns `[]` when **either** gt or dt is empty. `computeIoU` returns `[]` only when **both** are empty. Subtle inconsistency. | ce:171 vs ce:202 | **aligned**. Returning a 0-row or 0-col array of correct shape gives the same downstream behavior. | Ratified by [ADR-0012](../adr/0012-oks-keypoints-surface.md). Implemented by `OksSimilarity::compute` (zero-shape early return) at `crates/vernier-core/src/similarity/oks.rs`. |

## G. RLE encoding format

| # | Quirk | Source | Disposition | Wired-by |
|---|---|---|---|---|
| G1 | The "compressed counts" string uses **6-bit chars 48-111** (printable ASCII subset), not standard LEB128: 5 data bits per char, bit 5 = continuation, bit 4 = sign (when continuation cleared). | mc:237-250, mp:14 | **strict**. This is the on-the-wire format users ship in JSON. Must round-trip bit-exactly. | `vernier-mask/src/codec.rs::value_16_needs_two_chars_for_sign_disambiguation:135`. ✓ |
| G2 | Differential encoding: from `i==2` onward, `cnts[i] -= cnts[i-2]`. Decoder undoes via `x += cnts[m-2]`. This means even-indexed runs are differenced against even-indexed neighbors; same for odd. Chosen to keep deltas small for typical masks (gradual size change between consecutive runs). | mc:242, mc:263 | **strict**. | `codec.rs::differential_kicks_in_at_index_three:140`. ✓ |
| G3 | Sign extension on decode: `if(!more && (c & 0x10)) x \|= -1 << 5*k`. Implementation-defined behavior in C (left-shift of negative). Needs an explicit two's-complement formulation in Rust. | mc:261 | **aligned**. Rust port computes `x \|= -(1i64 << (5*k))` or equivalent unsigned bit-twiddle. | `codec.rs::negative_differential_uses_sign_extension:148`. ✓ |
| G4 | RLE counts can include zero-length foreground runs. `rleToBbox` and `rleArea` tolerate them. | mc:159, mc:88-91 | **strict**. | `vernier-mask/src/ops.rs::bbox_all_zero_length_foreground:268`. ✓ |
| G5 | `rleArea` sums **odd-indexed** counts (foreground = positions 1, 3, 5, ...). Implies the encoding always starts with a background run, even if zero-length. | mc:88-91 | **strict**. | `vernier-mask/src/raster.rs::all_foreground_starts_with_zero_length_background:145`, `ops.rs::area_sums_odd_indexed_runs:253`; fixture `crowd_rle_gt_segm`. ✓ |
| G6 | `rleEncode` treats any non-zero pixel as foreground (binarizes via `T[j]!=p` with `byte p`). So a uint8 mask with values `0,2` is indistinguishable from `0,1`. | mc:32-41 | **strict**. | `raster.rs::nonzero_bytes_binarize_per_g6:152`. ✓ (disposition reconcile in `phase2/audit-g6-disposition-reconcile`) |

## H. RLE decoding and rasterization

| # | Quirk | Source | Disposition | Wired-by |
|---|---|---|---|---|
| H1 | `rleDecode` returns `0` (failure) if the RLE counts overflow the mask buffer. `_mask.pyx::decode` raises ValueError on `0`. So malformed RLE is recoverable, not a segfault. | mc:45-62, mp:148-149 | **strict**. | `codec.rs::rejects_out_of_range_byte:157`, `codec.rs::rejects_truncated_run:168`. ✓ |
| H2 | `rleMerge` silently produces an empty 0×0 RLE if dimensions mismatch — no error. | mc:73 | **corrected**. We should raise an error. The silent empty result has bitten downstream tools. | `ops.rs::merge_dimension_mismatch_errors:312`, `similarity/segm.rs::rle_dimension_mismatch_returns_typed_error:230`, `segmentation.rs::rle_size_mismatch_errors_h2_corrected:209`. ✓ |
| H3 | `rleFrPoly` rasterizes via 5x supersampling: `x = (int)(scale * xy[i] + 0.5)`. Then walks the boundary with Bresenham-like line drawing, then downsamples. The exact pixels filled depend on **point order** (boundary direction): a polygon and its reverse trace produce slightly different masks at thin edges. | mc:193-235 | **strict**. Rasterization differences propagate through every segm score. | fixture `self_intersecting_polygon_segm`. ✓ |
| H4 | `rleFrPoly` clips x-points outside `[0, w-1]` via `if(floor(xd)!=xd \|\| xd<0 \|\| xd>w-1) continue` — drops that entire boundary segment. y-points outside `[0, h]` are clamped (`if(yd<0) yd=0; else if(yd>h) yd=h`). Asymmetric handling of x vs y boundary. | mc:220-222 | **strict**. | fixture `polygon_at_image_edge_segm`; `polygon.rs::polygon_clipped_to_image_bounds:368`. ✓ |
| H5 | `rleFrPoly` rounds y via `ceil(yd)` after clamp; rounds x via `floor` (implicit in `(int)`). So polygon vertices on a pixel boundary land in different pixels along x vs y. | mc:222 | **strict**. | fixture `polygon_at_image_edge_segm`; `polygon.rs::axis_aligned_2x2_square_in_4x4_image:333`. ✓ |
| H6 | `rleFrBbox(bb)` rasterizes by going through `rleFrPoly` with the corner polygon — so a "bbox to mask" goes through the same supersampled rasterizer as polygon-to-mask, including any rounding quirks. | mc:180-187 | **strict**. | `polygon.rs::from_bbox_matches_explicit_polygon:351`. ✓ |
| H7 | `rleNms` is O(n²) sequential pairwise. No spatial index. For large detection sets this is the wall-clock bottleneck. | mc:114-123 | **corrected** in vernier (or out-of-scope, since cocoeval doesn't use NMS). | *n/a (out-of-scope)* — `rleNms` deliberately not ported |

## I. IoU computation

| # | Quirk | Source | Disposition | Wired-by |
|---|---|---|---|---|
| I1 | `rleIou` uses bbox IoU as a prefilter: pairs with `bbIou == 0` are skipped (kept at 0); only pairs with `bbIou > 0` get exact RLE IoU. The bbox bounding the RLE is conservative (encloses the RLE), so the prefilter is exact for non-overlap. | mc:95-98 | **strict**. | `similarity/segm.rs::disjoint_masks_are_zero_via_bbox_prefilter:170`. ✓ |
| I2 | `rleIou` returns `-1` when `dt.h != gt.h \|\| dt.w != gt.w`. Sentinel meaning "skip"; not a real IoU value. Downstream code in `evaluateImg` doesn't filter -1 explicitly — it relies on `-1 < threshold` to fail the match. | mc:100 | **corrected**. We should raise on dimension mismatch instead of returning a magic value that flows into the scoring. | `ops.rs::intersect_area_dimension_mismatch_errors:393`, `similarity/segm.rs::rle_dimension_mismatch_returns_typed_error:230`. ✓ |
| I3 | `rleIou` `i==0 → u=1` to avoid 0/0; `bbIou` short-circuits when `w<=0` or `h<=0` (no division at all). Asymmetric guards. | mc:109, mc:131-132 | **aligned**. Both cases yield IoU=0; we can use a single clean guard. | `similarity/bbox.rs::zero_area_gt_with_zero_inter_yields_zero_not_nan:247`, `:258`; `segm.rs::empty_gt_or_dt_pair_is_zero_not_nan:218`. ✓ |
| I4 | `bbIou` non-overlap check is `w<=0` and `h<=0` (after `fmin - fmax`). So two boxes that share an edge (e.g., `[0,0,1,1]` and `[1,0,1,1]`) have zero IoU — no edge-sharing intersection. | mc:131-132 | **strict**. Standard convention. | `similarity/bbox.rs::i4_edge_sharing_is_zero:198`. ✓ |
| I5 | `iou(dts, gts, iscrowd)` Cython entry returns Python `[]` when either side is empty. Not a 0-shape numpy array. | mp:220-221 | **aligned**. | `similarity/bbox.rs::empty_inputs_return_unchanged_matrix:285`, `segm.rs:258`. ✓ |
| I6 | `_preproc` for a 1D ndarray does `objs.reshape((objs[0], 1))` — uses **the value of objs[0]** as the row count. Almost certainly a bug (should likely be `(1, 4)`); reachable only if a caller passes a 1-D array, which neither cocoeval nor coco ever does. Latent landmine. | mp:177 | **corrected** (or just refuse 1-D input outright). | *enforced by type system* — kernels accept slices/`Rle`, not 1-D ndarrays. ✓ |

## J. Loading detection results (`loadRes`)

| # | Quirk | Source | Disposition | Wired-by |
|---|---|---|---|---|
| J1 | DT `id` is overwritten with sequential 1..N regardless of any `id` field in the input. So `evalImgs['dtIds']` does not match user input. Round-tripping requires a separate user→assigned mapping. | co:333, co:343, co:352, co:362 | **aligned**. We can preserve user ids when present. | `dataset.rs::j1_auto_assigns_ids_when_absent:860`, `j1_preserves_user_supplied_ids:871`. ✓ |
| J2 | Bbox-only DT: a fake rectangular "polygon" segmentation `[[x1,y1, x1,y2, x2,y2, x2,y1]]` is synthesized. If the user later switches `iouType='segm'`, those DTs are evaluated against this rasterized rectangle, not the user's intended box. | co:341 | **strict** (silent path-dependence). Document loudly. | *currently diverges* (raises `InvalidAnnotation`); policy decision in *pending: PR `phase2/audit-j2-j6-policy`* |
| J3 | DT `area` is auto-derived: from `bbox` for bbox results; from `maskUtils.area(seg)` for segm; from the keypoint-bbox `(x1-x0)*(y1-y0)` for keypoints. The keypoint area is *not* the GT person area; it can be tiny if the visible keypoints are clustered, dropping a DT into the "small" bucket unexpectedly. | co:342, co:349, co:361 | **strict** (drives small/medium/large bucketing — changing breaks scores). | `dataset.rs::j3_derives_area_from_bbox:882`. ✓ |
| J4 | DT `iscrowd = 0` always. Per E2. | co:344, co:353 | **strict**. | `dataset.rs:469-484`; test `loads_detections_from_json_array:917`. ✓ |
| J5 | `assert set(annsImgIds) == (set(annsImgIds) & set(self.getImgIds()))` — DT can cover *fewer* images than GT; only forbids unknown image ids. So missing-image silently scores 0 recall on that image. | co:328 | **strict**. | `evaluate.rs::missing_dt_image_yields_none_cells:981`; fixtures `missing_dt_image`(_segm). ✓ |
| J6 | First DT entry's keys decide the dispatch ("if `bbox` in anns[0]" / "elif `segmentation` in anns[0]" / "elif `keypoints` in anns[0]"). Heterogeneous DT lists where some entries have segm and some have only bbox are handled by the **first** entry's type; later entries get whatever derivation that path chose. | co:330-363 | **corrected**. We should require homogeneous result lists or dispatch per-entry. | *unwired*; policy decision in *pending: PR `phase2/audit-j2-j6-policy`* |

## K. Polymorphism and type discrimination

| # | Quirk | Source | Disposition | Wired-by |
|---|---|---|---|---|
| K1 | `frPyObjects` discriminates by `type(pyobj) == list` and `len(pyobj[0])`: `==4` → bbox(es), `>4` → polygon(s), `==4 and pyobj[0]` is dict-with-counts → uncompressed RLE. **A polygon with exactly 2 points (4 floats) collides with bbox detection.** Such polygons are legal in the COCO schema but get silently treated as a single bbox. | mp:290-310 | **corrected**. Reject ambiguous input or require an explicit type tag. | `polygon.rs::rejects_odd_coordinate_count:284`, `:293` `rejects_fewer_than_three_vertices`, `:309` `rejects_nan_and_infinity`; `segmentation.rs::polygon_with_too_few_vertices_propagates_k1_error:234`. ✓ |
| K2 | `annToRLE` for polygon segm: multi-polygon GT (e.g., a person split across an occluder) is **merged into a single RLE via union**. Subsequent IoU treats the GT as one mask. | co:427-429 | **strict**. | fixture `multi_polygon_gt_segm`; `polygon.rs::from_polygons_multi_unions_disjoint_regions_k2:401`; `segmentation.rs::polygon_to_rle_rasterizes_and_unions_k2:169`. ✓ |
| K3 | `_frString` accepts `bytes` or `str` for `counts` but raises only on `PYTHON_VERSION` not in {2,3}. The bytes/str branching also hides a subtle issue: when `counts` is `bytes`, no encoding pass happens; when `str`, it gets `str.encode(...)` (default utf-8). The COCO RLE chars are all in the 0x30–0x6F range so utf-8 = ascii here. Fine in practice, fragile in principle. | mp:123-129 | **aligned**. We accept either and decode as ASCII (rejecting non-ASCII). | `segmentation.rs::parses_compressed_rle_shape:142`, `:183`. ✓ |
| K4 | `iou()` requires `type(dt) == type(gt)`. A list of bboxes vs a list of RLE dicts raises with a generic message. Mixed input is unsupported. | mp:222-223 | **aligned**. | *enforced by associated-type discrimination* (`similarity/mod.rs::Annotation:39`). ✓ |

## L. API and parameter quirks

| # | Quirk | Source | Disposition | Wired-by |
|---|---|---|---|---|
| L1 | `iouThrs` constructed via `np.linspace(.5, 0.95, int(np.round((0.95-0.5)/.05)) + 1, endpoint=True)`. The author explicitly avoids `np.arange` because of accumulated float error. Float values of the 10 thresholds are reproducible across numpy versions. | ce:506 | **strict**. | `parity.rs::iou_thresholds_match_numpy_linspace:124`. ✓ |
| L2 | `recThrs` constructed identically. R=101. | ce:507 | **strict**. | `parity.rs::recall_thresholds_have_101_points_endpoints_pinned:156`. ✓ |
| L3 | `Params.useSegm` deprecated but still honored: if set, overrides `iouType`. Prints a warning. | ce:130-132, ce:534 | **corrected**. Drop in vernier; useSegm has been deprecated for years. | *unwired*; rejection added in *pending: PR `phase2/audit-l3-reject-usesegm`* |
| L4 | `Params.useCats` is `1` (int) not `True` (bool). Comparisons via truthiness throughout. | ce:511, ce:328 | **aligned**. | `dataset.rs::indices_for_image:912`; `tests/python/test_compat.py:151`. ✓ |
| L5 | `summarize()` is hardcoded to print to stdout via `print(iStr.format(...))`. There is no machine-readable summary API; users either capture stdout or read `self.stats` (12-element array for det, 10 for kp) and reconstruct labels from defaults. | ce:456, ce:493 | **corrected**. Return a structured `Summary` value. | `summarize.rs::pretty_lines:135`; tests `tests/python/test_compat.py::test_summarize_strict_mode_prints:77`, `:93`. ✓ |
| L6 | `summarize()` returns `None`; sets `self.stats` as side effect. `__str__` calls `summarize()` and also returns `None`, which makes `str(coco_eval)` print but evaluate to `'None'`. | ce:493, ce:495-496 | **corrected**. | `summarize.rs::Summary:119`, `python/vernier/__init__.py::Evaluator.evaluate:79`; test `tests/python/test_evaluator.py:59`. ✓ |
| L7 | The `_summarize(ap=1, ...)` inner function raises `IndexError` if user passes `iouThr` not in `iouThrs` (np.where returns empty). Passes silently if `areaRng` label is wrong (aind=[], `s[:,:,:,[],mind]` is 0-shape, mean of empty = -1 by the `s>-1` filter). Inconsistent error handling. | ce:441-456 | **corrected**. | `summarize.rs::missing_max_det_value_is_typed_error:556`, `out_of_range_area_index_is_typed_error:665`. ✓ |
| L8 | `Params.kpt_oks_sigmas` is set inside `setKpParams()` but **not** inside `setDetParams()`. Switching `iouType` from 'keypoints' back to 'bbox' leaves `kpt_oks_sigmas` defined as a stale attribute on the params object. | ce:502, ce:513-523 | **aligned**. Per-mode params object. | Ratified by [ADR-0012](../adr/0012-oks-keypoints-surface.md). The future `Keypoints` IouKind variant carries `sigmas` per-instance (the discriminated-union design from ADR-0011), so switching `iou` to `Bbox()` cannot leak `kpt_oks_sigmas` — there is no shared `Params.kpt_oks_sigmas` to leak. |

---

## Glossary cross-reference (for ADR-0002 authors)

When ADR-0002 is drafted, every row above must be cited by ID (e.g. "B1, B3 — strict; D1 — corrected; H2 — corrected"). The disposition column is the author's *proposal*; the ADR is the venue where each row is signed off.

A short cheat-sheet for the proposed defaults:

- **Most quirks: strict.** pycocotools is the reference; reproducing every quirk is the only way to claim parity.
- **A handful of corrected:** A3 (flat-index re-accumulate), C3 (bare except), D1 (overwritten ignore field), H2 (silent merge), I2 (-1 sentinel), I6 (1-D reshape bug), J6 (heterogeneous DT dispatch), K1 (2-point polygon collision), L3/L5/L6/L7 (API hygiene), F1 (per-category sigmas).
- **Aligned (semantic match, cleaner implementation):** A2, B5, C6, C8, D3, F5, G3, H8 (none yet), I3, I5, J1, K3, K4, L4, L8.

Each "corrected" item is a behavior change that must be opt-in or documented as an intentional break. The "aligned" items are pure implementation cleanups: same outputs, faster/safer code.

## Open questions

These are quirks where the reading is uncertain and we should write a small reproducer:

1. **B1 + integer IoU**: does `min(t, 1-1e-10)` actually allow IoU=t to match? The `<` comparison at ce:286 says yes for `t<1`, no for `t=1`. Need a test fixture at `iouThr=1.0`.
2. **D6 vs D7**: confirm the inequality difference between `getAnnIds(areaRng=...)` and the eval-time `_ignore` filter via a fixture with a GT of area exactly equal to a bucket boundary.
3. **H3**: how much does polygon traversal direction affect mask output? Build a fixture with a known polygon and its reverse, diff the masks. Suspect: 1-2 pixels around the boundary.
4. **K1 (2-point polygon)**: can we reach the bbox/polygon collision via valid COCO input? Need to check whether the schema permits 2-vertex polygons.
5. **G3 sign extension**: cross-check pycocotools's behavior on counts ≥ 2³⁰ with a Rust two's-complement port; do the two implementations agree on edge-case big counts?

Each open question is a ~30-minute fixture; a follow-up commit should resolve them and update the disposition table.
