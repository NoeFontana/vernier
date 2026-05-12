# ADR-0026: Add LVIS federated evaluation in `vernier-core`

- **Status:** accepted
- **Date:** 2026-05-03
- **Accepted on:** 2026-05-03
- **Deciders:** @NoeFontana
- **Consulted:** —
- **Informed:** all contributors

## Context and problem statement

LVIS (Gupta et al., CVPR 2019) is the canonical long-tail instance
segmentation benchmark — 1203 categories spanning a heavy tail. The
benchmark introduces **federated evaluation**: each image's GT is
annotated only for a subset of categories, so the standard COCO
assumption "every unmatched detection is a false positive" misfires
on long-tail data. LVIS solves this with per-image positive and
negative category sets and reports AP partitioned by category
frequency (`AP_r` / `AP_c` / `AP_f`).

The integration shape is structurally different from panoptic
(ADR-0025). Panoptic doesn't share the AP fold and lands as a sibling
crate; LVIS **does** share the AP fold completely. The IoU kernels
are identical to COCO, the matching engine already supports per-cell
`dt_ignore`, and the accumulator/summarizer already iterates the
`(T, R, K, A)` tensor (LVIS has no M-axis — single `max_dets`, AF5).
Federated evaluation is an **orchestration extension** on top of the
locked spine: a per-(image, category) skip mask, a per-cell
`dt_ignore` extension, and a category-subset filter at summarize time.

This ADR triggers ADR-0001 §"Affect the public API" (federated fields
on the dataset; new summary plan entries) and §"Add or remove a
top-level dependency" (`lvis` test-only; no runtime deps).

## Decision drivers

- **ADR-0005 invariant.** No edits to `matching.rs` or
  `accumulate.rs`. Federated semantics flow through the existing
  `dt_ignore` and per-cell skip mechanisms.
- **Single dataset surface.** LVIS JSON is COCO JSON with extra
  per-image and per-category fields (AG1). Users don't learn a new
  dataset type to read an LVIS file.
- **Frequency buckets are not a `Breakdown` axis.** ADR-0016's
  `Breakdown` is f64-keyed; frequency (`r`/`c`/`f`) is a
  category-subset selector, a different tensor axis (K vs A).
- **Pre-1.0 freedom.** Project remains on 0.0.x (release-pace memo).
  Federated surface is provisional within the patch line; the parity
  contract (lvis-api pinned, strict-mode bit-equality on LVIS v1 val)
  is durable from first ship.
- **Boundary IoU on LVIS is already covered.** Boundary survey quirk
  **Q1** disposed `dilation_ratio` as exposed at every entry point;
  LVIS users pass `Boundary(dilation_ratio=0.008)` (AE3).

## Considered options

| Axis | Options (chosen in **bold**) |
|---|---|
| A. Workspace placement | A1 new crate `vernier-lvis`. **A2 modules in `vernier-core`.** A3 thin shim crate `vernier-lvis-api`. |
| B. Public-API integration | B1 new `LVISEvaluator`. **B2 `Evaluator` with federated-aware `CocoDataset`.** B3 top-level function. |
| C. Dataset model | C1 new `LvisDataset`. **C2 `CocoDataset` with optional federated fields.** |
| D. Frequency-bucketed AP | D1 hardcoded summarize branches. **D2 `StatRequest` with `CategoryFilter`.** D3 lift through `Breakdown`. |
| E. `LVISEval` shim | E1 ship `patch_lvis_api`. **E2 no shim; first-class API.** E3 migration guide only. |
| F. `max_dets` default | F1 per-dataset sentinel. **F2 explicit user choice.** |
| G. Parity oracle | **G1 `lvis` PyPI pinned + bowenc0221 for boundary.** G2 NumPy reference only. |

## Decision outcome

**Chosen: A2 + B2 + C2 + D2 + E2 + F2 + G1.**

### Workspace placement

LVIS code lives in `vernier-core`. New modules:

- `vernier-core::dataset::federated` — optional per-image
  `pos_category_ids` / `neg_category_ids` /
  `not_exhaustive_category_ids` and per-category `category_frequency`
  on `CocoDataset`.
- `vernier-core::summarize::category_filter` — `CategoryFilter` enum
  consumed by `StatRequest` for frequency-bucketed AP.
- `vernier-core::lvis_parity` — pinned LVIS oracle constants.

No new crate. The leaf-direction discipline that motivated
`vernier-mask` (ADR-0009) and `vernier-panoptic` (ADR-0025) doesn't
apply: LVIS shares the kernel, not just the codec. A separate crate
would force a circular dependency or duplicate the matching engine.

### Public Python surface

```python
@dataclass(frozen=True, slots=True)
class CocoDataset:
    """Existing dataset, extended with optional federated fields.

    Federated metadata activates LVIS-style evaluation when present;
    its absence is the COCO default. There is no `federated=True`
    flag — the dataset shape carries the semantics (ADR-0005's
    "behavior is a property of data, not a runtime switch" pattern).
    """

    # ... existing fields ...

    # All optional. None means "not provided"; the evaluator uses
    # COCO semantics on those cells.
    pos_category_ids: Mapping[ImageId, frozenset[CategoryId]] | None = None
    neg_category_ids: Mapping[ImageId, frozenset[CategoryId]] | None = None
    not_exhaustive_category_ids: Mapping[ImageId, frozenset[CategoryId]] | None = None

    # Per-category frequency tag for AP_r/c/f. None means
    # "frequency unknown" — affected summary entries become -1
    # (the LVIS undefined-AP sentinel; AA6 / AF6).
    category_frequency: Mapping[CategoryId, Frequency] | None = None

    @classmethod
    def from_lvis_json(cls, path: str | Path) -> CocoDataset: ...


class Frequency(StrEnum):
    """Boundaries per `lvis-api` eval.py:537-541 (AB1).

    Note these are the *code's* boundaries; the LVIS paper's prose
    ("1-10 / 11-100 / >100") is loose. The `frequency` field is
    precomputed at dataset publication; vernier reads it as-is and
    never derives it from `image_count` (AB2).
    """
    RARE = "r"       # < 10 train images
    COMMON = "c"     # [10, 100) train images
    FREQUENT = "f"   # ≥ 100 train images
```

`Evaluator` is **not** modified. LVIS users construct
`Evaluator(iou=Bbox() | Segm() | Boundary(dilation_ratio=0.008),
max_dets=(300,))` and pass an LVIS-loaded dataset. `max_dets=(300,)`
is explicit (F2): per-dataset defaults couple kernel choice and
dataset convention through a hidden lookup, surprising users who
override `max_dets`. The migration guide and tutorial set the value.

### Federated semantics through the matching engine

The orchestrator gains two responsibilities, both above the locked
spine:

1. **Per-(image, category) cell skip (AA4).** A DT is kept only if
   `C ∈ pos[I] ∪ neg[I]` (the lvis-api mechanism: filter DTs at
   prepare time, ev:99-103; GTs trivially satisfy `C ∈ pos[I]` since
   `pos` is derived from GTs at AA1). Cells with no GT and no DT
   produce no `eval_imgs` entry, which the accumulator already
   discriminates from non-`None` zero-cells (ev:336).
2. **Per-cell `dt_ignore` extension (AA3).** When
   `C ∈ not_exhaustive[I]`, every unmatched DT in the cell has its
   `dt_ignore` set to `true`. lvis-api ORs this into the existing
   area-bucket ignore mask (ev:269-278); vernier extends the same
   `dt_ignore` source the matching engine already reads.

`matching.rs` does not change. `accumulate.rs` does not change. The
ADR-0005 invariant is preserved by construction.

### Frequency-bucketed AP

`StatRequest` (the summarizer's plan unit) gains an optional
`category_filter`:

```rust
pub enum CategoryFilter {
    All,                        // current behavior
    Frequency(Frequency),       // r | c | f
    ByIds(BTreeSet<CategoryId>) // explicit subset (e.g., per-supercategory)
}
```

The summarizer's K-axis mean gains a pre-filter step: categories
outside the filter contribute nothing. Three new entries land in the
LVIS default plan: `lvis_ap_rare()`, `lvis_ap_common()`,
`lvis_ap_frequent()`. The plan is a public constant
`summarize::lvis_default()` parallel to `coco_detection_default()`;
users select it explicitly:
`Evaluator.evaluate(..., summary_plan=vernier.summarize.lvis_default())`.

Frequency buckets are *not* a `Breakdown` axis. `Breakdown` keys an
annotation by `f64` and slices A; `Frequency` keys a *category* by
enum and selects a subset of K. Forcing the round trip through `f64`
encodes a sum type that doesn't generalize to non-numeric axes
(ADR-0016's "Option 2 generic Bucket<K>" cons section).

### `LVISEval` drop-in shim

No `patch_lvis_api` in this ADR. The `lvis` package is much smaller
than `pycocotools` and its API is closer to vernier's; the migration
guide does the work. If demand materializes after ship, a follow-up
ADR (parallel to ADR-0007) adds the shim. Until then, the LVIS
migration guide is the documented path.

### Parity strategy

- **Vendored, pinned oracle.** `lvis` PyPI at a frozen version
  vendored at `tests/python/parity_lvis/oracle/lvis/` with
  `VENDORING.md` recording the version, the **transitively-pinned
  `pycocotools` version** (must align with vernier's
  `pycocotools==2.0.11` per ADR-0002 — see open question 6), and a
  fork plan.
- **Strict-mode parity claim.** Bit-equal AP / AP50 / AP75 / APs /
  APm / APl / APr / APc / APf and the four AR entries on LVIS v1 val
  against pinned `lvis`. The default plan has **13 entries** total
  (9 AP + 4 AR), not 9 + 1 (AF1, AF4).
- **Disposition vocabulary.** `strict` (bit-exact) / `aligned`
  (semantics match within tolerance) / `corrected` (opt-in fix) /
  `informational` (relied-upon upstream property, no vernier behavior
  follows) / `(deferred)` qualifier. The full row table is in the
  appendix; the global `ParityMode` enum
  (`crates/vernier-core/src/parity.rs`) is reused.
- **Constants module.** `crates/vernier-core/src/lvis_parity.rs`
  pins `LVIS_DEFAULT_MAX_DETS = 300` (ev:528, AC1),
  `LVIS_BOUNDARY_DILATION_RATIO_DEFAULT = 0.008` (boundary survey
  Q1, AE3), `LVIS_PARITY_EPS` (aligned-mode tolerance, value pinned
  by appendix open question 6), `ORACLE_LVIS_VERSION`,
  `ORACLE_PYCOCOTOOLS_PIN`. Drift between this module and
  `VENDORING.md` is a build failure (mirrors `boundary_parity.rs`).

### Numerical layout

`f64` end-to-end. LVIS shares the AP fold; ADR-0008's bbox-IoU
precedent is already in force. The K-axis filter (D2) is a boolean
mask applied before the unweighted mean; it doesn't change summation
order. The undefined-AP sentinel is **`-1`**, not `nan` or `0`
(ev:314, ev:441-442, AA6 / AF6) — an important migration-guide
warning since panopticapi (ADR-0025 W6) chose `0`/`ZeroDivisionError`
on the same shape.

## Consequences

- **Positive.** LVIS ships behind the same parity discipline as COCO
  and boundary IoU. ADR-0005's invariant survives without
  modification. Users coming from `lvis-api` find a familiar shape
  (LVIS-loaded dataset, max_dets=300, AP_r/c/f reported); users
  coming from `pycocotools` find LVIS as "COCO with extra fields on
  the dataset." `CategoryFilter` generalizes to other category-subset
  needs (per-supercategory AP, ablation subsets).
- **Negative.** New top-level test-only dep (`lvis` PyPI), which
  transitively pins `pycocotools` and may diverge from vernier's
  `==2.0.11` (open question 6). New parity surface with its own
  governance lifecycle. `CategoryFilter` is new public API; growing
  it later requires a deprecation cycle once the project moves to
  0.1.0+. Users loading an LVIS JSON expecting COCO defaults get
  federated semantics silently — the migration guide must lead with
  this.
- **Neutral.** `Evaluator` and `IouKind` are unchanged. `Summary`
  structure is unchanged (gains three new fields populated when
  frequency metadata is present). Boundary-IoU LVIS is already
  covered by Q1; this ADR doesn't re-litigate it.

## What this ADR explicitly does *not* decide

- **`patch_lvis_api` shim.** Deferred until demand materializes
  post-ship; follow-up ADR parallel to ADR-0007.
- **LVIS challenge-year variants.** vernier ships the canonical LVIS
  v1 protocol from the `lvis` package; challenge variants
  (e.g., 2021 max_dets changes) are follow-ups if requested.
- **LVIS v0.5 loader.** Different field names
  (`non_exhaustive_category_ids`) and frequency boundaries (AG2);
  out-of-scope for the initial ship.
- **Score normalization for long-tail models.** vernier evaluates
  whatever scores the user provides (AD1); per-class re-calibration
  is a user-side transform on the prediction file. Consistent with
  ADR-0018's "LVIS / open-vocabulary calibration" deferral.
- **Open Images V7 federated evaluation.** Group-of bounding boxes
  and a different per-image label semantics; not LVIS, not in scope.
- **`vernier eval --dataset lvis` CLI flag.** ADR-0015's verb
  extensibility applies; CLI path is a follow-up.
- **Federated `StreamingEvaluator` semantics.** Per-image batches
  flow through the existing ADR-0013 surface; per-batch federated
  metadata is a construction-time-only decision in v1.

## Pros and cons of the non-chosen options

- **A1** new crate forces a duplicate matching engine or a circular
  dep on `vernier-core`. **A3** thin shim is premature given E2.
- **B1** new evaluator duplicates `Evaluator`'s surface for no
  benefit; LVIS uses every knob identically. **B3** top-level
  function has nowhere to hang `parity_mode`/`max_dets`/IoU choice.
- **C1** new `LvisDataset` forces a parallel ingest pipeline for
  what is structurally COCO JSON.
- **D1** hardcoded summarize branches grow with every "give me AP
  for category subset X" request. **D3** through `Breakdown` is a
  type-system mismatch (f64-keyed vs categorical).
- **E1** `patch_lvis_api` introduces a `sys.modules` monkey-patch
  surface ahead of demand.
- **F1** per-dataset sentinel couples kernel choice and dataset
  convention via a hidden lookup.
- **G2** NumPy reference loses the "matches the reference everyone
  cites" claim.

## Links and references

- ADR-0001 — Triggers §"Affect the public API" and §"Add or remove
  a top-level dependency".
- ADR-0002 — Three-tier parity model (vocabulary reused; LVIS
  disposition table is its own).
- ADR-0005 — `Similarity` trait and matching-engine API lock.
  Validated: federated semantics flow through `dt_ignore` and
  orchestrator cell-skip, not through new kernel surface (AE4).
- ADR-0007 — `patch_pycocotools` shim. Pattern reserved for an
  eventual `patch_lvis_api`.
- ADR-0008 — Bbox IoU `f64` end-to-end (already in force; AE1).
- ADR-0010 — Boundary IoU isolated subsystem. Quirk **Q1** is the
  source of AE3.
- ADR-0011 — Discriminated kernel config. AH1 maps `iou_type`
  string to `IouKind`.
- ADR-0015 — `vernier-cli` (LVIS CLI path is a follow-up).
- ADR-0016 — Generalized `Breakdown` axis (validates D3 wrong-fit).
- ADR-0018 — Calibration. Deferral consistent with AD1.
- ADR-0025 — Panoptic-quality evaluation as a sibling crate. The
  asymmetry — panoptic gets a sibling crate, LVIS lives in
  `vernier-core` — is justified by AE4 (LVIS shares the AP fold;
  panoptic doesn't).
- `docs/engineering/pycocotools-quirks.md` — pycocotools survey
  (AC4 stable-sort, AE1 crowd asymmetry, AH3 stdout printing).
- `docs/engineering/boundary-iou-quirks.md` — boundary survey
  (Q1 → AE3).
- `tests/python/parity_lvis/oracle/VENDORING.md` — to be created
  alongside the implementation; pins the `lvis` version and the
  transitively-required `pycocotools` version.
- Gupta, Dollár, Girshick. *LVIS: A Dataset for Large Vocabulary
  Instance Segmentation.* CVPR 2019. arXiv:1908.03195.
- `lvis-dataset/lvis-api` — strict-mode oracle.

---

## Appendix: lvis-api quirks survey

Per-row evidence base for the parity strategy above. Each row
discovered by reading the oracle. Disposition column uses the
vocabulary defined in *Parity strategy*. Citations: `ev` =
`lvis/eval.py`, `rs` = `lvis/results.py`, `lv` = `lvis/lvis.py`;
`schema` = LVIS v1 JSON schema. Line numbers pinned against
`master` HEAD as of this ADR's drafting; the version in
`VENDORING.md` is authoritative once vendored.

The quirks namespace starts at **AA** because pycocotools used A–L,
boundary M–Q, and panoptic R–Z.

### AA. Federated cell-skip semantics

| # | Quirk | Source | Disposition |
|---|---|---|---|
| AA1 | Per-image `pos_category_ids` is **derived implicitly** from GT annotations: `pos[img] = {ann.category_id for ann in annotations[img]}`. Not a JSON field. A category with zero annotations on an image is *not* in `pos[img]`. | ev:91-94 | **strict**. Reproduced in `CocoDataset::from_lvis_json`. |
| AA2 | `neg_category_ids` is **explicit** in the LVIS JSON. Categories listed are verified absent; the cell evaluates with no GT and all DTs become FP candidates. | ev:90 (load), schema | **strict** (metric-defining; without it, recall on rare classes would be nonsense). |
| AA3 | `not_exhaustive_category_ids` is **explicit** in the JSON, semantically a subset of pos. Matched DTs still TP, unmatched DTs treated as ignore. The flag is consumed at ev:269-274 by ORing `cat_id ∈ img_nel[image_id]` into the area-bucket `dt_ig_mask`. | ev:97 (load), ev:269-278 (consume) | **strict**. Mapped to per-cell `dt_ignore = true` on every unmatched DT, threaded through the existing matching engine without kernel changes. |
| AA4 | **Cell skip rule.** For `(image I, cat C)`, the cell is evaluated only if `C ∈ pos[I] ∪ neg[I]`. Mechanism: ev:99-103 filters DTs (`if cat_id not in img_nl and cat_id not in img_pl: continue`); GTs trivially satisfy `C ∈ pos[I]` (pos is derived from them). With both empty, `evaluate_img` returns `None` (ev:197-198), and `accumulate` skips `None` cells (ev:336). | ev:99-103, 197-198, 336 | **strict** (denominator reduction is the core federated contract). |
| AA5 | Skipped cell (`None`) is structurally distinct from a non-None empty cell (e.g., `C ∈ neg[I]` with no DTs — produces a dict with empty arrays). The accumulator's filter `E = [e for e in E if not e is None]` keeps the latter and drops the former. | ev:336 | **strict**. vernier maps to "no `eval_imgs` entry" vs "entry with empty arrays". |
| AA6 | A category that is never in `pos[I] ∪ neg[I]` for any image contributes no `eval_imgs` row; its precision row stays at the **`-1`** initialized at ev:314 and is excluded from the unweighted mean by `s[s>-1]` at ev:444. The earlier draft of this row claimed `nan`; corrected against ev:314, 441-442. | ev:314, 441-444 | **strict**. The `-1` sentinel matches pycocotools' convention. |
| AA7 | Conflict between `not_exhaustive` and `neg`: by spec, `not_exhaustive[I] ⊆ pos[I]` and `neg[I] ∩ pos[I] = ∅`, so `not_exhaustive[I] ∩ neg[I] = ∅`. lvis-api does not validate; on overlap, the `not_exhaustive` branch wins at consumption (the `dt_ig_mask` is built from `img_nel` and ORed with the `dt_m == 0` mask). | ev:269-278 | **corrected**. vernier validates disjointness at `from_lvis_json` (`LvisFederatedConflict { image_id, category_id }`); strict reproduces the silent fall-through. |

### AB. Frequency buckets and per-bucket AP

| # | Quirk | Source | Disposition |
|---|---|---|---|
| AB1 | Each category carries `frequency: Literal["r","c","f"]`. The eval-code comments pin the boundaries: **`r: <10`, `c: [10, 100)`, `f: ≥100`** train images. (The LVIS paper's prose "1-10 / 11-100 / >100" is loose — a 10-image category is `c`, not `r`. The `frequency` field is precomputed at dataset publication; eval reads it as-is and never derives from `image_count`.) | ev:537-541 (boundary comments), ev:111 (read) | **strict**. |
| AB2 | `image_count` is also stored per category but **not consulted by the eval**. lvis-api does not cross-check it against `frequency`. | schema; not read in eval.py | **strict**. vernier preserves `image_count` on the dataset handle for downstream consumers; not used by the kernel. |
| AB3 | **AP_r / AP_c / AP_f** = unweighted mean of per-category precision over categories whose `frequency` matches the bucket. Categories whose precision row is `-1` (AA6) are excluded by `s[s>-1]`. | ev:431, 444 | **strict**. |
| AB4 | Overall **AP** = unweighted mean across **all** categories with the same `s>-1` filter. AP ≠ (AP_r + AP_c + AP_f) / 3 in general — depends on the bucket cardinalities. | ev:425-433, 444 | **strict**. |
| AB5 | The default plan reports AP / AP50 / AP75 / APs / APm / APl / APr / APc / APf; cross-products like `AP50_r` are **not** in it. | ev:454-462 | **strict** for the default; vernier's `CategoryFilter` mechanism composes naturally with IoU-threshold and area-bucket selectors, so users *can* request `AP50_r` via a custom plan. |
| AB6 | Missing `frequency` field: `_prepare_freq_group` does dict access at ev:111 → `KeyError`; an unrecognized value would hit `index()` at ev:112 → `ValueError`. | ev:111-112 | **corrected**. vernier surfaces `MissingFrequency { category_id }` at dataset load (not eval time); when `category_frequency` is `None`, AP_r/c/f entries become `-1` (the AP-undefined sentinel). |

### AC. Max-detections trim

| # | Quirk | Source | Disposition |
|---|---|---|---|
| AC1 | `Params.max_dets = 300` per image. `LVISResults` defaults its constructor's `max_dets` argument to 300, used by the input-side trim (AC2). | ev:528, rs:10 | **strict** as a default; users can override. |
| AC2 | **Input-side trim is per IMAGE, not per (image, category).** `LVISResults.limit_dets_per_image` (rs:73-84) buckets by `image_id`, sorts each bucket descending by score, keeps the top-K across **all categories combined**. The earlier draft of this row claimed per-(image, category) — corrected against the actual code. | rs:73-84 | **strict**. The trim is observable via `LVISResults.dataset['annotations']` and reproduced in `CocoDetections::lvis_trim` at the FFI boundary. |
| AC3 | The per-image granularity has a real consequence: a result file with 250 cat-A and 350 cat-B predictions on one image trims to **300 total** (top-300 across both classes), not 250 + min(350, 300) = 550. A frequent-class flood can crowd out rare-class predictions. | rs:73-84 | **strict** (open question 2 pins this with a fixture). |
| AC4 | Trim sort is `sorted(_anns, key=lambda ann: ann["score"], reverse=True)` — Python's Timsort, stable. **Different code path** from the matching-stage sort (eval.py uses `np.argsort(-scores, kind="mergesort")` at ev:174, 212, 344). Both are stable, so no AP impact for non-tied scores; tie-break order may differ between the two sites in degenerate inputs. | rs:81 (trim); ev:174, 212, 344 (matching) | **strict** for both sites. vernier mirrors with stable Rust sorts. |
| AC5 | If an image has `≤ max_dets` predictions, no trim applied. Setting `max_dets = -1` (or any negative value) disables the trim entirely (`if max_dets >= 0` guard). | rs:39-40, 79-80 | **strict**. |

### AD. Score handling

| # | Quirk | Source | Disposition |
|---|---|---|---|
| AD1 | LVISEval does **not** normalize scores per class; raw user scores feed the global descending sort. Long-tail models with miscalibrated rare-class scores rank rare detections low. Per-class score calibration is a separate tool (ADR-0018). | ev:174, 344 | **strict**. |
| AD2 | Scores read as Python `float` (== f64) from JSON; no normalization, clipping, or transform. | rs:30 | **strict**. |
| AD3 | NaN/inf scores: lvis-api does not validate. With `np.argsort(-scores, kind="mergesort")` ascending, NaN values sort to the **end** of ascending order — so in the descending-by-score reordering, NaN-score detections rank **lowest** (sorted to end), **not highest** as an earlier draft claimed. `+inf` ranks highest; `-inf` ranks lowest. | ev:174, 344 | **corrected**. vernier rejects NaN/inf at `CocoDetections::from_inputs` (`InvalidScore { detection_id, value }`); strict reproduces NumPy's NaN-to-end behavior. |

### AE. IoU and matching

| # | Quirk | Source | Disposition |
|---|---|---|---|
| AE1 | Bbox / segm IoU are computed via `pycocotools.mask.iou`, but **LVIS hardcodes `iscrowd = [int(False)] * len(gt)`** at ev:177 before calling the kernel — the pycocotools crowd-asymmetric path (pycocotools quirk **E1**) is bypassed regardless of annotation flags. The earlier draft of this row claimed crowd-asymmetric inheritance; corrected against ev:177. | ev:177-190 | **strict**. In practice LVIS v1 has no crowd annotations so the override is a no-op for valid data; for ill-formed datasets with crowd-flagged GT, LVIS produces *symmetric* IoU where pycocotools would produce asymmetric. |
| AE2 | The annotation-level `iscrowd` field is inherited from the COCO schema and **ignored by the eval** (consequence of AE1). Real LVIS v1 emits `iscrowd = 0` everywhere. | ev:177; schema | **strict**. |
| AE3 | Boundary IoU on LVIS is in the bowenc0221 fork — not in upstream `lvis-api`, which only supports `bbox`/`segm` (ev:25 raises on others). The fork's `LVISEval` exposes `dilation_ratio`; LVIS convention is `0.008` (boundary survey **Q1**). | boundary survey **Q1**; ev:25-26 | **strict** for the value; **corrected** for the API: `Boundary(dilation_ratio=0.008)` is explicit, not auto-set per F2. |
| AE4 | The matching loop (ev:233-265) is structurally the same as COCOeval's per-IoU-threshold greedy match with the early-break-on-ignore optimization (ev:247-248). Only orchestration changes are AA4 (cell skip) and AA3 (`dt_ignore` extension). `matching.rs` and `accumulate.rs` not edited. | ev:233-265 | **strict** by inheritance (ADR-0005 invariant). |

### AF. Aggregation and summary

| # | Quirk | Source | Disposition |
|---|---|---|---|
| AF1 | The default plan reports **13 entries**: AP, AP50, AP75, APs, APm, APl, APr, APc, APf (9 AP) + AR@300, ARs@300, ARm@300, ARl@300 (4 AR). Earlier draft said 9 + 1; corrected against ev:454-469. | ev:454-469 | **strict**. vernier ships it as `summarize::lvis_default()`. |
| AF2 | AP averages 10 IoU thresholds at `[0.5, 0.55, ..., 0.95]`. | ev:522-524 | **strict** by inheritance. |
| AF3 | Area buckets `s/m/l` use the same boundaries as COCO: `[0², 32²)`, `[32², 96²)`, `[96², 1e5²)`; "all" is `[0², 1e5²)`. | ev:529-534 | **strict**. |
| AF4 | AR is reported at `max_dets = 300` only — **AR@300 plus ARs/m/l@300** (4 entries). The COCO triplet `AR@1 / AR@10 / AR@100` is **not** in the default plan. Earlier draft said "single number"; corrected. | ev:464-469 | **strict**. |
| AF5 | `LVISEval.eval['precision']` is `(T, R, K, A)` — **no M-axis**, single max_dets (ev:314-316). `LVISEval.results` is a flat OrderedDict keyed by metric name; **there is no `categories` field**. Per-category AP is only in the precision tensor. Earlier draft claimed `(T, R, K, A, M)` and a `results['categories']` list; corrected. | ev:314-316 (shape), ev:454-469 (results keys) | **aligned**. vernier returns `Summary.per_class` from the underlying tensor; the public shape doesn't mimic an absent `lvis-api` field. |
| AF6 | When no categories survive a filter, `_summarize` returns **`-1`** (ev:441-442) — **not `nan`, not `0`**. Earlier draft said `nan`; corrected. Different from panopticapi (ADR-0025 W6 → ZeroDivisionError → vernier corrects to `0`). | ev:441-442 | **strict**. The migration guide leads with the cross-codebase difference: LVIS `-1` vs panoptic `0` vs uninitialized `nan`. |

### AG. JSON schema and dataset loading

| # | Quirk | Source | Disposition |
|---|---|---|---|
| AG1 | LVIS v1 JSON is structurally COCO JSON plus per-image `neg_category_ids` / `not_exhaustive_category_ids` and per-category `frequency` / `image_count`. Standard COCO loaders read it without error; extras are silently ignored. | schema | **strict**. `CocoDataset::from_lvis_json` projects the LVIS-specific fields into the optional federated metadata. |
| AG2 | LVIS v0.5 used different field names (`non_exhaustive_category_ids`) and frequency boundaries. v1 is the assumed schema. | historical | **out-of-scope** for v0.5 in the initial ship. |
| AG3 | Predictions are loaded by **instantiating `LVISResults` directly**: `LVISResults(lvis_gt, results, max_dets=300)`. There is **no `LVIS.loadRes()` method** (different from `COCO.loadRes`). The constructor applies the AC2 input-side trim. Earlier draft cited a non-existent `LVIS.loadRes()`; corrected against lvis.py. | rs:9-71; lvis.py (no `loadRes`) | **strict**. vernier's `CocoDetections::from_lvis_results` mirrors the constructor shape (GT + results + max_dets). |
| AG4 | Result files for segm use the same compressed-RLE format as COCO. | rs (transitively from pycocotools) | **strict** by inheritance. |
| AG5 | `LVIS.get_ann_ids` filters by area with **strict inequality on both ends** (`area > rng[0] and area < rng[1]`, lv:90-96). A GT with area exactly `32**2` falls into neither `small` nor `medium`. (In practice integer pixel-count areas almost never hit a boundary precisely; documented for property-test design.) | lv:90-96 | **strict**. |

### AH. API surface

| # | Quirk | Source | Disposition |
|---|---|---|---|
| AH1 | `LVISEval(lvis_gt, lvis_dt, iou_type="segm")` accepts **both file-path strings and pre-loaded `LVIS`/`LVISResults` objects**. `iou_type ∈ {"bbox","segm"}`; the upstream rejects others (ev:25-26). The bowenc0221 fork adds `"boundary"`. Earlier draft said file paths were not accepted; corrected against ev:28-40. | ev:14-40 | **aligned**. vernier's `Evaluator(iou=Bbox()|Segm()|Boundary(0.008))` consumes pre-loaded shapes; the difference is `IouKind` (ADR-0011) vs string literal. |
| AH2 | `LVISEval.run()` calls `evaluate(); accumulate(); summarize()` in one shot (vs COCOeval's three-step path). | ev:471-475 | **aligned**. vernier's `Evaluator.evaluate(...)` returns a `Summary` directly. |
| AH3 | `LVISEval.print_results()` prints one line per `self.results` key using a hand-rolled template (ev:477-507). The print format is the user-visible API for shell scrapers (mirrors pycocotools L5, ADR-0025 panoptic Y2). | ev:477-507 | **corrected**. vernier returns structured `Summary`; CLI subcommand reproduces under `--format text`. |
| AH4 | Errors are bare `Exception`/`KeyError`/`ValueError`/`TypeError` with f-string messages. Raise sites: ev:26 (unsupported `iou_type`), ev:33 (wrong `lvis_gt` type), ev:40 (wrong `lvis_dt` type), ev:184 (unknown `iou_type` for IoU), ev:450 (`accumulate()` not run). | ev:26, 33, 40, 184, 450 | **corrected**. Typed `LvisError` variants matching each raise site; strict replays via thin wrapper. |
| AH5 | Progress: `self.logger.info(...)` calls during evaluate / accumulate. Configurable via stdlib `logging`, but no structured progress signal. | ev:120-121, 297 | **corrected**. No stdout from vernier core; structured `progress=Callable[[int, int], None]` arg on `Evaluator`. |
| AH6 | The `lvis` package transitively pins `pycocotools`. Strict-mode parity is keyed to *both* pins. If they drift from vernier's `pycocotools==2.0.11` (ADR-0002), either: vernier accepts the drift and re-runs the pycocotools parity harness against the LVIS-pinned version, or pin to a slightly older `lvis` that ships with `pycocotools==2.0.11`. | `lvis` package metadata | **informational**. Pinned in `tests/python/parity_lvis/oracle/VENDORING.md` alongside `lvis`; survey gets a row if pycocotools drifts. |

### Open questions

The list below was the pre-acceptance gate; every entry was a
~30-minute fixture and is **resolved** in the implementation rollout
(PRs #114–#119). The acceptance status is recorded inline; the
ADR is now `accepted`.

1. **AC2 trim observability — RESOLVED (PR #118).** `tests/python/parity_lvis/test_lvis_trim.py::test_q1_500_single_category_dts_trim_to_300` plus the Rust unit test `dataset::tests::ac2_q1_trims_500_single_category_to_300` pin byte-equal trim output against `LVISResults` for a 500-DT single-category fixture.
2. **AC3 cross-class crowding — RESOLVED (PR #118).** `test_q2_cross_class_crowding_trims_to_300_total` and the Rust counterpart `ac3_q2_cross_class_crowding_keeps_300_total_across_classes` confirm 250 cat-A + 350 cat-B detections trim to **300 total** (the corrected reading from the appendix). The well-separated-scores fixture in `test_lvis_trim_uses_well_separated_scores_for_strict_membership` pins strict membership equality.
3. **AA7 federated conflict — RESOLVED (PR #115).** `test_aa7_pos_intersect_neg_raises` and `test_aa7_not_exhaustive_outside_pos_raises` plus the Rust `dataset::tests::aa7_*` tests lock in the corrected disjointness validation; `LvisFederatedConflict { image_id, category_id, detail }` surfaces the offender pair with a typed error.
4. **AB6 missing-frequency on partial datasets — RESOLVED (PR #115).** `test_ab6_missing_frequency_collects_all_offenders` and `dataset::tests::ab6_missing_frequency_collects_all_offenders` assert that vernier collects **all** missing-`frequency` categories and surfaces them in one sorted `MissingFrequency { category_ids }` error at load time.
5. **AF6 sentinel-vs-zero migration trap — RESOLVED (PR #117).** `test_q5_af6_lvis_minus_one_sentinel_on_empty_frequency_bucket` plus the Rust unit tests `summarize::tests::af6_empty_frequency_bucket_returns_minus_one_not_zero_or_nan` and `ab6_no_frequency_map_yields_minus_one_for_frequency_filtered_lines` pin the LVIS `-1` distinct from the panoptic ADR-0025 W6 corrected `0.0` and from the uninitialized `nan`. The cross-codebase contract is documented in `docs/migrate/from-lvis-api.md` (PR #120).
6. **AH6 pycocotools pin compatibility — RESOLVED (PR #114).** `lvis==0.5.3` declares no `pycocotools` requirement in its package metadata (verified via `importlib.metadata.requires('lvis')`); the oracle imports `pycocotools.mask` opportunistically and coexists cleanly with vernier's `pycocotools==2.0.11` (ADR-0002). The fallback ("pin older `lvis`") is unnecessary; `ORACLE_PYCOCOTOOLS_PIN` in `crates/vernier-core/src/lvis_parity.rs` mirrors `pyproject.toml`'s pin.

### Implementation rollout

The seven-PR rollout from `/home/dev/.claude/plans/imperative-drifting-toucan.md`:

- PR-1 #114 — Vendor `lvis-api` oracle + `lvis_parity` constants.
- PR-2 #115 — Federated metadata fields on `CocoDataset` + `from_lvis_json`.
- PR-3 #116 — Orchestrator cell-skip (AA4) + `not_exhaustive` `dt_ignore` (AA3).
- PR-4 #117 — `CategoryFilter` + `lvis_default()` 13-entry plan.
- PR-5 #118 — `CocoDetections::lvis_trim` per-image top-K (AC2/AC3/AC4/AC5).
- PR-6 #119 — LVIS v1 val parity smoke + `lvis_val_cache`.
- PR-7 #120 — Migration guide + this ADR's acceptance.

ADR-0005 invariant (no edits to `matching.rs`, `accumulate.rs`, or
`similarity/`) was preserved across every PR; verifiable via
`git diff --stat 28254b8..af7d74a -- crates/vernier-core/src/{matching,accumulate}.rs crates/vernier-core/src/similarity/` (empty).

### Known follow-up

**Dense-grid memory peak — structurally fixed by PR #179.** The
orchestrator grid was `Vec<Option<PerImageEval>>` at the time of
this ADR's acceptance — ~232 bytes per `(category, area, image)`
slot regardless of population, projecting to >22 GB peak resident
on full LVIS v1 val (1203 × 4 × 19809 ≈ 95M slots). PR-6's val
smoke subsamples to 1000 images to keep the measurement tractable.
PR #179 ("perf(evaluate): box per-cell results so EvalGrid skips
268 MB zero-init") shifted the slot type to
`Vec<Option<Box<PerImageEval>>>`: `Box`'s `NonNull` niche absorbs
the discriminant, so each unpopulated slot is 8 B and populated
cells pay one heap allocation. On the LVIS dimensions the structural
floor drops to 95M × 8 B ≈ 760 MB before populated cells land —
the 22 GB ceiling is no longer load-bearing. The original perf push
to sparse cell storage is unblocked; whether it's worth chasing
depends on what full-val measurement reveals (the bench-side LVIS
cell wired ahead of the next patch release, ADR-0026 follow-up,
captures the new peak in `bench/results/<sha>/<fp>/lvis/`).

**Bench-side strict parity at full val — open.** The release-mode
LVIS bench cell (`lvis_v1_val_perfect` × bbox, ADR-0033 + bench
matrix) shows ~0.06% per-cell divergence between vernier_lvis and
lvis-api on the (T, R, K, A) precision tensor — concentrated on
two categories (K=168, K=817) out of 1203. Vernier reports 1.0
on every divergent cell (the perfect-DT expectation); lvis-api
reports 0.987–0.999. The 1000-image parity smoke doesn't
populate enough multi-GT-per-image cells in those two categories
to surface the divergence. Hypothesis: score-tie ordering inside
`LVISResults` (perfect-DT detections all carry score=1.0, so
DT-to-GT tie-breaking is the load-bearing path); root cause not
yet pinned. Tracked alongside the bench cell's `divergence_report.json`.
