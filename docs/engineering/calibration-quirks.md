# Calibration metrics — quirks and dispositions (ADR-0018)

A working note (not an ADR) cataloguing the numerical and structural
quirks of the detection-family calibration summarizer (Shape 1 in
[ADR-0018](../adr/0018-calibration.md)). The disposition column is
ratified by ADR-0018; this survey is its per-row evidence base.

This survey is intentionally **independent of**
`docs/engineering/pycocotools-quirks.md` — calibration is a new metric
family with no pycocotools precedent (cocoeval's spine is a ranking
metric, calibration is a probability metric), so it gets its own
isolated three-tier table per the ADR-0010 pattern that
`boundary-iou-quirks.md` and `sem-seg-quirks.md` already follow. The
documents share only the `ParityMode` enum (`strict` / `aligned` /
`corrected`) — calibration retains the three-tier model because the
oracle is a clean-room numpy implementation with documented `f64`
re-association leeway, not pycocotools (where the 2026-05-10
amendment collapsed `aligned` into `strict`).

Each row below was disposed against the clean-room numpy oracle at
`tests/python/parity_calibration/numpy_oracle.py`. The disposition
column is one of:

- **strict** — vernier reproduces the oracle bit-exactly (`f64`-equal,
  no tolerance budget).
- **aligned** — vernier matches the *semantics*; user-visible outputs
  agree within a 4-ULP relative tolerance. The tolerance budget is the
  `f64`-comparison helper precedent in `tests/python/parity/harness.py`.
- **corrected** — vernier opts to fix this. Default behavior diverges
  from a naïve definition of the metric and the divergence is
  documented as an opinionated improvement; the legacy behavior is
  recoverable via the appropriate `CalibrationParams` knob.

Quirk-ID prefix: **`P`** (probability / calibration). The prefix is
chosen because A–L are taken by `pycocotools-quirks.md` and `B`
(boundary) is the only other crowded namespace. Because each survey
is isolated (separate oracle, separate fixtures, separate harness),
the local namespace is unambiguous — a row cited as "P3" in vernier
source comments refers to this document by construction.

The wired-in implementation lives at
`crates/vernier-core/src/calibration.rs`. Some `Wired-by` paths
below carry `:<TBD>` line markers — those will be filled in after
Unit 1 of the ADR-0018 implementation plan merges
(see `~/.claude/plans/adr-0018-calibration-metrics-zany-wall.md`).
Similarly, `numpy_oracle.py:<TBD>` placeholders will be replaced
with concrete line citations once Unit 4 lands the oracle.

## Source reference

- `tests/python/parity_calibration/numpy_oracle.py` (Unit 4) —
  clean-room numpy implementation; the parity oracle for this
  survey.
- `crates/vernier-core/src/calibration.rs` (Unit 1) — the wired
  kernel; bit-equal to the oracle under `ParityMode::Strict`.
- `crates/vernier-core/src/parity.rs` — pinned constants
  (`CALIBRATION_QUANTILE_METHOD`) and helpers (`quantile_linear`).
- `docs/adr/0018-calibration.md` — risk register (R1–R6),
  DETR-aware defaults section, parity-model section.

Conventions used below: `np.quantile`, `min_score`, `effective_n_bins`
and similar identifiers refer to vernier API surface unless prefixed
with an upstream module (e.g., `scipy.stats.norm.ppf`,
`sklearn.calibration.calibration_curve`).

---

## P. Probability / calibration quirks

| # | Quirk | Source | Disposition | Wired-by |
|---|---|---|---|---|
| P1 | Quantile bin-edges via `np.quantile(values, q, method='linear')` with `q = np.linspace(0, 1, n_bins + 1)`. The `linear` interpolation method (numpy's default, formerly `interpolation='linear'`) is pinned in `parity.rs` next to `linspace`; bit-equal to the oracle across numpy versions. | `numpy_oracle.py:<TBD>` | **strict** | `parity::quantile_linear` + `parity::CALIBRATION_QUANTILE_METHOD`; `calibration.rs:<TBD>`. |
| P2 | Detections matched into ignore regions (`dt_ignore[iou_index, d] == true` per the ADR-0013 cell store) drop from the histogram entirely — neither TP nor FP, no contribution to any bin. Mirrors the spirit of pycocotools quirk **B6** but applied to the calibration histogram rather than the PR curve. | ADR-0018 R3; `numpy_oracle.py:<TBD>` | **strict** | `calibration.rs:<TBD>` (the cell-load filter that drops `dt_ignore[t, d]` rows before binning). Fixture: `parity_calibration/fixtures/cal_ignore_regions/`. |
| P3 | Default `min_score = 0.05` filter on detection scores. Detections with `score < min_score` are dropped from the calibration histogram before binning. pycocotools has no analog (its PR curve has no score floor); the divergence is the DETR-no-object-tail rationale documented in ADR-0018 "Decision outcome → DETR-aware defaults". The naïve default (`min_score = 0.0`) is recoverable via `CalibrationParams { min_score: 0.0, .. }`. | ADR-0018 "DETR-aware defaults" | **corrected** | `calibration.rs:<TBD>` (score filter pass). `CalibrationParams::min_score` default = `0.05`. |
| P4 | 95% Wilson confidence intervals on per-bin accuracy. `z = 1.959963984540054` (the `f64` value of `scipy.stats.norm.ppf(0.975)`); pinned as a literal in the kernel rather than re-derived to keep the oracle bit-equal across scipy versions. Formula: `(p̂ + z²/(2n) ± z√(p̂(1-p̂)/n + z²/(4n²))) / (1 + z²/n)`. | ADR-0018 R2; `numpy_oracle.py:<TBD>` | **strict** | `calibration.rs:<TBD>` (`wilson_interval` helper, `z` pinned as `pub(crate) const`). |
| P5 | Duplicate-quantile-edge merging under clustered scores. When `np.quantile` returns edges with `edge[i] == edge[i+1]` (the R1 degeneracy), the offending bins are merged in-place and the result surfaces an `effective_n_bins` field with `effective_n_bins <= n_bins`. Bit-equal to the oracle, which performs the same merge. | ADR-0018 R1; `numpy_oracle.py:<TBD>` | **strict** | `calibration.rs:<TBD>` (`merge_duplicate_edges`); `CalibrationSummary::effective_n_bins`. Fixture: `parity_calibration/fixtures/cal_overconfident/` exercises bimodal clustering. |
| P6 | Default per-class aggregation is **macro** (weights rare classes equally; `ece_macro = mean_k(ece_k)`). Micro aggregation (`ece_micro` over the pooled histogram) is available via `CalibrationParams { per_class_aggregation: Aggregation::Micro, .. }`. The macro default diverges from sklearn-style implementations (which default to micro / sample-weighted); the safety-case rationale is in ADR-0018 R4. | ADR-0018 R4; `numpy_oracle.py:<TBD>` | **corrected** | `calibration.rs:<TBD>` (`PerClassAggregation` enum, macro arm default). |
| P7 | NaN propagation for zero-count bins: `accuracy`, `mean_score`, `gap`, `ci_lo`, `ci_hi` all emit `f64::NAN`; the `count` column emits `0u64`. Downstream consumers filter on `count > 0` rather than reading sentinel `-1` values (cf. pycocotools quirk **C5**'s `-1` sentinel for absent categories). | ADR-0018 R2; `numpy_oracle.py:<TBD>` | **strict** | `calibration.rs:<TBD>` (zero-count-bin branch in the reliability-table builder). |
| P8 | Area-bucket invariance: calibration uses the `all` area bucket only — there is no per-area calibration breakdown. Calibration is not an AP-style breakdown by object size; the keypoints-specific no-`small`-bucket quirk (pycocotools **D5** / ADR-0012) is therefore irrelevant to calibration. | ADR-0018 "Shape 1" footnote on keypoints | **strict** | `calibration.rs:<TBD>` (the kernel reads cell-store rows at the `all` area-index slot; no area-axis parameter on `CalibrationParams`). |
| P9 | Clopper-Pearson CI is deferred to Phase 2. The current kernel returns `EvalError::Unsupported` if `ConfidenceKind::ClopperPearson` is selected; only `ConfidenceKind::Wilson` is implemented. Documented as a Phase-2 follow-up rather than a silent fallback to Wilson, so users cannot accidentally read Wilson numbers under a Clopper-Pearson label. | ADR-0018 "DETR-aware defaults"; ADR-0018 R2 | **aligned** (Phase-2 follow-up) | `calibration.rs:<TBD>` (the `ConfidenceKind::ClopperPearson` match arm returns `Err(EvalError::Unsupported)`). |
| P10 | Keypoints footnote — histogram denominator under `max_dets`. The kernel does **not** apply `max_dets` internally; the cell store handed to `summarize_calibration` already reflects whichever `max_dets` cap the streaming evaluator (or the one-shot `evaluate`) applied. Under the keypoints-canonical `max_dets=[20]` cap, the denominator shifts vs the detection-canonical `max_dets=[1, 10, 100]` — surfaced transparently because the cell store is the only input. A misconfigured streaming evaluator that passes an unintended cap produces surprising calibration numbers, but the divergence is in the caller, not the kernel. | ADR-0018 "Shape 1" footnote on keypoints | **strict** | `calibration.rs:<TBD>` (no `max_dets` parameter on `CalibrationParams`; the cell store is authoritative). Fixture: `parity_calibration/fixtures/cal_keypoints_smoketest/`. |

---

## Notes

### Cross-reference to `sklearn.calibration.calibration_curve`

ADR-0018 explicitly calls out `sklearn.calibration.calibration_curve`
as a **sanity cross-check, not the oracle**. The bin-edge semantics
differ in two ways:

1. **Default binning strategy.** sklearn's `strategy='uniform'`
   (equal-width) is the documented default; vernier's quantile
   default (P1) was chosen because equal-width produces garbage ECE
   on the bimodal score distributions DETR-family detectors emit
   (ADR-0018 "DETR-aware defaults"). sklearn does expose
   `strategy='quantile'`, but its quantile path uses the same
   `np.quantile(..., method='linear')` semantics as vernier (P1), so
   under matched parameters the bin edges agree.
2. **Score filtering.** sklearn has no `min_score` cutoff;
   `min_score=0.0` recovers a comparable input set (P3).

The sanity-cross-check test
(`tests/python/parity_calibration/test_sklearn_crosscheck.py`,
`not slow` marker) runs sklearn with matched parameters and asserts
agreement under a relaxed tolerance — documentation-grade evidence,
not a CI gate.

### Tolerance policy for `aligned` mode

The 4-ULP relative tolerance for `aligned`-mode rows is computed via
the `f64`-comparison helpers in `parity.rs` (precedent: the existing
tolerance kit consumed by `tests/python/parity/harness.py`). Only P9
currently carries an `aligned` disposition (and only in the trivial
"Phase-2 follow-up" sense — the code path errors today rather than
emitting drifted numbers). The remaining rows are bit-equal under
`ParityMode::Strict`.

### Quirks-survey discoverability

Every `P<n>` cited in `crates/vernier-core/src/calibration.rs` source
comments must appear as a row in this table. Verification:
`rg "P[0-9]+" docs/engineering/calibration-quirks.md` should be a
superset of `rg "P[0-9]+" crates/vernier-core/src/calibration.rs`.
