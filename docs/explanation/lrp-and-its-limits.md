# LRP: what it answers, and what it doesn't

vernier ships an LRP / oLRP decomposition (Oksuz et al., ECCV 2018;
TPAMI 2021) via `vernier.instance.optimal_lrp` and
`vernier.panoptic.optimal_lrp`. Three components — `oLRP_Loc`,
`oLRP_FP`, `oLRP_FN` — collapse to a single oLRP score per class, and
the search returns the per-class confidence threshold `tau` at which
the model achieves that score. A worked example tutorial
(`docs/tutorials/debugging-with-lrp.md`, planned alongside this page)
walks the *how*; this page is the *what for, and what against*. Long
enough that an issue starting with "the tau looks wrong" can be
closed by a link.

## What oLRP answers

**What confidence threshold should I deploy this model at?** The
"o" in oLRP is *optimal*: the metric searches over a grid of
confidence thresholds and reports the one that minimizes localization-
recall-precision error. The reported `tau` is what a practitioner
would set on the model to reproduce the reported behavior. That is
the headline deliverable — not the score in isolation, but the
*(score, threshold)* pair.

**How does my model trade off the three error modes?** oLRP
decomposes into `oLRP_Loc` (localization error on TPs, computed as
`1 − mean(IoU)` or `1 − mean(OKS)` over the matched pairs), `oLRP_FP`
(false-positive rate at the optimal threshold), and `oLRP_FN`
(false-negative rate at the optimal threshold). The three are
orthogonal in a way AP's single number isn't — a model with `oLRP_Loc
= 0.1, oLRP_FP = 0.4, oLRP_FN = 0.1` is making a very different
mistake than `oLRP_Loc = 0.4, oLRP_FP = 0.1, oLRP_FN = 0.1`, and the
fix is different even though both might have similar AP.

**Comparing two models on the same data at their respective
operating points.** Each model's `tau` is found independently, so
the comparison is "model A at its best deployable threshold versus
model B at its best deployable threshold." That is the comparison a
practitioner deploying one of them cares about.

## How LRP differs from AP

AP is a single number that integrates the precision-recall curve over
all confidence thresholds. It does not report the threshold; the
curve *is* the threshold-swept signal, and the integration averages
across it. oLRP, by contrast, picks a single point on the threshold
axis (the optimum) and reports the error structure at that point.

The two answer different questions. AP tells you "how good is this
model's ranking, across all operating points?" oLRP tells you "at
this model's best operating point, what's costing it?" A model with
strong AP can have surprising error structure at its optimal tau;
oLRP surfaces what AP averaged out.

## How LRP differs from TIDE

TIDE ([`tide-and-its-limits.md`](tide-and-its-limits.md)) decomposes
the gap between a model's mAP and the perfect-mAP upper bound into
six bins (Cls, Loc, Both, Dupe, Bkg, Missed) by running corrected
accumulations per bin. It is a *mAP-relative* decomposition — every
number is a delta against baseline.

oLRP is not mAP-relative. The three components are computed at a
single operating point and reported in their own units (`oLRP_Loc`
is the average IoU gap; `oLRP_FP` and `oLRP_FN` are rates at that
operating point). Three differences fall out:

- **Continuous IoU, not binned.** TIDE's bin assignment is a
  phase diagram over `(t_b, t_f)`. oLRP's localization term is a
  mean of similarity scores across matched TPs — no binning, no
  threshold defining "almost matched". That is why oLRP-on-OKS
  ships (ADR-0045) where TIDE-on-OKS does not (ADR-0024): the
  continuous-similarity geometry transfers; the phase diagram does
  not.
- **Ships a tau.** TIDE picks its `t_f` and `t_b` *a priori* (per
  ADR-0022 defaults). oLRP picks `tau` from data, per class. That
  makes oLRP a deployment-relevant answer in a way TIDE is not —
  TIDE tells you the shape of your errors, oLRP tells you the
  threshold to set.
- **No Cls bin.** TIDE has Cls (and Both) bins that compare
  detections against GTs of other classes. oLRP is a per-class
  metric that does not look across classes. The cost: oLRP is
  silent on cross-class confusion (see "Limits" below). The
  benefit: oLRP works cleanly on single-class workloads (notably
  COCO keypoints) where TIDE's cross-class bins are
  algorithmically zero.

The two are complementary. Read TIDE to understand error structure;
read oLRP to understand operating point.

## Limits

### oLRP is single-class-tolerant — it doesn't see cross-class confusion

The components are computed per class, and the dataset-level
reduction is a mean over classes. If a model systematically
mislabels person-on-bike as just person, oLRP's per-class accounting
shows up as missing-class-bicycle FNs and wrong-class-person FPs —
both penalized in the right components numerically, but the *causal*
link (one class's FP is the same instance as another class's FN) is
invisible to oLRP. TIDE's Cls bin or the
`vernier.instance.confusion_matrix` output is what you want for
cross-class diagnosis.

### Tau is per-class — multi-class deployment needs per-class thresholds in production

A reported `tau = 0.42` for `person` and `tau = 0.31` for `dog` means
that to reproduce the oLRP numbers in production you set those two
thresholds *separately*. Many deployment stacks use a single
confidence cutoff. There is no canonical aggregation of per-class
taus into a single threshold — taking the mean or median loses the
guarantee that the reported score is achievable. If your deployment
constraint is "one threshold for the whole model," oLRP is reporting
an upper-bound on achievable performance, not your achievable
performance.

The result table reports per-class taus exactly so this gap is
visible.

### Empty-TP classes report `tau = NaN, oLRP = 1.0`

A class with no TPs at any threshold in the grid (the model never
found a true instance of it) carries `oLRP = 1.0` (the worst
possible score) and `tau = NaN`. These are flagged in the result
table separately so they do not silently pull the dataset-level
mean. If half a dataset's classes are empty-TP, the headline oLRP
is meaningless before you filter; the per-class table is the safe
read in that regime.

### Tau search is discrete — the reported threshold is grid-precision

oLRP's `tau` comes from a discrete grid (default `0.01` step over
`[0.0, 1.0]`, per [ADR-0044](../adr/0044-lrp-thresholds-and-tau-grid.md)).
The reported threshold is accurate to ±half-a-grid-step — fine for
deployment intent at the practitioner's typical 1%-increment tuning
granularity, but not exact. Two consecutive grid points are typically
within `<0.001` of LRP from each other in dense regions, so the
reported score is reliable; the *exact* optimum within the grid step
is not pinned.

If you need a different grid, pass `tau_grid=np.linspace(0, 1, 1001)`
per call. The result's `config` field records the resolved grid.

### Argmin-tau ties resolve to the larger threshold (deliberately)

When two thresholds yield identical oLRP, the larger one wins. The
reading is "given a tie, deploy the more conservative threshold —
fewer FPs, slightly less recall, same score." This is a design
choice, not a property of the metric — `kemaloksuz/LRP-Error`
documents its tie-handling ambiguously, and other implementations
may pick the smaller. ADR-0043 commits us to "larger wins" because
it matches deployment intuition.

## Where it disagrees with `kemaloksuz/LRP-Error` on real models

The LRP-Error repo is the first-party reference implementation
(Oksuz wrote both the paper and the code). vernier vendors it
commit-pinned as a CI tripwire (one synthetic fixture cross-checks
`|oracle − kemaloksuz| < 1e-6`) — see
[ADR-0043](../adr/0043-lrp-oracle-and-namespace.md). The tripwire is
a sanity gate, not a parity contract: where the two diverge on real
data, the numpy oracle at `tests/python/oracle/lrp/oracle.py` is
authoritative.

**This section is a placeholder.** The empirical comparison against
`kemaloksuz/LRP-Error` on real models on COCO val2017 is gated on
the deferred whole-dataset parity infrastructure
(`project_coco_val_regression.md`); vernier does not commit COCO
val data in CI per the licensing policy. Once the 0.5.x follow-up
runs the comparison, this section will be filled in with the
specific divergence patterns observed — the categories where ties
resolve differently, the low-recall edge cases LRP-Error handles
ambiguously, the empty-TP class flagging differences.

The pattern is identical to the equivalent paragraph in
[`tide-and-its-limits.md`](tide-and-its-limits.md): we know where to
look, the comparison is on the roadmap, and the gap closes when
infrastructure lands.

## Where the kernel-specific defaults come from

[ADR-0044](../adr/0044-lrp-thresholds-and-tau-grid.md) is canonical;
in summary:

- **bbox** — `tp_threshold=0.5`, `tau_grid=0.01-step`. Both anchored
  on the Oksuz TPAMI 2021 paper. The paper's recommended operating
  point reproduces against these defaults on canonical models.
- **segm** — same `tp_threshold=0.5`. *Tentative*: argued from the
  bound that segmentation IoU is at most bbox IoU on the same
  instance, and tracks within ~10% on standard models. Empirical
  anchoring on real models is deferred to a 0.5.x follow-up.
- **boundary** — same `tp_threshold=0.5` at `dilation_ratio=0.02`,
  tentative. Note: this is the *opposite* posture from
  ADR-0022's TIDE boundary default (which carves a tighter floor),
  because LRP's `tp_threshold` is the "this is a TP" line, not a
  phase-diagram cutoff. Different metric, different role for the
  same number.
- **keypoints** — same `tp_threshold=0.5` on OKS. Anchored on OKS
  `0.5` as the meaningful-match operating point. Empirical
  anchoring deferred to the same 0.5.x follow-up.

The "tentative" warning in ADR-0044 is honest; if your workload's
models cluster differently, override per call. The result's
`config.tp_threshold` makes the resolution audit-trail available.

## Where the validation comes from

The Rust core is gated against a numpy oracle at
`tests/python/oracle/lrp/oracle.py`. Hand-computed assertions on
small fixtures pin oracle correctness; `1e-9` parity between Rust
and oracle pins implementation correctness; a single CI tripwire
fixture cross-checks `|oracle − kemaloksuz| < 1e-6` to catch the
whole-class regressions a self-consistent oracle would not.

Real-model behavior on COCO val2017 is the same deferred-validation
story as TIDE — gated on the whole-dataset parity infrastructure
(`project_coco_val_regression.md`, the equivalent follow-up doc).
Until that runs, the LRP entry points are correct against the
oracle and sanity-gated against the published reference; the
empirical real-model behavior assertion is on the 0.5.x line.

## See also

- `docs/tutorials/debugging-with-lrp.md` (planned) — worked example
  of using oLRP on a model, end-to-end.
- [ADR-0043](../adr/0043-lrp-oracle-and-namespace.md) — correctness
  model and namespace.
- [ADR-0044](../adr/0044-lrp-thresholds-and-tau-grid.md) — threshold
  and tau-grid defaults.
- [ADR-0045](../adr/0045-lrp-keypoints-shipped.md) — why LRP-on-OKS
  ships where TIDE-on-OKS does not.
- [`tide-and-its-limits.md`](tide-and-its-limits.md) — the sibling
  decomposition; read both for the full diagnostic picture.
- [`why-no-per-image-ap.md`](why-no-per-image-ap.md) — same
  statistical reasoning for "scale matters" applied to a different
  metric.
