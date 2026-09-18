# ADR-0058: Dispose of keypoints-under-`useCats=0` as `corrected` (quirk F6)

- **Status:** proposed
- **Date:** 2026-09-17
- **Deciders:** @NoeFontana
- **Consulted:** —
- **Informed:** —

## Context and problem statement

`pycocotools.cocoeval.COCOeval.evaluate` computes the per-cell similarity
matrix over a category axis that collapses when category labels are
ignored:

```python
catIds = p.catIds if p.useCats else [-1]                       # ce:142
self.ious = {(imgId, catId): computeIoU(imgId, catId) ...}
```

`computeIoU` copes with the `-1` sentinel because it forks on `p.useCats`
and gathers annotations across every real category (`ce:165-170`).
`computeOks` — the keypoints kernel — does not. It indexes
`self._gts[imgId, catId]` / `self._dts[imgId, catId]` directly
(`ce:195-196`), finds nothing under the `-1` sentinel, and returns `[]`.

`evaluateImg` *does* have the `useCats` fork (`ce:241-246`), so the cell
is still populated with the image's GTs and DTs; only the IoU matrix is
empty. The guard at `ce:266`
(`ious = self.ious[imgId, catId][:, gtind] if len(self.ious[imgId, catId]) > 0 else ...`)
then hands the matching loop an empty array, `if not len(ious)==0` skips
matching outright, and the run finishes with every DT an unmatched FP and
every GT an unmatched FN. **Keypoint AP and AR come out 0** — even for a
byte-identical prediction. Verified against `pycocotools==2.0.11`:

```
pycocotools useCats=1 stats: [ 1.  1.  1.  1. -1.  1.  1.  1.  1. -1.]
pycocotools useCats=0 stats: [ 0.  0.  0.  0. -1.  0.  0.  0.  0. -1.]
```

Vernier's L4 collapse (`use_cats=false` → one virtual category bucket)
is kernel-agnostic: it gathers correctly for OKS as it does for bbox.
Before this ADR, vernier therefore reported AP 1 on the same input — in
**strict** mode, which is the drop-in's default (ADR-0007) and is
supposed to reproduce pycocotools bugs included.

So this is not a question of whether to fix a vernier bug; it is a
disposition question under ADR-0002 for a pycocotools quirk vernier had
not yet catalogued.

## Decision drivers

- ADR-0002: every pycocotools behavior gets exactly one disposition, and
  `strict` means bit-exact *including* upstream bugs.
- ADR-0007: `parity_mode="strict"` is the drop-in's default, because the
  drop-in is the migration path. A migrating user's numbers must not
  move silently.
- The blackout is unambiguously an omission, not a semantic: the sibling
  code path (`evaluateImg`) gathers across categories in the very same
  configuration, so the intent was category-agnostic matching.
- Blast radius is small. COCO keypoints ships exactly one category
  (`person`), so `useCats=0` is close to degenerate there and has no
  known real caller. The quirk bites the non-COCO keypoint datasets
  (multi-category pose: animal/vehicle keypoints, CrowdPose-style forks)
  where category-agnostic scoring is a meaningful request.

## Considered options

1. **Document only** — add the quirk row, change no code. Vernier keeps
   reporting AP 1 in both modes.
2. **`strict` disposition** — reproduce the blackout in both modes.
3. **`corrected` disposition** — blackout under `ParityMode::Strict`,
   category-agnostic gather under `ParityMode::Corrected`.

## Decision outcome

Chosen option: **Option 3 (`corrected`)**, because it is the only option
that satisfies both halves of the contract: strict callers get the
oracle's number (bugs included) and corrected callers get the answer
`useCats=0` is asking for.

Implementation is a single predicate,
`evaluate::strict_oks_use_cats_blackout(kernel, use_cats, parity_mode)`,
consulted at the one place each evaluate path calls `kernel.compute`
(`evaluate.rs`, `evaluate_parallel.rs`). When it fires, the already
zero-filled per-cell IoU scratch is left untouched. An all-zero `g × d`
matrix is observationally identical to pycocotools' empty `ious`: quirk
**B1**'s `min(t, 1 - 1e-10)` seed admits no match at IoU 0, so the
matching loop produces the same all-FP outcome, while the cell keeps the
GT/DT bookkeeping `evaluateImg` also keeps.

Scoping is via the existing `EvalKernel::is_keypoints()` marker, so
bbox / segm / boundary — which route through `computeIoU` and have the
fork — are untouched.

### Consequences

- **Positive:** the drop-in's default path is now bit-equal to
  pycocotools for keypoints under `useCats=0`, closing a silent strict-mode
  divergence. Corrected mode gains a genuinely useful behavior for
  multi-category pose datasets.
- **Negative:** this is a behavior change in the default (strict) path —
  keypoint AP under `useCats=0` goes from 1 to 0 on a perfect
  prediction. That looks like a regression and reads like one in a diff.
  It is the contract working as designed, but it needs the quirk row and
  this ADR to be legible. Mitigated by the near-certain absence of real
  callers (single-category COCO keypoints).
- **Neutral:** one more `corrected` row on the disposition table (F6),
  and a keypoints-only branch in two hot loops — a `bool` already
  computed once per call, not per cell, in the sequential path.

## Pros and cons of the options

### Option 1 — document only

- 👍 Zero code, zero risk, zero behavior change.
- 👎 Leaves a known strict-mode divergence in the shipped drop-in, which
  is exactly the failure mode ADR-0002 exists to prevent. A documented
  divergence that nothing enforces decays into an undocumented one.

### Option 2 — `strict` both ways

- 👍 Simplest possible story: vernier always equals pycocotools.
- 👎 Propagates an omission bug to users who explicitly asked to ignore
  category labels, with no opt-out. Contradicts the precedent set by
  D1 / I2 / K1, which all correct plain upstream mistakes.

### Option 3 — `corrected` (chosen)

- 👍 Both audiences served; consistent with every other `corrected` row.
- 👎 Costs a default-path behavior change and a kernel-scoped branch.

## Links and references

- Related ADRs: [0002](0002-three-tier-parity-model.md) (disposition
  model), [0007](0007-patch-pycocotools-policy.md) (strict is the
  drop-in default), [0012](0012-oks-keypoints-surface.md) (keypoints /
  OKS surface; quirks D2, D5, F1–F5, L8).
- Quirk row: `docs/engineering/pycocotools-quirks.md` § F, row **F6**.
- Upstream source: `pycocotools==2.0.11`, `cocoeval.py` lines 142,
  165-170, 195-196, 241-246, 266.
