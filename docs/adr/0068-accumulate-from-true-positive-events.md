# ADR-0068: Accumulate PR curves from their true-positive events

- **Status:** proposed
- **Date:** 2026-10-29
- **Deciders:** @NoeFontana
- **Consulted:** —
- **Informed:** all contributors

Supersedes the *Bucket lookup for recall thresholds* clause of
[ADR-0004](0004-numerical-layout-policy.md) (quirk **C1**). The rest of
ADR-0004 stands.

## Context and problem statement

ADR-0004 fixed how the accumulator finds each recall threshold on a PR
curve: `partition_point` over the cumulative-recall slice with the
predicate `|r| r < threshold`, identical to numpy's
`searchsorted(rc, recThrs, side='left')`. That presumes the curve exists:
for every `(t, k, a, m)` lane the accumulator materialized `rc` and `pr` at
every detection (a division on every true and false positive), ran
**C2**'s right-to-left running max over all of it, then binary-searched
each of the 101 recall thresholds.

On dense-query detectors that is a large share of an evaluation: on
RF-DETR Nano over COCO val2017 (514k detections) `accumulate` took 132 ms
of a 520 ms single-thread run, and partitioned evaluation pays it once
per slice.

## Decision drivers

- Output must stay bit-identical to pycocotools 2.0.11 (the parity
  contract): same f64 expressions, same envelope, same lookup.
- The hot loop runs once per detection per threshold lane; it should be
  branch-light and free of division.
- One algorithm for the packed and the dense accumulators.

## Considered options

1. **Keep the curve** (ADR-0004 as written) and micro-optimize the sweep.
2. **Sweep true-positive events**: record only where true positives land
   and read every recall threshold off them.

## Decision outcome

Chosen option: **2**, because it removes the per-detection division, the
curve stores, the full-length envelope pass and the binary searches, and
two exact facts make it bit-identical:

- **The envelope peaks only at a true positive.** Between two true
  positives precision only falls — a false positive grows the
  denominator, an ignored detection (**C7**) changes nothing — so C2's
  running max over the curve equals the running max over the true
  positives. Each is computed with the same expression,
  `tp / (tp + fp + eps)`, so the same bits.
- **A recall threshold is a true-positive count.** `rc` changes only at a
  true positive, to `k / npig` after the `k`-th one, so the first position
  with `rc >= r` (**C1**) is the `k`-th true positive for the smallest `k`
  with `k / npig >= r` under the same f64 division. `k` depends on `npig`
  alone and is resolved once per `(category, area)` for every threshold
  and `maxDet` lane. `r <= 0` lands on position 0; `NaN` lands past the
  curve, as numpy sorts it; a `k` beyond the lane's true positives is
  past the curve (**C3**).

Where thresholds fit the packed grid's 16-bit word (at most 16), one pass
sweeps every lane: the per-lane false-positive counts are a 16-wide add
and each hit records an event on its lane.

### Consequences

- **Positive:** `accumulate` on RF-DETR Nano 132 → 52 ms (−60%); a
  two-slice partitioned evaluation 1056 → 726 ms (−31%). Bit-identical.
- **Negative:** the lookup is no longer the textbook `searchsorted`; its
  equivalence is an argument, so it is pinned by tests against a
  line-by-line transcription of pycocotools' `accumulate`
  (`accumulate::reference`): per lane over random lanes and recall grids,
  and per grid over random cells, crowds, area ranges, `maxDet`
  truncation and both flag layouts.
- **Neutral:** the public `accumulate` / `accumulate_grid` signatures and
  outputs are unchanged.
