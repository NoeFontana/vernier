# `federated_min` — first end-to-end LVIS parity fixture (PR-3)

Exercises every federated branch in one ~30-line fixture. Diff target:
`LVISEval` strict-mode `eval_imgs` and `(T, R, K, A)` precision tensor
at `max_dets=300`. Quirk citations are appendix rows of
`docs/adr/0026-lvis-support.md`.

## Layout

| image | width × height | GTs (id, cat, bbox)         | neg | not_exhaustive |
|------:|---------------:|------------------------------|-----|----------------|
| 1     | 100 × 100      | (1, cat 1, [0,0,20,20])      | [2] | []             |
| 2     | 100 × 100      | (2, cat 1, [10,10,20,20])    | []  | [1]            |

| category | frequency |
|---------:|-----------|
| 1 alpha  | `f`       |
| 2 beta   | `c`       |
| 3 gamma  | `r`       |

## Detections

| image | cat | score | bbox             | branch exercised |
|------:|----:|------:|------------------|------------------|
| 1     | 1   | 0.95  | [0,0,20,20]      | TP — pos cell, normal eval |
| 1     | 2   | 0.72  | [50,50,10,10]    | FP — `cat 2 ∈ neg[1]`, cell evaluates with no GTs |
| 1     | 3   | 0.88  | [60,60,10,10]    | dropped — `cat 3 ∉ pos[1] ∪ neg[1]` (**AA4** cell skip) |
| 2     | 1   | 0.91  | [10,10,20,20]    | TP — pos cell, but `cat 1 ∈ not_exhaustive[2]` (AA3 applies to **unmatched** DTs only, so this matched DT is unaffected) |
| 2     | 1   | 0.30  | [60,60,10,10]    | unmatched, but `cat 1 ∈ not_exhaustive[2]` → `dt_ignore=true` (**AA3**) |

## Branch coverage

- **AA1** — `pos[1] = {1}`, `pos[2] = {1}` derived from GTs.
- **AA2** — `neg[1] = {2}`, `neg[2] = {}` read verbatim.
- **AA3** — image 2 × cat 1: unmatched DT (`score=0.30`) gains
  `dt_ignore = true`; matched DT (`score=0.91`) keeps `dt_ignore = false`.
- **AA4** — image 1 × cat 3: cell produces no `eval_imgs` entry
  (vernier `Option<PerImageEval> = None`; `lvis-api` `eval_imgs[idx]
  = None` per `eval.py:336`).
- **AA5** — image 1 × cat 2 produces a *populated* cell (no GTs but DTs)
  because `cat 2 ∈ neg[1]`. Distinct from the `None` of AA4.
- **AB1** — every category has a `frequency` tag — required for
  `from_lvis_json_bytes` to load (the AB6 corrected branch).

## Why this fixture is enough for PR-3

The vertical slice is small on purpose: PR-3 ships **bbox + AP only**,
diffing the per-cell payload and raw `(T, R, K, A)` precision tensor.
PR-4 lands the 13-entry summary plan and extends the harness to diff
those entries. Larger fixtures (frequency-bucket coverage, segm/
boundary, AA7 conflict — already a load-time test in PR-2) belong to
later PRs in the rollout.
