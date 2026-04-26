# `score_ties`

Pins quirk **A1** (cocoeval.py:259, `kind='mergesort'`): when two
detections share a score, pycocotools' `np.argsort([-d['score'] for d in
dt], kind='mergesort')` is *stable*, so the tied detections retain their
input order rather than being shuffled by an unstable sort.

Two GTs and two DTs, each DT perfectly matched to one GT, all four with
score 0.7. With stable sort:

- DT 0 (input position 0) is processed first → claims GT 0.
- DT 1 (input position 1) is processed second → claims GT 1.

If the sort were unstable, ties could swap order and the matching
assignment would flip — observable in `dtMatches` (which GT id ends up
in row 0 vs row 1) even though `stats[0]` (mAP) is invariant.

The fixture is included in `ALL_FIXTURES` so the parity harness diffs
the per-image `dtMatches`/`gtMatches` arrays, catching any regression in
the sort kind or its tiebreaker.
