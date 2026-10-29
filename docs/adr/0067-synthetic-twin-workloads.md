# ADR-0067: Gate synthetic twins of real bench workloads on measured equivalence

- **Status:** proposed
- **Date:** 2026-10-29
- **Deciders:** @NoeFontana
- **Consulted:** —
- **Informed:** —

## Context and problem statement

The bench harness ([ADR-0017](0017-local-bench-harness.md),
[ADR-0033](0033-multi-paradigm-bench.md)) measures every implementation on
real datasets: COCO val2017, LVIS v1, Objects365 v2, COCO panoptic, ADE20K.
Their predictions are either jittered ground truth or real-model output.
Real data is what makes the published numbers credible. It is also what
keeps the bench out of the development loop:

- each machine has to download the data (≈ 2 GiB of ground truth), under
  licences that rule out committing or redistributing it;
- the cells are slow: Objects365 is 80k images and 1.24M boxes, and
  pycocotools takes about 6 min per rep on it;
- an agent working in a worktree cannot run any of it before it has the data.

`synthetic:` workloads exist, but they draw boxes uniformly and detections
independently of ground truth. Their cost profile is unlike any real
dataset, so a speedup measured on them predicts nothing about real data
(the perf-discipline notes record exactly that disagreement between divan
and val2017 + jitter). A synthetic stand-in is only useful if it costs the
same as the real cell, for every implementation.

## Decision drivers

- A stand-in must reproduce the real cell's cost: per-stage time and peak
  memory, for every implementation, not just vernier.
- "Similar-looking data" is not evidence. Equivalence has to be measured,
  and re-measured whenever the code changes.
- No dataset bytes in the repository: whatever is committed must be
  aggregate statistics.
- Real cells stay the ground truth for published numbers.

## Considered options

1. **Keep real data only.** Make it cheaper to fetch (shared caches, an
   HF mirror of generated predictions), but keep every timing cell on real
   data.
2. **Parametric synthetic workloads tuned by hand.** Extend `synthetic:`
   with knobs and choose values that look like COCO.
3. **Profiled twins behind a measured gate.** Extract a statistics profile
   from the real workload, generate a twin from it deterministically, and
   accept the twin only when a gate shows equal structure *and* equal
   measured cost.

## Decision outcome

Chosen option: **Option 3**, because it is the only one that turns
"equivalent" into a check that fails loudly when it stops being true.

**Profile.** `bench twin profile <workload> --name N` writes
`bench/twin_profiles/N.json`. It records:

- the joint per-(image, category) cell size distribution `(g, d)`, since
  matching performs `g·d` IoU evaluations per cell;
- category rank-frequency, as integer ids only;
- box log-area, aspect and fill: area drives the COCO area-range split;
- crowd share;
- each detection's best same-category IoU, and scores split by match;
- mask complexity, as RLE characters per bbox column, conditioned on the
  polygon's vertex count;
- the JSON shape: key order, separators, digits per number, id ranges,
  the per-record segmentation kinds, and the sizes of the extra fields.

Real text never enters the profile, only lengths and integer ranges.

**Twin.** `twin:N[,seed=S]` generates the GT/DT pair, cached per
`(profile, seed, TWIN_VERSION)`. Discrete distributions use stratified
quotas, so counts that set the cost reproduce exactly rather than within
sampling noise: images, cells, GT, DT, IoU pairs and crowd. Mask outlines
are star polygons whose roughness is searched to hit a sampled RLE length.

**Gate.** `bench twin compare <real> twin:N --iou T` passes only if all of
the following hold:

1. **Structure.** Every total (counts, IoU pairs, JSON bytes, and RLE
   characters for mask cells) is within 3%, and every cost-relevant
   histogram is within total-variation distance 0.05.
2. **Cost.** For every implementation and every stage, the median wall
   and CPU time are within 10% (or 20 ms), and the peak RSS is within 10%
   (or 16 MiB). This uses `bench run --mode release` results for both
   workloads from the same commit and machine, under the ADR-0049 CPU
   budget. Each implementation runs on the real workload and the twin back
   to back: on a shared host, speed drifts by about the tolerance between
   runs minutes apart. A cell that failed its IQR gate fails the check as
   unstable, rather than counting as evidence either way.
3. **Ranking.** Each implementation's CPU-time ratio to vernier is within
   10% of the real ratio, so no published comparison would change.

Each report is committed under `docs/engineering/benchmarking/twins/`.

**Tiers.** A real cell whose twin passes moves to a *release* tier: it no
longer runs in the development loop, and it runs before every release
together with the twin gate. A real cell is never deleted. A gate failure
at release blocks the release until the twin is regenerated (bump
`TWIN_VERSION` or re-profile) or the real cell is promoted back.

**Scope.** This ADR covers instance workloads (bbox, and mask cells once
they pass). LVIS federated metadata, keypoints, panoptic and semantic need
their own profile dimensions and stay on real data until they have them.
Real-model prediction cells are not candidates: their distribution is the
whole point of those cells.

### Consequences

- **Positive:** agents and CI can run instance timing cells without
  downloads, and a twin can be scaled or re-seeded. A drifting twin is
  caught by the release gate instead of quietly misleading.
- **Negative:** a release now runs the real cells plus the gate, which
  makes it longer. The profile captures only the dimensions we know drive
  cost, so an optimisation that keys on something else can make a twin
  and its real cell disagree. The release gate catches that, but only at
  release time.
- **Neutral:** the twin's AP values are meaningless and are not a parity
  target. Parity stays on real data and fixtures.

## Pros and cons of the options

### Option 1

- 👍 Nothing new to trust.
- 👎 The loop stays as slow as the data. Licences and downloads stay in
  every setup.

### Option 2

- 👍 Simple.
- 👎 Hand-tuned knobs drift, and nothing tells you when. That is the
  current `synthetic:` situation.

### Option 3 (chosen)

- 👍 Equivalence is measured and re-measured. Twins are reproducible and
  cheap.
- 👎 More machinery, and a gate that has to be kept honest.

## Links and references

- Related ADRs: [ADR-0017](0017-local-bench-harness.md),
  [ADR-0033](0033-multi-paradigm-bench.md),
  [ADR-0049](0049-bench-cpu-budget.md)
- Implementation: `bench/bench/workloads/twin.py`, `bench twin` in
  `bench/bench/cli.py`
