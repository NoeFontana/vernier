# ADR-0032: Distributed evaluation across paradigms

- **Status:** accepted (amended by [ADR-0035](0035-api-surface-consolidation.md))
  — the wire envelope, paradigm enum, per-paradigm strict-mode determinism
  rows, and the shared ``Partial*`` exception family are all unchanged.
  ``to_partial`` / ``from_partials`` move from ``Streaming{,Panoptic,Semantic}Evaluator``
  to ``Evaluator`` on each paradigm namespace; the streaming pyclasses are
  removed from Python entirely (their Rust counterparts remain as the
  implementation substrate behind the new pyfunctions).
- **Date:** 2026-05-05
- **Deciders:** @NoeFontana
- **Consulted:** —
- **Informed:** all contributors

## Context and problem statement

ADR-0031 shipped `from_partials` / `to_partial` / `with_rank` for
instance detection (bbox, segm, boundary, keypoints) and explicitly
carved out semantic and panoptic in §"What this ADR explicitly does
not decide": *"`vernier.semantic` and `vernier.panoptic` have their
own evaluator classes (per ADR-0028, ADR-0025) with their own state
shapes. … Separate ADRs when the demand surfaces; this ADR is
instance-detection only."*

Demand surfaced immediately. The same DDP / multi-node training rigs
that motivated ADR-0031 also produce panoptic and semantic metrics:
Cityscapes mIoU, COCO panoptic PQ, ADE20K mIoU, Mapillary panoptic.
A user with three paradigms in their training loop should not have
three different transport stories, three different validation
surfaces, three different exception namespaces, and one of them
(panoptic) missing entirely. ADR-0031's instance-only stop was the
right ship-it cut for that ADR; this ADR lifts the carve-out.

The state shapes are different — instance has a sparse cells store
plus `meta_cells` plus `retained_ious`; semantic has a dense u64
confusion matrix; panoptic has a per-category PqStat fold. But the
*envelope* around those state shapes is identical: magic + version,
parity_mode, dataset_hash, params_hash, a paradigm-specific shape
fingerprint, an optional `rank_id`, an opaque body archive, a CRC32
footer. The cross-rank policy is identical too: image-id partitions
must be disjoint; strict-mode rank ids must be distinct; framing
errors fail cheapest-first before any body decode. One envelope,
one validation pipeline, one error vocabulary, three bodies.

This ADR triggers ADR-0001 §"Affect the public API" (three new
paradigms gain four methods each plus a constructor parameter),
§"Cross the FFI boundary" (three new pyclasses on the FFI surface),
and §"Set a project-wide convention" (the partial wire format is
now a forward-compat commitment for all three paradigms, not just
instance). It does not add a top-level dependency: `rkyv 0.8`,
`blake3 1`, and `crc32fast 1` already shipped via ADR-0031 and are
re-used through a new leaf crate.

## Decision drivers

- **Robustness — one merge policy, three paradigms.** Image-id
  disjointness, rank-id distinctness, dataset-hash equality, and
  params-hash equality are paradigm-agnostic invariants. Any design
  that re-derives them per-paradigm risks divergent semantics — one
  paradigm tolerating overlap that another refuses, or one paradigm
  raising a different exception class than another for the same
  failure. We force a single source of truth: a `BaseMergeAccumulator`
  in a leaf crate that both extends and constrains the paradigm
  body.
- **Robustness — paradigm-mismatch is a structural rejection, not
  a body-decode failure.** A semantic partial loaded by an instance
  evaluator must be refused before any rkyv archive validation runs
  on the body. The header carries `paradigm_kind: u8` exactly so the
  receiver can integer-compare and reject in microseconds. This is
  cheap, and it makes the diagnostic ("you handed a panoptic partial
  to a semantic merger") legible without a stack trace.
- **Robustness — the five exception types must be *the same Python
  class object* across paradigms.** A user catching
  `PartialDatasetMismatch` in a top-level handler should not need to
  know which paradigm's evaluator raised it. The FFI defines each
  class once in `vernier._core` and every paradigm's `__init__.py`
  re-exports it, so `vernier.instance.PartialDatasetMismatch is
  vernier.semantic.PartialDatasetMismatch` holds at runtime.
- **Efficiency — share the wire format.** Two formats means two
  rkyv schemas to evolve, two CRCs to test, two version bytes,
  two hash domains. Three formats means three. One format with a
  closed-world body enum (`Instance | Semantic | Panoptic`) means
  adding a paradigm bumps the format version once and updates one
  enum, instead of requiring a fresh ADR for the format substrate.
- **Efficiency — strict-mode bit-equality where it's free.**
  Semantic confusion-matrix sums are integer (u64) and therefore
  associative. Strict-mode merge for semantic is unconditionally
  bit-equal to a batch run over the union — no `pytest.skip`, no
  tiebreak, no caveat. We pin this property explicitly because it
  is *the* contrast against ADR-0031's instance strict-mode
  conditional skip, and it is the headline determinism story for
  semantic users.
- **Maintainability — ADR-0005 invariant survives.** `matching.rs`,
  `accumulate.rs` (instance), the `pq_image_with_id` /
  `attribute_image` kernels (panoptic), and the confusion-matrix
  fold (semantic) are untouched. Merge is orchestration above the
  spine in every paradigm.
- **Maintainability — no DAG regression.** ADR-0025 keeps
  `vernier-panoptic` independent of `vernier-core`. A naive design
  would route the envelope through `vernier-core` and force panoptic
  to take a dependency it explicitly avoided. We extract the
  envelope into a new leaf crate `vernier-partial` that all three
  paradigm crates depend on but that depends on nothing in the
  workspace. The DAG flattens, not deepens.

## Considered options

The decision space splits across four orthogonal axes:

1. **Axis A — Format substrate.** One shared envelope with a closed
   body enum vs three independent rkyv schemas vs a cross-paradigm
   "extension table" the way Arrow IPC handles dictionaries.
2. **Axis B — Strict-mode bit-equality story.** Per-paradigm
   determination vs uniform "always bit-equal" vs uniform "never
   bit-equal".
3. **Axis C — Where the shared code lives.** New leaf crate
   `vernier-partial` vs in-tree generic module under `vernier-core`
   vs duplicated per paradigm.
4. **Axis D — Exception types.** Five paradigm-shared classes vs
   per-paradigm subclasses vs a single tagged class.

### Axis A — Format substrate

**A1 (chosen) — one envelope, closed-world body enum.**

The wire layout is:

```
 0    4    5    6   10   11    [...]
 +----+----+----+---+----+----+...+----+----+
 |MAG | V  | PK | DISC | parity_mode | header rkyv | body rkyv | CRC32 |
 +----+----+----+------+-------------+-------------+-----------+-------+
```

`MAG = "VRPS"`, `V = FORMAT_VERSION = 2`. `PK` is `paradigm_kind`
(0 = Instance, 1 = Semantic, 2 = Panoptic). `DISC` is a paradigm-
specific u32 discriminator (instance: `KernelKind` discriminant;
semantic / panoptic: 0). The header is one rkyv archive with the
two hashes, the four-slot `shape_fingerprint`, and `rank_id`. The
body is a separate rkyv archive — `WireInstanceBody`,
`WireSemanticBody`, or `WirePanopticBody` — selected by `PK`. The
CRC32 covers everything before the footer.

The closed body enum (rather than a `Vec<u8>` length-prefixed slot)
matters for two reasons. First, rkyv's `bytecheck` validation runs
at decode time against the typed schema, not against opaque bytes —
a malformed body is rejected with a typed `RkyvDecode` error
naming the offset. Second, double-archiving (a `Vec<u8>` rkyv
archive whose contents are also a rkyv archive) is wasted work; a
single archive per body is the right cost.

**A2 — three independent rkyv schemas.**

Each paradigm owns its own magic, version, format. Avoids the
closed enum but at the cost of three test matrices, three CRC
implementations, three hash domain commitments. The
`paradigm_mismatch` rejection becomes "didn't recognize the
magic" instead of a typed integer compare, losing the diagnostic.

**A3 — Arrow IPC extension table.**

Arrow IPC handles paradigm-style polymorphism via dictionary
extension types. Mature, well-specified, and strictly heavier
than rkyv for the same payload (Arrow's header overhead is in the
hundreds of bytes vs rkyv's tens). Also pulls in a much larger
dep graph. Would be the right call if vernier had a separate
"share data with non-vernier consumers" goal; it does not.

### Axis B — Strict-mode bit-equality

**B1 (chosen) — per-paradigm determination, documented per row.**

The bit-equality cell of the determinism contract table is paradigm-
specific:

| Paradigm | Strict mode bit-equal-to-batch? |
|---|---|
| Instance | conditional (skipped pending the `(score, rank_id, local_position)` tiebreak — ADR-0013 follow-up) |
| Semantic | **yes — unconditionally**. Confusion-matrix sums are u64-additive. |
| Panoptic | **yes — opt-in via `retain_per_image_deltas=True`**. The merge accumulator re-sorts per-image deltas by `image_id` and re-sums in batch order. |

The asymmetry is honest: semantic gets bit-equality for free
because integer sums commute; panoptic pays a small memory cost
(per-image PqStat deltas, ~few hundred bytes per image) for the
same property, opt-in via a constructor flag; instance is still
held back by the same ADR-0013 tiebreak as ADR-0031.

**B2 — uniform "always bit-equal".**

Would require lifting the per-image delta machinery into instance
too. Different cell-store shape (sparse, not dense), different
sort key (per-detection score, not per-image id), different memory
profile. Coupling all three paradigms to a single bit-equality
strategy across that variation is the wrong abstraction.

**B3 — uniform "never bit-equal, always 4-ULP envelope".**

Discards the headline semantic property (which is real and free).
Discards the panoptic opt-in path (which users explicitly want for
deterministic CI gates). Pessimizes the surface for no maintenance
benefit.

### Axis C — Where the shared code lives

**C1 (chosen) — new leaf crate `vernier-partial`.**

Owns: the envelope (magic, version, header layout, validation
pipeline), the five `Partial*` error variants, the
`BaseMergeAccumulator` (image_id partition + rank-collision policy),
and the `Partial` / `PartialExpectation` traits. Depends on
`rkyv 0.8`, `blake3 1`, `crc32fast 1`, `thiserror`. Depended on
by `vernier-core`, `vernier-semantic`, `vernier-panoptic`. ~1k
lines.

**C2 — in-tree generic module under `vernier-core`.**

Forces `vernier-panoptic` (which ADR-0025 deliberately kept
independent of `vernier-core`) to take a `vernier-core` dep. The
DAG regression is explicit; the alternative leaf crate is
inexpensive to extract.

**C3 — duplicate per paradigm.**

Three copies of magic / version / CRC / partition logic. Three
test matrices for the same code. Drift is inevitable. Refused.

### Axis D — Exception types

**D1 (chosen) — five paradigm-shared classes.**

`PartialFormatMismatch`, `PartialDatasetMismatch`,
`PartialParamsMismatch`, `PartialPartitionOverlap`,
`PartialRankCollision` are defined once in `vernier._core` and
re-exported from `vernier.instance`, `vernier.semantic`,
`vernier.panoptic`. The Python identity test asserts:

```python
import vernier.instance, vernier.semantic, vernier.panoptic
assert (
    vernier.instance.PartialDatasetMismatch
    is vernier.semantic.PartialDatasetMismatch
    is vernier.panoptic.PartialDatasetMismatch
)
```

A user's top-level handler catches one class and gets all three
paradigms.

**D2 — per-paradigm subclasses (`InstancePartialDatasetMismatch`,
…).**

Triples the surface for no semantic gain. A multi-paradigm handler
would catch a tuple of three classes; a logger would print three
different qualnames for the same condition.

**D3 — single tagged class.**

One `VernierPartialError` with a `kind: str` field. Loses static
catchability; ergonomically worse than the five-class variant.

## Decision outcome

Chosen: **A1 (one envelope, closed body enum) + B1 (per-paradigm
strict-mode bit-equality, documented per row) + C1 (new
`vernier-partial` leaf crate) + D1 (paradigm-shared exception
types).**

The choice is forced by the decision drivers: A1 is the only option
that avoids drift across three format substrates; B1 is the only
option that doesn't lie about the headline semantic determinism
property; C1 is the only option that doesn't regress the DAG that
ADR-0025 deliberately built; D1 is the only option that gives the
user a single class to catch.

### Consequences

- **Positive.** The DDP / multi-host training story closes for *all
  three* paradigms with one transport idiom. Semantic ships
  unconditional strict-mode bit-equality (no `pytest.skip`) — first
  paradigm in vernier where the strongest cell of the determinism
  table is filled. Panoptic ships strict-mode bit-equality opt-in
  via `retain_per_image_deltas=True`; the corrected default stays
  within ADR-0004's 4-ULP envelope. The five exception classes
  unify catch sites across paradigms. Adding a fourth paradigm
  later (e.g., 3D detection) requires bumping `FORMAT_VERSION`
  once and adding one `WireEnvelopeBody::Foo(WireFooBody)`
  variant — no fresh ADR for the substrate.
- **Negative.** The `FORMAT_VERSION 1 → 2` bump invalidates any
  v1 instance partial captured between ADR-0031 ship and this
  ADR. The window is <24 hours; per `project_release_pace`
  the project is on 0.0.x and the cost is acceptable; users with
  v1 partials in flight pin their vernier version. The closed-world
  body enum means a fourth paradigm cannot ship as a pure additive
  release — it requires a coordinated `vernier-partial` change.
  This is the right cost: a leaf crate change is cheap; format
  drift is not.
- **Neutral.** `vernier-panoptic`, which had no streaming surface
  pre-PR-E, gains streaming + distributed merge as a single
  release. The new `StreamingPanopticEvaluator` and its
  `from_partials` constructor are net-additive; no existing API
  reshapes. The `retain_per_image_deltas` constructor flag is
  paradigm-only (not surfaced on instance or semantic) and
  defaults to `False` to keep single-rank ergonomics lean.

## What this ADR explicitly does *not* decide

- **Cross-paradigm merge.** Loading an instance partial into a
  semantic evaluator is rejected with `PartialFormatMismatch{kind:
  paradigm_mismatch}`. Cross-paradigm fusion (e.g., panoptic
  derived from a paired instance + semantic run) is a different
  shape — the math for "merge an instance Summary with a semantic
  Summary" is undefined; refusing structurally is the right cut.
- **Async / streaming gather.** `from_partials` is a single-shot
  N-way constructor (same as ADR-0031). Streaming partial
  consumption (e.g., partials trickling in over a long-running
  socket) is a different threading model; out of scope.
- **Distributed TIDE / `error_decomposition`.** Per ADR-0021,
  TIDE is its own structural axis. The bin-attribution histograms
  aggregate differently from the AP fold; a distributed TIDE
  merge needs its own design.
- **Distributed boundary-PQ.** ADR-0025 §"explicitly does not
  decide" Q3 / Z1 deferred boundary-PQ to a follow-up; that
  follow-up will need to extend `WirePanopticBody` (or add a
  paradigm variant) but the ADR-0032 envelope already supports
  the discriminator slot.
- **Partial compression.** rkyv archived payloads are dense; at
  ImageNet val scale a semantic partial is ~tens of KB; at COCO
  panoptic val with `retain_per_image_deltas=True` it climbs to
  ~few MB. Users who need wire-time compression apply zstd at
  the transport layer (`gather_object` over zstd-pickle is
  trivial).
- **Cross-version partial compatibility.** `FORMAT_VERSION 1`
  partials are refused by v2 with `PartialFormatMismatch{kind:
  wrong_version}`; the same rule applies to any future bump.
  Pre-1.0 stability promise unchanged.
- **Endianness.** rkyv archives are little-endian; both x86_64
  and aarch64 are LE. Big-endian deployment is hypothetical;
  same status as ADR-0031.

## Pros and cons of the options

### Axis A — Format substrate

**A1 (chosen) — one envelope, closed body enum.**

- 👍 One CRC, one version byte, one validation pipeline, one
  test matrix.
- 👍 Paradigm-mismatch is a typed integer compare on the header —
  microsecond-level rejection without body archive validation.
- 👍 rkyv `bytecheck` validates the typed body schema at decode
  time, surfacing structural errors as `RkyvDecode` with the
  offending offset.
- 👎 Adding a fourth paradigm requires a `FORMAT_VERSION` bump
  and a `vernier-partial` change. We pay this cost willingly:
  format drift is the real risk, not coordination overhead.

**A2 — three independent rkyv schemas.**

- 👍 Per-paradigm format evolution proceeds independently.
- 👎 Three magic-byte conventions, three CRCs, three header
  layouts. The cross-paradigm test matrix grows quadratically
  with paradigm count.
- 👎 `paradigm_mismatch` becomes "magic mismatch" — the diagnostic
  is structurally weaker.

**A3 — Arrow IPC extension table.**

- 👍 Industry-standard polymorphism story.
- 👎 ~10× the header overhead of rkyv at the per-partial level.
- 👎 Pulls in a much larger dep graph (`arrow-rs` is multi-crate
  with its own version churn).

### Axis B — Strict-mode bit-equality

**B1 (chosen) — per-paradigm.**

- 👍 Honest documentation: each row of the contract table reflects
  what the paradigm actually delivers.
- 👍 Semantic ships the headline "no `pytest.skip`" property.
- 👍 Panoptic users who want the property pay for it explicitly
  via `retain_per_image_deltas=True`.
- 👎 The contract table is no longer uniform across paradigms.
  A user reading the table needs to know which paradigm they're
  on. We mitigate by colocating the table with each paradigm's
  ADR.

**B2 — uniform "always bit-equal".**

- 👍 One mental model.
- 👎 Forces instance and panoptic to converge on a single
  bit-equality strategy across very different cell-store shapes.
  Wrong abstraction.
- 👎 Pessimizes panoptic memory at single-rank scale because the
  per-image deltas would no longer be opt-in.

**B3 — uniform "never bit-equal".**

- 👍 One mental model.
- 👎 Discards the free semantic property.
- 👎 Discards the panoptic opt-in story users explicitly want.

### Axis C — Where the shared code lives

**C1 (chosen) — `vernier-partial` leaf crate.**

- 👍 No DAG regression. `vernier-panoptic` does not take
  `vernier-core` as a dep.
- 👍 The `BaseMergeAccumulator` partition + rank-collision policy
  is a single source of truth.
- 👍 The five `Partial*` error variants live in one place,
  re-exported by every paradigm crate.
- 👎 One more crate to publish. We accept this cost: extraction
  is a one-time tax, drift is a recurring one.

**C2 — in-tree under `vernier-core`.**

- 👍 No new crate boundary.
- 👎 Forces `vernier-panoptic` to depend on `vernier-core`,
  reversing the explicit ADR-0025 isolation.
- 👎 Conflates instance-paradigm code with paradigm-agnostic
  envelope code in one crate.

**C3 — duplicate per paradigm.**

- 👍 Each paradigm crate is self-contained.
- 👎 Three copies of the validation pipeline. Drift over time
  is certain.
- 👎 The five exception types would need to be defined three
  times, breaking the `is`-identity guarantee.

### Axis D — Exception types

**D1 (chosen) — five paradigm-shared classes.**

- 👍 One catch site handles all paradigms.
- 👍 Identity test (`is`-equality) provides a strong contract.
- 👍 The `kind` attribute on `PartialFormatMismatch` carries the
  sub-discriminator (`paradigm_mismatch`, `kernel_mismatch`,
  `grid_mismatch`, etc.) for fine-grained branching.
- 👎 Three paradigm namespaces re-export the same five classes —
  the docstring repetition is a minor surface tax.

**D2 — per-paradigm subclasses.**

- 👍 Each paradigm's exceptions are distinct identifiers.
- 👎 Triples the surface area for no semantic gain.
- 👎 A user catching `PartialDatasetMismatch` in a top-level
  handler must catch a tuple of three classes.

**D3 — single tagged class.**

- 👍 One identifier.
- 👎 Loses static catchability — `try: … except PartialDatasetMismatch:`
  becomes `try: … except VernierPartialError as e:
  if e.kind == "dataset": …`.
- 👎 Surface is ergonomically worse than D1 with no benefit.

## Links and references

- Related ADRs:
  - ADR-0013 (`StreamingEvaluator`) — the predecessor whose
    `merge` deferral this ADR closes for all paradigms.
  - ADR-0014 (`BackgroundEvaluator`) — same `merge` deferral;
    background evaluators inherit the partial surface for free
    once their underlying streaming evaluator gains it.
  - ADR-0025 (panoptic API) — the panoptic batch-only constraint
    is lifted by this ADR.
  - ADR-0028 (semantic API) — the semantic streaming surface
    gains `from_partials`.
  - ADR-0029 (namespace) — the per-paradigm submodule structure
    that hosts the re-exported exception classes.
  - ADR-0031 (distributed eval, instance) — the predecessor whose
    §"explicitly does not decide" carve-out for semantic / panoptic
    this ADR supersedes.
- Implementation PRs: #157 (PR-C extract `vernier-partial`),
  #158 (PR-D semantic merge), #159 (PR-E panoptic streaming +
  merge).
