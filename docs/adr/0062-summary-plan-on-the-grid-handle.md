# ADR-0062: Carry the summary plan on the grid handle

- **Status:** proposed
- **Date:** 2026-09-19
- **Deciders:** @NoeFontana
- **Consulted:** —
- **Informed:** all contributors
- **Amends:** ADR-0012 (OKS keypoints surface) — the 3-bucket area grid
  is unchanged; what changes is who remembers that it needs its own
  summary plan.

## Context and problem statement

`PyAccumulated.summarize(plan=None)` defaulted to the literal string
`"detection"`. That default is wrong for exactly one kernel, and wrong
in a way that cannot be shrugged off: keypoints evaluates over a
3-bucket area grid (ADR-0012, quirk **D5**) while the detection plan
indexes a fourth bucket. Pairing them does not produce slightly-off
numbers, it raises `AreaRng index 3 is out of range`. The reverse
pairing is quieter and worse — a detection grid summarized with the
keypoints plan reads re-indexed buckets and returns numbers.

The docstring on `summarize` said so: *"match the plan to the kernel
that built the grid."* A docstring instructing the caller to re-supply a
fact is a sign the callee discarded something it had. `evaluate_grid_impl`
holds the `EvalIouType`, calls `is_keypoints()` on it while resolving the
area ranges, and then drops it when constructing `PyEvalGrid` — while
deliberately carrying `parity`, `iou_thresholds` and `recall_thresholds`
onto the same handle, for the stated reason that summarize must match
the grid it came from. The summary plan is the same class of fact and
was the one member of that class left to the caller.

So the mapping "which plan does this kernel use" was re-derived at four
sites that could not see one another:

| site | shape |
| --- | --- |
| `python/vernier/instance/__init__.py` | per-arm in the kernel `match` |
| `python/vernier/_compat.py` | ternary on `params.iouType` |
| `crates/vernier-ffi/src/partition_py.rs` | an `is_keypoints: bool` parameter carried *alongside* the `EvalIouType` it duplicates |
| `crates/vernier-ffi/src/lib.rs` (`summarize_grid`) | an `is_keypoints: bool` parameter, fed from `iou_type.is_keypoints()` by both callers |

One of them was wrong. The keypoints branch of `_evaluate_with_tables`
summarized with the detection plan, so
`Evaluator(iou=Keypoints()).evaluate(gt, dt, calibration=True)` raised
for *every* input — the path was unreachable rather than merely
inaccurate, and stayed that way because keypoints rejects `tables=`,
leaving `calibration=True` as the only route in and nothing covering it.

## Decision drivers

- A fact the callee already holds should not be re-supplied by the
  caller. Four independent derivations of one mapping is four chances to
  disagree, and they did.
- The failure mode is a crash or silently re-indexed buckets on a
  parity-critical surface, not a cosmetic difference.
- LVIS and user-defined summaries legitimately need to name a plan that
  their grid's kernel does not imply, so the argument cannot simply be
  removed.
- Precedent already exists on this exact handle: `parity` and both
  threshold ladders are carried for this reason and documented as such.

## Considered options

1. **Keep re-deriving, add a test per call site.** Leave the default at
   `"detection"` and cover each derivation.
2. **Put the plan on the kernel dataclasses** (`Bbox`/`Segm`/`Boundary`/
   `Keypoints` in `vernier.instance`) and read it at each call site.
3. **Carry the plan on `PyEvalGrid`, propagate to `PyAccumulated`, and
   default `summarize(plan=None)` to it.**

## Decision outcome

Chosen option: **Option 3.** The grid knows which kernel built it; the
accumulator comes from the grid; the summary comes from the
accumulator. Threading the plan along that existing chain makes the
wrong pairing unreachable by default instead of documented against, and
the explicit `plan=` argument stays as an override so nothing is lost.

`SummarizePlan` gains `for_iou_type` — the single mapping — and
`to_core`, since `vernier_core::SummaryPlan` carries a lifetime for its
`Custom` variant and cannot be the `Copy` field these handles store.

### Consequences

- **Positive:** all four re-derivations delete. The two Rust
  `is_keypoints: bool` parameters were redundant with an `EvalIouType`
  their functions already received; both are gone, and with them the
  possibility that the flag and the type disagree.
- **Positive:** the documented footgun becomes structural. `summarize()`
  on a keypoints accumulator returns the 10-stat vector rather than
  raising, without the caller knowing anything about area buckets.
- **Negative:** `PyEvalGrid` and `PyAccumulated` each carry one more
  field. Both are already carrying four such facts, so this is more of
  the same shape rather than a new one.
- **Negative:** the default is now context-dependent — reading
  `summarize()` at a call site no longer tells you which plan runs. That
  is the point, and it is why the override remains and is tested, but it
  does trade local legibility for a guarantee.
- **Neutral:** no behaviour changes for any call that works today.
  Detection grids defaulted to `"detection"` and still do; keypoints
  grids raised and now succeed. There is no input for which a
  previously-correct result changes.

## Pros and cons of the options

### Option 1 — test each derivation

- 👍 Smallest diff; no signature or default changes.
- 👎 Tests the symptom. A fifth call site is still free to get it wrong,
  and the keypoints bug proves that "three of four are correct" is the
  natural resting state, not an accident.
- 👎 Leaves two Rust functions taking a boolean that duplicates a
  parameter beside it.

### Option 2 — plan on the kernel dataclasses

- 👍 Puts the mapping next to the kernel selector, which reads well in
  Python.
- 👎 Only fixes the Python half; the two Rust sites are below that layer
  and keep their booleans.
- 👎 The dataclasses are bare selectors today. Giving them behaviour
  makes them a second place to look for kernel semantics alongside
  `EvalIouType`.

### Option 3 — plan on the grid handle (chosen)

- 👍 One mapping, at the boundary where the kernel is already known.
- 👍 Deletes both redundant Rust parameters and both Python
  derivations.
- 👍 Matches how `parity` and the threshold ladders are already handled
  on the same struct.
- 👎 Two new struct fields, and a default whose value is not visible at
  the call site.

## Links and references

- ADR-0012 — the keypoints 3-bucket area grid that makes the pairing a
  correctness constraint rather than a preference.
- ADR-0040 — `iou_thresholds` / `recall_thresholds` on the same handle,
  the precedent this follows.
- ADR-0061 — the review that surfaced the mis-bound keypoints branch.
- Quirk **D5** — `docs/engineering/pycocotools-quirks.md`.
