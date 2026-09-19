# ADR-0061: One ground-truth parameter, and keyword-only options, on the instance evaluate surface

- **Status:** accepted
- **Date:** 2026-09-18
- **Accepted:** 2026-09-19 (shipped in 0.5.0)
- **Deciders:** @NoeFontana
- **Supersedes:** none
- **Amends:** ADR-0020 (parsed-once dataset handle) — the handle stays; the
  way it is reached changes.

## Context and problem statement

The instance evaluate surface is a hand-maintained combinatorial matrix:
four kernels (`bbox`, `segm`, `boundary`, `keypoints`) × two output
shapes (`grid`, `summary`) × two ground-truth input forms (JSON bytes,
parsed `CocoDataset` handle) = **sixteen functions**. Fourteen exist.

| kernel | `grid` | `grid_with_dataset` | `summary` | `summary_with_dataset` |
| --- | :--: | :--: | :--: | :--: |
| bbox | ✓ | ✓ | ✓ | ✓ |
| segm | ✓ | ✓ | ✓ | ✓ |
| boundary | ✓ | **✗** | ✓ | ✓ |
| keypoints | ✓ | **✗** | ✓ | ✓ |

The gaps are not a backlog; they are the *shape* of the problem. Each
cell is written by hand, so a missing one is invisible until a caller
needs it — and then it is a release-blocking gap in a shipped API:

- **0.4.0** shipped `evaluate_bbox_grid_with_dataset` present in `_core`
  but absent from the `vernier.instance` re-export list. ADR-0020's
  handle served summary evaluation and not grid evaluation. Found by a
  downstream integration, fixed in 0.4.1, and pinned by an invariant
  test over the whole `_with_dataset` family — which is a guard on the
  symptom, not the cause.
- **0.4.1** had no `evaluate_segm_grid_with_dataset` at all, so a caller
  evaluating both instance kernels parsed the same ground truth twice.
  Added in 0.4.2. Boundary's and keypoints' cells are still empty, and
  will be until someone trips on them.

Three further defects sit on the same surface:

1. **The ground-truth parameter has three names.** `gt_json` on the
   grid/summary functions, `gt_bytes` on the `_partitioned_lrp` ones,
   `gt` on the `_with_dataset` ones. The same argument, the same
   position, three spellings — so no keyword works across the pair a
   caller is choosing between, and the two forms cannot be swapped
   without editing the call.
2. **Options are positional, and three adjacent ones are booleans.**
   `(gt, dt, parity_mode, max_dets_per_image, use_cats, retain_iou=…,
   cast_inputs=…, …)`. `use_cats`, `retain_iou` and `cast_inputs` are
   all `bool`; a caller who miscounts sets a different flag and gets a
   silently different evaluation rather than a `TypeError`.
3. **Kernel-specific required arguments sit mid-signature.**
   `dilation_ratio` (boundary) and `sigmas` (keypoints) are wedged
   between `use_cats` and the optional tail, so the four kernels'
   signatures diverge in the middle rather than at the edge.

## Decision drivers

- A missing cell in a hand-maintained matrix must stop being
  expressible, rather than being caught by a test that enumerates the
  family after the fact.
- The parsed-once handle (ADR-0020) is the performance-relevant path for
  any caller evaluating more than one thing; reaching it should not
  require calling a differently-named function.
- Pre-1.0 permits breaking changes on a minor bump, and this surface is
  low-level: the documented public API is `Evaluator`, which already
  accepts either ground-truth form.

## Decision outcome

**One ground-truth parameter named `gt`, accepting `bytes |
CocoDataset`, and every option after `dt` keyword-only.**

```python
evaluate_bbox_grid(gt, dt, *, parity_mode, max_dets_per_image, use_cats, ...)
evaluate_segm_grid(gt, dt, *, parity_mode, max_dets_per_image, use_cats, ...)
evaluate_boundary_grid(gt, dt, *, parity_mode, ..., dilation_ratio, ...)
evaluate_keypoints_grid(gt, dt, *, parity_mode, ..., sigmas, ...)
#   ... and the four `_summary` siblings.
```

The six `_with_dataset` functions are **deleted**. Their behaviour is
reached by passing a `CocoDataset` to the base function, which
dispatches on the argument's type — `bytes` to the parsing path,
`CocoDataset` to the snapshot path. The dispatch is total and
unambiguous: the two types share no values.

### Why this is the fix and not a bigger signature

The matrix collapses from sixteen cells to eight functions, and the two
empty cells fill themselves: `evaluate_boundary_grid` and
`evaluate_keypoints_grid` accept a handle the moment `gt` widens,
without either function being written. **A gap of this class stops being
expressible** — there is no longer a per-(kernel, form) function that
someone could forget to add.

This is not a new idea in the codebase; it is the *existing* high-level
contract pushed down. `Evaluator(iou=Bbox()).evaluate(gt, dt)` has
always accepted bytes or a handle, and `test_dataset.py` has always
asserted the two are bit-equal on every kernel. The wrapper solved this
problem; the layer beneath it did not.

Going further — one `evaluate_grid(gt, dt, *, iou=Bbox())` — was
rejected. That function already exists: it is `Evaluator`. The
per-kernel entry points are the primitives the wrapper is built from,
and `dilation_ratio` / `sigmas` are genuinely per-kernel, so collapsing
the kernel axis would replace four honest signatures with one that
carries three mutually-exclusive optional arguments.

### Keyword-only, and what it is worth

Everything after `dt` becomes keyword-only. The argument is not style:
`use_cats`, `retain_iou` and `cast_inputs` are three adjacent `bool`s,
two of them defaulted, so a caller who passes one positional argument
too few or too many silently changes *which* flag is set. The failure is
a different evaluation, not an error — the worst shape a mistake can
take on a parity-critical surface.

Making the kernel-specific arguments keyword-only is what lets the four
signatures agree up to their own options: `dilation_ratio` and `sigmas`
stop being positional slots that shift everything behind them.

### `gt`, not `gt_json`, on the functions that still take only bytes

`evaluate_*_partitioned` and `evaluate_*_partitioned_lrp` keep taking
bytes for now — widening them is separate work with its own
orchestration questions. Their parameter is **renamed to `gt`
regardless**, so that widening is later a *widening* and not a rename:
the keyword a caller writes today is the keyword that will accept a
handle tomorrow.

## Consequences

Breaking, on a pre-1.0 minor bump:

- Six `_with_dataset` functions are removed. Callers pass the handle to
  the base function instead.
- `gt_json=` / `gt_bytes=` as keywords become `gt=`. (Zero occurrences
  in this repository; the parameter has only ever been passed
  positionally.)
- Every option after `dt` must be named. 35 internal call sites pass
  `parity_mode` positionally and are updated with this ADR.

The invariant test added in 0.4.1 —
`test_instance_reexports_every_dataset_taking_entry_point` — is
retired with the family it guarded. It was a guard against forgetting a
cell; there are no cells left to forget. What replaces it is a test that
every kernel's base function accepts both ground-truth forms and returns
bit-equal results, which is the property the family existed to provide.

`Evaluator` is unaffected: it already accepted both forms, and its own
signature does not change.

## Scope, as implemented

Two extensions beyond what this ADR first proposed, both recorded here
rather than left to drift:

1. **The `gt` rename reached further than the instance surface.**
   `confusion.rs`, `tide.rs` and `lrp.rs` also took `gt_bytes` — 13
   functions. Renaming only the evaluate family would have left the FFI
   with two names for the ground-truth argument, which is the
   inconsistency this ADR exists to remove. With zero keyword callers in
   the repository the rename was free, so it was applied uniformly:
   **23 functions across five modules now take `gt`.**
2. **The partitioned family became keyword-only too**, not
   rename-only as first written. Leaving `manifest` as a positional slot
   behind three booleans reproduces the exact hazard being fixed on the
   neighbouring function, and the hand-written stub had already been
   marked keyword-only — a runtime that disagreed would fail
   `test_module_function_signatures_match`.

## A note for the next migration of this shape

The call-site migration was done with a source rewriter driven by the
stub's own parameter order. It failed twice, and both failures are worth
knowing about because neither was caught by the obvious check:

- It treated an inline **comment** inside a call's argument list as an
  argument, producing `use_cats=# comment, retain_iou=which is opt-in`.
  One file failed to parse; the same bug could have mis-named arguments
  in a call that still parsed. The fix is to *refuse* calls containing
  comments and report them — two existed, both done by hand.
- It assumed every call's first two arguments were the `gt` / `dt` pair
  and began naming from the third. True for the eight instance
  functions; **false for `evaluate_semantic_partitioned`**, whose
  parameters are `gt_label_maps` / `dt_label_maps` / `n_classes`. It
  emitted `gt_label_maps=n_classes, dt_label_maps=parity_mode,
  n_classes=manifest` — arguments shifted by two, using names that all
  *exist* in the signature.

That second one is the instructive failure. A sweep for keywords absent
from the target signature came back clean, because every name it used
was real. What caught it was the test suite. The lesson is narrow and
worth stating: a name-based rewriter must verify the signature it is
rewriting against, and "it parsed, and the obvious check was clean" is
not evidence of correctness on a mechanical migration.

## Links and references

- ADR-0020 — the parsed-once dataset handle this makes reachable
  uniformly.
- ADR-0040 — the grid axes (`iou_thresholds`, `recall_thresholds`,
  `area_ranges`) that become keyword-only here.
- `docs/engineering/python-type-stubs.md` — `_core.pyi` is hand-written;
  the overloads for `bytes | CocoDataset` are added there by hand.
- ADR-0047 — supersedes it *only* on the spelling of the ground-truth
  argument. Its `evaluate_*_with_dataset` references describe the surface
  as it stood in 0.3.x; the threading decision it records is untouched.
  Its body is `accepted` and therefore immutable (ADR-0001), so the
  pointer lives in its Status line and here, not in its prose.
