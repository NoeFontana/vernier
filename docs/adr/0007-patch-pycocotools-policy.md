# ADR-0007: `patch_pycocotools` — opt-in `sys.modules` monkey-patch as the migration path

- **Status:** accepted
- **Date:** 2026-04-26
- **Deciders:** @NoeFontana
- **Consulted:** —
- **Informed:** all contributors

## Context and problem statement

Phase 1 Week 5 ships two Python entry points: `vernier.Evaluator`
(an extended-API class with builder semantics, parity-mode selection,
and a structured `Summary` return value), and `vernier.COCOeval`
(a faithful drop-in for `pycocotools.cocoeval.COCOeval`, including
the `Params` mutability pattern that downstream code relies on).
The drop-in is the migration tool: it lets a user with mmdetection /
ultralytics / detectron2 eval scripts swap pycocotools for vernier
without touching the eval logic.

The drop-in alone does not finish the migration story. Downstream
codebases import pycocotools directly:

```python
from pycocotools.cocoeval import COCOeval
```

To exercise vernier with that code unchanged, a user needs to
intercept the import. The standard mechanism is monkey-patching
`sys.modules["pycocotools.cocoeval"].COCOeval`. The question this ADR
answers is: *does vernier expose a sanctioned helper for that, and
if so, how?*

The competing tool faster-coco-evals exposes `init_as_pycocotools()`,
which does the same `sys.modules` swap. Two issues with copying that
shape:

1. **Naming.** "init_as_pycocotools" reads as initialization — like
   constructing an evaluator with a pycocotools-shaped API. It does
   not name the mechanism (sys.modules patching). Memory note from
   prior conversations explicitly flags borrowed names from
   competing tools as anti-pattern: the verb should describe what
   actually happens.
2. **Scope.** faster-coco-evals' helper has a single config knob.
   vernier's drop-in has to thread through ADR-0002's parity mode
   (strict/aligned/corrected); the migration story changes shape if
   the user can't pin strict at patch time.

A second, larger concern: monkey-patching `sys.modules` is a
debugging hazard. A user who imports `pycocotools.cocoeval.COCOeval`
expects to receive pycocotools' class. If vernier rewires that
silently — e.g., on `import vernier` — the resulting confusion when
debugging an unexpected score difference is severe and hard to trace.
The patch must be opt-in, explicit, reversible, and visible in
tracebacks.

## Decision drivers

- ADR-0001 §"Affect the public API" — every Python entry point is
  public surface.
- ADR-0001 §"Change the parity contract" — the patch carries a
  parity mode; the choice is part of the contract.
- ADR-0002 — three-tier parity model. The patch must accept a
  parity mode; default is **strict** (matching the migration intent —
  users coming from pycocotools want bit-exact behavior first, then
  opt into corrections).
- Reversibility — tests need to undo the patch; a forgotten patch
  poisons the test suite.
- Discoverability — the patch must be a deliberate call, not an
  import side-effect. Silently rewiring `sys.modules` on `import
  vernier` is hostile and would violate the principle of least
  surprise.
- Naming — the verb names the mechanism (per memory: avoid
  copy-paste names from competing tools).

## Considered options

1. **No monkey-patch helper. Users edit imports manually.** Cleanest
   in the abstract; loses the migration story (every downstream
   codebase has to change source).
2. **`patch_pycocotools()` — explicit function, returns an unpatch
   callable. Idempotent. Has a context-manager sibling.**
3. **`init_as_pycocotools()` — borrowed verb from faster-coco-evals.**
   Same mechanism; wrong name (per drivers).
4. **Auto-patch on `import vernier`.** Worst — invisible to the user.

## Decision outcome

Chosen option: **Option 2.**

### API

```python
import vernier.adapters as adapters

# Patch in place; returns an unpatch callable.
unpatch = adapters.patch_pycocotools(parity_mode="strict")
try:
    # Existing pycocotools-using code now exercises vernier.
    from pycocotools.cocoeval import COCOeval
    e = COCOeval(coco_gt, coco_dt, iouType="bbox")
    e.evaluate(); e.accumulate(); e.summarize()
finally:
    unpatch()  # Restores original pycocotools.cocoeval.COCOeval.
```

A context-manager variant for tests:

```python
with adapters.patched_pycocotools(parity_mode="strict"):
    run_downstream_test_suite()
```

### Behavior

- **Target.** Replaces `sys.modules["pycocotools.cocoeval"].COCOeval`
  with vernier's drop-in class. Other pycocotools symbols
  (`pycocotools.coco.COCO`, `pycocotools.mask`) are *not* patched
  in Phase 1; segm parity is Phase 2 and the mask API is its own
  surface (will be revisited in a Phase 2 ADR if appropriate).
- **Class identity.** The patched class is named
  `vernier._compat.PycocotoolsCOCOeval` so that the swap is visible
  in tracebacks and `repr()`. `COCOeval is pycocotools.cocoeval.COCOeval`
  remains true under the patch (both names point to the same vernier
  class object), but `type(eval_instance).__name__` reveals the
  swap.
- **Parity mode.** `parity_mode` is required positionally or by
  keyword. The default is `"strict"` — matching pycocotools bit-
  exactly, since that is the migration intent. Callers wanting
  opinionated fixes (`"corrected"`) opt in explicitly.
- **Idempotency.** Calling `patch_pycocotools` twice without
  unpatching returns an unpatch handle that, when called, restores
  the *original* pycocotools class — not the first-patch state. The
  helper records the original on first call only.
- **Reentrancy.** The context-manager variant nests correctly: an
  outer `patched_pycocotools` followed by an inner
  `patched_pycocotools` followed by inner exit restores the outer
  patch state, and the outer exit then restores pycocotools. The
  helper maintains a small LIFO of saved states keyed by the patched
  module, not by call site.
- **Thread safety.** Patching mutates module-level state and is not
  itself thread-safe — calling it concurrently from multiple threads
  is undefined. Documented; the expected usage is "patch once at
  test setup or process start, unpatch at teardown". Patching does
  not interfere with the GIL-drop pattern from ADR-0006: the patched
  class still drops the GIL during compute.

### What we deliberately do not do

- **Auto-patch on import.** Out — the patch must be a deliberate
  call. The package itself never mutates `sys.modules` at import
  time.
- **Patch beyond `COCOeval` in Phase 1.** Out — `COCO` (dataset
  loader) and `mask` (RLE codec) are separate surfaces with their
  own quirks. Phase 2 may extend the policy via a follow-up ADR if
  segm parity demands it.
- **Silent fallthrough on missing pycocotools.** If pycocotools is
  not importable, `patch_pycocotools` raises a typed error rather
  than silently registering the vernier class as if pycocotools
  existed. (The vernier drop-in is independently available as
  `vernier.COCOeval` for users who don't have pycocotools installed.)

### Consequences

- **Positive.** Three downstream OSS test suites (mmdetection,
  ultralytics, detectron2) run unchanged with one line of setup.
  The migration story moves from "rewrite imports" to "wrap with
  patched_pycocotools". Default-strict matches user intent at
  migration time.
- **Negative.** Users who introspect `pycocotools.cocoeval.COCOeval`
  (e.g., debugging an unexpected score) need to know about the
  patch. We mitigate by giving the patched class a distinctive
  qualified name (`vernier._compat.PycocotoolsCOCOeval`) and by
  requiring explicit opt-in. We do not mitigate by emitting a
  warning on every patched call — that is too noisy for the common
  case where the user knows exactly what they did.
- **Neutral.** Naming this `patch_pycocotools` rather than
  `init_as_pycocotools` is a deliberate divergence from the
  competing tool. Users coming from faster-coco-evals will notice;
  the divergence is intentional and signals that vernier is not a
  drop-in for faster-coco-evals — it is a drop-in for *pycocotools*.

## Pros and cons of the options

### Option 1 — No helper

- 👍 Cleanest abstraction; nothing to maintain.
- 👎 Every downstream codebase must edit imports. Migration tax is
  high enough that users may simply not migrate.

### Option 2 (chosen) — `patch_pycocotools`

- 👍 Explicit verb names the mechanism. Reversible (returns unpatch
  handle). Context-manager variant is test-friendly. Carries
  parity mode.
- 👎 Adds a small amount of monkey-patch machinery to maintain.
  Acceptable cost for the migration story.

### Option 3 — `init_as_pycocotools`

- 👍 Familiar to faster-coco-evals users.
- 👎 Verb does not name the mechanism. Borrowed shape from a
  competing tool — flagged in memory as anti-pattern.

### Option 4 — Auto-patch on import

- 👍 Zero-config from the user's perspective.
- 👎 Hostile. Breaks debugger expectations. Silent score
  differences cannot be traced to vernier without reading the
  source. Violates principle of least surprise.

## Links and references

- ADR-0001 — Record architecture decisions (§"Affect the public
  API", §"Change the parity contract").
- ADR-0002 — Three-tier parity model.
- ADR-0006 — Threading model (GIL-drop, single-threaded compute).
- `tests/python/parity/harness.py:97` — `_run_vernier` swap point;
  the harness will exercise the drop-in once Week 5 lands.
- faster-coco-evals — competing tool whose `init_as_pycocotools`
  shape was deliberately not adopted (memory note: avoid copy-paste
  names from competing tools).
