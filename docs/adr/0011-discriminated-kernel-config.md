# ADR-0011: Discriminated kernel config replaces the `iou_type` string literal

- **Status:** accepted
- **Date:** 2026-04-28
- **Deciders:** @NoeFontana
- **Consulted:** —
- **Informed:** —

## Context and problem statement

Phase 1 shipped `vernier.Evaluator` with `iou_type: Literal["bbox", "segm"]`,
mirroring `pycocotools.cocoeval.Params.iouType`. This shape works as long as
every kernel takes its parameters from the same place — pycocotools'
`Params` carries the union of every kernel's tunables (`iouThrs`, `recThrs`,
`maxDets`, `kpt_oks_sigmas`, …) on a single object, with the contract that
fields are silently inert when the wrong `iouType` is selected. ADR-0010
breaks the assumption: `BoundaryIoU` carries a `dilation_ratio` parameter
that has no analogue in the existing `Params` and that LVIS sets to a
non-default value (`0.008` rather than `0.02`).

ADR-0010's "Public API" section therefore proposed *two* parallel surfaces —
the literal `iou_type="boundary"` (default ratio) *and* a `BoundaryIoU`
config class (custom ratio) — accepted in alternation by the same field.
That works but bakes a string-or-instance dispatch into a public surface
that Phase 3 keypoints will inherit (per-category sigmas have the same
shape as `dilation_ratio`: a kernel-local parameter with no parity-harness
analogue). Each new kernel that takes parameters either lengthens the
`Evaluator` dataclass (parameter sprawl, conditionally-relevant fields) or
adds a new "string-or-instance" path.

The single-field, multi-variant shape that fits this is a discriminated
union: `IouKind = Bbox | Segm | Boundary`, with each variant a frozen
dataclass carrying its own parameters. Modern Python (PEP 604 unions, PEP
634 `match`, `pyright --strict`) types this naturally; the Rust side
already has it (`EvalIouType` is an enum with a `Boundary { dilation_ratio
}` arm in this PR).

## Decision drivers

- **No `Evaluator` parameter sprawl as kernels gain config.** Phase 3's
  keypoints OKS adds per-category sigmas; this should not become another
  conditional field on `Evaluator`.
- **Single source of truth for what's selectable.** The set of kernels
  appears in exactly one place (the `IouKind` union); pyright catches
  exhaustiveness on `match` without a runtime sentinel.
- **Mirror the Rust enum.** The FFI layer is already a `match self.iou:`
  away from the Rust `EvalIouType` enum; the Python surface should carry
  the same shape rather than translate string sentinels into enum arms.
- **Pre-1.0 freedom.** No deprecation cycle is owed; the old
  `iou_type="bbox"` literal has shipped only in 0.x test/parity code, and
  the breaking change happens in the same PR that adds boundary support.

## Considered options

1. **Keep the `Literal` and add `dilation_ratio` to `Evaluator`.** Sibling
   fields whose meaning depends on which sibling is set.
2. **String-or-instance dispatch** (`iou_type: Literal[…] | BoundaryIoU`).
   The shape ADR-0010 sketched. Adds a parallel surface per kernel-with-
   parameters.
3. **Discriminated dataclass union** (`iou: Bbox | Segm | Boundary`). One
   field, parameterized variants, exhaustive `match` typing.
4. **ABC class hierarchy** (`class IouKind(ABC)` + subclasses with
   `to_ffi_args()` methods). Couples the user-facing type to dispatch
   behavior; less ergonomic with frozen dataclasses; weaker pyright
   narrowing inside `match`.

## Decision outcome

Chosen option: **3 — discriminated dataclass union**.

The Python public API becomes:

```python
@dataclass(frozen=True, slots=True)
class Bbox: ...
@dataclass(frozen=True, slots=True)
class Segm: ...
@dataclass(frozen=True, slots=True)
class Boundary:
    dilation_ratio: float = 0.02

IouKind = Bbox | Segm | Boundary

@dataclass(frozen=True, slots=True)
class Evaluator:
    iou: IouKind = field(default_factory=Bbox)
    parity_mode: ParityMode = "corrected"
    max_dets: tuple[int, ...] = (1, 10, 100)
    use_cats: bool = True
```

`Evaluator.evaluate` dispatches via `match`:

```python
match self.iou:
    case Bbox():     run = evaluate_bbox_summary; extra = ()
    case Segm():     run = evaluate_segm_summary; extra = ()
    case Boundary(dilation_ratio=r):
                     run = evaluate_boundary_summary; extra = (r,)
```

The migration in this PR drops the old `iou_type: IouType` field and the
`IouType = Literal["bbox", "segm"]` alias from `vernier/__init__.py`.
Net-new users get the union; pre-1.0 callers update once. The
`patch_pycocotools` shim continues to accept `iouType="bbox" |
"segm" | "boundary"` (per ADR-0010) and translates internally — that
surface mirrors `pycocotools` and is decoupled from this decision.

The Rust FFI surface follows the same shape: separate
`evaluate_bbox_summary`, `evaluate_segm_summary`, `evaluate_boundary_summary`
pyfunctions. Boundary takes `dilation_ratio: f64` as a positional argument;
the other two don't. This keeps each FFI signature a function of exactly
the parameters its kernel needs — no `Option<f64>` placeholder threading
through bbox/segm.

ADR-0005 locks the `Similarity` trait and the matching-engine API; it does
not lock the `Evaluator` Python surface, so no superseding is required.
ADR-0010 §"Public API" is partially reshaped by this ADR — the
`iou_type="boundary"` literal goes away in favor of `Boundary(...)`, the
`BoundaryIoU` class name becomes `Boundary`, and the `vernier.evaluate(...,
iou_type=...)` example becomes `vernier.Evaluator(iou=Boundary(...))`. The
mathematical content of ADR-0010 (algorithm, oracle, parity infrastructure,
performance baseline) is untouched.

### Consequences

- **Positive:** Phase 3 keypoints add a `Keypoints(sigmas: ...)` variant
  with no `Evaluator` change. `pyright` flags any kernel added without a
  `match` arm. The Python surface mirrors the Rust enum, so the FFI layer
  is a one-line `match` per kernel.
- **Positive:** Custom kernel parameters live where they're typed:
  `Boundary(dilation_ratio=0.008)` for LVIS. No
  conditionally-relevant `Evaluator` fields.
- **Negative:** Breaking change for the (small, internal) population
  passing `iou_type="bbox"`. The migration is mechanical.
- **Negative:** Frozen dataclasses with empty bodies (`Bbox`, `Segm`)
  feel ceremonious vs `"bbox"` / `"segm"`. The cost buys uniform shape
  across kernels-with-parameters and kernels-without.
- **Neutral:** Slight drift from `pycocotools.cocoeval.Params.iouType`
  (still a string). The `patch_pycocotools` shim absorbs the difference;
  vernier's own extended API does not need to mirror pycocotools shape-for-
  shape.

## Pros and cons of the options

### Option 1 — `Literal` + sibling fields

- 👍 Smallest patch; one new `Evaluator` field per kernel parameter.
- 👎 Conditionally-relevant fields are stringly-typed invariants. Pyright
  cannot express "this field is only valid when `iou_type='boundary'`".
- 👎 Phase 3 doubles down: per-category sigmas as another optional field.

### Option 2 — string-or-instance

- 👍 Backward-compatible; `iou_type="bbox"` keeps working.
- 👎 Two parallel selection axes for kernels with parameters. Every
  kernel-with-config has a "default" string form *and* a class form, with
  divergent ergonomics.
- 👎 Pyright cannot exhaustively match `Literal["bbox","segm"] |
  BoundaryIoU` arms in a single `match` without ad-hoc isinstance checks.

### Option 3 — discriminated dataclass union (chosen)

- 👍 Single field, single `match`, exhaustive typing.
- 👍 Mirrors the Rust enum 1:1; FFI dispatch is mechanical.
- 👎 Pre-1.0 breaking change for callers using `iou_type=` keyword.
- 👎 Empty-bodied dataclasses for parameterless kernels.

### Option 4 — ABC class hierarchy

- 👍 Methods (e.g., `to_ffi_args()`) live on the type.
- 👎 Couples user-facing data type to dispatch behavior; harder to keep
  the dispatch logic in one place near the FFI boundary.
- 👎 Pyright's narrowing inside `match isinstance(...)` is weaker than
  inside `match Bbox():`.

## Links and references

- ADR-0001 §"Affect the public API" — change requires an ADR.
- ADR-0005 — locks `Similarity` and matching-engine APIs (not affected).
- ADR-0010 §"Public API" — partially reshaped; mathematical content unchanged.
- PEP 634 (Structural Pattern Matching), PEP 604 (Union types `X | Y`).
