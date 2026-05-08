# ADR-0016: Generalize the A-axis as a value-typed `Breakdown`

- **Status:** accepted
- **Date:** 2026-04-29
- **Deciders:** @NoeFontana
- **Consulted:** —
- **Informed:** —

## Context and problem statement

The pycocotools accumulator and summarizer expose a five-axis tensor
`(T, R, K, A, M)` — IoU thresholds, recall thresholds, categories,
**area ranges**, and per-image detection caps. The `A`-axis has been
synonymous with COCO's four area buckets (`all` / `small` / `medium` /
`large`) since the COCO eval shipped, and `vernier-core` followed:
`evaluate::AreaRange` carries `(index, lo, hi)` for the orchestrator's
binning predicate, `summarize::AreaRng` carries `(index, label)` for
the summary table, and a fixed pair of `coco_default()` / 
`keypoints_default()` constructors hardcoded the layout. Quirk **D5**
(strict, ratified by ADR-0012) drops the `small` bucket for keypoints,
so even the "default" is already two layouts.

Two inbound forces stretch this shape:

1. **CrowdPose's `crowdIndex` axis.** CrowdPose's eval slices the same
   `(T, R, K, ?, M)` accumulator on a *crowd-density index* rather than
   on area. The slot is identical — a per-annotation key, an
   inclusion predicate, a label, and a position on the same A-axis;
   only the meaning of "area" differs. Naming the axis after one of
   its instances ("area") locks future Breakdowns out of the same
   slicing infrastructure.
2. **Robotics evaluation use cases.** Depth, occlusion, and viewpoint
   are equally natural slice axes; users have asked for fine-grained
   five- or six-bucket area splits in regression suites. Today the
   only path is to hand-build a parallel accumulator and summarize
   plan, defeating the point of the shared kernel.

Both forces want the same thing: the A-axis as a *value*, not as a
hardcoded predicate. The pycocotools-strict default must keep
producing byte-identical numbers — quirks **D4** (literal `areaRng`
values), **D5** (kp drops `small`), and **D6** (closed-on-both-ends
inclusion) are all keyed to the canonical four-bucket layout.

## Decision drivers

- **Strict-parity contract is non-negotiable.** ~25 fixtures, plus
  boundary parity, plus the queued COCO val2017 non-regression check,
  must continue producing byte-identical numbers under the default.
  The new shape has to *be* the old shape when both buckets and
  labels match the canonical layout.
- **Quirks survey honesty (ADR-0002 three-tier).** `D6` is closed on
  both ends. A "bucket" type that uses `Range<f64>` (half-open) silently
  re-bins boundary annotations and breaks parity. The data type must
  encode the actual semantics, not a Rust convention.
- **Surface stays internal until it's not.** The PR exposing this in
  Python (CrowdPose-driven) is a separate ADR. Today the FFI shape
  must stay byte-identical to keep the wheel ABI and the `~/python`
  tests stable.
- **Ergonomics for custom plans.** A user who wants a 5-bucket area
  split should be able to construct it in five lines of safe Rust and
  pass it through the existing `summarize_with` plumbing.

## Considered options

1. **Status quo plus per-kernel hardcoded constructors.** Ship
   `AreaRange::crowdpose_default()` and call it done.
2. **Break `AreaRange` open into a generic `Bucket<K>` with a `Key`
   trait.** Type-parameterize the accumulator's A-axis on the bucket
   key type so `f64` (area) and `f64` (crowd index) are distinct types.
3. **Value-typed `Breakdown { axis, buckets }` over `f64` keys, with
   `coco_area_det` / `coco_area_keypoints` constructors and
   per-bucket `to_area_range` / `to_area_rng` lift methods.** A
   non-generic bridge type that keeps the existing `AreaRange` /
   `AreaRng` shapes intact.
4. **Move the breakdown into `EvaluateParams` as a closure
   `Fn(&Annotation) -> usize`.** Treat the A-axis as a free function
   from annotation to bucket index, with the labels supplied
   separately to the summarizer.

## Decision outcome

Chosen option: **3 — value-typed `Breakdown { axis, buckets }`**.

The new module `vernier_core::breakdown` introduces:

```rust
pub struct Bucket {
    pub index: usize,
    pub label: Cow<'static, str>,
    pub lo: f64,
    pub hi: f64,
}

pub struct Breakdown {
    axis: Cow<'static, str>,
    buckets: Vec<Bucket>,
}
```

`Bucket::contains(key)` is `key >= lo && key <= hi` — closed on both
ends, mirroring pycocotools' `cocoeval.py:251` `not (area < lo or area
> hi)` (quirk **D6** strict). The doc-comment calls this out so a
future reader doesn't reach for `Range<f64>`.

`Breakdown::coco_area_det()` and `Breakdown::coco_area_keypoints()`
construct the canonical four- and three-bucket layouts. Two pinning
tests assert bit-equal `f64::to_bits()` bounds against the legacy
`AreaRange::coco_default()` / `keypoints_default()`, and a third test
runs the static `StatRequest::coco_detection_default()` and the
Breakdown-built `bd.detection_plan()` over the same `Accumulated` and
asserts every stat's bits match.

`Breakdown::area_ranges()` and `summary_areas()` lift a `Breakdown`
into the existing `Vec<AreaRange>` and `Vec<AreaRng>` shapes the
orchestrator and summarizer already consume. `AreaRange` and `AreaRng`
themselves are untouched — no FFI breakage, no fixture drift.

The JSON output schema stays at v1 (ADR-0015): no `breakdown_axis`
field is added to `cli-output-schema.md`, because no public surface
yet emits one. When the CrowdPose follow-up promotes `Breakdown` to
the FFI, the schema bump (v1 → v2) is part of *that* ADR; today's PR
preserves the v1 contract verbatim.

### Consequences

- **Positive:** Fine-grained area splits, depth/occlusion axes, and
  the CrowdPose `crowdIndex` axis become a leaf change at the
  Breakdown construction site rather than a fork of the accumulator
  / summarizer pair. The `axis` field reserves a name for the
  upcoming JSON schema bump without committing to it today.
- **Positive:** `Bucket` encodes the closed-on-both-ends semantics in
  its public type. A future reader cannot accidentally introduce
  half-open semantics by typing `Range<f64>` — the type does not
  permit it.
- **Negative:** Two coexisting shapes for "an A-axis bucket"
  (`Bucket` + `AreaRange` + `AreaRng`) until the FFI follow-up. The
  duplication is by-design — collapsing them now would break the
  byte-identical-default invariant — but it is real overhead.
- **Negative:** `Breakdown` is `f64`-keyed. Future axes that need
  non-numeric keys (e.g., a per-image `subset` enum) will need
  another ADR; option 2 (generic `Bucket<K>`) is the natural
  successor when that arrives.
- **Neutral:** No change to wheel ABI, no change to `vernier._core`
  exports, no change to the JSON schema. Pyright `--strict` and
  `cargo deny` see no diff.

## Pros and cons of the options

### Option 1 — hardcoded constructors

- Pros — minimal surface, no new module.
- Cons — every new axis hardcodes a layout in `vernier-core`;
  CrowdPose, depth, occlusion all become library-owned constants. No
  affordance for user-defined breakdowns. Doesn't address the
  duplication between `AreaRange` and `AreaRng`.

### Option 2 — generic `Bucket<K>` over a `Key` trait

- Pros — most expressive long-term shape; non-numeric keys come for
  free; the type system rules out cross-axis index mixups.
- Cons — propagates a type parameter through `AreaRange`,
  `AreaRng`, `EvaluateParams`, `AccumulateParams`, and
  `summarize_with`, breaking every call site at once. The strict-
  parity contract is hard to satisfy under that scale of change in
  one PR. The FFI surface (`vernier._core` exports `AreaRange` by
  value) doesn't have a natural way to surface a generic.

### Option 3 — value-typed `Breakdown` (chosen)

- Pros — non-generic, fits in one new module, leaves every existing
  type untouched. The `axis_name` field gives the CrowdPose follow-
  up an unambiguous slot to extend without a rename. Closed
  semantics are encoded in the constructor, not on a boolean
  parameter. Default constructors are byte-identical to the legacy
  path under `to_bits()` comparison.
- Cons — `f64`-keyed only. A future ADR will need to handle
  non-numeric keys, but the upgrade path is well-known (option 2).
  Two parallel shapes (`Bucket` + `AreaRange`) coexist until the FFI
  follow-up, which is honest duplication, not hidden complexity.

### Option 4 — closure `Fn(&Annotation) -> usize`

- Pros — most general; user can compute the bucket from any
  annotation field, including Python ones via FFI.
- Cons — the orchestrator currently binds bucket *bounds* into
  `EvalImageMeta::area_rng` so a downstream consumer can recover
  what bucket an image was sliced into. A closure throws that
  affordance away — labels and bounds disappear at the FFI
  boundary, breaking the JSON schema's `area` field and the
  pretty-print output. Also inverts the data flow: today the
  evaluator owns the predicate, not the caller.

## Links and references

- ADR-0001: Records architecture decisions; gates significance.
- ADR-0011: Discriminated kernel config — the precedent for
  evolving a typed-enum surface ahead of new kernels rather than
  retrofitting.
- ADR-0012: OKS keypoints surface — explicitly mentions a future
  Breakdown ADR as the home for kp-A-axis quirks (D5).
- ADR-0015: vernier CLI — the JSON schema (v1) this ADR
  *deliberately leaves alone*; a future ADR will bump it when the
  FFI surfaces `Breakdown`.
- `docs/engineering/pycocotools-quirks.md` — quirks **D4 / D5 / D6**
  are the strict-parity invariants this design preserves.
