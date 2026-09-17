# ADR-0054: Split JSON ingestion across the thread budget, and round floats correctly

- **Status:** proposed
- **Date:** 2026-09-17
- **Deciders:** @NoeFontana
- **Consulted:** —
- **Informed:** —

## Context and problem statement

Loading is the largest phase of an evaluation that isn't matching. On
LVIS v1 val (191 MiB of ground truth) `serde_json::from_slice` takes
750 ms of a 2.7 s run — 30 % — and every millisecond of it is serial,
on one core, while the rest of the thread budget waits. ADR-0047
parallelized matching and ADR-0050 `accumulate`; ingestion was left
whole, so it now caps the achievable speedup.

The same code path also carries a known parity defect. `serde_json`'s
default float parser is not correctly rounded: it sends some near-tie
decimals to the adjacent double. On the DETR-R50 real-prediction gate
that surfaced as ~16 % of `eval_imgs.dtScores` drifting by exactly
1 ULP against pycocotools, which reads the same bytes through CPython's
`strtod`. `docs/engineering/real-predictions-parity.md` records the
consequence: `dtScores` and the `scores` tensor sit at **aligned** tier
while the 12-stat summary stays **strict**, because AP depends on
detection *order* and 1 ULP does not reorder. The doc carries a
follow-up to "tighten `serde_json`'s f64 parser or normalise scores at
ingest".

These are one change because they are one dependency and one hot path,
and because splitting the parse would otherwise have to be re-validated
against a parser that is about to change.

## Decision drivers

- Parsed values must be bit-identical to the serial path. The loader
  feeds every downstream tier; an ingestion that is "equivalent" is not
  good enough.
- Input order must be preserved exactly: `loadRes` assigns detection
  ids by position (quirk **J1**), so a reordered parse is a
  *semantically* different dataset, not just a differently-ordered one.
- ADR-0047's contract: `num_threads=None` stays serial and never
  enters rayon.
- Errors on malformed input must keep their current text and byte
  offsets. Users debug against them.

## Considered options

1. **Keep the serial parse.** Accept the Amdahl ceiling.
2. **Swap in a SIMD JSON parser** (`simd-json`).
3. **Structurally split the arrays, deserialize the elements with
   `serde_json` in parallel**, and enable `float_roundtrip`.
4. **A schema-specialized columnar visitor** deserializing straight
   into per-field vectors.

## Decision outcome

Chosen option: **Option 3.**

`json_split` performs one structural scan — tracking brackets, quotes
and escapes, building no values — that yields the byte range of every
element of the requested arrays. Ranges are grouped into byte-balanced
chunks and each chunk is handed to the *same* `serde_json` element
deserializer the serial path uses. Nothing here reimplements number
parsing, string unescaping or field matching, which is what makes the
result bit-identical rather than merely equal; order is preserved
because chunks are concatenated in input order.

`float_roundtrip` switches `serde_json` to a correctly-rounded parser,
making vernier's doubles bit-equal to CPython's for every literal.

### Fallback is the design, not an afterthought

Anything the splitter does not plainly recognize returns `None` and the
caller runs the serial loader: a document shape other than the expected
one, elements that are not objects, a **duplicate key** (which `serde`
rejects and a first-match scan would silently accept), and any parse
error. That last case is what contains scanner bugs: a mis-split leaves
a fragment that does not parse, and the serial re-run then reports the
canonical error at the canonical offset. The failure mode is "slower
and correct", never "faster and wrong".

### Two measured details

- **The scan had to be fused.** Finding the arrays and splitting them
  into elements are both full passes over the document. Done in
  sequence they cost 168 ms + 160 ms at LVIS scale — more than the
  parallel parse they enable, for a net speedup of 1.5×. One pass that
  emits both takes the same 168 ms once.
- **`memchr` belongs in exactly one of the two loops.** In the
  container walk, where runs of digits sit between structural bytes,
  it cut the scan from 168 ms to 29 ms. In the string walk it was a
  *loss*: JSON field names put the next quote ~10 bytes away, and at
  that distance the per-call SIMD setup costs more than the bytes it
  skips. The string scanner is a plain byte loop, deliberately, with a
  comment saying so.

### Consequences

- **Positive:** GT parse on LVIS v1 val (191 MiB, 244 707 annotations)
  drops 732 ms → 204 ms at 8 threads, a **3.6×**, with annotations,
  images and categories identical (`cargo run --release --example
  json_parse_profile`). End to end, `evaluate_bbox_grid` on that
  dataset at `num_threads=8` goes **1142 ms → 554 ms (−51 %)**.
- **Positive:** the 1-ULP parser drift is retired at the root. The
  `float_roundtrip` guard is pinned by a Rust test against Rust's own
  correctly-rounded `str::parse` and by a Python test that compares
  `dtScores` bit-for-bit against CPython's `json` module — both
  verified to fail with the feature off.
- **Negative:** `num_threads=None` gets *slower*: 1606 ms → 1670 ms
  (+4 %) on the same cell, because correct rounding is not free and the
  serial path gains no parallelism to pay for it. This is the price of
  retiring an aligned-tier band, and it is charged to the path that
  opted out of threads.
- **Negative:** a hand-written JSON scanner is a new class of thing to
  get wrong. It is ~150 lines, has no `unsafe`, and its failure mode is
  a fallback; the test module covers structural bytes inside strings,
  escaped quotes and backslashes, nested containers, whitespace in
  every structural position, duplicate keys, trailing commas, empty and
  non-object arrays, and every chunk count from 1 to 16 against
  `serde_json::from_slice`.
- **Neutral:** peak memory grows by one `Range` per element during the
  scan (~4 MB at LVIS scale, ~20 MB at Objects365 scale).
- **Neutral:** `Dataset.from_json` / `from_lvis_json` are static Python
  constructors with no thread policy, so they stay serial. Wiring a
  budget through them is an API change and a separate decision; the
  evaluate entries, which already carry a `ThreadPolicy`, get the win
  today.

## Pros and cons of the options

### Option 1 (serial parse)

- 👍 Nothing to review.
- 👎 Leaves the largest non-matching phase entirely serial, and leaves
  the parser drift in place.

### Option 2 (`simd-json`)

- 👍 Fast, and well-tested as parsers go.
- 👎 A different number parser is a new parity surface on the exact
  axis this ADR is trying to close, needs a mutable input buffer, and
  replaces a dependency the project already trusts.

### Option 3 (structural split + `float_roundtrip`, chosen)

- 👍 Bit-identical by construction; order-preserving; degrades to the
  serial loader; retires a documented parity band.
- 👎 A hand-written scanner; a 4 % tax on the sequential path.

### Option 4 (columnar visitor)

- 👍 Would cut allocation as well as parallelize, which is the next
  ceiling (the parallel parse scales 4.6×, not 8×, on 8 threads —
  allocator contention).
- 👎 Reimplements field matching and per-field parsing for every
  annotation shape vernier supports, which is precisely the parity
  surface option 3 avoids. Worth revisiting only with option 3's tests
  already in place.

## Links and references

- Retires the follow-up in
  `docs/engineering/real-predictions-parity.md` §"Follow-up parity
  item"; the DETR-R50 aligned-tier band should be re-measured against a
  build with this change before it is narrowed.
- Composes with [ADR-0047](0047-threading-model.md) (thread policy),
  [ADR-0050](0050-parallel-accumulate.md) and
  [ADR-0051](0051-occupied-cell-visiting.md).
- The array-ingest path of [ADR-0030](0030-buffer-protocol.md) is
  already parse-free and is unaffected.
- Measurements: `crates/vernier-core/examples/json_parse_profile.rs`.
