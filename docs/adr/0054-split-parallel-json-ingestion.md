# ADR-0054: Split JSON ingestion across the thread budget, and round floats correctly

- **Status:** proposed (amended 2026-09-17, still unmerged — see
  "Amendment: hardening the split")
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
  comment saying so. (Amended below: it is a byte loop for the first
  32 bytes and `memchr2` after that.)

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
- **Positive (amendment):** `num_threads=1` is no longer a 0.74–0.94×
  regression against the serial loader; it *is* the serial loader.
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

## Amendment: hardening the split (2026-09-17)

Written against this ADR while it is still `proposed` and unmerged, so
it amends the text above rather than superseding it. Three claims above
did not survive review; the decision itself does.

### The error-fidelity guarantee was aspirational, not enforced

The driver "errors on malformed input must keep their current text and
byte offsets" was not met. The splitter parses only the elements of the
arrays it was asked for; the rest of the document it *walked*. Three
walks accepted what `serde_json::from_slice` rejects, so acceptance was
a function of `num_threads`:

| input                                    | serial                                    | parallel (before) |
| ---------------------------------------- | ----------------------------------------- | ----------------- |
| `[{…}][{…}]` (two concatenated arrays)   | `trailing characters at line 1 column 70` | `Ok`, first array only — silent data loss |
| `{…}}}}garbage`                          | `trailing characters at line 1 column 117`| `Ok`              |
| `{"info": tru, …}`                       | `expected ident at line 1 column 13`      | `Ok`              |
| `{"info": NaN, …}`                       | `expected value at line 1 column 10`      | `Ok`              |
| `{"info": 01+2x, …}`                     | `invalid number at line 1 column 11`      | `Ok`              |

One root cause: the walk was never asked whether what it walked over
was legal. Two rules close it, both resolving *towards* the serial
loader rather than towards a second error-reporting implementation:

- **Nothing may follow the document.** Both entry points now require
  whitespace after the closing bracket.
- **A skipped member is skipped by `serde_json` itself.** The scalar
  "run to the next structural byte" arm is gone; an unrequested member
  now goes through the same `IgnoredAny` deserializer a derived
  `Deserialize` impl uses for an unknown field, which both validates it
  and reports where it ends. Its *key* goes through `String`, because a
  derived impl always decodes a key to match it against the known field
  names — `IgnoredAny` skips a string without decoding, and let a raw
  non-UTF-8 byte in a key through. That one was found by the property
  test, not by hand.

Both rules return `None`, so the serial loader produces the message.
The guarantee is now "the parallel path accepts exactly what the serial
path accepts, and rejects with byte-identical text", pinned by an
equivalence test over the malformed shapes above and by three proptest
properties that overwrite, insert and truncate bytes anywhere in the GT
and DT fixtures and assert both paths agree — on acceptance, on the
loaded values, and on the error string — for every `num_threads` in
1..=8.

One residual is accepted: `serde_json`'s 128-deep recursion limit is
counted from the start of whatever slice it is handed, so an element
nested within one or two levels of that limit is accepted by the split
and rejected serially. Closing it costs a second pass to count true
depth; real COCO and LVIS payloads nest four deep.

### `num_threads=1` was slower than serial, not equal to it

`chunks_for(1)` still cut four chunks, so a one-thread budget paid for
the structural scan, the `Range` vector and the rayon dispatch and got
nothing back. Measured best-of-5, 8 cores, release profile, parallel ÷
serial:

| payload                       | t1   | t2   | t3   | t4   | t8   |
| ----------------------------- | ---- | ---- | ---- | ---- | ---- |
| Objects365 bbox DT, 300k det  | 0.87 | 1.09 | 1.27 | 1.32 | 1.47 |
| COCO segm DT (RLE)            | 0.74 | 1.17 | 1.19 | 1.42 | 1.24 |
| LVIS segm DT (RLE)            | 0.84 | 1.35 | 1.50 | 1.66 | 1.97 |
| COCO GT `instances_val2017`   | 0.94 | 1.67 | 1.90 | 2.14 | 2.98 |

`num_threads=1` is a setting callers pass — `test_json_float_parity.py`
parametrizes it — so all three parallel loaders now short-circuit to
their serial sibling below two threads. The crossover is between one
thread and two on every payload measured, which is lower than a guess
would have put it. After the change t1 measures 1.00–1.02×.

### `memchr` in the string walk was a loss only for short strings

The original measurement is right about field names and wrong about
`segmentation.counts`, which is multi-KB and dominates segm GT and DT —
and the string walk is fully serial, so it caps the achievable speedup.
`skip_string` is now a byte loop for the first 32 bytes and `memchr2`
past that. The gate was measured, not guessed (structural scan only,
best of 40, median ms):

| payload                      | scalar | memchr2 | gate 16 | gate 32 | gate 64 |
| ---------------------------- | ------ | ------- | ------- | ------- | ------- |
| Objects365 bbox DT, 171 MiB  | 78.6   | 92.6    | 76.3    | 76.2    | 77.6    |
| COCO GT (polygons), 20 MiB   | 4.67   | 5.62    | 4.76    | 4.65    | 4.81    |
| COCO segm DT (RLE), 18 MiB   | 8.61   | 7.90    | 6.89    | 6.99    | 7.54    |
| LVIS segm DT (RLE), 96 MiB   | 49.2   | 49.6    | 42.6    | 43.1    | 46.3    |

Gate 32 is best-or-tied on three of four and never worse than the byte
loop, which neither 16 (COCO GT) nor 64 (both RLE payloads) manages.
Pure `memchr2` is a 10–18 % loss on the bbox and polygon payloads, so
the original "it was a loss" reading was sound on the payloads it was
taken from. End to end at 8 threads the gate moves COCO segm DT from
1.24× to 1.98–2.07× and COCO GT from 2.98× to 3.57–3.59×.

### `bench-timings` on the parallel path

`GT_FROM_PARTS_NS` / `DT_FROM_INPUTS_NS` were never recorded on the
split path, so `read_and_reset_dataset_timings()` reported index-build
as 0 ms for every parallel bench, and a fallback added the abandoned
scan to `GT_PARSE_NS` on top of the serial loader's own figure. Both
phases are now recorded once each on the success path; the fallback
records nothing of its own and lets the serial loader own both
counters.

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
