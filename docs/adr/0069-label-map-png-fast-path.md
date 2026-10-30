# ADR-0069: Decode label-map PNGs through a fast path, the `png` crate as fallback

- **Status:** proposed
- **Date:** 2026-10-30
- **Deciders:** @NoeFontana
- **Consulted:** —
- **Informed:** all contributors

## Context and problem statement

Panoptic and semantic evaluation read their label maps from PNG: 8-bit RGB
for panoptic segment ids (ADR-0025), 8-bit grayscale for semantic class ids
(ADR-0037's `evaluate_from_pngs`). Both decode with the `png` crate, and once the PQ kernel and
the confusion fold were cheap, decoding was most of either evaluation.
Within decoding, inflating is the small part: on COCO panoptic val2017 a
PNG takes 0.50 ms to decode, of which inflating its ~8 kB of zlib data
takes 0.10 ms. The rest is unfiltering, and most of that is Paeth rows.

A general decoder has to treat Paeth as a serial per-byte dependency
chain: each byte's predictor reads the byte reconstructed just before it.
A label map is piecewise constant, which collapses the chain. When a
byte's left neighbour equals the byte above that neighbour, the Paeth
predictor is the byte above, so a zero residual reproduces it — and the
next byte along then sees the same equality. A run of zero residuals that
starts in that state is a copy of the row above. Likewise a zero Sub
residual repeats the pixel to its left. On label maps both runs are long,
and both are found sixteen bytes at a time.

## Decision drivers

- Decoded pixels must be exactly the PNG's: no approximation, nothing
  for parity to tolerate.
- Every error, for every malformed or unusual file, must stay what it
  is: the `png` crate's, through the same `PanopticError` /
  `SemanticError` variants.
- No new unsafe code, and no new dependency for `vernier-core` or the
  wasm build.

## Considered options

1. **Keep the `png` crate** for everything.
2. **Upstream** a label-map fast path to the `png` crate.
3. **Own a label-map decoder** — chunk walker, CRCs, inflate and unfilter.
4. **Drive the `png` crate's public `StreamingDecoder`** for the container
   and inflate, and own only a label-map-aware unfilter, falling back to
   the `png` crate's `Decoder` for anything else.

## Decision outcome

Chosen option: **4**. `vernier_mask::label_png` (behind the crate's `png`
feature, which the panoptic and semantic crates enable) runs
`png::StreamingDecoder` with its default options over the file, inflating
the image data into the caller's buffer, so the fast path accepts a file
exactly when the `png` crate's reader would. It then takes only 8-bit
grayscale or RGB, non-interlaced, of the layout the caller asked for,
whose image data is exactly the length the header implies, with every
filter type valid — all checked before any pixel is returned — and
unfilters it in place. Any refusal returns `None` and the caller decodes
with `png::Decoder` as before, which then produces the value or the
error.

The unfilter is the only part vernier owns: the span shortcuts above,
written as fills and block copies, monomorphized per bytes-per-pixel,
and run row by row as the consumer reads, while the row above is in
cache.
Semantic decodes straight into its reusable label-map buffer; panoptic
reads rows from a per-thread scratch buffer.

This widens `vernier-mask` (ADR-0009: RLE codec, rasterizer, mask kernels)
to label-map image input. It is the one leaf crate both paradigm crates
depend on, the code is mask-shaped data in, pixels out, and the feature
keeps the `png` dependency out of `vernier-core` and the wasm build; a
new crate for ~250 lines would not pay for itself.

Measured on COCO val2017, one CPU, each pair in the same run: decoding
the 5000 panoptic GT PNGs 2.55 → 1.10 s, the 5000 semantic GT PNGs
1.18 → 0.54 s, all 10,000 byte-identical to the `png` crate. On images
that are not label maps (random RGB) it is no slower than the `png`
crate: the unfilter keeps the left and upper-left pixels in registers
and uses stb_image's branch-free Paeth, as the `png` crate does, and the
span checks cost one compare per pixel when they do not fire.

### Consequences

- **Positive:** decoding stops being most of a panoptic or semantic
  evaluation. Strict parity is untouched: the pixels are the file's, and
  acceptance is the `png` crate's.
- **Negative:** vernier owns a PNG unfilter. Its correctness rests on
  differential tests against the `png` crate: proptest images under each
  filter type and the adaptive choice, corrupted files (refused, or
  decoded identically), hand-built rows for paths an encoder does not
  choose, and the COCO val2017 label maps.
- **Neutral:** the fast path holds the whole inflated image (`H × (1 +
  stride)` bytes) instead of streaming through two rows; a 640 × 480 RGB
  map is 0.9 MB, and panoptic releases a scratch buffer above 16 MiB.

## Pros and cons of the options

### Option 1

- 👍 No decoder code to own.
- 👎 Leaves decoding at roughly half of panoptic evaluation time.

### Option 2

- 👍 Everyone using the `png` crate gains.
- 👎 The span shortcut pays only on piecewise-constant images, which a
  general-purpose decoder rarely sees; acceptance and release timing are
  outside this project. Worth offering upstream regardless.

### Option 3

- 👍 No dependency on the `png` crate's lower-level API.
- 👎 Re-implements chunk parsing, CRC checks and inflate driving that the
  `png` crate already exposes, and duplicates its acceptance rules —
  ~110 more lines to own, and a second definition of "valid PNG".

### Option 4 (chosen)

- 👍 The speedup where it is, with the container and every acceptance and
  error decision left to the `png` crate.
- 👎 Two decoders in the path; the fast one must refuse rather than
  guess, which its up-front checks are there to guarantee.

## Links and references

- Related ADRs: [ADR-0009](0009-split-mask-kernels-into-vernier-mask-crate.md)
  (`vernier-mask` as the leaf mask crate), [ADR-0025](0025-panoptic-api.md)
  (panoptic), [ADR-0037](0037-generalize-semantic-kernel-class-id.md)
  (semantic `evaluate_from_pngs`).
- PNG specification §9 (filtering), §9.4 (Paeth predictor).
