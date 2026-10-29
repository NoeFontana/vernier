//! Structural splitting of a JSON array, so its elements can be
//! deserialized in parallel.
//!
//! `serde_json::from_slice` on a ground-truth file is a single serial
//! pass: 750 ms of a 2.7 s LVIS v1 val evaluation, the largest phase
//! after matching. The elements of `annotations` are independent, so
//! the only thing between that pass and the rest of the thread budget
//! is knowing where each element starts.
//!
//! This module finds those boundaries with **one** structural scan — no
//! values are built, only brackets, quotes and escapes are tracked —
//! and hands back byte ranges. Each range is then deserialized by
//! exactly the same `serde_json` element deserializer the serial path
//! uses, in input order. The parsed values are bit-identical, not
//! merely equivalent: nothing here reimplements number parsing, string
//! unescaping or field matching. Input order is preserved because
//! `loadRes` auto-ids are positional (quirk **J1**).
//!
//! Scanning cost is why the scan is fused rather than layered. Finding
//! the arrays and then splitting them into elements are both full
//! passes over the document, and at LVIS scale each costs ~168 ms —
//! together more than the parallel parse they enable. One pass emits
//! both.
//!
//! Everything not plainly recognizable returns `None` and the caller
//! falls back to the serial loader: malformed input, duplicate keys
//! (which `serde` rejects and a first-match scan would silently
//! accept), an unexpected document shape, elements that are not
//! objects, and any scanner bug — a mis-split leaves a fragment that
//! does not parse, and the fallback re-runs the serial loader to
//! produce the canonical error at the canonical byte offset.
//!
//! ## Acceptance must not depend on the thread count
//!
//! The splitter sees the whole document but only *parses* the elements
//! of the arrays it was asked for. Everything else — trailing bytes
//! after the document, and the members the caller did not ask for — is
//! walked, not validated, and a walk that is happy to step over
//! nonsense would accept input `serde_json::from_slice` rejects. That
//! would make acceptance a function of `num_threads`, which ADR-0054
//! rules out: errors on malformed input keep their text and their byte
//! offsets.
//!
//! Two rules close that gap, and both resolve towards the serial
//! loader rather than towards a second error-reporting implementation:
//!
//! - **Nothing may follow the document.** Both entry points require
//!   the bytes after the closing bracket to be whitespace, so
//!   `[{…}][{…}]` and `{…}}}}garbage` fall back and get serde's
//!   `trailing characters` error at serde's offset.
//! - **A skipped member is skipped by `serde_json` itself.**
//!   [`skip_ignored_member`] runs the same `IgnoredAny` deserializer a
//!   derived `Deserialize` impl uses for an unknown field, which both
//!   validates the member and reports where it ends. Nothing here
//!   re-implements "is this a number"; a malformed `"info"` value
//!   returns `None` and the serial loader produces the message.
//!
//! One residual divergence is accepted and not worth a second pass to
//! close: `serde_json`'s 128-deep recursion limit is counted from the
//! start of whatever slice it is handed, so an element (or a skipped
//! member) nested within one or two levels of the document limit is
//! accepted here and rejected serially. Real COCO/LVIS payloads nest
//! four deep.

use std::ops::Range;

use rayon::prelude::*;
use serde::de::{DeserializeOwned, IgnoredAny};

/// Index of the first non-whitespace byte at or after `i`.
///
/// JSON whitespace is exactly space, tab, CR and LF (RFC 8259 §2) — a
/// narrower set than `u8::is_ascii_whitespace`, which also accepts form
/// feed.
fn skip_ws(bytes: &[u8], mut i: usize) -> usize {
    while i < bytes.len() && matches!(bytes[i], b' ' | b'\t' | b'\r' | b'\n') {
        i += 1;
    }
    i
}

/// Bytes of a string scanned by the plain byte loop before
/// [`skip_string`] hands the rest to `memchr2`.
///
/// Measured, not guessed. `memchr` everywhere is a loss on field names
/// — the next quote is ~10 bytes away and the per-call SIMD setup costs
/// more than the bytes it skips — but a `segmentation.counts` RLE
/// string is multi-KB and the byte loop is the serial cap on the whole
/// split. Gating on length gets both. Structural scan, best of 40, 8
/// cores, no `target-cpu=native`, median ms:
///
/// | payload                     | scalar | memchr2 | gate 16 | gate 32 | gate 64 |
/// |-----------------------------|--------|---------|---------|---------|---------|
/// | Objects365 bbox DT, 171 MiB | 78.6   | 92.6    | 76.3    | 76.2    | 77.6    |
/// | COCO GT (polygons), 20 MiB  | 4.67   | 5.62    | 4.76    | 4.65    | 4.81    |
/// | COCO segm DT (RLE), 18 MiB  | 8.61   | 7.90    | 6.89    | 6.99    | 7.54    |
/// | LVIS segm DT (RLE), 96 MiB  | 49.2   | 49.6    | 42.6    | 43.1    | 46.3    |
///
/// 32 is best-or-tied on three of the four and never worse than the
/// byte loop, which 16 is not (COCO GT) and 64 is not (both RLE
/// payloads).
const STRING_SCALAR_GATE: usize = 32;

/// Index just past the string starting at `i` (its opening quote).
///
/// A byte loop for the first [`STRING_SCALAR_GATE`] bytes, `memchr2`
/// beyond it — see that constant for the measurements behind the gate.
fn skip_string(bytes: &[u8], i: usize) -> Option<usize> {
    let mut j = i + 1;
    let gate = (j + STRING_SCALAR_GATE).min(bytes.len());
    while j < gate {
        match bytes[j] {
            // An escape consumes the next byte whatever it is, which is
            // what keeps `\"` from ending the string. `\uXXXX` needs no
            // special case: the four hex digits are ordinary bytes.
            b'\\' => j += 2,
            b'"' => return Some(j + 1),
            _ => j += 1,
        }
    }
    // Past the gate the same two rules hold, found with SIMD. An
    // escape at the gate boundary may have pushed `j` past the end;
    // `bytes.get` then yields `None`, exactly as the byte loop did.
    loop {
        let rel = memchr::memchr2(b'\\', b'"', bytes.get(j..)?)?;
        let at = j + rel;
        if bytes[at] == b'"' {
            return Some(at + 1);
        }
        j = at + 2;
    }
}

/// Index just past the object or array starting at `i`.
///
/// Counts only the bracket kind it opened with. Nested containers of
/// the other kind are balanced within a well-formed document, so
/// ignoring them is safe; a malformed one runs off the end and yields
/// `None`.
fn skip_container(bytes: &[u8], i: usize) -> Option<usize> {
    let open = *bytes.get(i)?;
    let close = match open {
        b'{' => b'}',
        b'[' => b']',
        _ => return None,
    };
    let mut depth = 0usize;
    let mut j = i;
    loop {
        let rel = memchr::memchr3(b'"', open, close, bytes.get(j..)?)?;
        let at = j + rel;
        if bytes[at] == b'"' {
            j = skip_string(bytes, at)?;
            continue;
        }
        if bytes[at] == open {
            depth += 1;
        } else {
            depth -= 1;
            if depth == 0 {
                return Some(at + 1);
            }
        }
        j = at + 1;
    }
}

/// Index just past the `"key": value` pair starting at `key` (its
/// opening quote), for a member the caller did not ask for — validated
/// by `serde_json`, not by this module.
///
/// A skipped member still has to be *legal*, or the parallel path would
/// accept documents the serial path rejects: `{"info": tru, …}` and
/// `{"info": 01+2x, …}` are errors at a specific byte offset and must
/// stay errors. Rather than re-derive JSON's number and literal
/// grammars — a second implementation to keep in step with serde's —
/// this hands the bytes to the very deserializer a derived
/// `Deserialize` impl uses for an unknown field, `IgnoredAny`, and
/// reads the end position back off the stream. Wrong input therefore
/// returns `None` and the serial loader reports it, with serde's own
/// message at serde's own offset.
///
/// The key goes through `String`, not `IgnoredAny`, and the difference
/// is load-bearing: a derived impl always *decodes* an unknown key —
/// it has to, to compare it against the known field names — so an
/// invalid escape, an unpaired surrogate or a raw non-UTF-8 byte in a
/// key is an error serially. `IgnoredAny` skips a string without
/// decoding it and would have let those through. (A requested key
/// matched raw bytes and is therefore escape-free ASCII, so it needs
/// no check.) Inside the ignored *value*, by contrast, `IgnoredAny` is
/// exactly right: it is what the serial path uses there too, so the
/// two agree on strings it does not decode.
///
/// Cost is bounded by the skipped member, which on COCO and LVIS is
/// `info` and `licenses` — a few hundred bytes against the hundreds of
/// megabytes in `annotations`.
fn skip_ignored_member(bytes: &[u8], key: usize, value: usize) -> Option<usize> {
    json_value_end::<String>(bytes, key)?;
    json_value_end::<IgnoredAny>(bytes, value)
}

/// Index just past the single JSON value starting at `i`, or `None` if
/// `serde_json` will not read a `T` there.
fn json_value_end<T: DeserializeOwned>(bytes: &[u8], i: usize) -> Option<usize> {
    let mut stream = serde_json::Deserializer::from_slice(bytes.get(i..)?).into_iter::<T>();
    stream.next()?.ok()?;
    Some(i + stream.byte_offset())
}

/// `true` when nothing but JSON whitespace follows `end`.
///
/// Without this the parallel path stops reading at the first complete
/// document and silently drops whatever came after it —
/// `[{…}][{…}]` would load half the detections instead of raising
/// `trailing characters`.
fn only_whitespace_follows(bytes: &[u8], end: usize) -> bool {
    skip_ws(bytes, end) == bytes.len()
}

/// Element ranges of an array of objects, plus the index just past its
/// closing bracket. `i` must be the opening `[`.
///
/// `None` when any element is not an object — the arrays this module
/// targets (`images`, `annotations`, `categories`, and the detection
/// payload) always hold objects, and anything else is the caller's
/// cue to fall back.
fn scan_object_array(bytes: &[u8], i: usize) -> Option<(Vec<Range<usize>>, usize)> {
    if bytes.get(i)? != &b'[' {
        return None;
    }
    let mut elements: Vec<Range<usize>> = Vec::new();
    let mut at = skip_ws(bytes, i + 1);
    if bytes.get(at)? == &b']' {
        return Some((elements, at + 1));
    }
    loop {
        if bytes.get(at)? != &b'{' {
            return None;
        }
        let end = skip_container(bytes, at)?;
        elements.push(at..end);
        at = skip_ws(bytes, end);
        match bytes.get(at)? {
            b',' => at = skip_ws(bytes, at + 1),
            b']' => return Some((elements, at + 1)),
            _ => return None,
        }
    }
}

/// Element ranges of a whole-document array of objects.
///
/// The array must *be* the document: anything but whitespace after its
/// closing bracket falls back, so `trailing characters` stays an error
/// instead of becoming a silent truncation.
pub(crate) fn document_object_array(bytes: &[u8]) -> Option<Vec<Range<usize>>> {
    let start = skip_ws(bytes, 0);
    let (elements, end) = scan_object_array(bytes, start)?;
    only_whitespace_follows(bytes, end).then_some(elements)
}

/// Element ranges of the named array members of a top-level object, in
/// the order `keys` lists them.
///
/// One pass over the document: skipping a member's value means walking
/// it, and `annotations` is most of the file.
///
/// `None` when the document is not a plain object, a requested key is
/// missing, a requested value is not an array of objects, a key
/// appears more than once, a skipped member is malformed, or anything
/// but whitespace follows the closing brace.
pub(crate) fn top_level_object_arrays(
    bytes: &[u8],
    keys: &[&str],
) -> Option<Vec<Vec<Range<usize>>>> {
    let start = skip_ws(bytes, 0);
    if bytes.get(start)? != &b'{' {
        return None;
    }
    let mut found: Vec<Option<Vec<Range<usize>>>> = (0..keys.len()).map(|_| None).collect();
    let mut i = skip_ws(bytes, start + 1);
    if bytes.get(i)? == &b'}' {
        return None;
    }
    loop {
        if bytes.get(i)? != &b'"' {
            return None;
        }
        let key_end = skip_string(bytes, i)?;
        // Compare raw bytes, so a key written with escapes simply does
        // not match and the caller falls back.
        let name = &bytes[i + 1..key_end - 1];
        let colon = skip_ws(bytes, key_end);
        if bytes.get(colon)? != &b':' {
            return None;
        }
        let value = skip_ws(bytes, colon + 1);
        let value_end = match keys.iter().position(|k| k.as_bytes() == name) {
            Some(slot) => {
                if found[slot].is_some() {
                    return None;
                }
                let (elements, end) = scan_object_array(bytes, value)?;
                found[slot] = Some(elements);
                end
            }
            None => skip_ignored_member(bytes, i, value)?,
        };
        i = skip_ws(bytes, value_end);
        match bytes.get(i)? {
            b',' => i = skip_ws(bytes, i + 1),
            b'}' => {
                if !only_whitespace_follows(bytes, i + 1) {
                    return None;
                }
                break;
            }
            _ => return None,
        }
    }
    found.into_iter().collect()
}

/// Deserialize every element of one chunk, in order, handing each to
/// `sink`.
///
/// Each element goes through `serde_json` exactly as on the serial
/// path. `StreamDeserializer::byte_offset` reports where the value
/// stopped, so the separating commas are stepped over without
/// re-scanning. Anything other than a comma between two values is left
/// in place for the next `next()` to choke on, turning a splitter bug
/// into a parse error — and so into the caller's fallback — rather
/// than a silently truncated chunk.
fn parse_chunk<T: DeserializeOwned>(
    slice: &[u8],
    mut sink: impl FnMut(T),
) -> Result<(), serde_json::Error> {
    let mut pos = 0usize;
    while pos < slice.len() {
        let mut stream = serde_json::Deserializer::from_slice(&slice[pos..]).into_iter::<T>();
        match stream.next() {
            Some(Ok(value)) => sink(value),
            Some(Err(err)) => return Err(err),
            None => break,
        }
        pos = skip_ws(slice, pos + stream.byte_offset());
        if pos < slice.len() && slice[pos] == b',' {
            pos = skip_ws(slice, pos + 1);
        }
    }
    Ok(())
}

/// Deserialize pre-scanned elements in parallel.
///
/// Elements are grouped into `n_chunks` byte-balanced runs; a run is
/// one contiguous slice, so the commas between its elements come along
/// for free. `Some(Err(..))` means a chunk failed and the caller should
/// fall back to the serial loader, which reports the error against the
/// whole document at the right offset.
///
/// Must be called with a rayon pool installed.
pub(crate) fn parse_elements_parallel<T>(
    bytes: &[u8],
    elements: &[Range<usize>],
    n_chunks: usize,
) -> Result<Vec<T>, serde_json::Error>
where
    T: DeserializeOwned + Send,
{
    let parts =
        parse_elements_parallel_into(bytes, elements, n_chunks, Vec::with_capacity, Vec::push)?;
    let mut out = Vec::with_capacity(parts.iter().map(Vec::len).sum());
    for part in parts {
        out.extend(part);
    }
    Ok(out)
}

/// [`parse_elements_parallel`] into caller-chosen per-chunk accumulators:
/// `init(count)` makes one for a chunk of `count` elements, `push`
/// appends an element to it, and the accumulators come back in document
/// order. Lets a caller build columns straight from the parse, with no
/// row-major intermediate.
pub(crate) fn parse_elements_parallel_into<T, A>(
    bytes: &[u8],
    elements: &[Range<usize>],
    n_chunks: usize,
    init: impl Fn(usize) -> A + Sync,
    push: impl Fn(&mut A, T) + Sync,
) -> Result<Vec<A>, serde_json::Error>
where
    T: DeserializeOwned,
    A: Send,
{
    if elements.is_empty() {
        return Ok(Vec::new());
    }
    // Group by bytes, not by element count: an annotation carrying a
    // polygon mask is an order of magnitude larger than a bare bbox, so
    // equal element counts would not be equal work.
    let total: usize = elements.iter().map(std::ops::Range::len).sum();
    let target = total.div_ceil(n_chunks.max(1)).max(1);
    let mut chunks: Vec<(usize, Range<usize>)> = Vec::with_capacity(n_chunks);
    let mut first = 0usize;
    let mut bytes_in_chunk = 0usize;
    for (idx, element) in elements.iter().enumerate() {
        bytes_in_chunk += element.len();
        let last = idx + 1 == elements.len();
        if bytes_in_chunk >= target || last {
            // The chunk spans from the *first* element's start, not the
            // previous chunk's end: starting at the end would put the
            // separating comma at byte 0 of the slice.
            chunks.push((idx + 1 - first, elements[first].start..element.end));
            first = idx + 1;
            bytes_in_chunk = 0;
        }
    }

    chunks
        .into_par_iter()
        .map(|(count, range)| {
            let mut acc = init(count);
            parse_chunk(&bytes[range], |value| push(&mut acc, value)).map(|()| acc)
        })
        .collect()
}

/// Deepest brace nesting at a cut that [`document_object_array_chunks`]
/// still resolves an element start for. A cut deeper than this is not
/// an error: that segment yields no boundary and its bytes join the
/// previous chunk. COCO results nest three deep.
const MAX_CUT_DEPTH: usize = 8;

/// One segment of a document, scanned under one assumption about the
/// string state at its first byte.
#[derive(Clone, Copy)]
struct SegmentScan {
    /// Whether the segment ends inside a string.
    end_in_string: bool,
    /// Net brace depth change across the segment.
    delta: i64,
    /// `first_open[d]`: the first `{` outside strings reached at brace
    /// depth `-d` relative to the segment start — the first element
    /// start for a segment entered at absolute depth `d`.
    first_open: [Option<usize>; MAX_CUT_DEPTH + 1],
}

/// Index just past the closing quote of a string whose body starts at
/// `pos`, within `..end`; `None` when the string runs past `end`.
///
/// [`skip_string`]'s two rules — an escape consumes the next byte, a
/// bare quote ends the string — with its byte-loop / `memchr2` split.
fn string_end(bytes: &[u8], mut pos: usize, end: usize) -> Option<usize> {
    let gate = (pos + STRING_SCALAR_GATE).min(end);
    while pos < gate {
        match bytes[pos] {
            b'\\' => pos += 2,
            b'"' => return Some(pos + 1),
            _ => pos += 1,
        }
    }
    loop {
        let rel = memchr::memchr2(b'\\', b'"', bytes.get(pos..end)?)?;
        let at = pos + rel;
        if bytes[at] == b'"' {
            return Some(at + 1);
        }
        pos = at + 2;
    }
}

/// Scan `range` for string state and brace depth, starting inside a
/// string when `in_string`.
///
/// Cuts never fall right after a backslash, so an escape and the byte it
/// consumes are in the same segment. Under the wrong assumption the
/// result is meaningless but harmless: [`document_object_array_chunks`]
/// only reads the scan of the assumption the previous segment proved.
fn scan_segment(bytes: &[u8], range: Range<usize>, mut in_string: bool) -> SegmentScan {
    let end = range.end;
    let mut pos = range.start;
    let mut depth = 0_i64;
    let mut first_open = [None; MAX_CUT_DEPTH + 1];
    while pos < end {
        if in_string {
            match string_end(bytes, pos, end) {
                Some(after) => {
                    in_string = false;
                    pos = after;
                }
                None => break,
            }
            continue;
        }
        let Some(rel) = memchr::memchr3(b'"', b'{', b'}', &bytes[pos..end]) else {
            break;
        };
        let at = pos + rel;
        match bytes[at] {
            b'"' => in_string = true,
            b'{' => {
                if let Ok(d) = usize::try_from(-depth) {
                    if let Some(slot) = first_open.get_mut(d) {
                        slot.get_or_insert(at);
                    }
                }
                depth += 1;
            }
            _ => depth -= 1,
        }
        pos = at + 1;
    }
    SegmentScan {
        end_in_string: in_string,
        delta: depth,
        first_open,
    }
}

/// Chunks of a whole-document array of objects, for parallel parsing:
/// byte ranges that each start at an element and run up to the next
/// chunk's first element (the last one up to the closing bracket).
///
/// [`document_object_array`] walks the document serially to find every
/// element, which at 186 MB of RLE results costs 83 ms — the serial floor
/// under the parallel parse. The parse only needs about one boundary per
/// chunk, and those can be found in parallel without guessing:
///
/// 1. Cut the array body into `n_chunks` segments, never right after a
///    backslash, so no escape spans a cut.
/// 2. Scan every segment twice in parallel — once assuming it starts
///    outside a string, once inside. (A string's state never
///    resynchronizes, so both are needed.)
/// 3. Compose sequentially: the document starts outside any string at
///    brace depth 0, and each segment's proven scan gives the next
///    segment's entry state, and its own first element start: the first
///    `{` outside strings at brace depth 0.
///
/// The ranges carry the separating commas; [`parse_chunks_parallel_into`]
/// checks them. `None` for anything not plainly an array of objects
/// followed by whitespace, as [`document_object_array`].
///
/// Must be called with a rayon pool installed.
pub(crate) fn document_object_array_chunks(
    bytes: &[u8],
    n_chunks: usize,
) -> Option<Vec<Range<usize>>> {
    let open = skip_ws(bytes, 0);
    if bytes.get(open)? != &b'[' {
        return None;
    }
    let close = bytes
        .iter()
        .rposition(|b| !matches!(b, b' ' | b'\t' | b'\r' | b'\n'))?;
    if close <= open || bytes[close] != b']' {
        return None;
    }
    let first = skip_ws(bytes, open + 1);
    if first == close {
        return Some(Vec::new());
    }
    if bytes[first] != b'{' {
        return None;
    }

    let body = close - first;
    let n = n_chunks.max(1);
    let mut cuts: Vec<usize> = Vec::with_capacity(n + 1);
    cuts.push(first);
    for c in 1..n {
        let mut cut = first + body * c / n;
        while cut < close && bytes[cut - 1] == b'\\' {
            cut += 1;
        }
        if cut > *cuts.last()? && cut < close {
            cuts.push(cut);
        }
    }
    cuts.push(close);

    let scans: Vec<[SegmentScan; 2]> = cuts
        .par_windows(2)
        .map(|w| {
            let (outside, inside) = rayon::join(
                || scan_segment(bytes, w[0]..w[1], false),
                || scan_segment(bytes, w[0]..w[1], true),
            );
            [outside, inside]
        })
        .collect();

    let mut starts: Vec<usize> = Vec::with_capacity(scans.len());
    let (mut in_string, mut depth) = (false, 0_i64);
    for scan in &scans {
        let scan = &scan[usize::from(in_string)];
        if let Some(start) = usize::try_from(depth)
            .ok()
            .and_then(|d| scan.first_open.get(d).copied().flatten())
        {
            starts.push(start);
        }
        in_string = scan.end_in_string;
        depth += scan.delta;
        if depth < 0 {
            return None;
        }
    }
    if in_string || depth != 0 || starts.first() != Some(&first) {
        return None;
    }
    let mut chunks: Vec<Range<usize>> = starts.windows(2).map(|w| w[0]..w[1]).collect();
    chunks.push(*starts.last()?..close);
    Some(chunks)
}

/// Deserialize a run of comma-separated elements, in order, handing each
/// to `sink`: one chunk of [`document_object_array_chunks`].
///
/// Stricter than [`parse_chunk`], because no serial scan has checked the
/// separators: exactly one comma between two elements, and the run ends
/// with one (`trailing_comma`) unless it is the array's last. Anything
/// else is an error, and so the caller's fallback to the serial loader.
fn parse_comma_run<T: DeserializeOwned>(
    slice: &[u8],
    trailing_comma: bool,
    mut sink: impl FnMut(T),
) -> Result<(), serde_json::Error> {
    let mut pos = 0usize;
    loop {
        let mut stream = serde_json::Deserializer::from_slice(&slice[pos..]).into_iter::<T>();
        match stream.next() {
            Some(Ok(value)) => sink(value),
            Some(Err(err)) => return Err(err),
            None => return Err(serde::de::Error::custom("expected an array element")),
        }
        pos = skip_ws(slice, pos + stream.byte_offset());
        if pos == slice.len() {
            return if trailing_comma {
                Err(serde::de::Error::custom(
                    "expected `,` after an array element",
                ))
            } else {
                Ok(())
            };
        }
        if slice[pos] != b',' {
            return Err(serde::de::Error::custom("expected `,` or `]`"));
        }
        pos = skip_ws(slice, pos + 1);
        if pos == slice.len() {
            return if trailing_comma {
                Ok(())
            } else {
                Err(serde::de::Error::custom("trailing comma"))
            };
        }
    }
}

/// Deserialize [`document_object_array_chunks`]'s chunks in parallel into
/// caller-chosen accumulators (`init` makes one, `push` appends an
/// element), returned in document order.
///
/// Must be called with a rayon pool installed.
pub(crate) fn parse_chunks_parallel_into<T, A>(
    bytes: &[u8],
    chunks: &[Range<usize>],
    init: impl Fn() -> A + Sync,
    push: impl Fn(&mut A, T) + Sync,
) -> Result<Vec<A>, serde_json::Error>
where
    T: DeserializeOwned,
    A: Send,
{
    let last = chunks.len().saturating_sub(1);
    chunks
        .par_iter()
        .enumerate()
        .map(|(c, range)| {
            let mut acc = init();
            parse_comma_run(&bytes[range.clone()], c != last, |value| {
                push(&mut acc, value)
            })
            .map(|()| acc)
        })
        .collect()
}

/// Chunk count for a thread budget. More chunks than threads, because
/// element sizes vary by an order of magnitude and rayon can only
/// rebalance at chunk granularity.
pub(crate) fn chunks_for(threads: usize) -> usize {
    threads.max(1).saturating_mul(4)
}

/// Smallest thread budget at which [`document_object_array_chunks`]'s
/// parallel scan beats [`document_object_array`]'s serial one.
///
/// The parallel scan does twice the work — every segment under both
/// string-state assumptions — so it needs threads to spare. Measured on
/// COCO val2017 results, parse + evaluate, parallel ÷ serial scan
/// (< 1 is a speedup): RF-DETR Nano bbox 1.06 / 0.94 / 0.89 and Seg-Nano
/// bbox 0.99 / 0.77 / 0.65 at 2 / 4 / 8 threads.
pub(crate) const PARALLEL_SCAN_MIN_THREADS: usize = 4;

/// Smallest thread budget at which splitting beats parsing serially.
///
/// The split is not free: it walks the whole document once before any
/// element is deserialized, and it builds a `Range` per element. One
/// thread has nothing to spend that on, so it pays the scan and keeps
/// the serial parse — and [`chunks_for`] would still cut four chunks
/// for it. `num_threads=1` is a setting callers actually pass, so the
/// loaders short-circuit below this value instead of taking the loss.
///
/// Measured best-of-5, 8 cores, release profile, parallel ÷ serial
/// (>1 is a speedup):
///
/// | payload                      | t1   | t2   | t3   | t4   | t8   |
/// |------------------------------|------|------|------|------|------|
/// | Objects365 bbox DT, 300k det | 0.87 | 1.09 | 1.27 | 1.32 | 1.47 |
/// | COCO segm DT (RLE)           | 0.74 | 1.17 | 1.19 | 1.42 | 1.24 |
/// | LVIS segm DT (RLE)           | 0.84 | 1.35 | 1.50 | 1.66 | 1.97 |
/// | COCO GT (`instances_val2017`)| 0.94 | 1.67 | 1.90 | 2.14 | 2.98 |
///
/// The crossover is between one thread and two on every payload, so
/// the threshold is 2 — not the "a few threads" one would guess.
pub(crate) const MIN_THREADS_TO_SPLIT: usize = 2;

#[cfg(test)]
mod tests {
    use super::*;
    use serde::Deserialize;

    #[derive(Debug, Deserialize, PartialEq)]
    struct Item {
        id: i64,
        #[serde(default)]
        name: String,
        #[serde(default)]
        score: f64,
    }

    fn parse_all(json: &str, keys: &[&str], chunks: usize) -> Option<Vec<Vec<Item>>> {
        let bytes = json.as_bytes();
        let arrays = top_level_object_arrays(bytes, keys)?;
        Some(
            arrays
                .iter()
                .map(|elements| {
                    parse_elements_parallel::<Item>(bytes, elements, chunks).expect("parse")
                })
                .collect(),
        )
    }

    /// The parallel array splitter cuts only at true element starts and
    /// parses to exactly what `serde_json::from_slice` produces, for every
    /// chunk count — over strings full of braces, commas, escaped quotes
    /// and backslash runs at every possible cut position.
    #[test]
    fn parallel_array_chunks_cut_at_elements_and_parse_like_serde() {
        let names = [
            r#"}, {\"id\": 99}"#,
            r#"a\\"#,
            r#"[,],{,}"#,
            r#"\\\\\"{\\"#,
            r#"\u007b\u0022"#,
            "",
            r#"x\\\\"#,
        ];
        let mut json = String::from(" \n[");
        for i in 0..240 {
            if i > 0 {
                json.push_str(if i % 3 == 0 { ",\n " } else { "," });
            }
            let name = names[i % names.len()];
            let nested = if i % 5 == 0 {
                r#", "extra": {"a": [{"b": "}"}], "c": {}}"#
            } else {
                ""
            };
            json.push_str(&format!(
                r#"{{"id":{i},"name":"{name}","score":0.{i}{nested}}}"#
            ));
        }
        json.push_str("]\n");
        let bytes = json.as_bytes();
        let reference: Vec<Item> =
            serde_json::from_str(&json).unwrap_or_else(|e| panic!("{e}: {}", &json[..400]));
        let starts: Vec<usize> = document_object_array(bytes)
            .expect("serial scan")
            .iter()
            .map(|r| r.start)
            .collect();
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .expect("pool");
        for n in 1..=40 {
            let chunks = pool
                .install(|| document_object_array_chunks(bytes, n))
                .expect("chunks");
            assert!(chunks.iter().all(|c| starts.contains(&c.start)), "n={n}");
            let parts: Vec<Vec<Item>> = pool
                .install(|| parse_chunks_parallel_into(bytes, &chunks, Vec::new, Vec::push))
                .expect("parse");
            assert_eq!(
                parts.into_iter().flatten().collect::<Vec<_>>(),
                reference,
                "n={n}"
            );
        }
    }

    /// Acceptance does not depend on the split: the chunked path yields a
    /// value exactly when `serde_json` accepts the document, and then the
    /// same value — malformed separators, non-object elements, trailing
    /// bytes, an unterminated string or a bracket inside a string at the
    /// end all fall back.
    #[test]
    fn parallel_array_chunks_accept_exactly_what_serde_accepts() {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(3)
            .build()
            .expect("pool");
        for doc in [
            r#"[]"#,
            r#"  [ ]  "#,
            r#"[{"id":1}]"#,
            r#"[{"id":1},{"id":2}]"#,
            r#"[{"id":1} {"id":2}]"#,
            r#"[{"id":1},,{"id":2}]"#,
            r#"[{"id":1},{"id":2},]"#,
            r#"[,{"id":1}]"#,
            r#"[{"id":1},2]"#,
            r#"[{"id":1}]]"#,
            r#"[{"id":1}][{"id":2}]"#,
            r#"[{"id":1,"name":"]"}"#,
            r#"[{"id":1,"name":"\"]"}]"#,
            r#"{"id":1}"#,
            r#"[{"id":1}"#,
        ] {
            let serial = serde_json::from_str::<Vec<Item>>(doc).ok();
            for n in 1..=8 {
                let parallel = pool.install(|| {
                    let chunks = document_object_array_chunks(doc.as_bytes(), n)?;
                    parse_chunks_parallel_into(doc.as_bytes(), &chunks, Vec::new, Vec::push)
                        .ok()
                        .map(|parts: Vec<Vec<Item>>| {
                            parts.into_iter().flatten().collect::<Vec<_>>()
                        })
                });
                assert!(
                    parallel.is_none() || parallel == serial,
                    "accepted what serde rejects or differs: {doc} n={n}"
                );
                if serial.is_some() && doc.contains("id") {
                    assert_eq!(parallel, serial, "{doc} n={n}");
                }
            }
        }
    }

    /// The whole point: every chunk count must produce the same values
    /// in the same order as one serial `from_slice`.
    #[test]
    fn every_chunk_count_agrees_with_serde_from_slice() {
        let json = r#"{"items":[
            {"id":1,"name":"a","score":0.5},
            {"id":2,"name":"b","score":0.25},
            {"id":3,"name":"c","score":0.125},
            {"id":4,"name":"d","score":1e-3},
            {"id":5,"name":"e","score":-0.0},
            {"id":6,"name":"f","score":12345.6789},
            {"id":7,"name":"g","score":0.9992794394493103}
        ]}"#;
        #[derive(Deserialize)]
        struct Doc {
            items: Vec<Item>,
        }
        let reference: Doc = serde_json::from_str(json).expect("serial");

        for chunks in 1..=16 {
            let got = parse_all(json, &["items"], chunks).expect("split");
            assert_eq!(got[0], reference.items, "chunks={chunks}");
        }
    }

    /// Element boundaries must survive every byte that could be
    /// mistaken for structure: braces, brackets and commas inside
    /// strings, and escaped quotes and backslashes before them.
    #[test]
    fn structural_bytes_inside_strings_do_not_split_elements() {
        let json = r#"{"items":[
            {"id":1,"name":"}, {\"id\": 99}"},
            {"id":2,"name":"a\\"},
            {"id":3,"name":"[,],{,}"},
            {"id":4,"name":"trailing backslash \\"},
            {"id":5,"name":"},"}
        ]}"#;
        #[derive(Deserialize)]
        struct Doc {
            items: Vec<Item>,
        }
        let reference: Doc = serde_json::from_str(json).expect("serial");
        for chunks in 1..=8 {
            let got = parse_all(json, &["items"], chunks).expect("split");
            assert_eq!(got[0], reference.items, "chunks={chunks}");
            assert_eq!(got[0].len(), 5);
        }
    }

    /// Nested containers inside an element, including an object nested
    /// in an array nested in an object — the RLE-in-segmentation shape.
    #[test]
    fn nested_containers_close_at_the_right_depth() {
        let json = r#"{"items":[
            {"id":1,"extra":{"counts":"abc}{","size":[10,20]}},
            {"id":2,"extra":[[1,2],[3,4],{"deep":{"deeper":[{}]}}]},
            {"id":3,"extra":[]}
        ]}"#;
        let got = parse_all(json, &["items"], 3).expect("split");
        assert_eq!(got[0].len(), 3);
        assert_eq!(got[0][2].id, 3);
    }

    /// Several arrays, one pass, in the caller's key order — not
    /// document order.
    #[test]
    fn multiple_keys_come_back_in_the_requested_order() {
        let json = r#"{"b":[{"id":2}],"ignored":{"x":[1,2,3]},"a":[{"id":1}],"c":[{"id":3}]}"#;
        let got = parse_all(json, &["a", "b", "c"], 2).expect("split");
        assert_eq!(got[0][0].id, 1);
        assert_eq!(got[1][0].id, 2);
        assert_eq!(got[2][0].id, 3);
    }

    /// Empty arrays are legal and produce no elements.
    #[test]
    fn empty_array_yields_no_elements() {
        let json = r#"{"items":[],"other":[{"id":1}]}"#;
        let got = parse_all(json, &["items"], 4).expect("split");
        assert!(got[0].is_empty());
    }

    /// Whitespace anywhere structural: the scanner must not assume the
    /// compact spelling `serde_json::to_string` produces.
    #[test]
    fn whitespace_between_every_token_is_tolerated() {
        let json = "{\n\t\"items\"  :  [\r\n  { \"id\" : 1 }  ,\n  { \"id\" : 2 }\n ]\n}";
        for chunks in 1..=4 {
            let got = parse_all(json, &["items"], chunks).expect("split");
            assert_eq!(got[0].len(), 2, "chunks={chunks}");
            assert_eq!(got[0][1].id, 2);
        }
    }

    /// A duplicate key is a `serde` error, so the splitter must refuse
    /// it rather than silently take the first — otherwise the parallel
    /// and serial loaders would disagree about a malformed document.
    #[test]
    fn duplicate_key_refuses_to_split() {
        let json = r#"{"items":[{"id":1}],"items":[{"id":2}]}"#;
        assert!(top_level_object_arrays(json.as_bytes(), &["items"]).is_none());
        // And `serde` does indeed reject it, which is the behaviour the
        // fallback then produces.
        #[derive(Deserialize)]
        struct Doc {
            #[allow(dead_code)]
            items: Vec<Item>,
        }
        assert!(serde_json::from_str::<Doc>(json).is_err());
    }

    /// Shapes the splitter does not handle hand back `None` so the
    /// caller falls back to the serial loader, which owns the error.
    #[test]
    fn unhandled_shapes_fall_back() {
        for json in [
            r#"[{"id":1}]"#,            // array, not an object
            r#"{"items":{"id":1}}"#,    // member is not an array
            r#"{"items":[1,2,3]}"#,     // elements are not objects
            r#"{"other":[{"id":1}]}"#,  // key missing
            r#"{"items":[{"id":1}"#,    // truncated
            r#"{}"#,                    // empty object
            r#"   "#,                   // no document
            r#"{"items":[{"id":1},]}"#, // trailing comma
        ] {
            assert!(
                top_level_object_arrays(json.as_bytes(), &["items"]).is_none(),
                "expected fallback for {json}"
            );
        }
    }

    /// Nothing may follow the document. Before this check the splitter
    /// stopped at the first closing bracket and silently dropped the
    /// rest, so `[{..}][{..}]` parsed as half a payload while
    /// `serde_json` called it `trailing characters`.
    #[test]
    fn trailing_bytes_after_the_document_fall_back() {
        for json in [
            r#"[{"id":1}][{"id":2}]"#,
            r#"[{"id":1}] xyz"#,
            r#"[{"id":1}]]"#,
            "[{\"id\":1}]\u{0}",
        ] {
            assert!(
                document_object_array(json.as_bytes()).is_none(),
                "expected fallback for {json}"
            );
        }
        // Whitespace, however, is not trailing content.
        assert!(document_object_array(b" [{\"id\":1}] \n\t\r").is_some());

        for json in [
            r#"{"items":[{"id":1}]}}}}garbage"#,
            r#"{"items":[{"id":1}]} ["#,
            r#"{"items":[{"id":1}]}{"items":[]}"#,
        ] {
            assert!(
                top_level_object_arrays(json.as_bytes(), &["items"]).is_none(),
                "expected fallback for {json}"
            );
        }
        assert!(top_level_object_arrays(b"{\"items\":[{\"id\":1}]}  \n", &["items"]).is_some());
    }

    /// A member the caller did not ask for is still part of the
    /// document, so it has to be as legal as the rest of it. The
    /// scalar walk it used to get accepted `tru`, `NaN` and `01+2x`,
    /// which `serde_json` rejects at a named byte offset.
    #[test]
    fn a_malformed_skipped_member_falls_back() {
        for tail in [
            "tru",
            "NaN",
            "01+2x",
            "-",
            "1.",
            "0x10",
            "+1",
            "'x'",
            "[1,]",
            "{\"a\":}",
            "\"a\\q\"",
            "01",
            "1e",
            "nul",
            "truefalse",
            "{\"a\":1,}",
        ] {
            let json = format!(r#"{{"info": {tail}, "items":[{{"id":1}}]}}"#);
            assert!(
                top_level_object_arrays(json.as_bytes(), &["items"]).is_none(),
                "expected fallback for skipped value {tail}"
            );
            assert!(
                serde_json::from_str::<serde_json::Value>(&json).is_err(),
                "fixture {tail} should be malformed"
            );
        }
        // A skipped key is validated too: a bad escape in one is an
        // error serially and must not be walked past here.
        let bad_key = r#"{"in\qfo": 1, "items":[{"id":1}]}"#;
        assert!(top_level_object_arrays(bad_key.as_bytes(), &["items"]).is_none());
        assert!(serde_json::from_str::<serde_json::Value>(bad_key).is_err());

        // ...and the key is decoded, not merely skipped. `IgnoredAny`
        // walks a string without validating its bytes, so it accepted a
        // raw non-UTF-8 byte here while the serial path — which has to
        // decode every key to match it against the known fields —
        // raised `invalid unicode code point`. Found by
        // `mutated_gt_loads_identically_on_both_paths`.
        #[derive(Deserialize)]
        struct Doc {
            #[allow(dead_code)]
            items: Vec<Item>,
        }
        let mut raw_key = br#"{"info": 1, "items":[{"id":1}]}"#.to_vec();
        raw_key[2] = 0x80;
        assert!(top_level_object_arrays(&raw_key, &["items"]).is_none());
        assert!(serde_json::from_slice::<Doc>(&raw_key).is_err());
        // An undecoded byte inside a skipped *value* is the mirror
        // case: the serial path skips that string with `IgnoredAny`
        // too, so both accept it and the splitter must not get
        // stricter than the loader it stands in for.
        let mut raw_value = br#"{"info": "xx", "items":[{"id":1}]}"#.to_vec();
        raw_value[11] = 0x80;
        assert!(top_level_object_arrays(&raw_value, &["items"]).is_some());
        assert!(serde_json::from_slice::<Doc>(&raw_value).is_ok());
    }

    /// The legal spellings a skipped member can take must all still be
    /// stepped over, or every COCO file with an `info` block would
    /// take the slow path.
    #[test]
    fn well_formed_skipped_members_are_stepped_over() {
        for tail in [
            "true",
            "false",
            "null",
            "0",
            "-0.0",
            "1e-3",
            "12345.6789",
            "\"a string with } and , in it\"",
            "[1, 2, [3, {\"x\": null}]]",
            "{\"nested\": {\"deep\": [1]}}",
            "{}",
            "[]",
            "\"\\u00e9\\\"\"",
        ] {
            let json = format!(r#"{{"info": {tail} , "items":[{{"id":1}}]}}"#);
            let arrays = top_level_object_arrays(json.as_bytes(), &["items"])
                .unwrap_or_else(|| panic!("expected a split for skipped value {tail}"));
            assert_eq!(arrays[0].len(), 1, "skipped value {tail}");
        }
    }

    /// The gate in `skip_string` changes which loop finds the closing
    /// quote; both sides of it must agree, including when an escape
    /// straddles the boundary.
    #[test]
    fn the_string_scan_gate_is_invisible() {
        for len in 0..(STRING_SCALAR_GATE * 3) {
            for prefix in [0usize, 1] {
                // `prefix` shifts an escaped quote across the gate.
                let mut s = String::from("\"");
                s.push_str(&"x".repeat(prefix));
                s.push_str(&"\\\"".repeat(len));
                s.push('"');
                let bytes = s.as_bytes();
                assert_eq!(
                    skip_string(bytes, 0),
                    Some(bytes.len()),
                    "len={len} prefix={prefix}"
                );
                // Unterminated: dropping the closing quote must yield
                // `None`, not a runaway index.
                assert!(skip_string(&bytes[..bytes.len() - 1], 0).is_none());
            }
        }
        // A trailing backslash consumes the byte that would have been
        // the terminator, on either side of the gate.
        assert!(skip_string(b"\"abc\\\"", 0).is_none());
        let long = format!("\"{}\\\"", "y".repeat(STRING_SCALAR_GATE * 2));
        assert!(skip_string(long.as_bytes(), 0).is_none());
    }

    /// A whole-document array is the detection payload's shape.
    #[test]
    fn document_array_splits_into_elements() {
        let json = r#" [ {"id":1}, {"id":2}, {"id":3} ] "#;
        let elements = document_object_array(json.as_bytes()).expect("split");
        assert_eq!(elements.len(), 3);
        let parsed = parse_elements_parallel::<Item>(json.as_bytes(), &elements, 2).expect("parse");
        assert_eq!(parsed.iter().map(|i| i.id).collect::<Vec<_>>(), [1, 2, 3]);
        assert!(document_object_array(br#"{"a":1}"#).is_none());
        assert_eq!(document_object_array(b"[]").expect("empty").len(), 0);
    }

    /// A malformed *element* is a parse error, not a silent drop: the
    /// caller turns it into a serial re-parse that reports the real
    /// offset.
    #[test]
    fn malformed_element_surfaces_as_an_error() {
        let json = r#"{"items":[{"id":1},{"id":"not a number"},{"id":3}]}"#;
        let bytes = json.as_bytes();
        let arrays = top_level_object_arrays(bytes, &["items"]).expect("split");
        for chunks in 1..=4 {
            assert!(parse_elements_parallel::<Item>(bytes, &arrays[0], chunks).is_err());
        }
    }

    /// Chunk grouping is by bytes, so one huge element does not drag a
    /// chunk's worth of small ones along with it — and, more
    /// importantly, every element still lands exactly once.
    #[test]
    fn byte_balanced_grouping_keeps_every_element_exactly_once() {
        let big = "x".repeat(4096);
        let mut items = String::from(r#"{"items":["#);
        for i in 0..64 {
            if i > 0 {
                items.push(',');
            }
            if i % 8 == 0 {
                items.push_str(&format!(r#"{{"id":{i},"name":"{big}"}}"#));
            } else {
                items.push_str(&format!(r#"{{"id":{i}}}"#));
            }
        }
        items.push_str("]}");

        for chunks in [1usize, 2, 3, 8, 64, 1000] {
            let got = parse_all(&items, &["items"], chunks).expect("split");
            assert_eq!(got[0].len(), 64, "chunks={chunks}");
            assert_eq!(
                got[0].iter().map(|i| i.id).collect::<Vec<_>>(),
                (0..64).collect::<Vec<i64>>(),
                "chunks={chunks}"
            );
        }
    }
}
