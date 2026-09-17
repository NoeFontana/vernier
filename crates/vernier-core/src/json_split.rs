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

use std::ops::Range;

use rayon::prelude::*;
use serde::de::DeserializeOwned;

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

/// Index just past the string starting at `i` (its opening quote).
///
/// A plain byte loop, deliberately. `memchr` was measured here and is
/// slower: JSON field names put the next quote ~10 bytes away, and at
/// that distance the per-call SIMD setup costs more than the bytes it
/// skips.
fn skip_string(bytes: &[u8], i: usize) -> Option<usize> {
    let mut j = i + 1;
    while j < bytes.len() {
        match bytes[j] {
            // An escape consumes the next byte whatever it is, which is
            // what keeps `\"` from ending the string. `\uXXXX` needs no
            // special case: the four hex digits are ordinary bytes.
            b'\\' => j += 2,
            b'"' => return Some(j + 1),
            _ => j += 1,
        }
    }
    None
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

/// Index just past the JSON value starting at `i`.
fn skip_value(bytes: &[u8], i: usize) -> Option<usize> {
    match *bytes.get(i)? {
        b'"' => skip_string(bytes, i),
        b'{' | b'[' => skip_container(bytes, i),
        // Numbers, `true`, `false`, `null`: run to the next structural
        // byte. Their exact spelling is `serde_json`'s business — we
        // only need to know where the value stops.
        _ => {
            let mut j = i;
            while j < bytes.len()
                && !matches!(bytes[j], b',' | b'}' | b']' | b' ' | b'\t' | b'\r' | b'\n')
            {
                j += 1;
            }
            (j > i).then_some(j)
        }
    }
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
pub(crate) fn document_object_array(bytes: &[u8]) -> Option<Vec<Range<usize>>> {
    let start = skip_ws(bytes, 0);
    scan_object_array(bytes, start).map(|(elements, _)| elements)
}

/// Element ranges of the named array members of a top-level object, in
/// the order `keys` lists them.
///
/// One pass over the document: skipping a member's value means walking
/// it, and `annotations` is most of the file.
///
/// `None` when the document is not a plain object, a requested key is
/// missing, a requested value is not an array of objects, or a key
/// appears more than once.
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
            None => skip_value(bytes, value)?,
        };
        i = skip_ws(bytes, value_end);
        match bytes.get(i)? {
            b',' => i = skip_ws(bytes, i + 1),
            b'}' => break,
            _ => return None,
        }
    }
    found.into_iter().collect()
}

/// Deserialize every element of one chunk, in order, appending to
/// `out`.
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
    out: &mut Vec<T>,
) -> Result<(), serde_json::Error> {
    let mut pos = 0usize;
    while pos < slice.len() {
        let mut stream = serde_json::Deserializer::from_slice(&slice[pos..]).into_iter::<T>();
        match stream.next() {
            Some(Ok(value)) => out.push(value),
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

    let parsed: Result<Vec<Vec<T>>, serde_json::Error> = chunks
        .into_par_iter()
        .map(|(count, range)| {
            let mut out = Vec::with_capacity(count);
            parse_chunk(&bytes[range], &mut out).map(|()| out)
        })
        .collect();
    parsed.map(|parts| {
        let mut out = Vec::with_capacity(parts.iter().map(Vec::len).sum());
        for part in parts {
            out.extend(part);
        }
        out
    })
}

/// Chunk count for a thread budget. More chunks than threads, because
/// element sizes vary by an order of magnitude and rayon can only
/// rebalance at chunk granularity.
pub(crate) fn chunks_for(threads: usize) -> usize {
    threads.max(1).saturating_mul(4)
}

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
