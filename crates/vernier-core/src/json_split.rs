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
