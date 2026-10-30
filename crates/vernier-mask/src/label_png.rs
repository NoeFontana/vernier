//! A fast decoder for label-map PNGs: 8-bit grayscale (semantic class ids)
//! and 8-bit RGB (panoptic segment ids), non-interlaced.
//!
//! Decoding is most of a panoptic or semantic evaluation, and unfiltering is
//! most of decoding: on COCO panoptic val2017, 0.40 of the 0.50 ms a PNG
//! takes, where inflating takes 0.10. A general decoder has to treat every
//! Paeth-filtered row as a serial per-byte dependency chain. A label map is
//! piecewise constant, and that collapses: once a pixel equals the one above
//! it, a zero Paeth residual reproduces the pixel above — which keeps the
//! next pixel in the same state — so a zero-residual span is a copy of the
//! row above. A zero Sub residual likewise repeats the last pixel. Both spans
//! are found sixteen bytes at a time and written as block copies.
//!
//! Only the unfilter is vernier's. The container — signature, chunk CRCs,
//! IHDR, IDAT contiguity, inflate — is the `png` crate's own
//! [`png::StreamingDecoder`]; iCCP and text chunks, which nothing here
//! reads, are skipped unparsed. [`decode_label_png`] returns `None` for any
//! other layout (color type, bit depth, interlacing), for a file the `png`
//! crate rejects, and for image data whose length is not what the header
//! implies; the caller then decodes with the `png` crate's [`png::Decoder`],
//! which also owns every error message. On the files it accepts, the output
//! is the PNG's pixels: there is no approximation to disagree with.
//!
//! This is the panoptic and semantic crates' implementation detail, not a
//! stable API.

use png::{Decoded, StreamingDecoder, UnfilterRegion};

/// Pixel layout of a label-map PNG this decoder accepts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LabelColor {
    /// 8-bit grayscale: one byte per pixel.
    Gray8,
    /// 8-bit RGB: three bytes per pixel.
    Rgb8,
}

impl LabelColor {
    /// Bytes per pixel.
    #[must_use]
    pub const fn bytes_per_pixel(self) -> usize {
        match self {
            Self::Gray8 => 1,
            Self::Rgb8 => 3,
        }
    }

    fn png_color_type(self) -> png::ColorType {
        match self {
            Self::Gray8 => png::ColorType::Grayscale,
            Self::Rgb8 => png::ColorType::Rgb,
        }
    }
}

/// A validated label map, inflated into the caller's buffer and unfiltered
/// as it is read: by [`Self::into_rows`] or [`Self::into_pixels`].
pub struct LabelPng<'a> {
    width: u32,
    height: u32,
    color: LabelColor,
    stride: usize,
    /// `height` filtered rows of `1 + stride` bytes, each led by its
    /// (already checked) filter byte.
    data: &'a mut Vec<u8>,
}

impl<'a> LabelPng<'a> {
    /// Image width in pixels.
    #[must_use]
    pub fn width(&self) -> u32 {
        self.width
    }

    /// Image height in pixels.
    #[must_use]
    pub fn height(&self) -> u32 {
        self.height
    }

    /// The rows' pixel bytes, top to bottom, each unfiltered in place as it
    /// is reached — while the row above is still in cache.
    #[must_use]
    pub fn into_rows(self) -> LabelRows<'a> {
        LabelRows {
            rest: self.data.as_mut_slice(),
            prev: &[],
            stride: self.stride,
            color: self.color,
        }
    }

    /// Unfilters every row and packs it at the front of the buffer, which
    /// is then truncated to `height * width * bytes_per_pixel` pixel bytes,
    /// row-major. Each row is moved right after it is unfiltered; it lands
    /// before the next row's bytes, so nothing unread is overwritten.
    pub fn into_pixels(self) {
        let s = self.stride;
        for r in 0..self.height as usize {
            let (done, rest) = self.data.split_at_mut(r * (s + 1));
            // Row `r - 1`, already unfiltered and packed at `(r - 1) * s`.
            let prev = if r == 0 {
                &[][..]
            } else {
                &done[(r - 1) * s..r * s]
            };
            let (kind, cur) = rest[..=s].split_at_mut(1);
            unfilter_row(self.color, kind[0], prev, cur);
            let from = r * (s + 1) + 1;
            self.data.copy_within(from..from + s, r * s);
        }
        self.data.truncate(self.height as usize * s);
    }
}

/// Rows of a decoded label map; see [`LabelPng::into_rows`].
pub struct LabelRows<'a> {
    rest: &'a mut [u8],
    /// The last row handed out, unfiltered; empty before the first.
    prev: &'a [u8],
    stride: usize,
    color: LabelColor,
}

impl<'a> Iterator for LabelRows<'a> {
    type Item = &'a [u8];

    fn next(&mut self) -> Option<&'a [u8]> {
        if self.rest.is_empty() {
            return None;
        }
        let (row, tail) = std::mem::take(&mut self.rest).split_at_mut(self.stride + 1);
        self.rest = tail;
        let (kind, cur) = row.split_at_mut(1);
        unfilter_row(self.color, kind[0], self.prev, cur);
        let cur: &'a [u8] = cur;
        self.prev = cur;
        Some(cur)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let n = self.rest.len() / (self.stride + 1);
        (n, Some(n))
    }
}

impl ExactSizeIterator for LabelRows<'_> {}

/// Decodes a label-map PNG of layout `color` into `buf`, which is resized
/// as needed and can be reused across images.
///
/// Returns `None` — `buf` then holds no meaningful bytes — when the file is
/// not `color`, not an 8-bit non-interlaced PNG, not one the `png` crate
/// accepts, carries image data of the wrong length or an invalid filter
/// type, or would not fit in memory. Decode it with the `png` crate then.
pub fn decode_label_png<'a>(
    bytes: &[u8],
    color: LabelColor,
    buf: &'a mut Vec<u8>,
) -> Option<LabelPng<'a>> {
    let (width, height, stride) = inflate(bytes, color, buf)?;
    // Every filter type is checked here, so reading rows cannot fail.
    if buf.chunks_exact(stride + 1).any(|row| row[0] > 4) {
        return None;
    }
    Some(LabelPng {
        width,
        height,
        color,
        stride,
        data: buf,
    })
}

/// Runs the `png` crate's streaming decoder over `bytes`, inflating the
/// image data into `buf`, sized exactly `height * (1 + stride)`. Returns
/// `(width, height, stride)`.
fn inflate(bytes: &[u8], color: LabelColor, buf: &mut Vec<u8>) -> Option<(u32, u32, usize)> {
    let mut decoder = StreamingDecoder::new();
    // Nothing here reads them, and they are where a small file can hide a
    // large allocation (a compressed ICC profile or text chunk).
    decoder.set_ignore_iccp_chunk(true);
    decoder.set_ignore_text_chunk(true);
    let mut region = UnfilterRegion::default();
    let mut rest = bytes;
    // (width, height, stride, expected data length), once the header is in.
    let mut header: Option<(u32, u32, usize, usize)> = None;
    let mut sized = false;
    let mut stalled = false;
    loop {
        if header.is_none() {
            if let Some(info) = decoder.info() {
                if info.color_type != color.png_color_type()
                    || info.bit_depth != png::BitDepth::Eight
                    || info.interlaced
                {
                    return None;
                }
                let stride = usize::try_from(info.width)
                    .ok()?
                    .checked_mul(color.bytes_per_pixel())?;
                let expected = usize::try_from(info.height)
                    .ok()?
                    .checked_mul(stride.checked_add(1)?)?;
                header = Some((info.width, info.height, stride, expected));
            }
        }
        if rest.is_empty() {
            return None; // ended before IEND
        }
        let before = (rest.len(), region.filled);
        let (consumed, event) = match header {
            Some(_) if sized => decoder.update(rest, Some(&mut region.as_buf(buf))),
            _ => decoder.update(rest, None),
        }
        .ok()?;
        rest = &rest[consumed..];
        match event {
            // Sized only once image data begins: a header alone allocates
            // nothing. One byte of slack makes a stream longer than the
            // header implies overrun, and be refused, rather than be cut
            // off; a failed allocation falls back rather than aborts.
            Decoded::ChunkBegin(_, png::chunk::IDAT) if !sized => {
                let (_, _, _, expected) = header?;
                let len = expected.checked_add(1)?;
                buf.clear();
                buf.try_reserve_exact(len).ok()?;
                buf.resize(len, 0);
                sized = true;
            }
            Decoded::ChunkComplete(png::chunk::IEND) => break,
            _ => {}
        }
        // No input consumed and no output produced, twice running: the
        // stream cannot advance (in practice, more data than the header
        // implies has filled the buffer).
        if before == (rest.len(), region.filled) {
            if stalled {
                return None;
            }
            stalled = true;
        } else {
            stalled = false;
        }
    }
    let (width, height, stride, expected) = header?;
    if !sized || region.filled != expected {
        return None;
    }
    buf.truncate(expected);
    Some((width, height, stride))
}

/// Unfilters one row in place; `prev` is the unfiltered row above, empty
/// for the first row.
fn unfilter_row(color: LabelColor, kind: u8, prev: &[u8], cur: &mut [u8]) {
    match (color, prev.is_empty()) {
        (LabelColor::Gray8, true) => unfilter_first::<1>(kind, cur),
        (LabelColor::Gray8, false) => unfilter::<1>(kind, prev, cur),
        (LabelColor::Rgb8, true) => unfilter_first::<3>(kind, cur),
        (LabelColor::Rgb8, false) => unfilter::<3>(kind, prev, cur),
    }
}

/// How many leading bytes of `s` are zero, sixteen at a time.
#[inline]
fn zero_run(s: &[u8]) -> usize {
    let mut n = 0;
    for chunk in s.chunks_exact(16) {
        if chunk.iter().fold(0u8, |acc, &x| acc | x) != 0 {
            break;
        }
        n += 16;
    }
    n + s[n..].iter().take_while(|&&x| x == 0).count()
}

/// Whole pixels of zero residual starting at byte `i` (a multiple of
/// `BPP`), in bytes.
#[inline]
fn zero_pixels<const BPP: usize>(cur: &[u8], i: usize) -> usize {
    zero_run(&cur[i..]) / BPP * BPP
}

/// The Paeth predictor (PNG spec §9.4) in stb_image's branch-free form
/// (public domain; the form the `png` crate itself decodes with): the
/// spec's three distances reduce to comparisons against `3c - (a + b)`.
#[inline]
fn paeth(a: i16, b: i16, c: i16) -> u8 {
    let thresh = c * 3 - (a + b);
    let lo = a.min(b);
    let hi = a.max(b);
    let t0 = if hi <= thresh { lo } else { c };
    let t1 = if thresh <= lo { hi } else { t0 };
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)] // one of a, b, c: 0..=255
    let out = t1 as u8;
    out
}

/// Writes `pixel` over `cur[i..i + len]` (`len` a multiple of `BPP`): a
/// fill for one byte per pixel, doubling block copies otherwise.
#[inline]
fn repeat_pixel<const BPP: usize>(cur: &mut [u8], i: usize, len: usize, pixel: [u8; BPP]) {
    if len == 0 {
        return;
    }
    cur[i..i + BPP].copy_from_slice(&pixel);
    let mut done = BPP;
    while done < len {
        let n = done.min(len - done);
        cur.copy_within(i..i + n, i + done);
        done += n;
    }
}

/// Sub: `x += left`. The left pixel stays in registers, so no store feeds
/// the next load, and a zero-residual span repeats it as block copies.
fn unfilter_sub<const BPP: usize>(cur: &mut [u8]) {
    let n = cur.len();
    let Some(first) = cur.get(..BPP) else { return };
    let mut a: [u8; BPP] = first.try_into().unwrap_or([0; BPP]);
    let mut i = BPP;
    while i < n {
        if cur[i] == 0 {
            let run = zero_pixels::<BPP>(cur, i);
            if run > 0 {
                repeat_pixel(cur, i, run, a);
                i += run;
                continue;
            }
        }
        for k in 0..BPP {
            a[k] = cur[i + k].wrapping_add(a[k]);
        }
        cur[i..i + BPP].copy_from_slice(&a);
        i += BPP;
    }
}

/// Average: `x += floor((left + above) / 2)`; `prev` is `None` for the
/// first row, whose row above is all zero (spec §9.2).
fn unfilter_avg<const BPP: usize>(prev: Option<&[u8]>, cur: &mut [u8]) {
    let mut a = [0u16; BPP];
    for (p, px) in cur.chunks_exact_mut(BPP).enumerate() {
        for k in 0..BPP {
            let b = prev.map_or(0, |prev| u16::from(prev[p * BPP + k]));
            #[allow(clippy::cast_possible_truncation)] // ≤ 255
            let x = px[k].wrapping_add(((a[k] + b) / 2) as u8);
            px[k] = x;
            a[k] = u16::from(x);
        }
    }
}

/// Paeth, with the left (`a`) and upper-left (`c`) pixels in registers.
///
/// When a pixel's left equals its upper-left, the predictor is the pixel
/// above, so a zero residual reproduces it — and the next pixel then sees
/// left == upper-left again. A zero-residual span starting in that state is
/// therefore a copy of the row above. On a label map that is most of the
/// row; elsewhere the check almost never fires and costs one compare.
fn unfilter_paeth<const BPP: usize>(prev: &[u8], cur: &mut [u8]) {
    let n = cur.len();
    let prev = &prev[..n];
    // Pixel 0 has no left or upper-left: zero, which predicts the pixel
    // above, as the spec requires.
    let mut a = [0i16; BPP];
    let mut c = [0i16; BPP];
    let mut i = 0;
    while i < n {
        if cur[i] == 0 && a == c {
            let run = zero_pixels::<BPP>(cur, i);
            if run > 0 {
                cur[i..i + run].copy_from_slice(&prev[i..i + run]);
                i += run;
                for k in 0..BPP {
                    a[k] = i16::from(prev[i - BPP + k]);
                }
                c = a;
                continue;
            }
        }
        for k in 0..BPP {
            let b = i16::from(prev[i + k]);
            let x = cur[i + k].wrapping_add(paeth(a[k], b, c[k]));
            cur[i + k] = x;
            a[k] = i16::from(x);
            c[k] = b;
        }
        i += BPP;
    }
}

/// The first row: the row above is all zero (spec §9.2), so Up is the
/// identity and Paeth predicts the left pixel, as Sub does.
fn unfilter_first<const BPP: usize>(kind: u8, cur: &mut [u8]) {
    match kind {
        1 | 4 => unfilter_sub::<BPP>(cur),
        3 => unfilter_avg::<BPP>(None, cur),
        _ => {}
    }
}

/// Reverses one row's filter in place; `prev` is the unfiltered row above,
/// as long as `cur`. `kind` is at most 4.
fn unfilter<const BPP: usize>(kind: u8, prev: &[u8], cur: &mut [u8]) {
    match kind {
        1 => unfilter_sub::<BPP>(cur),
        // Up: `x += above`.
        2 => {
            for (x, &b) in cur.iter_mut().zip(prev) {
                *x = x.wrapping_add(b);
            }
        }
        3 => unfilter_avg::<BPP>(Some(prev), cur),
        4 => unfilter_paeth::<BPP>(prev, cur),
        // None.
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    /// Encodes `pixels` with the `png` crate, every row with `filter`
    /// (`None` for its adaptive choice).
    fn encode(
        w: u32,
        h: u32,
        color: LabelColor,
        pixels: &[u8],
        filter: Option<png::Filter>,
    ) -> Vec<u8> {
        let mut out = Vec::new();
        {
            let mut enc = png::Encoder::new(&mut out, w, h);
            enc.set_color(match color {
                LabelColor::Gray8 => png::ColorType::Grayscale,
                LabelColor::Rgb8 => png::ColorType::Rgb,
            });
            enc.set_depth(png::BitDepth::Eight);
            enc.set_filter(filter.unwrap_or(png::Filter::Adaptive));
            let mut writer = enc.write_header().unwrap();
            writer.write_image_data(pixels).unwrap();
        }
        out
    }

    fn reference(bytes: &[u8]) -> Option<Vec<u8>> {
        let mut reader = png::Decoder::new(std::io::Cursor::new(bytes))
            .read_info()
            .ok()?;
        let mut buf = vec![0; reader.output_buffer_size()?];
        reader.next_frame(&mut buf).ok()?;
        Some(buf)
    }

    /// Decodes through both views — the row iterator and the packed
    /// pixels — and checks they agree.
    fn fast_as(bytes: &[u8], color: LabelColor) -> Option<((u32, u32), Vec<u8>)> {
        let mut buf = Vec::new();
        let png = decode_label_png(bytes, color, &mut buf)?;
        let dims = (png.width(), png.height());
        let rows: Vec<u8> = png.into_rows().flatten().copied().collect();
        decode_label_png(bytes, color, &mut buf)?.into_pixels();
        assert_eq!(rows, buf, "row view and packed pixels disagree");
        Some((dims, buf))
    }

    fn fast(bytes: &[u8]) -> Option<((u32, u32), Vec<u8>)> {
        fast_as(bytes, LabelColor::Gray8).or_else(|| fast_as(bytes, LabelColor::Rgb8))
    }

    /// A piecewise-constant label map with a few rectangles, so every
    /// filter meets both flat spans and edges.
    fn label_map(
        w: usize,
        h: usize,
        bpp: usize,
        rects: &[(usize, usize, usize, usize, u8)],
    ) -> Vec<u8> {
        let mut px = vec![0u8; w * h * bpp];
        for &(x0, y0, rw, rh, v) in rects {
            for y in y0..(y0 + rh).min(h) {
                for x in x0..(x0 + rw).min(w) {
                    for c in 0..bpp {
                        px[(y * w + x) * bpp + c] = v.wrapping_add(c as u8 * 37);
                    }
                }
            }
        }
        px
    }

    const FILTERS: [Option<png::Filter>; 6] = [
        None,
        Some(png::Filter::NoFilter),
        Some(png::Filter::Sub),
        Some(png::Filter::Up),
        Some(png::Filter::Avg),
        Some(png::Filter::Paeth),
    ];

    proptest! {
        #[test]
        fn matches_the_png_crate(
            w in 1usize..70,
            h in 1usize..40,
            rgb in any::<bool>(),
            rects in prop::collection::vec((0usize..70, 0usize..40, 1usize..40, 1usize..30, any::<u8>()), 0..6),
            noise in prop::collection::vec((any::<prop::sample::Index>(), any::<u8>()), 0..20),
            filter in 0usize..6,
        ) {
            let color = if rgb { LabelColor::Rgb8 } else { LabelColor::Gray8 };
            let bpp = color.bytes_per_pixel();
            let mut px = label_map(w, h, bpp, &rects);
            for (at, v) in noise {
                let i = at.index(px.len());
                px[i] = v;
            }
            let bytes = encode(w as u32, h as u32, color, &px, FILTERS[filter]);
            let (dims, got) = fast(&bytes).expect("a plain label-map PNG takes the fast path");
            prop_assert_eq!(dims, (w as u32, h as u32));
            prop_assert_eq!(&got, &px);
            prop_assert_eq!(Some(got), reference(&bytes));
        }

        /// Corrupted files: refused, or decoded exactly as the `png` crate
        /// decodes them.
        #[test]
        fn corruption_is_refused_or_agrees(
            w in 1usize..30,
            h in 1usize..20,
            rects in prop::collection::vec((0usize..30, 0usize..20, 1usize..20, 1usize..20, any::<u8>()), 0..4),
            edits in prop::collection::vec((any::<prop::sample::Index>(), any::<u8>()), 1..3),
        ) {
            let px = label_map(w, h, 3, &rects);
            let mut bytes = encode(w as u32, h as u32, LabelColor::Rgb8, &px, None);
            for (at, v) in edits {
                let i = at.index(bytes.len());
                bytes[i] = v;
            }
            if let Some((_, got)) = fast(&bytes) {
                prop_assert_eq!(Some(got), reference(&bytes));
            }
        }
    }

    #[test]
    fn refuses_what_it_does_not_handle() {
        let px = label_map(8, 4, 1, &[(1, 1, 3, 2, 9)]);
        let gray = encode(8, 4, LabelColor::Gray8, &px, None);
        assert!(fast_as(&gray, LabelColor::Gray8).is_some());
        assert!(fast_as(&gray, LabelColor::Rgb8).is_none(), "wrong layout");
        // Interlaced, 16-bit, palette: the general decoder's.
        let mut out = Vec::new();
        {
            let mut enc = png::Encoder::new(&mut out, 8, 4);
            enc.set_color(png::ColorType::Grayscale);
            enc.set_depth(png::BitDepth::Sixteen);
            let mut writer = enc.write_header().unwrap();
            writer.write_image_data(&[0u8; 64]).unwrap();
        }
        assert!(fast(&out).is_none());
        assert!(fast(&gray[..gray.len() - 5]).is_none(), "truncated");
        assert!(fast(b"not a png").is_none());
        assert!(fast(&[]).is_none());
    }

    /// A PNG built from raw filtered rows, bypassing any encoder.
    fn png_from_raw(w: u32, h: u32, color_type: u8, raw: &[u8]) -> Vec<u8> {
        fn chunk(out: &mut Vec<u8>, kind: &[u8; 4], data: &[u8]) {
            out.extend_from_slice(&u32::try_from(data.len()).unwrap().to_be_bytes());
            let mut body = kind.to_vec();
            body.extend_from_slice(data);
            out.extend_from_slice(&body);
            out.extend_from_slice(&crc32fast::hash(&body).to_be_bytes());
        }
        let mut out = vec![0x89, b'P', b'N', b'G', b'\r', b'\n', 0x1a, b'\n'];
        let mut ihdr = w.to_be_bytes().to_vec();
        ihdr.extend_from_slice(&h.to_be_bytes());
        ihdr.extend_from_slice(&[8, color_type, 0, 0, 0]);
        chunk(&mut out, b"IHDR", &ihdr);
        chunk(&mut out, b"IDAT", &fdeflate::compress_to_vec(raw));
        chunk(&mut out, b"IEND", &[]);
        out
    }

    #[test]
    fn hand_built_rows_decode() {
        // 4x2 gray: row 0 Sub, row 1 Average.
        let raw = [1, 10, 1, 1, 1, 3, 0, 4, 0, 0];
        let (_, px) = fast(&png_from_raw(4, 2, 0, &raw)).unwrap();
        assert_eq!(px, reference(&png_from_raw(4, 2, 0, &raw)).unwrap());
        assert_eq!(px[..4], [10, 11, 12, 13]);
    }

    #[test]
    fn a_bad_filter_type_is_refused_up_front() {
        // In the last row: refused before any row is handed out.
        let raw = [0, 1, 2, 3, 0, 4, 5, 6, 9, 7, 8, 9];
        let bytes = png_from_raw(3, 3, 0, &raw);
        assert!(decode_label_png(&bytes, LabelColor::Gray8, &mut Vec::new()).is_none());
    }

    #[test]
    fn a_header_alone_allocates_nothing() {
        // IHDR declaring 100000 × 100000 RGB, then IEND: no image data.
        let mut bytes = png_from_raw(100_000, 100_000, 2, &[]);
        let idat_start = 8 + 25; // after the signature and IHDR chunk
        let idat_len =
            12 + u32::from_be_bytes(bytes[idat_start..idat_start + 4].try_into().unwrap()) as usize;
        bytes.drain(idat_start..idat_start + idat_len);
        let mut buf = Vec::new();
        assert!(decode_label_png(&bytes, LabelColor::Rgb8, &mut buf).is_none());
        assert_eq!(buf.capacity(), 0);
    }

    proptest! {
        /// Noise, not a label map: every byte takes the scalar paths.
        #[test]
        fn random_pixels_match_the_png_crate(
            w in 1usize..40,
            h in 1usize..20,
            rgb in any::<bool>(),
            seed in any::<u64>(),
            filter in 0usize..6,
        ) {
            let color = if rgb { LabelColor::Rgb8 } else { LabelColor::Gray8 };
            let mut state = seed | 1;
            let px: Vec<u8> = (0..w * h * color.bytes_per_pixel())
                .map(|_| {
                    state ^= state << 13;
                    state ^= state >> 7;
                    state ^= state << 17;
                    state.to_le_bytes()[0]
                })
                .collect();
            let bytes = encode(w as u32, h as u32, color, &px, FILTERS[filter]);
            let (_, got) = fast_as(&bytes, color).expect("takes the fast path");
            prop_assert_eq!(&got, &px);
        }
    }

    #[test]
    fn image_data_must_match_the_header() {
        let short = png_from_raw(3, 3, 0, &[0, 1, 2, 3, 0, 4, 5, 6]);
        let long = png_from_raw(3, 2, 0, &[0, 1, 2, 3, 0, 4, 5, 6, 0]);
        assert!(fast(&short).is_none());
        assert!(fast(&long).is_none());
    }
}
