//! Wire envelope: magic, version, header, CRC framing.
//!
//! Two-archive layout, little-endian:
//!
//! ```text
//! [ 4 bytes : MAGIC = b"VRPS"        ]
//! [ 1 byte  : FORMAT_VERSION         ]
//! [ 4 bytes : header_archive_len u32 ]
//! [ N bytes : header rkyv archive    ]
//! [ M bytes : body rkyv archive      ]
//! [ 4 bytes : CRC32 over the preceding (9 + N + M) bytes ]
//! ```
//!
//! The header archive carries paradigm-agnostic metadata
//! ([`WireEnvelopeHeader`]). The body archive belongs to each
//! paradigm crate — `vernier-partial` ships and validates body bytes
//! as opaque `&[u8]` and never touches paradigm-specific rkyv types.
//! This keeps the dep DAG flat: paradigm crates depend on
//! `vernier-partial`; `vernier-partial` does not depend on them.

use rkyv::rancor::Error as RkyvError;
use rkyv::util::AlignedVec;

use crate::error::{PartialError, PartialFormatErrorKind};
use crate::traits::PartialExpectation;

/// Wire-format magic: ASCII `"VRPS"` (vernier partial state). Every
/// valid partial starts with these four bytes.
pub const MAGIC: [u8; 4] = *b"VRPS";

/// Wire-format version. Bumped on any breaking change to the framing
/// or the archived header layout. Old versions are refused at decode
/// with [`PartialFormatErrorKind::WrongVersion`].
///
/// **v2 (ADR-0032):** generalized `kernel_kind` u8 → `paradigm_kind`
/// u8 + `discriminator` u32; added `shape_fingerprint: [u32; 4]`;
/// split the body off the header archive (two archives separated by
/// a length prefix) so paradigm crates own their own rkyv invocations.
pub const FORMAT_VERSION: u8 = 2;

/// Minimum bytes a partial must carry to even attempt parsing:
/// 4 magic + 1 version + 4 header_len + 4 CRC. Header and body
/// archive bodies can each be empty in principle (an empty rkyv
/// archive is non-zero bytes, but we don't enforce that here — the
/// rkyv access call surfaces the right error).
const MIN_PARTIAL_BYTES: usize = MAGIC.len() + 1 + 4 + 4;

/// rkyv alignment we copy archive bytes into before
/// [`rkyv::access`]. Covers every primitive rkyv writes on x86_64 /
/// aarch64. Caller-supplied transport bytes are not aligned; we copy
/// once on decode rather than imposing alignment on the wire.
const RKYV_ALIGN: usize = 16;

/// Paradigm-agnostic envelope header. Carries everything
/// `vernier-partial` needs to validate cross-rank compatibility:
/// paradigm + sub-kind, parity mode, dataset / params hashes, the
/// shape fingerprint, and the rank id.
///
/// **Field order is wire-format load-bearing.** rkyv's archived
/// layout follows declaration order; reordering or inserting a field
/// changes the byte layout. Add new fields at the end and bump
/// [`FORMAT_VERSION`].
#[derive(rkyv::Archive, rkyv::Serialize, rkyv::Deserialize, Debug, Clone)]
pub struct WireEnvelopeHeader {
    /// [`ParadigmKind::as_u8`](crate::traits::ParadigmKind::as_u8).
    pub paradigm_kind: u8,
    /// Paradigm-defined sub-kind. Instance: `KernelKind` discriminant.
    /// Semantic: `0`. Panoptic: `0`.
    pub discriminator: u32,
    /// `ParityMode` discriminant.
    pub parity_mode: u8,
    /// Optional rank identifier (none for single-rank flows).
    pub rank_id: Option<u32>,
    /// BLAKE3 over the canonical dataset form.
    pub dataset_hash: [u8; 32],
    /// BLAKE3 over the canonical params archive.
    pub params_hash: [u8; 32],
    /// Paradigm-specific four-slot shape fingerprint. Instance:
    /// `(n_categories, n_area_ranges, n_images, retain_iou as u32)`.
    /// Semantic: `(n_classes, 0, n_images, 0)`. Panoptic:
    /// `(n_categories, 0, n_images, things_stuff_split as u32)`.
    pub shape_fingerprint: [u32; 4],
}

/// Serialize a header + already-archived body into a framed partial
/// blob.
///
/// Each paradigm archives its body via its own rkyv path (the body
/// type lives in the paradigm crate and rkyv's generic bounds are
/// paradigm-specific) and hands the resulting bytes here. We frame
/// with magic, version, header archive, body archive, CRC.
pub fn encode(header: &WireEnvelopeHeader, body_archive: &[u8]) -> Result<Vec<u8>, PartialError> {
    let header_archive = rkyv::to_bytes::<RkyvError>(header).map_err(|e| PartialError::Format {
        kind: PartialFormatErrorKind::RkyvDecode {
            detail: format!("rkyv::to_bytes(header) failed: {e}"),
        },
    })?;
    let header_len = u32::try_from(header_archive.len()).map_err(|_| PartialError::Format {
        kind: PartialFormatErrorKind::RkyvDecode {
            detail: format!("header archive too large: {} bytes", header_archive.len()),
        },
    })?;

    let mut out = Vec::with_capacity(MIN_PARTIAL_BYTES + header_archive.len() + body_archive.len());
    out.extend_from_slice(&MAGIC);
    out.push(FORMAT_VERSION);
    out.extend_from_slice(&header_len.to_le_bytes());
    out.extend_from_slice(&header_archive);
    out.extend_from_slice(body_archive);
    let crc = crc32fast::hash(&out);
    out.extend_from_slice(&crc.to_le_bytes());
    Ok(out)
}

/// Validated archive view passed to the body callback in
/// [`with_validated_envelope`]. Carries the typed rank id (decoded
/// from the header) plus the unaligned body bytes the callback will
/// rkyv-access itself.
pub struct ValidatedView<'a> {
    /// The validated archived header. Borrows from the same aligned
    /// buffer that the callback's lifetime is tied to.
    pub header: &'a ArchivedWireEnvelopeHeader,
    /// Body archive bytes. Caller copies into its own
    /// [`rkyv::util::AlignedVec`] before [`rkyv::access`].
    pub body_archive: &'a [u8],
}

/// Validate a partial blob's framing + header fields and run a
/// callback on the validated view.
///
/// The callback runs while the aligned buffer holding the archived
/// header is still in scope; this is why the API is callback-based
/// rather than returning an owned view.
///
/// **Validation order** (cheapest-first; same as ADR-0031):
///
/// 1. Length: at least the framing-overhead minimum (magic + version
///    byte + header-length prefix + CRC trailer = 13 bytes).
/// 2. Magic: `bytes[..4] == MAGIC`.
/// 3. Version: `bytes[4] == FORMAT_VERSION`.
/// 4. CRC: stored CRC matches `crc32(bytes[..len-4])`.
/// 5. rkyv access on the header archive.
/// 6. Header field comparison against `expected`.
///
/// If any step fails the callback is not invoked.
pub fn with_validated_envelope<R>(
    bytes: &[u8],
    expected: &PartialExpectation,
    body_callback: impl FnOnce(ValidatedView<'_>) -> Result<R, PartialError>,
) -> Result<R, PartialError> {
    let (header_bytes, body_bytes) = validate_framing(bytes)?;

    let mut aligned: AlignedVec<RKYV_ALIGN> = AlignedVec::with_capacity(header_bytes.len());
    aligned.extend_from_slice(header_bytes);
    let archived =
        rkyv::access::<ArchivedWireEnvelopeHeader, RkyvError>(&aligned).map_err(|e| {
            PartialError::Format {
                kind: PartialFormatErrorKind::RkyvDecode {
                    detail: format!("rkyv::access(header) failed: {e}"),
                },
            }
        })?;
    validate_header_fields(archived, expected)?;
    body_callback(ValidatedView {
        header: archived,
        body_archive: body_bytes,
    })
}

/// Validate magic / version / CRC / header_len framing. On success
/// returns `(header_archive_bytes, body_archive_bytes)` slices of the
/// input.
fn validate_framing(bytes: &[u8]) -> Result<(&[u8], &[u8]), PartialError> {
    if bytes.len() < MIN_PARTIAL_BYTES {
        return Err(PartialError::Format {
            kind: PartialFormatErrorKind::TooShort {
                observed: bytes.len(),
                minimum: MIN_PARTIAL_BYTES,
            },
        });
    }

    let magic: [u8; 4] = bytes[..4].try_into().map_err(|_| PartialError::Format {
        kind: PartialFormatErrorKind::TooShort {
            observed: bytes.len(),
            minimum: MIN_PARTIAL_BYTES,
        },
    })?;
    if magic != MAGIC {
        return Err(PartialError::Format {
            kind: PartialFormatErrorKind::WrongMagic { found: magic },
        });
    }

    let version = bytes[4];
    if version != FORMAT_VERSION {
        return Err(PartialError::Format {
            kind: PartialFormatErrorKind::WrongVersion {
                expected: FORMAT_VERSION,
                found: version,
            },
        });
    }

    // CRC must match before we trust any further byte read. Otherwise
    // a corrupted header_len could point past the end of the buffer
    // and we'd surface a confusing error.
    let crc_split = bytes.len() - 4;
    let stored_crc =
        u32::from_le_bytes(
            bytes[crc_split..]
                .try_into()
                .map_err(|_| PartialError::Format {
                    kind: PartialFormatErrorKind::Crc,
                })?,
        );
    let actual_crc = crc32fast::hash(&bytes[..crc_split]);
    if stored_crc != actual_crc {
        return Err(PartialError::Format {
            kind: PartialFormatErrorKind::Crc,
        });
    }

    let header_len =
        u32::from_le_bytes(bytes[5..9].try_into().map_err(|_| PartialError::Format {
            kind: PartialFormatErrorKind::TooShort {
                observed: bytes.len(),
                minimum: MIN_PARTIAL_BYTES,
            },
        })?) as usize;
    let header_end = 9usize.saturating_add(header_len);
    if header_end > crc_split {
        return Err(PartialError::Format {
            kind: PartialFormatErrorKind::TooShort {
                observed: bytes.len(),
                minimum: header_end + 4,
            },
        });
    }

    Ok((&bytes[9..header_end], &bytes[header_end..crc_split]))
}

fn validate_header_fields(
    archived: &ArchivedWireEnvelopeHeader,
    expected: &PartialExpectation,
) -> Result<(), PartialError> {
    let paradigm = archived.paradigm_kind;
    if paradigm != expected.paradigm.as_u8() {
        return Err(PartialError::Format {
            kind: PartialFormatErrorKind::ParadigmMismatch {
                expected: expected.paradigm.as_u8(),
                found: paradigm,
            },
        });
    }
    // Paradigm matched; sub-kind discriminator must match too.
    let discriminator = archived.discriminator.to_native();
    if discriminator != expected.discriminator {
        return Err(PartialError::Format {
            kind: PartialFormatErrorKind::KernelMismatch {
                expected: expected.discriminator,
                found: discriminator,
            },
        });
    }
    let parity_mode = archived.parity_mode;
    if parity_mode != expected.parity_mode {
        return Err(PartialError::Format {
            kind: PartialFormatErrorKind::ParityMismatch {
                expected: expected.parity_mode,
                found: parity_mode,
            },
        });
    }
    let fingerprint = [
        archived.shape_fingerprint[0].to_native(),
        archived.shape_fingerprint[1].to_native(),
        archived.shape_fingerprint[2].to_native(),
        archived.shape_fingerprint[3].to_native(),
    ];
    if fingerprint != expected.shape_fingerprint {
        return Err(PartialError::Format {
            kind: PartialFormatErrorKind::GridMismatch {
                detail: format!(
                    "expected {:?}, got {:?}",
                    expected.shape_fingerprint, fingerprint
                ),
            },
        });
    }
    let dataset_hash: [u8; 32] = archived.dataset_hash;
    if dataset_hash != expected.dataset_hash {
        return Err(PartialError::DatasetMismatch {
            expected: expected.dataset_hash,
            actual: dataset_hash,
        });
    }
    let params_hash: [u8; 32] = archived.params_hash;
    if params_hash != expected.params_hash {
        return Err(PartialError::ParamsMismatch {
            expected: expected.params_hash,
            actual: params_hash,
        });
    }
    Ok(())
}

/// Read the typed `rank_id` field off an archived header. Centralized
/// because [`Option<u32>`]'s archived form is `ArchivedOption<u32>`
/// which needs the `.as_ref().map(...)` pattern at every read site.
pub fn rank_id_from_archive(header: &ArchivedWireEnvelopeHeader) -> Option<u32> {
    header.rank_id.as_ref().map(|v| v.to_native())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::traits::ParadigmKind;

    fn fake_expectation() -> PartialExpectation {
        PartialExpectation {
            paradigm: ParadigmKind::Instance,
            discriminator: 0,
            parity_mode: 1,
            dataset_hash: [0xAB; 32],
            params_hash: [0xCD; 32],
            shape_fingerprint: [80, 4, 5000, 0],
            strict_mode: false,
        }
    }

    fn fake_header() -> WireEnvelopeHeader {
        WireEnvelopeHeader {
            paradigm_kind: ParadigmKind::Instance.as_u8(),
            discriminator: 0,
            parity_mode: 1,
            rank_id: None,
            dataset_hash: [0xAB; 32],
            params_hash: [0xCD; 32],
            shape_fingerprint: [80, 4, 5000, 0],
        }
    }

    #[test]
    fn round_trip_empty_body() {
        let bytes = encode(&fake_header(), &[]).unwrap();
        let exp = fake_expectation();
        with_validated_envelope(&bytes, &exp, |view| {
            assert!(view.body_archive.is_empty());
            assert_eq!(view.header.paradigm_kind, ParadigmKind::Instance.as_u8());
            Ok(())
        })
        .unwrap();
    }

    #[test]
    fn rejects_too_short() {
        let err = validate_framing(b"VRP").unwrap_err();
        assert!(matches!(
            err,
            PartialError::Format {
                kind: PartialFormatErrorKind::TooShort { .. }
            }
        ));
    }

    #[test]
    fn rejects_wrong_magic() {
        let mut bytes = vec![0u8; MIN_PARTIAL_BYTES + 8];
        bytes[..4].copy_from_slice(b"FAKE");
        bytes[4] = FORMAT_VERSION;
        // crc still wrong but magic check fires first
        let err = validate_framing(&bytes).unwrap_err();
        assert!(matches!(
            err,
            PartialError::Format {
                kind: PartialFormatErrorKind::WrongMagic { .. }
            }
        ));
    }

    #[test]
    fn rejects_wrong_version() {
        let mut bytes = vec![0u8; MIN_PARTIAL_BYTES + 8];
        bytes[..4].copy_from_slice(&MAGIC);
        bytes[4] = 99;
        let err = validate_framing(&bytes).unwrap_err();
        assert!(matches!(
            err,
            PartialError::Format {
                kind: PartialFormatErrorKind::WrongVersion { .. }
            }
        ));
    }

    #[test]
    fn rejects_bad_crc() {
        let mut bytes = encode(&fake_header(), &[]).unwrap();
        let n = bytes.len();
        bytes[n - 1] ^= 0xFF;
        let exp = fake_expectation();
        let err = with_validated_envelope(&bytes, &exp, |_| Ok(())).unwrap_err();
        assert!(matches!(
            err,
            PartialError::Format {
                kind: PartialFormatErrorKind::Crc
            }
        ));
    }

    #[test]
    fn rejects_paradigm_mismatch() {
        let bytes = encode(&fake_header(), &[]).unwrap();
        let mut exp = fake_expectation();
        exp.paradigm = ParadigmKind::Semantic;
        let err = with_validated_envelope(&bytes, &exp, |_| Ok(())).unwrap_err();
        match err {
            PartialError::Format {
                kind: PartialFormatErrorKind::ParadigmMismatch { expected, found },
            } => {
                assert_eq!(expected, ParadigmKind::Semantic.as_u8());
                assert_eq!(found, ParadigmKind::Instance.as_u8());
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn rejects_discriminator_mismatch() {
        let bytes = encode(&fake_header(), &[]).unwrap();
        let mut exp = fake_expectation();
        exp.discriminator = 1;
        let err = with_validated_envelope(&bytes, &exp, |_| Ok(())).unwrap_err();
        assert!(matches!(
            err,
            PartialError::Format {
                kind: PartialFormatErrorKind::KernelMismatch { .. }
            }
        ));
    }

    #[test]
    fn rejects_dataset_hash_mismatch() {
        let bytes = encode(&fake_header(), &[]).unwrap();
        let mut exp = fake_expectation();
        exp.dataset_hash = [0; 32];
        let err = with_validated_envelope(&bytes, &exp, |_| Ok(())).unwrap_err();
        assert!(matches!(err, PartialError::DatasetMismatch { .. }));
    }

    #[test]
    fn round_trip_with_body() {
        // body archive is just opaque bytes from this crate's perspective
        let body = b"\x00\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a\x0b";
        let bytes = encode(&fake_header(), body).unwrap();
        let exp = fake_expectation();
        with_validated_envelope(&bytes, &exp, |view| {
            assert_eq!(view.body_archive, body);
            Ok(())
        })
        .unwrap();
    }
}
