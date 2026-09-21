#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::print_stderr
)]

//! Bridge test: the Rust D2 replica against detectron2's own C++ kernel.
//!
//! ADR-0063 M3's exit gate is *matrix bit-equality*, and this is where
//! that claim is actually made. Everything else in the replica's unit
//! tests checks that the numbers are plausible; only this file checks
//! that they are the oracle's.
//!
//! The harness is not built automatically, because building it pins
//! compiler flags and that is a decision, not a side effect. Build it
//! and point this test at it:
//!
//! ```bash
//! ./tests/python/parity_obb/oracle/detectron2/build.sh /tmp/d2harness
//! VERNIER_OBB_D2_HARNESS=/tmp/d2harness cargo test -p vernier-geom --test d2_bridge
//! ```
//!
//! Without the variable the test reports what it skipped and passes, so
//! a clean checkout stays green — the harness needs a C++ toolchain and
//! a deliberate flag set, neither of which a `cargo test` should
//! conjure. `just test-obb-parity` builds it and runs this test, and
//! CI's `obb-parity` job runs that recipe.
//!
//! Because "skipped" and "passed" look identical to a CI summary, that
//! job also sets `VERNIER_OBB_REQUIRE_BRIDGE=1`, which turns a missing
//! harness into a failure. Without it the M3 exit gate could quietly
//! stop being checked and the lane would stay green.

use std::io::Write;
use std::path::PathBuf;
use std::process::Command;

/// Deterministic xorshift64*, so a failure is reproducible from the
/// seed alone.
struct Rng(u64);

impl Rng {
    fn next_f64(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        ((self.0 >> 11) as f64) * (1.0 / 9_007_199_254_740_992.0)
    }

    fn range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + self.next_f64() * (hi - lo)
    }
}

/// One generated case, already narrowed to f32 so both sides see
/// identical inputs and the comparison is about the kernel, not about
/// who rounded first.
struct Case {
    b1: [f32; 5],
    b2: [f32; 5],
}

/// Cases span the regimes that actually break OBB kernels: near-total
/// overlap (where the epsilon-relaxed hull matters), grazing contact
/// (where the `EPS` slack decides whether a point is collected at all),
/// exact quadrant angles, square aspect ratios where the hull sort's
/// distance tie-break fires, and large coordinates where f32 runs out
/// of mantissa.
fn cases(n: usize) -> Vec<Case> {
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let regime = i % 5;
        let (cx, cy) = match regime {
            4 => (rng.range(-2.0e4, 2.0e4), rng.range(-2.0e4, 2.0e4)),
            _ => (rng.range(-50.0, 50.0), rng.range(-50.0, 50.0)),
        };
        let (w, h) = match regime {
            3 => {
                let s = rng.range(1.0, 20.0);
                (s, s)
            }
            _ => (rng.range(0.5, 40.0), rng.range(0.5, 40.0)),
        };
        let a1 = match regime {
            2 => 90.0 * (rng.range(0.0, 4.0).floor()),
            _ => rng.range(-180.0, 180.0),
        };
        // How far the second box drifts: regime 0 nearly coincides,
        // regime 1 grazes, the rest are ordinary.
        let spread = match regime {
            0 => 0.05,
            1 => (w + h) * 0.5,
            _ => 25.0,
        };
        let a2 = match regime {
            2 => 90.0 * (rng.range(0.0, 4.0).floor()),
            _ => a1 + rng.range(-40.0, 40.0),
        };
        #[allow(clippy::cast_possible_truncation)]
        let f = |v: f64| v as f32;
        out.push(Case {
            b1: [f(cx), f(cy), f(w), f(h), f(a1)],
            b2: [
                f(cx + rng.range(-spread, spread)),
                f(cy + rng.range(-spread, spread)),
                f(w * rng.range(0.5, 1.6)),
                f(h * rng.range(0.5, 1.6)),
                f(a2),
            ],
        });
    }
    out
}

#[test]
fn replica_is_bit_equal_to_the_vendored_kernel() {
    let Ok(harness) = std::env::var("VERNIER_OBB_D2_HARNESS") else {
        // A test that reports a skip and passes is a test that can
        // silently stop testing. `VERNIER_OBB_REQUIRE_BRIDGE` is how
        // the lane that is *supposed* to run it says so: with it set,
        // a missing harness is a failure rather than a shrug.
        assert!(
            std::env::var_os("VERNIER_OBB_REQUIRE_BRIDGE").is_none(),
            "VERNIER_OBB_REQUIRE_BRIDGE is set but VERNIER_OBB_D2_HARNESS is not: \
             the D2 bridge was expected to run and would have skipped. Build it with \
             tests/python/parity_obb/oracle/detectron2/build.sh"
        );
        eprintln!(
            "skipped: set VERNIER_OBB_D2_HARNESS to the binary built by \
             tests/python/parity_obb/oracle/detectron2/build.sh, or run \
             `just test-obb-parity`"
        );
        return;
    };

    // Scale is a CI knob, not a source constant: the local default
    // runs in well under a second, while the nightly OBB lane sets this
    // to 1e8 for the full ADR-0063 M3 sweep.
    let n: usize = std::env::var("VERNIER_OBB_BRIDGE_N")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(200_000);
    let cases = cases(n);

    let dir = std::env::temp_dir().join(format!("vernier-obb-bridge-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("temp dir");
    let pairs_path: PathBuf = dir.join("pairs.bin");
    let out_path: PathBuf = dir.join("out.bin");

    {
        let mut f = std::fs::File::create(&pairs_path).expect("write pairs");
        let mut buf = Vec::with_capacity(n * 10 * 4);
        for c in &cases {
            for v in c.b1.iter().chain(c.b2.iter()) {
                buf.extend_from_slice(&v.to_le_bytes());
            }
        }
        f.write_all(&buf).expect("write pairs");
    }

    let status = Command::new(&harness)
        .arg(&pairs_path)
        .arg(&out_path)
        .status()
        .expect("run harness");
    assert!(status.success(), "harness exited with {status}");

    let raw = std::fs::read(&out_path).expect("read results");
    assert_eq!(raw.len(), n * 4, "harness produced the wrong record count");

    let mut mismatches = 0_usize;
    let mut first: Option<String> = None;
    for (i, c) in cases.iter().enumerate() {
        let mut b = [0_u8; 4];
        b.copy_from_slice(&raw[i * 4..i * 4 + 4]);
        let oracle = f32::from_le_bytes(b);

        let b1 = c.b1.map(f64::from);
        let b2 = c.b2.map(f64::from);
        let ours = vernier_geom::replica::d2::iou(&b1, &b2);

        // The oracle's f32 widened to f64 is exact, and the replica
        // normalizes `-0.0` on the way out, so the comparison is on
        // bits modulo signed zero -- which is exactly ADR-0063's
        // equivalence relation.
        let want = f64::from(oracle) + 0.0;
        if want.is_finite() != ours.is_finite()
            || (want.is_finite() && want.to_bits() != ours.to_bits())
        {
            mismatches += 1;
            if first.is_none() {
                first = Some(format!(
                    "case {i}: b1={:?} b2={:?} oracle={want:?} ({:#x}) replica={ours:?} ({:#x})",
                    c.b1,
                    c.b2,
                    want.to_bits(),
                    ours.to_bits()
                ));
            }
        }
    }

    let _ = std::fs::remove_dir_all(&dir);
    assert_eq!(
        mismatches,
        0,
        "{mismatches}/{n} pairs diverged; first: {}",
        first.unwrap_or_default()
    );
}
