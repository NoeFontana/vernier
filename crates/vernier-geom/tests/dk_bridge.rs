#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::print_stderr
)]

//! Bridge test: the Rust DK replica against DOTA_devkit's own C++
//! `iou_poly`.
//!
//! ADR-0063 M3's exit gate for the `Quad` geometry. The oracle is not
//! vendored — DOTA_devkit carries no license — so the harness is built
//! from the developer-provisioned cache:
//!
//! ```bash
//! uv run python tests/python/parity_obb/oracle/dota_devkit/fetch.py
//! VERNIER_OBB_DK_HARNESS=.cache/dota-devkit/dk_harness \
//!   cargo test -p vernier-geom --test dk_bridge --release
//! ```
//!
//! Without the variable the test reports the skip and passes, so a
//! clean checkout stays green. `just test-obb-parity` provisions and
//! runs it; set `VERNIER_OBB_REQUIRE_BRIDGE=1` to make a skip a
//! failure, which is what a lane that means to run it should do.
//!
//! The provisioning step needs the network and reaches an unlicensed
//! upstream, so — unlike the D2 bridge — this one is **not** wired into
//! CI. That is a deliberate limitation of the `Quad` strict claim and
//! is recorded as such in `docs/engineering/obb-quirks.md`.
//!
//! The harness deliberately runs the **raw** kernel, with no prefilter,
//! so this file compares [`vernier_geom::replica::dk::iou_poly_raw`]
//! rather than the composed [`vernier_geom::replica::dk::iou`]. The
//! horizontal-box gate is checked separately, in the replica's own unit
//! tests, because conflating the two would let a prefilter bug hide
//! behind a kernel that returns zero anyway.

use std::io::Write;
use std::process::Command;

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

/// A rotated rectangle as a quad, which is what a real DOTA submission
/// looks like.
fn rect_quad(cx: f64, cy: f64, w: f64, h: f64, deg: f64) -> [f64; 8] {
    let a = deg.to_radians();
    let (s, c) = a.sin_cos();
    let (hw, hh) = (w * 0.5, h * 0.5);
    let (ux, uy) = (c * hw, s * hw);
    let (vx, vy) = (-s * hh, c * hh);
    [
        cx + ux + vx,
        cy + uy + vy,
        cx - ux + vx,
        cy - uy + vy,
        cx - ux - vx,
        cy - uy - vy,
        cx + ux - vx,
        cy + uy - vy,
    ]
}

/// Regimes that stress the fan decomposition: integer coordinates (real
/// DOTA annotations), tile-scale offsets from the origin (the fan hinges
/// there, so this is where its conditioning is worst), near-coincident
/// quads, grazing pairs, and free-form quads that are neither convex nor
/// consistently wound.
fn cases(n: usize) -> Vec<([f64; 8], [f64; 8])> {
    let mut rng = Rng(0xDEAD_BEEF_CAFE_F00D);
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let regime = i % 5;
        let (cx, cy) = match regime {
            1 => (rng.range(0.0, 2.0e4), rng.range(0.0, 2.0e4)),
            _ => (rng.range(-200.0, 200.0), rng.range(-200.0, 200.0)),
        };
        let w = rng.range(2.0, 80.0);
        let h = rng.range(2.0, 80.0);
        let a = rng.range(-180.0, 180.0);
        let spread = match regime {
            2 => 0.1,
            3 => (w + h) * 0.5,
            _ => 40.0,
        };
        let mut g = rect_quad(cx, cy, w, h, a);
        let (dcx, dcy) = (
            cx + rng.range(-spread, spread),
            cy + rng.range(-spread, spread),
        );
        let (dw, dh) = (w * rng.range(0.6, 1.5), h * rng.range(0.6, 1.5));
        let da = a + rng.range(-50.0, 50.0);
        let mut d = rect_quad(dcx, dcy, dw, dh, da);
        match regime {
            0 => {
                // Integer coordinates, as DOTA ships them.
                for v in g.iter_mut().chain(d.iter_mut()) {
                    *v = v.round();
                }
            }
            4 => {
                // Free-form: perturb one vertex hard enough to make the
                // quad non-convex, and sometimes reverse the winding.
                d[4] += rng.range(-w, w);
                d[5] += rng.range(-h, h);
                if rng.next_f64() < 0.5 {
                    let r = [d[6], d[7], d[4], d[5], d[2], d[3], d[0], d[1]];
                    d = r;
                }
            }
            _ => {}
        }
        out.push((g, d));
    }
    out
}

#[test]
fn replica_is_bit_equal_to_the_cached_kernel() {
    let Ok(harness) = std::env::var("VERNIER_OBB_DK_HARNESS") else {
        // See the D2 bridge: an opt-in gate that reports a skip and
        // passes is a gate that can stop gating without anyone noticing.
        assert!(
            std::env::var_os("VERNIER_OBB_REQUIRE_BRIDGE").is_none(),
            "VERNIER_OBB_REQUIRE_BRIDGE is set but VERNIER_OBB_DK_HARNESS is not: \
             the DK bridge was expected to run and would have skipped. Provision it \
             with tests/python/parity_obb/oracle/dota_devkit/fetch.py"
        );
        eprintln!(
            "skipped: run tests/python/parity_obb/oracle/dota_devkit/fetch.py, \
             then set VERNIER_OBB_DK_HARNESS to the built binary — or run \
             `just test-obb-parity`, which does both when the network allows"
        );
        return;
    };

    let n: usize = std::env::var("VERNIER_OBB_BRIDGE_N")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(200_000);
    let cases = cases(n);

    let dir = std::env::temp_dir().join(format!("vernier-dk-bridge-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("temp dir");
    let pairs_path = dir.join("pairs.bin");
    let out_path = dir.join("out.bin");

    {
        let mut f = std::fs::File::create(&pairs_path).expect("write pairs");
        let mut buf = Vec::with_capacity(n * 16 * 8);
        for (g, d) in &cases {
            for v in g.iter().chain(d.iter()) {
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
    assert_eq!(raw.len(), n * 8, "harness produced the wrong record count");

    let mut mismatches = 0_usize;
    let mut nans = 0_usize;
    let mut stale = 0_u64;
    let mut first: Option<String> = None;
    for (i, (g, d)) in cases.iter().enumerate() {
        let mut b = [0_u8; 8];
        b.copy_from_slice(&raw[i * 8..i * 8 + 8]);
        let oracle = f64::from_le_bytes(b);
        let (ours, trace) = vernier_geom::replica::dk::iou_poly_raw(g, d);
        stale += trace.stale_slots;

        let agree = if oracle.is_nan() {
            nans += 1;
            ours.is_nan()
        } else {
            // Modulo signed zero, per ADR-0063's equivalence relation.
            (oracle + 0.0).to_bits() == (ours + 0.0).to_bits()
        };
        if !agree {
            mismatches += 1;
            if first.is_none() {
                first = Some(format!(
                    "case {i}: gt={g:?} dt={d:?} oracle={oracle:?} replica={ours:?}"
                ));
            }
        }
    }

    let _ = std::fs::remove_dir_all(&dir);
    eprintln!("dk bridge: {n} pairs, {nans} NaN, {stale} stale-slot reads");
    assert_eq!(
        mismatches,
        0,
        "{mismatches}/{n} pairs diverged; first: {}",
        first.unwrap_or_default()
    );
}
