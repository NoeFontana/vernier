//! Serial vs split-parallel JSON loading (ADR-0054).
//!
//! `o365_phase_profile` reports the parse phase as one number. This
//! example splits that number the other way — serial `from_json_bytes`
//! against `from_json_bytes_parallel` at a given thread count — and
//! asserts the two produce the same annotations, so a speedup can never
//! be read without its parity check alongside.
//!
//! Run:
//! ```sh
//! cargo run --release --example json_parse_profile -p vernier-core -- <gt.json> [threads]
//! ```

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::print_stdout)]

use std::time::Instant;

use vernier_core::{CocoDataset, EvalDataset};

fn time_ms<F: FnOnce() -> R, R>(f: F) -> (R, f64) {
    let t = Instant::now();
    let r = f();
    (r, t.elapsed().as_secs_f64() * 1000.0)
}

fn main() {
    let mut args = std::env::args().skip(1);
    let path = args
        .next()
        .expect("usage: json_parse_profile <gt.json> [threads]");
    let threads: usize = args
        .next()
        .map_or_else(rayon::current_num_threads, |s| s.parse().expect("threads"));

    let bytes = std::fs::read(&path).expect("read GT");
    println!(
        "{path}: {:.1} MiB, {threads} threads",
        bytes.len() as f64 / 1_048_576.0
    );

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .expect("build pool");

    // Two rounds: the first warms the page cache and the allocator, the
    // second is the one to read.
    for round in 0..2 {
        let (serial, serial_ms) =
            time_ms(|| CocoDataset::from_json_bytes(&bytes).expect("serial parse"));
        let (parallel, parallel_ms) = time_ms(|| {
            pool.install(|| {
                CocoDataset::from_json_bytes_parallel(&bytes, threads).expect("parallel parse")
            })
        });

        let same = format!("{:?}", serial.annotations()) == format!("{:?}", parallel.annotations())
            && format!("{:?}", serial.images()) == format!("{:?}", parallel.images())
            && format!("{:?}", serial.categories()) == format!("{:?}", parallel.categories());

        println!(
            "round {round}: serial {serial_ms:>7.0} ms | parallel {parallel_ms:>7.0} ms | \
             {:.2}x | {} annotations | identical: {same}",
            serial_ms / parallel_ms,
            serial.annotations().len(),
        );
        assert!(same, "parallel loader diverged from serial");
    }
}
