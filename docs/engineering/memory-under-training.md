# Memory under training-loop load

`BackgroundEvaluator` (ADR-0006) lives on a dedicated worker thread
behind a bounded `mpsc::sync_channel`. We want empirical evidence that
RSS plateaus rather than creeps under fast `submit` calls from a
training inner loop.

## How to run

```bash
uv run python bench/bench/runners/memory_bench.py
```

Output: `bench/results/memory/training-loop.csv`, schema
`timestamp_s, rss_bytes` (one row per 100 ms sample, captured by
`bench.harness.rss.RSSSampler`). The runner synthesizes a small
COCO-shaped GT (50 images, 80 categories, 1 GT/image) and fakes the
training loop with a generator — no `torch` import.

## Expected shape

RSS climbs during the first few epochs while the cell store warms
up, then flattens. A monotone climb past ~epoch 10 indicates the
worker is retaining per-submit state instead of folding it into the
accumulator.
