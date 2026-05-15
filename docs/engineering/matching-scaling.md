# Matching scaling (Phase 3 robustness)

`crates/vernier-core/src/matching.rs` is intentionally `O(T · D · G)`
per image — its inner triple loop mirrors
`pycocotools.cocoeval.COCOeval.evaluateImg` line-for-line for parity.
The 10× synthetic sweep stresses the dense regime at 10× val2017's
image count and captures every `match_image` call's `(G, D, wall_ns)`
to rule out a hidden quadratic-in-N blowup leaking in via accumulator
state or a stray nested scan upstream.

## Hypothesis

`wall_ns` vs `G·D` (log–log) clusters along slope ≈ 1 over 4+ decades.
A visible knee around `G·D ≈ 1000`, or a 10× cloud sitting above the
1× cloud at matched `G·D`, falsifies the hypothesis.

## Workload (`make_workload_scaled(scale=10)`)

50 000 images, 80 categories, 100 GT, 100 DT per image, default IoU
ladder (10 thresholds), seed 0.

## How to run

```bash
cargo build --release --features bench-histogram -p vernier-core
```

```python
from bench.workloads.synthetic import make_workload_scaled
gt_path, dt_path = make_workload_scaled(scale=10)
# ... feed into the evaluator ...
```

Drain via `vernier_core::matching::histogram::dump_csv(path)`. Schema:
`g,d,wall_ns`. An FFI hook analogous to
`vernier._core.dump_bbox_iou_histogram` is a follow-up.
