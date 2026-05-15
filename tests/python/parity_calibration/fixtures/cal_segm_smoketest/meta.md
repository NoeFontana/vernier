# cal_segm_smoketest

Identical detection cells to `cal_perfect`, tagged
`iou_type="segm"` in `meta.json`.

- Exercises ADR-0018 Shape-1 iou_type-genericity at the *data level*:
  the per-image cell layout (`accumulate.rs:62`) is bit-identical across
  bbox / segm / boundary. The oracle reads the same fields and produces
  the same numbers regardless of which similarity kernel populated them.
- No quirks specific to segm here; the fixture is the iou_type-coverage
  guard. Expected ECE equals `cal_perfect`'s.
