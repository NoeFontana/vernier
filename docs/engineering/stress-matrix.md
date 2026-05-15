# Stress matrix

Verifies vernier scales along the axes modern ML setups actually push:
detection density (DETR-style), GT density (LVIS-crowded), category
count (LVIS / Open Images), and image dimensions (satellite /
pathology). Companion to `matching-scaling.md`, which only varies image
count.

## Named regimes

`bench/bench/workloads/stress_matrix.py::REGIMES`:

| Regime | Images | Cats | DT/img | GT/img | Dims | IoU |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| `coco-baseline` | 5000 | 80 | 100 | 10 | 640×480 | bbox |
| `detr-output` | 5000 | 80 | 500 | 10 | 640×480 | bbox |
| `lvis-crowded` | 2000 | 1203 | 100 | 50 | 1024×1024 | segm |
| `open-images-cats` | 1000 | 10000 | 50 | 30 | 1024×1024 | bbox |
| `satellite-4k` | 500 | 50 | 50 | 50 | 4096×4096 | segm |
| `pathology-8k` | 100 | 20 | 100 | 100 | 8000×8000 | segm |

## Per-axis sweeps

Each sweep fixes everything at a baseline (2000 images, 80 cats, 100
DT, 30 GT, 640×480, bbox) and varies one knob:

- `--axis dt` — `dt_per_image ∈ {10, 100, 500}`
- `--axis gt` — `gt_per_image ∈ {10, 100, 500}`
- `--axis cats` — `n_categories ∈ {80, 1203, 10000}` (image count tapers to keep wall time bounded)
- `--axis dims` — image dims ∈ `{640×480, 1920×1080, 4096×4096}` with segm IoU (exercises the RLE/raster path)

## How to run

```bash
uv run python bench/bench/runners/stress_runner.py --regime detr-output
uv run python bench/bench/runners/stress_runner.py --axis dt
uv run python bench/bench/runners/stress_runner.py --all
```

Results land at `bench/results/stress/results.json` — one record per
regime with wall time and AP. For per-call distributions, build with
`--features bench-histogram` and dump via
`vernier_core::matching::histogram::dump_csv` between runs (see
`matching-scaling.md`).

## What to look for

- **DT sweep** stresses the `match_image` inner loop and the
  score-sorted argsort. Wall time should grow linearly with
  `dt_per_image` at fixed everything else.
- **GT sweep** stresses the per-image gather and the ignore-mask path.
  Wall time should grow linearly with `gt_per_image`.
- **Category sweep** stresses per-(image, category) sharding and the
  aggregator. Wall time should grow approximately linearly with
  `n_categories` for fixed total `n_images * n_categories` work.
- **Dims sweep** stresses `vernier-mask`'s RLE codec and polygon
  rasterizer (segm IoU). Wall time grows with image area, dominated
  by mask decode at large dims.

A sub-linear envelope is the goal; super-linear growth at any axis is
the alarm signal.
