# vernier

[![PyPI](https://img.shields.io/pypi/v/vernier.svg)](https://pypi.org/project/vernier/)
[![Python](https://img.shields.io/pypi/pyversions/vernier.svg)](https://pypi.org/project/vernier/)
[![Crates.io](https://img.shields.io/crates/v/vernier.svg)](https://crates.io/crates/vernier)
[![License](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](#license)

**Fast, auditable evaluation for 2D vision models.** Detection, instance and
panoptic segmentation, semantic segmentation, keypoints, and LVIS, all in one
package with a Rust core, a Python API, and a standalone CLI.

- **Bit-exact** with `pycocotools==2.0.11`, `panopticapi`, `lvis-api` and
  `boundary-iou-api` in strict mode. Every upstream quirk has a documented
  disposition ([quirks survey](docs/engineering/pycocotools-quirks.md)).
- **Drop-in** for `pycocotools.cocoeval.COCOeval`: change one import, or none.
- **3–17× faster** than faster-coco-eval and pycocotools at equal CPU budget
  ([benchmarks](#performance)).
- **Built for real pipelines**: training-loop evaluation, multi-rank
  gathering, per-image tables, error decomposition, calibration, scenario
  slicing.

## Install

```sh
pip install vernier               # Python ≥ 3.10, abi3 wheels
cargo binstall vernier-cli        # standalone `vernier` binary, no Python needed
cargo add vernier                 # Rust library
```

## Quickstart: the COCO API you already know

`vernier.COCOeval` has the same constructor, the same
`evaluate() / accumulate() / summarize()` sequence, and the same `.stats`
as pycocotools. It defaults to `parity_mode="strict"`, so the output is
bit-identical.

```python
from pycocotools.coco import COCO
from vernier import COCOeval  # was: from pycocotools.cocoeval import COCOeval

coco_gt = COCO("instances_val2017.json")
coco_dt = coco_gt.loadRes("detections.json")

E = COCOeval(coco_gt, coco_dt, iouType="bbox")  # "segm" | "keypoints" | "boundary"
E.evaluate()
E.accumulate()
E.summarize()
```

**Code you can't edit** (mmdetection, detectron2, ultralytics, …): patch
the symbol before anything imports `pycocotools.cocoeval`.

```python
import vernier

unpatch = vernier.patch_pycocotools()  # pycocotools.cocoeval.COCOeval -> vernier
run_existing_eval()
unpatch()
```

The patch is explicit, reversible, and never happens on import. A context
manager (`vernier.adapters.patched_pycocotools`) and a one-fixture pytest
recipe are in the [pycocotools migration guide](docs/migrate/from-pycocotools.md#pytest-integration).

## Recommended: the native API

The shim exists for migration. New code should use the native `Evaluator`:
immutable configuration, typed results, no pycocotools dependency, and
access to everything below.

```python
from pathlib import Path
from vernier.instance import Bbox, CocoDataset, Evaluator

gt = CocoDataset.from_json(Path("instances_val2017.json").read_bytes())
dt = Path("detections.json").read_bytes()

evaluator = Evaluator(iou=Bbox(), parity_mode="strict")
summary = evaluator.evaluate(gt, dt, num_threads=8)

print("\n".join(summary.pretty_lines()))  # the familiar 12-line table
ap = summary.stats[0]
```

> The native `Evaluator` defaults to `parity_mode="corrected"`, which applies
> the [itemized fixes](docs/engineering/pycocotools-quirks.md) to upstream
> bugs. Pass `"strict"` when your numbers must match published pycocotools
> results.

**Inside a training loop**, evaluate on a background worker and feed it
tensors directly (torch, JAX, CuPy, NumPy via DLPack, zero-copy):

```python
with evaluator.background(gt) as bg:
    for images, targets in val_loader:
        preds = model(images)
        bg.submit([{"image_id": int(t["image_id"]), **p} for t, p in zip(targets, preds)])
    summary = bg.finalize()
```

**Per-image and per-class diagnostics** as Polars DataFrames
(`pip install "vernier[tables]"`):

```python
result = evaluator.evaluate(gt, dt, tables="all")
result.per_class
```

### Paradigms

Pick the submodule that matches your model's output. They have different
data models and matching rules, so they are separate evaluators rather than
one class with a mode switch ([why](docs/explanation/three-paradigms.md)).

| Submodule | Input | Metrics |
| --- | --- | --- |
| `vernier.instance` | Scored detections: boxes, masks, keypoints | AP / AR (bbox, segm, boundary, OKS), LVIS federated AP |
| `vernier.panoptic` | Panoptic PNGs + `segments_info` | PQ / SQ / RQ, boundary PQ |
| `vernier.semantic` | Class-id label maps | mIoU, FWIoU, pixel accuracy, mean accuracy |

### Beyond the COCO API

| Need | Feature | Guide |
| --- | --- | --- |
| Evaluate without blocking training | `Evaluator.background(...)` | [how-to](docs/how-to/background-evaluator.md) |
| Evaluate across DDP ranks | `evaluate_to_partial` / `from_partials` | [how-to](docs/how-to/distributed-eval.md) |
| Find which images/classes regressed | `tables="all"` | [how-to](docs/how-to/result-tables.md) |
| Explain an AP gap | TIDE error decomposition, oLRP | [tutorial](docs/tutorials/debugging-with-tide.md) |
| Check score calibration | ECE / MCE / reliability (`calibration=True`) | [how-to](docs/how-to/calibration.md) |
| Metrics per weather, time of day, … | Manifest slicing, `vernier aggregate` (mPC / rPC) | [how-to](docs/how-to/scenario-slicing.md) |
| Non-standard IoU / recall / area grids | `iou_thresholds=`, `recall_thresholds=`, `area_ranges=` | [how-to](docs/how-to/custom-evaluation-grids.md) |

## CLI

A static binary for CI gates and robotics replay pipelines. Output is
byte-deterministic (sorted keys, no timestamps), so artifacts diff cleanly.

```sh
vernier eval --gt gt.json --dt dt.json --iou-type bbox                  # pycocotools-identical stdout
vernier eval --gt gt.json --dt dt.json --iou-type segm --emit json=result.json --threads 8
```

Exit codes: `0` success, `1` evaluation error, `2` invalid arguments.
Full reference: [`crates/vernier-cli`](crates/vernier-cli/README.md).

## Status & validation

Every row is checked by a parity harness that runs the reference
implementation and vernier on the same inputs. "Bit-exact" means
`parity_mode="strict"`.

| Metric | Reference | Parity | Notes |
| --- | --- | --- | --- |
| bbox / segm / keypoints AP | `pycocotools==2.0.11` | bit-exact | |
| Boundary AP | `boundary-iou-api` | bit-exact | |
| LVIS federated AP | `lvis-api` 0.5.3 | bit-exact | full v1 val, bbox |
| Panoptic PQ, boundary PQ | `panopticapi` (single-core path) | bit-exact | Cityscapes panoptic deferred |
| Semantic mIoU / FWIoU / pAcc / mAcc | `mmseg.IoUMetric` v1.2.2 (vendored) | bit-exact on class marginals | [ADR-0036](docs/adr/0036-vendor-mmsegmentation-ioumetric.md) proposed; ADE20K-scale check pending |
| oLRP | clean-room NumPy oracle | ≤ 1e-9 | panoptic not supported |
| Calibration (ECE / MCE) | clean-room NumPy oracle | bit-exact | detection family only |
| TIDE thresholds (segm, boundary) | none | corrected only | [ADR-0022](docs/adr/0022-tide-thresholds.md) proposed |

Parity model: [ADR-0002](docs/adr/0002-three-tier-parity-model.md).
Library-by-library comparison: [`docs/comparison.md`](docs/comparison.md).

## Performance

Median wall time at a one-CPU budget (CPU/wall is 1.00 in every cell).
COCO val2017 throughout, except the LVIS row, which is LVIS v1 val.
Speedup is the other library's time divided by vernier's.

<!-- Hand-mirrored from docs/benchmarks.md (generated by tools/render_benchmarks.py).
     After a bench round, re-run the renderer, then update these numbers. -->

| Workload | vernier | vs pycocotools | vs faster-coco-eval | vs hotcoco |
| --- | ---: | ---: | ---: | ---: |
| bbox AP | 354 ms | 16.0× | 4.7× | 1.6× |
| segm AP | 968 ms | 6.7× | 3.5× | 1.4× |
| keypoints AP | 136 ms | 16.9× | 5.7× | 1.6× |
| boundary AP | 3.2 s | 19.5× ¹ | 16.7× | — |
| Panoptic PQ | 10.6 s | 3.3× ² | — | — |
| Semantic mIoU | 2.9 s | 14.0× ³ | — | — |
| LVIS v1 bbox AP | 2.6 s | 73.1× ⁴ | — | 1.4× |

¹ boundary-iou-api · ² panopticapi · ³ mmsegmentation · ⁴ lvis-api, with 10× lower peak memory (1.45 vs 15.0 GiB)

<details>
<summary>Thread scaling (<code>num_threads</code>, 8-vCPU host)</summary>

faster-coco-eval ≥ 1.8 and hotcoco are multi-threaded by default, so each
column compares equal CPU budgets.

| Workload | 1 | 2 | 4 | 8 | vs hotcoco @ 8 | vs faster-coco-eval @ 8 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| bbox | 354 ms | 267 ms | 229 ms | 226 ms | 1.6× | 6.5× |
| segm | 983 ms | 569 ms | 375 ms | 319 ms | 1.7× | 11.0× |
| boundary | 3.20 s | 1.70 s | 937 ms | 790 ms | — | 21.5× |
| keypoints | 136 ms | 118 ms | 109 ms | 104 ms | 1.5× | 7.3× |
| bbox, Objects365 (1.06M dets) † | 8.8 s | — | — | 4.7 s | 1.7× | OOM at ~30 GiB |

Strict-mode results are bit-identical across thread counts.
† Single-rep dev run; pycocotools takes ~6 min per rep at this size.

</details>

AMD EPYC-Milan KVM VPS (4 cores x 2 threads = 8 logical CPUs), harness
mode `release` (N=10 measurement reps + 2 warmup, randomised impl order,
5% relative-IQR gate), against `hotcoco==1.0.1`,
`faster-coco-eval==1.8.0` and `pycocotools==2.0.11`. Full methodology,
every baseline pin and the complete results:
[`docs/benchmarks.md`](docs/benchmarks.md).

## Documentation

- [Tutorials](docs/tutorials/): first evaluation, training-loop integration
- [Migration guides](docs/migrate/): pycocotools, faster-coco-eval,
  panopticapi, lvis-api, mmsegmentation
- [How-to guides](docs/how-to/) · [Reference](docs/reference/) ·
  [Design decisions (ADRs)](docs/adr/)

## Contributing

```sh
just lint && just test && just audit
```

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for the ADR workflow, vendoring
policy, and code style.

## License

Licensed under either of [Apache-2.0](LICENSE-APACHE) or [MIT](LICENSE-MIT)
at your option. Test-only reference implementations used for parity checks
are listed in [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md). They are not
shipped in wheels or binaries.
