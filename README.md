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
  Same constructor, same `params` mutations, same `eval` / `evalImgs` /
  `ious` / `stats` read back off the instance ([what carries
  over](docs/migrate/from-pycocotools.md#params-what-the-shim-honors-and-what-it-rejects)).
- **8–54× faster** than faster-coco-eval and pycocotools at equal CPU budget,
  measured on real detector output rather than jittered ground truth
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
bit-identical (see [Status & validation](#status--validation) for what
that is measured against).

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

TorchMetrics' `MeanAveragePrecision` runs under the patch unchanged,
including `class_metrics=True` and `extended_summary=True`
(`tests/python/test_compat_torchmetrics.py` asserts the patched and
unpatched results are equal). Two `params` fields are the exception and
raise rather than diverge quietly: `imgIds` subsetting, and `areaRng`
(custom area ranges are a native-`Evaluator` feature, ADR-0040). The
[migration guide](docs/migrate/from-pycocotools.md#params-what-the-shim-honors-and-what-it-rejects)
has the field-by-field table.

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
`parity_mode="strict"`, against the reference **as published on PyPI**
— vernier's own output is identical on x86-64 and ARM. (A pycocotools
you recompile from source can round bbox IoU 1-2 ULP differently on
ARM; see [ADR-0056](docs/adr/0056-pin-no-fp-contraction-for-bbox-iou.md)
and the [migration guide](docs/migrate/from-pycocotools.md#bit-for-bit-and-against-which-build).)

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

The **DT source** column is load-bearing. A *jittered* workload perturbs the
ground truth to synthesise detections, which inherits the GT's per-image
class distribution — not the distribution a detector actually produces. The
bbox and segm rows are measured on real RF-DETR predictions on val2017
(513,808 and 430,516 detections) and the keypoints row on real ViTPose
output; the remaining rows still run against synthetic DT, and are
labelled as such rather than presented as equivalent.

<!-- Hand-mirrored from docs/benchmarks.md (generated by
     tools/render_benchmarks.py), except the keypoints row, which is from
     docs/engineering/benchmarking/2026-09-rfdetr-real-predictions.md.
     After a bench round, re-run the renderer, then update these numbers. -->

| Workload | DT source | vernier | vs pycocotools | vs faster-coco-eval | vs hotcoco | vs ufcoco |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| bbox AP | RF-DETR Nano | 434 ms | 53.9× ⁶ | 11.3× | 1.56× | 1.25× |
| segm AP | RF-DETR Seg-Nano | 1.20 s | 20.0× | 8.2× | 1.68× | 1.52× |
| keypoints AP | ViTPose-B ⁵ | 129 ms | 18.6× | 6.1× | 1.6× | — |
| boundary AP | jittered GT | 2.68 s | 22.3× ¹ | 19.9× | — | 78.6× |
| Panoptic PQ | perfect DT | 3.68 s | 9.6× ² | — | 1.92× | — |
| Semantic mIoU | perfect DT | 1.56 s | 28.6× ³ | — | — | — |
| LVIS v1 bbox AP | jittered GT | 1.04 s | 217.0× ⁴ | — | 1.78× | — |

¹ boundary-iou-api · ² panopticapi · ³ mmsegmentation · ⁴ lvis-api, with 20× lower peak memory (787 MiB vs 15.08 GiB) · ⁵ from the 2026-09 real-predictions snapshot, not re-run in this round · ⁶ pycocotools' reps narrowly missed the 5% IQR gate (5.07%), so read the ratio as approximate

A second real detector on the same kernel and GT gives a different point
estimate: RF-DETR Seg-Nano's bbox output (16% fewer detections, higher AP)
measures 41.3× / 9.5× / 1.58× / 1.21×. Treat a single-workload speedup as a
point estimate, not a property of the library — the
[latest benchmark snapshot](docs/engineering/benchmarking/2026-11-hotcoco-1.5-fce-1.8.1-round.md)
gives per-stage breakdowns. `vernier` leads every cell in the matrix, including
1-CPU jittered bbox (`134 ms` against `ufcoco`'s `155 ms` and `hotcoco`'s `211 ms`).

Peak RSS on the real Nano bbox cell is 276 MiB, against 267 MiB (hotcoco),
224 MiB (ufcoco), 1384 MiB (faster-coco-eval) and 1429 MiB (pycocotools). On
real Seg-Nano masks it is 502 MiB vs 648 MiB (hotcoco), 380 MiB (ufcoco),
1752 MiB (faster-coco-eval) and 1486 MiB (pycocotools).

<details>
<summary>Thread scaling (<code>num_threads</code>, 8-vCPU host)</summary>

faster-coco-eval ≥ 1.8, hotcoco and ufcoco are multi-threaded by default, so
each column compares equal CPU budgets.

| Workload | 1 | 2 | 4 | 8 | vs hotcoco @ 8 | vs faster-coco-eval @ 8 | vs ufcoco @ 1 / 8 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| bbox | 136 ms | 94 ms | 61 ms | 57 ms | 1.32× | 25.9× | 1.15× / 1.94× |
| segm | 364 ms | 208 ms | 121 ms | 101 ms | 2.00× | 34.5× | 2.00× / 2.21× |
| boundary | 2.67 s | 1.36 s | 719 ms | 603 ms | — | 28.1× | 80.2× / 83.7× |
| keypoints | 98 ms | 62 ms | 38 ms | 32 ms | 1.34× | 24.7× | 1.20× / 2.15× |
| bbox, Objects365 (1.06M dets) † | 2.5 s | — | — | 962 ms | 1.45× | OOM (>16.8 GiB) | 1.29× / 1.98× |

Strict-mode results are bit-identical across thread counts.
† Single-rep dev run; pycocotools takes ~6.6 min per rep at this size. A
memory-capped probe of `faster-coco-eval==1.8.1` exceeded 16.8 GiB RSS in 56 s
(and 1.8.0 OOM-killed at ~30 GiB), so it is aborted/omitted rather than
destabilising the host mid-round.

</details>

AMD EPYC-Milan KVM VPS (4 cores x 2 threads = 8 logical CPUs), harness
mode `release` (N=10 measurement reps + 2 warmup, randomised impl order,
5% relative-IQR gate), against `hotcoco==1.5.0`,
`ultrafast-pycocotools==0.1.14`, `faster-coco-eval==1.8.1` and
`pycocotools==2.0.11`. Full methodology,
every baseline pin and the complete results:
[`docs/benchmarks.md`](docs/benchmarks.md) for every timed row except keypoints,
[the latest snapshot](docs/engineering/benchmarking/2026-11-hotcoco-1.5-fce-1.8.1-round.md)
for the round's narrative.

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
