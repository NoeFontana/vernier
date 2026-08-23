# vernier

A parity-preserving COCO-style evaluator for instance segmentation, panoptic
segmentation, boundary IoU, OKS keypoints, semantic segmentation, LVIS
federated evaluation, LRP / oLRP error decomposition, and detection-family
calibration (ECE / MCE / reliability). Bit-exact against
`pycocotools==2.0.11`, `panopticapi`, and `lvis-api` in strict parity mode;
semantic mIoU is calibrated against a vendored `mmsegmentation` `IoUMetric`.
See the per-paradigm matrix in the
[README §Status & validation](https://github.com/NoeFontana/vernier/#status--validation)
for the full per-paradigm picture, plus a documented quirks survey for every
place the reference implementations disagree with themselves.

## Why vernier

`pycocotools==2.0.11` is the reference for COCO evaluation and it is slow,
unmaintained, and full of edge-case quirks that downstream tools either
silently fix or silently inherit. Faster reimplementations exist, but each
chooses its own quirk dispositions, leaving users to discover the divergences
empirically. vernier takes a third path:

- **Auditable parity.** Every divergence from `pycocotools` is filed in the
  quirks survey under
  [ADR-0002](https://github.com/NoeFontana/vernier/blob/main/docs/adr/0002-three-tier-parity-model.md)
  as either `strict` (bit-equal output, even when vernier's implementation
  is structurally different) or `corrected` (opt-in opinionated fix). The
  default is strict; corrected fixes are itemized so you know exactly when
  your numbers diverge from a reference run.
- **A unified evaluation toolkit.** bbox / segm / keypoints AP, boundary IoU,
  panoptic PQ, semantic mIoU, and LVIS federated evaluation all live in one
  package, behind one Python API and one CLI. No more wrestling with
  fragmented `pycocotools`, `boundary-iou-api`, `panopticapi`, `lvis-api`,
  and `mmsegmentation` installs (each has a per-paradigm
  [migration guide](migrate/README.md)).
- **Scenario slicing and cross-run aggregation.** A partition manifest
  (`weather`, `time_of_day`, …) feeds `vernier eval --manifest` for
  per-slice headline metrics and `vernier aggregate` for cross-run
  corruption tables (mPC / rPC) — one matching pass, N slices
  ([ADR-0046](https://github.com/NoeFontana/vernier/blob/main/docs/adr/0046-slice-and-aggregate.md)).
- **Drop-in shim.** `vernier.patch_pycocotools()` swaps the `COCOeval` symbol
  in place — existing pycocotools-based scripts switch with one line
  ([ADR-0007](https://github.com/NoeFontana/vernier/blob/main/docs/adr/0007-patch-pycocotools-policy.md)).
- **Rust core.** The matching kernel is Rust with runtime SIMD dispatch via
  [`pulp`](https://github.com/sarah-quinones/pulp); the FFI layer is data
  conversion only. The CLI ships as a static binary, so CI pipelines can call
  vernier without provisioning a Python interpreter.
- **Opt-in parallelism with zero-overhead default.** Every public
  evaluate surface accepts `num_threads=N` (or `vernier eval --threads
  N`) to parallelise inside a single eval call across `N` rayon
  workers. The default `None` is byte-for-byte the sequential path —
  no rayon symbol entered — so users not opting in see no behaviour
  change. Measured ~3.25× scaling at `num_threads=4` on val2017
  boundary IoU; strict-mode results stay bit-equal across thread
  counts ([ADR-0047](https://github.com/NoeFontana/vernier/blob/main/docs/adr/0047-threading-model.md)).

## Install

```sh
pip install vernier
```

For the Rust library:

```sh
cargo add vernier
```

`vernier` is a facade
([ADR-0048](https://github.com/NoeFontana/vernier/blob/main/docs/adr/0048-vernier-facade-crate.md)):
it re-exports the paradigm crates under one dependency and one module
map — `vernier::{instance, mask, panoptic, semantic, partial}` —
mirroring the Python namespace. Depending on a leaf crate directly
(`cargo add vernier-core`) is equally supported when a narrower
dependency tree matters.

For the CLI binary on its own:

```sh
cargo binstall vernier-cli
# Or compile from source:
cargo install vernier-cli
```

## 60-second example

```python
from pathlib import Path
from vernier.instance import Bbox, CocoDataset, Evaluator

gt_bytes = Path("instances_val2017.json").read_bytes()
dt_bytes = Path("detections.json").read_bytes()

dataset = CocoDataset.from_json(gt_bytes)
summary = Evaluator(iou=Bbox()).evaluate(dataset, dt_bytes)
for line in summary.pretty_lines():
    print(line)
```

## Three evaluation paradigms

vernier ships three sibling submodules — pick the one whose **input shape**
matches your model's output:

```python
import vernier

# Detections (bbox / segm / boundary / keypoints) with scores → AP fold
vernier.instance.Evaluator()

# RGB-encoded panoptic PNGs + segments_info JSON → PQ
vernier.panoptic.Evaluator()

# Single-channel class-id label maps → mIoU / FWIoU / pAcc / mAcc
vernier.semantic.Evaluator()
```

The submodules are mutually exclusive (different data models, different
matching rules, different parity oracles). See
[Three paradigms](explanation/three-paradigms.md) for when to use which.

## Where to go next

- **New to vernier?** Start with [Tutorials](tutorials/README.md).
- **Migrating from pycocotools, faster-coco-eval, panopticapi, lvis-api, or mmsegmentation?**
  See [Migrate](migrate/README.md).
- **Comparing alternatives?** [How vernier compares](comparison.md) is a
  per-library decision aid (when to pick vernier, when to keep what you
  have).
- **Curious about speed?** [Benchmarks](benchmarks.md) carries the
  per-cell medians and methodology.
- **Looking for a specific recipe?** See [How-to](how-to/README.md).
- **Need API details?** See [Reference](reference/README.md).
- **Want to understand the design?** See [Explanation](explanation/README.md)
  or browse the
  [ADRs](https://github.com/NoeFontana/vernier/tree/main/docs/adr).
