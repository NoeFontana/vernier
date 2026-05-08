# vernier

A parity-preserving COCO-style evaluator for instance segmentation, panoptic
segmentation, boundary IoU, OKS keypoints, semantic segmentation, and LVIS
federated evaluation. Bit-exact against `pycocotools==2.0.11`, `panopticapi`,
and `lvis-api` in strict parity mode — with a documented quirks survey for
every place the reference implementations disagree with themselves.

## Why vernier

`pycocotools==2.0.11` is the reference for COCO evaluation and it is slow,
unmaintained, and full of edge-case quirks that downstream tools either
silently fix or silently inherit. Faster reimplementations exist, but each
chooses its own quirk dispositions, leaving users to discover the divergences
empirically. vernier takes a third path:

- **Auditable parity.** Every divergence from `pycocotools` is filed in a
  three-tier quirks survey under
  [ADR-0002](https://github.com/NoeFontana/vernier/blob/main/docs/adr/0002-three-tier-parity-model.md):
  `strict` (bit-equal reproduction), `aligned` (within a documented
  tolerance), or `corrected` (opt-in opinionated fix). The default is
  strict; corrected fixes are itemized so you know exactly when your numbers
  diverge from a reference run.
- **A unified evaluation toolkit.** bbox / segm / keypoints AP, boundary IoU,
  panoptic PQ, semantic mIoU, and LVIS federated evaluation all live in one
  package, behind one Python API and one CLI. No more wrestling with
  fragmented `pycocotools`, `boundary-iou-api`, `panopticapi`, `lvis-api`,
  and `mmsegmentation` installs (each has a per-paradigm
  [migration guide](migrate/README.md)).
- **Drop-in shim.** `vernier.patch_pycocotools()` swaps the `COCOeval` symbol
  in place — existing pycocotools-based scripts switch with one line
  ([ADR-0007](https://github.com/NoeFontana/vernier/blob/main/docs/adr/0007-patch-pycocotools-policy.md)).
- **Rust core.** The matching kernel is Rust with runtime SIMD dispatch via
  [`pulp`](https://github.com/sarah-quinones/pulp); the FFI layer is data
  conversion only. The CLI ships as a static binary, so CI pipelines can call
  vernier without provisioning a Python interpreter.

## Install

```sh
pip install vernier
```

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
