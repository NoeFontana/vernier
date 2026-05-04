# vernier

A parity-preserving COCO-style evaluator for instance segmentation, panoptic
segmentation, boundary IoU, OKS keypoints, and LVIS federated evaluation.
Bit-exact against `pycocotools==2.0.11`, `panopticapi`, and `lvis-api` in
strict parity mode — with a documented quirks survey for every place the
reference implementations disagree with themselves.

## Install

```sh
pip install vernier
```

## 60-second example

```python
from pathlib import Path
from vernier.instance import Bbox, Dataset, Evaluator

gt_bytes = Path("instances_val2017.json").read_bytes()
dt_bytes = Path("detections.json").read_bytes()

dataset = Dataset.from_json(gt_bytes)
summary = Evaluator(iou=Bbox()).evaluate(dataset, dt_bytes)
for line in summary.pretty_lines():
    print(line)
```

## Where to go next

- **New to vernier?** Start with [Tutorials](tutorials/README.md).
- **Moving from pycocotools, lvis-api, or panopticapi?** See [Migrate](migrate/README.md).
- **Looking for a specific recipe?** See [How-to](how-to/README.md).
- **Need API details?** See [Reference](reference/README.md).
- **Want to understand the design?** See [Explanation](explanation/README.md).
