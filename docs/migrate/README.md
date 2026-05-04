# Migrate

Step-by-step guides for moving an existing evaluation pipeline onto vernier.
Each guide covers the API mapping, behavioural differences, and the one or two
sentinel/quirk traps most likely to affect real workloads.

- [From `pycocotools`](from-pycocotools.md) — the upstream case; covers the
  `COCOeval` shim and `patch_pycocotools` drop-in.
- [From `faster-coco-eval`](from-faster-coco-eval.md) — the divergent-stance
  case; vernier's auditable parity contract vs faster-coco-eval's opaque
  quirk fixes.
- [From `panopticapi`](from-panopticapi.md) — panoptic-quality (PQ).
- [From `lvis-api`](from-lvis-api.md) — long-tail / federated detection.
- [From `mmsegmentation`](from-mmsegmentation.md) — semantic segmentation.
- [From flat-root imports (0.0.x → 0.1.x)](from-flat-root.md) — internal
  upgrade path for users on the pre-namespace API.
