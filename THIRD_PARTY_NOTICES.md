# Third-party notices

vernier carries verbatim copies of selected third-party reference
implementations, vendored at pinned commit SHAs and used only in the
test harness. None of this code is included in the published wheel
or linked into the Rust binary; all of it is preserved with its
original license text.

For each entry, see the linked `VENDORING.md` for provenance,
modifications policy, and refresh procedure. Adding a new vendored
reference is an ADR-level decision — see
[`docs/engineering/vendoring.md`](docs/engineering/vendoring.md).

## boundary-iou-api

- **Role:** bit-exact parity oracle for boundary-IoU evaluation
  (ADR-0010). Consumed only by `tests/python/parity_boundary/`; not
  imported by `python/vernier/` or any code that ships in the wheel.
  The bench harness reaches the same tree through a symlink at
  [`bench/envs/boundary-iou-api/oracle/`](bench/envs/boundary-iou-api/oracle/);
  there is one canonical vendored copy.
- **Path:** [`tests/python/parity_boundary/oracle/boundary_iou_api/`](tests/python/parity_boundary/oracle/boundary_iou_api/)
- **Upstream:** <https://github.com/bowenc0221/boundary-iou-api>
- **Pinned commit:** `37d25586a677b043ed585f10e5c42d4e80176ea9` (2021-04-05)
- **Primary license:** BSD-2-Clause. Copyright © 2021 Bowen Cheng.
- **License text:** [`tests/python/parity_boundary/oracle/boundary_iou_api/LICENSE`](tests/python/parity_boundary/oracle/boundary_iou_api/LICENSE)
- **Vendoring details:** [`tests/python/parity_boundary/oracle/VENDORING.md`](tests/python/parity_boundary/oracle/VENDORING.md)

### Bundled attributions

The upstream's `LICENSE` file bundles four further notices that the
upstream chose to redistribute. Per BSD-2-Clause clause 2 we preserve
each verbatim regardless of whether we ship the corresponding source
subtree. The "code in our tree?" column makes that disposition
explicit.

| Notice                  | Copyright                                              | License       | Code in our tree? |
| ----------------------- | ------------------------------------------------------ | ------------- | ----------------- |
| **COCOAPI**             | © 2014 Piotr Dollar and Tsung-Yi Lin                   | BSD-2-Clause  | **Yes** — `boundary_iou/coco_instance_api/{coco,cocoeval}.py` are derivatives of pycocotools. |
| **LVIS API**            | © 2019 Agrim Gupta and Ross Girshick                   | BSD-2-Clause  | No — upstream's `lvis_instance_api/` is skipped (LVIS dropped from Phase 2 per ADR-0010). Notice preserved per clause 2. |
| **PANOPTICAPI**         | © 2018 Alexander Kirillov                              | BSD-2-Clause  | No — upstream's `coco_panoptic_api/` is skipped (panoptic out of scope). Notice preserved per clause 2. |
| **Cityscapes Dataset**  | Daimler AG, MPI Informatics, TU Darmstadt              | Custom (non-commercial; see LICENSE) | No — upstream's `cityscapes_*_api/` subtrees are skipped. Notice preserved per clause 2. |

The Cityscapes notice carries a non-commercial restriction. Because
no Cityscapes code or data is shipped in our tree, that restriction
does not propagate to vernier's MIT/Apache-2.0 dual license; the
notice is preserved as documentation only. If a future ADR proposes
vendoring any Cityscapes subtree, that ADR has to address the
licensing implications first.

<!-- Future vendored references append here, same shape. -->
