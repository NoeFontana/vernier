# Third-party notices

vernier carries pinned references to a small set of third-party
reference implementations, used only in the test and bench harnesses.
None of these are imported by `python/vernier/` or linked into the
Rust binary; the published wheel does not contain vendored bytes and
does not depend on vendored packages.

Vendoring takes two flavors here, both covered by the policy in
[`docs/engineering/vendoring.md`](docs/engineering/vendoring.md):

- **In-tree source vendoring** — verbatim upstream source checked into
  the repo at a pinned commit SHA (e.g. `boundary-iou-api`).
- **Pinned-package envs** — third-party packages pinned at exact
  versions in `pyproject.toml` + `uv.lock`, where the pin itself is
  the parity / comparator claim (e.g. `pycocotools`,
  `faster-coco-eval`).

For each entry, see the linked `VENDORING.md` (in-tree flavor) or
the linked pin sites (pinned-package flavor) for provenance and
refresh discipline. Adding a new vendored reference is an ADR-level
decision regardless of flavor.

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

## pycocotools

- **Role:** the canonical parity oracle for COCO-style evaluation
  (ADR-0002). Every quirk vernier reproduces in `strict` mode is
  keyed to the exact bytes this pin selects; bumping it is an
  ADR-level decision (see [`CLAUDE.md`](CLAUDE.md#parity-contract--read-before-changing-eval-logic)).
- **Vendoring flavor:** pinned-package env. The pin is the artifact;
  no source tree lives in our repo.
- **Pin sites:**
  - Root [`pyproject.toml`](pyproject.toml) — `pycocotools==2.0.11`,
    consumed by `tests/python/parity/`.
  - [`bench/envs/pycocotools/pyproject.toml`](bench/envs/pycocotools/pyproject.toml)
    — `pycocotools==2.0.11`, consumed by the bench harness's
    pycocotools runner subprocess (ADR-0017). Mirrors the root pin.
- **Lockfiles:** [`uv.lock`](uv.lock) (root) and
  [`bench/envs/pycocotools/uv.lock`](bench/envs/pycocotools/uv.lock).
- **Upstream:** <https://github.com/cocodataset/cocoapi>
  (Python package: <https://pypi.org/project/pycocotools/>).
- **License:** BSD-2-Clause ("FreeBSD" in upstream metadata).
  Copyright © 2014 Piotr Dollar and Tsung-Yi Lin. Same license text
  as the bundled COCOAPI notice in
  [`tests/python/parity_boundary/oracle/boundary_iou_api/LICENSE`](tests/python/parity_boundary/oracle/boundary_iou_api/LICENSE)
  (the boundary-iou-api LICENSE preserves the COCOAPI notice
  verbatim because it ships pycocotools derivatives).
- **Quirks survey:** [`docs/engineering/pycocotools-quirks.md`](docs/engineering/pycocotools-quirks.md)
  — every vernier-side disposition (`strict` / `aligned` / `corrected`)
  is keyed against this exact version.

## faster-coco-eval

- **Role:** comparator implementation for the bench harness
  (ADR-0017). vernier's bench compares throughput against
  faster-coco-eval as one of the reference points; the pin keeps
  bench numbers reproducible across runs.
- **Vendoring flavor:** pinned-package env.
- **Pin site:** [`bench/envs/faster-coco-eval/pyproject.toml`](bench/envs/faster-coco-eval/pyproject.toml)
  — `faster-coco-eval>=1.6` resolved to an exact version by the
  lockfile.
- **Lockfile:** [`bench/envs/faster-coco-eval/uv.lock`](bench/envs/faster-coco-eval/uv.lock).
- **Upstream:** <https://github.com/MiXaiLL76/faster_coco_eval>
  (Python package: <https://pypi.org/project/faster-coco-eval/>).
- **License:** Apache-2.0.
- **Notes:** faster-coco-eval ships shims that monkey-patch the
  `pycocotools` namespace so existing call sites stay verbatim
  (per the bench-env's pyproject comment); the pin therefore
  behaves as a drop-in replacement at the import layer, but the
  numerical behavior is the upstream's, not pycocotools'. Treated
  as a comparator, not an oracle.

<!-- Future vendored references append here, same shape. -->
