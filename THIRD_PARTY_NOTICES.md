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
- **Hash-pinned, not redistributed** — an upstream we read and
  reproduce but do not copy, because it carries no license grant. The
  SHA-256 of each file is pinned and a fetch script provisions it into
  a git-ignored dev cache (`DOTA_devkit`). This flavor exists because
  the alternative — quietly vendoring unlicensed source — is not one.

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
| **PANOPTICAPI**         | © 2018 Alexander Kirillov                              | BSD-2-Clause  | **Yes** — vendored at [`tests/python/parity_panoptic/oracle/panopticapi/`](tests/python/parity_panoptic/oracle/panopticapi/) (ADR-0025) for the panoptic-quality parity oracle. The boundary-iou-api LICENSE preserves this notice independently. |
| **Cityscapes Dataset**  | Daimler AG, MPI Informatics, TU Darmstadt              | Custom (non-commercial; see LICENSE) | No — upstream's `cityscapes_*_api/` subtrees are skipped. Notice preserved per clause 2. |

The Cityscapes notice carries a non-commercial restriction. Because
no Cityscapes code or data is shipped in our tree, that restriction
does not propagate to vernier's MIT/Apache-2.0 dual license; the
notice is preserved as documentation only. If a future ADR proposes
vendoring any Cityscapes subtree, that ADR has to address the
licensing implications first.

## panopticapi

- **Role:** bit-exact parity oracle for panoptic-quality (PQ)
  evaluation (ADR-0025). Consumed only by
  `tests/python/parity_panoptic/`; not imported by `python/vernier/`
  or any code that ships in the wheel.
- **Path:** [`tests/python/parity_panoptic/oracle/panopticapi/`](tests/python/parity_panoptic/oracle/panopticapi/)
- **Upstream:** <https://github.com/cocodataset/panopticapi>
- **Pinned commit:** `7bb4655548f98f3fedc07bf37e9040a992b054b0` (2021-06-17)
- **Primary license:** BSD-2-Clause. Copyright © 2018 Alexander Kirillov.
- **License text:** [`tests/python/parity_panoptic/oracle/panopticapi/LICENSE`](tests/python/parity_panoptic/oracle/panopticapi/LICENSE)
  (renamed from upstream `license.txt` for tooling discoverability;
  contents byte-equal — see `VENDORING.md` byte-equality table).
- **Vendoring details:** [`tests/python/parity_panoptic/oracle/VENDORING.md`](tests/python/parity_panoptic/oracle/VENDORING.md)
- **Runtime dep:** `Pillow==12.2.0` (oracle imports `PIL.Image` at
  module load; pin mirrored by `ORACLE_PILLOW_PIN` in
  [`crates/vernier-panoptic/src/parity.rs`](crates/vernier-panoptic/src/parity.rs)).

## mmsegmentation

- **Role:** bit-exact parity oracle for semantic-segmentation
  evaluation (ADR-0036). Only the single file
  `mmseg/evaluation/metrics/iou_metric.py` is vendored — `mmcv`,
  `mmengine`, and the rest of the mmsegmentation package are
  satisfied by hand-written stubs in
  [`tests/python/parity_semantic/oracle/mmsegmentation/_mmengine_stub.py`](tests/python/parity_semantic/oracle/mmsegmentation/_mmengine_stub.py).
  Consumed only by `tests/python/parity_semantic/`; not imported by
  `python/vernier/` or any code that ships in the wheel.
- **Path:** [`tests/python/parity_semantic/oracle/mmsegmentation/`](tests/python/parity_semantic/oracle/mmsegmentation/)
- **Upstream:** <https://github.com/open-mmlab/mmsegmentation>
- **Pinned commit:** `c685fe6767c4cadf6b051983ca6208f1b9d1ccb8` (2023-12-14, tag `v1.2.2`)
- **Primary license:** Apache-2.0. Copyright 2020 The MMSegmentation Authors.
- **License text:** [`tests/python/parity_semantic/oracle/mmsegmentation/LICENSE`](tests/python/parity_semantic/oracle/mmsegmentation/LICENSE)
- **Vendoring details:** [`tests/python/parity_semantic/oracle/mmsegmentation/VENDORING.md`](tests/python/parity_semantic/oracle/mmsegmentation/VENDORING.md)
- **Runtime dep:** `torch>=2.4` (oracle calls `torch.histc` for label
  binning; floor mirrored by `ORACLE_TORCH_FLOOR` in
  [`crates/vernier-semantic/src/parity.rs`](crates/vernier-semantic/src/parity.rs)).
  `Pillow==12.2.0` is shared with the panopticapi vendor (ADR-0025).

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
  — every vernier-side disposition (`strict` / `corrected`)
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

## rfdetr

- **Role:** real-model source for the TIDE validation harness
  (Week 5 of the 0.x.x TIDE track). The bundled `RFDETRNano` (bbox)
  and `RFDETRSegNano` (instance segmentation) checkpoints generate
  COCO-format predictions on `val2017`; vernier's
  `error_decomposition` is then exercised against those predictions
  to confirm bin assignments behave coherently on real data. This is
  the "non-parity sanity-check" vendoring flavor anticipated in
  [`docs/engineering/vendoring.md`](docs/engineering/vendoring.md):
  not a parity oracle, not a comparator, just a load-bearing input
  for the validation harness.
- **Vendoring flavor:** pinned-package env. The pin is the artifact;
  no source tree lives in our repo.
- **Pin site:** root [`pyproject.toml`](pyproject.toml) under the
  `real-models` optional-dependency group — `rfdetr==1.6.5.post0`,
  consumed by `tests/python/integration/real_models/tide/`. The
  harness is gated on `@pytest.mark.real_models` and on the rfdetr
  import succeeding, so it skips cleanly when the extra is not
  installed.
- **Lockfile:** [`uv.lock`](uv.lock) (root). Refreshing the pin
  also rewrites the lockfile in the same commit.
- **Upstream:** <https://github.com/roboflow/rf-detr>
  (Python package: <https://pypi.org/project/rfdetr/>).
- **License:** Apache-2.0. Copyright © Roboflow, Inc.
- **Notes:** rfdetr's `.predict()` returns a `supervision.Detections`
  object; the harness adapts it to COCO JSON before passing the bytes
  through `vernier.error_decomposition`. The COCO category-id mapping
  is derived from the GT JSON at runtime (instances_val2017.json
  carries the canonical sparse 1..90 ids), not hard-coded.

<!-- Future vendored references append here, same shape. -->


## detectron2 (`box_iou_rotated`)

- **Role:** bit-exact parity oracle for oriented-box (`RotatedBox`)
  evaluation (ADR-0063). Consumed only by
  `tests/python/parity_obb/` and the Rust bridge test
  [`crates/vernier-geom/tests/d2_bridge.rs`](crates/vernier-geom/tests/d2_bridge.rs);
  not imported by `python/vernier/` or any code that ships in the
  wheel. Two files are vendored: the C++ kernel header and the
  `RotatedCOCOeval` evaluator that instantiates it.
- **Path:** [`tests/python/parity_obb/oracle/detectron2/`](tests/python/parity_obb/oracle/detectron2/)
- **Upstream:** <https://github.com/facebookresearch/detectron2>
- **Pinned commit:** `a25898a09d6ee232767647e92c6177fb1c642369` (2026-03-16)
- **Primary license:** Apache-2.0. Copyright (c) Facebook, Inc. and its
  affiliates.
- **License text:** [`tests/python/parity_obb/oracle/detectron2/LICENSE`](tests/python/parity_obb/oracle/detectron2/LICENSE)
- **Vendoring details:** [`tests/python/parity_obb/oracle/VENDORING.md`](tests/python/parity_obb/oracle/VENDORING.md)
- **Build note:** the claim is keyed to a `linux-x86_64` build. The
  kernel is a `float` template, and baseline `x86-64` has no FMA for
  the compiler to contract `a*b + c` into, while GCC on `aarch64`
  fuses by default and would move the last bits of every cross
  product. `build.sh` beside the header pins the flags.
- **Runtime dep:** `torch` (test-only, for the `RotatedCOCOeval` bridge
  and the threshold-dtype probe behind quirk **OB10**). Shared with the
  mmsegmentation vendor.

## DOTA_devkit (`polyiou`)

- **Role:** bit-exact parity oracle for quad (`Quad`) evaluation
  (ADR-0063). Consumed only by `tests/python/parity_obb/` and
  [`crates/vernier-geom/tests/dk_bridge.rs`](crates/vernier-geom/tests/dk_bridge.rs).
- **Path:** **none — no bytes of this upstream are in this
  repository.**
- **Upstream:** <https://github.com/CAPTAIN-WHU/DOTA_devkit>
- **Pinned commit:** `d3f8da45d4091b1dab37d9fbe4d6e6a50928e410` (2019-04-26)
- **License:** **none stated.** There is no `LICENSE` file in the
  repository, no license header in any source file, no license
  statement in `readme.md` or `setup.py`, and the GitHub API reports
  `"license": null`. With no grant there is no right to redistribute,
  so vernier pins the SHA-256 of each file instead of copying it.
  Reading a public source to reproduce its observable behavior is not
  redistribution;
  [`crates/vernier-geom/src/replica/dk.rs`](crates/vernier-geom/src/replica/dk.rs)
  carries no upstream text.
- **Provisioning:** `uv run python tests/python/parity_obb/oracle/dota_devkit/fetch.py`
  downloads the three pinned files into `.cache/dota-devkit/`, verifies
  each SHA-256, and builds the bridge harness. Tests that need it skip
  cleanly when it is absent.
- **Vendoring details:** [`tests/python/parity_obb/oracle/VENDORING.md`](tests/python/parity_obb/oracle/VENDORING.md)
