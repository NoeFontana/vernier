# Vendored oracle: `bowenc0221/boundary-iou-api`

This directory contains a frozen, verbatim copy of the
[`bowenc0221/boundary-iou-api`](https://github.com/bowenc0221/boundary-iou-api)
reference implementation of Boundary IoU (Cheng, Girshick, Dollár, Berg,
Kirillov; *Boundary IoU: Improving Object-Centric Image Segmentation
Evaluation*, CVPR 2021; arXiv:2103.16562).

The oracle is consumed only by `tests/python/parity_boundary/` — vernier's
boundary-IoU parity harness. It is not imported by `python/vernier/`,
`crates/`, or any code that ships in the published wheel. `cargo deny`'s
license check is unaffected (the oracle is Python, never linked).

Per **ADR-0010 §"Oracle (E2 + E3)"** this is a vendored, version-pinned
dependency: every quirk vernier reproduces in strict mode is keyed to
the exact commit recorded below, in the same way that pycocotools
parity is keyed to `pycocotools==2.0.11`.

## Provenance

| Field                | Value |
| -------------------- | ----- |
| Upstream repo        | https://github.com/bowenc0221/boundary-iou-api |
| Upstream commit SHA  | `37d25586a677b043ed585f10e5c42d4e80176ea9` |
| Upstream commit date | 2021-04-05 |
| Upstream branch      | `master` |
| Vendored on          | 2026-04-28 |
| Vendored by          | @NoeFontana |
| Modifications        | **None.** Files are verbatim copies of the upstream tree at the pinned SHA. |

The pinned constants live in
[`crates/vernier-core/src/boundary_parity.rs`](../../../../crates/vernier-core/src/boundary_parity.rs)
as `ORACLE_COMMIT_SHA` and `ORACLE_OPENCV_PIN`. A unit test in that
module asserts the SHA recorded here matches the constant — drift
between the two is a build failure.

## License

The upstream is licensed under the **BSD 2-Clause "Simplified" License**
(see [`boundary_iou_api/LICENSE`](boundary_iou_api/LICENSE), preserved
verbatim from upstream `license.txt`). The same file bundles two further
BSD-2-Clause notices the upstream chose to redistribute:

- **COCOAPI** (Copyright © 2014 Piotr Dollar and Tsung-Yi Lin) —
  `boundary_iou/coco_instance_api/{coco,cocoeval}.py` are derivatives
  of pycocotools and require the COCOAPI attribution.
- **LVIS API** (Copyright © 2019 Agrim Gupta and Ross Girshick) — applies
  to the LVIS eval files in upstream; we do not vendor those (see
  inventory below) but preserve the notice as-is, because we do not
  modify license files.

BSD-2-Clause is compatible with vernier's MIT/Apache-2.0 dual license
(more permissive). Redistribution requires preserving the copyright
notice and disclaimer; both are retained in `boundary_iou_api/LICENSE`,
and this file plus the per-file headers in `boundary_iou/` constitute
the documentation-side notice required by clause 2.

## Inventory — what we vendored, and what we did not

Vendored:

```
boundary_iou_api/
├── LICENSE                                            # upstream license.txt, verbatim
├── README.md                                          # upstream README, verbatim
└── boundary_iou/
    ├── __init__.py                                    # empty package marker
    ├── utils/
    │   ├── __init__.py
    │   └── boundary_utils.py                          # mask_to_boundary core
    └── coco_instance_api/
        ├── __init__.py
        ├── coco.py                                    # COCO loader (extends pycocotools)
        └── cocoeval.py                                # COCOeval with iouType="boundary"
```

Skipped:

| Upstream path | Why skipped |
| ------------- | ----------- |
| `boundary_iou/cityscapes_instance_api/` | Cityscapes parity is out of scope for vernier v0.1 (see ADR-0010 §"Performance baseline"). |
| `boundary_iou/cityscapes_panoptic_api/` | Panoptic parity is out of scope. |
| `boundary_iou/coco_panoptic_api/` | Panoptic parity is out of scope. |
| `boundary_iou/lvis_instance_api/` | LVIS dropped from Phase 2 (see ADR-0010 §"Decision drivers" — not a primary deliverable). |
| `tools/` | Demo evaluation scripts — not needed; the harness will exercise `cocoeval.COCOeval` directly. |
| `setup.py` | We do not pip-install the oracle; it is imported from this checked-in path. |
| `.gitignore` | Vernier's root `.gitignore` covers the same patterns. |

Skipping a subtree is a vendoring discipline choice, not a license
question — BSD-2-Clause permits redistribution of subsets. If a
follow-up adds a Cityscapes/Panoptic/LVIS parity oracle, the
corresponding upstream subtree is added at the same pinned SHA and
this inventory is updated.

## Runtime dependency: OpenCV

The upstream's `boundary_utils.mask_to_boundary` calls `cv2.erode` and
`cv2.copyMakeBorder` from `opencv-python`. The oracle therefore requires
`opencv-python` at test time. Pinned in `pyproject.toml` under
`[dependency-groups].dev`:

```
opencv-python==4.10.0.84
```

Both `cv2.erode` and `cv2.copyMakeBorder` have been API-stable since
OpenCV 2.x, so the pin is conservative against the test surface — the
risk is OpenCV-side build/wheel breakage, not behavior drift.
`ORACLE_OPENCV_PIN` in `boundary_parity.rs` mirrors this string for the
Rust side. If the pin changes, both must change atomically.

## Fork plan

Per **ADR-0010 §"Oracle (E2 + E3)"** the upstream is unmaintained
("Beta version" since 2021; last commit 2021-04-05). The plan, decided
in advance so it is not a panic decision later:

1. If the pinned commit becomes uninstallable or behaviorally broken
   (CVE in a transitive dep, OpenCV API break that reaches into
   `cv2.erode`'s signature, or a vendored-NumPy issue), we fork to
   `NoeFontana/boundary-iou-api-vendored` from the same SHA.
2. The fork is the *new* upstream. This `VENDORING.md` is updated in
   place: the upstream URL points at the fork, the pinned SHA reflects
   the fork's HEAD, and the modification field documents what changed.
3. The fork's commit history retains the original `bowenc0221` lineage
   so the BSD-2-Clause attribution is preserved by construction.
4. The fork is itself frozen at a SHA in this file — we do not track
   a moving branch.

The NumPy reference at `tests/python/parity_boundary/numpy_reference.py`
(ADR-0010 §"Oracle (E3 sidecar)") is the spec-side oracle that lets us
distinguish "vernier diverges from upstream" from "vernier and upstream
both diverge from the spec". It is implemented from this ADR rather
than from the upstream code; if the fork plan above ever fires, the
NumPy reference is the diagnostic that determines whether the upstream
break was a behavior change or a packaging change.

## How to refresh

This is a deliberate, ADR-level operation — not a routine update. To
move to a different upstream commit:

1. Open a `proposed` ADR titled "Refresh boundary-IoU oracle to
   `<short-sha>`" describing what changed in upstream and why we are
   moving forward. Cite the disposition impact on
   `docs/engineering/boundary-iou-quirks.md`.
2. Run the diff: `git fetch upstream && git diff <old-sha>..<new-sha>
   -- boundary_iou/utils boundary_iou/coco_instance_api`. Anything
   beyond cosmetic must be reflected in either the quirks survey or
   the ADR.
3. Re-fetch the eight files (LICENSE, README.md, six Python files —
   the `gh api` invocation in this repo's PR #41-or-equivalent is the
   canonical recipe).
4. Update the table at the top of this file: SHA, commit date,
   vendored-on date.
5. Update `ORACLE_COMMIT_SHA` in `crates/vernier-core/src/boundary_parity.rs`
   in the same commit.
6. Re-run the parity harness on the full fixture corpus.
   Differential output is the regression signal.

The "no modifications" invariant is structural — it is what makes the
SHA pin a parity claim rather than a snapshot. If a fix is needed, it
goes upstream-or-fork, not into the vendored tree.
