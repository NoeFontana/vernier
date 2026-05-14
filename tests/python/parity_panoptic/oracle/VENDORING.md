# Vendored oracles: `cocodataset/panopticapi` + `bowenc0221/boundary-iou-api`

This directory contains two frozen, version-pinned upstream oracles
consumed by the panoptic parity tree. They live side-by-side because
they share the surface (`pq_compute_single_core`) — the second is a
fork-with-modifications of the first that adds the boundary-PQ
``iou_type="boundary"`` code path. Both are test-only, both are
imported from a checked-in path via ``sys.path`` insertion in the
harness modules (see ``boundary_harness.py``).

---

## Oracle A — `cocodataset/panopticapi`

This subtree contains a frozen, verbatim copy of the
[`cocodataset/panopticapi`](https://github.com/cocodataset/panopticapi)
reference implementation of panoptic-quality (PQ) evaluation
(Kirillov, He, Girshick, Rother, Dollár; *Panoptic Segmentation*;
CVPR 2019; arXiv:1801.00868).

The pinned commit is **`7bb4655548f98f3fedc07bf37e9040a992b054b0`** —
upstream `master` HEAD as of 2021-06-17, the most recent commit at the
time of vendoring (the repo has had no upstream activity since).
panopticapi has had only one PyPI release (`panopticapi==0.1`,
uploaded 2018-04-03) which predates this commit by three years; the
sdist on PyPI does not represent the current oracle behavior, and
canonical install is `pip install
git+https://github.com/cocodataset/panopticapi.git` per the upstream
README. Because the pinned commit is *not* a release point, the
parity-claim ground truth is the GitHub tree at the SHA above; SHA-256
hashes for every vendored file are recorded below as the structural
equivalent of a release-tarball checksum.

The oracle is consumed only by `tests/python/parity_panoptic/` —
vernier's panoptic parity harness. It is not imported by
`python/vernier/`, `crates/`, or any code that ships in the published
wheel. `cargo deny`'s license check is unaffected (the oracle is
Python, never linked).

Per **ADR-0025 §"Parity strategy"** this is a vendored, version-pinned
dependency: every quirk vernier reproduces in strict mode is keyed to
the exact commit recorded below, in the same way that pycocotools
parity is keyed to `pycocotools==2.0.11` (ADR-0002), boundary-IoU
parity is keyed to `bowenc0221/boundary-iou-api` at `37d25586`
(ADR-0010), and LVIS parity is keyed to `lvis-dataset/lvis-api` at
`031ac21f` (ADR-0026).

## Provenance

| Field                | Value |
| -------------------- | ----- |
| Upstream repo        | https://github.com/cocodataset/panopticapi |
| Upstream commit SHA  | `7bb4655548f98f3fedc07bf37e9040a992b054b0` |
| Upstream commit date | 2021-06-17 |
| Upstream branch      | `master` |
| PyPI release         | none at this SHA (PyPI `panopticapi==0.1` from 2018 is not the current behavior; canonical install per upstream is `pip install git+https://github.com/cocodataset/panopticapi.git`) |
| Vendored on          | 2026-05-03 |
| Vendored by          | @NoeFontana |
| Modifications        | **None.** Files are verbatim copies of the upstream tree at the pinned SHA. |

The pinned constants live in
[`crates/vernier-panoptic/src/parity.rs`](../../../../crates/vernier-panoptic/src/parity.rs)
as `ORACLE_COMMIT_SHA` and `ORACLE_PILLOW_PIN`. A unit test in that
module asserts the SHA recorded here matches the constant — drift
between the two is a build failure.

### Provenance — byte-equality

Each file in this tree is byte-equal to the same path inside the
`cocodataset/panopticapi` GitHub repo at the pinned commit. SHA-256
hashes (verified at vendor time on 2026-05-03 against a fresh
`git clone --depth=1` of the upstream repo):

| Path                                              | SHA-256                                                            |
| ------------------------------------------------- | ------------------------------------------------------------------ |
| `panopticapi/LICENSE`                             | `f7a02588d6199e6bf040747098928e2eee1e11276478c0a808a6ccaa7cca7f25` |
| `panopticapi/README.md`                           | `ab67daba14c631701398bf1746930c52216214012b410aa5b8b00acdd575d805` |
| `panopticapi/panopticapi/__init__.py`             | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `panopticapi/panopticapi/evaluation.py`           | `04b55dd0128079be1d9f793a01e9ebed95f661cb2240c2ec9e45adcabda85425` |
| `panopticapi/panopticapi/utils.py`                | `826f4d4c5a54fdb64bc1edd1d9e6b508249649716457df96db09b7d23b85a27a` |

Note: the upstream LICENSE file is named `license.txt` (lowercase) on
disk. We rename it to `LICENSE` here to match the convention used by
the boundary-IoU and LVIS oracles in this repo and to make the file
discoverable by tooling that scans for the canonical name. The
contents are byte-equal to the upstream `license.txt`; the SHA above
attests that.

The upstream `panopticapi/__init__.py` is empty (zero bytes); the
SHA-256 of the empty string is the standard
`e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.

## License

The upstream is licensed under the **BSD 2-Clause "Simplified" License**
(see [`panopticapi/LICENSE`](panopticapi/LICENSE), verbatim from
upstream `license.txt`). Copyright © 2018, Alexander Kirillov.

BSD-2-Clause is compatible with vernier's MIT/Apache-2.0 dual license
(more permissive). Redistribution requires preserving the copyright
notice and disclaimer; both are retained in `panopticapi/LICENSE`,
and this file constitutes the documentation-side notice required by
clause 2.

## Inventory — what we vendored, and what we did not

Vendored:

```
panopticapi/
├── LICENSE                          # upstream license.txt, verbatim (BSD-2-Clause)
├── README.md                        # upstream README, verbatim
└── panopticapi/
    ├── __init__.py                  # empty; package marker
    ├── evaluation.py                # PQ orchestrator (referenced as ev: in ADR-0025)
    └── utils.py                     # rgb2id, IdGenerator, get_traceback (referenced as ut: in ADR-0025)
```

Skipped (from the upstream GitHub tree):

| Upstream path | Why skipped |
| ------------- | ----------- |
| `panopticapi/combine_semantic_and_instance_predictions.py` | Pre-evaluation utility for fusing two model heads' outputs; not part of the PQ-eval surface. The parity claim is against the eval kernel, not the data-prep tools. |
| `visualization.py` | Visualization helper; depends on matplotlib at module load. The parity claim does not touch rendering. |
| `cityscapes_gt_converter/`, `converters/` | Data-format conversion utilities targeting datasets out of the v1 scope (per ADR-0025 §"does not decide"). |
| `sample_data/`, `panoptic_coco_categories.json` | Sample inputs / category taxonomy; the parity harness builds its own fixtures with hand-picked categories so it can drive the strict / corrected disposition split. |
| `setup.py` | We do not pip-install the oracle; it is imported from this checked-in path with a `sys.path` insertion in the harness `conftest.py`. |
| `CONVERTERS.md`, `.git/` | Documentation for the skipped converters / version-control metadata. |

Skipping a subtree is a vendoring-discipline choice, not a license
question — BSD-2-Clause permits redistribution of subsets.

## Runtime dependency: Pillow

The upstream's `evaluation.py` imports `PIL.Image as Image` at module
scope (`evaluation.py:14`) and decodes panoptic PNGs with
`np.array(Image.open(path), dtype=np.uint32)` (`evaluation.py:86-89`).
**Pillow's PNG decoder is the load-bearing dep**, not NumPy — Pillow
determines whether RGB is preserved as 3-channel uint8 or implicitly
converted (R2: RGBA silently drops alpha; P / L modes crash mid-eval
because `rgb2id` falls into the scalar branch on a 2-D array). NumPy
is incidental; any 2.x release works.

The oracle therefore requires `Pillow` at test time. Pinned in
`pyproject.toml` under `[dependency-groups].dev`:

```
Pillow==12.2.0
```

Pillow's PNG decode path has been API-stable since Pillow 5.x, so the
pin is conservative against the test surface — the risk is
Pillow-side build/wheel breakage on a future Python release, not
behavior drift. `ORACLE_PILLOW_PIN` in `parity.rs` mirrors this
string for the Rust side. If the pin changes, both must change
atomically.

## Fork plan

Per **ADR-0025 §"Parity strategy"** the upstream is unmaintained as
of this vendoring (last upstream commit 2021-06-17; no PyPI release
representing current behavior). The plan, decided in advance so it
is not a panic decision later:

1. If the pinned commit becomes uninstallable or behaviorally broken
   (CVE in a transitive dep, NumPy API break that reaches the
   `np.unique` / `np.array_split` calls in `evaluation.py`, Pillow
   API change that reaches into `Image.open`), we fork to
   `NoeFontana/panopticapi-vendored` from the same SHA.
2. The fork is the *new* upstream. This `VENDORING.md` is updated in
   place: the upstream URL points at the fork, the pinned SHA reflects
   the fork's HEAD, and the modification field documents what changed.
3. The fork's commit history retains the original `cocodataset` lineage
   so the BSD-2-Clause attribution is preserved by construction.
4. The fork is itself frozen at a SHA in this file — we do not track
   a moving branch.

## How to refresh

This is a deliberate, ADR-level operation — not a routine update. To
move to a different upstream commit:

1. Open a `proposed` ADR titled "Refresh panopticapi oracle to
   `<short-sha>`" describing what changed in upstream and why we are
   moving forward. Cite the disposition impact on the ADR-0025 quirks
   appendix (rows R–Z).
2. Run the diff: `git fetch upstream && git diff <old-sha>..<new-sha>
   -- panopticapi/`. Anything beyond cosmetic must be reflected in
   either the quirks appendix or the ADR.
3. Re-fetch the five files (LICENSE, README.md, three Python files) at
   the new SHA via the `gh api` recipe used at vendor time.
4. Update the table at the top of this file: SHA, commit date,
   vendored-on date, byte-equality hashes.
5. Update `ORACLE_COMMIT_SHA` (and any dependent constants) in
   `crates/vernier-panoptic/src/parity.rs` in the same commit.
6. Re-run the parity harness on the full fixture corpus and (if the
   COCO panoptic val cache is provisioned) the val2017 whole-dataset
   smoke. Differential output is the regression signal.

The "no modifications" invariant is structural — it is what makes
the SHA pin a parity claim rather than a snapshot. If a fix is
needed, it goes upstream-or-fork, not into the vendored tree.

---

## Oracle B — `bowenc0221/boundary-iou-api` (panoptic surface)

This subtree contains a frozen copy of the **panoptic** evaluation
files from
[`bowenc0221/boundary-iou-api`](https://github.com/bowenc0221/boundary-iou-api),
the upstream that adds boundary-IoU composition to the panopticapi
PQ kernel (Cheng, Girshick, Dollár, Schwing, Kirillov; *Boundary IoU:
Improving Object-Centric Image Segmentation Evaluation*; CVPR 2021;
arXiv:2103.16562). The same upstream repo is the reference oracle for
the boundary-IoU instance surface — see
[`tests/python/parity_boundary/oracle/VENDORING.md`](../../parity_boundary/oracle/VENDORING.md);
the pinned SHA is identical.

The oracle is consumed only by the boundary-PQ parity tests in
[`tests/python/parity_panoptic/test_boundary_parity.py`](../test_boundary_parity.py)
via [`tests/python/parity_panoptic/boundary_harness.py`](../boundary_harness.py).
It is not imported by `python/vernier/`, `crates/`, or anything that
ships in the published wheel.

Per **ADR-0010 §"Oracle (E2 + E3)"** this is a vendored, version-
pinned dependency: every quirk vernier reproduces under
``parity_mode="strict", boundary=True`` is keyed to this exact commit,
in the same way that pycocotools parity is keyed to
``pycocotools==2.0.11`` (ADR-0002) and panopticapi parity is keyed to
``7bb46555`` (Oracle A above).

### Provenance

| Field                | Value |
| -------------------- | ----- |
| Upstream repo        | https://github.com/bowenc0221/boundary-iou-api |
| Upstream commit SHA  | `37d25586a677b043ed585f10e5c42d4e80176ea9` |
| Upstream commit date | 2021-04-25 |
| Upstream branch      | `master` |
| PyPI release         | none (upstream is install-from-source via `pip install git+https://...`) |
| License              | BSD 2-Clause "Simplified" (per upstream `license.txt`; copyright © 2021, Bowen Cheng) |
| Vendored on          | 2026-05-14 |
| Vendored by          | @NoeFontana |
| Modifications        | One file (`coco_panoptic_api/evaluation.py`) gets a single trailing one-line comment marker — see below. All other vendored bytes are verbatim. |

The pinned SHA matches the boundary-IoU instance oracle vendored at
``tests/python/parity_boundary/oracle/boundary_iou_api`` — both
oracles ride on the same upstream commit because they are the same
repo (the same `pip install git+...` checkout gives you both). One
SHA, two consumers.

### Provenance — byte-equality

Each file in this subtree is byte-equal to the same path inside the
`bowenc0221/boundary-iou-api` GitHub repo at the pinned commit, with
the single exception noted under "Modifications". SHA-256 hashes
(verified at vendor time on 2026-05-14 against the `gh api` fetch at
the pinned ref):

| Path                                              | SHA-256                                                            | Notes |
| ------------------------------------------------- | ------------------------------------------------------------------ | ----- |
| `boundary_iou_api/__init__.py`                    | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` | Empty package marker (not present upstream as `boundary_iou_api`; introduced here as the import root — see "Modifications"). |
| `boundary_iou_api/coco_panoptic_api/__init__.py`  | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` | Verbatim (empty upstream). |
| `boundary_iou_api/coco_panoptic_api/evaluation.py` (upstream-equal bytes) | `8537746529d216a231c67ec51ba1a5e993c181baccebcadf6b7bca2c51356dc7` | Bytes-as-fetched-from-upstream. |
| `boundary_iou_api/coco_panoptic_api/evaluation.py` (as shipped) | `b10f595f6b67962a372ca7f4d5c9a591ef5eb3521af0010038fa55de68207e83` | After appending a single trailing one-line vendor-marker comment. |
| `boundary_iou_api/utils/__init__.py`              | `2c712855027cbf2cfcc1841f7fb132efc2d25e6bb83e14c69391e572fee86fd2` | **Local shim** (not vendored from upstream); re-exports `mask_to_boundary` from the parity_boundary vendor — see "Modifications". |

The empty `__init__.py` files have SHA-256 of the empty string, which
is the standard `e3b0c44...8b855`.

### Modifications

This is the only oracle subtree in the repo with non-zero modifications.
Two changes, both motivated by import-resolution mechanics and neither
touching evaluation semantics:

1. **`coco_panoptic_api/evaluation.py`**: appended a single trailing
   one-line comment:

   ```text
   # Vendored from bowenc0221/boundary-iou-api @ 37d25586a677b043ed585f10e5c42d4e80176ea9 — do not edit; refresh by re-fetching.
   ```

   All other bytes are verbatim from upstream. The upstream-equal
   SHA-256 (`8537746...`) is recorded alongside the shipped SHA-256
   (`b10f595...`) so a future auditor can confirm the file is the
   pinned upstream content plus the marker line.

   The marker is documentation, not behavior — it does not run because
   it is a comment. We chose the trailing-comment approach (rather than
   a sibling ``PROVENANCE`` file) because the marker is most discoverable
   to anyone reading the file in isolation, including auto-tooling that
   greps for vendor markers.

2. **`utils/__init__.py`** is a **local shim**, not vendored bytes.
   The upstream ``coco_panoptic_api/evaluation.py`` line 24 does
   ``from ..utils import mask_to_boundary``. In the upstream package
   layout the relative parent is ``boundary_iou`` (housing both
   ``coco_panoptic_api`` and ``utils.boundary_utils``); here we keep
   ``coco_panoptic_api`` as the only sub-package under
   ``boundary_iou_api`` and let the sibling ``utils`` package satisfy
   the relative import. The shim is a single line:

   ```python
   from boundary_iou.utils.boundary_utils import mask_to_boundary  # noqa: F401
   ```

   It re-exports the **same upstream function** that the instance-side
   oracle (`tests/python/parity_boundary/oracle/boundary_iou_api/`)
   already vendored at the same pinned SHA. Duplicating
   ``boundary_utils.py`` here would create two copies that could drift;
   re-exporting keeps a single source of truth.

   The boundary harness (`boundary_harness.py`) inserts the
   parity_boundary tree on ``sys.path`` at import time so this shim's
   import resolves.

### Inventory — what we vendored, and what we did not

Vendored:

```
boundary_iou_api/
├── __init__.py                     # empty package marker (local)
├── coco_panoptic_api/
│   ├── __init__.py                 # empty upstream package marker
│   └── evaluation.py               # PQ orchestrator w/ boundary support
└── utils/
    └── __init__.py                 # local shim → parity_boundary vendor
```

Skipped (from the upstream GitHub tree):

| Upstream path | Why skipped |
| ------------- | ----------- |
| `boundary_iou/coco_instance_api/`  | Instance-IoU surface; vendored separately under `tests/python/parity_boundary/oracle/boundary_iou_api/`. Vendoring twice would create drift risk. |
| `boundary_iou/utils/boundary_utils.py` | Already vendored under `parity_boundary`; re-exported here via the local shim above. |
| `tools/`, `setup.py`, `README.md`, `license.txt`, `.gitignore` | Project metadata / install glue / docs; not part of the panoptic-PQ eval surface. The license attestation lives in the parity_boundary tree (`tests/python/parity_boundary/oracle/boundary_iou_api/LICENSE`); this VENDORING.md records the same SHA so the BSD-2-Clause attribution applies. |

Skipping a subtree is a vendoring-discipline choice, not a license
question — BSD-2-Clause permits redistribution of subsets.

### Runtime dependencies

Same as the instance-side oracle:

- ``opencv-python`` (pinned in ``pyproject.toml``; the upstream's
  ``mask_to_boundary`` calls ``cv2.erode`` / ``cv2.copyMakeBorder``).
- ``Pillow`` (pinned; the upstream's ``pq_compute_single_core`` opens
  PNGs via ``PIL.Image.open``).
- The panopticapi oracle's ``panopticapi.utils.get_traceback`` and
  ``panopticapi.utils.rgb2id`` (Oracle A above; same vendoring tree).

### How to refresh

Same shape as Oracle A above, with two additions:

1. Open a `proposed` ADR titled "Refresh boundary-IoU oracles to
   `<short-sha>`" — the SHA refresh must move **both** Oracle B here
   and the instance-side vendor (`tests/python/parity_boundary/oracle/`)
   atomically. They share an upstream commit; if they diverge that's
   itself a parity break.
2. Re-fetch the two upstream files via:

   ```bash
   gh api repos/bowenc0221/boundary-iou-api/contents/boundary_iou/coco_panoptic_api/evaluation.py?ref=<new-sha> \
     --jq '.content' | base64 -d > tests/python/parity_panoptic/oracle/boundary_iou_api/coco_panoptic_api/evaluation.py
   gh api repos/bowenc0221/boundary-iou-api/contents/boundary_iou/coco_panoptic_api/__init__.py?ref=<new-sha> \
     --jq '.content' | base64 -d > tests/python/parity_panoptic/oracle/boundary_iou_api/coco_panoptic_api/__init__.py
   ```
3. Re-append the trailing vendor-marker comment to ``evaluation.py``
   (or update it to mention the new SHA).
4. Update the byte-equality table above with the new SHA-256s for both
   the upstream-equal and the as-shipped versions of ``evaluation.py``.
5. Verify the relative import ``from ..utils import mask_to_boundary``
   still points at our local shim (upstream rarely moves the layout
   but a refresh diff might).

The "no semantic modifications" invariant holds for Oracle B too: the
trailing comment doesn't change behavior, and the utils shim re-exports
the same upstream function. Any future fix to ``evaluation.py`` semantics
goes upstream-or-fork, not into the vendored tree.
