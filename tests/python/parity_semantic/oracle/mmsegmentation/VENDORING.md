# Vendored oracle: `open-mmlab/mmsegmentation`

This directory contains a frozen, verbatim copy of the
[`open-mmlab/mmsegmentation`](https://github.com/open-mmlab/mmsegmentation)
reference implementation of the semantic-segmentation evaluation
metric (`mmseg.evaluation.metrics.IoUMetric`). The `IoUMetric` class
is the de-facto research reference for mIoU / mAcc / mDice / mFscore
on Cityscapes, ADE20K, Pascal VOC, and the broader semantic-eval
ecosystem.

The pinned commit is **`c685fe6767c4cadf6b051983ca6208f1b9d1ccb8`** —
upstream `main` at tag `v1.2.2` (released 2023-12-14). v1.2.2 is the
most recent stable release at the time of vendoring; upstream entered
maintenance mode under the OpenMMLab organization shortly after.
Because `IoUMetric` is the durable parity surface (the mIoU formula
has been stable across mmsegmentation 0.x → 1.x), the SHA pin is
keyed to a release tag rather than a free-floating `main` commit;
this gives the parity claim a citable upstream artifact.

The oracle is consumed only by `tests/python/parity_semantic/` —
vernier's semantic-segmentation parity harness. It is not imported
by `python/vernier/`, `crates/`, or any code that ships in the
published wheel. `cargo deny`'s license check is unaffected (the
oracle is Python, never linked).

Per **ADR-0028 §"Parity strategy"** and **ADR-0036** this is a
vendored, version-pinned dependency: every quirk vernier reproduces
in strict mode against `IoUMetric` is keyed to the exact commit
recorded below, in the same way that pycocotools parity is keyed to
`pycocotools==2.0.11` (ADR-0002), boundary-IoU parity is keyed to
`bowenc0221/boundary-iou-api` at `37d25586` (ADR-0010), LVIS parity
is keyed to `lvis-dataset/lvis-api` at `031ac21f` (ADR-0026), and
panoptic parity is keyed to `cocodataset/panopticapi` at `7bb46555`
(ADR-0025).

## Provenance

| Field                | Value |
| -------------------- | ----- |
| Upstream repo        | https://github.com/open-mmlab/mmsegmentation |
| Upstream commit SHA  | `c685fe6767c4cadf6b051983ca6208f1b9d1ccb8` |
| Upstream commit date | 2023-12-14 |
| Upstream tag         | `v1.2.2` |
| PyPI release         | `mmsegmentation==1.2.2` (2023-12-14) — equivalent bytes for `mmseg/evaluation/metrics/iou_metric.py`, but pip-installing pulls the full ~3 GB transitive (mmcv, mmengine, torch); we vendor the single file instead. See AP5 in `docs/engineering/sem-seg-quirks.md`. |
| Vendored on          | 2026-05-09 |
| Vendored by          | @NoeFontana |
| Modifications        | **None.** `mmseg/evaluation/metrics/iou_metric.py` and `LICENSE` are byte-equal to the upstream tree at the pinned SHA. The empty `mmseg/__init__.py`, `mmseg/evaluation/__init__.py`, and `mmseg/evaluation/metrics/__init__.py` package markers are *our* code (the upstream `__init__.py` files import the rest of mmsegmentation, which would defeat the point of the vendor); `_mmengine_stub.py` and `_verify_against_pip.py` are also our code. The vendored bytes themselves are unmodified. |

The pinned constants live in
[`crates/vernier-semantic/src/parity.rs`](../../../../crates/vernier-semantic/src/parity.rs)
as `ORACLE_MMSEGMENTATION_COMMIT_SHA`. A unit test in that module
asserts the SHA recorded here matches the constant — drift between
the two is a build failure.

### Provenance — byte-equality

Each vendored file is byte-equal to the same path inside the
`open-mmlab/mmsegmentation` GitHub repo at the pinned commit.
SHA-256 hashes (verified at vendor time on 2026-05-09 against
`gh api repos/open-mmlab/mmsegmentation/contents/<path>?ref=<sha>`):

| Path                                          | SHA-256                                                            |
| --------------------------------------------- | ------------------------------------------------------------------ |
| `LICENSE`                                     | `98b89585ea28b480797870476cd82584425432d1badbe01ac2b9b9bc9cbc2a9b` |
| `mmseg/evaluation/metrics/iou_metric.py`      | `59812ad9c7c0ae1277fb53c23715f80c90be5c0801ac986796aa2320345d67a8` |

The empty `__init__.py` files are not in this table — they are not
vendored upstream content, they are package markers we created so
Python can resolve `from mmseg.evaluation.metrics.iou_metric import
IoUMetric` without pulling the upstream `mmseg/__init__.py` (which
imports the rest of the mmsegmentation package).

## License

The upstream is licensed under the **Apache License 2.0** (see
[`LICENSE`](LICENSE), verbatim from upstream). Copyright 2020 The
MMSegmentation Authors. All rights reserved.

Apache-2.0 is compatible with vernier's MIT/Apache-2.0 dual license.
Redistribution requires preserving the copyright notice, the license
text, and any NOTICE file (mmsegmentation does not ship a separate
NOTICE; the copyright header in `LICENSE` is sufficient). All
requirements are satisfied by retaining `LICENSE` verbatim alongside
the vendored file, and this `VENDORING.md` constitutes the
documentation-side notice required by §4 of the license.

The Apache-2.0 patent grant (§3) extends to vernier's contributors
through the inbound Apache-2.0; no additional CLA or per-PR notice is
required.

## Inventory — what we vendored, and what we did not

Vendored:

```
mmsegmentation/
├── LICENSE                                            # upstream LICENSE, verbatim (Apache-2.0)
└── mmseg/
    └── evaluation/
        └── metrics/
            └── iou_metric.py                          # IoUMetric class, verbatim
```

Skipped (from the upstream GitHub tree):

| Upstream path | Why skipped |
| ------------- | ----------- |
| `mmseg/__init__.py` (and parent `__init__.py` files) | Upstream's package init imports the full mmsegmentation package (registry, models, datasets); pulling it would defeat the point of the vendor. We replace with empty package markers. |
| `mmseg/evaluation/metrics/citys_metric.py`, `dice_metric.py`, `depth_metric.py` | Sibling metrics; the parity contract is `IoUMetric` only. The other metrics use the same `BaseMetric` interface and could be vendored later under a separate ADR if a parity claim against them is wanted. |
| `mmseg/evaluation/metrics/__init__.py` (upstream version) | Upstream re-exports all metrics; we only need `IoUMetric`. Replaced with empty marker. |
| `mmseg/datasets/`, `mmseg/models/`, `mmseg/registry.py`, `mmseg/apis/`, etc. | Out of scope. The parity claim is the eval kernel, not data loading or model inference. The single symbol `mmseg.registry.METRICS` that `iou_metric.py` reads is provided by `_mmengine_stub.py`. |
| `requirements/`, `setup.py`, `tools/`, `configs/`, `docs/`, `tests/`, etc. | Build / training / docs infrastructure; not load-bearing for the parity claim. |

Skipping a subtree is a vendoring-discipline choice, not a license
question — Apache-2.0 permits redistribution of subsets.

## Stubbed dependencies

`mmseg.evaluation.metrics.iou_metric` imports the following at module
scope; the parity harness provides minimal replacements in
[`_mmengine_stub.py`](_mmengine_stub.py) (registered into
`sys.modules` by `tests/python/parity_semantic/conftest.py` before
`iou_metric` is imported):

| Upstream symbol                                   | Stubbed | Notes |
| ------------------------------------------------- | ------- | ----- |
| `mmengine.dist.is_main_process`                   | yes     | Returns `True`; parity harness is single-process. |
| `mmengine.evaluator.BaseMetric`                   | yes     | Bare class with `results: list`, `dataset_meta: dict`, no rank-collation. The parity harness sets `dataset_meta` directly. |
| `mmengine.logging.MMLogger`, `print_log`          | yes     | No-op. Parity harness asserts on returned metrics, not log output. |
| `mmengine.utils.mkdir_or_exist`                   | yes     | `os.makedirs(..., exist_ok=True)`. Reachable only via `output_dir`, never set by the harness. |
| `mmseg.registry.METRICS`                          | yes     | Identity-decorator. The parity harness imports `IoUMetric` directly; the registry is not read. |
| `prettytable.PrettyTable`                         | yes     | No-op class with `add_column`, `get_string`. Used only for log formatting in `compute_metrics`. |
| `torch.histc`, `torch.tensor`, `Tensor.float()` etc. | **no — real torch** | `IoUMetric.intersect_and_union` calls `torch.histc(intersect.float(), bins=N, min=0, max=N-1)` for label binning. `torch.histc`'s float-edge bin semantics do not have a bit-exact numpy equivalent for the general case; replacing it would weaken the parity claim. The torch dep is real and version-pinned (see "Runtime dependencies" below). |
| `PIL.Image`                                       | **no — real Pillow** | Reachable only via `output_dir` (PNG export); the parity harness never sets `output_dir`. We keep the real import for surface fidelity at refresh time and because Pillow is already a parity dep (panopticapi, ADR-0025). |

The stubs are deliberately minimal. If a future SHA refresh
introduces new symbols, the import will fail loudly and the stub is
extended in lock-step with the SHA bump (ADR-level, not a routine
update).

## Runtime dependencies

### `torch`

`IoUMetric.intersect_and_union` (line 190 in `iou_metric.py`) calls
`torch.histc(input, bins, min, max)` to bin label tensors into per-class
histograms. The same call in `total_area_to_metrics` constructs
`torch.tensor([...])` for the mFscore branch. **`torch.histc`'s
float-edge binning is the load-bearing kernel** — the rest of the file
is integer indexing and elementwise arithmetic that any tensor
library would implement equivalently.

The torch pin is **`>= 2.4`** (the existing floor in the project's
`[project.optional-dependencies].torch` extra). `torch.histc`'s API
has been stable since PyTorch 1.x; the `>= 2.4` floor matches what
other vernier consumers (rfdetr, the real-models extra) already
require. Bumping the floor is *not* an ADR-level decision unless
upstream changes `histc`'s rounding or boundary behavior — in that
case the parity claim re-grades and the bump goes through ADR-0036's
refresh procedure.

`ORACLE_TORCH_FLOOR` in `crates/vernier-semantic/src/parity.rs`
mirrors this constraint for the Rust side. The parity harness asserts
the installed torch satisfies the floor.

### `Pillow`

Already pinned at `Pillow==12.2.0` in the project's
`[dependency-groups].dev` for the panopticapi parity oracle (ADR-0025).
mmsegmentation's `IoUMetric` imports `PIL.Image` for the output_dir
PNG-export path, never reached by the parity harness. The pin is
shared; no second pin is needed.

### `numpy`

Already a top-level project dependency (`numpy>=2.0` in `[project]`).
`iou_metric.py` uses `np.round`, `np.nanmean`, `np.nan_to_num`; all
stable across NumPy 1.x and 2.x.

## Fork plan

Per **ADR-0036 §"Fork plan"** the upstream is in maintenance mode as
of this vendoring (last release `v1.2.2`, 2023-12-14). The plan,
decided in advance so it is not a panic decision later:

1. If the pinned commit becomes uninstallable or behaviorally broken
   (CVE in a transitive dep, NumPy API break that reaches the
   `np.round` / `np.nanmean` calls in `iou_metric.py`, torch API
   change that reaches `torch.histc`), we fork to
   `NoeFontana/mmsegmentation-vendored` from the same SHA.
2. The fork is the *new* upstream. This `VENDORING.md` is updated in
   place: the upstream URL points at the fork, the pinned SHA reflects
   the fork's HEAD, and the modification field documents what changed.
3. The fork's commit history retains the original `open-mmlab` lineage
   so the Apache-2.0 attribution is preserved by construction.
4. The fork is itself frozen at a SHA in this file — we do not track
   a moving branch.

If `IoUMetric`'s mIoU formula is the *correct* divergence point (a
behavioral fix we want), it goes through ADR-0036's "Refresh
procedure", not the fork plan — the fork is for emergency
unavailability, not for opinion divergence.

## How to refresh

This is a deliberate, ADR-level operation — not a routine update. To
move to a different upstream commit:

1. Open a `proposed` ADR titled "Refresh mmsegmentation oracle to
   `<short-sha>`" describing what changed in upstream and why we are
   moving forward. Cite the disposition impact on the
   [`sem-seg-quirks.md`](../../../../docs/engineering/sem-seg-quirks.md)
   table (rows AI–AP keyed `ms`).
2. Run the diff:
   `git fetch upstream && git diff <old-sha>..<new-sha> -- mmseg/evaluation/metrics/iou_metric.py`.
   Anything beyond cosmetic must be reflected in either the quirks
   appendix or the ADR.
3. Re-fetch the two files (LICENSE, iou_metric.py) at the new SHA via
   the `gh api` recipe used at vendor time:
   ```
   gh api repos/open-mmlab/mmsegmentation/contents/<path>?ref=<sha> --jq .content | base64 -d > <path>
   ```
4. Update the table at the top of this file: SHA, commit date,
   vendored-on date, byte-equality hashes.
5. Update `ORACLE_MMSEGMENTATION_COMMIT_SHA` in
   `crates/vernier-semantic/src/parity.rs` in the same commit.
6. Re-run the parity harness on the full fixture corpus:
   `uv run pytest tests/python/parity_semantic/ -v`.
   Differential output is the regression signal.
7. Run `_verify_against_pip.py` once locally in a separate venv with
   the corresponding `pip install mmsegmentation==<version>`. This
   confirms the vendored bytes + our stubs produce bit-identical
   metrics to a real install. Document the run in this file's commit
   message; do not check `_verify_against_pip.py` into CI (the
   `pip install mmsegmentation` is the ~3 GB transitive we vendor to
   avoid).

The "no modifications" invariant is structural — it is what makes
the SHA pin a parity claim rather than a snapshot. If a fix is
needed, it goes upstream-or-fork, not into the vendored tree.
