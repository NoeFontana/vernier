# Vendored oracle: `lvis-dataset/lvis-api` (PyPI `lvis==0.5.3`)

This directory contains a frozen, verbatim copy of the
[`lvis-dataset/lvis-api`](https://github.com/lvis-dataset/lvis-api)
reference implementation of LVIS evaluation
(Gupta, Dollár, Girshick; *LVIS: A Dataset for Large Vocabulary
Instance Segmentation*; CVPR 2019; arXiv:1908.03195).

The pinned commit is **`031ac21f939bcb5f1ca8de2ab8704082e101ff9b`** —
the release point of `lvis==0.5.3` on PyPI (uploaded 2020-06-18). Every
file in `lvis_api/lvis/` is byte-equal to the corresponding file in the
PyPI sdist (see *Provenance — byte-equality* below); the sdist itself
ships no LICENSE, so the BSD-2-Clause notice is sourced from the
upstream GitHub repo at the same SHA.

The oracle is consumed only by `tests/python/parity_lvis/` — vernier's
LVIS parity harness. It is not imported by `python/vernier/`,
`crates/`, or any code that ships in the published wheel. `cargo deny`'s
license check is unaffected (the oracle is Python, never linked).

Per **ADR-0026 §"Parity strategy"** this is a vendored, version-pinned
dependency: every quirk vernier reproduces in strict mode is keyed to
the exact commit recorded below, in the same way that pycocotools
parity is keyed to `pycocotools==2.0.11` (ADR-0002) and boundary-IoU
parity is keyed to `bowenc0221/boundary-iou-api` at `37d25586` (ADR-0010).

## Provenance

| Field                | Value |
| -------------------- | ----- |
| Upstream repo        | https://github.com/lvis-dataset/lvis-api |
| Upstream commit SHA  | `031ac21f939bcb5f1ca8de2ab8704082e101ff9b` |
| Upstream commit date | 2020-06-18 |
| Upstream branch      | `master` |
| PyPI release         | `lvis==0.5.3` (sdist uploaded 2020-06-18) |
| PyPI sdist sha256    | `4f07153330df342b3161fafb46641ce7c02864113a8ddf0d6ffab6b02407bef0` (wheel) |
| Vendored on          | 2026-05-03 |
| Vendored by          | @NoeFontana |
| Modifications        | **None.** Files are verbatim copies of the upstream tree at the pinned SHA. |

The pinned constants live in
[`crates/vernier-core/src/lvis_parity.rs`](../../../../crates/vernier-core/src/lvis_parity.rs)
as `ORACLE_LVIS_VERSION`, `ORACLE_LVIS_COMMIT_SHA`, and
`ORACLE_PYCOCOTOOLS_PIN`. A unit test in that module asserts the SHA
recorded here matches the constant — drift between the two is a build
failure.

### Provenance — byte-equality

Each `lvis/*.py` file in this tree is byte-equal to the same path inside
the `lvis-0.5.3.tar.gz` PyPI sdist. SHA-256 hashes (verified at vendor
time on 2026-05-03 against a fresh PyPI download):

| Path                  | SHA-256                                                            |
| --------------------- | ------------------------------------------------------------------ |
| `lvis_api/lvis/__init__.py` | `c376f0493e9e13b2cc56a5f2269c276b1a0ee8a35ee9d2e0bfc5883249193130` |
| `lvis_api/lvis/colormap.py` | `b9b1726f2029c72b5482e7aff4125c6a94eea418cab815abf885a6e176c9033d` |
| `lvis_api/lvis/eval.py`     | `e048fb2ba7372b9de12966625391aef5612db9f0b247b0e7aca241e38b34cb46` |
| `lvis_api/lvis/lvis.py`     | `a789a4d730b0ee2fc5579a2c09ee8a37c7e73293caf3b2537eeb38c3ac10c24d` |
| `lvis_api/lvis/results.py`  | `e84e401787c45513bd0eea8eca37ab0998c4a92ce795031ebc9383239661cc0a` |
| `lvis_api/lvis/vis.py`      | `93f36639c26f14fffc8ffd6c7a3d51e5129a97968223559f1e8ef7a2d437d439` |
| `lvis_api/LICENSE`          | `4648c944cf9cacdc4050aa2be0a6efc2883e725b01911dccf392989ef46ebf32` |
| `lvis_api/README.md`        | `88a6e7fee2963bb43b6269d3dad94f57b56cf0f37c54199b0de51dbac9800847` |

The PyPI sdist ships no LICENSE; the BSD-2-Clause notice in this tree
is fetched from the upstream GitHub repo at the same pinned SHA, where
it is the canonical project license file.

## License

The upstream is licensed under the **BSD 2-Clause "Simplified" License**
(see [`lvis_api/LICENSE`](lvis_api/LICENSE), verbatim from upstream
`LICENSE`). Copyright © 2019, Agrim Gupta and Ross Girshick.

BSD-2-Clause is compatible with vernier's MIT/Apache-2.0 dual license
(more permissive). Redistribution requires preserving the copyright
notice and disclaimer; both are retained in `lvis_api/LICENSE`, and
this file constitutes the documentation-side notice required by
clause 2.

## Inventory — what we vendored, and what we did not

Vendored:

```
lvis_api/
├── LICENSE                     # upstream LICENSE, verbatim (BSD-2-Clause)
├── README.md                   # upstream README, verbatim
└── lvis/
    ├── __init__.py             # re-exports LVIS, LVISResults, LVISEval, LVISVis
    ├── colormap.py             # transitively used by vis.py
    ├── eval.py                 # LVISEval orchestrator (referenced as ev: in ADR-0026)
    ├── lvis.py                 # LVIS dataset loader (referenced as lv: in ADR-0026)
    ├── results.py              # LVISResults trim (referenced as rs: in ADR-0026)
    └── vis.py                  # visualisation; imported transitively by __init__.py
```

`lvis/vis.py` and `lvis/colormap.py` are kept because the upstream
`lvis/__init__.py` imports them at package load (`from lvis.vis import
LVISVis`); modifying `__init__.py` to drop the import would violate
the no-modifications invariant. They depend on `matplotlib` at module
load, which is a transitive dependency of `lvis` on PyPI and is
satisfied by `pip install lvis==0.5.3`.

Skipped (from the upstream GitHub tree):

| Upstream path | Why skipped |
| ------------- | ----------- |
| `.github/`, `.gitignore` | vernier's root `.gitignore` covers the same patterns. |
| `data/`, `images/` | empty placeholder directories in the upstream repo. |
| `requirements.txt` | runtime deps are pinned via `pyproject.toml` `[dependency-groups].dev` — `lvis==0.5.3` pulls them in transitively. |
| `setup.py` | we do not pip-install the oracle; it is imported from this checked-in path with a `sys.path` insertion in the harness `conftest.py` (lands in PR-3). |
| `test.py` | upstream's smoke test — superseded by vernier's parity harness which exercises the full surface. |

Skipping a subtree is a vendoring-discipline choice, not a license
question — BSD-2-Clause permits redistribution of subsets.

## Runtime dependency: pycocotools

`lvis==0.5.3` does not declare `pycocotools` as a direct dependency in
its package metadata (verified via `importlib.metadata.requires('lvis')`
on 2026-05-03 — the declared list is `numpy`, `cycler`, `Cython`,
`matplotlib`, `opencv-python`, `kiwisolver`, `pyparsing`, `python-dateutil`,
`six`). It does, however, **import** `pycocotools.mask` at runtime
inside `lvis/lvis.py:15` and uses its RLE codec for ground-truth mask
decoding. The oracle therefore requires `pycocotools` to be installed
when the parity harness runs.

vernier already pins `pycocotools==2.0.11` (ADR-0002) for the
pycocotools parity oracle. Because `lvis==0.5.3` declares no
`pycocotools` constraint, the two co-exist cleanly with no version
conflict — this resolves ADR-0026 appendix open question 6 (AH6).
`ORACLE_PYCOCOTOOLS_PIN` in `lvis_parity.rs` mirrors the
`pycocotools==2.0.11` string so the LVIS parity surface is keyed to
both pins atomically.

If a future `pycocotools` bump (ADR-level) breaks LVIS oracle behavior,
the resolution path is the same as boundary-IoU: either accept the
drift and re-run the LVIS parity harness, or pin to an older `lvis`
release that ships with the desired `pycocotools`. No such drift exists
today.

## Fork plan

Per **ADR-0026 §"Parity strategy"** the upstream is unmaintained as of
this vendoring (last meaningful release 2020-06-18; the only
post-release activity through 2024 is a `np.float` deprecation fix on
`master` that did *not* produce a new PyPI release). The plan, decided
in advance so it is not a panic decision later:

1. If the pinned commit becomes uninstallable or behaviorally broken
   (CVE in a transitive dep, NumPy API break that reaches `np.argsort`
   or `np.linspace`, pycocotools API change that reaches `mask_utils`),
   we fork to `NoeFontana/lvis-api-vendored` from the same SHA.
2. The fork is the *new* upstream. This `VENDORING.md` is updated in
   place: the upstream URL points at the fork, the pinned SHA reflects
   the fork's HEAD, and the modification field documents what changed.
3. The fork's commit history retains the original `lvis-dataset` lineage
   so the BSD-2-Clause attribution is preserved by construction.
4. The fork is itself frozen at a SHA in this file — we do not track
   a moving branch.

Note that the upstream `master` HEAD (`7d7f07de`, Feb 2024) carries the
`np.float` → `float` deprecation fix that PyPI 0.5.3 lacks. Since
vernier pins NumPy ≥ 2.0 (`pyproject.toml`), running this oracle under
NumPy 2.x requires either (a) accepting the conftest-side `np.float =
float` shim that already exists for the boundary oracle (see
`tests/python/parity_boundary/conftest.py`), or (b) advancing the SHA
to a successor commit. Option (a) preserves the parity-claim invariant
("oracle is byte-equal to PyPI 0.5.3"); option (b) re-opens the parity
claim against a different oracle. We default to (a); the conftest patch
lands with the parity harness in PR-3.

## How to refresh

This is a deliberate, ADR-level operation — not a routine update. To
move to a different upstream commit:

1. Open a `proposed` ADR titled "Refresh LVIS oracle to `<short-sha>`"
   describing what changed in upstream and why we are moving forward.
   Cite the disposition impact on the ADR-0026 quirks appendix
   (rows AA–AH).
2. Run the diff: `git fetch upstream && git diff <old-sha>..<new-sha>
   -- lvis/`. Anything beyond cosmetic must be reflected in either the
   quirks appendix or the ADR.
3. Re-fetch the eight files (LICENSE, README.md, six Python files) at
   the new SHA via the `gh api` recipe used at vendor time.
4. Update the table at the top of this file: SHA, commit date,
   vendored-on date, byte-equality hashes.
5. Update `ORACLE_LVIS_COMMIT_SHA` (and any dependent constants) in
   `crates/vernier-core/src/lvis_parity.rs` in the same commit.
6. Re-run the parity harness on the full fixture corpus and the LVIS
   v1 val whole-dataset smoke. Differential output is the regression
   signal.

The "no modifications" invariant is structural — it is what makes the
SHA pin a parity claim rather than a snapshot. If a fix is needed, it
goes upstream-or-fork, not into the vendored tree.
