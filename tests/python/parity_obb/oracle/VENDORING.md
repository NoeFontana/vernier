# Vendored oracles: oriented-box evaluation (ADR-0063)

Two oracles back the `strict` claims for oriented boxes. They are
treated differently, and the difference is a licensing fact, not a
preference.

| Key | Upstream | Geometry | License | Held how |
| --- | --- | --- | --- | --- |
| `D2` | [`facebookresearch/detectron2`](https://github.com/facebookresearch/detectron2) | `RotatedBox` | Apache-2.0 | **Vendored** under `detectron2/` |
| `DK` | [`CAPTAIN-WHU/DOTA_devkit`](https://github.com/CAPTAIN-WHU/DOTA_devkit) | `Quad` | **none — see below** | **Not vendored**; fetched into a dev cache |

Neither is imported by `python/vernier/`, `crates/`, or anything that
ships in the published wheel. They are consumed only by
`tests/python/parity_obb/`.

## `D2` — vendored verbatim

| Field | Value |
| --- | --- |
| Upstream repo | https://github.com/facebookresearch/detectron2 |
| Upstream commit SHA | `a25898a09d6ee232767647e92c6177fb1c642369` |
| Upstream commit date | 2026-03-16 |
| Upstream branch | `main` |
| Vendored on | 2026-09-19 |
| Vendored by | @NoeFontana |
| Modifications | **None.** Verbatim copies at the pinned SHA. |

| File | SHA-256 |
| --- | --- |
| `detectron2/box_iou_rotated_utils.h` | `8540cf5a4652ce8c1b3e9f98aa277c0920c7778f4356545b73df70547b684a0b` |
| `detectron2/rotated_coco_evaluation.py` | `55816d89c6958b45bea5ecbf6d6e5e007ca5db84e07a7dc1e4a50762e3b1e691` |
| `detectron2/LICENSE` | `ceb78ee847e76f67a6a521a5732dc3373aed07eee003d738a572dc96705f08ff` |

Apache-2.0 is compatible with vernier's MIT/Apache-2.0 dual license.
Clause 4 requires the license text and a statement of changes; the
license is preserved verbatim beside the sources and there are no
changes to state. `THIRD_PARTY_NOTICES.md` carries the notice-side
entry.

The pinned constants live in
[`crates/vernier-geom/src/pinned.rs`](../../../../crates/vernier-geom/src/pinned.rs)
as `D2_COMMIT_SHA` and `D2_UTILS_SHA256`. A unit test in that module
asserts the strings recorded here match the constants — drift between
the two is a build failure.

### Why the header, and why a flag-pinned build

detectron2's rotated IoU is a C++ template instantiated at `float`.
Its bits are a function of its compiler flags: baseline `x86-64` has no
FMA for the compiler to contract `a*b + c` into, while GCC on `aarch64`
fuses by default and would change the last bits of every cross product.
The `strict` claim is therefore keyed to the triple
`(D2, a25898a0, linux-x86_64)`, and the harness that builds this header
pins release-equivalent flags rather than inheriting whatever the
ambient toolchain prefers. ADR-0063 §"Oracles" states the same thing
from the decision side.

Rust never contracts FMA implicitly, so
[`crates/vernier-geom/src/replica/d2.rs`](../../../../crates/vernier-geom/src/replica/d2.rs)
matches by construction; a unit test additionally scans the replica's
own source for `mul_add` so the invariant cannot be edited away.

## `DK` — pinned by hash, never redistributed

**DOTA_devkit carries no license.** There is no `LICENSE` file in the
repository, no license header in any source file, no license statement
in `readme.md` or `setup.py`, and the GitHub API reports
`"license": null`. ADR-0063's oracle table recorded it as MIT; that was
an error, corrected here and in the ADR.

With no license grant there is no right to redistribute, so this
directory holds **no DOTA_devkit bytes**. The oracle is instead
developer-provisioned, exactly like the COCO val2017 and LVIS caches:
`tests/python/parity_obb/oracle/dota_devkit/fetch.py` downloads the
pinned files into the git-ignored dev cache and verifies each SHA-256
before use. Tests that need the DK oracle skip cleanly when the cache
is absent, so a clean checkout still runs green.

| Field | Value |
| --- | --- |
| Upstream repo | https://github.com/CAPTAIN-WHU/DOTA_devkit |
| Upstream commit SHA | `d3f8da45d4091b1dab37d9fbe4d6e6a50928e410` |
| Upstream commit date | 2019-04-26 |
| Upstream branch | `master` |
| License | **none stated** — no redistribution right |

| File | SHA-256 |
| --- | --- |
| `polyiou.cpp` | `ffbe0459419f962ce1695cd4c49beacb97b95ca42381f244da91f5b56dcb301a` |
| `polyiou.h` | `470c332bb8313efd38e4ba92b7ccbaf9717f106e2918eef1ea37946ddd0b5f9b` |
| `dota_evaluation_task1.py` | `c334f2986ba83e368f13f1e36bc93cba88f728ac24e61049ac8a3545c59e346b` |

Reading a public source to reimplement its observable behavior is not
redistribution, and
[`crates/vernier-geom/src/replica/dk.rs`](../../../../crates/vernier-geom/src/replica/dk.rs)
is such a reimplementation: it carries no upstream text, only vernier's
own prose describing what the algorithm does and why each step is
reproduced. The hashes above are what make the description checkable.

### Upstream fork plan

The repository has been unmaintained since 2019 and the licensing gap is
unlikely to close on its own. If the pinned commit becomes unreachable:

1. The SHA-256 table above remains the contract; a developer who already
   has the files can keep running the harness offline.
2. We do **not** fork and relicense — there is nothing to relicense.
3. If the DK oracle becomes permanently unobtainable, the `Quad` strict
   claim is demoted to the vernier-side replica plus its
   property-and-fuzz suite, and ADR-0063's oracle table is amended by a
   superseding ADR that says so. That is a worse claim, and the honest
   move is to record it rather than to keep asserting parity against
   something nobody can run.

## How to refresh

Same discipline as the other oracle directories: a `proposed` ADR
describing what changed upstream and why, a diff of the pinned range, an
update to the tables above, an atomic update of the constants in
`crates/vernier-geom/src/pinned.rs`, and a full re-run of
`tests/python/parity_obb/`.
