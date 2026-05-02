# Boundary-IoU oracle — bench-side wiring

`oracle/` is a **symlink** to the parity-suite vendored checkout at
[`tests/python/parity_boundary/oracle/boundary_iou_api/`](../../../tests/python/parity_boundary/oracle/boundary_iou_api/).
The bench harness does not maintain its own copy: provenance, license,
fork plan, and refresh procedure all live alongside the vendored tree.

The runner in
[`bench/runners/boundary_iou_runner.py`](../../bench/runners/boundary_iou_runner.py)
inserts `oracle/` onto `sys.path` and applies the same three conftest
fixups as `tests/python/parity_boundary/conftest.py` (matplotlib stub,
single-core multiprocessing, `np.float` alias) before importing from
`boundary_iou`.

Linux-only (per ADR-0017 §G1) — symlinks are unproblematic.
