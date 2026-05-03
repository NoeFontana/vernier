"""TIDE error-decomposition reference oracle.

Per ADR-0021, this oracle is the spec the Rust `error_decomposition`
implementation is validated against. It is pure numpy / Python; no
vernier imports. Correctness is pinned by hand-computed assertions on
six small synthetic fixtures (see `test_oracle.py`).

Scope (Week 1, this PR):
    - bbox kernel only (segm / boundary lift in Week 3)
    - mode="single" only (single t_f for bin assignment)

Out of scope:
    - per-threshold mode (future)
    - keypoints (ADR-0024 deferred)
    - Rust-vs-oracle parity test (Week 2, when Rust ships)
"""
