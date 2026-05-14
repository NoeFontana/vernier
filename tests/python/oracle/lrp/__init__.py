"""LRP / oLRP error decomposition reference oracle.

Mirrors ADR-0021's "numpy oracle as correctness contract" pattern for the
Localization-Recall-Precision (LRP) and optimal-LRP metrics of Oksuz et
al. (ECCV 2018 / TPAMI 2021). The oracle in `oracle.py` is pure numpy /
Python with no vernier imports; its correctness is pinned by
hand-computed assertions on small synthetic fixtures in `test_oracle.py`.

Scope (this PR):
    - Single-IoU-threshold (``tp_threshold``) bin assignment.
    - Per-class oLRP plus the three additive components (Loc, FP, FN).
    - Generic similarity callable (bbox IoU, segm IoU, OKS — caller-supplied).

Out of scope:
    - Rust-vs-oracle parity test fires once `vernier.instance.optimal_lrp`
      exists (the parity test file is shipped pre-skipped against the
      missing symbol).
    - Cross-check against kemaloksuz/LRP-Error is a tripwire (vendored
      under ``vendor/lrp_error/``), not a parity contract.
"""
