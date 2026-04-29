"""Phase D fixtures and helpers for `StreamingEvaluator` parity tests.

The shard helper lives in `tests/python/parity/conftest.py` so the
streaming and background suites share a single source of truth.

## Skipped per Phase D scope (follow-up tickets)

- **Order-invariance** — needs the `(score, stream_position)`
  tiebreak which is currently deferred (ADR-0013 §Determinism). Until
  the strict-mode tiebreak lands, two orderings of the same DTs can
  produce a small ULP wobble in tied-score regions, so we cannot pin
  the bit-equality contract.
- **Checkpoint round-trip** — `StreamingEvaluator.checkpoint` /
  `restore` raise `NotImplementedError` in v0; the rkyv-based
  implementation is its own ADR.
- **Crash mid-stream** — needs the Phase E (`BackgroundEvaluator`)
  worker / panic-recovery surface.
- **`snapshot_running` regression** — v0 `snapshot_running` delegates
  to `snapshot()`, so there is nothing to regress against until the
  fast-path lands.
"""
