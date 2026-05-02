"""Schema v1 for bench-harness IPC and persisted results (ADR-0017 §F2).

Two distinct shapes share one ``schema_version`` line:

- ``RunnerRepOutput`` — what a runner subprocess writes to its
  ``--output`` JSON. One per (impl, rep) call. The orchestrator owns
  the run-level identity (git sha, machine fingerprint, mode, run seed,
  rep index) and stitches it together with the runner's per-rep fields.
- ``BenchResult`` — what the orchestrator persists at
  ``results/<git-sha>/<machine-fp>/<workload>/<iou>/<impl>.json``. One
  per (impl) over the whole run.

Both serialize with ``extra="forbid"``: a stray field is a bug, not
forward-compat. The migration reader (see ``migrations/``) sets
``extra="ignore"`` so v1 code can still read a v2 file's known fields.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

IouType = Literal["bbox", "segm", "keypoints", "boundary"]
Mode = Literal["dev", "release", "profile"]


class StageTimings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    wall_ns: int
    notes: list[str] = []


class BenchWarning(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: str
    message: str


class RunnerRepOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    impl: str
    impl_version: str
    iou_type: IouType
    workload_id: str
    # Stages keys are open by convention (load / evaluate / accumulate /
    # summarize / total); a runner that splits one of these into
    # sub-stages adds a new key — readers join on ``total``.
    stages: dict[str, StageTimings]
    summary_stats: dict[str, float]
    tensor_sha256: str
    warnings: list[BenchWarning] = []


class RepResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rep: int
    warmup: bool
    stages: dict[str, StageTimings]
    summary_stats: dict[str, float]
    ru_maxrss_bytes: int
    parent_wall_ns: int


class StageAggregation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    median_ns: int
    iqr_ns: int
    min_ns: int
    max_ns: int


class IqrGateResult(BaseModel):
    """Outcome of the release-mode IQR-relative-to-median gate."""

    model_config = ConfigDict(extra="forbid")

    stage: str
    relative: float
    threshold: float
    passed: bool


class Aggregation(BaseModel):
    """Across-rep summary. ``None`` in dev mode (one rep)."""

    model_config = ConfigDict(extra="forbid")

    stages: dict[str, StageAggregation]
    iqr_gate: IqrGateResult | None = None


class BenchResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1

    impl: str
    impl_version: str
    iou_type: IouType
    workload_id: str

    git_sha: str
    machine_fingerprint: str
    harness_version: str
    mode: Mode
    run_seed: int

    reps_count: int
    warmup_discarded: int
    reps: list[RepResult]
    aggregation: Aggregation | None = None

    tensor_path: str
    tensor_sha256: str
    warnings: list[BenchWarning] = []
