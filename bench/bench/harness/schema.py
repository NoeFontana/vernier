"""Schema v2 for bench-harness IPC and persisted results (ADR-0017 §F2,
extended by ADR-0033 across paradigms).

Two distinct shapes share one ``schema_version`` line:

- ``RunnerRepOutput`` — what a runner subprocess writes to its
  ``--output`` JSON. One per (impl, rep) call. The orchestrator owns
  the run-level identity (git sha, machine fingerprint, mode, run seed,
  rep index) and stitches it together with the runner's per-rep fields.
- ``BenchResult`` — what the orchestrator persists at
  ``results/<git-sha>/<machine-fp>/<paradigm>/<workload>/<metric>/<impl>.json``
  (paradigm segment per ADR-0033). One per (impl) over the whole run.

Both serialize with ``extra="forbid"``: a stray field is a bug, not
forward-compat. The migration reader (see ``migrations/``) sets
``extra="ignore"`` so v1 code can still read a v2 file's known fields.

Schema version history:

- **v1** — flat detection-only: ``iou_type``, ``tensor_path``,
  ``tensor_sha256``. No paradigm field. Read via the v1→v2 compat shim
  in ``migrations.v1_to_v2``.
- **v2** — paradigm-aware: adds ``paradigm`` (required), generalizes
  artifact handling from a single tensor to ``artifact_paths`` /
  ``artifact_sha256`` dicts. Detection runners populate
  ``{"tensor": "vernier.npy"}``; panoptic / streaming populate
  multi-artifact dicts.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

# Detection iou-types. ADR-0017 invariant; ADR-0033 keeps the literal
# but generalizes the schema's ``metric`` slot — bbox/segm/keypoints/
# boundary remain valid, plus pq (panoptic), miou (semantic), and the
# streaming throughput/p99/rss family.
IouType = Literal["bbox", "segm", "keypoints", "boundary"]

# Per-paradigm metric discriminator. Stored under the result-store path
# segment that v1 called ``<iou>``; v2 generalizes it. Open per-paradigm
# (e.g., streaming has multiple metrics) — the literal lists the
# canonical names B-streams populate.
Metric = Literal[
    # instance
    "bbox",
    "segm",
    "keypoints",
    "boundary",
    # panoptic
    "pq",
    # semantic
    "miou",
    # streaming (B3 populates per ADR-0033). Each cell name doubles as
    # the path segment under ``bench/results/<sha>/<fp>/streaming/<workload>/<metric>/``
    # and is therefore a closed-world Literal. ``vs_naive`` is the
    # ``naive_python`` baseline cell; ``dlpack`` is the JSON-vs-array
    # ingest cell per ADR-0030.
    "throughput",
    "p99",
    "rss",
    "vs_naive",
    "dlpack",
]

# Paradigm discriminator for both the workload tagged union (per
# ADR-0033) and the persisted ``BenchResult``. The four-way closed-
# world union mirrors ADR-0029's per-paradigm namespace split and
# ADR-0032's ``WireEnvelopeBody`` pattern.
Paradigm = Literal["instance", "panoptic", "semantic", "streaming"]

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

    schema_version: Literal[2] = 2
    paradigm: Paradigm
    impl: str
    impl_version: str
    iou_type: IouType
    workload_id: str
    # Stages keys are open by convention (load / evaluate / accumulate /
    # summarize / total); a runner that splits one of these into
    # sub-stages adds a new key — readers join on ``total``.
    stages: dict[str, StageTimings]
    summary_stats: dict[str, float]
    # Generalized from v1's single-tensor pair to a name→path /
    # name→sha map. Detection populates ``{"tensor": "vernier.npy"}``;
    # panoptic populates ``{"snapshot": "panoptic.json", "per_class":
    # "per_class.npy"}``; streaming populates ``{"summary":
    # "stats.json", "rss_curve": "rss_curve.json"}``. Keys are open
    # per-paradigm; readers MUST tolerate unknown keys.
    artifact_paths: dict[str, str]
    artifact_sha256: dict[str, str]
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


class MemoryAggregation(BaseModel):
    """Across-rep ``ru_maxrss`` summary (warmup reps excluded)."""

    model_config = ConfigDict(extra="forbid")

    median_bytes: int
    min_bytes: int
    max_bytes: int


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
    # Backfilled lazily — older v1 result files written before RAM
    # aggregation landed parse with this as ``None``.
    memory: MemoryAggregation | None = None


class BenchResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[2] = 2
    paradigm: Paradigm

    impl: str
    impl_version: str
    iou_type: IouType
    workload_id: str

    git_sha: str
    machine_fingerprint: str
    # Human-readable provenance carried alongside the fingerprint. Both
    # are optional so result files written before these fields landed
    # still parse. **Not** folded into the fingerprint hash — see
    # ``MachineInputs`` (re-bucketing existing results is not free).
    cpu_model: str | None = None
    cpu_arch: str | None = None
    harness_version: str
    mode: Mode
    run_seed: int

    reps_count: int
    warmup_discarded: int
    reps: list[RepResult]
    aggregation: Aggregation | None = None

    # See ``RunnerRepOutput.artifact_paths`` for the per-paradigm key
    # conventions. v1 results lift their single ``tensor_path`` /
    # ``tensor_sha256`` into the ``"tensor"`` slot via the v1→v2
    # compat shim in ``migrations.v1_to_v2``.
    artifact_paths: dict[str, str]
    artifact_sha256: dict[str, str]
    warnings: list[BenchWarning] = []
