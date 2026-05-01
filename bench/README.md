# vernier-bench

Local-only benchmarking harness for vernier. Specified by
[ADR-0017](../docs/adr/0017-local-bench-harness.md). Linux-only.

The harness produces a single artifact answering, for a given commit on a
given machine: *is vernier faster than the baselines, and did the speedup
come at the cost of correctness?* Parity is a side effect of every timing
run, not a separate test pass.

## Quickstart

From the repository root:

```
just bootstrap          # one-time: build vernier wheel
just bench-sync         # sync the harness env and per-impl envs
just bench-run -- --impl vernier --workload smoke --iou bbox
```

The result lands at `bench/results/<git-sha>/<machine-fp>/<workload>/<iou>/<impl>.{json,npy}`.

## Layout

- `bench/` — Python sources for the orchestrator, runners, workloads,
  result schema, and reporting.
- `envs/<impl>/` — one uv-managed venv per implementation. Each has its
  own `pyproject.toml` and `uv.lock`. The orchestrator invokes runners via
  `uv run --directory envs/<impl>` so each subprocess sees exactly one
  pycocotools-flavored package.
- `tests/` — the harness's own pytest suite. Runs in `bench/.venv`, not in
  the repo's root venv. Driven by `just bench-test`.
- `results/`, `profiles/` — gitignored output trees, scoped by git sha and
  machine fingerprint. Cross-machine result aggregation is out of scope.

## Milestones

This package is built up in 6 milestones (see
`/home/dev/.claude/plans/let-s-design-the-plan-moonlit-hickey.md`). The
current milestone scope is annotated at the top of each module.
