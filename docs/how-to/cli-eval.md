# How to evaluate with `vernier eval`

Recipe-style guide for the most common shell-driven uses of `vernier-cli` (ADR-0015). Each recipe is one short heading, a shell snippet, and a one-paragraph context note. The flag surface is pinned in ADR-0015 §"Surface"; the JSON output shape is pinned in [`docs/reference/cli-output-schema.md`](../reference/cli-output-schema.md).

The CLI is a sibling of the in-process `Evaluator` / `patch_pycocotools` shim (ADR-0007): same kernel, different entry point. Pick the surface that matches your environment — these recipes assume a shell, no Python interpreter.

## Reproduce the pycocotools summary table

```sh
vernier eval --gt instances_val2017.json --dt detections.json --iou-type bbox
```

The default `--emit` is text and the default `--parity-mode` is `strict`. Together that makes the CLI's stdout byte-equal to what `COCOeval(coco_gt, coco_dt, "bbox").summarize()` writes to stdout via `print()`. This is the canonical parity test — `vernier eval ... > vernier.txt && diff vernier.txt cocoeval.txt` is the verification, not a special incantation. Strict-mode byte-equality is parity-pinned to `pycocotools==2.0.11` (ADR-0002, ADR-0015).

## Store the result as a CI artifact

```sh
vernier eval --gt gt.json --dt dt.json --iou-type bbox --emit json=result.json
jq '.stats[0]' result.json
```

`--emit json=PATH` writes the v1 JSON document atomically to `PATH` and produces no stdout. `.stats[0]` is the headline AP for the bbox / segm / boundary plan; the keypoints plan uses the same `stats` index (also AP). The output is byte-deterministic for byte-equal input — fixed key order, no timestamps, no environment leakage — so `git diff result.json` between two CI runs is empty unless the eval inputs actually changed. ADR-0015 §"Output determinism" pins this contract.

## Capture human and machine output in one run

```sh
vernier eval --gt gt.json --dt dt.json --iou-type bbox \
    --emit text \
    --emit json=result.json
```

`--emit` is repeatable. The eval pipeline runs once; each `--emit` adds one render pass on the borrowed `Summary`, which is cheap compared to matching and accumulation. The text output goes to stdout (so a human reading the CI logs sees the COCO table); `result.json` lands as an archived artifact for downstream tooling. At most one `--emit` per invocation may target stdout — passing two implicit-stdout emits (e.g. `--emit text --emit json` with no `=PATH`) is rejected at parse time with exit code 2.

## Parity-gate against pycocotools in a shell

```sh
vernier eval --gt gt.json --dt dt.json --iou-type bbox > vernier.txt

python - <<'PY' > cocoeval.txt
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
gt = COCO("gt.json")
dt = gt.loadRes("dt.json")
e = COCOeval(gt, dt, iouType="bbox")
e.evaluate(); e.accumulate(); e.summarize()
PY

diff vernier.txt cocoeval.txt
```

Both invocations capture stdout to separate files; `diff` exits 0 only if the bytes match. This is the strict-mode byte-equality property exercised at the binary boundary, complementing the in-process parity harness (`tests/python/parity/test_cli.py::test_strict_text_matches_pycocotools_stdout` — see ADR-0015 §"Parity harness"). For CI, swap the inline heredoc for a checked-in script and pin the pycocotools version explicitly so the oracle is reproducible.

## Evaluate boundary IoU

```sh
vernier eval --gt gt.json --dt dt.json --iou-type boundary --dilation-ratio 0.02
```

`--iou-type boundary` is the entry point for the boundary-IoU subsystem (ADR-0010). `--dilation-ratio` selects the band thickness as a fraction of the image diagonal; the default `0.02` is the bowenc0221 reference value, and the LVIS variant is `0.008`. The flag is kind-coupled: passing `--dilation-ratio` with `--iou-type bbox` (or `segm`, or `keypoints`) is rejected at parse time with exit code 2, not silently ignored. The strict-mode oracle for boundary IoU is the vendored `bowenc0221/boundary-iou-api` rather than pycocotools — see [`docs/engineering/boundary-iou-quirks.md`](../engineering/boundary-iou-quirks.md).

## Evaluate keypoints with custom sigmas

```sh
vernier eval --gt gt.json --dt dt.json --iou-type keypoints --sigmas sigmas.json
```

The `sigmas.json` file is a JSON object keyed by category id (as a string), with each value a list of per-keypoint standard deviations:

```json
{
  "1": [0.026, 0.025, 0.025, 0.035, 0.035, 0.079, 0.079, 0.072, 0.072, 0.062, 0.062, 0.107, 0.107, 0.087, 0.087, 0.089, 0.089],
  "2": [0.030, 0.030, 0.040, 0.040, 0.045, 0.045, 0.080, 0.080, 0.080, 0.080, 0.080, 0.090, 0.090, 0.090, 0.090, 0.090, 0.090]
}
```

Categories absent from `sigmas.json` fall back to the COCO-person 17-keypoint defaults pinned at [`crates/vernier-core/src/similarity/oks.rs`](../../crates/vernier-core/src/similarity/oks.rs) as `COCO_PERSON_SIGMAS`. The list length per category must equal one third of the `keypoints` annotation length for that category; mismatches surface as a typed error with exit code 1. `--sigmas` is kind-coupled to `--iou-type keypoints` — passing it with any other IoU type is rejected at parse time with exit code 2. ADR-0012 §"Decision outcome" pins the per-category sigma map as the F1-corrected disposition.

## Suppress stderr progress messages

```sh
vernier eval --gt gt.json --dt dt.json --iou-type bbox --quiet
```

`--quiet` silences the (currently-empty, but reserved) stderr diagnostic stream. Stdout — the summary itself — is unaffected; `--quiet` and `--emit text` compose without surprises. Errors that abort the run still write a typed message to stderr and exit non-zero regardless of `--quiet`. The CLI does not ship a `--verbose` flag at v0.2; structured logging is a follow-up ADR (ADR-0015 §"What this ADR explicitly does *not* decide").

## Pin the JSON schema version for archived results

```sh
# v0.2: the only shipped schema is "1"; this is what every --emit json writes today.
vernier eval --gt gt.json --dt dt.json --iou-type bbox --emit json=result.json
```

At v0.2 the JSON formatter writes `"version": "1"` and there is no other shipped schema — `--emit json,version=N` is the planned per-formatter knob (ADR-0015 §"Formatter: JSON") for opting into a future schema before the major bump that retires `"1"`. The schema is a SemVer-major contract surface: archived `"version": "1"` documents remain consumable through the next major-release window. If you store eval results long-term, pin both `vernier-cli` (e.g. `cargo install vernier-cli@^0.2`) and the schema version field on read — `jq -e '.version == "1"' result.json` is enough to fail fast when a future toolchain change starts emitting a different shape.

## See also

- [`docs/adr/0015-vernier-cli.md`](../adr/0015-vernier-cli.md) — the CLI surface, exit codes, and stability commitments.
- [`docs/reference/cli-output-schema.md`](../reference/cli-output-schema.md) — JSON output schema, field by field.
- [`docs/reference/coco-summary-stats.md`](../reference/coco-summary-stats.md) — `stats[i]` index → metric mapping.
- [`docs/adr/0007-patch-pycocotools-policy.md`](../adr/0007-patch-pycocotools-policy.md) — the in-process complement of the CLI for users who can run a Python interpreter.
- [`docs/adr/0010-boundary-iou-isolated-subsystem.md`](../adr/0010-boundary-iou-isolated-subsystem.md) — boundary IoU oracle, dilation-ratio semantics.
- [`docs/adr/0012-oks-keypoints-surface.md`](../adr/0012-oks-keypoints-surface.md) — per-category sigmas, kp-canonical `max_dets = [20]`.
