# vernier-coco-val-cache

Single source of truth for the COCO val2017 development cache contract
(URL, SHA256, filename, layout, fetch+verify flow). Consumed by:

- `tools/fetch-coco-val.sh` — user-facing bash entry point (a thin
  shim over `python -m coco_val_cache fetch`).
- `bench/bench/workloads/coco_val2017.py` — bench harness's GT loader.
- `tests/python/coco_val_paths.py` — parity tests' cache-root
  convention.

This package is dev-only: not part of the published `vernier` wheel.
It's a `[tool.uv.sources]` path dependency of the root `pyproject.toml`
and `bench/pyproject.toml` (both editable so a code change is picked
up without a re-sync).

Common entry point:

```bash
./tools/fetch-coco-val.sh                    # GT + perfect-DTs
./tools/fetch-coco-val.sh --with-images      # also fetch val2017/ images
```

Pinning a different SHA256 / URL is an ADR-level decision per
[`docs/engineering/coco-val-parity.md`](../../docs/engineering/coco-val-parity.md).
