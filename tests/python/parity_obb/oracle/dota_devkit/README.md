# DOTA_devkit oracle — not vendored

This directory is intentionally empty of upstream code. DOTA_devkit
states no license, so there is no right to redistribute its sources.
See [`../VENDORING.md`](../VENDORING.md) for the full record.

Provision the oracle locally with:

```bash
uv run python tests/python/parity_obb/oracle/dota_devkit/fetch.py
```

which downloads the three pinned files into the git-ignored dev cache,
verifies each SHA-256, and builds the `polyiou` extension. Tests that
need it skip when the cache is absent.
