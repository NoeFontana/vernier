# LRP-Error vendoring

This directory holds the verbatim upstream of
[kemaloksuz/LRP-Error](https://github.com/kemaloksuz/LRP-Error) used as
the **first-party reference implementation** for vernier's LRP /
oLRP oracle tripwire (`../test_kemaloksuz_tripwire.py`).

The vendor tree is intentionally **NOT** checked into the repo. The
tripwire test skips cleanly when the directory is absent. Re-vendor
when:

- The Oksuz reference receives a bug-fix release that materially
  changes its numerics.
- The pinned commit goes stale (the upstream `master` head and the
  pinned SHA drift on a non-trivial diff and we want to check whether
  vernier needs to follow).

## Disposition

This is a **sanity gate, not a parity contract**. The vendored
implementation is treated as one of two independent witnesses to the
LRP algorithm; when the oracle and the witness disagree on a
constructed fixture, a human investigates and decides which one is
right. The oracle (`../oracle.py`) is the authoritative implementation
for vernier; this directory only feeds the tripwire.

Drift is documented case-by-case; we do not auto-resolve by editing
the oracle to match.

## Commit pin

| Field | Value |
| --- | --- |
| Upstream | https://github.com/kemaloksuz/LRP-Error |
| Pinned SHA | _UNPINNED_ — fill in on first vendoring |
| Pinned tag | _UNTAGGED_ — fill in on first vendoring |
| License | Apache License 2.0 (per upstream `LICENSE`) |
| Vendoring date | _TBD_ — fill in on first vendoring |

When you fill these in, also bump `THIRD_PARTY_NOTICES.md` at the repo
root with the matching attribution.

## Bootstrap recipe

The upstream is a Python package; vendor only what the tripwire
imports (the `LRPError` evaluator class + its module dependencies),
not the model-zoo download scripts. The expected layout below mirrors
the upstream's `LRP-Error/` subdirectory:

```bash
# From the repo root.
TARGET="tests/python/oracle/lrp/vendor/lrp_error"
mkdir -p "$TARGET"

# Pick one of the two recipes below.

# (A) Submodule (preferred if it goes upstream cleanly).
git submodule add https://github.com/kemaloksuz/LRP-Error "$TARGET"
# Pin to the SHA recorded in the table above.
git -C "$TARGET" checkout "$PINNED_SHA"

# (B) Tarball drop (simpler if upstream is a flat repo).
PINNED_SHA="<sha>"
curl -L "https://github.com/kemaloksuz/LRP-Error/archive/${PINNED_SHA}.tar.gz" \
    | tar xz -C "$TARGET" --strip-components=1

# Sanity-check the import path the tripwire uses.
test -f "$TARGET/__init__.py" || (
    echo "Expected $TARGET/__init__.py; the upstream may not declare lrp_error as a package."
    echo "If so, drop an __init__.py here to make the test's import work, or amend the test."
    exit 1
)
```

The tripwire test (`../test_kemaloksuz_tripwire.py`) imports from
`lrp_error` after putting `vendor/` on `sys.path`. If the upstream's
top-level package name differs, edit the test's `_run_kemaloksuz`
helper rather than monkey-patching `sys.path` in production code.

## Refresh recipe

To pull in a newer pinned SHA:

1. Update the table above with the new SHA / tag / date.
2. Re-run the bootstrap recipe with the new pin (delete the existing
   tree first; we do not preserve local edits to the vendor).
3. Run the tripwire (`uv run pytest -m tripwire`); investigate any
   tolerance breaches as new sanity-gate hits, not bugs to silence.
4. Bump the attribution in `THIRD_PARTY_NOTICES.md`.

## Why this lives here and not under `tests/python/parity_*/oracle/`

The `parity_*` oracles (pycocotools, lvis, panopticapi, etc.) are
**parity contracts**: vernier reproduces them in version-pinned
detail. LRP is different: the numpy oracle in `../oracle.py` is the
spec, and the kemaloksuz reference is one *additional* witness. Putting
the vendor tree alongside the parity oracles would invite contributors
to treat it as a parity reference. It is not.
