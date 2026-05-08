# TIDE oracle expected outputs

Hand-computed expected values for the per-fixture parity assertions. One
file per fixture; the `<name>` here matches the directory under
`fixtures/<name>/`. Loaded by both:

- `tests/python/oracle/tide/test_oracle.py` (oracle pinning, tolerance `1e-6`)
- `crates/vernier-core/tests/tide_oracle_parity.rs` (Rust parity, tolerance `1e-9`)

The `_why_ each value is what it is_ documentation lives in the Python
test docstrings — re-derive from there rather than from these files.

## Schema

```json
{
  "baseline_map": <float>,
  "deltas": {"cls": <float>, "loc": <float>, "both": <float>,
             "dupe": <float>, "bkg": <float>, "missed": <float>},
  "delta_all_fp_removed": <float>
}
```

`all_dupe.json` stores the closest f64 to `76/101` and `25/101`
(`repr(76/101)` and `repr(25/101)` in Python); both Rust and Python
parsers round-trip the literals to the same f64 bit pattern.
