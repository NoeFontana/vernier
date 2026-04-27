# crowd_overlap_tiebreak

Regression fixture for **ADR-0008** (bbox IoU in `f64` end-to-end). The
exact GT/DT triple lifted from COCO val2017 image 2299 that surfaced
the original divergence.

- GT 1284714: real person, bbox=[394.3, 127.59, 40.72, 92.29], iscrowd=0
- GT 900100002299: crowd region, bbox=[0, 18, 499, 263], iscrowd=1 (encloses GT 1284714)
- DT: identical bbox to GT 1284714, score=0.9

Both candidate matches sit at IoU values that f64 distinguishes by ~8 ULPs
(`0.9999999999999983` for the symmetric self-intersect, `0.9999999999999991`
for the asymmetric crowd path) but f32 collapses to `1.0` exactly vs
`1.0000001`. Pycocotools' "later equal wins" (quirk **B2**) strict-mode
matcher picks the crowd. An f32 kernel widened to f64 produces the wrong
ordering and matches the non-crowd. This fixture asserts strict bit-parity
at exactly that decision point.

Tests:
- E1 / mc:109: asymmetric crowd IoU computed in f64.
- B2 / ce:288: later-equal-wins is bit-sensitive; f64 internals are required
  to reproduce the pycocotools tie-break.
