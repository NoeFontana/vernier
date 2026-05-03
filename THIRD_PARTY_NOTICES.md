# Third-party notices

vernier carries verbatim copies of selected third-party reference
implementations, vendored at pinned commit SHAs and used only in the
test harness. None of this code is included in the published wheel
or linked into the Rust binary; all of it is preserved with its
original license text.

For each entry, see the linked `VENDORING.md` for provenance,
modifications policy, and refresh procedure. Adding a new vendored
reference is an ADR-level decision — see
[`docs/engineering/vendoring.md`](docs/engineering/vendoring.md).

## boundary-iou-api

- **Role:** bit-exact parity oracle for boundary-IoU evaluation
  (ADR-0010). Consumed only by `tests/python/parity_boundary/`; not
  imported by `python/vernier/` or any code that ships in the wheel.
- **Path:** [`tests/python/parity_boundary/oracle/boundary_iou_api/`](tests/python/parity_boundary/oracle/boundary_iou_api/)
- **Upstream:** <https://github.com/bowenc0221/boundary-iou-api>
- **Pinned commit:** `37d25586a677b043ed585f10e5c42d4e80176ea9` (2021-04-05)
- **License:** BSD-2-Clause. Copyright © 2021 Bowen Cheng.
  Bundled COCOAPI (© 2014 Piotr Dollar, Tsung-Yi Lin) and LVIS API
  (© 2019 Agrim Gupta, Ross Girshick) notices preserved verbatim
  per the upstream's own redistribution.
- **License text:** [`tests/python/parity_boundary/oracle/boundary_iou_api/LICENSE`](tests/python/parity_boundary/oracle/boundary_iou_api/LICENSE)
- **Vendoring details:** [`tests/python/parity_boundary/oracle/VENDORING.md`](tests/python/parity_boundary/oracle/VENDORING.md)

<!-- Future vendored references append here, same shape. -->
