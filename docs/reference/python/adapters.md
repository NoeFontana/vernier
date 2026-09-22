# `vernier.adapters`

The ingest route, which turns a training loop's per-image or columnar
state into vernier's evaluation inputs without importing its framework
(ADR-0063), plus the migration shims: a pycocotools `COCOeval` drop-in
and the COCO-JSON normalizers (ADR-0007, ADR-0055).

::: vernier.adapters
    options:
      show_root_heading: false
      show_root_toc_entry: false
