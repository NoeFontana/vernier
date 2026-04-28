# heterogeneous_dt_segm

Heterogeneous DT list under `iouType="segm"`: DT[0] carries an explicit
`segmentation` polygon, DT[1] carries only `bbox`. Both DTs perfectly
overlap their respective GT polygons.

Pins quirks **J2** (`strict`) and **J6** (`corrected`):

- **J2** (`strict`): pycocotools' `coco.py:341` synthesizes
  `[[x1,y1, x1,y2, x2,y2, x2,y1]]` for DT[1] from its bbox and rasterizes.
  Vernier's strict-mode `evaluate_segm` reproduces that synthesis
  per-entry (no first-entry-decides dispatch), so this fixture is a
  positive-equivalence parity test against pycocotools.
- **J6** (`corrected`): the per-entry dispatch — vernier inspects each
  DT independently for the segm/bbox kind, rather than letting `anns[0]`
  decide for the whole list. The strict-mode harness exercises the
  parity path; corrected-mode rejection is asserted in the
  `j6_heterogeneous_dt_list_*` Rust unit tests and in
  `test_heterogeneous_dt_segm_rejects_in_corrected_mode`.

DT[0] segm covers GT[0] exactly → IoU 1.0 at every threshold. DT[1]'s
synthesized rectangle covers GT[1] exactly → IoU 1.0 at every threshold.
AP collapses to 1.0; if the per-entry dispatch is broken (e.g., DT[1]
silently dropped, or first-entry behavior chosen for the whole list)
the AP drops below 1.0 and the harness flags the divergence.
