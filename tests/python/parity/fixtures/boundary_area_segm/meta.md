# boundary_area_segm

Pins **D6** (`docs/engineering/pycocotools-quirks.md`): pycocotools'
`cocoeval.py:251` filters out-of-range annotations with non-strict
inequalities (`g['area'] < aRng[0] or g['area'] > aRng[1]`), so an
annotation whose `area` equals a bucket boundary lands in *both*
adjacent buckets. Buckets are not partitions.

- One image (100x100), one GT polygon, one identical DT polygon.
- Polygon `[[0, 0, 32, 0, 32, 32, 0, 32]]` rasterizes to a 32x32 mask
  with area exactly `1024` — the small/medium boundary.
- GT JSON sets `area: 1024` explicitly so the eval-time filter on the
  GT-side reads the boundary value verbatim.
- DT bbox is also `[0, 0, 32, 32]`; vernier derives DT area from bbox
  (J3 disposition for vernier-side), pycocotools derives it from the
  segmentation mask — both arrive at `1024`.

The GT lands in `small ([0, 1024])`, `medium ([1024, 9216])`, and `all`
buckets, but **not** in `large`. Earlier vernier used strict `>` / `<`
on `AreaRange::contains`, dropping the GT from both small and medium —
this fixture would have failed parity.

Failure here means the area-range filter has regressed back to strict
exclusion, and pycocotools' bucket-overlap behavior is no longer being
mirrored.
