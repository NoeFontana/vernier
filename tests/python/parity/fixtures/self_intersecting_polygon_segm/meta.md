# self_intersecting_polygon_segm

Pins quirk **H3** (`strict`): the rleFrPoly rasterizer is
sensitive to point order. A self-intersecting polygon (a "bowtie")
walks the boundary with Bresenham line drawing across the
self-crossing, so the exact set of filled pixels depends on the
order of vertices and on the 5x supersampling tie-breaks at the
crossing.

- Image 100×100
- One GT bowtie polygon `[[10, 10, 90, 90, 10, 90, 90, 10]]` —
  vertices traced as TL → BR → BL → TR → close, crossing itself
  at the centre `(50, 50)`. Two filled triangles meeting tip-to-tip.
- One identical DT.

A divergence here is the canonical H3 regression signal — the
matched rasterizer outputs differ at the diagonals, not at the
boundary, so it can't be confused with the H4/H5 boundary-clip
fixture.
