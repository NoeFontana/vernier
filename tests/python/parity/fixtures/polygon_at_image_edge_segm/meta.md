# polygon_at_image_edge_segm

Pins quirks **H4/H5** (`strict`): the rleFrPoly rasterizer's
asymmetric x/y boundary handling — x outside `[0, w-1]` is dropped
via `continue`, y outside `[0, h]` is clamped (H4); x rounds via
`floor`, y rounds via `ceil` after clamp (H5) — must match pycocotools
at exactly the boundary, not one pixel inside or outside.

- Image 100×100
- One GT polygon with vertices at `(W, *)` and `(*, H)` — the bottom-
  right 20×20 corner. The polygon's right edge lies exactly on x=100
  (one past the last valid column index) and bottom edge on y=100.
- One identical DT.

Catches a vernier-mask `Rle::from_polygon` port that off-by-ones the
boundary clip (e.g. clamping x to `W` instead of dropping x>W-1,
treating the y-edge inclusively when pycocotools does not, or vice
versa). On a clean implementation, mask-IoU=1.0 because both sides
rasterize the clipped polygon identically. A divergence here would
surface as a non-zero IoU difference at the highest iouThr, and the
parity harness diffs every cell.
