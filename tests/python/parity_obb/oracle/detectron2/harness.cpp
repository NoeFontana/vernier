// Standalone harness around detectron2's vendored rotated-box IoU.
//
// ADR-0063 M0 PR-0.2. The point is to run `single_box_iou_rotated<float>`
// with nothing else in the process: no torch, no Python, no dispatch
// layer -- just the pinned header compiled with release-equivalent
// flags, so the Rust replica in `crates/vernier-geom/src/replica/d2.rs`
// can be diffed against it bit for bit.
//
// The flags matter as much as the source. detectron2's shipped
// linux-x86_64 wheels target baseline x86-64, which has no FMA for the
// compiler to contract `a*b + c` into; building this harness with
// `-march=native` on a machine that does have FMA would change the last
// bits of every cross product and quietly invalidate the comparison.
// `build.sh` beside this file pins the flags; do not build it by hand.
//
// Usage:
//     harness <pairs.bin> <out.bin>
//
// `pairs.bin` is a flat little-endian float32 array of 10*N values:
// box1[5] then box2[5] per record, in `(cx, cy, w, h, angle_degrees)`
// order. `out.bin` receives N float32 IoU values, one per record.
//
// Build: ./build.sh     Verify: the `d2_bridge` Rust test.

#include <cstdio>
#include <cstdlib>
#include <vector>

#include "box_iou_rotated_utils.h"

int main(int argc, char** argv) {
  if (argc != 3) {
    std::fprintf(stderr, "usage: %s <pairs.bin> <out.bin>\n", argv[0]);
    return 2;
  }

  std::FILE* in = std::fopen(argv[1], "rb");
  if (!in) {
    std::fprintf(stderr, "cannot open %s\n", argv[1]);
    return 1;
  }
  std::fseek(in, 0, SEEK_END);
  long bytes = std::ftell(in);
  std::fseek(in, 0, SEEK_SET);
  if (bytes < 0 || bytes % (10 * (long)sizeof(float)) != 0) {
    std::fprintf(stderr, "pairs file is not a whole number of 10-float records\n");
    std::fclose(in);
    return 1;
  }
  size_t n = (size_t)bytes / (10 * sizeof(float));
  std::vector<float> pairs(n * 10);
  if (n > 0 && std::fread(pairs.data(), sizeof(float), n * 10, in) != n * 10) {
    std::fprintf(stderr, "short read\n");
    std::fclose(in);
    return 1;
  }
  std::fclose(in);

  std::vector<float> out(n);
  for (size_t i = 0; i < n; i++) {
    const float* b1 = &pairs[i * 10];
    const float* b2 = &pairs[i * 10 + 5];
    out[i] = detectron2::single_box_iou_rotated<float>(b1, b2);
  }

  std::FILE* fo = std::fopen(argv[2], "wb");
  if (!fo) {
    std::fprintf(stderr, "cannot open %s\n", argv[2]);
    return 1;
  }
  if (n > 0 && std::fwrite(out.data(), sizeof(float), n, fo) != n) {
    std::fprintf(stderr, "short write\n");
    std::fclose(fo);
    return 1;
  }
  std::fclose(fo);
  return 0;
}
