#!/usr/bin/env python3
"""Provision the DOTA_devkit oracle into the git-ignored dev cache.

DOTA_devkit states no license anywhere -- no ``LICENSE`` file, no
per-file header, no statement in ``readme.md`` or ``setup.py``, and the
GitHub API reports ``"license": null``. With no grant there is no right
to redistribute, so vernier does not vendor its bytes. What vernier
pins instead is the SHA-256 of each file at a fixed commit, which is
enough to make the replica in ``crates/vernier-geom/src/replica/dk.rs``
checkable by anyone who fetches the originals themselves.

Run once per machine::

    uv run python tests/python/parity_obb/oracle/dota_devkit/fetch.py

Files land in ``.cache/dota-devkit/`` and the C++ harness is built
beside them. Every test that needs the DK oracle skips cleanly when the
cache is absent, so a clean checkout still runs green.

See ``tests/python/parity_obb/oracle/VENDORING.md`` and ADR-0063 M0
PR-0.1 for the full position.
"""

from __future__ import annotations

import hashlib
import subprocess
import sys
import urllib.request
from pathlib import Path

#: Pinned upstream commit. Mirrored by ``DK_COMMIT_SHA`` in
#: ``crates/vernier-geom/src/pinned.rs``; the two update atomically.
COMMIT = "d3f8da45d4091b1dab37d9fbe4d6e6a50928e410"

RAW = f"https://raw.githubusercontent.com/CAPTAIN-WHU/DOTA_devkit/{COMMIT}"

#: filename -> SHA-256 at ``COMMIT``. A mismatch is a hard failure: it
#: means the file is not the one the parity claim was made against.
FILES = {
    "polyiou.cpp": "ffbe0459419f962ce1695cd4c49beacb97b95ca42381f244da91f5b56dcb301a",
    "polyiou.h": "470c332bb8313efd38e4ba92b7ccbaf9717f106e2918eef1ea37946ddd0b5f9b",
    "dota_evaluation_task1.py": (
        "c334f2986ba83e368f13f1e36bc93cba88f728ac24e61049ac8a3545c59e346b"
    ),
}

REPO = Path(__file__).resolve().parents[5]
CACHE = REPO / ".cache" / "dota-devkit"

#: Standalone harness. Not part of upstream -- it is vernier's own
#: three-line driver, so it lives here rather than in the cache.
HARNESS_SRC = """\
// Standalone harness around DOTA_devkit's polygon IoU (ADR-0063 M0).
//
// Reads a flat little-endian float64 array of 16*N values (gt quad's
// eight coordinates, then the detection quad's eight) and writes N
// float64 `iou_poly` results. No prefilter is applied here: the
// composed oracle's horizontal-box gate belongs to the caller, so that
// the bridge test can check the raw kernel and the gate separately.
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "polyiou.h"

int main(int argc, char** argv) {
  if (argc != 3) {
    std::fprintf(stderr, "usage: %s <pairs.bin> <out.bin>\\n", argv[0]);
    return 2;
  }
  std::FILE* in = std::fopen(argv[1], "rb");
  if (!in) { std::fprintf(stderr, "cannot open %s\\n", argv[1]); return 1; }
  std::fseek(in, 0, SEEK_END);
  long bytes = std::ftell(in);
  std::fseek(in, 0, SEEK_SET);
  if (bytes < 0 || bytes % (16 * (long)sizeof(double)) != 0) {
    std::fprintf(stderr, "not a whole number of 16-double records\\n");
    std::fclose(in);
    return 1;
  }
  size_t n = (size_t)bytes / (16 * sizeof(double));
  std::vector<double> pairs(n * 16);
  if (n > 0 && std::fread(pairs.data(), sizeof(double), n * 16, in) != n * 16) {
    std::fprintf(stderr, "short read\\n");
    std::fclose(in);
    return 1;
  }
  std::fclose(in);

  std::vector<double> out(n);
  for (size_t i = 0; i < n; i++) {
    std::vector<double> p(pairs.begin() + i * 16, pairs.begin() + i * 16 + 8);
    std::vector<double> q(pairs.begin() + i * 16 + 8, pairs.begin() + i * 16 + 16);
    out[i] = iou_poly(p, q);
  }

  std::FILE* fo = std::fopen(argv[2], "wb");
  if (!fo) { std::fprintf(stderr, "cannot open %s\\n", argv[2]); return 1; }
  if (n > 0 && std::fwrite(out.data(), sizeof(double), n, fo) != n) {
    std::fprintf(stderr, "short write\\n");
    std::fclose(fo);
    return 1;
  }
  std::fclose(fo);
  return 0;
}
"""


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def fetch() -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    for name, want in FILES.items():
        dest = CACHE / name
        if dest.is_file() and sha256(dest) == want:
            print(f"  ok      {name}")
            continue
        url = f"{RAW}/{name}"
        print(f"  fetch   {name}  <- {url}")
        with urllib.request.urlopen(url, timeout=60) as r:
            dest.write_bytes(r.read())
        got = sha256(dest)
        if got != want:
            dest.unlink(missing_ok=True)
            raise SystemExit(
                f"SHA-256 mismatch for {name}:\n"
                f"  expected {want}\n"
                f"  got      {got}\n"
                "The pinned commit is not what was served. This is an "
                "ADR-level event, not a retry."
            )
        print(f"  ok      {name}")


def build() -> None:
    src = CACHE / "dk_harness.cpp"
    src.write_text(HARNESS_SRC, encoding="utf-8")
    out = CACHE / "dk_harness"
    # f64 throughout upstream, so FMA contraction is the only flag that
    # could move a bit. Turn it off explicitly rather than relying on
    # the absence of an ISA feature.
    cmd = [
        "g++",
        "-std=c++14",
        "-O2",
        "-fno-fast-math",
        "-ffp-contract=off",
        f"-I{CACHE}",
        "-o",
        str(out),
        str(src),
        str(CACHE / "polyiou.cpp"),
    ]
    print("  build  ", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print(f"\nHarness: {out}")
    print("Bridge test:")
    print(f"  VERNIER_OBB_DK_HARNESS={out} \\")
    print("    cargo test -p vernier-geom --test dk_bridge --release")


def main() -> int:
    print(f"DOTA_devkit oracle @ {COMMIT[:12]} -> {CACHE}")
    fetch()
    build()
    return 0


if __name__ == "__main__":
    sys.exit(main())
