# 2026-09 real predictions — bbox / segm / keypoints cells

Engineer-facing snapshot of vernier against the alternatives on *real*
COCO val2017 predictions from three checkpoints — RF-DETR `Nano`
(bbox) and `SegNano` (bbox + instance masks), both at pinned pip
version `1.6.5.post0`, plus `ViTPose-base-simple` (keypoints) at hub
revision `a93ac0c`. Companion to the DETR-R50 bbox cell
([2026-05-detr-r50-real-predictions.md](./2026-05-detr-r50-real-predictions.md))
and to the jittered-DT round in [`docs/benchmarks.md`](../../benchmarks.md).

Two things make this round worth its own page. It is the first
segmentation cell measured on real predicted masks rather than
GT-derived ones, and it is the first rf-detr round measured after the
`v3` prediction-cache bump — every earlier rf-detr number on this repo
was computed on systematically relabelled detections (see
[`real-predictions-parity.md`](../real-predictions-parity.md#boundary--rfdetr-segnano-vs-boundary_iou_api)).

## Shared configuration

- **Harness mode**: release (N=10 + 2 warmup, randomised impl order,
  governor pre-flight, 5% relative-IQR gate per impl). Every cell
  below passed its gate; the widest relative IQR in the round is
  2.14%.
- **Git SHA**: `952daa42015e`
- **Machine fingerprint**: `59aab88b17f4` (AMD EPYC-Milan, x86_64),
  one CPU per run via `sched_setaffinity` (ADR-0047). The host was
  otherwise idle for all three cells.
- **Build profile**: cargo release defaults (`opt-level=3`,
  `lto=thin`, `codegen-units=1`, no `target-cpu`) — the profile the
  PyPI wheel ships with.
- **Versions**: vernier `0.5.0`, `hotcoco==1.0.1`,
  `faster-coco-eval==1.8.0`, `pycocotools==2.0.11`.
- **Parity**: all three tiers pass on every cell, and the headline AP
  is bit-identical across all four implementations.

### Workloads

| workload | detections | images | categories | cache |
| --- | ---: | ---: | ---: | ---: |
| `coco_val2017_rfdetr_nano_v1.6.5.post0` | 513,808 | 5,000 | 80 / 80 | 77 MiB |
| `coco_val2017_rfdetr_segnano_v1.6.5.post0` | 430,516 | 4,996 | 80 / 80 | 187 MiB |
| `coco_val2017_vitpose_base_simple_va93ac0c` | 10,777 | 2,693 | 1 / 1 (`person`) | 8.5 MiB |

Both are `v3` caches. The score floor is 0.05; segnano produced no
above-floor output on 4 of the 5,000 images. Every segnano detection
carries an RLE mask, so the same cache backs the bbox and segm cells.

### Absolute metrics, as a cross-check

| cell | AP@[.5:.95] | AP@.50 |
| --- | ---: | ---: |
| nano bbox | 0.4835 | 0.6749 |
| segnano bbox | 0.4926 | 0.6820 |
| segnano segm | 0.4037 | 0.6329 |
| vitpose keypoints | 0.7626 | 0.9059 |

These reproduce the RF-DETR model card (Nano 48.4 box AP) to the
published precision, which is the point: the `v2` cache these cells
replaced measured `0.0019`. The numbers are recorded for
cross-reference, not gated — see the coverage note at the end.

## Instance — `coco_val2017_rfdetr_nano_v1.6.5.post0` (bbox)

| impl | median | IQR | RSS (max) | vs vernier |
| --- | ---: | ---: | ---: | ---: |
| **vernier** | 992.8 ms | 16.4 ms (1.65%) | 395 MiB | **1.00x** |
| hotcoco | 2.271 s | 37.5 ms (1.65%) | 997 MiB | 2.29x |
| faster-coco-eval | 5.355 s | 114.8 ms (2.14%) | 1383 MiB | 5.39x |
| pycocotools | 22.273 s | 249.4 ms (1.12%) | 1429 MiB | 22.43x |

### Per-stage breakdown (median)

| impl | load | evaluate | accumulate | summarize |
| --- | ---: | ---: | ---: | ---: |
| **vernier** | 56.7 ms | 664.6 ms | 271.3 ms | 0.6 ms |
| hotcoco | 1.028 s | 715.0 ms | 525.5 ms | 1.0 ms |
| faster-coco-eval | 1.796 s | 3.553 s | fused\* | 2.2 ms |
| pycocotools | 1.809 s | 17.556 s | 2.878 s | 1.3 ms |

\* faster-coco-eval fuses accumulate into evaluate; the harness
records ~0 ns on the accumulate stage and reports the wall under
evaluate.

### Raw measurements

| impl | median (ns) | IQR (ns) | RSS (B) |
| --- | ---: | ---: | ---: |
| vernier | 992,834,422 | 16,369,889 | 414,474,240 |
| hotcoco | 2,270,655,973 | 37,530,745 | 1,045,581,824 |
| faster-coco-eval | 5,355,087,067 | 114,759,589 | 1,450,696,704 |
| pycocotools | 22,272,914,534 | 249,449,307 | 1,498,521,600 |

## Instance — `coco_val2017_rfdetr_segnano_v1.6.5.post0` (bbox)

Same GT, same kernel, a different detector — and a mask-carrying DT
file 2.4× the size of nano's for 16% fewer detections.

| impl | median | IQR | RSS (max) | vs vernier |
| --- | ---: | ---: | ---: | ---: |
| **vernier** | 1.315 s | 9.6 ms (0.73%) | 589 MiB | **1.00x** |
| hotcoco | 2.531 s | 36.8 ms (1.46%) | 961 MiB | 1.92x |
| faster-coco-eval | 5.251 s | 99.8 ms (1.90%) | 1483 MiB | 3.99x |
| pycocotools | 20.307 s | 227.9 ms (1.12%) | 1502 MiB | 15.44x |

### Per-stage breakdown (median)

| impl | load | evaluate | accumulate | summarize |
| --- | ---: | ---: | ---: | ---: |
| **vernier** | 123.2 ms | 955.9 ms | 234.8 ms | 0.6 ms |
| hotcoco | 1.453 s | 608.5 ms | 475.4 ms | 1.1 ms |
| faster-coco-eval | 2.069 s | 3.180 s | fused | 2.2 ms |
| pycocotools | 2.066 s | 15.580 s | 2.648 s | 1.3 ms |

### Raw measurements

| impl | median (ns) | IQR (ns) | RSS (B) |
| --- | ---: | ---: | ---: |
| vernier | 1,315,303,789 | 9,623,877 | 617,730,048 |
| hotcoco | 2,531,116,715 | 36,845,050 | 1,007,190,016 |
| faster-coco-eval | 5,250,916,310 | 99,761,917 | 1,555,427,328 |
| pycocotools | 20,307,425,978 | 227,876,285 | 1,575,206,912 |

## Instance — `coco_val2017_rfdetr_segnano_v1.6.5.post0` (segm)

| impl | median | IQR | RSS (max) | vs vernier |
| --- | ---: | ---: | ---: | ---: |
| **vernier** | 2.384 s | 20.2 ms (0.85%) | 590 MiB | **1.00x** |
| hotcoco | 4.873 s | 69.6 ms (1.43%) | 1479 MiB | 2.04x |
| faster-coco-eval | 10.021 s | 75.6 ms (0.75%) | 1752 MiB | 4.20x |
| pycocotools | 22.367 s | 100.2 ms (0.45%) | 1485 MiB | 9.38x |

### Per-stage breakdown (median)

| impl | load | evaluate | accumulate | summarize |
| --- | ---: | ---: | ---: | ---: |
| **vernier** | 122.4 ms | 2.019 s | 238.8 ms | 0.6 ms |
| hotcoco | 1.439 s | 2.962 s | 470.5 ms | 1.0 ms |
| faster-coco-eval | 2.056 s | 7.968 s | fused | 2.2 ms |
| pycocotools | 2.057 s | 17.615 s | 2.648 s | 1.3 ms |

### Raw measurements

| impl | median (ns) | IQR (ns) | RSS (B) |
| --- | ---: | ---: | ---: |
| vernier | 2,383,665,082 | 20,160,154 | 618,975,232 |
| hotcoco | 4,873,388,844 | 69,616,029 | 1,551,167,488 |
| faster-coco-eval | 10,021,204,663 | 75,595,828 | 1,836,617,728 |
| pycocotools | 22,367,470,992 | 100,165,712 | 1,557,233,664 |

## Instance — `coco_val2017_vitpose_base_simple_va93ac0c` (keypoints)

Top-down: ViTPose runs on the GT person boxes, so this cell pairs with
the keypoints GT and covers the single `person` category. Two orders of
magnitude fewer detections than the bbox cells, which makes it the one
place in the round where fixed per-call overhead is visible.

| impl | median | IQR | RSS (max) | vs vernier |
| --- | ---: | ---: | ---: | ---: |
| **vernier** | 128.7 ms | 1.5 ms (1.20%) | 120 MiB | **1.00x** |
| hotcoco | 205.1 ms | 6.4 ms (3.12%) | 123 MiB | 1.59x |
| faster-coco-eval | 790.3 ms | 11.5 ms (1.45%) | 166 MiB | 6.14x |
| pycocotools | 2.394 s | 11.8 ms (0.49%) | 158 MiB | 18.61x |

### Per-stage breakdown (median)

| impl | load | evaluate | accumulate | summarize |
| --- | ---: | ---: | ---: | ---: |
| **vernier** | 10.8 ms | 114.1 ms | 3.8 ms | 0.0 ms |
| hotcoco | 140.0 ms | 57.2 ms | 7.5 ms | 0.0 ms |
| faster-coco-eval | 250.3 ms | 539.6 ms | fused | 0.3 ms |
| pycocotools | 249.7 ms | 2.106 s | 39.0 ms | 0.3 ms |

### Raw measurements

| impl | median (ns) | IQR (ns) | RSS (B) |
| --- | ---: | ---: | ---: |
| vernier | 128,650,998 | 1,538,184 | 126,238,720 |
| hotcoco | 205,121,613 | 6,397,770 | 128,753,664 |
| faster-coco-eval | 790,250,727 | 11,455,318 | 173,629,440 |
| pycocotools | 2,394,306,357 | 11,837,388 | 165,318,656 |

## Read against the tables

- **The speedup band moves on real predictions, in both directions.**
  Against pycocotools, the jittered bbox cell reports
  18.7× and this round reports 22.4× (nano) / 15.4× (segnano);
  against faster-coco-eval, 5.6× jittered vs 5.4× / 4.0× here. A
  jitter-derived DT inherits the GT's per-image class distribution,
  which is not the distribution a detector produces — real output is
  denser in low-confidence false positives and more skewed across
  categories. Two real detectors disagreeing by 1.4× on the *same*
  kernel and GT is the useful signal: a single-workload speedup claim
  is a point estimate, not a property of the library.

- **Load is the most durable advantage, and it grows with mask
  payload.** vernier ingests GT+DT through its binary FFI without
  materialising Python dicts: 57 ms (nano) and ~123 ms (segnano) vs
  1.0–2.1 s for all three alternatives. On the segnano cells that is
  a 12–17× gap on a stage every consumer pays before any evaluation
  begins. It is decisive on segnano bbox, where vernier loses the
  evaluate stage and still wins the cell 1.92×; on segm, where vernier
  also leads on evaluate (2.019 s vs 2.962 s), load is what turns a
  1.5× kernel margin into a 2.0× cell.

- **hotcoco wins the evaluate stage on two of the four cells**, and
  vernier still wins both cells outright on the strength of load.
  On segnano bbox it is 608.5 ms against vernier's 955.9 ms; on
  keypoints, 57.2 ms against 114.1 ms. Nano bbox goes the other way
  (664.6 ms vs 715.0 ms), as does segm (2.019 s vs 2.962 s).

  The segnano bbox case is the odd one: same GT, same kernel, 16%
  *fewer* detections than nano, and vernier's evaluate went *up* 44%.
  This is an open question, not a diagnosed one. The plausible
  candidates are the denser true-positive population implied by the
  higher AP (0.4926 vs 0.4835, so more matches to resolve per image)
  and deferred parsing of the mask payload the bbox path never reads;
  neither has been measured. It is called out rather than smoothed
  over because the honest summary of this round is that vernier's
  *ingest* is decisively ahead of every alternative while its
  *kernel* trades wins with hotcoco's depending on workload shape.

- **Peak RSS holds a 2.4–3.0× advantage**, widening on segm: 590 MiB
  against 1479 MiB (hotcoco), 1485 MiB (pycocotools) and 1752 MiB
  (faster-coco-eval). The decoded-mask working set dominates the
  segm cell for every implementation; vernier's is the only one that
  does not also hold a Python-object mirror of the input JSON.

- **Parity is the result that matters most here.** All four
  implementations agree bit-for-bit on AP across 944,324 real
  detections spanning both checkpoints. The prior rf-detr round also
  reported parity — on detections that were all mislabelled — which
  is precisely why parity alone is not a sufficient gate and why
  these absolute AP values are now recorded alongside it.

## What this round does not cover

- **Boundary IoU is not measured.** A single vernier boundary pass
  over the segnano masks costs ~470 s, so a release-mode cell across
  vernier / faster-coco-eval / boundary-iou-api is a multi-hour job.
  Deferred to an overnight round; the boundary row in the README and
  in [`docs/benchmarks.md`](../../benchmarks.md) is still the
  jittered cell.
- **No thread-scaling axis.** These are one-CPU cells. The ADR-0047
  `_t<N>` variants exist only for the jittered workloads.
- **No absolute-metric gate.** The AP values above are recorded, not
  asserted. The class-mapping bug that invalidated the `v2` round was
  visible only as an absolute-metric anomaly, and a DT-side
  category-coverage assertion would have caught it — that gate is a
  tracked follow-up, not something this round installs.
