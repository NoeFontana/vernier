# Twin check: `twin:coco_val2017_jittered` vs `coco_val2017_jittered_seed0` (bbox)

**FAIL**: 84/85 checks

| check | detail | ok |
|---|---|---|
| total gt | 36781 vs 36781 (+0.0%) | ✅ |
| total dt | 36323 vs 36323 (+0.0%) | ✅ |
| total cells | 16382 vs 16382 (+0.0%) | ✅ |
| total iou_pairs | 203378 vs 203378 (+0.0%) | ✅ |
| total crowd | 446 vs 446 (+0.0%) | ✅ |
| total gt_bytes | 19987840 vs 19698006 (-1.5%) | ✅ |
| total dt_bytes | 17787373 vs 18035484 (+1.4%) | ✅ |
| hist category_cells | TV 0.038 | ✅ |
| hist cell_gd | TV 0.000 | ✅ |
| hist cells_per_image | TV 0.000 | ✅ |
| hist dt_best_iou | TV 0.020 | ✅ |
| hist dt_score_matched | TV 0.007 | ✅ |
| hist dt_score_unmatched | TV 0.015 | ✅ |
| hist dt_segmentation | TV 0.000 | ✅ |
| hist gt_crowd_counts_len | TV 0.002 | ✅ |
| hist gt_fill | TV 0.001 | ✅ |
| hist gt_log_area | TV 0.011 | ✅ |
| hist gt_log_aspect | TV 0.010 | ✅ |
| hist gt_polygon_parts | TV 0.000 | ✅ |
| hist gt_polygon_vertices | TV 0.000 | ✅ |
| hist gt_segmentation | TV 0.000 | ✅ |
| hist id_ranges | TV 0.000 | ✅ |
| faster-coco-eval accumulate.cpu_ns | 0.0 vs 0.0 ms (-2.9%) | ✅ |
| faster-coco-eval accumulate.peak_rss_bytes | 422.8 vs 412.2 MiB (-2.5%) | ✅ |
| faster-coco-eval accumulate.wall_ns | 0.0 vs 0.0 ms (-3.7%) | ✅ |
| faster-coco-eval evaluate.cpu_ns | 1311.7 vs 1181.1 ms (-10.0%) | ✅ |
| faster-coco-eval evaluate.peak_rss_bytes | 691.2 vs 680.6 MiB (-1.5%) | ✅ |
| faster-coco-eval evaluate.wall_ns | 2654.6 vs 2366.1 ms (-10.9%) | ❌ |
| faster-coco-eval load.cpu_ns | 430.1 vs 427.8 ms (-0.5%) | ✅ |
| faster-coco-eval load.peak_rss_bytes | 232.6 vs 230.5 MiB (-0.9%) | ✅ |
| faster-coco-eval load.wall_ns | 874.3 vs 846.1 ms (-3.2%) | ✅ |
| faster-coco-eval summarize.cpu_ns | 2.4 vs 2.4 ms (-1.4%) | ✅ |
| faster-coco-eval summarize.peak_rss_bytes | 422.8 vs 412.2 MiB (-2.5%) | ✅ |
| faster-coco-eval summarize.wall_ns | 5.4 vs 5.4 ms (-0.7%) | ✅ |
| faster-coco-eval total.cpu_ns | 1744.4 vs 1610.8 ms (-7.7%) | ✅ |
| faster-coco-eval total.peak_rss_bytes | 691.2 vs 680.6 MiB (-1.5%) | ✅ |
| faster-coco-eval total.wall_ns | 3533.7 vs 3213.9 ms (-9.0%) | ✅ |
| hotcoco accumulate.cpu_ns | 123.3 vs 123.0 ms (-0.3%) | ✅ |
| hotcoco accumulate.peak_rss_bytes | 274.4 vs 274.5 MiB (+0.1%) | ✅ |
| hotcoco accumulate.wall_ns | 250.8 vs 243.9 ms (-2.8%) | ✅ |
| hotcoco evaluate.cpu_ns | 171.3 vs 164.8 ms (-3.8%) | ✅ |
| hotcoco evaluate.peak_rss_bytes | 238.6 vs 238.9 MiB (+0.1%) | ✅ |
| hotcoco evaluate.wall_ns | 351.6 vs 334.6 ms (-4.8%) | ✅ |
| hotcoco load.cpu_ns | 284.6 vs 287.0 ms (+0.8%) | ✅ |
| hotcoco load.peak_rss_bytes | 178.3 vs 177.6 MiB (-0.4%) | ✅ |
| hotcoco load.wall_ns | 575.4 vs 572.3 ms (-0.5%) | ✅ |
| hotcoco summarize.cpu_ns | 1.4 vs 1.3 ms (-4.9%) | ✅ |
| hotcoco summarize.peak_rss_bytes | 274.4 vs 274.5 MiB (+0.1%) | ✅ |
| hotcoco summarize.wall_ns | 1.4 vs 4.2 ms (+207.2%) | ✅ |
| hotcoco total.cpu_ns | 580.2 vs 575.9 ms (-0.7%) | ✅ |
| hotcoco total.peak_rss_bytes | 274.4 vs 274.5 MiB (+0.1%) | ✅ |
| hotcoco total.wall_ns | 1178.0 vs 1148.9 ms (-2.5%) | ✅ |
| pycocotools accumulate.cpu_ns | 764.4 vs 754.5 ms (-1.3%) | ✅ |
| pycocotools accumulate.peak_rss_bytes | 576.1 vs 571.5 MiB (-0.8%) | ✅ |
| pycocotools accumulate.wall_ns | 1529.8 vs 1518.0 ms (-0.8%) | ✅ |
| pycocotools evaluate.cpu_ns | 5103.4 vs 5002.0 ms (-2.0%) | ✅ |
| pycocotools evaluate.peak_rss_bytes | 551.8 vs 547.5 MiB (-0.8%) | ✅ |
| pycocotools evaluate.wall_ns | 10161.4 vs 10030.3 ms (-1.3%) | ✅ |
| pycocotools load.cpu_ns | 431.7 vs 425.8 ms (-1.4%) | ✅ |
| pycocotools load.peak_rss_bytes | 230.7 vs 228.7 MiB (-0.9%) | ✅ |
| pycocotools load.wall_ns | 865.0 vs 854.5 ms (-1.2%) | ✅ |
| pycocotools summarize.cpu_ns | 1.5 vs 1.4 ms (-2.7%) | ✅ |
| pycocotools summarize.peak_rss_bytes | 576.2 vs 571.5 MiB (-0.8%) | ✅ |
| pycocotools summarize.wall_ns | 1.5 vs 4.4 ms (+200.7%) | ✅ |
| pycocotools total.cpu_ns | 6297.7 vs 6194.6 ms (-1.6%) | ✅ |
| pycocotools total.peak_rss_bytes | 576.2 vs 571.5 MiB (-0.8%) | ✅ |
| pycocotools total.wall_ns | 12534.0 vs 12389.9 ms (-1.1%) | ✅ |
| vernier accumulate.cpu_ns | 40.5 vs 41.0 ms (+1.3%) | ✅ |
| vernier accumulate.peak_rss_bytes | 165.7 vs 166.1 MiB (+0.2%) | ✅ |
| vernier accumulate.wall_ns | 82.2 vs 83.0 ms (+1.0%) | ✅ |
| vernier evaluate.cpu_ns | 176.3 vs 180.1 ms (+2.1%) | ✅ |
| vernier evaluate.peak_rss_bytes | 154.6 vs 154.9 MiB (+0.2%) | ✅ |
| vernier evaluate.wall_ns | 352.9 vs 360.6 ms (+2.2%) | ✅ |
| vernier load.cpu_ns | 22.3 vs 22.1 ms (-1.0%) | ✅ |
| vernier load.peak_rss_bytes | 112.5 vs 112.6 MiB (+0.1%) | ✅ |
| vernier load.wall_ns | 46.2 vs 43.1 ms (-6.7%) | ✅ |
| vernier summarize.cpu_ns | 0.8 vs 0.8 ms (+5.4%) | ✅ |
| vernier summarize.peak_rss_bytes | 165.8 vs 166.2 MiB (+0.2%) | ✅ |
| vernier summarize.wall_ns | 0.8 vs 0.8 ms (+5.8%) | ✅ |
| vernier total.cpu_ns | 240.0 vs 243.9 ms (+1.6%) | ✅ |
| vernier total.peak_rss_bytes | 165.8 vs 166.2 MiB (+0.2%) | ✅ |
| vernier total.wall_ns | 479.9 vs 488.0 ms (+1.7%) | ✅ |
| ranking faster-coco-eval/vernier | 7.27x vs 6.60x (-9.1%) | ✅ |
| ranking hotcoco/vernier | 2.42x vs 2.36x (-2.3%) | ✅ |
| ranking pycocotools/vernier | 26.24x vs 25.40x (-3.2%) | ✅ |

Results: `bench/results/3c5c1a406034/a6ada888c850/instance/coco_val2017_jittered_seed0/bbox` vs `bench/results/3c5c1a406034/a6ada888c850/instance/twin_coco_val2017_jittered_s0/bbox`
