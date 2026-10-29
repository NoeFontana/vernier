# Twin check: `twin:coco_val2017_jittered` vs `coco_val2017_jittered_seed0` (bbox)

**FAIL**: 82/84 checks

| check | detail | ok |
|---|---|---|
| total gt | 36781 vs 36781 (+0.0%) | ✅ |
| total dt | 36323 vs 36323 (+0.0%) | ✅ |
| total cells | 16382 vs 16382 (+0.0%) | ✅ |
| total iou_pairs | 203378 vs 203378 (+0.0%) | ✅ |
| total crowd | 446 vs 446 (+0.0%) | ✅ |
| total gt_bytes | 19987840 vs 19698006 (-1.5%) | ✅ |
| total dt_bytes | 17787373 vs 18035484 (+1.4%) | ✅ |
| hist cells_per_image | TV 0.000 | ✅ |
| hist cell_gd | TV 0.000 | ✅ |
| hist category_cells | TV 0.038 | ✅ |
| hist gt_log_area | TV 0.011 | ✅ |
| hist gt_log_aspect | TV 0.010 | ✅ |
| hist gt_fill | TV 0.001 | ✅ |
| hist gt_segmentation | TV 0.000 | ✅ |
| hist gt_polygon_parts | TV 0.000 | ✅ |
| hist gt_polygon_vertices | TV 0.000 | ✅ |
| hist gt_crowd_counts_len | TV 0.002 | ✅ |
| hist dt_best_iou | TV 0.020 | ✅ |
| hist dt_score_matched | TV 0.007 | ✅ |
| hist dt_score_unmatched | TV 0.015 | ✅ |
| hist dt_segmentation | TV 0.000 | ✅ |
| faster-coco-eval accumulate.cpu_ns | 0.0 vs 0.0 ms (-1.1%) | ✅ |
| faster-coco-eval accumulate.peak_rss_bytes | 422.9 vs 412.3 MiB (-2.5%) | ✅ |
| faster-coco-eval accumulate.wall_ns | 0.0 vs 0.0 ms (-0.8%) | ✅ |
| faster-coco-eval evaluate.cpu_ns | 1362.3 vs 1225.4 ms (-10.1%) | ❌ |
| faster-coco-eval evaluate.peak_rss_bytes | 691.4 vs 680.7 MiB (-1.5%) | ✅ |
| faster-coco-eval evaluate.wall_ns | 2747.2 vs 2461.5 ms (-10.4%) | ❌ |
| faster-coco-eval load.cpu_ns | 433.7 vs 430.6 ms (-0.7%) | ✅ |
| faster-coco-eval load.peak_rss_bytes | 232.6 vs 230.6 MiB (-0.9%) | ✅ |
| faster-coco-eval load.wall_ns | 872.1 vs 869.6 ms (-0.3%) | ✅ |
| faster-coco-eval summarize.cpu_ns | 2.5 vs 2.4 ms (-1.9%) | ✅ |
| faster-coco-eval summarize.peak_rss_bytes | 422.9 vs 412.3 MiB (-2.5%) | ✅ |
| faster-coco-eval summarize.wall_ns | 5.4 vs 5.4 ms (-1.0%) | ✅ |
| faster-coco-eval total.cpu_ns | 1797.0 vs 1660.9 ms (-7.6%) | ✅ |
| faster-coco-eval total.peak_rss_bytes | 691.4 vs 680.7 MiB (-1.5%) | ✅ |
| faster-coco-eval total.wall_ns | 3619.6 vs 3331.1 ms (-8.0%) | ✅ |
| hotcoco accumulate.cpu_ns | 125.0 vs 123.4 ms (-1.3%) | ✅ |
| hotcoco accumulate.peak_rss_bytes | 274.3 vs 274.3 MiB (+0.0%) | ✅ |
| hotcoco accumulate.wall_ns | 249.7 vs 247.5 ms (-0.9%) | ✅ |
| hotcoco evaluate.cpu_ns | 175.7 vs 166.0 ms (-5.5%) | ✅ |
| hotcoco evaluate.peak_rss_bytes | 238.5 vs 238.7 MiB (+0.1%) | ✅ |
| hotcoco evaluate.wall_ns | 359.3 vs 332.7 ms (-7.4%) | ✅ |
| hotcoco load.cpu_ns | 286.8 vs 290.0 ms (+1.1%) | ✅ |
| hotcoco load.peak_rss_bytes | 178.3 vs 177.4 MiB (-0.5%) | ✅ |
| hotcoco load.wall_ns | 578.0 vs 581.4 ms (+0.6%) | ✅ |
| hotcoco summarize.cpu_ns | 1.4 vs 1.3 ms (-3.0%) | ✅ |
| hotcoco summarize.peak_rss_bytes | 274.3 vs 274.3 MiB (+0.0%) | ✅ |
| hotcoco summarize.wall_ns | 1.4 vs 4.3 ms (+198.7%) | ✅ |
| hotcoco total.cpu_ns | 590.6 vs 582.8 ms (-1.3%) | ✅ |
| hotcoco total.peak_rss_bytes | 274.3 vs 274.3 MiB (+0.0%) | ✅ |
| hotcoco total.wall_ns | 1181.0 vs 1161.9 ms (-1.6%) | ✅ |
| pycocotools accumulate.cpu_ns | 780.5 vs 769.3 ms (-1.4%) | ✅ |
| pycocotools accumulate.peak_rss_bytes | 576.1 vs 571.5 MiB (-0.8%) | ✅ |
| pycocotools accumulate.wall_ns | 1575.0 vs 1555.3 ms (-1.3%) | ✅ |
| pycocotools evaluate.cpu_ns | 5602.6 vs 5506.7 ms (-1.7%) | ✅ |
| pycocotools evaluate.peak_rss_bytes | 551.9 vs 547.7 MiB (-0.8%) | ✅ |
| pycocotools evaluate.wall_ns | 11363.3 vs 11089.4 ms (-2.4%) | ✅ |
| pycocotools load.cpu_ns | 441.6 vs 432.5 ms (-2.1%) | ✅ |
| pycocotools load.peak_rss_bytes | 230.8 vs 228.8 MiB (-0.9%) | ✅ |
| pycocotools load.wall_ns | 890.0 vs 873.4 ms (-1.9%) | ✅ |
| pycocotools summarize.cpu_ns | 1.4 vs 1.5 ms (+2.6%) | ✅ |
| pycocotools summarize.peak_rss_bytes | 576.2 vs 571.6 MiB (-0.8%) | ✅ |
| pycocotools summarize.wall_ns | 2.9 vs 4.5 ms (+51.9%) | ✅ |
| pycocotools total.cpu_ns | 6839.6 vs 6713.1 ms (-1.8%) | ✅ |
| pycocotools total.peak_rss_bytes | 576.2 vs 571.6 MiB (-0.8%) | ✅ |
| pycocotools total.wall_ns | 13861.4 vs 13518.0 ms (-2.5%) | ✅ |
| vernier accumulate.cpu_ns | 41.4 vs 42.0 ms (+1.3%) | ✅ |
| vernier accumulate.peak_rss_bytes | 165.5 vs 165.8 MiB (+0.2%) | ✅ |
| vernier accumulate.wall_ns | 83.5 vs 85.3 ms (+2.2%) | ✅ |
| vernier evaluate.cpu_ns | 180.1 vs 185.5 ms (+3.0%) | ✅ |
| vernier evaluate.peak_rss_bytes | 154.4 vs 154.5 MiB (+0.1%) | ✅ |
| vernier evaluate.wall_ns | 365.4 vs 376.8 ms (+3.1%) | ✅ |
| vernier load.cpu_ns | 22.6 vs 22.5 ms (-0.2%) | ✅ |
| vernier load.peak_rss_bytes | 112.3 vs 112.3 MiB (-0.0%) | ✅ |
| vernier load.wall_ns | 45.0 vs 46.6 ms (+3.4%) | ✅ |
| vernier summarize.cpu_ns | 0.8 vs 0.8 ms (+4.9%) | ✅ |
| vernier summarize.peak_rss_bytes | 165.5 vs 165.8 MiB (+0.2%) | ✅ |
| vernier summarize.wall_ns | 0.8 vs 0.8 ms (+2.8%) | ✅ |
| vernier total.cpu_ns | 245.0 vs 250.8 ms (+2.4%) | ✅ |
| vernier total.peak_rss_bytes | 165.5 vs 165.8 MiB (+0.2%) | ✅ |
| vernier total.wall_ns | 496.5 vs 511.0 ms (+2.9%) | ✅ |
| ranking faster-coco-eval/vernier | 7.33x vs 6.62x (-9.7%) | ✅ |
| ranking hotcoco/vernier | 2.41x vs 2.32x (-3.6%) | ✅ |
| ranking pycocotools/vernier | 27.92x vs 26.76x (-4.1%) | ✅ |

Results: `bench/results/ad49243d651f/a6ada888c850/instance/coco_val2017_jittered_seed0/bbox` vs `bench/results/ad49243d651f/a6ada888c850/instance/twin_coco_val2017_jittered_s0_61e0427341e6/bbox`
