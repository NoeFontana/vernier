# Twin check: `twin:objects365_val_jittered` vs `objects365_val_jittered_seed0` (bbox)

**FAIL**: 66/68 checks

| check | detail | ok |
|---|---|---|
| total gt | 1240587 vs 1240587 (+0.0%) | ✅ |
| total dt | 1062412 vs 1062412 (+0.0%) | ✅ |
| total cells | 625894 vs 625894 (+0.0%) | ✅ |
| total iou_pairs | 7123464 vs 7123464 (+0.0%) | ✅ |
| total crowd | 178349 vs 178349 (+0.0%) | ✅ |
| total gt_bytes | 269422006 vs 265595908 (-1.4%) | ✅ |
| total dt_bytes | 171098982 vs 169135450 (-1.1%) | ✅ |
| hist cells_per_image | TV 0.000 | ✅ |
| hist cell_gd | TV 0.000 | ✅ |
| hist category_cells | TV 0.020 | ✅ |
| hist gt_log_area | TV 0.009 | ✅ |
| hist gt_log_aspect | TV 0.009 | ✅ |
| hist gt_fill | TV 0.000 | ✅ |
| hist gt_segmentation | TV 0.000 | ✅ |
| hist gt_polygon_parts | TV 0.000 | ✅ |
| hist gt_polygon_vertices | TV 0.000 | ✅ |
| hist gt_crowd_counts_len | TV 0.000 | ✅ |
| hist dt_best_iou | TV 0.010 | ✅ |
| hist dt_score_matched | TV 0.001 | ✅ |
| hist dt_score_unmatched | TV 0.003 | ✅ |
| hist dt_segmentation | TV 0.000 | ✅ |
| hotcoco accumulate.cpu_ns | 3574.4 vs 3485.6 ms (-2.5%) | ✅ |
| hotcoco accumulate.peak_rss_bytes | 6649.4 vs 6652.0 MiB (+0.0%) | ✅ |
| hotcoco accumulate.wall_ns | 7134.7 vs 6977.6 ms (-2.2%) | ✅ |
| hotcoco evaluate.cpu_ns | 6666.3 vs 6665.3 ms (-0.0%) | ✅ |
| hotcoco evaluate.peak_rss_bytes | 6420.6 vs 6421.7 MiB (+0.0%) | ✅ |
| hotcoco evaluate.wall_ns | 13358.3 vs 13330.3 ms (-0.2%) | ✅ |
| hotcoco load.cpu_ns | 6002.4 vs 6110.0 ms (+1.8%) | ✅ |
| hotcoco load.peak_rss_bytes | 4025.5 vs 4026.9 MiB (+0.0%) | ✅ |
| hotcoco load.wall_ns | 12052.5 vs 12201.8 ms (+1.2%) | ✅ |
| hotcoco summarize.cpu_ns | 7.1 vs 7.0 ms (-0.9%) | ✅ |
| hotcoco summarize.peak_rss_bytes | 6649.4 vs 6652.0 MiB (+0.0%) | ✅ |
| hotcoco summarize.wall_ns | 14.7 vs 13.0 ms (-11.1%) | ✅ |
| hotcoco total.cpu_ns | 16242.5 vs 16281.4 ms (+0.2%) | ✅ |
| hotcoco total.peak_rss_bytes | 6649.4 vs 6652.0 MiB (+0.0%) | ✅ |
| hotcoco total.wall_ns | 32511.2 vs 32541.1 ms (+0.1%) | ✅ |
| pycocotools accumulate.cpu_ns | 40641.1 vs 40370.3 ms (-0.7%) | ✅ |
| pycocotools accumulate.peak_rss_bytes | 21854.1 vs 21940.9 MiB (+0.4%) | ✅ |
| pycocotools accumulate.wall_ns | 81297.2 vs 80737.5 ms (-0.7%) | ✅ |
| pycocotools evaluate.cpu_ns | 345533.6 vs 348530.6 ms (+0.9%) | ✅ |
| pycocotools evaluate.peak_rss_bytes | 21724.1 vs 21828.5 MiB (+0.5%) | ✅ |
| pycocotools evaluate.wall_ns | 691854.9 vs 697141.0 ms (+0.8%) | ✅ |
| pycocotools load.cpu_ns | 6713.5 vs 6985.5 ms (+4.1%) | ✅ |
| pycocotools load.peak_rss_bytes | 1920.0 vs 2039.9 MiB (+6.2%) | ✅ |
| pycocotools load.wall_ns | 13352.2 vs 13972.2 ms (+4.6%) | ✅ |
| pycocotools summarize.cpu_ns | 6.8 vs 6.1 ms (-9.8%) | ✅ |
| pycocotools summarize.peak_rss_bytes | 21848.6 vs 21936.8 MiB (+0.4%) | ✅ |
| pycocotools summarize.wall_ns | 12.8 vs 12.1 ms (-5.1%) | ✅ |
| pycocotools total.cpu_ns | 392895.0 vs 395892.6 ms (+0.8%) | ✅ |
| pycocotools total.peak_rss_bytes | 21854.1 vs 21940.9 MiB (+0.4%) | ✅ |
| pycocotools total.wall_ns | 786517.0 vs 791862.9 ms (+0.7%) | ✅ |
| vernier accumulate.cpu_ns | 735.2 vs 834.2 ms (+13.5%) | ❌ |
| vernier accumulate.peak_rss_bytes | 874.3 vs 885.3 MiB (+1.3%) | ✅ |
| vernier accumulate.wall_ns | 1464.2 vs 1666.3 ms (+13.8%) | ❌ |
| vernier evaluate.cpu_ns | 2431.0 vs 2657.1 ms (+9.3%) | ✅ |
| vernier evaluate.peak_rss_bytes | 928.7 vs 931.6 MiB (+0.3%) | ✅ |
| vernier evaluate.wall_ns | 4862.1 vs 5319.7 ms (+9.4%) | ✅ |
| vernier load.cpu_ns | 264.4 vs 262.3 ms (-0.8%) | ✅ |
| vernier load.peak_rss_bytes | 496.4 vs 490.9 MiB (-1.1%) | ✅ |
| vernier load.wall_ns | 527.9 vs 526.2 ms (-0.3%) | ✅ |
| vernier summarize.cpu_ns | 4.1 vs 4.1 ms (-0.3%) | ✅ |
| vernier summarize.peak_rss_bytes | 874.3 vs 885.4 MiB (+1.3%) | ✅ |
| vernier summarize.wall_ns | 7.1 vs 7.2 ms (+1.2%) | ✅ |
| vernier total.cpu_ns | 3437.0 vs 3758.4 ms (+9.4%) | ✅ |
| vernier total.peak_rss_bytes | 928.7 vs 931.6 MiB (+0.3%) | ✅ |
| vernier total.wall_ns | 6868.8 vs 7520.6 ms (+9.5%) | ✅ |
| ranking hotcoco/vernier | 4.73x vs 4.33x (-8.3%) | ✅ |
| ranking pycocotools/vernier | 114.31x vs 105.33x (-7.9%) | ✅ |

Results: `bench/results/ad49243d651f/a6ada888c850/instance/objects365_val_jittered_seed0/bbox` vs `bench/results/ad49243d651f/a6ada888c850/instance/twin_objects365_val_jittered_s0_ad5537dd6ba8/bbox`
