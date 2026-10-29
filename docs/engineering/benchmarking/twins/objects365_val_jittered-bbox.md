# Twin check: `twin:objects365_val_jittered` vs `objects365_val_jittered_seed0` (bbox)

**FAIL**: 66/71 checks

| check | detail | ok |
|---|---|---|
| total gt | 1240587 vs 1240587 (+0.0%) | ✅ |
| total dt | 1062412 vs 1062412 (+0.0%) | ✅ |
| total cells | 625894 vs 625894 (+0.0%) | ✅ |
| total iou_pairs | 7123464 vs 7123464 (+0.0%) | ✅ |
| total crowd | 178349 vs 178349 (+0.0%) | ✅ |
| total gt_bytes | 269422006 vs 265595908 (-1.4%) | ✅ |
| total dt_bytes | 171098982 vs 169135450 (-1.1%) | ✅ |
| hist category_cells | TV 0.020 | ✅ |
| hist cell_gd | TV 0.000 | ✅ |
| hist cells_per_image | TV 0.000 | ✅ |
| hist dt_best_iou | TV 0.010 | ✅ |
| hist dt_score_matched | TV 0.001 | ✅ |
| hist dt_score_unmatched | TV 0.003 | ✅ |
| hist dt_segmentation | TV 0.000 | ✅ |
| hist gt_crowd_counts_len | TV 0.000 | ✅ |
| hist gt_fill | TV 0.000 | ✅ |
| hist gt_log_area | TV 0.009 | ✅ |
| hist gt_log_aspect | TV 0.009 | ✅ |
| hist gt_polygon_parts | TV 0.000 | ✅ |
| hist gt_polygon_vertices | TV 0.000 | ✅ |
| hist gt_segmentation | TV 0.000 | ✅ |
| hist id_ranges | TV 0.000 | ✅ |
| hotcoco stable measurement | IQR gate failed on twin: re-run on a quiet host | ❌ |
| hotcoco accumulate.cpu_ns | 3421.7 vs 3656.5 ms (+6.9%) | ✅ |
| hotcoco accumulate.peak_rss_bytes | 6649.6 vs 6651.9 MiB (+0.0%) | ✅ |
| hotcoco accumulate.wall_ns | 6880.5 vs 6375.3 ms (-7.3%) | ✅ |
| hotcoco evaluate.cpu_ns | 6604.5 vs 6742.3 ms (+2.1%) | ✅ |
| hotcoco evaluate.peak_rss_bytes | 6420.8 vs 6421.8 MiB (+0.0%) | ✅ |
| hotcoco evaluate.wall_ns | 13195.7 vs 12475.3 ms (-5.5%) | ✅ |
| hotcoco load.cpu_ns | 6001.3 vs 6136.2 ms (+2.2%) | ✅ |
| hotcoco load.peak_rss_bytes | 4025.7 vs 4027.0 MiB (+0.0%) | ✅ |
| hotcoco load.wall_ns | 11987.9 vs 12283.7 ms (+2.5%) | ✅ |
| hotcoco summarize.cpu_ns | 7.0 vs 7.0 ms (-0.5%) | ✅ |
| hotcoco summarize.peak_rss_bytes | 6649.6 vs 6651.9 MiB (+0.0%) | ✅ |
| hotcoco summarize.wall_ns | 13.1 vs 13.0 ms (-0.7%) | ✅ |
| hotcoco total.cpu_ns | 16077.0 vs 16558.7 ms (+3.0%) | ✅ |
| hotcoco total.peak_rss_bytes | 6649.6 vs 6651.9 MiB (+0.0%) | ✅ |
| hotcoco total.wall_ns | 32133.1 vs 30492.3 ms (-5.1%) | ✅ |
| pycocotools accumulate.cpu_ns | 40696.7 vs 41270.4 ms (+1.4%) | ✅ |
| pycocotools accumulate.peak_rss_bytes | 21854.1 vs 21941.1 MiB (+0.4%) | ✅ |
| pycocotools accumulate.wall_ns | 81366.9 vs 82381.3 ms (+1.2%) | ✅ |
| pycocotools evaluate.cpu_ns | 345715.2 vs 346868.2 ms (+0.3%) | ✅ |
| pycocotools evaluate.peak_rss_bytes | 21724.0 vs 21828.6 MiB (+0.5%) | ✅ |
| pycocotools evaluate.wall_ns | 691653.1 vs 491806.6 ms (-28.9%) | ❌ |
| pycocotools load.cpu_ns | 6875.4 vs 6955.5 ms (+1.2%) | ✅ |
| pycocotools load.peak_rss_bytes | 1920.0 vs 2040.0 MiB (+6.3%) | ✅ |
| pycocotools load.wall_ns | 13753.4 vs 9015.1 ms (-34.5%) | ❌ |
| pycocotools summarize.cpu_ns | 6.4 vs 6.5 ms (+0.2%) | ✅ |
| pycocotools summarize.peak_rss_bytes | 21849.1 vs 21937.2 MiB (+0.4%) | ✅ |
| pycocotools summarize.wall_ns | 12.4 vs 12.5 ms (+0.3%) | ✅ |
| pycocotools total.cpu_ns | 393293.8 vs 395100.5 ms (+0.5%) | ✅ |
| pycocotools total.peak_rss_bytes | 21854.1 vs 21941.1 MiB (+0.4%) | ✅ |
| pycocotools total.wall_ns | 786785.8 vs 583215.5 ms (-25.9%) | ❌ |
| vernier stable measurement | IQR gate failed on real: re-run on a quiet host | ❌ |
| vernier accumulate.cpu_ns | 754.6 vs 828.7 ms (+9.8%) | ✅ |
| vernier accumulate.peak_rss_bytes | 874.3 vs 885.3 MiB (+1.3%) | ✅ |
| vernier accumulate.wall_ns | 754.6 vs 828.7 ms (+9.8%) | ✅ |
| vernier evaluate.cpu_ns | 2626.2 vs 2587.3 ms (-1.5%) | ✅ |
| vernier evaluate.peak_rss_bytes | 928.8 vs 931.6 MiB (+0.3%) | ✅ |
| vernier evaluate.wall_ns | 2626.3 vs 2587.5 ms (-1.5%) | ✅ |
| vernier load.cpu_ns | 273.5 vs 260.4 ms (-4.8%) | ✅ |
| vernier load.peak_rss_bytes | 496.4 vs 490.8 MiB (-1.1%) | ✅ |
| vernier load.wall_ns | 273.5 vs 260.4 ms (-4.8%) | ✅ |
| vernier summarize.cpu_ns | 4.1 vs 4.0 ms (-1.0%) | ✅ |
| vernier summarize.peak_rss_bytes | 874.3 vs 885.3 MiB (+1.3%) | ✅ |
| vernier summarize.wall_ns | 4.1 vs 4.0 ms (-1.0%) | ✅ |
| vernier total.cpu_ns | 3674.8 vs 3682.8 ms (+0.2%) | ✅ |
| vernier total.peak_rss_bytes | 928.8 vs 931.6 MiB (+0.3%) | ✅ |
| vernier total.wall_ns | 3675.0 vs 3683.0 ms (+0.2%) | ✅ |
| ranking hotcoco/vernier | 4.37x vs 4.50x (+2.8%) | ✅ |
| ranking pycocotools/vernier | 107.02x vs 107.28x (+0.2%) | ✅ |

Results: `bench/results/3c5c1a406034/a6ada888c850/instance/objects365_val_jittered_seed0/bbox` vs `bench/results/3c5c1a406034/a6ada888c850/instance/twin_objects365_val_jittered_s0/bbox`
