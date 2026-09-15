# vernier-objects365-val-cache

Single source of truth for the Objects365 v2 val annotation cache, the
scale workload of the bench harness (`objects365_val_jittered_seed<N>`).
It mirrors the shape of `vernier-lvis-val-cache`: an idempotent
download-and-verify step pinned by URL and SHA-256.

| Field          | Value |
| -------------- | ----- |
| GT URL         | `https://dorc.ks3-cn-beijing.ksyun.com/data-set/2020Objects365%E6%95%B0%E6%8D%AE%E9%9B%86/val/zhiyuan_objv2_val.json` |
| GT SHA-256     | `b5b6a043f3b36c1865240a3e23fc4cbbf627052d72b6a4428a993bf2549d4424` |
| Size           | 269,422,006 bytes · 80,000 images · 1,240,587 boxes · 365 categories |
| Cache env var  | `VERNIER_OBJECTS365_CACHE` (defaults to `<repo>/.cache/objects365-val`) |

## Licensing

- **Annotations** belong to the Objects365 Consortium and are licensed
  under [Creative Commons Attribution 4.0](https://creativecommons.org/licenses/by/4.0/)
  ([objects365.org](https://www.objects365.org/download.html)). Anything
  published from them, such as the benchmark tables, carries the
  attribution in `objects365_val_cache.ATTRIBUTION`.
- **Images** are not owned by the consortium, fall under the Flickr
  Terms of Use, and must not be redistributed. This cache never
  downloads, references, or needs them: the bench workload is bbox-only
  and its detections are synthesized from the annotations.
- The consortium's own download portal asks users to register and
  describes the dataset as intended for academic use. This cache fetches
  the publicly served annotation file from the consortium's bucket, the
  same object the Ultralytics and Detectron2 dataset configs use, and
  uses it only for benchmarking.
- No dataset bytes are ever committed to this repository.

## Usage

```
$ python -m objects365_val_cache
Cache directory: /path/to/.cache/objects365-val
GT ready: …/zhiyuan_objv2_val.json
```
