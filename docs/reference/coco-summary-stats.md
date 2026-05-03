# COCO 12-stat detection summary

vernier's [`summarize_detection`](https://github.com/NoeFontana/vernier/blob/main/crates/vernier-core/src/summarize.rs) reproduces the twelve numbers that `pycocotools.cocoeval.COCOeval.summarize` writes into `eval.stats` for the `bbox` and `segm` IoU types. This page is the canonical mapping from each stat index to the slice of the accumulator's output tensors that produces it.

The accumulator's output is two tensors:

- `precision` — shape `(T, R, K, A, M)`. T = IoU thresholds, R = recall thresholds, K = categories, A = area ranges, M = maxDet caps.
- `recall` — shape `(T, K, A, M)`. The terminal cumulative recall (quirk **C4**), not an integral.

Both tensors use `-1` as the sentinel for absent cells (quirk **C5**). `summarize_detection` filters those out with `s > -1` before averaging — slices made entirely of sentinels return `-1`.

For COCO defaults, `T = 10`, `R = 101`, `A = 4` (`all`, `small`, `medium`, `large`), `M = 3` (`[1, 10, 100]`).

## Stat index → slice

| Idx | Name      | Metric    | IoU                          | Area     | maxDets    | Slice                                            |
|----:|-----------|-----------|------------------------------|----------|-----------:|--------------------------------------------------|
|   0 | AP        | Precision | `0.50:0.95` (mean over T)    | `all`    |        100 | `precision[:, :, :, 0, 2]`                       |
|   1 | AP@.50    | Precision | `0.50`                       | `all`    |        100 | `precision[t=0.50, :, :, 0, 2]`                  |
|   2 | AP@.75    | Precision | `0.75`                       | `all`    |        100 | `precision[t=0.75, :, :, 0, 2]`                  |
|   3 | AP_S      | Precision | `0.50:0.95`                  | `small`  |        100 | `precision[:, :, :, 1, 2]`                       |
|   4 | AP_M      | Precision | `0.50:0.95`                  | `medium` |        100 | `precision[:, :, :, 2, 2]`                       |
|   5 | AP_L      | Precision | `0.50:0.95`                  | `large`  |        100 | `precision[:, :, :, 3, 2]`                       |
|   6 | AR_1      | Recall    | `0.50:0.95`                  | `all`    |          1 | `recall[:, :, 0, 0]`                             |
|   7 | AR_10     | Recall    | `0.50:0.95`                  | `all`    |         10 | `recall[:, :, 0, 1]`                             |
|   8 | AR_100    | Recall    | `0.50:0.95`                  | `all`    |        100 | `recall[:, :, 0, 2]`                             |
|   9 | AR_S      | Recall    | `0.50:0.95`                  | `small`  |        100 | `recall[:, :, 1, 2]`                             |
|  10 | AR_M      | Recall    | `0.50:0.95`                  | `medium` |        100 | `recall[:, :, 2, 2]`                             |
|  11 | AR_L      | Recall    | `0.50:0.95`                  | `large`  |        100 | `recall[:, :, 3, 2]`                             |

For each row, the reported number is the mean of every cell in the slice whose value is `> -1`.

## How `summarize_detection` resolves the M-axis

vernier does not assume the M-axis is `[1, 10, 100]`. The summarizer:

- Uses `max_dets[m_max]` (the largest cap) for every AP line and for `AR_S` / `AR_M` / `AR_L`.
- Looks up `1`, `10`, `100` by value for `AR_1`, `AR_10`, `AR_100` respectively. If any of those three are missing from the supplied `max_dets`, summarization fails with [`EvalError::InvalidConfig`].

This matches what pycocotools does at `cocoeval.py:466-471` (`maxDets[0]`, `maxDets[1]`, `maxDets[2]`), but the value-lookup makes it robust to the default being changed by a caller.

## How `summarize_detection` resolves the IoU axis

For the AP@.50 / AP@.75 lines, vernier scans `iou_thresholds` for an entry within `1e-12` of the target. The pycocotools default is `numpy.linspace(0.5, 0.95, 10)`, which has `0.50` exactly and `0.7499999999999999` for `0.75` — within tolerance. If the threshold is missing entirely, summarization fails with [`EvalError::InvalidConfig`].

For all other lines, the IoU axis is averaged across (length `T`).

## Custom summary plans

`summarize_detection` is a thin wrapper over `summarize_with`, which evaluates an arbitrary `&[StatRequest]` plan against an `Accumulated`. The canonical 12-entry plan is exposed as `StatRequest::coco_detection_default()` for callers that want to extend rather than replace it (push extra `StatRequest` entries, swap the M-axis selectors, pin different IoU thresholds). Bit-exact parity with cocoeval is by construction: the wrapper produces the same `Summary` whether called directly or via `summarize_with(.., StatRequest::coco_detection_default(), ..)`.

`MaxDetSelector::Largest` picks the trailing entry of the supplied `max_dets`; `MaxDetSelector::Value(n)` looks the value up. This preserves the cocoeval intent ("AR_1 means maxDets=1") without binding to fixed positional indices, so callers passing non-default `max_dets` (e.g. keypoint `[20]`) still get sensible plans.

## Custom area buckets

`AreaRng` is a `(index, label)` value type, not a closed enum. The canonical pycocotools layout is exposed as `AreaRng::ALL` / `SMALL` / `MEDIUM` / `LARGE` (constants matching `[all, small, medium, large]` at A-axis indices `0..4`). Callers that build an `Accumulated` with a different number of A-axis buckets — finer breakdowns (e.g. add a "tiny" bucket below "small") or robotics-relevant axes — construct `AreaRng::new(index, label)` and address them in their plan; the label flows through to `pretty_lines`.

Note: the *bounds* that turn an annotation's area into a bucket index live upstream, on the orchestrator that builds `PerImageEval` cells (not yet implemented — Phase 1 Week 5). When that lands, it will accept an `AreaBucketing` config that pairs each label with its `[min_area, max_area]` bounds and emits the matching A-axis indices to the accumulator. The summarizer-side flexibility added here is the consumer of that config.

## Keypoint summary

The 10-stat `_summarizeKps` table is not yet implemented (Phase 3). Its layout differs only in `maxDets = [20]` and a 2-bucket area range (`medium`, `large`); the slicing is otherwise the same. With the plan/execution split, this is a custom plan, not a new entry point.
