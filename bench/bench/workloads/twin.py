"""Synthetic twins of real instance workloads (ADR-0067).

A *profile* is aggregate statistics of a real COCO-format GT/DT pair — no
records, no coordinates, nothing re-identifiable — recording what drives
evaluation cost:

* the per-(image, category) cell sizes ``(g, d)``: matching does ``g * d``
  IoU evaluations per cell, so this joint distribution is the work;
* box geometry (area drives the COCO area-range split), crowd share;
* how detections overlap ground truth (best same-category IoU) and score;
* mask complexity (RLE string length) when masks are present;
* how the JSON is written — key order, extra fields, separators, digits per
  number — since parsing dominates at scale.

``generate`` rebuilds a GT/DT pair with the same statistics, deterministic
per ``(profile, seed)``. A twin replaces its real workload only once
``bench twin compare`` shows matching structure *and* matching per-stage
time and memory for every implementation (see ADR-0067).
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from pycocotools import mask as mask_utils

from bench.harness.migrations import upgrade as upgrade_schema
from bench.harness.paths import BENCH_ROOT, bench_cache_root
from bench.harness.schema import BenchResult
from bench.harness.stats import aggregate_stage

PROFILE_VERSION = 1
# Bump when generation changes, so cached twins regenerate.
TWIN_VERSION = 1
PROFILE_DIR = BENCH_ROOT / "twin_profiles"
TWIN_PREFIX = "twin:"

# Continuous-histogram bin edges. Fine enough that sampling uniformly
# within a bin cannot move a box across a COCO area-range threshold
# (32^2, 96^2) by more than a few percent of the bin.
_LOG_AREA_EDGES = np.linspace(0.0, 16.0, 129)  # ln(area), area up to ~8.9e6 px
_LOG_ASPECT_EDGES = np.linspace(-4.0, 4.0, 81)  # ln(w / h)
_FILL_EDGES = np.linspace(0.0, 1.5, 61)  # GT `area` / bbox area
_UNIT_EDGES = np.linspace(0.0, 1.0, 51)  # scores, IoUs
_PER_COLUMN_EDGES = np.linspace(0.0, 32.0, 257)  # RLE chars per bbox column
_IOU_EDGES = np.concatenate([[0.0, 1e-9], np.linspace(0.05, 1.0, 20)])  # bin 0: no overlap

_SEMANTIC_IMAGE_KEYS = frozenset({"id", "width", "height"})
_SEMANTIC_ANN_KEYS = frozenset(
    {"id", "image_id", "category_id", "bbox", "area", "iscrowd", "segmentation"}
)
_SEMANTIC_DT_KEYS = frozenset({"image_id", "category_id", "bbox", "score", "area", "segmentation"})

# ---------------------------------------------------------------------------
# Histogram helpers. Discrete histograms are {value: count} with string keys
# (JSON); continuous ones are counts over fixed edges.
# ---------------------------------------------------------------------------


def _discrete(values: Any) -> dict[str, int]:
    return {str(k): int(v) for k, v in sorted(Counter(values).items())}


def _continuous(values: Any, edges: np.ndarray) -> list[int]:
    arr = np.clip(np.asarray(values, dtype=np.float64), edges[0], edges[-1])
    return np.histogram(arr, bins=edges)[0].astype(int).tolist()


def _allocate(counts: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """`n` bin indices following `counts`, shuffled.

    Stratified (largest-remainder quotas) rather than i.i.d.: with `n`
    equal to the source size it reproduces the histogram exactly, so
    totals such as GT count or IoU pairs carry no sampling noise.
    """
    p = counts / counts.sum()
    quota = p * n
    take = np.floor(quota).astype(np.int64)
    short = n - int(take.sum())
    if short:
        take[np.argsort(take - quota, kind="stable")[:short]] += 1
    return rng.permutation(np.repeat(np.arange(len(counts)), take))


def _sample_discrete(hist: dict[str, int], n: int, rng: np.random.Generator) -> np.ndarray:
    keys = list(hist)
    counts = np.array([hist[k] for k in keys], dtype=np.float64)
    return np.array(keys, dtype=object)[_allocate(counts, n, rng)]


def _sample_continuous(
    counts: list[int], edges: np.ndarray, n: int, rng: np.random.Generator
) -> np.ndarray:
    p = np.asarray(counts, dtype=np.float64)
    if p.sum() == 0:
        return np.zeros(n)
    bins = _allocate(p, n, rng)
    return rng.uniform(edges[bins], edges[bins + 1])


def _total_variation(a: list[int] | dict[str, int], b: list[int] | dict[str, int]) -> float:
    if isinstance(a, dict) and isinstance(b, dict):
        keys = sorted(set(a) | set(b))
        va = np.array([a.get(k, 0) for k in keys], dtype=np.float64)
        vb = np.array([b.get(k, 0) for k in keys], dtype=np.float64)
    else:
        va, vb = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if va.sum() == 0 and vb.sum() == 0:
        return 0.0
    if va.sum() == 0 or vb.sum() == 0:
        return 1.0
    return float(0.5 * np.abs(va / va.sum() - vb / vb.sum()).sum())


# ---------------------------------------------------------------------------
# JSON shape: digits per number and the non-semantic fields of each record.
# ---------------------------------------------------------------------------


def _frac_digits(x: Any) -> int:
    """Digits after the point in the number's JSON form; -1 for integers."""
    if isinstance(x, int):
        return -1
    s = repr(float(x))
    if "e" in s or "." not in s:
        return 17
    return len(s) - s.index(".") - 1


def _render(x: float, digits: int) -> float | int:
    return round(x) if digits < 0 else round(x, digits)


def _record_template(records: list[dict[str, Any]], semantic: frozenset[str]) -> dict[str, Any]:
    """Key order plus, per extra field, a constant or a string length."""
    sample = records[: min(len(records), 20000)]
    order = list(sample[0]) if sample else []
    extras: dict[str, Any] = {}
    for key in order:
        if key in semantic:
            continue
        values = [r.get(key) for r in sample]
        # Only shape is kept: integers' range, other values' JSON length.
        # Real text (file names, URLs, dates) never enters the profile.
        if all(isinstance(v, int) and not isinstance(v, bool) for v in values):
            extras[key] = {"int_max": max(values)}
        elif all(v in (None, "", False) for v in values[:2000]):
            extras[key] = {"const": values[0]}
        else:
            extras[key] = {"strlen": float(np.mean([len(json.dumps(v)) - 2 for v in values]))}
    return {"order": order, "extras": extras}


def _seg_kind(segmentation: Any) -> str:
    if isinstance(segmentation, list):
        return "polygon"
    return "rle" if isinstance(segmentation, dict) else "none"


def _separators(path: Path) -> tuple[str, str]:
    with path.open("rb") as f:
        head = f.read(4096).decode("utf-8", "replace")
    item = ", " if re.search(r'[\]}"\d], "', head) else ","
    key = ": " if '": ' in head else ":"
    return item, key


def _bbox_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU, (n, 4) xywh against (m, 4) xywh."""
    ax2, ay2 = a[:, 0] + a[:, 2], a[:, 1] + a[:, 3]
    bx2, by2 = b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]
    iw = np.clip(np.minimum(ax2[:, None], bx2) - np.maximum(a[:, None, 0], b[:, 0]), 0, None)
    ih = np.clip(np.minimum(ay2[:, None], by2) - np.maximum(a[:, None, 1], b[:, 1]), 0, None)
    inter = iw * ih
    union = (a[:, 2] * a[:, 3])[:, None] + b[:, 2] * b[:, 3] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0)


def _rle_len(segmentation: Any, h: int, w: int) -> int | None:
    """Length of the compressed RLE string: the mask-complexity measure."""
    if isinstance(segmentation, list):
        if not segmentation:
            return None
        rles = mask_utils.frPyObjects(segmentation, h, w)
        rle = mask_utils.merge(rles)
    elif isinstance(segmentation, dict):
        rle = segmentation
        if isinstance(rle["counts"], list):
            rle = mask_utils.frPyObjects(rle, h, w)
    else:
        return None
    counts = rle["counts"]
    return len(counts)


# ---------------------------------------------------------------------------
# Profile extraction.
# ---------------------------------------------------------------------------


def extract(gt_path: Path, dt_path: Path, *, masks: bool, source: str) -> dict[str, Any]:
    gt = json.loads(gt_path.read_text())
    dts: list[dict[str, Any]] = json.loads(dt_path.read_text())
    images: list[dict[str, Any]] = gt["images"]
    anns: list[dict[str, Any]] = gt["annotations"]
    size_of = {im["id"]: (int(im["height"]), int(im["width"])) for im in images}

    gt_by_cell: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    dt_by_cell: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for a in anns:
        gt_by_cell[(a["image_id"], a["category_id"])].append(a)
    for d in dts:
        dt_by_cell[(d["image_id"], d["category_id"])].append(d)
    cells = set(gt_by_cell) | set(dt_by_cell)

    cells_per_image = Counter(img for img, _ in cells)
    cell_gd = Counter(f"{len(gt_by_cell.get(c, ()))},{len(dt_by_cell.get(c, ()))}" for c in cells)
    by_category = Counter(cat for _, cat in cells)
    ranked_ids = [cat for cat, _ in by_category.most_common()]
    ranked_ids += sorted({c["id"] for c in gt["categories"]} - set(ranked_ids))
    category_cells = [by_category[c] for c in ranked_ids]

    best_iou: list[float] = []
    matched_scores: list[float] = []
    unmatched_scores: list[float] = []
    for cell, cell_dts in dt_by_cell.items():
        gts = [g for g in gt_by_cell.get(cell, ()) if not g.get("iscrowd")]
        boxes = np.array([d["bbox"] for d in cell_dts], dtype=np.float64)
        if gts:
            ious = _bbox_iou(boxes, np.array([g["bbox"] for g in gts], dtype=np.float64))
            best = ious.max(axis=1)
            best_iou.extend(best.tolist())
        else:
            best = np.zeros(len(cell_dts))
        for d, iou in zip(cell_dts, best, strict=True):
            (matched_scores if iou >= 0.5 else unmatched_scores).append(d["score"])

    gt_boxes = np.array([a["bbox"] for a in anns], dtype=np.float64).reshape(-1, 4)
    bbox_area = np.maximum(gt_boxes[:, 2] * gt_boxes[:, 3], 1e-6)
    seg_kinds = Counter(_seg_kind(a.get("segmentation")) for a in anns)
    polygons = [a["segmentation"] for a in anns if isinstance(a.get("segmentation"), list)]
    crowd_rles = [a["segmentation"] for a in anns if isinstance(a.get("segmentation"), dict)]
    coord_sample = [c for poly in polygons[:5000] for part in poly for c in part][:200000]

    profile: dict[str, Any] = {
        "profile_version": PROFILE_VERSION,
        "source": source,
        "masks": masks,
        "n_images": len(images),
        "id_ranges": {
            "image": [min(im["id"] for im in images), max(im["id"] for im in images)],
            "annotation": [min(a["id"] for a in anns), max(a["id"] for a in anns)],
        },
        "image_sizes": _discrete(f"{h},{w}" for h, w in size_of.values()),
        "categories": len(gt["categories"]),
        "cells_per_image": _discrete(cells_per_image[im["id"]] for im in images),
        "cell_gd": dict(sorted(cell_gd.items())),
        "category_cells": category_cells,
        # Integer ids only, most-used first: their digits are parse cost.
        "category_ids_by_rank": ranked_ids,
        "crowd_fraction": float(np.mean([bool(a.get("iscrowd")) for a in anns])),
        "gt_log_area": _continuous(np.log(bbox_area), _LOG_AREA_EDGES),
        "gt_log_aspect": _continuous(
            np.log(np.maximum(gt_boxes[:, 2], 1e-3) / np.maximum(gt_boxes[:, 3], 1e-3)),
            _LOG_ASPECT_EDGES,
        ),
        "gt_fill": _continuous([a["area"] for a in anns] / bbox_area, _FILL_EDGES),
        "gt_segmentation": dict(seg_kinds),
        "gt_polygon_parts": _discrete(len(p) for p in polygons),
        "gt_polygon_vertices": _discrete(min(len(part) // 2, 1024) for p in polygons for part in p),
        "gt_crowd_counts_len": _discrete(
            min(len(r["counts"]), 4096) // 16 * 16 for r in crowd_rles
        ),
        "dt_best_iou": _continuous(best_iou, _IOU_EDGES),
        "dt_score_matched": _continuous(matched_scores, _UNIT_EDGES),
        "dt_score_unmatched": _continuous(unmatched_scores, _UNIT_EDGES),
        # Per-record kinds: the jitterer gives only its false positives a
        # rectangle polygon (and `area`), real detectors an RLE per record.
        "dt_segmentation": dict(Counter(_seg_kind(d.get("segmentation")) for d in dts)),
        "dt_area_share": float(np.mean(["area" in d for d in dts])) if dts else 0.0,
        "digits": {
            "gt_bbox": _discrete(_frac_digits(v) for v in gt_boxes.ravel()[:200000]),
            "gt_area": _discrete(_frac_digits(a["area"]) for a in anns[:50000]),
            "gt_polygon": _discrete(_frac_digits(v) for v in coord_sample),
            "dt_bbox": _discrete(_frac_digits(v) for d in dts[:50000] for v in d["bbox"]),
            "dt_score": _discrete(_frac_digits(d["score"]) for d in dts[:50000]),
            "dt_area": _discrete(_frac_digits(d["area"]) for d in dts[:50000] if "area" in d),
        },
        "json": {
            "gt_top_keys": list(gt),
            # Sizes only (info, licenses): their content is the dataset's.
            "gt_extra_top": {
                k: [type(gt[k]).__name__, len(json.dumps(gt[k]))]
                for k in gt
                if k not in ("images", "annotations", "categories")
            },
            "image": _record_template(images, _SEMANTIC_IMAGE_KEYS),
            "annotation": _record_template(anns, _SEMANTIC_ANN_KEYS),
            "detection": _record_template(dts, _SEMANTIC_DT_KEYS),
            "gt_separators": _separators(gt_path),
            "dt_separators": _separators(dt_path),
        },
        "totals": {
            "gt": len(anns),
            "dt": len(dts),
            "cells": len(cells),
            "iou_pairs": int(
                sum(len(gt_by_cell.get(c, ())) * len(dt_by_cell.get(c, ())) for c in cells)
            ),
            "crowd": int(sum(bool(a.get("iscrowd")) for a in anns)),
            "gt_bytes": gt_path.stat().st_size,
            "dt_bytes": dt_path.stat().st_size,
        },
    }
    if masks:
        gt_rle: list[tuple[int, float]] = []
        # Per vertex-count bucket: a 6-vertex outline cannot carry the RLE of
        # a 60-vertex one, so GT complexity is drawn conditional on it.
        by_bucket: dict[int, list[float]] = defaultdict(list)
        for a in anns:
            seg = a.get("segmentation")
            if a.get("iscrowd") or not (n := _rle_len(seg, *size_of[a["image_id"]])):
                continue
            gt_rle.append((n, a["bbox"][2]))
            if isinstance(seg, list):
                vertices = sum(len(part) // 2 for part in seg)
                by_bucket[_vertex_bucket(vertices)].append(n / max(a["bbox"][2], 1.0))
        profile["gt_rle_per_column_by_vertices"] = {
            str(b): _continuous(v, _PER_COLUMN_EDGES) for b, v in sorted(by_bucket.items())
        }
        dt_rle = [
            (n, d["bbox"][2])
            for d in dts
            if (n := _rle_len(d.get("segmentation"), *size_of[d["image_id"]]))
        ]
        # RLE length grows with the columns a mask spans; generation draws
        # chars-per-column and multiplies by the generated box's width.
        profile["gt_rle_per_column"] = _continuous(
            [n / max(w, 1.0) for n, w in gt_rle], _PER_COLUMN_EDGES
        )
        profile["dt_rle_per_column"] = _continuous(
            [n / max(w, 1.0) for n, w in dt_rle], _PER_COLUMN_EDGES
        )
        profile["gt_rle_len"] = _discrete(min(n, 8192) // 32 * 32 for n, _ in gt_rle)
        profile["dt_rle_len"] = _discrete(min(n, 8192) // 32 * 32 for n, _ in dt_rle)
        profile["totals"]["gt_rle_chars"] = sum(n for n, _ in gt_rle)
        profile["totals"]["dt_rle_chars"] = sum(n for n, _ in dt_rle)
    return profile


# ---------------------------------------------------------------------------
# Generation.
# ---------------------------------------------------------------------------


def _filler(spec: dict[str, Any], rng: np.random.Generator, index: int) -> Any:
    if "const" in spec:
        return spec["const"]
    if "strlen" in spec:
        n = max(round(spec["strlen"]), 1)
        return f"{index:0{n}d}"[-n:]
    return int(rng.integers(0, spec["int_max"] + 1))


def _ordered(record: dict[str, Any], template: dict[str, Any], rng, index) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in template["order"]:
        if key in record:
            out[key] = record[key]
        elif key in template["extras"]:
            out[key] = _filler(template["extras"][key], rng, index)
    for key, value in record.items():
        out.setdefault(key, value)
    return out


def generate(profile: dict[str, Any], seed: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Build a GT dict and DT list with the profile's statistics.

    Vectorized pass by pass (cells, GT, DT, then records) so a twin of
    Objects365 (1.2M GT, 1M DT) generates in seconds, not hours.
    """
    rng = np.random.default_rng([seed, TWIN_VERSION])
    templates = profile["json"]
    n_images = profile["n_images"]
    n_categories = profile["categories"]

    # Images and their cells.
    sizes = np.array(
        [s.split(",") for s in _sample_discrete(profile["image_sizes"], n_images, rng)],
        dtype=np.int64,
    )  # (n, 2) as (h, w)
    cells_per_image = np.minimum(
        _sample_discrete(profile["cells_per_image"], n_images, rng).astype(np.int64),
        n_categories,
    )
    weights = np.zeros(n_categories)
    ranked = np.asarray(profile["category_cells"], dtype=np.float64)[:n_categories]
    weights[: len(ranked)] = ranked
    weights = (weights + 1e-9) / (weights + 1e-9).sum()
    cell_image = np.repeat(np.arange(n_images), cells_per_image)
    ids_by_rank = np.asarray(profile["category_ids_by_rank"], dtype=np.int64)
    cell_cat = ids_by_rank[
        np.concatenate(
            [rng.choice(n_categories, size=k, replace=False, p=weights) for k in cells_per_image]
        ).astype(np.int64)
    ]
    gd = np.array(
        [s.split(",") for s in _sample_discrete(profile["cell_gd"], len(cell_image), rng)],
        dtype=np.int64,
    ).reshape(-1, 2)
    g_per_cell, d_per_cell = gd[:, 0], gd[:, 1]

    # Ground truth.
    gt_cell = np.repeat(np.arange(len(cell_image)), g_per_cell)
    n_gt = len(gt_cell)
    gt_img = cell_image[gt_cell]
    img_h, img_w = sizes[gt_img, 0].astype(np.float64), sizes[gt_img, 1].astype(np.float64)
    area = np.exp(_sample_continuous(profile["gt_log_area"], _LOG_AREA_EDGES, n_gt, rng))
    aspect = np.exp(_sample_continuous(profile["gt_log_aspect"], _LOG_ASPECT_EDGES, n_gt, rng))
    gw = np.minimum(np.sqrt(area * aspect), img_w)
    gh = np.minimum(np.sqrt(area / aspect), img_h)
    gx = rng.uniform(0, 1, n_gt) * np.maximum(img_w - gw, 0)
    gy = rng.uniform(0, 1, n_gt) * np.maximum(img_h - gh, 0)
    gt_box = np.stack([gx, gy, gw, gh], axis=1)
    fill = _sample_continuous(profile["gt_fill"], _FILL_EDGES, n_gt, rng)
    crowd = _exactly(round(profile["crowd_fraction"] * n_gt), n_gt, rng)
    kinds = profile["gt_segmentation"]
    polygon_share = kinds.get("polygon", 0) / max(sum(kinds.values()), 1)
    has_polygon = np.zeros(n_gt, dtype=bool)
    has_polygon[~crowd] = _exactly(round(polygon_share * n_gt), int((~crowd).sum()), rng)
    crowd_rle = crowd & bool(profile["gt_crowd_counts_len"])
    # Per-annotation shape draws, stratified over the whole population.
    n_poly = int(has_polygon.sum())
    parts = np.zeros(n_gt, dtype=np.int64)
    parts[has_polygon] = _ints(profile["gt_polygon_parts"] or {"1": 1}, n_poly, rng)
    part_vertices = _ints(profile["gt_polygon_vertices"] or {"16": 1}, int(parts.sum()), rng)
    poly_vertices = np.split(part_vertices, np.cumsum(parts[has_polygon])[:-1]) if n_poly else []
    polygon_specs = iter(
        zip(poly_vertices, _gt_per_column(profile, poly_vertices, rng), strict=True)
    )
    crowd_len = iter(_ints(profile["gt_crowd_counts_len"] or {"2": 1}, int(crowd_rle.sum()), rng))

    # Detections: each draws a best-IoU target; overlapping ones are a GT
    # of the same cell shifted sideways by exactly the amount giving that IoU.
    dt_cell = np.repeat(np.arange(len(cell_image)), d_per_cell)
    n_dt = len(dt_cell)
    dt_img = cell_image[dt_cell]
    gt_start = np.concatenate([[0], np.cumsum(g_per_cell)])[:-1]
    iou_bin = _allocate(np.asarray(profile["dt_best_iou"], dtype=np.float64), n_dt, rng)
    iou_target = rng.uniform(_IOU_EDGES[iou_bin], _IOU_EDGES[iou_bin + 1])
    # Anchor on non-crowd GT only: the profile's IoU is measured against those.
    by_cell_noncrowd_first = np.lexsort((crowd, gt_cell))
    cell_nc = np.bincount(gt_cell[~crowd], minlength=len(cell_image))[dt_cell]
    overlaps = (cell_nc > 0) & (iou_bin > 0)
    slot = gt_start[dt_cell] + (rng.random(n_dt) * np.maximum(cell_nc, 1)).astype(np.int64)
    pick = by_cell_noncrowd_first[np.minimum(slot, max(n_gt - 1, 0))] if n_gt else slot
    base = gt_box[pick] if n_gt else np.zeros((n_dt, 4))
    sign = np.where(rng.random(n_dt) < 0.5, -1.0, 1.0)
    shifted = base.copy()
    shifted[:, 0] += sign * base[:, 2] * (1 - iou_target) / (1 + iou_target)
    free_side = np.sqrt(
        np.exp(_sample_continuous(profile["gt_log_area"], _LOG_AREA_EDGES, n_dt, rng))
    )
    dh, dw = sizes[dt_img, 0].astype(np.float64), sizes[dt_img, 1].astype(np.float64)
    fw, fh = np.minimum(free_side, dw), np.minimum(free_side, dh)
    free = np.stack(
        [
            rng.random(n_dt) * np.maximum(dw - fw, 0),
            rng.random(n_dt) * np.maximum(dh - fh, 0),
            fw,
            fh,
        ],
        axis=1,
    )
    dt_box = np.where(overlaps[:, None], shifted, free)
    matched = overlaps & (iou_target >= 0.5)
    score = np.where(
        matched,
        _sample_continuous(profile["dt_score_matched"], _UNIT_EDGES, n_dt, rng),
        _sample_continuous(profile["dt_score_unmatched"], _UNIT_EDGES, n_dt, rng),
    )

    # Records, with numbers written to the source's digit counts.
    fmt = _Formatter(profile["digits"], rng)
    # Ids span the source's ranges: their digit counts are bytes to parse.
    image_ids = _ids(profile["id_ranges"]["image"], n_images, rng)
    ann_ids = _ids(profile["id_ranges"]["annotation"], n_gt, rng)

    images = [
        _ordered(
            {"id": int(image_ids[i]), "width": int(sizes[i, 1]), "height": int(sizes[i, 0])},
            templates["image"],
            rng,
            i + 1,
        )
        for i in range(n_images)
    ]
    anns: list[dict[str, Any]] = []
    bbox_values = fmt.many("gt_bbox", gt_box.ravel()).reshape(-1, 4)
    area_values = fmt.many("gt_area", fill * gw * gh)
    for j in range(n_gt):
        h, w = int(sizes[gt_img[j], 0]), int(sizes[gt_img[j], 1])
        ann: dict[str, Any] = {
            "id": int(ann_ids[j]),
            "image_id": int(image_ids[gt_img[j]]),
            "category_id": int(cell_cat[gt_cell[j]]),
            "bbox": bbox_values[j].tolist(),
            "area": area_values[j],
            "iscrowd": int(crowd[j]),
        }
        if crowd_rle[j]:
            counts = _crowd_counts(gt_box[j], int(next(crowd_len)) + 8, h, w, rng)
            ann["segmentation"] = {"counts": counts, "size": [h, w]}
        elif has_polygon[j]:
            verts, per_column = next(polygon_specs)
            n_parts = len(verts)
            rle_target = per_column * gw[j] / n_parts
            polygons = [
                _shape(_part(gt_box[j], p, n_parts), int(verts[p]), rle_target, h, w, rng)[0]
                for p in range(n_parts)
            ]
            ann["segmentation"] = [fmt.many("gt_polygon", np.asarray(c)).tolist() for c in polygons]
        anns.append(_ordered(ann, templates["annotation"], rng, j + 1))

    # Segmentation kinds in the source's exact shares; polygons (the
    # jitterer's false positives) go to non-overlapping detections first,
    # and `area` to records that carry a segmentation first.
    dt_kinds = profile["dt_segmentation"]
    total_kinds = max(sum(dt_kinds.values()), 1)
    order = np.lexsort((rng.random(n_dt), overlaps))
    n_poly_dt = round(dt_kinds.get("polygon", 0) / total_kinds * n_dt)
    n_rle_dt = round(dt_kinds.get("rle", 0) / total_kinds * n_dt)
    dt_kind = np.zeros(n_dt, dtype=np.int8)  # 0 none, 1 polygon, 2 rle
    dt_kind[order[:n_poly_dt]] = 1
    dt_kind[order[n_poly_dt : n_poly_dt + n_rle_dt]] = 2
    has_area = np.zeros(n_dt, dtype=bool)
    has_area[order[: round(profile["dt_area_share"] * n_dt)]] = True
    dt_bbox_values = fmt.many("dt_bbox", dt_box.ravel()).reshape(-1, 4)
    score_values = fmt.many("dt_score", score)
    dt_area_values = fmt.many("dt_area", dt_box[:, 2] * dt_box[:, 3])
    dt_per_column = iter(_per_column(profile, "dt", int((dt_kind == 2).sum()), rng))
    dts: list[dict[str, Any]] = []
    for j in range(n_dt):
        det: dict[str, Any] = {
            "image_id": int(image_ids[dt_img[j]]),
            "category_id": int(cell_cat[dt_cell[j]]),
            "bbox": dt_bbox_values[j].tolist(),
            "score": score_values[j],
        }
        if has_area[j]:
            det["area"] = dt_area_values[j]
        if dt_kind[j] == 1:
            x, y, bw, bh = det["bbox"]
            det["segmentation"] = [[x, y, x + bw, y, x + bw, y + bh, x, y + bh]]
        elif dt_kind[j] == 2:
            h, w = int(sizes[dt_img[j], 0]), int(sizes[dt_img[j], 1])
            _, rle = _shape(dt_box[j], 32, next(dt_per_column) * dt_box[j, 2], h, w, rng)
            det["segmentation"] = {"size": [h, w], "counts": rle["counts"].decode()}
        dts.append(_ordered(det, templates["detection"], rng, j + 1))

    gt: dict[str, Any] = {}
    for key in templates["gt_top_keys"]:
        if key == "images":
            gt[key] = images
        elif key == "annotations":
            gt[key] = anns
        elif key == "categories":
            gt[key] = [{"id": int(c), "name": f"cat{c}"} for c in sorted(ids_by_rank)]
        else:
            kind, size = templates["gt_extra_top"].get(key, ["dict", 0])
            pad = {"pad": "x" * size}
            gt[key] = [pad] if kind == "list" else pad
    return gt, dts


class _Formatter:
    """Rounds values to digit counts drawn from the source's distribution."""

    def __init__(self, digits: dict[str, dict[str, int]], rng: np.random.Generator) -> None:
        self._digits = digits
        self._rng = rng

    def many(self, field: str, values: np.ndarray) -> np.ndarray:
        hist = self._digits.get(field) or {"17": 1}
        counts = _sample_discrete(hist, len(values), self._rng).astype(np.int64)
        out = np.empty(len(values), dtype=object)
        for i, (x, d) in enumerate(zip(values.tolist(), counts.tolist(), strict=True)):
            out[i] = _render(x, d)
        return out


def _rle(polygon: list[float], h: int, w: int) -> dict[str, Any]:
    codec: Any = mask_utils
    return codec.merge(codec.frPyObjects([polygon], h, w))


def _ids(span: list[int], n: int, rng: np.random.Generator) -> np.ndarray:
    """`n` distinct increasing ids spread over [lo, hi], shuffled."""
    lo, hi = span
    if n == 0:
        return np.zeros(0, np.int64)
    mean_gap = max((hi - lo) / n, 1.0)
    gaps = 1 + rng.integers(0, max(int(2 * mean_gap) - 1, 1), size=n)
    return rng.permutation(lo + np.cumsum(gaps) - gaps[0])


def _exactly(k: int, n: int, rng: np.random.Generator) -> np.ndarray:
    """Boolean mask of length `n` with exactly `k` (clipped) set entries."""
    out = np.zeros(n, dtype=bool)
    out[rng.permutation(n)[: max(0, min(k, n))]] = True
    return out


def _ints(hist: dict[str, int], n: int, rng: np.random.Generator) -> np.ndarray:
    return _sample_discrete(hist, n, rng).astype(np.int64) if n else np.zeros(0, np.int64)


_VERTEX_BUCKETS = (8, 16, 32, 64, 128)


def _vertex_bucket(vertices: int) -> int:
    return int(np.searchsorted(_VERTEX_BUCKETS, vertices))


def _gt_per_column(profile: dict[str, Any], poly_vertices: list[np.ndarray], rng) -> np.ndarray:
    """Per GT polygon, RLE chars per column drawn for its vertex-count bucket."""
    out = np.zeros(len(poly_vertices))
    by_bucket = profile.get("gt_rle_per_column_by_vertices") or {}
    buckets = np.array([_vertex_bucket(int(v.sum())) for v in poly_vertices], dtype=np.int64)
    for bucket in np.unique(buckets):
        idx = np.flatnonzero(buckets == bucket)
        counts = by_bucket.get(str(bucket)) or profile.get("gt_rle_per_column")
        if counts:
            out[idx] = _sample_continuous(counts, _PER_COLUMN_EDGES, len(idx), rng)
    return out


def _per_column(profile: dict[str, Any], side: str, n: int, rng) -> np.ndarray:
    """`n` draws of RLE chars per bbox column (zeros without mask stats)."""
    counts = profile.get(f"{side}_rle_per_column")
    if not counts or not n:
        return np.zeros(n)
    return _sample_continuous(counts, _PER_COLUMN_EDGES, n, rng)


def _part(box: np.ndarray, index: int, parts: int) -> np.ndarray:
    """Slice `index` of `parts` side-by-side slices of `box`."""
    bx, by, bw, bh = box.tolist()
    return np.array([bx + index * bw / parts, by, bw / parts, bh])


# Radial roughness levels tried, smooth first, to match a target RLE length.
_ROUGHNESS = tuple(np.linspace(0.0, 0.95, 12).tolist())


def _shape(
    box: np.ndarray, n: int, target: float, h: int, w: int, rng
) -> tuple[list[float], dict[str, Any]]:
    """An `n`-vertex polygon in `box` whose RLE length is nearest `target`,
    with that RLE.

    Mask cost scales with run count, which a smooth ellipse underestimates
    against real object outlines; radial roughness adds runs until the
    compressed RLE reaches the length drawn from the source distribution.
    """
    bx, by, bw, bh = box.tolist()
    n = max(n, 3)
    t = np.linspace(0.0, 2 * math.pi, n, endpoint=False)
    # Alternate deep and shallow vertices (a star): concavities put several
    # runs in one column, which is what makes real outlines expensive.
    jitter = 0.5 * rng.random(n) + 0.5 * (np.arange(n) % 2)
    best: tuple[list[float], dict[str, Any]] = ([], {})
    best_gap = math.inf
    for roughness in _ROUGHNESS:
        r = 1 - roughness * jitter
        xs = bx + bw / 2 * (1 + r * np.cos(t))
        ys = by + bh / 2 * (1 + r * np.sin(t))
        polygon = np.stack([xs, ys], axis=1).ravel().tolist()
        rle = _rle(polygon, h, w)
        length = len(rle["counts"])
        if abs(length - target) < best_gap:
            best, best_gap = (polygon, rle), abs(length - target)
        if length >= target:
            break
    return best


def _crowd_counts(box: np.ndarray, length: int, h: int, w: int, rng) -> list[int]:
    """Uncompressed crowd RLE of about `length` runs inside the box's columns."""
    lo = int(box[0]) * h
    hi = max(min(math.ceil(box[0] + box[2]), w) * h, lo + 2)
    k = max(min(length - 1, hi - lo - 1), 1)
    cuts = np.sort(lo + 1 + rng.choice(hi - lo - 1, size=k, replace=False))
    return np.diff(np.concatenate([[0], cuts, [h * w]])).astype(int).tolist()


# ---------------------------------------------------------------------------
# Workload plumbing and comparison.
# ---------------------------------------------------------------------------


def profile_path(name: str) -> Path:
    return PROFILE_DIR / f"{name}.json"


def load_profile(name: str) -> dict[str, Any]:
    profile = json.loads(profile_path(name).read_text())
    if (version := profile.get("profile_version")) != PROFILE_VERSION:
        raise ValueError(
            f"{profile_path(name)} is profile v{version}, this code reads v{PROFILE_VERSION}; "
            "re-run `bench twin profile`"
        )
    return profile


def parse_workload(workload_name: str) -> tuple[str, int] | None:
    """``twin:<profile>[,seed=N]`` -> (profile, seed)."""
    if not workload_name.startswith(TWIN_PREFIX):
        return None
    name, _, rest = workload_name.removeprefix(TWIN_PREFIX).partition(",")
    seed = int(rest.removeprefix("seed=")) if rest else 0
    return name, seed


@dataclass(frozen=True)
class Twin:
    """A generated twin. Its id carries the content digest, so results of a
    re-profiled or regenerated twin never land in an older one's cell."""

    workload_id: str
    gt_path: Path
    dt_path: Path
    profile: dict[str, Any]
    measured_profile_path: Path

    def measured_profile(self) -> dict[str, Any]:
        """The twin's own statistics, extracted once and cached beside it."""
        if self.measured_profile_path.exists():
            return json.loads(self.measured_profile_path.read_text())
        measured = extract(
            self.gt_path, self.dt_path, masks=self.profile["masks"], source=self.workload_id
        )
        self.measured_profile_path.write_text(json.dumps(measured))
        return measured


def materialize(name: str, seed: int) -> Twin:
    """Generate the twin's GT/DT files unless cached."""
    profile = load_profile(name)
    digest = hashlib.sha256(
        json.dumps(profile, sort_keys=True).encode() + f"{seed}:{TWIN_VERSION}".encode()
    ).hexdigest()[:12]
    stem = bench_cache_root() / "twin" / f"{name}_s{seed}_{digest}"
    gt_out, dt_out = Path(f"{stem}_gt.json"), Path(f"{stem}_dt.json")
    if not (gt_out.exists() and dt_out.exists()):
        gt, dts = generate(profile, seed)
        gt_out.parent.mkdir(parents=True, exist_ok=True)
        gt_out.write_text(json.dumps(gt, separators=tuple(profile["json"]["gt_separators"])))
        dt_out.write_text(json.dumps(dts, separators=tuple(profile["json"]["dt_separators"])))
    return Twin(
        workload_id=f"twin_{name}_s{seed}_{digest}",
        gt_path=gt_out,
        dt_path=dt_out,
        profile=profile,
        measured_profile_path=Path(f"{stem}_profile.json"),
    )


SCALAR_TOLERANCE = 0.03
HISTOGRAM_TOLERANCE = 0.05
_MASK_IOUS = frozenset({"segm", "boundary"})
# What the structural gate compares, and for which IoU types (None: all).
# Mask complexity gates only mask cells: bbox evaluation never rasterizes a
# segmentation, so there it shows up only as JSON bytes, gated for every type.
_GATED_HISTOGRAMS: dict[str, frozenset[str] | None] = {
    "cells_per_image": None,
    "cell_gd": None,
    "category_cells": None,
    "gt_log_area": None,
    "gt_log_aspect": None,
    "gt_fill": None,
    "gt_segmentation": None,
    "gt_polygon_parts": None,
    "gt_polygon_vertices": None,
    "gt_crowd_counts_len": None,
    "dt_best_iou": None,
    "dt_score_matched": None,
    "dt_score_unmatched": None,
    "dt_segmentation": None,
    "gt_rle_len": _MASK_IOUS,
    "dt_rle_len": _MASK_IOUS,
}
_MASK_TOTALS = frozenset({"gt_rle_chars", "dt_rle_chars"})


def compare_profiles(
    real: dict[str, Any], twin: dict[str, Any], iou: str
) -> list[tuple[str, str, bool]]:
    """Structural checks for an `iou` cell: (name, detail, passed)."""
    checks: list[tuple[str, str, bool]] = []
    for key, value in real["totals"].items():
        if key in _MASK_TOTALS and iou not in _MASK_IOUS:
            continue
        other = twin["totals"].get(key, 0)
        rel = (other - value) / max(abs(value), 1)
        checks.append(
            (f"total {key}", f"{value} vs {other} ({rel:+.1%})", abs(rel) <= SCALAR_TOLERANCE)
        )
    for key, ious in _GATED_HISTOGRAMS.items():
        if key not in real or (ious is not None and iou not in ious):
            continue
        a = real[key]
        tv = _total_variation(a, twin.get(key) or type(a)())
        checks.append((f"hist {key}", f"TV {tv:.3f}", tv <= HISTOGRAM_TOLERANCE))
    return checks


TIME_TOLERANCE = 0.10
MEMORY_TOLERANCE = 0.10
# Below these, a stage is noise-dominated; compare absolute differences.
_TIME_FLOOR_NS = 20_000_000
_MEMORY_FLOOR_BYTES = 16 << 20


def _load_result(path: Path) -> BenchResult:
    return BenchResult.model_validate(upgrade_schema(json.loads(path.read_bytes())))


def _cell_stats(result: BenchResult) -> dict[str, float]:
    """Median over measured reps of each stage's wall, CPU and peak RSS."""
    reps = [r for r in result.reps if not r.warmup]
    out: dict[str, float] = {}
    for stage in reps[0].stages:
        for field in ("wall_ns", "cpu_ns", "peak_rss_bytes"):
            values = [getattr(r.stages[stage], field) for r in reps if stage in r.stages]
            values = [v for v in values if v is not None]
            if values:
                out[f"{stage}.{field}"] = float(aggregate_stage(values).median_ns)
    return out


def _unstable(result: BenchResult) -> bool:
    """Failed release mode's IQR gate: measured on a busy host."""
    gate = result.aggregation.iqr_gate if result.aggregation else None
    return gate is not None and not gate.passed


def compare_results(real_dir: Path, twin_dir: Path) -> list[tuple[str, str, bool]]:
    """Cost checks between two result cells: every impl, stage and resource."""
    checks: list[tuple[str, str, bool]] = []
    totals: dict[str, tuple[float, float]] = {}
    for real_file in sorted(real_dir.glob("*.json")):
        impl = real_file.stem
        twin_file = twin_dir / real_file.name
        if not twin_file.exists():
            checks.append((f"{impl}", "no twin result", False))
            continue
        results = {"real": _load_result(real_file), "twin": _load_result(twin_file)}
        # Unstable medians can neither confirm nor refute equivalence.
        if unstable := [side for side, r in results.items() if _unstable(r)]:
            detail = f"IQR gate failed on {' and '.join(unstable)}: re-run on a quiet host"
            checks.append((f"{impl} stable measurement", detail, False))
        real, other = _cell_stats(results["real"]), _cell_stats(results["twin"])
        for key, a in sorted(real.items()):
            if key not in other:
                continue
            b = other[key]
            is_memory = key.endswith("peak_rss_bytes")
            tolerance = MEMORY_TOLERANCE if is_memory else TIME_TOLERANCE
            floor = _MEMORY_FLOOR_BYTES if is_memory else _TIME_FLOOR_NS
            rel = (b - a) / a if a else 0.0
            unit, scale = ("MiB", 1 << 20) if is_memory else ("ms", 1e6)
            checks.append(
                (
                    f"{impl} {key}",
                    f"{a / scale:.1f} vs {b / scale:.1f} {unit} ({rel:+.1%})",
                    abs(rel) <= tolerance or abs(b - a) < floor,
                )
            )
        if "total.cpu_ns" in real and "total.cpu_ns" in other:
            totals[impl] = (real["total.cpu_ns"], other["total.cpu_ns"])
    if "vernier" in totals:
        rv, tv = totals["vernier"]
        for impl, (ri, ti) in sorted(totals.items()):
            if impl == "vernier":
                continue
            real_ratio, twin_ratio = ri / rv, ti / tv
            rel = twin_ratio / real_ratio - 1
            checks.append(
                (
                    f"ranking {impl}/vernier",
                    f"{real_ratio:.2f}x vs {twin_ratio:.2f}x ({rel:+.1%})",
                    abs(rel) <= TIME_TOLERANCE,
                )
            )
    return checks
