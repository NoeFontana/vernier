import os
from collections.abc import Mapping, Sequence
from types import TracebackType
from typing import Any, Literal, TypeAlias, TypedDict

import numpy as np
from numpy.typing import NDArray
from typing_extensions import Self

from vernier._array_types import CompressedRLE as CompressedRLE
from vernier._array_types import Detections as Detections
from vernier._array_types import DetectionsInput as DetectionsInput
from vernier._array_types import RLEInput as RLEInput
from vernier._array_types import UncompressedRLE as UncompressedRLE

#: LVIS category-frequency tier as a single-letter string (ADR-0026
#: AB1). The `CocoDataset.category_frequency` accessor returns these
#: values; the user-facing `vernier.Frequency` enum maps to and from
#: them.
LvisFrequencyLiteral: TypeAlias = Literal["r", "c", "f"]

__version__: str

_TablesResult: TypeAlias = tuple[
    Summary,
    ArrowRecordBatch | None,  # per_image
    ArrowRecordBatch | None,  # per_class
    ArrowRecordBatch | None,  # per_detection
    ArrowRecordBatch | None,  # per_pair
]

class OutOfBudgetError(RuntimeError):
    used_bytes: int
    budget_bytes: int
    breakdown: dict[str, int]

class QueueFullError(RuntimeError):
    queue_capacity: int
    timeout: float | None

class MemoryBudgetWarning(UserWarning): ...

class PartialFormatMismatch(RuntimeError):
    kind: Literal[
        "too_short",
        "wrong_magic",
        "wrong_version",
        "crc",
        "paradigm_mismatch",
        "kernel_mismatch",
        "grid_mismatch",
        "parity_mismatch",
        "rkyv_decode",
    ]

class PartialDatasetMismatch(RuntimeError):
    expected: bytes
    actual: bytes

class PartialParamsMismatch(RuntimeError):
    expected: bytes
    actual: bytes

class PartialPartitionOverlap(RuntimeError):
    rank_a: int
    rank_b: int
    image_id: int

class PartialRankCollision(RuntimeError):
    rank_id: int

class BackgroundEvaluator:
    def __init__(
        self,
        gt: bytes | CocoDataset,
        *,
        iou_type: Literal["bbox", "segm", "boundary", "keypoints"] = ...,
        parity_mode: Literal["strict", "corrected"] = ...,
        max_dets: list[int] = ...,
        use_cats: bool = ...,
        memory_budget_bytes: int | None = ...,
        dilation_ratio: float = ...,
        sigmas: dict[int, list[float]] | None = ...,
        queue_capacity: int = ...,
        worker_affinity: int | None = ...,
        worker_nice: int = ...,
        shutdown_timeout_seconds: float = ...,
        retain_iou: bool = ...,
        cast_inputs: bool = ...,
        rank_id: int | None = ...,
        record_latency_samples: bool = ...,
    ) -> None: ...
    def finalize_to_partial(self) -> bytes: ...
    def submit(self, detections: DetectionsInput, *, timeout: float | None = ...) -> None: ...
    def finalize(self) -> Summary: ...
    def finalize_with_tables(
        self,
        *,
        per_image: bool = ...,
        per_class: bool = ...,
        per_detection: bool = ...,
        per_pair: bool = ...,
        per_pair_iou_floor: float = ...,
        per_pair_max_rows: int = ...,
        per_detection_with_geometry: bool = ...,
    ) -> _TablesResult: ...
    def finalize_with_cells(self) -> tuple[Summary, EvalCells]: ...
    def __enter__(self) -> Self: ...
    def __exit__(self, exc_type: object, exc: object, tb: object) -> None: ...
    @property
    def images_seen(self) -> int: ...
    @property
    def detections_seen(self) -> int: ...
    @property
    def queue_depth(self) -> int: ...
    @property
    def memory_used_bytes(self) -> int: ...
    def drain_latency_samples_ns(self) -> list[int]: ...

class Breakdown:
    @classmethod
    def from_ranges(cls, axis: str, buckets: Sequence[tuple[str, float, float]]) -> Breakdown: ...
    @classmethod
    def from_class_groups(
        cls, axis: str, groups: Sequence[tuple[str, Sequence[int]]]
    ) -> Breakdown: ...
    @property
    def axis(self) -> str: ...
    @property
    def kind(self) -> Literal["range", "class_groups"]: ...
    @property
    def buckets(self) -> list[tuple[str, float, float]]: ...
    @property
    def class_groups(self) -> list[tuple[str, list[int]]]: ...
    def __len__(self) -> int: ...
    def __eq__(self, other: object) -> bool: ...
    def __hash__(self) -> int: ...

class Summary:
    @property
    def stats(self) -> list[float]: ...
    def pretty_lines(self) -> list[str]: ...

class EvalGrid:
    @property
    def n_categories(self) -> int: ...
    @property
    def n_area_ranges(self) -> int: ...
    @property
    def n_images(self) -> int: ...
    def eval_imgs(self) -> list[dict[str, Any] | None]: ...
    def accumulate(self, max_dets: list[int]) -> Accumulated: ...
    def dataset(self) -> CocoDataset: ...

class Accumulated:
    @property
    def precision(self) -> NDArray[np.float64]: ...
    @property
    def recall(self) -> NDArray[np.float64]: ...
    @property
    def scores(self) -> NDArray[np.float64]: ...
    @property
    def counts(self) -> list[int]: ...
    def summarize(
        self,
        max_dets: list[int] | None = ...,
        *,
        plan: Literal["detection", "keypoints"] | None = ...,
    ) -> Summary: ...
    def summarize_lvis(
        self,
        gt: CocoDataset,
        max_dets: list[int] | None = ...,
    ) -> Summary: ...

class CocoDataset:
    @staticmethod
    def from_json(gt_json: bytes) -> CocoDataset: ...
    @staticmethod
    def from_lvis_json(gt_json: bytes) -> CocoDataset: ...
    @property
    def num_annotations(self) -> int: ...
    @property
    def num_images(self) -> int: ...
    @property
    def num_categories(self) -> int: ...
    @property
    def is_federated(self) -> bool: ...
    @property
    def pos_category_ids(self) -> Mapping[int, frozenset[int]] | None: ...
    @property
    def neg_category_ids(self) -> Mapping[int, frozenset[int]] | None: ...
    @property
    def not_exhaustive_category_ids(self) -> Mapping[int, frozenset[int]] | None: ...
    @property
    def category_frequency(self) -> Mapping[int, LvisFrequencyLiteral] | None: ...
    @property
    def boundary_cache_len(self) -> int: ...
    @property
    def segm_cache_len(self) -> int: ...
    def clear_cache(self) -> None: ...

class ArrowRecordBatch:
    def __arrow_c_array__(self, requested_schema: object | None = ...) -> tuple[object, object]: ...

# ADR-0018 calibration: opaque cell-store handle. Constructed via
# `cells_from_grid(grid)`; consumed by `EvalResult.calibration(...)`.

class EvalCells:
    def iou_to_index(self, iou: float) -> int: ...
    def calibrate(
        self,
        iou_index: int,
        n_bins: int,
        binning: Literal["quantile", "equal_width"],
        min_score: float,
        confidence: Literal["wilson", "clopper_pearson"],
        per_class: bool,
        per_class_aggregation: Literal["macro", "micro"],
    ) -> tuple[float, float, int, int, ArrowRecordBatch, ArrowRecordBatch | None]: ...
    @staticmethod
    def from_python_cells(cells_json: Mapping[str, object]) -> EvalCells: ...

def cells_from_grid(grid: EvalGrid) -> EvalCells: ...
def version() -> str: ...
def evaluate_bbox_summary(
    gt_json: bytes,
    dt: DetectionsInput,
    parity_mode: str,
    max_dets: list[int],
    use_cats: bool,
    cast_inputs: bool = ...,
) -> Summary: ...
def evaluate_instance_to_partial(
    gt_json: bytes,
    detections: DetectionsInput,
    iou_type: Literal["bbox", "segm", "boundary", "keypoints"],
    rank_id: int,
    *,
    parity_mode: Literal["strict", "corrected"] = ...,
    max_dets: list[int] = ...,
    use_cats: bool = ...,
    memory_budget_bytes: int | None = ...,
    dilation_ratio: float = ...,
    sigmas: dict[int, list[float]] | None = ...,
    retain_iou: bool = ...,
    cast_inputs: bool = ...,
) -> bytes: ...
def merge_instance_partials(
    gt_json: bytes,
    partials: Sequence[bytes],
    iou_type: Literal["bbox", "segm", "boundary", "keypoints"],
    *,
    parity_mode: Literal["strict", "corrected"] = ...,
    max_dets: list[int] = ...,
    use_cats: bool = ...,
    memory_budget_bytes: int | None = ...,
    dilation_ratio: float = ...,
    sigmas: dict[int, list[float]] | None = ...,
    retain_iou: bool = ...,
    cast_inputs: bool = ...,
) -> Summary: ...
def evaluate_bbox_summary_with_dataset(
    gt: CocoDataset,
    dt: DetectionsInput,
    parity_mode: str,
    max_dets: list[int],
    use_cats: bool,
    cast_inputs: bool = ...,
) -> Summary: ...
def evaluate_bbox_grid(
    gt_json: bytes,
    dt: DetectionsInput,
    parity_mode: str,
    max_dets_per_image: int,
    use_cats: bool,
    retain_iou: bool = ...,
    cast_inputs: bool = ...,
    iou_thresholds: list[float] | None = ...,
    recall_thresholds: list[float] | None = ...,
    area_ranges: Breakdown | None = ...,
) -> EvalGrid: ...
def evaluate_bbox_grid_with_dataset(
    gt: CocoDataset,
    dt: DetectionsInput,
    parity_mode: str,
    max_dets_per_image: int,
    use_cats: bool,
    retain_iou: bool = ...,
    cast_inputs: bool = ...,
    iou_thresholds: list[float] | None = ...,
    recall_thresholds: list[float] | None = ...,
    area_ranges: Breakdown | None = ...,
) -> EvalGrid: ...
def evaluate_segm_summary(
    gt_json: bytes,
    dt: DetectionsInput,
    parity_mode: str,
    max_dets: list[int],
    use_cats: bool,
    cast_inputs: bool = ...,
) -> Summary: ...
def evaluate_segm_summary_with_dataset(
    gt: CocoDataset,
    dt: DetectionsInput,
    parity_mode: str,
    max_dets: list[int],
    use_cats: bool,
    cast_inputs: bool = ...,
) -> Summary: ...
def evaluate_segm_grid(
    gt_json: bytes,
    dt: DetectionsInput,
    parity_mode: str,
    max_dets_per_image: int,
    use_cats: bool,
    retain_iou: bool = ...,
    cast_inputs: bool = ...,
    iou_thresholds: list[float] | None = ...,
    recall_thresholds: list[float] | None = ...,
    area_ranges: Breakdown | None = ...,
) -> EvalGrid: ...
def evaluate_boundary_summary(
    gt_json: bytes,
    dt: DetectionsInput,
    parity_mode: str,
    max_dets: list[int],
    use_cats: bool,
    dilation_ratio: float,
    cast_inputs: bool = ...,
) -> Summary: ...
def evaluate_boundary_summary_with_dataset(
    gt: CocoDataset,
    dt: DetectionsInput,
    parity_mode: str,
    max_dets: list[int],
    use_cats: bool,
    dilation_ratio: float,
    cast_inputs: bool = ...,
) -> Summary: ...
def evaluate_boundary_grid(
    gt_json: bytes,
    dt: DetectionsInput,
    parity_mode: str,
    max_dets_per_image: int,
    use_cats: bool,
    dilation_ratio: float,
    retain_iou: bool = ...,
    cast_inputs: bool = ...,
    iou_thresholds: list[float] | None = ...,
    recall_thresholds: list[float] | None = ...,
    area_ranges: Breakdown | None = ...,
) -> EvalGrid: ...
def evaluate_keypoints_summary(
    gt_json: bytes,
    dt: DetectionsInput,
    parity_mode: str,
    max_dets: list[int],
    use_cats: bool,
    sigmas: dict[int, list[float]],
    cast_inputs: bool = ...,
) -> Summary: ...
def evaluate_keypoints_summary_with_dataset(
    gt: CocoDataset,
    dt: DetectionsInput,
    parity_mode: str,
    max_dets: list[int],
    use_cats: bool,
    sigmas: dict[int, list[float]],
    cast_inputs: bool = ...,
) -> Summary: ...
def evaluate_keypoints_grid(
    gt_json: bytes,
    dt: DetectionsInput,
    parity_mode: str,
    max_dets_per_image: int,
    use_cats: bool,
    sigmas: dict[int, list[float]],
    cast_inputs: bool = ...,
    iou_thresholds: list[float] | None = ...,
    recall_thresholds: list[float] | None = ...,
    area_ranges: Breakdown | None = ...,
) -> EvalGrid: ...

# ADR-0046 partitioned-eval surface.

class PartitionedSummary:
    @property
    def overall(self) -> Summary: ...
    @property
    def overall_n_images(self) -> int: ...
    @property
    def overall_n_detections(self) -> int: ...
    @property
    def n_slices(self) -> int: ...
    def slices_capsule(self) -> ArrowRecordBatch: ...

def evaluate_bbox_partitioned(
    gt_json: bytes,
    dt: DetectionsInput,
    parity_mode: str,
    max_dets_per_image: int,
    use_cats: bool,
    manifest: object,
    cast_inputs: bool = ...,
    iou_thresholds: list[float] | None = ...,
    recall_thresholds: list[float] | None = ...,
    area_ranges: Breakdown | None = ...,
    cross_axes: list[list[str]] | None = ...,
    key_kind: str = ...,
) -> PartitionedSummary: ...
def evaluate_segm_partitioned(
    gt_json: bytes,
    dt: DetectionsInput,
    parity_mode: str,
    max_dets_per_image: int,
    use_cats: bool,
    manifest: object,
    cast_inputs: bool = ...,
    iou_thresholds: list[float] | None = ...,
    recall_thresholds: list[float] | None = ...,
    area_ranges: Breakdown | None = ...,
    cross_axes: list[list[str]] | None = ...,
    key_kind: str = ...,
) -> PartitionedSummary: ...
def evaluate_boundary_partitioned(
    gt_json: bytes,
    dt: DetectionsInput,
    parity_mode: str,
    max_dets_per_image: int,
    use_cats: bool,
    dilation_ratio: float,
    manifest: object,
    cast_inputs: bool = ...,
    iou_thresholds: list[float] | None = ...,
    recall_thresholds: list[float] | None = ...,
    area_ranges: Breakdown | None = ...,
    cross_axes: list[list[str]] | None = ...,
    key_kind: str = ...,
) -> PartitionedSummary: ...
def evaluate_keypoints_partitioned(
    gt_json: bytes,
    dt: DetectionsInput,
    parity_mode: str,
    max_dets_per_image: int,
    use_cats: bool,
    sigmas: dict[int, list[float]],
    manifest: object,
    cast_inputs: bool = ...,
    iou_thresholds: list[float] | None = ...,
    recall_thresholds: list[float] | None = ...,
    area_ranges: Breakdown | None = ...,
    cross_axes: list[list[str]] | None = ...,
    key_kind: str = ...,
) -> PartitionedSummary: ...

class PartitionedLrpReport:
    @property
    def overall(self) -> _LrpReportDict: ...
    @property
    def overall_n_images(self) -> int: ...
    @property
    def overall_n_detections(self) -> int: ...
    @property
    def n_slices(self) -> int: ...
    def slices_capsule(self) -> ArrowRecordBatch: ...

def evaluate_bbox_partitioned_lrp(
    gt_bytes: bytes,
    dt_bytes: bytes,
    parity_mode: str,
    tp_threshold: float,
    tau_grid: list[float],
    max_dets_per_image: int,
    use_cats: bool,
    manifest: object,
    cross_axes: list[list[str]] | None = ...,
    key_kind: str = ...,
) -> PartitionedLrpReport: ...
def evaluate_segm_partitioned_lrp(
    gt_bytes: bytes,
    dt_bytes: bytes,
    parity_mode: str,
    tp_threshold: float,
    tau_grid: list[float],
    max_dets_per_image: int,
    use_cats: bool,
    manifest: object,
    cross_axes: list[list[str]] | None = ...,
    key_kind: str = ...,
) -> PartitionedLrpReport: ...
def evaluate_boundary_partitioned_lrp(
    gt_bytes: bytes,
    dt_bytes: bytes,
    parity_mode: str,
    tp_threshold: float,
    tau_grid: list[float],
    max_dets_per_image: int,
    use_cats: bool,
    dilation_ratio: float,
    manifest: object,
    cross_axes: list[list[str]] | None = ...,
    key_kind: str = ...,
) -> PartitionedLrpReport: ...
def evaluate_keypoints_partitioned_lrp(
    gt_bytes: bytes,
    dt_bytes: bytes,
    parity_mode: str,
    tp_threshold: float,
    tau_grid: list[float],
    max_dets_per_image: int,
    use_cats: bool,
    sigmas: dict[int, list[float]],
    manifest: object,
    cross_axes: list[list[str]] | None = ...,
    key_kind: str = ...,
) -> PartitionedLrpReport: ...
def slices_batch_panoptic(
    rows: list[tuple[str, str, int, int, float, float, float]],
) -> ArrowRecordBatch: ...
def slices_batch_semantic(
    rows: list[tuple[str, str, int, int, float, float, float, float]],
) -> ArrowRecordBatch: ...
def manifest_to_json_bytes(manifest: object, key_kind: str = ...) -> bytes: ...

class _TideDeltaDict(TypedDict):
    cls: float
    loc: float
    both: float
    dupe: float
    bkg: float
    missed: float

class _TideConfigDict(TypedDict):
    t_f: float
    t_b: float
    kernel: str

class _TideReportDict(TypedDict):
    baseline_map: float
    delta: _TideDeltaDict
    delta_all_fp_removed: float
    config: _TideConfigDict

def error_decomposition_bbox(
    gt_bytes: bytes,
    dt_bytes: bytes,
    parity_mode: str,
    t_f: float,
    t_b: float,
    max_dets_per_image: int,
    use_cats: bool,
) -> _TideReportDict: ...
def error_decomposition_segm(
    gt_bytes: bytes,
    dt_bytes: bytes,
    parity_mode: str,
    t_f: float,
    t_b: float,
    max_dets_per_image: int,
    use_cats: bool,
) -> _TideReportDict: ...
def error_decomposition_boundary(
    gt_bytes: bytes,
    dt_bytes: bytes,
    parity_mode: str,
    t_f: float,
    t_b: float,
    max_dets_per_image: int,
    use_cats: bool,
    dilation_ratio: float,
) -> _TideReportDict: ...

class _FpIouHistogramDict(TypedDict):
    iou_same: NDArray[np.float64]
    iou_cross: NDArray[np.float64]
    kernel: str
    t_f: float
    n_total_dts: int
    n_fps: int

def fp_iou_histogram_bbox(
    gt_bytes: bytes,
    dt_bytes: bytes,
    parity_mode: str,
    t_f: float,
    max_dets_per_image: int,
    use_cats: bool,
) -> _FpIouHistogramDict: ...
def fp_iou_histogram_segm(
    gt_bytes: bytes,
    dt_bytes: bytes,
    parity_mode: str,
    t_f: float,
    max_dets_per_image: int,
    use_cats: bool,
) -> _FpIouHistogramDict: ...
def fp_iou_histogram_boundary(
    gt_bytes: bytes,
    dt_bytes: bytes,
    parity_mode: str,
    t_f: float,
    max_dets_per_image: int,
    use_cats: bool,
    dilation_ratio: float,
) -> _FpIouHistogramDict: ...

class _LrpPerClassDict(TypedDict):
    category_id: int
    olrp: float | None
    olrp_loc: float | None
    olrp_fp: float | None
    olrp_fn: float | None
    tau: float | None

class _LrpConfigDict(TypedDict):
    tp_threshold: float
    tau_grid_len: int
    kernel: str

class _LrpReportDict(TypedDict):
    olrp: float
    loc: float
    fp: float
    fn: float
    per_class: list[_LrpPerClassDict]
    n_empty_classes: int
    config: _LrpConfigDict

def optimal_lrp_bbox(
    gt_bytes: bytes,
    dt_bytes: bytes,
    parity_mode: str,
    tp_threshold: float,
    tau_grid: list[float],
    max_dets_per_image: int,
    use_cats: bool,
) -> _LrpReportDict: ...
def optimal_lrp_segm(
    gt_bytes: bytes,
    dt_bytes: bytes,
    parity_mode: str,
    tp_threshold: float,
    tau_grid: list[float],
    max_dets_per_image: int,
    use_cats: bool,
) -> _LrpReportDict: ...
def optimal_lrp_boundary(
    gt_bytes: bytes,
    dt_bytes: bytes,
    parity_mode: str,
    tp_threshold: float,
    tau_grid: list[float],
    max_dets_per_image: int,
    use_cats: bool,
    dilation_ratio: float,
) -> _LrpReportDict: ...
def optimal_lrp_keypoints(
    gt_bytes: bytes,
    dt_bytes: bytes,
    parity_mode: str,
    tp_threshold: float,
    tau_grid: list[float],
    max_dets_per_image: int,
    use_cats: bool,
    sigmas: dict[int, list[float]],
) -> _LrpReportDict: ...
def lrp_default_tau_grid() -> list[float]: ...

class _ConfusionMatrixDict(TypedDict):
    gt_class: list[str]
    dt_class: list[str]
    count: list[int]
    iou_threshold: float
    kernel: str

def confusion_matrix_bbox(
    gt_bytes: bytes,
    dt_bytes: bytes,
    parity_mode: str,
    iou_threshold: float,
    max_dets_per_image: int,
    use_cats: bool,
) -> _ConfusionMatrixDict: ...
def confusion_matrix_segm(
    gt_bytes: bytes,
    dt_bytes: bytes,
    parity_mode: str,
    iou_threshold: float,
    max_dets_per_image: int,
    use_cats: bool,
) -> _ConfusionMatrixDict: ...
def confusion_matrix_boundary(
    gt_bytes: bytes,
    dt_bytes: bytes,
    parity_mode: str,
    iou_threshold: float,
    max_dets_per_image: int,
    use_cats: bool,
    dilation_ratio: float,
) -> _ConfusionMatrixDict: ...
def per_class_to_arrow_pycapsule(
    grid: EvalGrid,
    accum: Accumulated,
    dataset: CocoDataset,
) -> ArrowRecordBatch: ...
def per_image_to_arrow_pycapsule(
    grid: EvalGrid,
    dataset: CocoDataset,
) -> ArrowRecordBatch: ...
def per_detection_to_arrow_pycapsule(
    grid: EvalGrid,
    with_geometry: bool = ...,
) -> ArrowRecordBatch: ...
def per_pair_to_arrow_pycapsule(
    grid: EvalGrid,
    iou_floor: float = ...,
    max_rows: int = ...,
) -> ArrowRecordBatch: ...
def panoptic_per_class_to_arrow_pycapsule(
    summary: PanopticSummary,
) -> ArrowRecordBatch: ...
def semantic_per_class_to_arrow_pycapsule(
    summary: SemanticSummary,
) -> ArrowRecordBatch: ...

# Panoptic-quality (ADR-0025).

class ClassPanopticStats:
    @property
    def pq(self) -> float: ...
    @property
    def sq(self) -> float: ...
    @property
    def rq(self) -> float: ...
    @property
    def n_tp(self) -> int: ...
    @property
    def n_fp(self) -> int: ...
    @property
    def n_fn(self) -> int: ...
    @property
    def iou_sum(self) -> float: ...

class GroupPanopticStats:
    @property
    def label(self) -> str: ...
    @property
    def member_category_ids(self) -> list[int]: ...
    @property
    def pq(self) -> float: ...
    @property
    def sq(self) -> float: ...
    @property
    def rq(self) -> float: ...
    @property
    def n(self) -> int: ...

class PanopticSummary:
    @property
    def pq(self) -> float: ...
    @property
    def sq(self) -> float: ...
    @property
    def rq(self) -> float: ...
    @property
    def pq_things(self) -> float | None: ...
    @property
    def sq_things(self) -> float | None: ...
    @property
    def rq_things(self) -> float | None: ...
    @property
    def pq_stuff(self) -> float | None: ...
    @property
    def sq_stuff(self) -> float | None: ...
    @property
    def rq_stuff(self) -> float | None: ...
    @property
    def n(self) -> int: ...
    @property
    def n_things(self) -> int | None: ...
    @property
    def n_stuff(self) -> int | None: ...
    def per_class(self) -> dict[int, ClassPanopticStats]: ...
    def per_group(self) -> dict[str, GroupPanopticStats]: ...
    def to_dict(self, *, strict: bool = ...) -> dict[str, Any]: ...

class PanopticDataset:
    @staticmethod
    def from_arrays(
        label_maps: dict[int, NDArray[np.uint32]],
        segments_info: bytes,
        categories: bytes,
    ) -> PanopticDataset: ...
    @property
    def num_images(self) -> int: ...
    @property
    def num_categories(self) -> int: ...
    def image_ids(self) -> list[int]: ...
    def subset_by_image_ids(self, ids: list[int]) -> PanopticDataset: ...

class PanopticPredictions:
    @staticmethod
    def from_arrays(
        label_maps: dict[int, NDArray[np.uint32]],
        segments_info: bytes,
    ) -> PanopticPredictions: ...
    @property
    def num_images(self) -> int: ...
    @property
    def num_segments(self) -> int: ...
    def image_ids(self) -> list[int]: ...
    def num_segments_for(self, ids: list[int]) -> int: ...
    def subset_by_image_ids(self, ids: list[int]) -> PanopticPredictions: ...

def evaluate_panoptic(
    gt: PanopticDataset,
    dt: PanopticPredictions,
    parity_mode: str,
    things_stuff_split: bool = ...,
    *,
    pq_iou_threshold: float | None = ...,
    category_filter: list[int] | None = ...,
    class_grouping: list[tuple[str, list[int]]] | None = ...,
    stuff_thing_partition: tuple[list[int], list[int]] | None = ...,
    boundary: bool = ...,
    dilation_ratio: float = ...,
) -> PanopticSummary: ...
def evaluate_panoptic_to_partial(
    images: Sequence[tuple[int, NDArray[np.uint32], bytes, NDArray[np.uint32], bytes]],
    categories: bytes,
    parity_mode: str,
    rank_id: int,
    *,
    things_stuff_split: bool = ...,
    retain_per_image_deltas: bool = ...,
    boundary: bool = ...,
    dilation_ratio: float = ...,
) -> bytes: ...
def merge_panoptic_partials(
    categories: bytes,
    partials: Sequence[bytes],
    parity_mode: str,
    *,
    things_stuff_split: bool = ...,
    retain_per_image_deltas: bool = ...,
    boundary: bool = ...,
    dilation_ratio: float = ...,
) -> PanopticSummary: ...

class PartitionedPanopticReport:
    @property
    def overall(self) -> PanopticSummary: ...
    @property
    def overall_n_images(self) -> int: ...
    @property
    def overall_n_detections(self) -> int: ...
    @property
    def n_slices(self) -> int: ...
    def slices_capsule(self) -> ArrowRecordBatch: ...

def evaluate_panoptic_partitioned(
    gt: PanopticDataset,
    dt: PanopticPredictions,
    parity_mode: str,
    things_stuff_split: bool,
    boundary: bool,
    dilation_ratio: float,
    manifest: object,
    cross_axes: list[list[str]] | None = ...,
    key_kind: str = ...,
) -> PartitionedPanopticReport: ...

# ---------------------------------------------------------------------------
# Semantic-segmentation surface (ADR-0028).
# ---------------------------------------------------------------------------

class ClassSemanticStats:
    @property
    def class_id(self) -> int: ...
    @property
    def iou(self) -> float: ...
    @property
    def accuracy(self) -> float: ...
    @property
    def precision(self) -> float: ...
    @property
    def n_gt_pixels(self) -> int: ...
    @property
    def n_dt_pixels(self) -> int: ...

class ConfusionMatrix:
    @property
    def n_classes(self) -> int: ...
    @property
    def total(self) -> int: ...
    @property
    def trace(self) -> int: ...
    def get(self, g: int, d: int) -> int: ...
    def counts(self) -> NDArray[np.uint64]: ...

class GroupSemanticStats:
    @property
    def label(self) -> str: ...
    @property
    def member_class_ids(self) -> list[int]: ...
    @property
    def miou(self) -> float: ...
    @property
    def mean_accuracy(self) -> float: ...
    @property
    def pixel_accuracy(self) -> float: ...
    @property
    def fwiou(self) -> float: ...

class SemanticSummary:
    @property
    def miou(self) -> float: ...
    @property
    def fwiou(self) -> float: ...
    @property
    def pixel_accuracy(self) -> float: ...
    @property
    def mean_accuracy(self) -> float: ...
    @property
    def confusion_matrix(self) -> ConfusionMatrix: ...
    def per_class(self) -> dict[int, ClassSemanticStats]: ...
    def per_group(self) -> dict[str, GroupSemanticStats]: ...

def evaluate_semantic_from_arrays(
    gt_label_maps: dict[int, NDArray[np.unsignedinteger[Any]]],
    dt_label_maps: dict[int, NDArray[np.unsignedinteger[Any]]],
    n_classes: int,
    parity_mode: str,
    *,
    ignore_label: int | None = ...,
    label_remap: dict[int, int] | None = ...,
    class_filter: list[int] | None = ...,
    class_grouping: list[tuple[str, list[int]]] | None = ...,
) -> SemanticSummary: ...
def evaluate_semantic_from_pngs(
    gt_paths: dict[int, str | os.PathLike[str]],
    dt_paths: dict[int, str | os.PathLike[str]],
    n_classes: int,
    parity_mode: str,
    *,
    ignore_label: int | None = ...,
) -> SemanticSummary: ...
def evaluate_semantic_to_partial(
    gt_label_maps: dict[int, NDArray[np.unsignedinteger[Any]]],
    dt_label_maps: dict[int, NDArray[np.unsignedinteger[Any]]],
    n_classes: int,
    parity_mode: str,
    rank_id: int,
    *,
    ignore_label: int | None = ...,
) -> bytes: ...
def merge_semantic_partials(
    n_classes: int,
    partials: Sequence[bytes],
    parity_mode: str,
    *,
    ignore_label: int | None = ...,
) -> SemanticSummary: ...

class PartitionedSemanticReport:
    @property
    def overall(self) -> SemanticSummary: ...
    @property
    def overall_n_images(self) -> int: ...
    @property
    def overall_n_detections(self) -> int: ...
    @property
    def n_slices(self) -> int: ...
    def slices_capsule(self) -> ArrowRecordBatch: ...

def evaluate_semantic_partitioned(
    gt_label_maps: dict[int, NDArray[np.unsignedinteger[Any]]],
    dt_label_maps: dict[int, NDArray[np.unsignedinteger[Any]]],
    n_classes: int,
    parity_mode: str,
    manifest: object,
    *,
    ignore_label: int | None = ...,
    label_remap: dict[int, int] | None = ...,
    class_filter: list[int] | None = ...,
    class_grouping: list[tuple[str, list[int]]] | None = ...,
    cross_axes: list[list[str]] | None = ...,
    key_kind: str = ...,
) -> PartitionedSemanticReport: ...

class BackgroundSemanticEvaluator:
    def __init__(
        self,
        n_classes: int,
        parity_mode: str,
        *,
        ignore_label: int | None = ...,
        rank_id: int | None = ...,
        queue_capacity: int = ...,
        worker_affinity: int | None = ...,
        worker_nice: int = ...,
        shutdown_timeout_seconds: float = ...,
    ) -> None: ...
    @property
    def n_classes(self) -> int: ...
    @property
    def n_images(self) -> int: ...
    @property
    def queue_depth(self) -> int: ...
    def submit(
        self,
        image_id: int,
        gt: NDArray[np.unsignedinteger[Any]],
        dt: NDArray[np.unsignedinteger[Any]],
        *,
        timeout: float | None = ...,
    ) -> None: ...
    def submit_png(
        self,
        image_id: int,
        gt_png_bytes: bytes,
        dt_png_bytes: bytes,
        *,
        timeout: float | None = ...,
    ) -> None: ...
    def finalize(self) -> SemanticSummary: ...
    def finalize_to_partial(self) -> bytes: ...
    def __enter__(self) -> Self: ...
    def __exit__(
        self,
        exc_type: type[BaseException] | None = ...,
        exc: BaseException | None = ...,
        tb: TracebackType | None = ...,
    ) -> None: ...

class BackgroundPanopticEvaluator:
    def __init__(
        self,
        categories: bytes,
        parity_mode: str,
        *,
        things_stuff_split: bool = ...,
        retain_per_image_deltas: bool = ...,
        rank_id: int | None = ...,
        queue_capacity: int = ...,
        worker_affinity: int | None = ...,
        worker_nice: int = ...,
        shutdown_timeout_seconds: float = ...,
        boundary: bool = ...,
        dilation_ratio: float = ...,
    ) -> None: ...
    @property
    def n_categories(self) -> int: ...
    @property
    def n_images(self) -> int: ...
    @property
    def queue_depth(self) -> int: ...
    def submit(
        self,
        image_id: int,
        gt_label_map: NDArray[np.uint32],
        gt_segments_info: bytes,
        dt_label_map: NDArray[np.uint32],
        dt_segments_info: bytes,
        *,
        timeout: float | None = ...,
    ) -> None: ...
    def submit_png(
        self,
        image_id: int,
        gt_png_bytes: bytes,
        gt_segments_info: bytes,
        dt_png_bytes: bytes,
        dt_segments_info: bytes,
        *,
        timeout: float | None = ...,
    ) -> None: ...
    def finalize(self) -> PanopticSummary: ...
    def finalize_to_partial(self) -> bytes: ...
    def __enter__(self) -> Self: ...
    def __exit__(
        self,
        exc_type: type[BaseException] | None = ...,
        exc: BaseException | None = ...,
        tb: TracebackType | None = ...,
    ) -> None: ...
