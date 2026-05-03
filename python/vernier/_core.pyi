from collections.abc import Mapping
from typing import Any, Literal, TypeAlias, TypedDict

import numpy as np
from numpy.typing import NDArray
from typing_extensions import Self

#: LVIS category-frequency tier as a single-letter string (ADR-0026
#: AB1). The `Dataset.category_frequency` accessor returns these
#: values; the user-facing `vernier.Frequency` enum maps to and from
#: them.
LvisFrequencyLiteral: TypeAlias = Literal["r", "c", "f"]

__version__: str

class _UpdateReportDict(TypedDict):
    new_images: int
    new_detections: int
    memory_used_bytes: int
    soft_warn_triggered: bool

_TablesResult: TypeAlias = tuple[
    Summary,
    ArrowRecordBatch | None,  # per_image
    ArrowRecordBatch | None,  # per_class
    ArrowRecordBatch | None,  # per_detection
    ArrowRecordBatch | None,  # per_pair
]

class StreamingEvaluator:
    def __init__(
        self,
        gt_json: bytes,
        *,
        iou_type: Literal["bbox", "segm", "boundary", "keypoints"] = ...,
        parity_mode: Literal["strict", "corrected"] = ...,
        max_dets: list[int] = ...,
        use_cats: bool = ...,
        memory_budget_bytes: int | None = ...,
        dilation_ratio: float = ...,
        sigmas: dict[int, list[float]] | None = ...,
        retain_iou: bool = ...,
    ) -> None: ...
    def update(self, detections: bytes) -> _UpdateReportDict: ...
    def snapshot(self, *, running: bool = ...) -> Summary: ...
    def snapshot_with_tables(
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
    @property
    def images_seen(self) -> int: ...
    @property
    def detections_seen(self) -> int: ...
    @property
    def images_pending(self) -> int: ...
    @property
    def memory_used_bytes(self) -> int: ...
    @property
    def memory_budget_bytes(self) -> int: ...

class OutOfBudgetError(RuntimeError):
    used_bytes: int
    budget_bytes: int
    breakdown: dict[str, int]

class QueueFullError(RuntimeError):
    queue_capacity: int
    timeout: float | None

class MemoryBudgetWarning(UserWarning): ...

class BackgroundEvaluator:
    def __init__(
        self,
        gt_json: bytes,
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
    ) -> None: ...
    def submit(self, detections: bytes, *, timeout: float | None = ...) -> None: ...
    def snapshot(self, *, peek: bool = ...) -> Summary: ...
    def snapshot_with_tables(
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

class Dataset:
    @staticmethod
    def from_json(gt_json: bytes) -> Dataset: ...
    @staticmethod
    def from_lvis_json(gt_json: bytes) -> Dataset: ...
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
    def clear_cache(self) -> None: ...

class ArrowRecordBatch:
    def __arrow_c_array__(self, requested_schema: object | None = ...) -> tuple[object, object]: ...

def version() -> str: ...
def evaluate_bbox_summary(
    gt_json: bytes,
    dt_json: bytes,
    parity_mode: str,
    max_dets: list[int],
    use_cats: bool,
) -> Summary: ...
def evaluate_bbox_summary_with_dataset(
    gt: Dataset,
    dt_json: bytes,
    parity_mode: str,
    max_dets: list[int],
    use_cats: bool,
) -> Summary: ...
def evaluate_bbox_grid(
    gt_json: bytes,
    dt_json: bytes,
    parity_mode: str,
    max_dets_per_image: int,
    use_cats: bool,
    retain_iou: bool = ...,
) -> EvalGrid: ...
def evaluate_segm_summary(
    gt_json: bytes,
    dt_json: bytes,
    parity_mode: str,
    max_dets: list[int],
    use_cats: bool,
) -> Summary: ...
def evaluate_segm_summary_with_dataset(
    gt: Dataset,
    dt_json: bytes,
    parity_mode: str,
    max_dets: list[int],
    use_cats: bool,
) -> Summary: ...
def evaluate_segm_grid(
    gt_json: bytes,
    dt_json: bytes,
    parity_mode: str,
    max_dets_per_image: int,
    use_cats: bool,
    retain_iou: bool = ...,
) -> EvalGrid: ...
def evaluate_boundary_summary(
    gt_json: bytes,
    dt_json: bytes,
    parity_mode: str,
    max_dets: list[int],
    use_cats: bool,
    dilation_ratio: float,
) -> Summary: ...
def evaluate_boundary_summary_with_dataset(
    gt: Dataset,
    dt_json: bytes,
    parity_mode: str,
    max_dets: list[int],
    use_cats: bool,
    dilation_ratio: float,
) -> Summary: ...
def evaluate_boundary_grid(
    gt_json: bytes,
    dt_json: bytes,
    parity_mode: str,
    max_dets_per_image: int,
    use_cats: bool,
    dilation_ratio: float,
    retain_iou: bool = ...,
) -> EvalGrid: ...
def evaluate_keypoints_summary(
    gt_json: bytes,
    dt_json: bytes,
    parity_mode: str,
    max_dets: list[int],
    use_cats: bool,
    sigmas: dict[int, list[float]],
) -> Summary: ...
def evaluate_keypoints_summary_with_dataset(
    gt: Dataset,
    dt_json: bytes,
    parity_mode: str,
    max_dets: list[int],
    use_cats: bool,
    sigmas: dict[int, list[float]],
) -> Summary: ...
def evaluate_keypoints_grid(
    gt_json: bytes,
    dt_json: bytes,
    parity_mode: str,
    max_dets_per_image: int,
    use_cats: bool,
    sigmas: dict[int, list[float]],
    retain_iou: bool = ...,
) -> EvalGrid: ...

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
    dataset: Dataset,
) -> ArrowRecordBatch: ...
def per_image_to_arrow_pycapsule(
    grid: EvalGrid,
    dataset: Dataset,
) -> ArrowRecordBatch: ...
def per_detection_to_arrow_pycapsule(
    grid: EvalGrid,
    dt_json: bytes,
    with_geometry: bool = ...,
) -> ArrowRecordBatch: ...
def per_pair_to_arrow_pycapsule(
    grid: EvalGrid,
    iou_floor: float = ...,
    max_rows: int = ...,
) -> ArrowRecordBatch: ...
