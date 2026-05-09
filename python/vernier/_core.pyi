import os
from collections.abc import Mapping, Sequence
from types import TracebackType
from typing import Any, Literal, TypeAlias, TypedDict

import numpy as np
from numpy.typing import NDArray
from typing_extensions import Self

from vernier._array_types import RLE as RLE
from vernier._array_types import Detections as Detections
from vernier._array_types import DetectionsInput as DetectionsInput

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
    def clear_cache(self) -> None: ...

class ArrowRecordBatch:
    def __arrow_c_array__(self, requested_schema: object | None = ...) -> tuple[object, object]: ...

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
) -> EvalGrid: ...
def evaluate_bbox_grid_with_dataset(
    gt: CocoDataset,
    dt: DetectionsInput,
    parity_mode: str,
    max_dets_per_image: int,
    use_cats: bool,
    retain_iou: bool = ...,
    cast_inputs: bool = ...,
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

class PanopticPredictions:
    @staticmethod
    def from_arrays(
        label_maps: dict[int, NDArray[np.uint32]],
        segments_info: bytes,
    ) -> PanopticPredictions: ...
    @property
    def num_images(self) -> int: ...

def evaluate_panoptic(
    gt: PanopticDataset,
    dt: PanopticPredictions,
    parity_mode: str,
    things_stuff_split: bool = ...,
) -> PanopticSummary: ...
def evaluate_panoptic_to_partial(
    images: Sequence[tuple[int, NDArray[np.uint32], bytes, NDArray[np.uint32], bytes]],
    categories: bytes,
    parity_mode: str,
    rank_id: int,
    *,
    things_stuff_split: bool = ...,
    retain_per_image_deltas: bool = ...,
) -> bytes: ...
def merge_panoptic_partials(
    categories: bytes,
    partials: Sequence[bytes],
    parity_mode: str,
    *,
    things_stuff_split: bool = ...,
    retain_per_image_deltas: bool = ...,
) -> PanopticSummary: ...

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

def evaluate_semantic_from_arrays(
    gt_label_maps: dict[int, NDArray[np.uint32]],
    dt_label_maps: dict[int, NDArray[np.uint32]],
    n_classes: int,
    parity_mode: str,
    *,
    ignore_label: int | None = ...,
    label_remap: dict[int, int] | None = ...,
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
    gt_label_maps: dict[int, NDArray[np.uint32]],
    dt_label_maps: dict[int, NDArray[np.uint32]],
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
        gt: NDArray[np.uint32],
        dt: NDArray[np.uint32],
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
