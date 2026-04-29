from typing import Any, Literal, TypedDict

import numpy as np
from numpy.typing import NDArray
from typing_extensions import Self

__version__: str

class _UpdateReportDict(TypedDict):
    new_images: int
    new_detections: int
    memory_used_bytes: int
    soft_warn_triggered: bool

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
    ) -> None: ...
    def update(self, detections: bytes) -> _UpdateReportDict: ...
    def snapshot(self, *, running: bool = ...) -> Summary: ...
    def finalize(self) -> Summary: ...
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
    ) -> None: ...
    def submit(self, detections: bytes, *, timeout: float | None = ...) -> None: ...
    def snapshot(self, *, peek: bool = ...) -> Summary: ...
    def finalize(self) -> Summary: ...
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

def version() -> str: ...
def evaluate_bbox_summary(
    gt_json: bytes,
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
) -> EvalGrid: ...
def evaluate_segm_summary(
    gt_json: bytes,
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
) -> EvalGrid: ...
def evaluate_boundary_summary(
    gt_json: bytes,
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
) -> EvalGrid: ...
def evaluate_keypoints_summary(
    gt_json: bytes,
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
) -> EvalGrid: ...
