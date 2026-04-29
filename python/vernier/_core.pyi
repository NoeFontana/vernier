from typing import Any

import numpy as np
from numpy.typing import NDArray

__version__: str

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
    def summarize(self, max_dets: list[int] | None = ...) -> Summary: ...

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
