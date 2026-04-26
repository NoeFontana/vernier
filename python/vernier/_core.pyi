__version__: str

class Summary:
    @property
    def stats(self) -> list[float]: ...
    def pretty_lines(self) -> list[str]: ...

def version() -> str: ...
def evaluate_bbox_summary(
    gt_json: bytes,
    dt_json: bytes,
    parity_mode: str,
    max_dets: list[int],
    use_cats: bool,
) -> Summary: ...
