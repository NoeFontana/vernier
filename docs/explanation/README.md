# Explanation

Architecture overviews, theoretical background on COCO-style evaluation,
and design rationale. Audience: anyone who wants to understand *how
vernier thinks*.

- [Three paradigms: instance, panoptic, semantic](three-paradigms.md) —
  picking the submodule whose input shape matches your model's output.
- [TIDE: what it answers, and what it doesn't](tide-and-its-limits.md) —
  the six error bins, the assumptions behind them, and where they break.
- [Why `per_image` does not ship an AP column](why-no-per-image-ap.md) —
  why per-image AP is structurally ill-defined under the COCO fold.
