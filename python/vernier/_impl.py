"""Private substrate for the public evaluator surface (ADR-0035).

The public surface is :class:`vernier.instance.Evaluator` /
:class:`vernier.panoptic.Evaluator` / :class:`vernier.semantic.Evaluator`
plus the matching ``BackgroundEvaluator`` per paradigm. The streaming
evaluator pyclasses re-exported here are the implementation substrate
those public types delegate to (and that ``BackgroundEvaluator`` wraps
behind its worker thread). They are not part of the public API; do not
import from this module in user code.
"""

from __future__ import annotations

from vernier._core import (
    StreamingEvaluator,
    StreamingPanopticEvaluator,
    StreamingSemanticEvaluator,
)

__all__ = [
    "StreamingEvaluator",
    "StreamingPanopticEvaluator",
    "StreamingSemanticEvaluator",
]
