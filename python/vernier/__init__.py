"""vernier: high-performance, parity-preserving COCO-style evaluation.

The Python package is a thin wrapper around the Rust core. Public symbols
documented here are the supported API; anything imported from
:mod:`vernier._core` directly is considered implementation detail and may
change without a deprecation cycle.
"""

from __future__ import annotations

from vernier._core import version

__all__ = ["__version__", "version"]

__version__: str = version()
