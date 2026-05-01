"""vernier-bench harness package. See ADR-0017."""

__version__ = "0.0.0"
# Public alias for code that wants to embed the harness version into a
# result file (the ``__version__`` dunder name confuses ruff's N812 when
# imported under a renamed alias).
HARNESS_VERSION = __version__
