"""Framework adapters for vernier.

This subpackage holds optional integrations with external frameworks
(PyTorch tensors via DLPack, etc.). Each adapter should be importable
without its underlying framework being installed; gate framework-specific
imports behind ``try``/``except ImportError``.
"""
