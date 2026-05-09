"""Drift guard for ``docs/reference/python/`` mkdocstrings pages.

Each page renders ``::: vernier.<module>``, which auto-includes classes,
functions, dataclasses and enums in ``__all__`` — but mkdocstrings does
not render bare type aliases (``Union``/``Literal``) or module-level
``Final[...]`` constants from a top-level ``:::`` directive. Those need
an explicit ``::: vernier.<module>.<name>`` directive in the markdown.

This test asserts that for every name in each module's ``__all__`` that
mkdocstrings would skip from the top-level directive, the corresponding
page contains an explicit directive — so adding a new type alias or
constant to ``__all__`` fails the test until the reference page is
updated alongside it.
"""

from __future__ import annotations

import importlib
import inspect
from pathlib import Path

import pytest

REFERENCE_DIR = Path(__file__).resolve().parents[2] / "docs" / "reference" / "python"

# Per-module ignore lists for names that legitimately do not appear on the
# corresponding reference page: submodule re-exports that have their own
# canonical pages, and aliases that mkdocstrings would otherwise duplicate.
_IGNORE: dict[str, frozenset[str]] = {
    "vernier": frozenset({"instance", "panoptic", "semantic", "patch_pycocotools", "__version__"}),
    "vernier.instance": frozenset(),
    "vernier.panoptic": frozenset(),
    "vernier.semantic": frozenset(),
    "vernier.adapters": frozenset(),
}


def _page_filename(module_name: str) -> str:
    if module_name == "vernier":
        return "vernier.md"
    return module_name.removeprefix("vernier.") + ".md"


def _is_auto_rendered(obj: object) -> bool:
    """Return True if mkdocstrings will render ``obj`` from a module-level ``:::`` directive."""
    return inspect.isclass(obj) or inspect.isfunction(obj) or inspect.isbuiltin(obj)


@pytest.mark.parametrize("module_name", sorted(_IGNORE))
def test_reference_page_covers_all_public_names(module_name: str) -> None:
    module = importlib.import_module(module_name)
    page_filename = _page_filename(module_name)
    page_text = (REFERENCE_DIR / page_filename).read_text()
    ignore = _IGNORE[module_name]

    missing: list[str] = []
    for name in module.__all__:
        if name in ignore:
            continue
        obj = getattr(module, name)
        if _is_auto_rendered(obj):
            continue
        directive = f"::: {module_name}.{name}"
        if directive not in page_text:
            missing.append(name)

    assert not missing, (
        f"{page_filename} is missing explicit `::: {module_name}.<name>` "
        f"directives for these public names (mkdocstrings will not render "
        f"them from the module-level directive alone): {missing}. "
        f"Add a `::: {module_name}.<name>` block per name, or remove them "
        f"from {module_name}.__all__ if they should not be public."
    )
