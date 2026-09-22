"""Gate: vernier never takes a machine-learning framework dependency.

ADR-0031 §"Maintainability" states the rule as *"no torch dependency,
ever"*, and ADR-0055 restates it for TorchMetrics: *"a consumer we
satisfy, not a dependency we take."* Both are prose in an ADR, which
is exactly the kind of rule that decays — a lazy ``import torch``
inside one adapter function would satisfy every other test in this
suite.

Framework tensors reach vernier through the DLPack consumer instead
(ADR-0030; ``crates/vernier-ffi/src/dlpack.rs``), which duck-types on
``__dlpack__`` / ``__dlpack_device__`` and never names a framework. So
the rule is not "torch support is unimplemented" — it is "torch support
is implemented without depending on torch", and that property needs a
test or it is one convenient shortcut away from being false.

Three checks, each closing a different door:

1. **Static.** No shipped module imports a framework, anywhere in its
   AST — including inside a function body and inside an
   ``if TYPE_CHECKING:`` block. Parsed rather than grepped, because
   ``vernier.semantic`` and ``vernier.instance`` both mention torch in
   prose and a text search would trip on the docstrings.
2. **Runtime.** Importing the package and every public submodule pulls
   no framework into ``sys.modules``. Catches an import reached through
   a re-export the AST walk did not attribute to us.
3. **Metadata.** The built distribution's *base* requirements name no
   framework. The ``torch`` extra is exempt: it exists for the vendored
   mmsegmentation oracle (ADR-0036) and ships to nobody who does not
   ask for it by name.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, requires
from pathlib import Path

import pytest

#: Import roots that would make vernier depend on a training framework.
#: Matched against the first dotted component, so ``torch.utils.data``
#: and ``jax.numpy`` are covered by their roots.
FRAMEWORKS: frozenset[str] = frozenset(
    {
        "torch",
        "torchvision",
        "torchmetrics",
        "lightning",
        "pytorch_lightning",
        "jax",
        "tensorflow",
        "keras",
    }
)

#: Public submodules that must each be import-clean on their own. A
#: framework import reached only from a submodule the root does not
#: eagerly import would otherwise pass check 2.
PUBLIC_SUBMODULES: tuple[str, ...] = (
    "vernier.adapters",
    "vernier.aggregate",
    "vernier.calibration",
    "vernier.instance",
    "vernier.panoptic",
    "vernier.semantic",
)

PACKAGE_ROOT = Path(__file__).resolve().parents[2] / "python" / "vernier"


def _shipped_sources() -> list[Path]:
    """Every ``.py`` / ``.pyi`` file that ships inside the wheel."""
    return sorted(
        path
        for path in PACKAGE_ROOT.rglob("*.py*")
        if path.suffix in {".py", ".pyi"} and "__pycache__" not in path.parts
    )


def _imported_roots(source: str, filename: str) -> set[str]:
    """Return the first dotted component of every import in ``source``.

    Walks the whole tree rather than the module body, so a
    function-local import and one guarded by ``if TYPE_CHECKING:`` are
    both reported. A relative ``from . import x`` has ``node.module``
    ``None`` or a level above zero and is never a framework.
    """
    roots: set[str] = set()
    for node in ast.walk(ast.parse(source, filename=filename)):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            roots.add(node.module.split(".", 1)[0])
    return roots


def test_package_sources_are_present() -> None:
    """Guard the guard: a bad path would make every check below vacuous."""
    sources = _shipped_sources()
    assert len(sources) > 5, f"expected the vernier package at {PACKAGE_ROOT}, found {len(sources)}"
    assert PACKAGE_ROOT / "__init__.py" in sources


@pytest.mark.parametrize("source_path", _shipped_sources(), ids=lambda p: p.name)
def test_no_framework_import_in_shipped_source(source_path: Path) -> None:
    """No shipped module imports a training framework, at any nesting depth."""
    roots = _imported_roots(source_path.read_text(encoding="utf-8"), str(source_path))
    offending = roots & FRAMEWORKS
    assert not offending, (
        f"{source_path.relative_to(PACKAGE_ROOT.parents[1])} imports {sorted(offending)}. "
        "vernier reaches framework tensors through DLPack (ADR-0030), never through an import; "
        "see ADR-0031 and this module's docstring."
    )


@pytest.mark.parametrize("module", ["vernier", *PUBLIC_SUBMODULES])
def test_import_pulls_no_framework_into_sys_modules(module: str) -> None:
    """Importing the package loads no framework, even transitively.

    Runs in a subprocess: this test session already has torch imported
    (``test_compat_torchmetrics.py`` and the real-model harnesses need
    it), so an in-process ``sys.modules`` check would pass regardless of
    what vernier does.
    """
    probe = (
        "import sys, json;"
        f"__import__({module!r});"
        "roots = {n.split('.', 1)[0] for n in sys.modules};"
        f"print(json.dumps(sorted(roots.intersection({sorted(FRAMEWORKS)!r}))))"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True
    )
    loaded = result.stdout.strip()
    assert loaded == "[]", f"`import {module}` pulled {loaded} into sys.modules"


def test_base_requirements_name_no_framework() -> None:
    """The installed distribution's base requirements name no framework.

    Requirements carrying an ``extra ==`` marker are skipped: the
    ``torch`` extra is the vendored mmsegmentation oracle (ADR-0036) and
    ``real-models`` is the SOTA harness. Neither installs unless asked
    for by name, so neither is a dependency vernier takes.
    """
    try:
        declared = requires("vernier") or []
    except PackageNotFoundError:  # pragma: no cover - always installed under `just test-py`
        pytest.skip("vernier is not installed as a distribution")

    base = [req for req in declared if "extra ==" not in req]
    offending = [
        req
        for req in base
        if req.split(maxsplit=1)[0].split(";")[0].split("[")[0].strip() in FRAMEWORKS
    ]
    assert not offending, f"base requirements name a training framework: {offending}"
