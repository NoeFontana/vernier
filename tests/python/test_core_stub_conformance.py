"""Conformance check: the shipped ``vernier/_core.pyi`` vs. the compiled extension.

``_core.pyi`` is hand-written on purpose — it encodes ``Literal`` unions,
``TypedDict`` return shapes and ``NDArray`` dtypes that the Rust signatures do
not carry, so it cannot be generated from them (see
``docs/engineering/python-type-stubs.md``). The cost of that choice is drift:
CI's ``lint (python)`` job runs ``--no-install-project`` and type-checks against
the stub *without building the extension*, so a stale stub is invisible there.

This module closes that gap. It checks **shape** only — symbol names, class
members, parameter names, arity and the keyword-only boundary. Types stay
hand-curated and human-reviewed; nothing here inspects an annotation.

It runs against ``vernier.__file__``'s package directory, so under CI's
``test-py`` job it validates the stub *as packaged in the wheel*.
"""

from __future__ import annotations

import ast
import functools
import inspect
from pathlib import Path
from typing import Any

import pytest

import vernier
from vernier import _core

_STUB_PATH = Path(vernier.__file__).resolve().parent / "_core.pyi"

# Symbols behind `#[cfg(feature = ...)]` in crates/vernier-ffi. Never compiled
# into shipped wheels, so absent from the stub by design; they appear only in
# local `maturin develop --features ...` builds (e.g. `just
# bench-develop-histogram`). Keep in sync with `[features]` in
# crates/vernier-ffi/Cargo.toml, which documents the same mapping in prose.
_FEATURE_GATED_FUNCTIONS = frozenset(
    {
        # bench-histogram
        "dump_bbox_iou_histogram",
        # bench-timings
        "read_and_reset_evaluate_parallel_timings",
        "read_and_reset_build_anns_count",
        "read_and_reset_dataset_timings",
        # _test-counter
        "_test_reset_panoptic_matching_count",
        "_test_read_panoptic_matching_count",
        "_test_reset_semantic_fold_count",
        "_test_read_semantic_fold_count",
    }
)

# Same, but class members. Keys are ``ClassName.member``.
_FEATURE_GATED_MEMBERS = frozenset({"BackgroundEvaluator._inject_poison_for_tests"})

# Class members Python or PyO3 provides automatically. `#[pyclass(eq)]` /
# `(ord)` synthesise the comparison dunders; `__new__` is what the stub spells
# as `__init__`; `__del__` is a Drop shim with no typing meaning.
_AUTO_MEMBERS = frozenset(
    {
        "__dict__",
        "__del__",
        "__doc__",
        "__eq__",
        "__ge__",
        "__gt__",
        "__hash__",
        "__init__",
        "__le__",
        "__lt__",
        "__module__",
        "__ne__",
        "__new__",
        "__repr__",
        "__str__",
        "__weakref__",
    }
)


# --------------------------------------------------------------------------
# Stub parsing
# --------------------------------------------------------------------------


@functools.lru_cache(maxsize=1)
def _stub_toplevel() -> dict[str, ast.stmt]:
    """Top-level names the stub *declares* (imports are re-exports, not declarations)."""
    if not _STUB_PATH.is_file():
        pytest.fail(
            f"type stub not found at {_STUB_PATH}. It must ship beside the extension "
            "module; check `[tool.maturin] python-source` and the wheel contents."
        )
    tree = ast.parse(_STUB_PATH.read_text(encoding="utf-8"), filename=str(_STUB_PATH))
    declared: dict[str, ast.stmt] = {}
    for node in tree.body:
        if isinstance(node, ast.ClassDef | ast.FunctionDef):
            declared[node.name] = node
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            declared[node.target.id] = node
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    declared[target.id] = node
    return declared


def _stub_class(name: str) -> ast.ClassDef | None:
    node = _stub_toplevel().get(name)
    return node if isinstance(node, ast.ClassDef) else None


def _stub_class_members(node: ast.ClassDef) -> tuple[dict[str, ast.FunctionDef], set[str]]:
    """Return ``({method_name: def_node}, {annotated_attribute_names})``."""
    methods: dict[str, ast.FunctionDef] = {}
    attributes: set[str] = set()
    for item in node.body:
        if isinstance(item, ast.FunctionDef):
            methods[item.name] = item
        elif isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
            attributes.add(item.target.id)
    return methods, attributes


def _is_property(node: ast.FunctionDef) -> bool:
    return any(
        (isinstance(d, ast.Name) and d.id == "property")
        or (isinstance(d, ast.Attribute) and d.attr in {"setter", "getter", "deleter"})
        for d in node.decorator_list
    )


# --------------------------------------------------------------------------
# Runtime introspection
# --------------------------------------------------------------------------


def _runtime_exports() -> set[str]:
    """PyO3 0.29 synthesises ``__all__`` from every ``m.add*`` call."""
    exports = getattr(_core, "__all__", None)
    if exports is None:  # pragma: no cover - defensive; PyO3 always emits it
        pytest.fail("vernier._core has no __all__; PyO3 should synthesise one")
    return set(exports)


@functools.lru_cache(maxsize=1)
def _runtime_classes() -> dict[str, type]:
    return {
        name: obj
        for name in sorted(_runtime_exports())
        if isinstance(obj := getattr(_core, name, None), type)
    }


def _runtime_class_members(cls: type) -> dict[str, Any]:
    """Members defined on ``cls`` itself, minus auto-generated machinery.

    Everything not in ``_AUTO_MEMBERS`` must be declared in the stub — including
    non-callables such as ``#[classattr]`` constants. Filtering by type here
    would silently exempt whole categories of member from the check.
    """
    return {name: obj for name, obj in vars(cls).items() if name not in _AUTO_MEMBERS}


def _is_runtime_property(obj: Any) -> bool:
    """``#[getter]`` surfaces as a ``getset_descriptor``, not a callable."""
    return isinstance(obj, property) or inspect.isgetsetdescriptor(obj)


def _runtime_signature(obj: Any) -> inspect.Signature | None:
    """``inspect.signature`` for anything PyO3 gave a ``__text_signature__``."""
    try:
        return inspect.signature(obj)
    except (TypeError, ValueError):
        return None


# --------------------------------------------------------------------------
# Signature comparison (names / arity / keyword-only boundary only)
# --------------------------------------------------------------------------

_ParamShape = tuple[list[tuple[str, bool]], dict[str, bool]]
"""``([(positional_name, has_default), ...], {kwonly_name: has_default})``."""


def _drop_receiver(positional: list[tuple[str, bool]]) -> list[tuple[str, bool]]:
    if positional and positional[0][0] in {"self", "cls"}:
        return positional[1:]
    return positional


def _stub_params(node: ast.FunctionDef) -> _ParamShape:
    args = node.args
    ordered = [*args.posonlyargs, *args.args]
    # `defaults` right-aligns against posonlyargs + args.
    padding = len(ordered) - len(args.defaults)
    positional = [(arg.arg, index >= padding) for index, arg in enumerate(ordered)]
    keyword_only = {
        arg.arg: default is not None
        for arg, default in zip(args.kwonlyargs, args.kw_defaults, strict=True)
    }
    return _drop_receiver(positional), keyword_only


def _signature_params(signature: inspect.Signature) -> _ParamShape:
    positional: list[tuple[str, bool]] = []
    keyword_only: dict[str, bool] = {}
    for parameter in signature.parameters.values():
        has_default = parameter.default is not inspect.Parameter.empty
        if parameter.kind in {
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        }:
            positional.append((parameter.name, has_default))
        elif parameter.kind is inspect.Parameter.KEYWORD_ONLY:
            keyword_only[parameter.name] = has_default
    return _drop_receiver(positional), keyword_only


def _describe(params: _ParamShape) -> str:
    positional, keyword_only = params
    parts = [name + ("=..." if has_default else "") for name, has_default in positional]
    if keyword_only:
        parts.append("*")
        parts += [
            name + ("=..." if has_default else "") for name, has_default in keyword_only.items()
        ]
    return "(" + ", ".join(parts) + ")"


def _signature_mismatch(label: str, stub: ast.FunctionDef, runtime: Any) -> str | None:
    """Return a human-readable drift report, or ``None`` when the shapes agree."""
    signature = _runtime_signature(runtime)
    if signature is None:
        return (
            f"{label}: no __text_signature__ at runtime, but the stub declares one. "
            "Add #[pyo3(signature = ...)] to the Rust item, or drop it from the stub."
        )
    expected = _signature_params(signature)
    actual = _stub_params(stub)
    if actual == expected:
        return None
    return f"{label}\n      runtime: {_describe(expected)}\n      stub:    {_describe(actual)}"


def _assert_no_mismatches(mismatches: list[str]) -> None:
    assert not mismatches, (
        "signature drift between vernier._core and python/vernier/_core.pyi:\n    "
        + "\n    ".join(mismatches)
        + "\n  Parameter names, order, defaults-or-not and the keyword-only boundary must "
        "match. Types are not compared — reconcile _core.pyi with the Rust signature."
    )


# --------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------


def test_stub_declares_every_runtime_export() -> None:
    """Every symbol the extension exports must be declared in the stub."""
    missing = _runtime_exports() - set(_stub_toplevel()) - _FEATURE_GATED_FUNCTIONS
    assert not missing, (
        f"vernier._core exports {sorted(missing)} but python/vernier/_core.pyi does not "
        "declare them. Add them to the stub (or to _FEATURE_GATED_FUNCTIONS if the item "
        "is #[cfg(feature = ...)]-gated out of shipped wheels)."
    )


def test_stub_only_names_are_intentional() -> None:
    """A stub entry must not outlive the Rust item it describes."""
    extra = set(_stub_toplevel()) - _runtime_exports()
    # Type-only helpers (the `_TideReportDict` family, `_TablesResult`, …) describe
    # shapes the FFI builds ad hoc via `PyDict`, so no runtime symbol mirrors them.
    # They are private by convention, which is what makes them distinguishable here.
    unexplained = {name for name in extra if not name.startswith("_")}
    assert not unexplained, (
        f"python/vernier/_core.pyi declares {sorted(unexplained)}, which vernier._core does "
        "not export. Remove the stale declaration, or — if it is a type-only helper — give "
        "it a leading underscore."
    )


@pytest.mark.parametrize("class_name", sorted(_runtime_classes()))
def test_stub_class_members_match_runtime(class_name: str) -> None:
    """Methods and properties on each pyclass must be mirrored in the stub."""
    cls = _runtime_classes()[class_name]
    node = _stub_class(class_name)
    assert node is not None, f"_core.pyi has no class {class_name}"

    stub_methods, stub_attributes = _stub_class_members(node)
    runtime_members = _runtime_class_members(cls)

    # `_runtime_class_members` has already dropped `_AUTO_MEMBERS`, so this is
    # purely "everything the runtime exposes must be written down".
    declared = set(stub_methods) | stub_attributes
    missing = {
        name
        for name in runtime_members
        if name not in declared and f"{class_name}.{name}" not in _FEATURE_GATED_MEMBERS
    }
    assert not missing, (
        f"{class_name} exposes {sorted(missing)} at runtime but _core.pyi does not declare "
        "them. Add the method/property to the stub class."
    )

    # The reverse direction is checked against the *raw* class dict, not the
    # filtered view: a dunder the stub bothers to write down must actually
    # exist, even if `_AUTO_MEMBERS` exempts it from the `missing` check.
    # `__init__` is the one genuine exception — PyO3 spells it `__new__`.
    raw_members = vars(cls)
    stale = {name for name in stub_methods if name != "__init__" and name not in raw_members}
    assert not stale, (
        f"_core.pyi declares {sorted(stale)} on {class_name}, but the compiled class has no "
        "such member. Remove the stale declaration."
    )

    # A member PyO3 sets to `None` is disabled, not implemented — e.g.
    # `#[pyclass(eq)]` without `hash` makes the class unhashable. A stub that
    # declares it as a method promises an API that raises `TypeError`.
    disabled = {
        name
        for name in set(stub_methods) | stub_attributes
        if name in raw_members and raw_members[name] is None
    }
    nulled_as_method = {name for name in disabled if name in stub_methods}
    assert not nulled_as_method, (
        f"{class_name}: {sorted(nulled_as_method)} is None at runtime (disabled by PyO3) but "
        "_core.pyi declares it as a method. Spell it `ClassVar[None]` instead — see "
        "`Breakdown.__hash__`."
    )

    # A `#[getter]` declared as a plain `def` in the stub (or the reverse) type-checks
    # but blows up at runtime — `ev.images_seen()` on an int, or `ev.finalize` on a
    # method object. The signature test cannot catch this: getters are not callable.
    wrong_kind = {
        name
        for name in set(stub_methods) & set(runtime_members)
        if _is_property(stub_methods[name]) != _is_runtime_property(runtime_members[name])
    }
    assert not wrong_kind, (
        f"{class_name}: {sorted(wrong_kind)} disagree on property-vs-method between "
        "vernier._core and _core.pyi. A `#[getter]` must be `@property` in the stub, and "
        "a method must not be."
    )


def test_no_underscore_prefixed_public_parameters() -> None:
    """Rust's unused-parameter convention must not leak into the Python signature.

    PyO3 takes Python-visible parameter names verbatim from the Rust bindings, so
    naming one `_exc_type` to silence an unused-variable warning publishes that
    underscore as the keyword. Consume the binding with `drop((a, b, c));` instead.
    """
    offenders: list[str] = []
    for name in sorted(_runtime_exports()):
        obj = getattr(_core, name)
        targets = (
            [(f"{name}.{m}", getattr(cls, m)) for cls in [obj] for m in vars(cls)]
            if isinstance(obj, type)
            else [(name, obj)]
        )
        for label, target in targets:
            signature = _runtime_signature(target)
            if signature is None:
                continue
            offenders += [
                f"{label}({p})" for p in signature.parameters if p.startswith("_") and p != "_"
            ]
    assert not offenders, (
        "these Python-visible parameters are underscore-prefixed, leaking Rust's "
        f"unused-variable convention into the public API: {sorted(offenders)}"
    )


def test_module_function_signatures_match() -> None:
    """Free-function parameter names, arity and kw-only boundary must match.

    Batched rather than parametrized per function: a mass rename shows up as one
    failure listing every affected symbol, which is how this drift actually arrives.
    """
    mismatches: list[str] = []
    for name in sorted(_runtime_exports() - _FEATURE_GATED_FUNCTIONS):
        runtime = getattr(_core, name)
        if isinstance(runtime, type) or not callable(runtime):
            continue
        node = _stub_toplevel().get(name)
        assert isinstance(node, ast.FunctionDef), f"_core.pyi has no `def {name}`"
        if (report := _signature_mismatch(name, node, runtime)) is not None:
            mismatches.append(report)
    _assert_no_mismatches(mismatches)


@pytest.mark.parametrize("class_name", sorted(_runtime_classes()))
def test_class_signatures_match(class_name: str) -> None:
    """Constructor and method signatures must match, including the kw-only boundary."""
    cls = _runtime_classes()[class_name]
    node = _stub_class(class_name)
    assert node is not None, f"_core.pyi has no class {class_name}"
    stub_methods, _ = _stub_class_members(node)

    mismatches: list[str] = []
    constructor = stub_methods.get("__init__")
    if constructor is not None:
        report = _signature_mismatch(f"{class_name}.__init__", constructor, cls)
        if report is not None:
            mismatches.append(report)
    else:
        # No stub `__init__` is only correct when the class has no `#[new]`.
        class_signature = _runtime_signature(cls)
        assert class_signature is None or not class_signature.parameters, (
            f"{class_name} accepts {_describe(_signature_params(class_signature))} at runtime "
            "but _core.pyi declares no __init__. Add one to the stub class."
        )

    for name, stub_method in stub_methods.items():
        if name in _AUTO_MEMBERS or _is_property(stub_method):
            continue
        runtime_member = getattr(cls, name, None)
        if runtime_member is None or not callable(runtime_member):
            continue  # covered by test_stub_class_members_match_runtime
        report = _signature_mismatch(f"{class_name}.{name}", stub_method, runtime_member)
        if report is not None:
            mismatches.append(report)
    _assert_no_mismatches(mismatches)
