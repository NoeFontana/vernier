# Python type stubs

`python/vernier/_core.pyi` is the type stub for the PyO3 extension
module `vernier._core`. It is **hand-written on purpose**, and it is
**checked in**, not generated at build time.

This document is the authoritative statement of that policy. It covers
why we do not use a stub generator, what keeps the stub honest, and how
to maintain it when the FFI changes.

> **Erratum for ADR-0019.** ADR-0019 §"Crate dependency" states that
> "`pyo3-stub-gen` (already in the workspace) generates the
> PyCapsule-returning entry points correctly; no new stub infrastructure
> is required." That was never true: `pyo3-stub-gen` was declared in
> `[workspace.dependencies]` but depended on by no crate, never appeared
> in `Cargo.lock`, and no `#[gen_stub_*]` macro was ever used. The
> dangling declaration has been removed.
>
> ADR-0019 is `accepted` and therefore immutable, and only a new ADR can
> formally supersede it — so that sentence still stands in the ADR, with
> no forward pointer. This document records the erratum in the meantime;
> **do not re-add the dependency on the strength of that sentence.** If
> the discrepancy proves confusing in practice, the fix is a short
> superseding ADR, not an edit to ADR-0019.

## Why not a stub generator

`pyo3-stub-gen` is a mature, well-maintained crate, and version 0.23
supports our pinned `pyo3 >= 0.27, < 0.30` / `numpy 0.29`. Compatibility
is not the blocker. It was evaluated in July 2026 and rejected on two
grounds.

### 1. The stub is a curated contract, not a mechanical mirror

`_core.pyi` deliberately encodes types that the Rust signatures do not
carry, and a generator working from those signatures cannot recover
them:

| The stub says | The Rust says | Generated would be |
|---|---|---|
| `Literal["bbox", "segm", "boundary", "keypoints"]` | `&str` | `str` |
| `Literal["quantile", "equal_width"]`, `Literal["wilson", "clopper_pearson"]` | `&str` | `str` |
| `_TideReportDict`, `_LrpReportDict`, `_ConfusionMatrixDict`, … (9 `TypedDict`s) | `Py<PyDict>` | `dict[Any, Any]` |
| `NDArray[np.float64]`, `NDArray[np.uint32]` | `PyArray1<f64>` | untyped / unparameterized |
| `DetectionsInput` (the DLPack / Arrow / RLE duck-typed union) | `Bound<'py, PyAny>` | `Any` |
| `Self` on `__enter__` | `PyRef<'_, Self>` | the concrete class |
| `Sequence[T]` / `Mapping[K, V]` on inputs, `list` / `dict` on returns | `Vec<T>` / `HashMap<K, V>` | mostly right, but not the deliberate cases |

There are roughly fifteen string-typed pseudo-enums on the boundary —
`calibration.rs` calls them "String-typed enum arguments" in so many
words. Recovering the current precision under generation means
`#[gen_stub(override_type(...))]` on the bulk of 65 `#[pyfunction]`,
24 `#[pyclass]`, 23 `#[pymethods]` and 97 `#[getter]` sites. The types
would still be hand-written — just scattered across 15k lines of Rust
instead of one reviewable file. Adopting generation *without* that
annotation work is a straight type-safety regression for users.

### 2. It does not solve the problem we actually had

The real risk was never expressiveness; it was **drift**, and until
July 2026 nothing detected it. There was no justfile recipe, no CI
step, no test, and no pre-commit hook comparing the stub to the module.

Worse, the gate that looks like it should catch drift structurally
cannot: CI's `lint (python)` job runs `uv sync --no-install-project`
and type-checks with pyright *without ever building the extension*.
Pyright resolves `vernier._core` through `_core.pyi` itself, so a stub
that has drifted from the Rust still passes cleanly. The stub was
checking itself.

That gap is closable directly, and much more cheaply than adopting a
generator — see below.

## What keeps the stub honest

Two gates, in two different CI jobs, because they need different things.

### `tests/python/test_core_stub_conformance.py` — accuracy

Runs in `just test-py` / the `test-py` CI matrix, where the compiled
extension exists. It parses `_core.pyi` with `ast` and introspects the
imported module, asserting:

1. every symbol in `_core.__all__` is declared in the stub;
2. every stub declaration corresponds to a runtime symbol (or is a
   private type-only helper);
3. class members match in both directions — including **kind**, so a
   `#[getter]` must be `@property` in the stub and a method must not be
   (the signature check alone cannot catch this: getters are
   `getset_descriptor`s and are not callable);
4. a member PyO3 disables by setting it to `None` — e.g. `__hash__` on an
   `#[pyclass(eq)]` without `hash` — is not declared as a method (spell it
   `ClassVar[None]`, as `Breakdown.__hash__` does);
5. no Python-visible parameter is underscore-prefixed;
6. parameter **names**, **order**, **arity**, **which parameters have
   defaults**, and the **keyword-only boundary** match.

It checks *shape only*. It never inspects an annotation — types stay
hand-curated and human-reviewed. That division is the whole point:
**the machine checks shape, humans curate types.**

This works because PyO3 0.29 emits a complete `__text_signature__`,
including the keyword-only `*` marker and defaults, for all 106
functions and methods. The only callables without one are constructors
of opaque and exception classes that have no `#[new]` — and the test
asserts the stub correspondingly declares no `__init__` for them, so a
silently-skipped item cannot erode coverage.

Because it resolves the stub via `Path(vernier.__file__).parent`, under
CI it validates the stub **as packaged in the wheel**, not as it sits in
the source tree.

Symbols behind `#[cfg(feature = ...)]` (`bench-histogram`,
`bench-timings`, `_test-counter`, `test-poison`) are never compiled into
shipped wheels and so are deliberately absent from the stub. They are
listed in `_FEATURE_GATED_FUNCTIONS` / `_FEATURE_GATED_MEMBERS` in the
test. **Add to those lists when you add a gated FFI item**, otherwise
`maturin develop --features …` builds will fail the check.

### `pyright --verifytypes vernier --ignoreexternal` — completeness

Guards the completeness of the public typed surface; the baseline is
**100%** (0 unknown, 0 ambiguous) as of July 2026.

Runs in `just test-py` / the `test-py` CI job, **not** in the lint lane.
`--verifytypes` resolves the *installed distribution* and reads its
packaged `py.typed` — it does not honour pyright's `extraPaths`, so it
fails with `error: No py.typed file found` wherever `vernier` is not
installed. The `lint (python)` job runs `uv sync --no-install-project`, so
that is exactly where it would fail. Running it against the built wheel in
`test-py` is also the stronger check: it verifies the surface as shipped.

## Maintaining the stub

When you change the FFI:

1. Change the Rust.
2. Update `_core.pyi` in the same commit.
3. `just develop && uv run pytest tests/python/test_core_stub_conformance.py`

The conformance test tells you *what* is missing or misshapen. It will
not tell you *which type* to write — that is the judgement the policy
above deliberately keeps with a human. Prefer the most precise type the
runtime actually guarantees: `Literal` over `str` for string-typed
enums, a `TypedDict` over `dict[str, Any]` for fixed-shape returns,
`Sequence`/`Mapping` for inputs and concrete `list`/`dict` for returns.

Constraints to be aware of:

- **ruff's PYI rules apply.** No docstrings in the stub (PYI021), no
  `pass` bodies (PYI048) — use `...`. `ruff format --check` runs on
  `.pyi` in CI.
- **`N818` is disabled for this file** (`pyproject.toml`). The
  `Partial*Mismatch` / `Partial*Overlap` / `Partial*Collision`
  exception names are part of the shipped public surface per ADR-0032
  §"Errors" and cannot gain an `Error` suffix without a coordinated
  Rust + Python break.
- **Parameter names are API.** PyO3 takes the Python-visible parameter
  names verbatim from the Rust bindings, so naming one `_thing` to
  silence an unused-variable warning leaks that underscore into the
  public signature. Name it properly and consume it in the body with
  `drop((a, b, c));` — that satisfies both `unused_variables` and
  clippy's `needless_pass_by_value`, which an `#[allow(unused_variables)]`
  alone would not (see the three `__exit__` implementations). This is
  enforced, not just advised: `test_no_underscore_prefixed_public_parameters`.
- **Type-only helpers are private.** Names the stub declares that have no
  runtime counterpart — the `_TideReportDict` family, `_TablesResult`,
  `_LvisFrequencyLiteral` — carry a leading underscore. That convention is
  what lets the conformance test tell a deliberate helper from a stale
  declaration, so there is no allowlist to append to.
- **Use `*` in `#[pyo3(signature = ...)]`** for keyword-argument
  surfaces, so the runtime and the stub agree that they are
  keyword-only.

## Revisiting this decision

The tradeoff would change if the FFI's Python-visible types stopped
outrunning its Rust types — for example if the string-typed pseudo-enums
became real `#[pyclass]` enums and the ad-hoc `PyDict` returns became
`#[pyclass]` structs. At that point most of the `override_type`
annotation burden disappears and generation becomes attractive. Until
then, generation buys nothing the conformance test does not already
provide, and costs precision.
