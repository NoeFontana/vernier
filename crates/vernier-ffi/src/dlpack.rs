//! DLPack v1 ABI consumer (CPU-only) — sole array ingest path for ADR-0030.
//!
//! Numpy ≥ 1.22, torch CPU, jax CPU, and cupy host buffers all expose
//! `__dlpack__` / `__dlpack_device__`. Routing every input through this
//! module collapses the ingest matrix to one audited unsafe surface, one
//! validation table, and one parity test row. No `rust-numpy` or other
//! protocol is consulted on this path.
//!
//! ## Unsafe policy
//!
//! Every `unsafe` block is justified inline. The trust contract is the
//! DLPack spec at <https://dmlc.github.io/dlpack/latest/>:
//!
//! 1. Reading the capsule pointer through `pointer_checked` and casting
//!    it to `*const DLManagedTensor` / `*const DLManagedTensorVersioned`
//!    is sound because the producer guarantees that exact layout behind
//!    a capsule named `"dltensor"` / `"dltensor_versioned"` for the
//!    capsule's lifetime.
//! 2. Reinterpreting `shape`/`strides` raw pointers as slices of length
//!    `ndim` and `data` (offset by `byte_offset`) as a `&[T]` of
//!    `product(shape)` elements is sound because the producer
//!    guarantees those buffers stay valid while the capsule lives, and
//!    we hold the capsule by value inside [`DLPackView`] for the borrow's
//!    lifetime.
//! 3. We never call the producer's deleter directly. The capsule's
//!    Python-level destructor (set by the producer when constructing
//!    the capsule) calls it automatically when the last strong reference
//!    drops. The view owns the capsule, so the destructor runs when the
//!    view is dropped — releasing the producer's buffer exactly once.
//!
//! Every screen we can do without entering unsafe runs first
//! (`__dlpack_device__` for GPU rejection, capsule name check), so the
//! unsafe footprint is reached only after the input is known to satisfy
//! ADR-0030's CPU-only constraint.

use std::ffi::c_void;
use std::ptr::NonNull;

use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyCapsule, PyTuple};

// ---------------------------------------------------------------------------
// DLPack ABI: minimal #[repr(C)] mirror of dlpack.h. Only the fields we read
// are spelled out; the rest is byte-compatible because of the C layout.
// ---------------------------------------------------------------------------

const DL_DEVICE_CPU: i32 = 1;
const DL_DEVICE_CPU_PINNED: i32 = 3;

const DL_DTYPE_INT: u8 = 0;
const DL_DTYPE_UINT: u8 = 1;
const DL_DTYPE_FLOAT: u8 = 2;

#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct DLDevice {
    device_type: i32,
    device_id: i32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct DLDataType {
    code: u8,
    bits: u8,
    lanes: u16,
}

#[repr(C)]
struct DLTensor {
    data: *mut c_void,
    device: DLDevice,
    ndim: i32,
    dtype: DLDataType,
    shape: *mut i64,
    strides: *mut i64,
    byte_offset: u64,
}

#[repr(C)]
struct DLManagedTensor {
    dl_tensor: DLTensor,
    manager_ctx: *mut c_void,
    deleter: Option<unsafe extern "C" fn(*mut DLManagedTensor)>,
}

#[repr(C)]
struct DLPackVersion {
    major: u32,
    minor: u32,
}

#[repr(C)]
struct DLManagedTensorVersioned {
    version: DLPackVersion,
    manager_ctx: *mut c_void,
    deleter: Option<unsafe extern "C" fn(*mut DLManagedTensorVersioned)>,
    flags: u64,
    dl_tensor: DLTensor,
}

// ---------------------------------------------------------------------------
// Public surface — view-based, zero-copy.
//
// Each `extract_*` opens the DLPack capsule, validates layout, and returns a
// [`DLPackView`] that holds the capsule alive for the view's lifetime. Callers
// read through `view.as_slice()` (a `&[T]` borrow into the producer's buffer)
// and copy out only the elements they need into [`DetectionInput`].
// ---------------------------------------------------------------------------

/// Borrowed view into a DLPack producer's contiguous CPU buffer. Holds the
/// capsule by value so the producer's buffer stays valid for the view's
/// lifetime; dropping the view drops the capsule, which triggers the
/// producer's deleter exactly once.
pub(crate) struct DLPackView<'py, T> {
    /// Held purely for its `Drop`. Renamed `_capsule` only to silence
    /// "field never read" lints — the field is load-bearing for safety.
    _capsule: Bound<'py, PyCapsule>,
    data_ptr: NonNull<T>,
    len: usize,
}

impl<'py, T: Copy> DLPackView<'py, T> {
    /// Borrow the producer's buffer as a `&[T]`. Valid for the view's lifetime.
    pub(crate) fn as_slice(&self) -> &[T] {
        // SAFETY: `data_ptr` was validated by `open_cpu_tensor` to point at
        // `len` valid `T`-aligned elements within the producer's buffer.
        // The capsule we hold keeps that buffer alive for our lifetime.
        unsafe { std::slice::from_raw_parts(self.data_ptr.as_ptr(), self.len) }
    }

    /// Element count.
    pub(crate) fn len(&self) -> usize {
        self.len
    }
}

/// Extract `(N, K)` f64 row-major data from any DLPack-CPU producer. The
/// number of columns is checked against `expected_cols`. The `N` dim is
/// derivable from the returned view as `view.len() / expected_cols`.
pub(crate) fn extract_f64_2d<'py>(
    obj: &Bound<'py, PyAny>,
    field: &str,
    expected_cols: usize,
) -> PyResult<DLPackView<'py, f64>> {
    let (capsule, meta) = open_cpu_tensor(obj, field)?;
    expect_dtype(&meta, field, DL_DTYPE_FLOAT, 64)?;
    expect_ndim(&meta, field, 2)?;
    if meta.shape[1] as usize != expected_cols {
        return Err(PyValueError::new_err(format!(
            "detections.{field}: expected shape (N, {expected_cols}), got ({}, {})",
            meta.shape[0], meta.shape[1]
        )));
    }
    into_view::<f64>(capsule, &meta, field)
}

/// Extract `(N, K, 3)` f64 row-major data; returns the view and the K
/// dimension. Validates the trailing 3 (x, y, v).
pub(crate) fn extract_f64_3d_kp<'py>(
    obj: &Bound<'py, PyAny>,
    field: &str,
) -> PyResult<(DLPackView<'py, f64>, usize)> {
    let (capsule, meta) = open_cpu_tensor(obj, field)?;
    expect_dtype(&meta, field, DL_DTYPE_FLOAT, 64)?;
    expect_ndim(&meta, field, 3)?;
    if meta.shape[2] != 3 {
        return Err(PyValueError::new_err(format!(
            "detections.{field}: expected shape (N, K, 3), got ({}, {}, {})",
            meta.shape[0], meta.shape[1], meta.shape[2]
        )));
    }
    let k = meta.shape[1] as usize;
    let view = into_view::<f64>(capsule, &meta, field)?;
    Ok((view, k))
}

/// Extract `(N,)` f64 contiguous.
pub(crate) fn extract_f64_1d<'py>(
    obj: &Bound<'py, PyAny>,
    field: &str,
) -> PyResult<DLPackView<'py, f64>> {
    extract_1d::<f64>(obj, field, DL_DTYPE_FLOAT, 64)
}

/// Extract `(N,)` i64 contiguous.
pub(crate) fn extract_i64_1d<'py>(
    obj: &Bound<'py, PyAny>,
    field: &str,
) -> PyResult<DLPackView<'py, i64>> {
    extract_1d::<i64>(obj, field, DL_DTYPE_INT, 64)
}

/// Extract `(N,)` u32 contiguous.
pub(crate) fn extract_u32_1d<'py>(
    obj: &Bound<'py, PyAny>,
    field: &str,
) -> PyResult<DLPackView<'py, u32>> {
    extract_1d::<u32>(obj, field, DL_DTYPE_UINT, 32)
}

fn extract_1d<'py, T>(
    obj: &Bound<'py, PyAny>,
    field: &str,
    expected_code: u8,
    expected_bits: u8,
) -> PyResult<DLPackView<'py, T>> {
    let (capsule, meta) = open_cpu_tensor(obj, field)?;
    expect_dtype(&meta, field, expected_code, expected_bits)?;
    expect_ndim(&meta, field, 1)?;
    into_view::<T>(capsule, &meta, field)
}

// ---------------------------------------------------------------------------
// Internal: capsule consumption + validation
// ---------------------------------------------------------------------------

/// Validated DLPack metadata (shape, strides, dtype, base data pointer).
/// Returned by [`open_cpu_tensor`] alongside the live capsule, then
/// finalized by [`into_view`] into a typed [`DLPackView`].
struct CpuTensorMeta {
    shape: Vec<i64>,
    strides: Option<Vec<i64>>,
    dtype: DLDataType,
    /// Pointer into the producer's data buffer (`tensor.data + byte_offset`).
    data_ptr: NonNull<u8>,
}

/// Screen the object's device, pull a DLPack capsule, and validate the
/// shape/stride metadata. The capsule is returned by value so the caller
/// can hand it off to a [`DLPackView`] that keeps it alive for the
/// duration of the borrow; dropping the capsule invokes the producer's
/// deleter exactly once.
fn open_cpu_tensor<'py>(
    obj: &Bound<'py, PyAny>,
    field: &str,
) -> PyResult<(Bound<'py, PyCapsule>, CpuTensorMeta)> {
    screen_cpu_device(obj, field)?;

    let capsule_obj = obj.call_method0("__dlpack__").map_err(|e| {
        PyTypeError::new_err(format!(
            "detections.{field}: object does not support DLPack \
             (no __dlpack__ method): {e}"
        ))
    })?;
    let capsule: Bound<'py, PyCapsule> = capsule_obj.cast_into().map_err(|e| {
        PyTypeError::new_err(format!(
            "detections.{field}: __dlpack__ did not return a PyCapsule: {e}"
        ))
    })?;

    let dl_tensor: &DLTensor = read_capsule_tensor(&capsule, field)?;

    let device_type = dl_tensor.device.device_type;
    if device_type != DL_DEVICE_CPU && device_type != DL_DEVICE_CPU_PINNED {
        return Err(gpu_rejection_error(field, device_type));
    }

    if dl_tensor.ndim < 0 || dl_tensor.ndim > 8 {
        return Err(PyValueError::new_err(format!(
            "detections.{field}: implausible ndim {}",
            dl_tensor.ndim
        )));
    }
    let ndim = dl_tensor.ndim as usize;

    // SAFETY: the producer guarantees `shape` is valid for `ndim` int64
    // elements while the capsule lives.
    let shape: Vec<i64> = unsafe { std::slice::from_raw_parts(dl_tensor.shape, ndim) }.to_vec();
    for (i, &d) in shape.iter().enumerate() {
        if d < 0 {
            return Err(PyValueError::new_err(format!(
                "detections.{field}: shape[{i}] is negative ({d})"
            )));
        }
    }
    let strides: Option<Vec<i64>> = if dl_tensor.strides.is_null() {
        None
    } else {
        // SAFETY: same as shape; producer guarantees the buffer.
        Some(unsafe { std::slice::from_raw_parts(dl_tensor.strides, ndim) }.to_vec())
    };

    let data_addr = (dl_tensor.data as usize)
        .checked_add(dl_tensor.byte_offset as usize)
        .ok_or_else(|| {
            PyValueError::new_err(format!("detections.{field}: byte_offset overflow"))
        })?;
    let data_ptr = NonNull::new(data_addr as *mut u8).ok_or_else(|| {
        PyValueError::new_err(format!("detections.{field}: data pointer is null"))
    })?;

    let meta = CpuTensorMeta {
        shape,
        strides,
        dtype: dl_tensor.dtype,
        data_ptr,
    };
    Ok((capsule, meta))
}

/// Pair a validated capsule + meta into a typed view. Checks contiguity
/// and that `T`'s size matches the producer's dtype, then computes the
/// element count.
fn into_view<'py, T>(
    capsule: Bound<'py, PyCapsule>,
    meta: &CpuTensorMeta,
    field: &str,
) -> PyResult<DLPackView<'py, T>> {
    if !is_c_contiguous(meta) {
        return Err(PyTypeError::new_err(format!(
            "detections.{field}: array is not C-contiguous \
             (fix: pass np.ascontiguousarray(arr))"
        )));
    }
    let len = n_elements(meta).ok_or_else(|| {
        PyValueError::new_err(format!("detections.{field}: shape product overflows usize"))
    })?;
    let elem_bytes = (meta.dtype.bits as usize) / 8;
    if std::mem::size_of::<T>() != elem_bytes {
        return Err(PyValueError::new_err(format!(
            "detections.{field}: internal dtype size mismatch ({} vs {})",
            std::mem::size_of::<T>(),
            elem_bytes
        )));
    }
    Ok(DLPackView {
        _capsule: capsule,
        data_ptr: meta.data_ptr.cast::<T>(),
        len,
    })
}

/// Resolve the capsule pointer into a borrowed `&DLTensor`. The borrow is
/// valid for as long as `capsule` lives.
fn read_capsule_tensor<'a>(
    capsule: &'a Bound<'_, PyCapsule>,
    field: &str,
) -> PyResult<&'a DLTensor> {
    let name = capsule.name()?.ok_or_else(|| {
        PyTypeError::new_err(format!(
            "detections.{field}: DLPack capsule has no name (corrupt producer?)"
        ))
    })?;
    // SAFETY: `name` points at a NUL-terminated C string owned by the
    // capsule for the duration of this borrow.
    let name_bytes = unsafe { name.as_cstr() }.to_bytes();

    if name_bytes == b"dltensor" {
        let ptr = capsule.pointer_checked(Some(c"dltensor"))?;
        // SAFETY: the producer guarantees that capsules named "dltensor"
        // hold a valid `DLManagedTensor` pointer for the capsule lifetime.
        let mt: &DLManagedTensor = unsafe { &*ptr.as_ptr().cast::<DLManagedTensor>() };
        Ok(&mt.dl_tensor)
    } else if name_bytes == b"dltensor_versioned" {
        let ptr = capsule.pointer_checked(Some(c"dltensor_versioned"))?;
        // SAFETY: same as above for the versioned layout.
        let mt: &DLManagedTensorVersioned =
            unsafe { &*ptr.as_ptr().cast::<DLManagedTensorVersioned>() };
        Ok(&mt.dl_tensor)
    } else if name_bytes == b"used_dltensor" {
        Err(PyTypeError::new_err(format!(
            "detections.{field}: DLPack capsule has already been consumed"
        )))
    } else {
        Err(PyTypeError::new_err(format!(
            "detections.{field}: unexpected DLPack capsule name {:?}",
            String::from_utf8_lossy(name_bytes)
        )))
    }
}

/// Cheap GPU-rejection screen via `__dlpack_device__`. Errors with the
/// ADR-0030-mandated greppable message.
fn screen_cpu_device(obj: &Bound<'_, PyAny>, field: &str) -> PyResult<()> {
    let dev = obj.call_method0("__dlpack_device__").map_err(|e| {
        PyTypeError::new_err(format!(
            "detections.{field}: object does not support DLPack \
             (no __dlpack_device__ method): {e}"
        ))
    })?;
    let tup: Bound<'_, PyTuple> = dev.cast_into().map_err(|e| {
        PyTypeError::new_err(format!(
            "detections.{field}: __dlpack_device__ did not return a tuple: {e}"
        ))
    })?;
    let device_type: i32 = tup.get_item(0)?.extract()?;
    if device_type != DL_DEVICE_CPU && device_type != DL_DEVICE_CPU_PINNED {
        return Err(gpu_rejection_error(field, device_type));
    }
    Ok(())
}

/// ADR-0030 mandates a single greppable `vernier-0030` GPU-rejection
/// message; both the cheap pre-screen and the post-capsule check route
/// through here.
fn gpu_rejection_error(field: &str, device_type: i32) -> PyErr {
    PyTypeError::new_err(format!(
        "vernier-0030 does not accept GPU-resident detections; \
         move to CPU with .cpu() or .to('cpu') \
         (field detections.{field}, device_type={device_type})"
    ))
}

/// C-contiguity check: `strides == None` is contiguous by contract;
/// otherwise verify `strides[i] == product(shape[i+1..])` (in elements).
fn is_c_contiguous(meta: &CpuTensorMeta) -> bool {
    let strides = match &meta.strides {
        None => return true,
        Some(s) => s,
    };
    let mut expected: i64 = 1;
    for i in (0..meta.shape.len()).rev() {
        // Zero-length axes don't constrain stride.
        if meta.shape[i] != 0 && strides[i] != expected {
            return false;
        }
        // Treat overflow as non-contiguous rather than masking it as a match.
        match expected.checked_mul(meta.shape[i]) {
            Some(next) => expected = next,
            None => return false,
        }
    }
    true
}

fn n_elements(meta: &CpuTensorMeta) -> Option<usize> {
    let mut prod: usize = 1;
    for &d in &meta.shape {
        let d = usize::try_from(d).ok()?;
        prod = prod.checked_mul(d)?;
    }
    Some(prod)
}

fn expect_dtype(
    meta: &CpuTensorMeta,
    field: &str,
    expected_code: u8,
    expected_bits: u8,
) -> PyResult<()> {
    if meta.dtype.code == expected_code && meta.dtype.bits == expected_bits && meta.dtype.lanes == 1
    {
        return Ok(());
    }
    let got = describe_dtype(meta.dtype);
    let want = describe_dtype_code(expected_code, expected_bits);
    Err(PyTypeError::new_err(format!(
        "detections.{field}: expected {want}, got {got} \
         (fix: pass cast_inputs=True at construction or call .astype(np.{want}))"
    )))
}

fn expect_ndim(meta: &CpuTensorMeta, field: &str, expected: usize) -> PyResult<()> {
    if meta.shape.len() == expected {
        return Ok(());
    }
    Err(PyValueError::new_err(format!(
        "detections.{field}: expected {expected}-D array, got {}-D (shape {:?})",
        meta.shape.len(),
        meta.shape
    )))
}

fn describe_dtype(dt: DLDataType) -> String {
    let code = describe_dtype_code(dt.code, dt.bits);
    if dt.lanes == 1 {
        code
    } else {
        format!("{code} (lanes={})", dt.lanes)
    }
}

fn describe_dtype_code(code: u8, bits: u8) -> String {
    let prefix = match code {
        DL_DTYPE_INT => "int",
        DL_DTYPE_UINT => "uint",
        DL_DTYPE_FLOAT => "float",
        other => return format!("dtype(code={other}, bits={bits})"),
    };
    format!("{prefix}{bits}")
}
