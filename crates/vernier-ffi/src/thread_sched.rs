//! Best-effort thread scheduling helpers shared between
//! [`crate::background`] (instance, ADR-0014) and
//! [`crate::background_streaming`] (semantic + panoptic).
//!
//! Both wrappers spawn a single dedicated worker thread and apply the
//! user's nice / affinity preferences before pulling the first
//! message. Failures are reported as `Err(String)` so the FFI surface
//! can emit a single `UserWarning` and continue — scheduling
//! preferences are advisory, not load-bearing.

/// Linear interpolation: nice = -20 → 99 (highest priority),
/// nice = 19 → 0 (lowest). Constants are part of `set_thread_nice`'s
/// public contract, kept at module scope so both callers see the same
/// mapping.
const NICE_RANGE: f64 = 39.0;
const PRIORITY_MAX: f64 = 99.0;

/// Map a POSIX-style nice value (`-20` highest priority, `19` lowest)
/// to `thread-priority`'s 0..=99 cross-platform priority and apply it
/// to the current thread.
pub(crate) fn set_thread_nice(nice: i32) -> Result<(), String> {
    use thread_priority::{set_current_thread_priority, ThreadPriority, ThreadPriorityValue};
    let clamped = nice.clamp(-20, 19);
    let priority_value: u8 = ((f64::from(19 - clamped)) * (PRIORITY_MAX / NICE_RANGE))
        .round()
        .clamp(0.0, PRIORITY_MAX) as u8;
    let value = ThreadPriorityValue::try_from(priority_value)
        .map_err(|e| format!("invalid priority value {priority_value}: {e:?}"))?;
    set_current_thread_priority(ThreadPriority::Crossplatform(value))
        .map_err(|e| format!("set_current_thread_priority: {e:?}"))
}

/// Pin the current thread to `core`. `core_affinity::set_for_current`
/// returns `false` on failure rather than an error type, so we
/// synthesize a message here.
pub(crate) fn set_thread_affinity(core: usize) -> Result<(), String> {
    let core_id = core_affinity::CoreId { id: core };
    if core_affinity::set_for_current(core_id) {
        Ok(())
    } else {
        Err(format!("set_for_current({core}) returned false"))
    }
}

/// Apply nice and (optionally) affinity, in that order. The FFI side
/// reads the composite outcome once after spawn and warns if it's
/// `Err`. Both calls are best-effort.
pub(crate) fn apply_scheduling(nice: i32, affinity: Option<usize>) -> Result<(), String> {
    set_thread_nice(nice).map_err(|e| format!("nice: {e}"))?;
    if let Some(core) = affinity {
        set_thread_affinity(core).map_err(|e| format!("affinity: {e}"))?;
    }
    Ok(())
}
