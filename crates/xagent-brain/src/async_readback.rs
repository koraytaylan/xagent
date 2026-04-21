//! Reusable async-readback plumbing for `wgpu` buffer mapping.
//!
//! Each async readback in [`crate::gpu_kernel`] needs to:
//!
//! 1. Call `map_async(MapMode::Read, …)` on one or more buffer slices,
//! 2. Keep a shared completion counter so the main thread knows when every
//!    callback has fired,
//! 3. Keep a shared error flag so a single failing mapping short-circuits the
//!    whole group without hanging forever.
//!
//! Before this module each call site hand-rolled the `Arc<AtomicU32>` +
//! `Arc<AtomicBool>` + closure trio. [`ReadbackTracker`] centralises that
//! ceremony so the call sites express intent (which slices map, how many
//! callbacks to expect) instead of plumbing.
//!
//! Buffer ownership and the post-completion `get_mapped_range` / `unmap`
//! sequence stay with the caller: staging buffers are often pre-allocated
//! and reused across readback rounds, and only the caller knows the exact
//! byte ranges and data layout.

use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;

/// High bit of the packed state word — set when any callback reported an error.
const ERROR_BIT: u32 = 1 << 31;
/// Low 31 bits of the packed state word — completion count.
const COUNT_MASK: u32 = ERROR_BIT - 1;

/// Current state of a [`ReadbackTracker`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ReadbackStatus {
    /// Fewer than `expected` callbacks have fired.
    Pending,
    /// All expected callbacks fired without error.
    Ready,
    /// All expected callbacks fired and at least one reported an error.
    Failed,
}

/// Shared tracker for a group of `wgpu::BufferSlice::map_async` callbacks.
///
/// Call [`ReadbackTracker::install`] once per buffer slice that participates
/// in the group. Each installed callback encodes both "completed" and "had
/// error" into a single packed `AtomicU32` so [`status`] can decide between
/// `Ready`, `Failed`, and `Pending` with a single atomic load.
///
/// # Packed state layout
///
/// - Bit 31 — error flag; set by [`ERROR_BIT`] when any callback reports an
///   error.
/// - Bits 0..=30 — completion count; incremented on every callback.
///
/// Using a single atomic (instead of separate `AtomicU32` + `AtomicBool`)
/// avoids subtle release-sequence reasoning across two distinct atomics:
/// once the caller's `Acquire` load observes `count >= expected`, any error
/// bit set by prior callbacks is visible in the same word.
///
/// # Callback ordering
///
/// On an errored callback the error bit is set **before** the count is
/// incremented. Callbacks therefore publish the error to any later load that
/// observes the completion increment — `status()` never reports `Ready` when
/// an errored callback has already pushed the count past `expected`.
///
/// # Reuse
///
/// Trackers can be reused across readback rounds on the same set of
/// pre-allocated staging buffers by calling [`reset`] before issuing the
/// next batch of `map_async` calls.
///
/// [`status`]: ReadbackTracker::status
/// [`reset`]: ReadbackTracker::reset
pub(crate) struct ReadbackTracker {
    /// Packed (error_bit | count) state shared with every installed callback.
    state: Arc<AtomicU32>,
    expected: u32,
}

impl ReadbackTracker {
    /// Create a tracker that expects `expected` `map_async` callbacks.
    ///
    /// # Panics
    ///
    /// Panics in debug builds when `expected` is zero (a zero-callback
    /// tracker would return [`ReadbackStatus::Ready`] immediately and is
    /// almost certainly a bug) or when `expected` would overflow the
    /// 31-bit count field (callers track at most a handful of buffer
    /// mappings — this limit is a sanity guard, not a real constraint).
    pub(crate) fn new(expected: u32) -> Self {
        debug_assert!(
            expected > 0,
            "ReadbackTracker::new called with expected = 0 — the tracker would report Ready before any map_async ran"
        );
        debug_assert!(
            expected <= COUNT_MASK,
            "ReadbackTracker::new: expected = {expected} overflows the 31-bit count field"
        );
        Self {
            state: Arc::new(AtomicU32::new(0)),
            expected,
        }
    }

    /// Reset the packed state so this tracker can be reused for the next
    /// round of `map_async` calls on the same pre-allocated buffers.
    pub(crate) fn reset(&self) {
        self.state.store(0, Ordering::Release);
    }

    /// Install a `MapMode::Read` callback on `slice`.
    ///
    /// The callback:
    /// - sets the shared error bit (via `fetch_or`) before incrementing the
    ///   count when `result.is_err()`,
    /// - increments the shared completion count on every outcome.
    ///
    /// Performing `fetch_or` **before** `fetch_add` guarantees that if a
    /// future `status()` load observes `count >= expected`, it also observes
    /// the error bit set by any errored callback — both operations modify
    /// the same atomic, so a single `Acquire` load is sufficient.
    pub(crate) fn install(&self, slice: wgpu::BufferSlice<'_>) {
        let state = Arc::clone(&self.state);
        slice.map_async(wgpu::MapMode::Read, move |result| {
            if result.is_err() {
                state.fetch_or(ERROR_BIT, Ordering::Release);
            }
            state.fetch_add(1, Ordering::Release);
        });
    }

    /// Current status of the tracker.
    ///
    /// - [`Pending`](ReadbackStatus::Pending) while fewer than `expected`
    ///   callbacks have fired.
    /// - [`Failed`](ReadbackStatus::Failed) once every callback has fired and
    ///   at least one reported an error.
    /// - [`Ready`](ReadbackStatus::Ready) once every callback has fired and
    ///   none reported an error.
    ///
    /// A single `Acquire` load is enough because count and error flag live
    /// in the same atomic word — release sequences across separate atomics
    /// are not required for correctness.
    pub(crate) fn status(&self) -> ReadbackStatus {
        let state = self.state.load(Ordering::Acquire);
        let count = state & COUNT_MASK;
        let error = (state & ERROR_BIT) != 0;
        if count < self.expected {
            return ReadbackStatus::Pending;
        }
        if error {
            return ReadbackStatus::Failed;
        }
        ReadbackStatus::Ready
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Simulate the shared-state update the tracker installs on each slice.
    /// Tests call this instead of a real `map_async` callback so the
    /// tracker's bookkeeping can be exercised without a wgpu device.
    fn fake_callback(tracker: &ReadbackTracker, result: Result<(), wgpu::BufferAsyncError>) {
        let state = Arc::clone(&tracker.state);
        // Mirrors ReadbackTracker::install exactly: error bit set first so
        // the count increment publishes it.
        if result.is_err() {
            state.fetch_or(ERROR_BIT, Ordering::Release);
        }
        state.fetch_add(1, Ordering::Release);
    }

    #[test]
    fn pending_until_all_callbacks_fire() {
        let tracker = ReadbackTracker::new(3);
        assert_eq!(tracker.status(), ReadbackStatus::Pending);
        fake_callback(&tracker, Ok(()));
        assert_eq!(tracker.status(), ReadbackStatus::Pending);
        fake_callback(&tracker, Ok(()));
        assert_eq!(tracker.status(), ReadbackStatus::Pending);
        fake_callback(&tracker, Ok(()));
        assert_eq!(tracker.status(), ReadbackStatus::Ready);
    }

    #[test]
    fn ready_only_when_all_succeed() {
        let tracker = ReadbackTracker::new(2);
        fake_callback(&tracker, Ok(()));
        fake_callback(&tracker, Ok(()));
        assert_eq!(tracker.status(), ReadbackStatus::Ready);
    }

    #[test]
    fn failed_when_any_callback_errors_after_all_fired() {
        let tracker = ReadbackTracker::new(3);
        fake_callback(&tracker, Ok(()));
        fake_callback(&tracker, Err(wgpu::BufferAsyncError));
        // Still pending: only 2/3 callbacks fired.
        assert_eq!(tracker.status(), ReadbackStatus::Pending);
        fake_callback(&tracker, Ok(()));
        assert_eq!(tracker.status(), ReadbackStatus::Failed);
    }

    #[test]
    fn reset_clears_completion_and_error() {
        let tracker = ReadbackTracker::new(2);
        fake_callback(&tracker, Err(wgpu::BufferAsyncError));
        fake_callback(&tracker, Ok(()));
        assert_eq!(tracker.status(), ReadbackStatus::Failed);

        tracker.reset();
        assert_eq!(tracker.status(), ReadbackStatus::Pending);

        fake_callback(&tracker, Ok(()));
        fake_callback(&tracker, Ok(()));
        assert_eq!(tracker.status(), ReadbackStatus::Ready);
    }

    #[test]
    fn error_flag_short_circuits_once_all_callbacks_fire() {
        // Even when the very last callback is the failure, the tracker reports
        // Failed rather than Ready — the error flag is sticky.
        let tracker = ReadbackTracker::new(2);
        fake_callback(&tracker, Ok(()));
        fake_callback(&tracker, Err(wgpu::BufferAsyncError));
        assert_eq!(tracker.status(), ReadbackStatus::Failed);
    }

    #[test]
    fn expected_count_one_reports_ready_after_single_callback() {
        let tracker = ReadbackTracker::new(1);
        assert_eq!(tracker.status(), ReadbackStatus::Pending);
        fake_callback(&tracker, Ok(()));
        assert_eq!(tracker.status(), ReadbackStatus::Ready);
    }

    #[test]
    fn multiple_errors_collapse_into_single_error_bit() {
        // fetch_or is idempotent — two errored callbacks leave the error bit
        // set exactly once, and the count still reflects both callbacks.
        let tracker = ReadbackTracker::new(2);
        fake_callback(&tracker, Err(wgpu::BufferAsyncError));
        fake_callback(&tracker, Err(wgpu::BufferAsyncError));
        assert_eq!(tracker.status(), ReadbackStatus::Failed);
    }

    #[test]
    fn errored_callback_sets_error_bit_before_count_reaches_expected() {
        // The closure sets the error bit *before* incrementing the count,
        // so there is no window in which an observer sees count == expected
        // but misses an error from a callback that has already completed.
        let tracker = ReadbackTracker::new(2);
        fake_callback(&tracker, Err(wgpu::BufferAsyncError));
        // After only one callback, error bit is set but count < expected.
        let raw = tracker.state.load(Ordering::Acquire);
        assert_eq!(raw & COUNT_MASK, 1);
        assert_ne!(raw & ERROR_BIT, 0);
        fake_callback(&tracker, Ok(()));
        assert_eq!(tracker.status(), ReadbackStatus::Failed);
    }

    #[test]
    #[should_panic(expected = "expected = 0")]
    fn zero_expected_panics_in_debug() {
        let _ = ReadbackTracker::new(0);
    }
}
