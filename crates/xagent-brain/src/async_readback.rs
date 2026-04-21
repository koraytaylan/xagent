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

use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::Arc;

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
/// in the group. Each installed callback increments a shared counter on any
/// outcome and sets a shared error flag on failure, so [`status`] can report
/// `Ready`, `Failed`, or `Pending` in constant time.
///
/// # Error semantics
///
/// The counter is incremented on **every** callback invocation (success or
/// failure), so `completed >= expected` eventually holds whether or not a
/// mapping failed. Callers must therefore check [`ReadbackStatus::Failed`]
/// before consuming mapped ranges — a failed buffer is not safe to read.
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
    completed: Arc<AtomicU32>,
    had_error: Arc<AtomicBool>,
    expected: u32,
}

impl ReadbackTracker {
    /// Create a tracker that expects `expected` `map_async` callbacks.
    ///
    /// # Panics
    ///
    /// Panics in debug builds when `expected` is zero — a zero-callback
    /// tracker would return [`ReadbackStatus::Ready`] immediately and is
    /// almost certainly a bug.
    pub(crate) fn new(expected: u32) -> Self {
        debug_assert!(
            expected > 0,
            "ReadbackTracker::new called with expected = 0 — the tracker would report Ready before any map_async ran"
        );
        Self {
            completed: Arc::new(AtomicU32::new(0)),
            had_error: Arc::new(AtomicBool::new(false)),
            expected,
        }
    }

    /// Reset the completion counter and error flag so this tracker can be
    /// reused for the next round of `map_async` calls on the same pre-allocated
    /// buffers.
    pub(crate) fn reset(&self) {
        self.completed.store(0, Ordering::Release);
        self.had_error.store(false, Ordering::Release);
    }

    /// Install a `MapMode::Read` callback on `slice`.
    ///
    /// The callback:
    /// - increments the shared completion counter on any outcome,
    /// - sets the shared error flag when `result.is_err()`.
    pub(crate) fn install(&self, slice: wgpu::BufferSlice<'_>) {
        let completed = Arc::clone(&self.completed);
        let had_error = Arc::clone(&self.had_error);
        slice.map_async(wgpu::MapMode::Read, move |result| {
            if result.is_err() {
                had_error.store(true, Ordering::Release);
            }
            completed.fetch_add(1, Ordering::Release);
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
    pub(crate) fn status(&self) -> ReadbackStatus {
        if self.completed.load(Ordering::Acquire) < self.expected {
            return ReadbackStatus::Pending;
        }
        if self.had_error.load(Ordering::Acquire) {
            return ReadbackStatus::Failed;
        }
        ReadbackStatus::Ready
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Simulate the shared-counter closure the tracker installs on each
    /// slice. Tests call this instead of a real `map_async` callback so the
    /// tracker's bookkeeping can be exercised without a wgpu device.
    fn fake_callback(tracker: &ReadbackTracker, result: Result<(), wgpu::BufferAsyncError>) {
        let completed = Arc::clone(&tracker.completed);
        let had_error = Arc::clone(&tracker.had_error);
        // Mirrors ReadbackTracker::install exactly.
        if result.is_err() {
            had_error.store(true, Ordering::Release);
        }
        completed.fetch_add(1, Ordering::Release);
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
    #[should_panic(expected = "expected = 0")]
    fn zero_expected_panics_in_debug() {
        let _ = ReadbackTracker::new(0);
    }
}
