//! RFC-0106 rule 3: each recursive entry of a compiler walk runs through
//! `grow`. On a native target it moves the walk onto a fresh heap segment
//! when its thread's stack runs low, so a script within `NESTING_MAX`
//! compiles on any thread. `wasm32` cannot switch stacks; there the bound
//! alone keeps the walk inside the linear stack.

/// The stack a walk may still spend between two entries through `grow`.
/// The largest such stretch measured is not a walk's level but the parser's
/// `__reduce`, one frame of 668 KiB in a debug build of the grammar that
/// counts nesting (292 KiB before it), measured 2026-09-26.
/// `acvus-interpreter-test`'s `nesting_bound` compiles every kind of level at
/// the bound on a 256 KiB thread, and fails when this does not hold.
#[cfg(not(target_arch = "wasm32"))]
const RED_ZONE: usize = 1 << 20;

#[cfg(not(target_arch = "wasm32"))]
const SEGMENT: usize = 4 << 20;

#[cfg(not(target_arch = "wasm32"))]
#[inline]
pub fn grow<R, F>(f: F) -> R
where
    F: FnOnce() -> R,
{
    stacker::maybe_grow(RED_ZONE, SEGMENT, f)
}

#[cfg(target_arch = "wasm32")]
#[inline(always)]
pub fn grow<R, F>(f: F) -> R
where
    F: FnOnce() -> R,
{
    f()
}
