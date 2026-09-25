//! What a handler is called with, in place of `rt` and `frame`.

use crate::runtime::Runtime;

/// RFC-0023 rule 2, RFC-0067 rule 7.
///
/// `frame` is private: the owner of a `Ctx` writes the cells its frame
/// names after a handler returns, so a handler must not move that frame
/// out or put another in its place. The runtime reaches it through
/// `frame_mut`, which is `unsafe`.
pub struct Ctx<'a, Rt>
where
    Rt: Runtime,
{
    pub rt: &'a Rt,
    frame: Rt::Frame<'a>,
    recv: *const Rt::Value,
}

impl<'a, Rt> Ctx<'a, Rt>
where
    Rt: Runtime,
{
    const NO_RECEIVER: *const Rt::Value = std::ptr::null();

    /// # Safety
    /// This `Ctx` is the one owner of `frame`, and the cells `frame` names
    /// stay live for as long as the `Ctx` is used. A handler that could
    /// build a second `Ctx` safely could exchange its own with it, as
    /// `ctx_of`'s contract rules out.
    pub unsafe fn new(rt: &'a Rt, frame: Rt::Frame<'a>) -> Self {
        Ctx {
            rt,
            frame,
            recv: Self::NO_RECEIVER,
        }
    }

    /// The frame this `Ctx` was built over, for the runtime that runs a
    /// closure on it.
    ///
    /// # Safety
    /// The frame stays the one this `Ctx` was built with: the caller moves
    /// no frame out through the reference and writes no other frame into
    /// it (`mem::swap`, `mem::replace`, assignment).
    #[inline(always)]
    pub unsafe fn frame_mut(&mut self) -> &mut Rt::Frame<'a> {
        &mut self.frame
    }

    #[inline(always)]
    pub fn into_frame(self) -> Rt::Frame<'a> {
        self.frame
    }

    #[inline(always)]
    pub(crate) fn name_receiver(&mut self, at: &Rt::Value) {
        self.recv = at;
    }

    /// The mono glue's, and no handler's.
    ///
    /// # Safety
    /// An `Instance` or `InstanceOf` call named a receiver for the call now
    /// running, and the value it named is live for `'r`.
    #[doc(hidden)]
    #[inline(always)]
    pub unsafe fn receiver<'r>(&mut self) -> &'r Rt::Value {
        debug_assert!(
            self.recv != Self::NO_RECEIVER,
            "a mono glue read a receiver that no Instance::call named"
        );
        // SAFETY: the caller's contract.
        unsafe { &*self.recv }
    }
}

// SAFETY: the receiver word is an address into storage the caller of the
// running instance call owns for that call, and a `Ctx` crosses no thread
// between the `name_receiver` that wrote it and the `receiver` that reads
// it; the rest of a `Ctx` is `Send` where the host's frame is.
unsafe impl<'a, Rt> Send for Ctx<'a, Rt>
where
    Rt: Runtime,
    Rt::Frame<'a>: Send,
{
}
