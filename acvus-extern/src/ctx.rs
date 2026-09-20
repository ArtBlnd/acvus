//! What a handler is called with, in place of `rt` and `frame`.

use crate::runtime::Runtime;

/// RFC-0050 rule 6.
pub struct Ctx<'a, Rt>
where
    Rt: Runtime,
{
    pub rt: &'a Rt,
    pub frame: Rt::Frame<'a>,
    recv: *mut Rt::Value,
}

impl<'a, Rt> Ctx<'a, Rt>
where
    Rt: Runtime,
{
    const NO_RECEIVER: *mut Rt::Value = std::ptr::null_mut();

    pub fn new(rt: &'a Rt, frame: Rt::Frame<'a>) -> Self {
        Ctx {
            rt,
            frame,
            recv: Self::NO_RECEIVER,
        }
    }

    #[inline(always)]
    pub fn name_receiver(&mut self, at: &mut Rt::Value) {
        self.recv = at;
    }

    /// # Safety
    /// An `Instance::call` named a receiver for the call now running, and
    /// the value it named is live and exclusively named for `'r`.
    #[inline(always)]
    pub unsafe fn receiver<'r>(&mut self) -> &'r mut Rt::Value {
        debug_assert!(
            self.recv != Self::NO_RECEIVER,
            "a mono glue read a receiver that no Instance::call named"
        );
        // SAFETY: the caller's contract.
        unsafe { &mut *self.recv }
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
