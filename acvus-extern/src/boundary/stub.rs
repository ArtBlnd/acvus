//! A runtime for the crate's own tests: its only live operation is a value's
//! release, which it counts. Every kind is `NoKind`, reached by no
//! test.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use acvus_utils::Interner;

use crate::boundary::entry::Settled;
use crate::boundary::signed::{SignedS1, SignedS2, SignedS3};
use crate::boundary::uniform::{Primitive, Release, Releasing};
use crate::runtime::{
    Opaque, OpaqueKind, OptionKind, PrimitiveKind, RefKind, RefTarget, ResultKind, Runtime, StringKind, TupleKind,
};

/// A value that counts its releases and names no storage.
#[derive(Clone, Copy)]
pub(crate) struct Counted {
    pub(crate) releases: &'static AtomicUsize,
    pub(crate) id: u32,
}

impl Release for Counted {
    fn release(self, _releasing: Releasing) {
        self.releases.fetch_add(1, Ordering::SeqCst);
    }
}

pub(crate) struct Stub;

// SAFETY: `Stub` lends no storage, makes no reference value and keeps
// nothing: every operation that would is unreachable.
unsafe impl SignedS1 for Stub {}
// SAFETY: as for S1.
unsafe impl SignedS2 for Stub {}
// SAFETY: as for S1.
unsafe impl SignedS3 for Stub {}

pub(crate) enum NoKind {}

impl Runtime for Stub {
    type Value = Counted;

    type Opaque = NoKind;
    type Primitive = NoKind;
    type String = NoKind;
    type Option = NoKind;
    type Result = NoKind;
    type Tuple = NoKind;
    type Ref = NoKind;

    fn interner(&self) -> &Interner {
        unreachable!("the stub interns no name")
    }

    async fn sleep(&self, _: Duration) {
        unreachable!("the stub does not wait")
    }
}

impl OpaqueKind<Stub> for NoKind {

    fn give<T: Opaque>(_: &Stub, _: T, _: Settled<'_>) -> Counted {
        unreachable!()
    }
    fn take<T: Opaque>(_: &Stub, _: Counted, _: Settled<'_>) -> T {
        unreachable!()
    }
    fn shared<'a, T: Opaque + 'a>(_: &'a Stub, _: &'a Counted, _: Settled<'a>) -> &'a T {
        unreachable!()
    }
    fn exclusive<'a, T: Opaque + 'a>(_: &'a Stub, _: &'a mut Counted, _: Settled<'a>) -> &'a mut T {
        unreachable!()
    }
    fn shared_ref<'a, T: Opaque + 'a>(_: &'a Stub, _: Counted, _: Settled<'a>) -> &'a T {
        unreachable!()
    }
    fn exclusive_ref<'a, T: Opaque + 'a>(_: &'a Stub, _: Counted, _: Settled<'a>) -> &'a mut T {
        unreachable!()
    }
}

impl PrimitiveKind<Stub> for NoKind {

    fn give<P: Primitive>(_: &Stub, _: P, _: Settled<'_>) -> Counted {
        unreachable!()
    }
    fn take<P: Primitive>(_: &Stub, _: Counted, _: Settled<'_>) -> P {
        unreachable!()
    }
    fn shared<'a, P: Primitive>(_: &'a Stub, _: &'a Counted, _: Settled<'a>) -> &'a P {
        unreachable!()
    }
    fn exclusive<'a, P: Primitive>(_: &'a Stub, _: &'a mut Counted, _: Settled<'a>) -> &'a mut P {
        unreachable!()
    }
    fn shared_ref<'a, P: Primitive>(_: &'a Stub, _: Counted, _: Settled<'a>) -> &'a P {
        unreachable!()
    }
    fn exclusive_ref<'a, P: Primitive>(_: &'a Stub, _: Counted, _: Settled<'a>) -> &'a mut P {
        unreachable!()
    }
}

impl StringKind<Stub> for NoKind {

    fn give(_: &Stub, _: String, _: Settled<'_>) -> Counted {
        unreachable!()
    }
    fn take(_: &Stub, _: Counted, _: Settled<'_>) -> String {
        unreachable!()
    }
    fn shared<'a>(_: &'a Stub, _: &'a Counted, _: Settled<'a>) -> &'a String {
        unreachable!()
    }
    fn exclusive<'a>(_: &'a Stub, _: &'a mut Counted, _: Settled<'a>) -> &'a mut String {
        unreachable!()
    }
    fn shared_ref<'a>(_: &'a Stub, _: Counted, _: Settled<'a>) -> &'a String {
        unreachable!()
    }
    fn exclusive_ref<'a>(_: &'a Stub, _: Counted, _: Settled<'a>) -> &'a mut String {
        unreachable!()
    }
}

impl OptionKind<Stub> for NoKind {
    fn give(_: &Stub, _: Option<Counted>, _: Settled<'_>) -> Counted {
        unreachable!()
    }
    fn take(_: &Stub, _: Counted, _: Settled<'_>) -> Option<Counted> {
        unreachable!()
    }
    fn shared<'a>(_: &'a Stub, _: &'a Counted, _: Settled<'a>) -> Option<&'a Counted> {
        unreachable!()
    }
    fn exclusive<'a>(_: &'a Stub, _: &'a mut Counted, _: Settled<'a>) -> Option<&'a mut Counted> {
        unreachable!()
    }
    fn shared_ref<'a>(_: &'a Stub, _: Counted, _: Settled<'a>) -> Option<&'a Counted> {
        unreachable!()
    }
    fn exclusive_ref<'a>(_: &'a Stub, _: Counted, _: Settled<'a>) -> Option<&'a mut Counted> {
        unreachable!()
    }
}

impl ResultKind<Stub> for NoKind {
    fn give(_: &Stub, _: Result<Counted, Counted>, _: Settled<'_>) -> Counted {
        unreachable!()
    }
    fn take(_: &Stub, _: Counted, _: Settled<'_>) -> Result<Counted, Counted> {
        unreachable!()
    }
    fn shared<'a>(_: &'a Stub, _: &'a Counted, _: Settled<'a>) -> Result<&'a Counted, &'a Counted> {
        unreachable!()
    }
    fn exclusive<'a>(_: &'a Stub, _: &'a mut Counted, _: Settled<'a>) -> Result<&'a mut Counted, &'a mut Counted> {
        unreachable!()
    }
    fn shared_ref<'a>(_: &'a Stub, _: Counted, _: Settled<'a>) -> Result<&'a Counted, &'a Counted> {
        unreachable!()
    }
    fn exclusive_ref<'a>(_: &'a Stub, _: Counted, _: Settled<'a>) -> Result<&'a mut Counted, &'a mut Counted> {
        unreachable!()
    }
}

impl TupleKind<Stub> for NoKind {
    fn give<const N: usize>(_: &Stub, _: [Counted; N], _: Settled<'_>) -> Counted {
        unreachable!()
    }
    fn take<const N: usize>(_: &Stub, _: Counted, _: Settled<'_>) -> [Counted; N] {
        unreachable!()
    }
    fn shared<'a, const N: usize>(_: &'a Stub, _: &'a Counted, _: Settled<'a>) -> &'a [Counted; N] {
        unreachable!()
    }
    fn exclusive<'a, const N: usize>(_: &'a Stub, _: &'a mut Counted, _: Settled<'a>) -> &'a mut [Counted; N] {
        unreachable!()
    }
    fn shared_ref<'a, const N: usize>(_: &'a Stub, _: Counted, _: Settled<'a>) -> &'a [Counted; N] {
        unreachable!()
    }
    fn exclusive_ref<'a, const N: usize>(_: &'a Stub, _: Counted, _: Settled<'a>) -> &'a mut [Counted; N] {
        unreachable!()
    }
}

impl RefKind<Stub> for NoKind {
    fn reference<'a, P: ?Sized>(_: &'a Stub, _: RefTarget<'a, P>, _: Settled<'a>) -> Counted {
        unreachable!()
    }
    fn target<'l, P: ?Sized>(_: &'l Stub, _: Counted, _: Settled<'l>) -> &'l P {
        unreachable!()
    }
    fn target_mut<'l, P: ?Sized>(_: &'l Stub, _: Counted, _: Settled<'l>) -> &'l mut P {
        unreachable!()
    }
}
