//! A crossing receives the capability, which crosses any word at any type,
//! so implementing one is a promise the compiler cannot check: `OneValue` is
//! an `unsafe trait`, and a hand-written impl that does not say `unsafe impl`
//! is refused. The derives and the library's macros write theirs as
//! `unsafe impl`, so their users write no `unsafe`.
use acvus_extern::{Crossing, OneValue, Runtime};

struct MyType(i64);

acvus_extern::within_every!(MyType);

impl<Rt> OneValue<Rt> for MyType
where
    Rt: Runtime,
{
    fn erase(self, rt: Crossing<'_, Rt>) -> Rt::Value {
        <i64 as OneValue<Rt>>::erase(self.0, rt)
    }

    unsafe fn materialize(rt: Crossing<'_, Rt>, value: Rt::Value) -> Self {
        MyType(unsafe { <i64 as OneValue<Rt>>::materialize(rt, value) })
    }
}

fn main() {}
