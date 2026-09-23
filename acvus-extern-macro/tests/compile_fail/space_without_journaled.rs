//! A type a space holds is read and written through its `Journaled` impl,
//! which the author writes: the derive's `space` switch without one does
//! not compile (RFC-0033).
use std::marker::PhantomData;

use acvus_extern::{ExternType, Var, kind};

#[derive(ExternType)]
#[extern_type(name = "History", space)]
#[repr(transparent)]
struct History<I>(Vec<i64>, PhantomData<I>)
where
    I: Var<kind::Identity>;

fn main() {}
