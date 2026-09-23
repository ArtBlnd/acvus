//! `BTreeMap` is another crate's generic type, which `UniformPayload` does
//! not reach; the author asserts the payload's obligation by hand.
use std::collections::BTreeMap;

use acvus_extern::{ExternType, Var, kind};

#[derive(ExternType)]
#[repr(transparent)]
#[extern_type(name = "Index")]
#[extern_type(unsafe(uniform_payload))]
struct Index<T>(BTreeMap<i64, T>)
where
    T: Var<kind::Type>;

fn main() {}
