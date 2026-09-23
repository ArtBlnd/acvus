//! A payload of another crate's generic type, which `UniformPayload` does
//! not reach, is refused without `unsafe(uniform_payload)`.
use std::collections::BTreeMap;

use acvus_extern::{ExternType, Var, kind};

#[derive(ExternType)]
#[repr(transparent)]
#[extern_type(name = "Index")]
struct Index<T>(BTreeMap<i64, T>)
where
    T: Var<kind::Type>;

fn main() {}
