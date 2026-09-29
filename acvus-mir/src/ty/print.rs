//! A concrete type in the language's own notation, for the IR printer:
//! `i64`, `&mut [i64]`, `{a: i64, b: i64}`, `Option<String>`,
//! `fn(i64) -> i64`, a declared type by its qualified name with its
//! arguments. Names come from the interner; nothing is written but the
//! notation.
//!
//! Deterministic: the same type prints the same text. An object's fields and
//! an enum's variants print in the order the type holds them.

use acvus_utils::Interner;

use crate::ty::{Concrete, Ty, Types};

/// `ty` written in the language's notation.
pub fn ty(types: &Types<Concrete>, names: &Interner, ty: Ty) -> String {
    todo!("{:?} {:?} {ty:?}", std::ptr::from_ref(types), std::ptr::from_ref(names))
}
