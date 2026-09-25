//! A type-identity witness is made only by `repr::same_type`, which compares
//! the two `TypeId`s; its field is private, so it cannot be written as a
//! literal outside acvus-extern (RFC-0080 rule 5).
use std::marker::PhantomData;

use acvus_extern::repr::SameType;

fn main() {
    let _: SameType<u64, String> = SameType(PhantomData);
}
