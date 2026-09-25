//! A layout witness is made only by `same_layout!` inside acvus-extern
//! (RFC-0080 rule 5). Its constructor is private to the crate, so code
//! outside cannot vouch for two types, even in an `unsafe` block.
use acvus_extern::repr::SameLayout;

fn main() {
    // SAFETY: deliberately false: `u64` is not a `f64`'s every byte.
    let _ = unsafe { SameLayout::<u64, f64>::vouched() };
}
