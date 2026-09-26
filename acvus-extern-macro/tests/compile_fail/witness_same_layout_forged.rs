//! A layout witness is made only by `acvus_extern::repr`'s constructors, each
//! bound by the `unsafe impl` that proves it (RFC-0102 rule 2). The body they
//! share is private to `repr`, so no other code vouches, even in `unsafe`.
use acvus_extern::repr::SameLayout;

fn main() {
    // SAFETY: deliberately false: `u64` is not a `f64`'s every byte.
    let _ = unsafe { SameLayout::<u64, f64>::vouched() };
}
