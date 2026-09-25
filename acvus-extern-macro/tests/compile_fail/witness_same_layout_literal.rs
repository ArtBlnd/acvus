//! A layout witness has a private field, so code outside acvus-extern cannot
//! write one as a literal (RFC-0080 rule 5).
use std::marker::PhantomData;

use acvus_extern::repr::SameLayout;

fn main() {
    let _: SameLayout<u64, f64> = SameLayout(PhantomData);
}
