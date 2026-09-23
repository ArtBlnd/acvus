//! A box of `Vec<Erased<Rt, i64>>` is keyed at its canonical form,
//! `Vec<Owned<Rt>>`, and the two agree in size and alignment, so the read
//! between them builds. Its presence makes trybuild build, not only check,
//! the case beside it.
use acvus_extern::derive::canonical;
use acvus_extern::{Erased, TypesOnly};

fn main() {
    let _erase: fn(&TypesOnly, Vec<Erased<TypesOnly, i64>>) -> () =
        canonical::erase::<Vec<Erased<TypesOnly, i64>>, TypesOnly>;
}
