//! An `Args` owns words the call moved to its callee, and only the glue
//! knows which words those are: code outside `acvus-extern` builds none
//! (RFC-0097 rule 1).
use acvus_extern::{Args, Crossing, Owned, TypesOnly};

fn through_the_constructor<'c>(
    rt: Crossing<'c, TypesOnly>,
    run: &[()],
    site: &'c acvus_extern::ArgsSite,
) -> Args<'c, (Owned<TypesOnly>,), TypesOnly> {
    unsafe { Args::moved_in(rt, run, site) }
}

fn main() {}
