//! An `Output` is the glue's, filled at the type its call site settled: code
//! outside `acvus-extern` builds none, and so no `Finished` (RFC-0097 rule 3).
use acvus_extern::{Output, TypesOnly};

fn built() -> Output<'static, (), TypesOnly> {
    Output { sealed: false }
}

fn main() {}
