//! One binary: `cargo test -p acvus-mir-test --test all <file>::<test>`.
//! Each module here uses the harness (`Case`); the files under `tests/`
//! not named here are the old corpus, waiting to be moved onto it.

mod extern_fn;
mod harness;
