//! The layout half of a canonical read (RFC-0076): a read between two types
//! of one size and alignment builds, and one between two of different sizes
//! fails at monomorphization. The `const` assert is evaluated when the
//! reading function is instantiated, which `cargo check` does not reach, so
//! the pass case is what makes trybuild build the fail case.
#[test]
fn a_canonical_read_is_checked_for_layout_when_built() {
    let t = trybuild::TestCases::new();
    t.pass("tests/layout_pass/*.rs");
    t.compile_fail("tests/layout_fail/*.rs");
}
