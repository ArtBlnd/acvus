//! `#[extern_fn(ensures(..))]` (RFC-0082 rule 4): the forms the vocabulary
//! holds build, and a word outside it is refused.
#[test]
fn a_postcondition_in_the_vocabulary_builds() {
    trybuild::TestCases::new().pass("tests/ensures_pass/*.rs");
}
