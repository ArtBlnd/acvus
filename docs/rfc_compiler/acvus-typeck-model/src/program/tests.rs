use super::*;
use crate::fixture::decls;
use crate::join::ERROR;

fn error_of(part: &str) -> Term<Name> {
    Term::Enum(Name::new(ERROR), [(Name::new("A"), Name::new(part))].into())
}

#[test]
#[should_panic(expected = "no `Error` (l.91)")]
fn the_text_writes_no_error() {
    let mut p = ProgramBuilder::new(decls());
    p.written("E", error_of("A"));
}

#[test]
#[should_panic(expected = "no `Error` (l.91)")]
fn a_context_gives_no_error() {
    let mut p = ProgramBuilder::new(decls());
    p.host("E", error_of("A"));
}

#[test]
fn the_embeddings_result_may_hold_an_error() {
    let mut p = ProgramBuilder::new(decls());
    p.outward("E", error_of("A"));
}
