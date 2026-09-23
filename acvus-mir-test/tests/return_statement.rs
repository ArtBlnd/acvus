//! What the checker makes of `return e`: the shapes it admits, and the three
//! it refuses.

use acvus_mir::ty::Ty;
use acvus_mir_test::{compile_script_mode_raw, compile_to_ir};
use acvus_utils::{Astr, Interner};
use rustc_hash::FxHashMap;

fn no_context() -> FxHashMap<Astr, Ty> {
    FxHashMap::default()
}

#[test]
fn a_return_before_the_tail_compiles() {
    let i = Interner::new();
    compile_script_mode_raw(&i, "let x = 1; return x + 1; 99", &no_context())
        .unwrap_or_else(|e| panic!("{e}"));
}

/// `return` is an expression of type `!` and not a statement, which is what
/// lets it stand where a value is read.
#[test]
fn a_return_in_one_branch_leaves_the_other_branch_s_type() {
    let i = Interner::new();
    compile_script_mode_raw(
        &i,
        "let x = if true { return 1 } else { 2 }; x",
        &no_context(),
    )
    .unwrap_or_else(|e| panic!("{e}"));
}

#[test]
fn a_return_in_a_template_is_refused() {
    let i = Interner::new();
    let err = compile_to_ir(&i, "{{ return 1 }}", &no_context())
        .expect_err("a template is the body `?` is refused in too");
    assert!(
        err.contains("`return` needs a function to return from"),
        "{err}"
    );
}

/// There is no `return;`: every body returns a value (RFC-0054), so the
/// production takes an expression and the parse error names it.
#[test]
fn a_return_without_a_value_does_not_parse() {
    let i = Interner::new();
    let err = acvus_ast::parse_script(&i, "let x = 1; return; x").expect_err("no `return;`");
    assert_eq!(
        err.errors[0].kind.to_string(),
        "expected an expression, found `;`"
    );
}

/// A `return` fixes an undeclared body's return type, and what the source
/// wrote after it is checked like anything else: unreachable statements are
/// not exempt, and the tail still meets the type the `return` left with.
#[test]
fn what_stands_after_a_return_still_meets_the_body_s_return_type() {
    let i = Interner::new();
    let err = compile_script_mode_raw(&i, "return 1; \"no\".to_string()", &no_context())
        .expect_err("the tail leaves with a String where the `return` left with an i64");
    assert!(err.contains("type mismatch"), "{err}");
}

/// The array's release on the `return` edge covers the elements the loop
/// has not taken (RFC-0057 rule 6), so the `return` is admitted as `break`
/// and `?` are.
#[test]
fn a_return_out_of_an_array_of_owners_compiles() {
    let i = Interner::new();
    compile_script_mode_raw(
        &i,
        "let a = [\"ab\".to_string()]; for s in a { return len(&s) as i64; } 0",
        &no_context(),
    )
    .unwrap_or_else(|e| panic!("{e}"));
}
