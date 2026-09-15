//! E2E tests for script-mode IR: loops, branches, SSA, function calls.
//!
//! Each test compiles a script source -> MIR and snapshots the printed IR.
//! Tests are grouped by category with both soundness and completeness direction.

use acvus_mir::ty::{Param, Ty};
use acvus_mir_test::{compile_script_ir, compile_script_mode_raw};
use acvus_utils::Interner;
use rustc_hash::FxHashMap;

fn ctx(i: &Interner, entries: &[(&str, Ty)]) -> FxHashMap<acvus_utils::Astr, Ty> {
    entries
        .iter()
        .map(|(name, ty)| (i.intern(name), ty.clone()))
        .collect()
}

// =======================================================================
//  1. Loop (iteration)
// =======================================================================

#[test]
fn loop_context_write_phi() {
    // SSA PHI at loop header: @count written inside loop
    let i = Interner::new();
    let c = ctx(
        &i,
        &[
            (
                "items",
                Ty::Array(Box::new(Ty::Int), acvus_mir::ty::LenTerm::Known(3)),
            ),
            ("count", Ty::Int),
        ],
    );
    let ir = compile_script_mode_raw(
        &i,
        "let it = @items | iter; while let Some(x) = next(&mut it) { @count = @count + 1; } @count",
        &c,
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

// =======================================================================
//  2. Branch (match-bind / if-let)
// =======================================================================

#[test]
fn branch_simple_bind() {
    // Irrefutable: x = @data { body } - no branching needed
    let i = Interner::new();
    let c = ctx(&i, &[("data", Ty::Int), ("out", Ty::Int)]);
    let ir = compile_script_ir(&i, "x = @data { @out = x + 1; }; @out", &c).unwrap();
    insta::assert_snapshot!(ir);
}

#[test]
fn branch_refutable_literal() {
    // Refutable: literal match - needs test + branch
    let i = Interner::new();
    let c = ctx(&i, &[("val", Ty::Int), ("out", Ty::Int)]);
    let ir = compile_script_ir(&i, "42 = @val { @out = 1; }; @out", &c).unwrap();
    insta::assert_snapshot!(ir);
}

#[test]
fn branch_destructure_object() {
    let i = Interner::new();
    let obj_ty = Ty::Object(FxHashMap::from_iter([
        (i.intern("name"), Ty::String),
        (i.intern("age"), Ty::Int),
    ]));
    let c = ctx(&i, &[("user", obj_ty), ("out", Ty::String)]);
    let ir = compile_script_ir(&i, "{ name, age, } = @user { @out = name; }; @out", &c).unwrap();
    insta::assert_snapshot!(ir);
}

#[test]
fn branch_nested_match() {
    let i = Interner::new();
    let c = ctx(&i, &[("a", Ty::Int), ("b", Ty::Int), ("out", Ty::Int)]);
    let ir = compile_script_ir(&i, "x = @a { y = @b { @out = x + y; }; }; @out", &c).unwrap();
    insta::assert_snapshot!(ir);
}

#[test]
fn branch_context_write_in_refutable() {
    // Context write inside refutable branch - needs PHI at merge
    let i = Interner::new();
    let c = ctx(&i, &[("val", Ty::Int), ("out", Ty::Int)]);
    let ir = compile_script_ir(&i, "42 = @val { @out = 99; }; @out", &c).unwrap();
    insta::assert_snapshot!(ir);
}

// =======================================================================
//  3. SSA
// =======================================================================

#[test]
fn ssa_store_load_forwarding() {
    // Context write then read - SSA should forward the stored value
    let i = Interner::new();
    let c = ctx(&i, &[("x", Ty::Int)]);
    let ir = compile_script_ir(&i, "@x = 42; @x", &c).unwrap();
    insta::assert_snapshot!(ir);
}

#[test]
fn ssa_write_in_branch_phi() {
    // Context write in one branch - PHI at merge point
    let i = Interner::new();
    let c = ctx(&i, &[("cond", Ty::Int), ("x", Ty::Int)]);
    let ir = compile_script_ir(&i, "42 = @cond { @x = 1; }; @x", &c).unwrap();
    insta::assert_snapshot!(ir);
}

#[test]
fn ssa_write_in_loop_phi() {
    // Context write in loop - loop-carried PHI
    let i = Interner::new();
    let c = ctx(
        &i,
        &[
            (
                "items",
                Ty::Array(Box::new(Ty::Int), acvus_mir::ty::LenTerm::Known(3)),
            ),
            ("acc", Ty::Int),
        ],
    );
    let ir = compile_script_mode_raw(
        &i,
        "let it = @items | iter; while let Some(x) = next(&mut it) { @acc = @acc + x; } @acc",
        &c,
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

#[test]
fn ssa_multiple_contexts() {
    // Independent SSA chains for different contexts
    let i = Interner::new();
    let c = ctx(&i, &[("a", Ty::Int), ("b", Ty::Int)]);
    let ir = compile_script_ir(&i, "@a = @a + 1; @b = @b + 2; @a + @b", &c).unwrap();
    insta::assert_snapshot!(ir);
}

#[test]
fn ssa_sequential_writes() {
    // Multiple writes to same context - only last value visible
    let i = Interner::new();
    let c = ctx(&i, &[("x", Ty::Int)]);
    let ir = compile_script_ir(&i, "@x = 1; @x = 2; @x = 3; @x", &c).unwrap();
    insta::assert_snapshot!(ir);
}

// =======================================================================
//  4. Function calls
// =======================================================================

#[test]
fn func_pipe_chain() {
    let i = Interner::new();
    let c = ctx(&i, &[]);
    let ir = compile_script_ir(&i, "[1, 2, 3] | filter(|x| -> *x > 0) | collect", &c).unwrap();
    insta::assert_snapshot!(ir);
}

#[test]
fn func_to_string_in_bind() {
    let i = Interner::new();
    let c = ctx(&i, &[("val", Ty::Int)]);
    let ir = compile_script_ir(&i, "to_string(@val)", &c).unwrap();
    insta::assert_snapshot!(ir);
}

// =======================================================================
//  5. Combined scenarios
// =======================================================================

#[test]
fn combined_nested_loop_context() {
    // Inner loop writes, outer reads after
    let i = Interner::new();
    let c = ctx(
        &i,
        &[
            (
                "outer",
                Ty::Array(
                    Box::new(Ty::Array(
                        Box::new(Ty::Int),
                        acvus_mir::ty::LenTerm::Known(3),
                    )),
                    acvus_mir::ty::LenTerm::Known(3),
                ),
            ),
            ("total", Ty::Int),
        ],
    );
    let ir = compile_script_mode_raw(
        &i,
        "let rows = @outer | iter; while let Some(row) = next(&mut rows) { let xs = row | iter; while let Some(x) = next(&mut xs) { @total = @total + x; } } @total",
        &c,
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

// =======================================================================
//  6. Soundness - reject invalid programs
// =======================================================================

#[test]
fn reject_iterate_non_iterable() {
    // Int is not iterable
    let i = Interner::new();
    let c = ctx(&i, &[("val", Ty::Int)]);
    let result = compile_script_ir(&i, "x in @val { }; x", &c);
    assert!(result.is_err(), "expected error for iterating over Int");
}

#[test]
fn reject_type_mismatch_context_store() {
    // Storing String into Int context
    let i = Interner::new();
    let c = ctx(&i, &[("x", Ty::Int)]);
    let result = compile_script_ir(&i, r#"@x = "hello"; @x"#, &c);
    assert!(
        result.is_err(),
        "expected error for type mismatch on context store"
    );
}

// =======================================================================
//  Lent places (RFC-0018): a parameter's mode is its type, unified at the call
// =======================================================================

fn items_ctx(i: &Interner) -> rustc_hash::FxHashMap<acvus_utils::Astr, Ty> {
    ctx(
        &i,
        &[(
            "items",
            Ty::Array(Box::new(Ty::Int), acvus_mir::ty::LenTerm::Known(3)),
        )],
    )
}

#[test]
fn a_lending_parameter_rejects_a_value_argument() {
    let i = Interner::new();
    let err = compile_script_mode_raw(&i, "let it = @items | iter; next(it)", &items_ctx(&i))
        .unwrap_err();
    assert!(err.contains("type mismatch"), "{err}");
    assert!(err.contains(", got Iterator<"), "{err}");
}

#[test]
fn a_lending_parameter_rejects_the_other_mode() {
    let i = Interner::new();
    let err = compile_script_mode_raw(&i, "let it = @items | iter; next(&it)", &items_ctx(&i))
        .unwrap_err();
    assert!(err.contains("type mismatch"), "{err}");
    assert!(err.contains(", got &Iterator<"), "{err}");
}

#[test]
fn only_a_place_can_be_lent() {
    let i = Interner::new();
    let err =
        compile_script_mode_raw(&i, "next(&mut (@items | iter))", &items_ctx(&i)).unwrap_err();
    assert!(err.contains("can be referenced"), "{err}");
}
