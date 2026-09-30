//! E2E tests for script-mode IR: loops, branches, SSA, function calls.
//!
//! Each test compiles a script source -> MIR and snapshots the printed IR.
//! Tests are grouped by category with both soundness and completeness direction.

use acvus_mir::ty::{ObjectTy, Ty};
use acvus_mir_test::{compile_script_ir, compile_script_mode_raw, compile_script_optimized};
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
                Ty::Array(Box::new(Ty::I64), acvus_mir::ty::LenTerm::Known(3)),
            ),
            ("count", Ty::I64),
        ],
    );
    let ir = compile_script_mode_raw(
        &i,
        "let it = @items | into_iter; while let Some(x) = next(&mut it) { @count = @count + 1; } @count",
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
    let c = ctx(&i, &[("data", Ty::I64), ("out", Ty::I64)]);
    let ir = compile_script_ir(&i, "if let x = @data { @out = x + 1; }; @out", &c).unwrap();
    insta::assert_snapshot!(ir);
}

#[test]
fn branch_refutable_literal() {
    // Refutable: literal match - needs test + branch
    let i = Interner::new();
    let c = ctx(&i, &[("val", Ty::I64), ("out", Ty::I64)]);
    let ir = compile_script_ir(&i, "if let 42 = @val { @out = 1; }; @out", &c).unwrap();
    insta::assert_snapshot!(ir);
}

#[test]
fn branch_destructure_object() {
    let i = Interner::new();
    let obj_ty = Ty::Object(ObjectTy::written(FxHashMap::from_iter([
        (i.intern("name"), Ty::String),
        (i.intern("age"), Ty::I64),
    ])));
    let c = ctx(&i, &[("user", obj_ty), ("out", Ty::String)]);
    let ir = compile_script_ir(
        &i,
        "if let { name, age, } = @user { @out = name; }; @out",
        &c,
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

#[test]
fn branch_nested_match() {
    let i = Interner::new();
    let c = ctx(&i, &[("a", Ty::I64), ("b", Ty::I64), ("out", Ty::I64)]);
    let ir = compile_script_ir(
        &i,
        "if let x = @a { if let y = @b { @out = x + y; }; }; @out",
        &c,
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

#[test]
fn branch_context_write_in_refutable() {
    // Context write inside refutable branch - needs PHI at merge
    let i = Interner::new();
    let c = ctx(&i, &[("val", Ty::I64), ("out", Ty::I64)]);
    let ir = compile_script_ir(&i, "if let 42 = @val { @out = 99; }; @out", &c).unwrap();
    insta::assert_snapshot!(ir);
}

// =======================================================================
//  3. SSA
// =======================================================================

#[test]
fn ssa_store_load_forwarding() {
    // Context write then read - SSA should forward the stored value
    let i = Interner::new();
    let c = ctx(&i, &[("x", Ty::I64)]);
    let ir = compile_script_ir(&i, "@x = 42; @x", &c).unwrap();
    insta::assert_snapshot!(ir);
}

#[test]
fn ssa_write_in_branch_phi() {
    // Context write in one branch - PHI at merge point
    let i = Interner::new();
    let c = ctx(&i, &[("cond", Ty::I64), ("x", Ty::I64)]);
    let ir = compile_script_ir(&i, "if let 42 = @cond { @x = 1; }; @x", &c).unwrap();
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
                Ty::Array(Box::new(Ty::I64), acvus_mir::ty::LenTerm::Known(3)),
            ),
            ("acc", Ty::I64),
        ],
    );
    let ir = compile_script_mode_raw(
        &i,
        "let it = @items | into_iter; while let Some(x) = next(&mut it) { @acc = @acc + x; } @acc",
        &c,
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

#[test]
fn ssa_multiple_contexts() {
    // Independent SSA chains for different contexts
    let i = Interner::new();
    let c = ctx(&i, &[("a", Ty::I64), ("b", Ty::I64)]);
    let ir = compile_script_ir(&i, "@a = @a + 1; @b = @b + 2; @a + @b", &c).unwrap();
    insta::assert_snapshot!(ir);
}

#[test]
fn ssa_sequential_writes() {
    // Multiple writes to same context - only last value visible
    let i = Interner::new();
    let c = ctx(&i, &[("x", Ty::I64)]);
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
    let ir = compile_script_ir(
        &i,
        "[1, 2, 3] | into_iter | filter(|x| -> *x > 0) | collect",
        &c,
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

#[test]
fn func_to_string_in_bind() {
    let i = Interner::new();
    let c = ctx(&i, &[("val", Ty::I64)]);
    let ir = compile_script_ir(&i, "@val.to_string()", &c).unwrap();
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
                        Box::new(Ty::I64),
                        acvus_mir::ty::LenTerm::Known(3),
                    )),
                    acvus_mir::ty::LenTerm::Known(3),
                ),
            ),
            ("total", Ty::I64),
        ],
    );
    let ir = compile_script_mode_raw(
        &i,
        "let rows = @outer | into_iter; while let Some(row) = next(&mut rows) { let xs = row | into_iter; while let Some(x) = next(&mut xs) { @total = @total + x; } } @total",
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
    let c = ctx(&i, &[("val", Ty::I64)]);
    let result = compile_script_ir(&i, "x in @val { }; x", &c);
    assert!(result.is_err(), "expected error for iterating over Int");
}

#[test]
fn reject_type_mismatch_context_store() {
    // Storing String into Int context
    let i = Interner::new();
    let c = ctx(&i, &[("x", Ty::I64)]);
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
            Ty::Array(Box::new(Ty::I64), acvus_mir::ty::LenTerm::Known(3)),
        )],
    )
}

#[test]
fn a_lending_parameter_rejects_a_value_argument() {
    let i = Interner::new();
    let err = compile_script_mode_raw(&i, "let it = @items | into_iter; next(it)", &items_ctx(&i))
        .unwrap_err();
    assert!(err.contains("type mismatch"), "{err}");
    assert!(err.contains(", got Items<"), "{err}");
}

#[test]
fn a_lending_parameter_rejects_the_other_mode() {
    let i = Interner::new();
    let err = compile_script_mode_raw(&i, "let it = @items | into_iter; next(&it)", &items_ctx(&i))
        .unwrap_err();
    assert!(err.contains("type mismatch"), "{err}");
    assert!(err.contains(", got &Items<"), "{err}");
}

#[test]
fn a_value_lent_at_a_parameter_is_bound_to_a_temporary() {
    let i = Interner::new();
    let ir = compile_script_mode_raw(&i, "next(&mut (@items | into_iter))", &items_ctx(&i))
        .expect("the iterator is bound for the call that borrows it");
    assert!(ir.contains("assign v5 = r2"), "{ir}");
    assert!(ir.contains("ref &mut v5"), "{ir}");
}

// =======================================================================
//  `let` binds, `x = e;` assigns (RFC-0045)
// =======================================================================

#[test]
fn an_assignment_to_a_name_no_binding_introduced_is_refused() {
    let i = Interner::new();
    let err = compile_script_mode_raw(&i, "x = 1; x", &FxHashMap::default()).unwrap_err();
    assert!(
        err.contains(
            "cannot assign to `x`: no binding named `x` is in scope; `let x = ...;` binds it"
        ),
        "{err}"
    );
}

#[test]
fn an_assignment_to_a_name_the_lambda_captured_is_refused() {
    // A capture is by value (RFC-0018), so the store would write the
    // lambda's copy and never reach the binding the writer named.
    let i = Interner::new();
    let err = compile_script_mode_raw(
        &i,
        "let x = 1; let f = |q| -> { x = 2; x }; f(0)",
        &FxHashMap::default(),
    )
    .unwrap_err();
    assert!(
        err.contains("cannot assign to `x`: it is captured by the lambda, not bound in it"),
        "{err}"
    );
}

#[test]
fn a_name_the_lambda_bound_itself_is_assignable_in_it() {
    let i = Interner::new();
    compile_script_mode_raw(
        &i,
        "let f = |q| -> { let x = q; x = x + 1; x }; f(1)",
        &FxHashMap::default(),
    )
    .expect("`x` is the lambda body's own binding");
}

#[test]
fn a_let_in_a_nested_block_does_not_leak_past_it() {
    // The inner `let` shadows for the block only: after it, `n` is the
    // outer binding, and its type is the outer one's.
    let i = Interner::new();
    let err = compile_script_mode_raw(
        &i,
        "let n = 1; while n < 2 { let s = \"in\"; n = n + 1; } s",
        &FxHashMap::default(),
    )
    .unwrap_err();
    assert!(err.contains("undefined variable `s`"), "{err}");
}

#[test]
fn an_assignment_in_a_loop_body_is_the_outer_binding() {
    // The body writes the binding the loop's enclosing block introduced,
    // so the loop header joins the two reaching definitions.
    let i = Interner::new();
    let ir = compile_script_optimized(
        &i,
        "let n = 0; while n != 3 { n = n + 1; } n",
        &FxHashMap::default(),
    )
    .expect("`n` is assigned in the body and read after it");
    assert!(
        ir.contains("L0(r1: i64):"),
        "the loop header takes the assigned `n` as a block parameter: {ir}"
    );
}

// -- RFC-0069 rule 1: a closure does not leave the run it was made in ----------

#[test]
fn a_program_whose_result_is_a_closure_is_refused() {
    let i = Interner::new();
    let err = compile_script_mode_raw(&i, "|x| -> x + 1", &items_ctx(&i)).unwrap_err();
    assert!(
        err.contains("the program's result holds a closure"),
        "{err}"
    );
}

#[test]
fn a_program_whose_result_holds_a_closure_in_data_is_refused() {
    let i = Interner::new();
    let err = compile_script_mode_raw(&i, "Some(|x| -> x + 1)", &items_ctx(&i)).unwrap_err();
    assert!(
        err.contains("the program's result holds a closure"),
        "{err}"
    );
}

#[test]
fn a_lambda_may_return_a_closure_and_the_program_its_value() {
    let i = Interner::new();
    compile_script_mode_raw(
        &i,
        "let add = |a| -> |b| -> a + b; let add_two = add(2); add_two(3)",
        &items_ctx(&i),
    )
    .expect("a closure crosses between bodies of one run");
}
