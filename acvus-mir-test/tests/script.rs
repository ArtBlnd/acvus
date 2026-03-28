//! E2E tests for script-mode IR: loops, branches, SSA, function calls.
//!
//! Each test compiles a script source → MIR and snapshots the printed IR.
//! Tests are grouped by category with both soundness and completeness direction.

use acvus_mir::ty::{Effect, Param, Ty};
use acvus_mir_test::compile_script_ir;
use acvus_utils::Interner;
use rustc_hash::FxHashMap;

fn ctx(i: &Interner, entries: &[(&str, Ty)]) -> FxHashMap<acvus_utils::Astr, Ty> {
    entries
        .iter()
        .map(|(name, ty)| (i.intern(name), ty.clone()))
        .collect()
}

// ═══════════════════════════════════════════════════════════════════════
//  1. Loop (iteration)
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn loop_simple_iteration() {
    let i = Interner::new();
    let c = ctx(
        &i,
        &[("items", Ty::List(Box::new(Ty::Int))), ("sum", Ty::Int)],
    );
    let ir = compile_script_ir(&i, "x in @items { @sum = @sum + x; }; @sum", &c).unwrap();
    insta::assert_snapshot!(ir);
}

#[test]
fn loop_nested_iteration() {
    let i = Interner::new();
    let c = ctx(
        &i,
        &[
            ("matrix", Ty::List(Box::new(Ty::List(Box::new(Ty::Int))))),
            ("sum", Ty::Int),
        ],
    );
    let ir = compile_script_ir(
        &i,
        "row in @matrix { x in row { @sum = @sum + x; }; }; @sum",
        &c,
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

#[test]
fn loop_context_write_phi() {
    // SSA PHI at loop header: @count written inside loop
    let i = Interner::new();
    let c = ctx(
        &i,
        &[("items", Ty::List(Box::new(Ty::Int))), ("count", Ty::Int)],
    );
    let ir = compile_script_ir(&i, "x in @items { @count = @count + 1; }; @count", &c).unwrap();
    insta::assert_snapshot!(ir);
}

#[test]
fn loop_with_function_call() {
    let i = Interner::new();
    let c = ctx(
        &i,
        &[
            ("items", Ty::List(Box::new(Ty::Int))),
            ("result", Ty::String),
        ],
    );
    let ir = compile_script_ir(
        &i,
        r#"x in @items { @result = @result + to_string(x); }; @result"#,
        &c,
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

#[test]
fn loop_range_iteration() {
    let i = Interner::new();
    let c = ctx(&i, &[("sum", Ty::Int)]);
    let ir = compile_script_ir(&i, "x in 0..10 { @sum = @sum + x; }; @sum", &c).unwrap();
    insta::assert_snapshot!(ir);
}

// ═══════════════════════════════════════════════════════════════════════
//  2. Branch (match-bind / if-let)
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn branch_simple_bind() {
    // Irrefutable: x = @data { body } — no branching needed
    let i = Interner::new();
    let c = ctx(&i, &[("data", Ty::Int), ("out", Ty::Int)]);
    let ir = compile_script_ir(&i, "x = @data { @out = x + 1; }; @out", &c).unwrap();
    insta::assert_snapshot!(ir);
}

#[test]
fn branch_refutable_literal() {
    // Refutable: literal match — needs test + branch
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
    // Context write inside refutable branch — needs PHI at merge
    let i = Interner::new();
    let c = ctx(&i, &[("val", Ty::Int), ("out", Ty::Int)]);
    let ir = compile_script_ir(&i, "42 = @val { @out = 99; }; @out", &c).unwrap();
    insta::assert_snapshot!(ir);
}

// ═══════════════════════════════════════════════════════════════════════
//  3. SSA
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn ssa_store_load_forwarding() {
    // Context write then read — SSA should forward the stored value
    let i = Interner::new();
    let c = ctx(&i, &[("x", Ty::Int)]);
    let ir = compile_script_ir(&i, "@x = 42; @x", &c).unwrap();
    insta::assert_snapshot!(ir);
}

#[test]
fn ssa_write_in_branch_phi() {
    // Context write in one branch — PHI at merge point
    let i = Interner::new();
    let c = ctx(&i, &[("cond", Ty::Int), ("x", Ty::Int)]);
    let ir = compile_script_ir(&i, "42 = @cond { @x = 1; }; @x", &c).unwrap();
    insta::assert_snapshot!(ir);
}

#[test]
fn ssa_write_in_loop_phi() {
    // Context write in loop — loop-carried PHI
    let i = Interner::new();
    let c = ctx(
        &i,
        &[("items", Ty::List(Box::new(Ty::Int))), ("acc", Ty::Int)],
    );
    let ir = compile_script_ir(&i, "x in @items { @acc = @acc + x; }; @acc", &c).unwrap();
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
    // Multiple writes to same context — only last value visible
    let i = Interner::new();
    let c = ctx(&i, &[("x", Ty::Int)]);
    let ir = compile_script_ir(&i, "@x = 1; @x = 2; @x = 3; @x", &c).unwrap();
    insta::assert_snapshot!(ir);
}

// ═══════════════════════════════════════════════════════════════════════
//  4. Function calls
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn func_builtin_in_loop() {
    let i = Interner::new();
    let c = ctx(
        &i,
        &[("items", Ty::List(Box::new(Ty::Int))), ("out", Ty::String)],
    );
    let ir = compile_script_ir(
        &i,
        r#"x in @items { @out = @out + to_string(x); }; @out"#,
        &c,
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

#[test]
fn func_pipe_chain() {
    let i = Interner::new();
    let c = ctx(&i, &[("items", Ty::List(Box::new(Ty::Int)))]);
    let ir = compile_script_ir(&i, "@items | filter(|x| -> x > 0) | collect", &c).unwrap();
    insta::assert_snapshot!(ir);
}

#[test]
fn func_to_string_in_bind() {
    let i = Interner::new();
    let c = ctx(&i, &[("val", Ty::Int)]);
    let ir = compile_script_ir(&i, "to_string(@val)", &c).unwrap();
    insta::assert_snapshot!(ir);
}

// ═══════════════════════════════════════════════════════════════════════
//  5. Combined scenarios
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn combined_loop_with_branch() {
    // Iteration with conditional context write inside
    let i = Interner::new();
    let c = ctx(
        &i,
        &[("items", Ty::List(Box::new(Ty::Int))), ("count", Ty::Int)],
    );
    let ir = compile_script_ir(
        &i,
        "x in @items { 0 = x { @count = @count + 1; }; }; @count",
        &c,
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

#[test]
fn combined_accumulate_in_loop() {
    let i = Interner::new();
    let c = ctx(
        &i,
        &[
            ("items", Ty::List(Box::new(Ty::Int))),
            ("sum", Ty::Int),
            ("product", Ty::Int),
        ],
    );
    let ir = compile_script_ir(
        &i,
        "x in @items { @sum = @sum + x; @product = @product * x; }; @sum + @product",
        &c,
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

#[test]
fn combined_nested_loop_context() {
    // Inner loop writes, outer reads after
    let i = Interner::new();
    let c = ctx(
        &i,
        &[
            ("outer", Ty::List(Box::new(Ty::List(Box::new(Ty::Int))))),
            ("total", Ty::Int),
        ],
    );
    let ir = compile_script_ir(
        &i,
        "row in @outer { x in row { @total = @total + x; }; }; @total",
        &c,
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

#[test]
fn combined_bind_then_iterate() {
    // Bind a value, then iterate using it
    let i = Interner::new();
    let c = ctx(
        &i,
        &[
            (
                "data",
                Ty::Object(FxHashMap::from_iter([(
                    i.intern("items"),
                    Ty::List(Box::new(Ty::Int)),
                )])),
            ),
            ("sum", Ty::Int),
        ],
    );
    let ir = compile_script_ir(
        &i,
        "{ items, } = @data { x in items { @sum = @sum + x; }; }; @sum",
        &c,
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

#[test]
fn combined_multiple_loops_sequential() {
    // Two sequential loops writing to the same context
    let i = Interner::new();
    let c = ctx(
        &i,
        &[
            ("a", Ty::List(Box::new(Ty::Int))),
            ("b", Ty::List(Box::new(Ty::Int))),
            ("sum", Ty::Int),
        ],
    );
    let ir = compile_script_ir(
        &i,
        "x in @a { @sum = @sum + x; }; y in @b { @sum = @sum + y; }; @sum",
        &c,
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

// ═══════════════════════════════════════════════════════════════════════
//  6. Soundness — reject invalid programs
// ═══════════════════════════════════════════════════════════════════════

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
