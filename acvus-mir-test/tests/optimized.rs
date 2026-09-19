//! Full optimization pipeline E2E tests.
//!
//! Each test compiles a complex, real-world script through the **full** pipeline:
//! extract -> infer -> lower -> SROA -> SSA -> Inline -> SpawnSplit -> CodeMotion -> Reorder -> SSA -> RegColor -> Validate
//!
//! Two snapshots per test:
//! - `{name}@raw` - unoptimized, raw lowered MIR
//! - `{name}@optimized` - after full optimization pipeline

use acvus_mir::ty::{ObjectTy, Ty};
use acvus_mir_test::{compile_script_optimized, compile_script_raw};
use acvus_utils::Interner;
use rustc_hash::FxHashMap;

fn ctx(i: &Interner, entries: &[(&str, Ty)]) -> FxHashMap<acvus_utils::Astr, Ty> {
    entries
        .iter()
        .map(|(name, ty)| (i.intern(name), ty.clone()))
        .collect()
}

fn obj(i: &Interner, fields: &[(&str, Ty)]) -> Ty {
    Ty::Object(ObjectTy::written(
        fields
            .iter()
            .map(|(name, ty)| (i.intern(name), ty.clone()))
            .collect(),
    ))
}

fn snap_both(i: &Interner, source: &str, c: &FxHashMap<acvus_utils::Astr, Ty>) -> (String, String) {
    let raw = compile_script_raw(i, source, c).unwrap();
    let opt = compile_script_optimized(i, source, c).unwrap();
    (raw, opt)
}

// =======================================================================
//  1. Nested loop with conditional accumulator
//     - SSA: loop phi x 2 (pos_sum, neg_sum), branch phi within inner loop
//     - Reorder: context store ordering
// =======================================================================

// =======================================================================
//  2. Object field read-modify-write across branches
//     - SROA: multiple field projections on same context
//     - SSA: branch phi on context after conditional write
// =======================================================================

#[test]
fn field_read_modify_write_branch() {
    let i = Interner::new();
    let c = ctx(
        &i,
        &[(
            "stats",
            obj(
                &i,
                &[
                    ("count", Ty::I64),
                    ("threshold", Ty::I64),
                    ("exceeded", Ty::Bool),
                ],
            ),
        )],
    );
    let src = r#"
        let count = @stats.count + 1;
        if let true = count > @stats.threshold {
            @stats.count = count;
            @stats.exceeded = true;
        };
        @stats.count
    "#;
    let (raw, opt) = snap_both(&i, src, &c);
    insta::assert_snapshot!("field_read_modify_write_branch@raw", raw);
    insta::assert_snapshot!("field_read_modify_write_branch@optimized", opt);
}

// =======================================================================
//  3. Multi-context dataflow with transformation
//     - Inline: to_string inlined
//     - SSA: multiple context reads feeding into computation
//     - SROA: @output whole write
// =======================================================================

#[test]
fn multi_context_dataflow() {
    let i = Interner::new();
    let c = ctx(
        &i,
        &[
            ("a", Ty::I64),
            ("b", Ty::I64),
            ("scale", Ty::I64),
            ("output", Ty::I64),
        ],
    );
    let src = r#"
        let sum = @a + @b;
        let scaled = sum * @scale;
        @output = scaled + @a;
        @output
    "#;
    let (raw, opt) = snap_both(&i, src, &c);
    insta::assert_snapshot!("multi_context_dataflow@raw", raw);
    insta::assert_snapshot!("multi_context_dataflow@optimized", opt);
}

// =======================================================================
//  4. Object construction from context fields
//     - SROA: @user.name, @user.age field reads
//     - SSA: pure computation chain
//     - RegColor: many intermediate values
// =======================================================================

#[test]
fn object_construct_from_fields() {
    let i = Interner::new();
    let c = ctx(
        &i,
        &[
            ("user", obj(&i, &[("name", Ty::String), ("age", Ty::I64)])),
            ("min_score", Ty::I64),
            (
                "output",
                obj(
                    &i,
                    &[
                        ("label", Ty::String),
                        ("score", Ty::I64),
                        ("eligible", Ty::Bool),
                    ],
                ),
            ),
        ],
    );
    let src = r#"
        let score = @user.age * 2;
        let label = @user.name + " (score: " + score.to_string() + ")";
        let eligible = score > @min_score;
        @output = { label: label, score: score, eligible: eligible, };
        @output.score
    "#;
    let (raw, opt) = snap_both(&i, src, &c);
    insta::assert_snapshot!("object_construct_from_fields@raw", raw);
    insta::assert_snapshot!("object_construct_from_fields@optimized", opt);
}

// =======================================================================
//  5. Diamond control flow with divergent context mutations
//     - SSA: @high, @low writes in separate branches -> phi at join
//     - Multiple contexts mutated conditionally
// =======================================================================

#[test]
fn diamond_divergent_context_mutations() {
    let i = Interner::new();
    let c = ctx(
        &i,
        &[
            ("input", Ty::I64),
            ("high", Ty::I64),
            ("low", Ty::I64),
            ("output", Ty::I64),
        ],
    );
    let src = r#"
        let x = @input;
        if let true = x > 100 {
            @high = @high + 1;
            @output = x * 2;
        };
        if let true = x <= 100 {
            @low = @low + 1;
            @output = x + 10;
        };
        @output
    "#;
    let (raw, opt) = snap_both(&i, src, &c);
    insta::assert_snapshot!("diamond_divergent_context_mutations@raw", raw);
    insta::assert_snapshot!("diamond_divergent_context_mutations@optimized", opt);
}

// =======================================================================
//  6. Loop with search pattern + accumulator
//     - SSA: found, idx both loop phi + branch phi within loop body
//     - Complex phi nesting: loop x branch
// =======================================================================

// =======================================================================
//  7. Chained field mutations on same object
//     - SROA: 4 field projections on @state -> decompose each
//     - SSA: sequential writes, no phi but many SROA temporaries
//     - RegColor: high register pressure from SROA expansion
// =======================================================================

#[test]
fn chained_field_mutations() {
    let i = Interner::new();
    let c = ctx(
        &i,
        &[(
            "state",
            obj(
                &i,
                &[
                    ("step", Ty::I64),
                    ("value", Ty::I64),
                    ("multiplier", Ty::I64),
                    ("done", Ty::Bool),
                    ("max_steps", Ty::I64),
                ],
            ),
        )],
    );
    let src = r#"
        @state.step = @state.step + 1;
        @state.value = @state.value * @state.multiplier;
        @state.done = @state.step >= @state.max_steps;
        @state.value
    "#;
    let (raw, opt) = snap_both(&i, src, &c);
    insta::assert_snapshot!("chained_field_mutations@raw", raw);
    insta::assert_snapshot!("chained_field_mutations@optimized", opt);
}

// =======================================================================
//  8. Object destructure + multi-branch classification
//     - SROA: @user.name, @user.age field reads
//     - SSA: category vars from each branch -> sequential, no phi (each branch independent)
//     - String concat chain
// =======================================================================

#[test]
fn destructure_multi_branch_classify() {
    let i = Interner::new();
    let c = ctx(
        &i,
        &[
            ("user", obj(&i, &[("name", Ty::String), ("age", Ty::I64)])),
            ("output", Ty::String),
        ],
    );
    let src = r#"
        let age = @user.age;
        @output = "unknown".to_string();
        if let true = age >= 65 { @output = "senior".to_string(); };
        if let true = age >= 18 { @output = "adult".to_string(); };
        if let true = age < 18 { @output = "minor".to_string(); };
        @output = @user.name + " (" + @output + ")";
        @output
    "#;
    let (raw, opt) = snap_both(&i, src, &c);
    insta::assert_snapshot!("destructure_multi_branch_classify@raw", raw);
    insta::assert_snapshot!("destructure_multi_branch_classify@optimized", opt);
}

// =======================================================================
//  9. Iteration with stateful accumulation + conditional side-effects
//     - SROA: x.amount, x.id field reads on loop variable
//     - SSA: @balance, @overdraft_count, @last_overdraft - loop phi + branch phi
//     - Most complex phi pattern: loop x branch x multiple contexts
// =======================================================================

// =======================================================================
// 10. Pure computation with loop-invariant hoisting
//     - SROA: @config.base_rate, @config.multiplier field reads
//     - CodeMotion: `factor` computation is loop-invariant -> hoist
//     - SSA: @result loop phi
// =======================================================================

// =======================================================================
// 11. A small pure closure called where it was made (RFC-0060)
//     - Inline: the call becomes the add
//     - Fold: the add becomes the constant, and nothing is left of the closure
// =======================================================================

#[test]
fn closure_called_where_it_was_made() {
    let i = Interner::new();
    let c = ctx(&i, &[]);
    let src = "let f = |x| -> x + 1; f(5)";
    let (raw, opt) = snap_both(&i, src, &c);
    insta::assert_snapshot!("closure_called_where_it_was_made@raw", raw);
    insta::assert_snapshot!("closure_called_where_it_was_made@optimized", opt);
}

#[test]
fn closure_called_in_a_while_body() {
    let i = Interner::new();
    let c = ctx(&i, &[("n", Ty::I64)]);
    let src = "let step = |x| -> x + 1; let i = 0; while i < @n { i = step(i); } i";
    let (raw, opt) = snap_both(&i, src, &c);
    insta::assert_snapshot!("closure_called_in_a_while_body@raw", raw);
    insta::assert_snapshot!("closure_called_in_a_while_body@optimized", opt);
}

#[test]
fn closure_that_captures_a_closure() {
    let i = Interner::new();
    let c = ctx(&i, &[]);
    let src = "let f = |x| -> x + 1.0; let g = |t| -> f(1.0); g(0)";
    let (raw, opt) = snap_both(&i, src, &c);
    insta::assert_snapshot!("closure_that_captures_a_closure@raw", raw);
    insta::assert_snapshot!("closure_that_captures_a_closure@optimized", opt);
}
