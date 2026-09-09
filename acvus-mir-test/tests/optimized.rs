//! Full optimization pipeline E2E tests.
//!
//! Each test compiles a complex, real-world script through the **full** pipeline:
//! extract → infer → lower → SROA → SSA → Inline → SpawnSplit → CodeMotion → Reorder → SSA → RegColor → Validate
//!
//! Two snapshots per test:
//! - `{name}@raw` — unoptimized, raw lowered MIR
//! - `{name}@optimized` — after full optimization pipeline

use acvus_mir::ty::Ty;
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
    Ty::Object(
        fields
            .iter()
            .map(|(name, ty)| (i.intern(name), ty.clone()))
            .collect(),
    )
}

fn snap_both(i: &Interner, source: &str, c: &FxHashMap<acvus_utils::Astr, Ty>) -> (String, String) {
    let raw = compile_script_raw(i, source, c).unwrap();
    let opt = compile_script_optimized(i, source, c).unwrap();
    (raw, opt)
}

// =======================================================================
//  1. Nested loop with conditional accumulator
//     - SSA: loop phi × 2 (pos_sum, neg_sum), branch phi within inner loop
//     - Reorder: context store ordering
// =======================================================================

#[ignore = "pending identity integration"]
#[test]
fn nested_loop_conditional_accum() {
    let i = Interner::new();
    let c = ctx(
        &i,
        &[
            ("matrix", Ty::Array(Box::new(Ty::Array(Box::new(Ty::Int), acvus_mir::ty::LenTerm::Known(3))), acvus_mir::ty::LenTerm::Known(3))),
            ("pos_sum", Ty::Int),
            ("neg_sum", Ty::Int),
        ],
    );
    let src = r#"
        row in @matrix {
            y in row {
                true = y > 0 { @pos_sum = @pos_sum + y; };
                true = y < 0 { @neg_sum = @neg_sum + y; };
            };
        };
        { pos: @pos_sum, neg: @neg_sum, }
    "#;
    let (raw, opt) = snap_both(&i, src, &c);
    insta::assert_snapshot!("nested_loop_conditional_accum@raw", raw);
    insta::assert_snapshot!("nested_loop_conditional_accum@optimized", opt);
}

// =======================================================================
//  2. Object field read-modify-write across branches
//     - SROA: multiple field projections on same context
//     - SSA: branch phi on context after conditional write
// =======================================================================

#[ignore = "pending identity integration"]
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
                    ("count", Ty::Int),
                    ("threshold", Ty::Int),
                    ("exceeded", Ty::Bool),
                ],
            ),
        )],
    );
    let src = r#"
        count = @stats.count + 1;
        true = count > @stats.threshold {
            @stats.count = count;
            @stats.exceeded = true;
        };
        @stats
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

#[ignore = "pending identity integration"]
#[test]
fn multi_context_dataflow() {
    let i = Interner::new();
    let c = ctx(
        &i,
        &[
            ("a", Ty::Int),
            ("b", Ty::Int),
            ("scale", Ty::Int),
            ("output", Ty::Int),
        ],
    );
    let src = r#"
        sum = @a + @b;
        scaled = sum * @scale;
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

#[ignore = "pending identity integration"]
#[test]
fn object_construct_from_fields() {
    let i = Interner::new();
    let c = ctx(
        &i,
        &[
            (
                "user",
                obj(&i, &[("name", Ty::String), ("age", Ty::Int)]),
            ),
            ("min_score", Ty::Int),
            ("output", obj(&i, &[("label", Ty::String), ("score", Ty::Int), ("eligible", Ty::Bool)])),
        ],
    );
    let src = r#"
        score = @user.age * 2;
        label = @user.name + " (score: " + to_string(score) + ")";
        eligible = score > @min_score;
        @output = { label: label, score: score, eligible: eligible, };
        @output
    "#;
    let (raw, opt) = snap_both(&i, src, &c);
    insta::assert_snapshot!("object_construct_from_fields@raw", raw);
    insta::assert_snapshot!("object_construct_from_fields@optimized", opt);
}

// =======================================================================
//  5. Diamond control flow with divergent context mutations
//     - SSA: @high, @low writes in separate branches → phi at join
//     - Multiple contexts mutated conditionally
// =======================================================================

#[ignore = "pending identity integration"]
#[test]
fn diamond_divergent_context_mutations() {
    let i = Interner::new();
    let c = ctx(
        &i,
        &[
            ("input", Ty::Int),
            ("high", Ty::Int),
            ("low", Ty::Int),
            ("output", Ty::Int),
        ],
    );
    let src = r#"
        x = @input;
        true = x > 100 {
            @high = @high + 1;
            @output = x * 2;
        };
        true = x <= 100 {
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
//     - Complex phi nesting: loop × branch
// =======================================================================

#[ignore = "pending identity integration"]
#[test]
fn loop_search_with_accumulator() {
    let i = Interner::new();
    let c = ctx(
        &i,
        &[
            ("items", Ty::Array(Box::new(Ty::Int), acvus_mir::ty::LenTerm::Known(3))),
            ("target", Ty::Int),
            ("found", Ty::Bool),
            ("idx", Ty::Int),
        ],
    );
    let src = r#"
        x in @items {
            true = x == @target {
                @found = true;
            };
            @idx = @idx + 1;
        };
        { found: @found, index: @idx, }
    "#;
    let (raw, opt) = snap_both(&i, src, &c);
    insta::assert_snapshot!("loop_search_with_accumulator@raw", raw);
    insta::assert_snapshot!("loop_search_with_accumulator@optimized", opt);
}

// =======================================================================
//  7. Chained field mutations on same object
//     - SROA: 4 field projections on @state → decompose each
//     - SSA: sequential writes, no phi but many SROA temporaries
//     - RegColor: high register pressure from SROA expansion
// =======================================================================

#[ignore = "pending identity integration"]
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
                    ("step", Ty::Int),
                    ("value", Ty::Int),
                    ("multiplier", Ty::Int),
                    ("done", Ty::Bool),
                    ("max_steps", Ty::Int),
                ],
            ),
        )],
    );
    let src = r#"
        @state.step = @state.step + 1;
        @state.value = @state.value * @state.multiplier;
        @state.done = @state.step >= @state.max_steps;
        @state
    "#;
    let (raw, opt) = snap_both(&i, src, &c);
    insta::assert_snapshot!("chained_field_mutations@raw", raw);
    insta::assert_snapshot!("chained_field_mutations@optimized", opt);
}

// =======================================================================
//  8. Object destructure + multi-branch classification
//     - SROA: @user.name, @user.age field reads
//     - SSA: category vars from each branch → sequential, no phi (each branch independent)
//     - String concat chain
// =======================================================================

#[ignore = "pending identity integration"]
#[test]
fn destructure_multi_branch_classify() {
    let i = Interner::new();
    let c = ctx(
        &i,
        &[
            (
                "user",
                obj(&i, &[("name", Ty::String), ("age", Ty::Int)]),
            ),
            ("output", Ty::String),
        ],
    );
    let src = r#"
        age = @user.age;
        @output = "unknown";
        true = age >= 65 { @output = "senior"; };
        true = age >= 18 { @output = "adult"; };
        true = age < 18 { @output = "minor"; };
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
//     - SSA: @balance, @overdraft_count, @last_overdraft — loop phi + branch phi
//     - Most complex phi pattern: loop × branch × multiple contexts
// =======================================================================

#[ignore = "pending identity integration"]
#[test]
fn iter_stateful_accum_with_side_effects() {
    let i = Interner::new();
    let tx_ty = obj(&i, &[("amount", Ty::Int), ("id", Ty::String)]);
    let c = ctx(
        &i,
        &[
            ("transactions", Ty::Array(Box::new(tx_ty), acvus_mir::ty::LenTerm::Known(3))),
            ("balance", Ty::Int),
            ("overdraft_count", Ty::Int),
            ("last_overdraft", Ty::String),
        ],
    );
    let src = r#"
        x in @transactions {
            @balance = @balance + x.amount;
            true = @balance < 0 {
                @overdraft_count = @overdraft_count + 1;
                @last_overdraft = x.id;
            };
        };
        { balance: @balance, overdrafts: @overdraft_count, last: @last_overdraft, }
    "#;
    let (raw, opt) = snap_both(&i, src, &c);
    insta::assert_snapshot!("iter_stateful_accum_with_side_effects@raw", raw);
    insta::assert_snapshot!("iter_stateful_accum_with_side_effects@optimized", opt);
}

// =======================================================================
// 10. Pure computation with loop-invariant hoisting
//     - SROA: @config.base_rate, @config.multiplier field reads
//     - CodeMotion: `factor` computation is loop-invariant → hoist
//     - SSA: @result loop phi
// =======================================================================

#[ignore = "pending identity integration"]
#[test]
fn loop_invariant_hoisting() {
    let i = Interner::new();
    let c = ctx(
        &i,
        &[
            (
                "config",
                obj(&i, &[("base_rate", Ty::Int), ("multiplier", Ty::Int)]),
            ),
            ("items", Ty::Array(Box::new(Ty::Int), acvus_mir::ty::LenTerm::Known(3))),
            ("result", Ty::Int),
        ],
    );
    let src = r#"
        factor = @config.base_rate * @config.multiplier;
        x in @items {
            @result = @result + x * factor;
        };
        @result
    "#;
    let (raw, opt) = snap_both(&i, src, &c);
    insta::assert_snapshot!("loop_invariant_hoisting@raw", raw);
    insta::assert_snapshot!("loop_invariant_hoisting@optimized", opt);
}
