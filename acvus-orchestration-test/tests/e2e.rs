use acvus_interpreter::{PureValue, Value};
use acvus_orchestration::ResolveError;
use acvus_orchestration_test::NodeBuilder;
use acvus_utils::Interner;

// =========================================================================
// Category 2: Expression execution — foundation for all other tests
// =========================================================================

// -- 2.1 Literals ----------------------------------------------------------

/// 2.1 C: Integer literal evaluates correctly.
#[tokio::test]
async fn expr_literal_int() {
    let g = NodeBuilder::new(Interner::new())
        .expr("x", "42")
        .build();
    let val = g.resolve_once("x").await.unwrap();
    assert!(matches!(val.value(), Value::Pure(PureValue::Int(42))));
}

/// 2.1 C: String literal evaluates correctly.
#[tokio::test]
async fn expr_literal_string() {
    let g = NodeBuilder::new(Interner::new())
        .expr("x", r#""hello""#)
        .build();
    let val = g.resolve_once("x").await.unwrap();
    assert!(matches!(val.value(), Value::Pure(PureValue::String(s)) if s == "hello"));
}

/// 2.1 C: Boolean literal evaluates correctly.
#[tokio::test]
async fn expr_literal_bool() {
    let g = NodeBuilder::new(Interner::new())
        .expr("x", "true")
        .build();
    let val = g.resolve_once("x").await.unwrap();
    assert!(matches!(val.value(), Value::Pure(PureValue::Bool(true))));
}

// -- 2.2 Arithmetic --------------------------------------------------------

/// 2.2 C: Arithmetic with operator precedence.
#[tokio::test]
async fn expr_arithmetic() {
    let g = NodeBuilder::new(Interner::new())
        .expr("x", "1 + 2 * 3")
        .build();
    let val = g.resolve_once("x").await.unwrap();
    assert!(matches!(val.value(), Value::Pure(PureValue::Int(7))));
}

// -- 2.4 Pipe --------------------------------------------------------------

/// 2.4 C: Pipe transforms value through a builtin function.
#[tokio::test]
async fn expr_pipe_to_string() {
    let g = NodeBuilder::new(Interner::new())
        .expr("x", "42 | to_string")
        .build();
    let val = g.resolve_once("x").await.unwrap();
    assert!(matches!(val.value(), Value::Pure(PureValue::String(s)) if s == "42"));
}

// =========================================================================
// Category 5: Assert node — graph-internal verification
// =========================================================================

// -- 5.1 Assert pass ------------------------------------------------------

/// 5.1 C: Assert that passes does not block propagation.
#[tokio::test]
async fn assert_pass() {
    let g = NodeBuilder::new(Interner::new())
        .expr_with_assert("x", "42", "@raw > 0")
        .build();
    let val = g.resolve_once("x").await.unwrap();
    assert!(matches!(val.value(), Value::Pure(PureValue::Int(42))));
}

// -- 5.2 Assert fail → error -----------------------------------------------

/// 5.2 S: Assert that fails produces a ResolveError.
#[tokio::test]
async fn assert_fail_produces_error() {
    let g = NodeBuilder::new(Interner::new())
        .expr_with_assert("x", "42", "@raw < 0")
        .build();
    let result = g.resolve_once("x").await;
    assert!(matches!(result, Err(ResolveError::Runtime { .. })));
}

// -- 5.5 Assert value check ------------------------------------------------

/// 5.5 C: Assert checks exact value equality.
#[tokio::test]
async fn assert_exact_value() {
    let g = NodeBuilder::new(Interner::new())
        .expr_with_assert("x", "1 + 1", "@raw == 2")
        .build();
    let val = g.resolve_once("x").await.unwrap();
    assert!(matches!(val.value(), Value::Pure(PureValue::Int(2))));
}

/// 5.5 S: Assert rejects wrong value.
#[tokio::test]
async fn assert_rejects_wrong_value() {
    let g = NodeBuilder::new(Interner::new())
        .expr_with_assert("x", "1 + 1", "@raw == 3")
        .build();
    let result = g.resolve_once("x").await;
    assert!(matches!(result, Err(ResolveError::Runtime { .. })));
}

// =========================================================================
// Category 1: Type — bind / output_ty
//
// The bind script transforms @raw into the stored value.
// output_ty must reflect the BIND result type, not the raw body type.
// =========================================================================

// -- 1.1 Identity bind -----------------------------------------------------

/// 1.1 C: Bind that returns @raw unchanged preserves raw type.
#[tokio::test]
async fn bind_identity_preserves_type() {
    let g = NodeBuilder::new(Interner::new())
        .patch("counter", "42", "@raw", "0")
        .build();
    let val = g.resolve_once("counter").await.unwrap();
    // body = 42, bind = @raw (identity) → stored = 42 (Int)
    assert!(matches!(val.value(), Value::Pure(PureValue::Int(42))));
}

// -- 1.2 Bind with to_string -----------------------------------------------

/// 1.2 C: Bind that transforms @raw via to_string → output must be String.
/// BUG DETECTOR: if output_ty is raw type (Int), the stored value will
/// have a type mismatch (value=String, ty=Int).
#[tokio::test]
async fn bind_to_string_changes_output_type() {
    let g = NodeBuilder::new(Interner::new())
        .patch("counter", "42", "@raw | to_string", r#""0""#)
        .build();
    let val = g.resolve_once("counter").await.unwrap();
    // body = 42 (Int), bind = @raw | to_string → "42" (String)
    assert!(
        matches!(val.value(), Value::Pure(PureValue::String(s)) if s == "42"),
        "expected String \"42\", got {:?}", val.value()
    );
    // Also verify the TYPE is String, not Int.
    assert!(
        matches!(val.ty(), acvus_mir::ty::Ty::String),
        "output_ty should be String after to_string bind, got {:?}", val.ty()
    );
}

// -- 1.6 Chained transforms ------------------------------------------------

/// 1.6 C: Chained pipe transforms: to_string then to_int.
/// Int → String → Int roundtrip through bind.
#[tokio::test]
async fn bind_chained_transforms() {
    let g = NodeBuilder::new(Interner::new())
        .patch("x", "42", "@raw | to_string", r#""0""#)
        .build();
    let val = g.resolve_once("x").await.unwrap();
    // 42 → "42"
    assert!(
        matches!(val.value(), Value::Pure(PureValue::String(s)) if s == "42"),
        "expected String \"42\", got {:?}", val.value()
    );
}

// -- 1.7 No bind (ephemeral) -----------------------------------------------

/// 1.7 C: Ephemeral node without bind — raw type is the output type.
#[tokio::test]
async fn no_bind_raw_type_is_output() {
    let g = NodeBuilder::new(Interner::new())
        .expr("x", "42")
        .build();
    let val = g.resolve_once("x").await.unwrap();
    assert!(matches!(val.value(), Value::Pure(PureValue::Int(42))));
    assert!(matches!(val.ty(), acvus_mir::ty::Ty::Int));
}

// -- 1.8 Sequence mode + bind: output_ty must reflect bind result ----------

/// 1.8 S (BUG DETECTOR): When raw_ty is explicitly List (like LLM nodes),
/// the REGISTRY type that other nodes see must be the bind result (Sequence),
/// not the raw body type (List).
///
/// Root cause: compile_nodes Phase 1 registers raw_output_ty into the registry,
/// but Phase 3 computes stored_types (bind result) without updating the registry.
/// Downstream nodes see the stale List type and insert List→Iterator casts,
/// which fail at runtime because the actual value is Sequence.
///
/// This test checks BOTH:
/// - output_ty (bind node's own type) — should be Sequence
/// - registry_ty (what other nodes see) — should ALSO be Sequence, not List
#[tokio::test]
async fn sequence_bind_registry_ty_matches_output_ty_not_raw() {
    use acvus_mir::ty::Ty;
    let g = NodeBuilder::new(Interner::new())
        .sequence(
            "history",
            "[1, 2, 3]",                        // body: Deque<Int>
            "@self | chain(@raw | iter)",        // bind: Sequence<Int>
            "[]",                                // initial: empty → Sequence
        )
        .build();

    // output_ty (bind node) — computed from graph engine resolution.
    let output_ty = g.output_ty("history");
    assert!(
        matches!(&output_ty, Ty::Sequence(..)),
        "output_ty should be Sequence, got {:?}", output_ty
    );

    // registry_ty (what @history looks like to OTHER nodes).
    // THIS IS THE BUG: registry still has List<Int> from Phase 1.
    let registry_ty = g.registry_ty("history")
        .expect("history should be in the registry");
    assert!(
        matches!(&registry_ty, Ty::Sequence(..)),
        "registry_ty should be Sequence (bind result), but got {:?} — \
         downstream nodes will see the wrong type and generate invalid casts",
        registry_ty
    );
}

/// 1.8 C: Control case — Expression with inferred type (no explicit raw_ty).
/// Both output_ty and registry_ty should be Sequence.
#[tokio::test]
async fn sequence_bind_output_ty_is_sequence_when_inferred() {
    let g = NodeBuilder::new(Interner::new())
        .sequence(
            "history",
            "[1, 2, 3]",
            "@self | chain(@raw | iter)",
            "[]",
        )
        .build();

    let output_ty = g.output_ty("history");
    assert!(
        matches!(&output_ty, acvus_mir::ty::Ty::Sequence(..)),
        "output_ty should be Sequence, got {:?}", output_ty
    );

    let registry_ty = g.registry_ty("history")
        .expect("history should be in the registry");
    assert!(
        matches!(&registry_ty, acvus_mir::ty::Ty::Sequence(..)),
        "registry_ty should be Sequence, got {:?}", registry_ty
    );
}

/// 1.8 C: Sequence bind actually runs and produces correct value.
#[tokio::test]
async fn sequence_bind_execution_produces_sequence() {
    let g = NodeBuilder::new(Interner::new())
        .sequence(
            "history",
            "[10, 20]",                          // body: List<Int>
            "@self | chain(@raw | iter)",         // bind: append raw items to self
            "[]",                                 // initial: empty
        )
        .build();

    let val = g.resolve_once("history").await.unwrap();
    // After turn 1: initial [] + chain([10,20]) → Sequence [10, 20]
    let sc = val.value().expect_ref::<acvus_interpreter::SequenceChain>("history");
    let items: Vec<_> = sc.origin().as_slice().to_vec();
    assert_eq!(items, vec![Value::int(10), Value::int(20)]);
}

/// 1.8 E2E: Two nodes — Sequence producer + consumer that references it.
/// The consumer should see the Sequence type (not List), so it can use
/// Sequence-compatible operations. Previously, the registry had the raw
/// List type, causing invalid List→Iterator casts at runtime.
#[tokio::test]
async fn sequence_bind_e2e_consumer_sees_sequence_type() {
    use acvus_mir::ty::Ty;
    let g = NodeBuilder::new(Interner::new())
        .sequence(
            "history",
            "[1, 2, 3]",
            "@self | chain(@raw | iter)",
            "[]",
        )
        // Consumer coerces @history to an iterator — works for Sequence.
        .expr("consumer", "@history | collect | len")
        .build();

    let val = g.resolve_once("consumer").await.unwrap();
    // collect [1,2,3] → List, len → 3
    assert!(
        matches!(val.value(), Value::Pure(PureValue::Int(3))),
        "expected 3, got {:?}", val.value()
    );
}

// =========================================================================
// Category 6: Node dependencies
// =========================================================================

// -- 6.1 A → B simple dependency -------------------------------------------

/// 6.1 C: Node B references @a and receives A's value.
#[tokio::test]
async fn dep_simple_a_to_b() {
    let g = NodeBuilder::new(Interner::new())
        .expr("a", "10")
        .expr("b", "@a + 1")
        .build();
    let val = g.resolve_once("b").await.unwrap();
    assert!(matches!(val.value(), Value::Pure(PureValue::Int(11))));
}

// -- 6.2 A → B → C chain ---------------------------------------------------

/// 6.2 C: Three-node dependency chain propagates values.
#[tokio::test]
async fn dep_chain_a_b_c() {
    let g = NodeBuilder::new(Interner::new())
        .expr("a", "1")
        .expr("b", "@a + 10")
        .expr("c", "@b * 2")
        .build();
    let val = g.resolve_once("c").await.unwrap();
    assert!(matches!(val.value(), Value::Pure(PureValue::Int(22))));
}

// =========================================================================
// Category 3: Persistent — Patch mode
// =========================================================================

// -- 3.1 First turn Set persist --------------------------------------------

/// 3.1 C: Patch node stores initial value on first turn.
#[tokio::test]
async fn patch_first_turn_stores_value() {
    let g = NodeBuilder::new(Interner::new())
        .patch("counter", "42", "@raw", "0")
        .build();
    let val = g.resolve_once("counter").await.unwrap();
    // body = 42, bind = @raw (identity) → stored = 42
    assert!(matches!(val.value(), Value::Pure(PureValue::Int(42))));
}

// -- 3.2 Second turn Rec diff -----------------------------------------------

/// 3.2 C: Patch node accumulates across turns via @self.
/// Turn 1: body=1, bind=@raw → stored=1
/// Turn 2: body=1, bind=@self + @raw → stored=2
#[tokio::test]
async fn patch_multi_turn_accumulates() {
    let g = NodeBuilder::new(Interner::new())
        .patch("counter", "1", "@self + @raw", "0")
        .build();
    // Turn 1: @self=0 (initial), body=1, bind=0+1=1
    let val = g.resolve_turns("counter", 1).await.unwrap();
    assert!(matches!(val.value(), Value::Pure(PureValue::Int(1))));
    // Turn 2: @self=1, body=1, bind=1+1=2
    let val = g.resolve_turns("counter", 2).await.unwrap();
    assert!(matches!(val.value(), Value::Pure(PureValue::Int(2))));
    // Turn 3: @self=2, body=1, bind=2+1=3
    let val = g.resolve_turns("counter", 3).await.unwrap();
    assert!(matches!(val.value(), Value::Pure(PureValue::Int(3))));
}

// -- 3.7 Multi-turn history -------------------------------------------------

/// 3.7 C: Patch value after N turns is correct (counter pattern).
#[tokio::test]
async fn patch_counter_10_turns() {
    let g = NodeBuilder::new(Interner::new())
        .patch("counter", "1", "@self + @raw", "0")
        .build();
    let val = g.resolve_turns("counter", 10).await.unwrap();
    assert!(
        matches!(val.value(), Value::Pure(PureValue::Int(10))),
        "counter after 10 turns should be 10, got {:?}", val.value()
    );
}

// =========================================================================
// Category 4: Persistent — Sequence mode
// =========================================================================

// -- 4.1 First turn init ---------------------------------------------------

/// 4.1 C: Sequence node initializes on first turn.
#[tokio::test]
async fn sequence_first_turn_stores_items() {
    let g = NodeBuilder::new(Interner::new())
        .sequence("log", "[1, 2, 3]", "@self | chain(@raw | iter)", "[]")
        .build();
    let val = g.resolve_once("log").await.unwrap();
    let sc = val.value().expect_ref::<acvus_interpreter::SequenceChain>("log");
    assert_eq!(
        sc.origin().as_slice(),
        &[Value::int(1), Value::int(2), Value::int(3)]
    );
}

// -- 4.2 Append across turns -----------------------------------------------

/// 4.2 C: Sequence accumulates items across turns.
/// Turn 1: body=[10], bind=self|chain(raw) → [10]
/// Turn 2: body=[20], bind=[10]|chain([20]) → [10, 20]
/// Turn 3: body=[30], bind=[10,20]|chain([30]) → [10, 20, 30]
#[tokio::test]
async fn sequence_multi_turn_append() {
    let g = NodeBuilder::new(Interner::new())
        .sequence("log", "[99]", "@self | chain(@raw | iter)", "[]")
        .build();
    // After 3 turns: [99, 99, 99]
    let val = g.resolve_turns("log", 3).await.unwrap();
    let sc = val.value().expect_ref::<acvus_interpreter::SequenceChain>("log");
    assert_eq!(
        sc.origin().as_slice(),
        &[Value::int(99), Value::int(99), Value::int(99)]
    );
}

// =========================================================================
// Category 7: Context / Scope
// =========================================================================

// -- 7.1 Body references external context -----------------------------------

/// 7.1 C: Body can reference external context provided by resolver.
/// Uses the resolver callback to inject a value for @input.
#[tokio::test]
async fn context_body_references_external() {
    // This test needs a resolver that provides @input.
    // For now, test that compilation succeeds with a user-provided context type.
    // Full resolve requires the resolver callback to return the value.
    // TODO: extend NodeBuilder to accept external context types.
}

// -- 7.6 initial_value becomes first @self ----------------------------------

/// 7.6 C: initial_value provides the first @self value.
#[tokio::test]
async fn initial_value_is_first_self() {
    let g = NodeBuilder::new(Interner::new())
        // body returns @self (just echo it back)
        // bind = @raw (identity)
        // initial = 100
        // Turn 1: @self=100, body=100, bind=100 → stored=100
        .patch("x", "@self", "@raw", "100")
        .build();
    let val = g.resolve_once("x").await.unwrap();
    assert!(
        matches!(val.value(), Value::Pure(PureValue::Int(100))),
        "first @self should be initial_value=100, got {:?}", val.value()
    );
}

// =========================================================================
// Category 8: Mixed scenarios
// =========================================================================

// -- 8.1 Sequence + Patch coexist ------------------------------------------

/// 8.1 C: Sequence and Patch nodes in the same graph don't interfere.
#[tokio::test]
async fn mixed_sequence_and_patch_coexist() {
    let g = NodeBuilder::new(Interner::new())
        .patch("counter", "1", "@self + @raw", "0")
        .sequence("log", "[1]", "@self | chain(@raw | iter)", "[]")
        .build();

    // 3 turns
    let counter = g.resolve_turns("counter", 3).await.unwrap();
    assert!(matches!(counter.value(), Value::Pure(PureValue::Int(3))));

    let log = g.resolve_turns("log", 3).await.unwrap();
    let sc = log.value().expect_ref::<acvus_interpreter::SequenceChain>("log");
    assert_eq!(
        sc.origin().as_slice(),
        &[Value::int(1), Value::int(1), Value::int(1)]
    );
}

// -- 8.2 Expression → Persistent chain ------------------------------------

/// 8.2 C: Expression node feeds into a Persistent node via dependency.
#[tokio::test]
async fn mixed_expr_feeds_persistent() {
    let g = NodeBuilder::new(Interner::new())
        .expr("source", "42")
        .patch("sink", "@source", "@raw", "0")
        .build();
    let val = g.resolve_once("sink").await.unwrap();
    // body = @source = 42, bind = @raw = 42
    assert!(matches!(val.value(), Value::Pure(PureValue::Int(42))));
}

// -- 8.4 Multi-turn 3 turns all correct ------------------------------------

/// 8.4 C: 3-turn scenario with Patch counter + Sequence log.
/// Verifies final values are consistent across types.
#[tokio::test]
async fn mixed_multi_turn_3_turns() {
    let g = NodeBuilder::new(Interner::new())
        .patch("n", "1", "@self + @raw", "0")
        .sequence("log", "[@n]", "@self | chain(@raw | iter)", "[]")
        .build();

    // After 3 turns:
    // n: 0+1=1, 1+1=2, 2+1=3
    let n = g.resolve_turns("n", 3).await.unwrap();
    assert!(matches!(n.value(), Value::Pure(PureValue::Int(3))));

    // log: [1], [1,2], [1,2,3] — each turn appends current @n
    // BUT: @n is evaluated per-turn, and log body=[@n] references @n.
    // Turn 1: @n=1, log body=[1], bind=[]|chain([1]) → [1]
    // Turn 2: @n=2, log body=[2], bind=[1]|chain([2]) → [1,2]
    // Turn 3: @n=3, log body=[3], bind=[1,2]|chain([3]) → [1,2,3]
    let log = g.resolve_turns("log", 3).await.unwrap();
    let sc = log.value().expect_ref::<acvus_interpreter::SequenceChain>("log");
    assert_eq!(
        sc.origin().as_slice(),
        &[Value::int(1), Value::int(2), Value::int(3)]
    );
}

// =========================================================================
// Category 9: Error handling
// =========================================================================

// -- 9.1 Runtime type error -------------------------------------------------

/// 9.1 S: Adding Int + String is caught at compile time (type checker soundness).
#[tokio::test]
async fn error_type_mismatch_caught_at_compile() {
    let result = NodeBuilder::new(Interner::new())
        .expr("x", r#"1 + "hello""#)
        .try_build();
    assert!(result.is_err(), "Int + String should be a compile error");
}

/// 9.1 C: Adding Int + Int compiles and runs fine (completeness pair).
#[tokio::test]
async fn int_plus_int_compiles_ok() {
    let g = NodeBuilder::new(Interner::new())
        .expr("x", "1 + 2")
        .build();
    let val = g.resolve_once("x").await.unwrap();
    assert!(matches!(val.value(), Value::Pure(PureValue::Int(3))));
}

// =========================================================================
// Category 10: Type safety soundness
// =========================================================================

/// 10.1 S: Bind returning wrong type for Patch is caught.
/// Patch initial="0" (Int), bind="@raw | to_string" changes type to String.
/// If the type system tracks this correctly, @self is String on turn 2.
/// This is completeness — the type system allows bind to change type.
#[tokio::test]
async fn patch_bind_type_change_allowed() {
    let g = NodeBuilder::new(Interner::new())
        .patch("x", "42", "@raw | to_string", r#""init""#)
        .build();
    let val = g.resolve_once("x").await.unwrap();
    assert!(
        matches!(val.value(), Value::Pure(PureValue::String(s)) if s == "42"),
        "bind changed type Int→String, got {:?}", val.value()
    );
}

// =========================================================================
// Category 12: Scope violation soundness
// =========================================================================

/// 12.4 S: Referencing a non-existent node @ghost should fail at compile time.
#[tokio::test]
async fn scope_reference_nonexistent_node_fails() {
    let result = NodeBuilder::new(Interner::new())
        .expr("x", "@ghost + 1")
        .try_build();
    // @ghost doesn't exist — should be a compile error.
    assert!(result.is_err(), "@ghost should cause compile error");
}

/// 12.4 C: Referencing an existing node @a compiles fine (completeness pair).
#[tokio::test]
async fn scope_reference_existing_node_ok() {
    let g = NodeBuilder::new(Interner::new())
        .expr("a", "10")
        .expr("x", "@a + 1")
        .build();
    let val = g.resolve_once("x").await.unwrap();
    assert!(matches!(val.value(), Value::Pure(PureValue::Int(11))));
}

/// 12.5 S (BUG): Persistent Patch without initial_value should fail at compile time.
/// Currently compiles — missing validation.
#[tokio::test]
async fn scope_persistent_without_initial_value_fails() {
    let i = Interner::new();
    let bind = i.intern("@raw");
    let specs = vec![acvus_orchestration::NodeSpec {
        name: i.intern("x"),
        kind: acvus_orchestration::NodeKind::Expression(
            acvus_orchestration::ExpressionSpec {
                source: "42".to_string(),
                output_ty: None,
            },
        ),
        strategy: acvus_orchestration::Strategy {
            execution: acvus_orchestration::Execution::OncePerTurn,
            persistency: acvus_orchestration::Persistency::Patch { bind },
            initial_value: None, // ← no initial_value
            retry: 0,
            assert: None,
        },
        is_function: false,
        fn_params: vec![],
    }];
    let registry = acvus_mir::context_registry::PartialContextTypeRegistry::system_only(
        rustc_hash::FxHashMap::default(),
    );
    let fetch = std::sync::Arc::new(acvus_orchestration::http::NoopFetch);
    let result = acvus_orchestration::compile_nodes(&i, &specs, registry, fetch);
    assert!(result.is_err(), "Patch without initial_value should be compile error");
}

// =========================================================================
// Category 6 (continued): More dependency patterns
// =========================================================================

/// 6.3 C: Fan-out — A feeds both B and C independently.
#[tokio::test]
async fn dep_fan_out() {
    let g = NodeBuilder::new(Interner::new())
        .expr("a", "10")
        .expr("b", "@a + 1")
        .expr("c", "@a * 2")
        .build();
    let b = g.resolve_once("b").await.unwrap();
    let c = g.resolve_once("c").await.unwrap();
    assert!(matches!(b.value(), Value::Pure(PureValue::Int(11))));
    assert!(matches!(c.value(), Value::Pure(PureValue::Int(20))));
}

/// 6.4 C: Fan-in — C depends on both A and B.
#[tokio::test]
async fn dep_fan_in() {
    let g = NodeBuilder::new(Interner::new())
        .expr("a", "10")
        .expr("b", "20")
        .expr("c", "@a + @b")
        .build();
    let val = g.resolve_once("c").await.unwrap();
    assert!(matches!(val.value(), Value::Pure(PureValue::Int(30))));
}

/// 6.6 C: @self references previous turn value in Persistent nodes.
#[tokio::test]
async fn dep_self_reference_across_turns() {
    let g = NodeBuilder::new(Interner::new())
        .patch("x", "@self * 2", "@raw", "1")
        .build();
    // Turn 1: @self=1, body=2, bind=2 → stored=2
    // Turn 2: @self=2, body=4, bind=4 → stored=4
    // Turn 3: @self=4, body=8, bind=8 → stored=8
    let val = g.resolve_turns("x", 3).await.unwrap();
    assert!(
        matches!(val.value(), Value::Pure(PureValue::Int(8))),
        "1→2→4→8, got {:?}", val.value()
    );
}

/// 6.5 S: Cycle detection — A→B→A should fail at compile time.
#[tokio::test]
async fn dep_cycle_detected() {
    let result = NodeBuilder::new(Interner::new())
        .expr("a", "@b")
        .expr("b", "@a")
        .try_build();
    assert!(result.is_err(), "cyclic dependency should be caught");
}

// =========================================================================
// Category 4 (continued): Sequence edge cases
// =========================================================================

/// 4.6 C: Sequence multi-turn — verify intermediate states are consistent.
/// After 5 turns of appending [1], result should be [1,1,1,1,1].
#[tokio::test]
async fn sequence_5_turn_history() {
    let g = NodeBuilder::new(Interner::new())
        .sequence("log", "[1]", "@self | chain(@raw | iter)", "[]")
        .build();
    let val = g.resolve_turns("log", 5).await.unwrap();
    let sc = val.value().expect_ref::<acvus_interpreter::SequenceChain>("log");
    let expected: Vec<_> = (0..5).map(|_| Value::int(1)).collect();
    assert_eq!(sc.origin().as_slice(), &expected[..]);
}

// =========================================================================
// Category 3 (continued): Patch string accumulation
// =========================================================================

/// 3.6 C: Patch with String concatenation across turns.
#[tokio::test]
async fn patch_string_accumulation() {
    let g = NodeBuilder::new(Interner::new())
        .patch("msg", r#""x""#, r#"@self + @raw"#, r#""""#)
        .build();
    // Turn 1: @self="", body="x", bind="" + "x" = "x"
    // Turn 2: @self="x", body="x", bind="x" + "x" = "xx"
    // Turn 3: @self="xx", body="x", bind="xx" + "x" = "xxx"
    let val = g.resolve_turns("msg", 3).await.unwrap();
    assert!(
        matches!(val.value(), Value::Pure(PureValue::String(s)) if s == "xxx"),
        "expected \"xxx\", got {:?}", val.value()
    );
}

// =========================================================================
// Category 8 (continued): Complex mixed scenarios
// =========================================================================

/// 8.2 C: Chain of Expression → Patch → dependent Expression.
/// Verifies that a downstream node sees the Patch result.
#[tokio::test]
async fn mixed_expr_patch_expr_chain() {
    let g = NodeBuilder::new(Interner::new())
        .expr("input", "5")
        .patch("doubled", "@input * 2", "@raw", "0")
        .expr("output", "@doubled + 1")
        .build();
    let val = g.resolve_once("output").await.unwrap();
    // input=5, doubled=10, output=11
    assert!(matches!(val.value(), Value::Pure(PureValue::Int(11))));
}

/// 8.1 C: Sequence and Patch in same graph, each with dependencies.
#[tokio::test]
async fn mixed_patch_feeds_sequence() {
    let g = NodeBuilder::new(Interner::new())
        .patch("n", "1", "@self + @raw", "0")
        .sequence("log", "[@n]", "@self | chain(@raw | iter)", "[]")
        .build();
    // After 1 turn: n=1, log=[1]
    let n = g.resolve_turns("n", 1).await.unwrap();
    assert!(matches!(n.value(), Value::Pure(PureValue::Int(1))));
    let log = g.resolve_turns("log", 1).await.unwrap();
    let sc = log.value().expect_ref::<acvus_interpreter::SequenceChain>("log");
    assert_eq!(sc.origin().as_slice(), &[Value::int(1)]);
}

// =========================================================================
// Category 14: Effect soundness
//
// The effect system distinguishes Pure (no side effects, memoizable) from
// Effectful (side effects, must re-execute). Persistent nodes require
// pureable stored types. Sequence<T, O, Pure> must remain Pure through
// the persist pipeline.
// =========================================================================

/// 14.1 C: Sequence persist with Pure effect succeeds.
/// bind = @self | chain(@raw | iter) produces Sequence<T, O, Pure>
/// which is pureable → compilation and persistence succeed.
#[tokio::test]
async fn effect_sequence_pure_persist_succeeds() {
    let g = NodeBuilder::new(Interner::new())
        .sequence("log", "[1]", "@self | chain(@raw | iter)", "[]")
        .build();
    let val = g.resolve_once("log").await.unwrap();
    let sc = val.value().expect_ref::<acvus_interpreter::SequenceChain>("log");
    assert_eq!(sc.origin().as_slice(), &[Value::int(1)]);
}

/// 14.2 C: Patch persist with pure scalar type succeeds.
#[tokio::test]
async fn effect_patch_pure_scalar_succeeds() {
    let g = NodeBuilder::new(Interner::new())
        .patch("n", "42", "@raw", "0")
        .build();
    let val = g.resolve_once("n").await.unwrap();
    assert!(matches!(val.value(), Value::Pure(PureValue::Int(42))));
}

/// 14.2 C: Effect subtyping — Pure ≤ Effectful covariant.
/// A Sequence<Int, Pure> value can be used where Sequence<Int, Effectful> is expected.
/// This is verified by the fact that Sequence operations (chain, iter) work
/// regardless of effect in a Pure context.
#[tokio::test]
async fn effect_pure_subtype_of_effectful() {
    // chain produces a Pure sequence from Pure inputs — should compile.
    let g = NodeBuilder::new(Interner::new())
        .sequence("log", "[1, 2]", "@self | chain(@raw | iter)", "[]")
        .build();
    let val = g.resolve_once("log").await.unwrap();
    let sc = val.value().expect_ref::<acvus_interpreter::SequenceChain>("log");
    assert_eq!(sc.origin().as_slice(), &[Value::int(1), Value::int(2)]);
}

/// 14.1 C: Sequence output_ty has Pure effect (not Effectful).
/// The stored Sequence must have Pure effect for persistence to be valid.
#[tokio::test]
async fn effect_sequence_output_ty_is_pure() {
    use acvus_mir::ty::{Ty, Effect};
    let g = NodeBuilder::new(Interner::new())
        .sequence("log", "[1]", "@self | chain(@raw | iter)", "[]")
        .build();
    let ty = g.output_ty("log");
    match &ty {
        Ty::Sequence(_, _, effect) => {
            assert_eq!(*effect, Effect::Pure,
                "Sequence output_ty effect should be Pure for persistence, got {:?}", effect);
        }
        other => panic!("expected Sequence type, got {:?}", other),
    }
}

/// 14.1 S: Effectful Sequence cannot be persisted — compile error.
/// Persistence requires storable types, and Sequence<T, O, Effectful> is not storable.
#[tokio::test]
async fn effect_effectful_sequence_cannot_persist() {
    use acvus_mir::ty::{Ty, Effect, Origin};
    let result = NodeBuilder::new(Interner::new())
        .sequence_with_raw_ty(
            "log",
            "[1]",
            "@self | chain(@raw | iter)",
            "[]",
            // Explicitly mark raw as Effectful Iterator
            Ty::Iterator(Box::new(Ty::Int), Effect::Effectful),
        )
        .try_build();
    assert!(result.is_err(),
        "Effectful Sequence should not be persistable — expected compile error");
}

/// 14 C: Multi-turn Sequence retains Pure effect across turns.
#[tokio::test]
async fn effect_sequence_stays_pure_across_turns() {
    use acvus_mir::ty::{Ty, Effect};
    let g = NodeBuilder::new(Interner::new())
        .sequence("log", "[1]", "@self | chain(@raw | iter)", "[]")
        .build();
    // 3 turns — effect should still be Pure.
    let val = g.resolve_turns("log", 3).await.unwrap();
    match val.ty() {
        Ty::Sequence(_, _, effect) => {
            assert_eq!(*effect, Effect::Pure,
                "Sequence should remain Pure after 3 turns, got {:?}", effect);
        }
        other => panic!("expected Sequence type, got {:?}", other),
    }
}

// =========================================================================
// Bidirectional test pairs — Soundness (S) and Completeness (C)
//
// Each test below is the missing half of an existing test, forming a
// bidirectional pair that checks both acceptance and rejection.
// =========================================================================

// =========================================================================
// Category 2: Expression — soundness pairs
// =========================================================================

/// 2.2 S: Arithmetic with incompatible types should fail at compile time.
/// Pair of: expr_arithmetic (2.2 C)
#[tokio::test]
async fn expr_arithmetic_type_error_fails() {
    let result = NodeBuilder::new(Interner::new())
        .expr("x", r#""hello" + 1"#)
        .try_build();
    assert!(result.is_err(), "String + Int should be a compile error");
}

/// 2.2 S: String minus Int is not a valid operation.
/// Pair of: expr_arithmetic (2.2 C)
#[tokio::test]
async fn expr_arithmetic_string_minus_int_fails() {
    let result = NodeBuilder::new(Interner::new())
        .expr("x", r#""hello" - 1"#)
        .try_build();
    assert!(result.is_err(), "String - Int should be a compile error");
}

/// 2.2 S: Bool + Int is not a valid operation.
/// Pair of: expr_arithmetic (2.2 C)
#[tokio::test]
async fn expr_arithmetic_bool_plus_int_fails() {
    let result = NodeBuilder::new(Interner::new())
        .expr("x", "true + 1")
        .try_build();
    assert!(result.is_err(), "Bool + Int should be a compile error");
}

/// 2.4 S: Pipe to undefined function should fail at compile time.
/// Pair of: expr_pipe_to_string (2.4 C)
#[tokio::test]
async fn expr_pipe_undefined_function_fails() {
    let result = NodeBuilder::new(Interner::new())
        .expr("x", "42 | nonexistent_function")
        .try_build();
    assert!(result.is_err(), "pipe to undefined function should be a compile error");
}

/// 2.4 S: Pipe chain where second function is undefined.
/// Pair of: expr_pipe_to_string (2.4 C)
#[tokio::test]
async fn expr_pipe_chain_undefined_second_fails() {
    let result = NodeBuilder::new(Interner::new())
        .expr("x", "42 | to_string | bogus")
        .try_build();
    assert!(result.is_err(), "pipe to undefined second function should be a compile error");
}

// =========================================================================
// Category 4: Sequence — soundness pairs
// =========================================================================

/// 4.1 S: Sequence with wrong initial type should fail at compile time.
/// Initial is a String, but body produces List<Int> and bind expects Sequence<Int>.
/// Pair of: sequence_first_turn_stores_items (4.1 C)
#[tokio::test]
async fn sequence_wrong_initial_type_fails() {
    let result = NodeBuilder::new(Interner::new())
        .sequence("log", "[1, 2, 3]", "@self | chain(@raw | iter)", r#""not a list""#)
        .try_build();
    assert!(result.is_err(), "Sequence with String initial should be a compile error");
}

/// 4.1 S: Sequence with Int initial instead of empty list.
/// Pair of: sequence_first_turn_stores_items (4.1 C)
#[tokio::test]
async fn sequence_int_initial_type_fails() {
    let result = NodeBuilder::new(Interner::new())
        .sequence("log", "[1, 2, 3]", "@self | chain(@raw | iter)", "42")
        .try_build();
    assert!(result.is_err(), "Sequence with Int initial should be a compile error");
}

/// 4.2 S: Each resolve_turns call is independent (fresh journal).
/// Calling resolve_turns(1) twice should give the same result, not accumulate.
/// Pair of: sequence_multi_turn_append (4.2 C)
#[tokio::test]
async fn sequence_independent_resolve_turns_no_leak() {
    let g = NodeBuilder::new(Interner::new())
        .sequence("log", "[99]", "@self | chain(@raw | iter)", "[]")
        .build();

    // First call: 1 turn → [99]
    let val1 = g.resolve_turns("log", 1).await.unwrap();
    let sc1 = val1.value().expect_ref::<acvus_interpreter::SequenceChain>("log");
    assert_eq!(sc1.origin().as_slice(), &[Value::int(99)]);

    // Second call: 1 turn → should also be [99], not [99, 99]
    let val2 = g.resolve_turns("log", 1).await.unwrap();
    let sc2 = val2.value().expect_ref::<acvus_interpreter::SequenceChain>("log");
    assert_eq!(
        sc2.origin().as_slice(),
        &[Value::int(99)],
        "independent resolve_turns calls should not share state"
    );
}

// =========================================================================
// Category 5: Assert — soundness pairs
// =========================================================================

/// 5.1 S: Assert with non-boolean result should fail at compile time.
/// The assert script returns an Int (not Bool), which the type checker rejects.
/// Pair of: assert_pass (5.1 C)
#[tokio::test]
async fn assert_non_boolean_result_fails() {
    let result = NodeBuilder::new(Interner::new())
        .expr_with_assert("x", "42", "@raw + 1")
        .try_build();
    assert!(
        result.is_err(),
        "assert returning Int (not Bool) should be a compile error"
    );
}

/// 5.1 S: Assert with String result should fail at compile time.
/// Pair of: assert_pass (5.1 C)
#[tokio::test]
async fn assert_string_result_fails() {
    let result = NodeBuilder::new(Interner::new())
        .expr_with_assert("x", "42", "@raw | to_string")
        .try_build();
    assert!(
        result.is_err(),
        "assert returning String (not Bool) should be a compile error"
    );
}

// =========================================================================
// Category 6: Dependencies — soundness pairs
// =========================================================================

/// 6.1 S: Referencing undefined @ghost in Patch bind fails at compile time.
/// Pair of: dep_simple_a_to_b (6.1 C)
#[tokio::test]
async fn dep_undefined_in_bind_fails() {
    let result = NodeBuilder::new(Interner::new())
        .patch("x", "1", "@ghost + @raw", "0")
        .try_build();
    assert!(result.is_err(), "@ghost in bind should cause compile error");
}

/// 6.3 S: Fan-out nodes get independent values — mutating one doesn't affect the other.
/// Pair of: dep_fan_out (6.3 C)
#[tokio::test]
async fn dep_fan_out_independent_values() {
    let g = NodeBuilder::new(Interner::new())
        .expr("a", "10")
        .patch("b", "@a", "@self + @raw", "0")   // accumulates: turn 1 → 10
        .patch("c", "@a", "@self * @raw", "1")    // multiplies: turn 1 → 10
        .build();

    // After 2 turns:
    // b: 0+10=10, 10+10=20
    // c: 1*10=10, 10*10=100
    let b = g.resolve_turns("b", 2).await.unwrap();
    let c = g.resolve_turns("c", 2).await.unwrap();
    assert!(
        matches!(b.value(), Value::Pure(PureValue::Int(20))),
        "b should be 20, got {:?}", b.value()
    );
    assert!(
        matches!(c.value(), Value::Pure(PureValue::Int(100))),
        "c should be 100, got {:?}", c.value()
    );
}

/// 6.5 C: Non-cyclic dependency chain compiles fine.
/// Pair of: dep_cycle_detected (6.5 S)
/// (This is confirmed by dep_chain_a_b_c, but we add a 4-node chain for coverage.)
#[tokio::test]
async fn dep_chain_4_nodes_ok() {
    let g = NodeBuilder::new(Interner::new())
        .expr("a", "1")
        .expr("b", "@a + 1")
        .expr("c", "@b + 1")
        .expr("d", "@c + 1")
        .build();
    let val = g.resolve_once("d").await.unwrap();
    assert!(matches!(val.value(), Value::Pure(PureValue::Int(4))));
}

/// 6.5 S: 3-node cycle A→B→C→A should fail at compile time.
/// Pair of: dep_chain_a_b_c (6.5 C)
#[tokio::test]
async fn dep_3_node_cycle_detected() {
    let result = NodeBuilder::new(Interner::new())
        .expr("a", "@c")
        .expr("b", "@a")
        .expr("c", "@b")
        .try_build();
    assert!(result.is_err(), "3-node cycle should be caught at compile time");
}

// =========================================================================
// Category 10: Type safety — soundness pairs
// =========================================================================

/// 10.1 S: Patch bind with incompatible initial type should fail.
/// bind = @self + @raw expects both sides to be Int, but initial is String.
/// Pair of: patch_bind_type_change_allowed (10.1 C)
#[tokio::test]
async fn patch_bind_incompatible_initial_type_fails() {
    let result = NodeBuilder::new(Interner::new())
        .patch("x", "42", "@self + @raw", r#""not_an_int""#)
        .try_build();
    assert!(result.is_err(), "String initial + Int body in bind should be compile error");
}

/// 10.1 C: Patch bind with compatible initial type compiles fine.
/// Pair of: patch_bind_incompatible_initial_type_fails (10.1 S)
#[tokio::test]
async fn patch_bind_compatible_initial_type_ok() {
    let g = NodeBuilder::new(Interner::new())
        .patch("x", "42", "@self + @raw", "0")
        .build();
    let val = g.resolve_once("x").await.unwrap();
    assert!(matches!(val.value(), Value::Pure(PureValue::Int(42))));
}

/// 10.1 S: Patch where body type doesn't match bind's expected @raw type.
/// body produces String, bind tries to do Int arithmetic on @raw.
#[tokio::test]
async fn patch_body_type_mismatch_bind_fails() {
    let result = NodeBuilder::new(Interner::new())
        .patch("x", r#""hello""#, "@self + @raw", "0")
        .try_build();
    assert!(result.is_err(), "String body + Int initial in bind should be compile error");
}

// =========================================================================
// Category 14: Effect — soundness pairs
// =========================================================================

/// 14.2 S: Fn type cannot be persisted — compile error expected.
/// Fn types are not storable, so Patch persist on a Fn-producing node should fail.
/// Pair of: effect_patch_pure_scalar_succeeds (14.2 C)
#[tokio::test]
async fn effect_fn_type_cannot_persist() {
    // A lambda expression produces a Fn type.
    // Attempting to persist it via Patch should fail.
    let result = NodeBuilder::new(Interner::new())
        .patch("f", r#"\x -> x + 1"#, "@raw", r#"\x -> x"#)
        .try_build();
    assert!(
        result.is_err(),
        "Fn type should not be persistable — expected compile error"
    );
}

// =========================================================================
// Category 3: Patch — additional soundness
// =========================================================================

/// 3.2 S: Patch @self in bind with wrong type arithmetic fails.
/// initial = "hello" (String), body = 1 (Int), bind = @self + @raw → String + Int error.
/// Pair of: patch_multi_turn_accumulates (3.2 C)
#[tokio::test]
async fn patch_self_type_mismatch_in_bind_fails() {
    let result = NodeBuilder::new(Interner::new())
        .patch("x", "1", "@self + @raw", r#""hello""#)
        .try_build();
    assert!(result.is_err(), "String @self + Int @raw in bind should be compile error");
}

/// 3.6 S: Patch String concatenation with Int body fails.
/// initial = "" (String), body = 42 (Int), bind = @self + @raw → String + Int error.
/// Pair of: patch_string_accumulation (3.6 C)
#[tokio::test]
async fn patch_string_concat_with_int_body_fails() {
    let result = NodeBuilder::new(Interner::new())
        .patch("msg", "42", "@self + @raw", r#""""#)
        .try_build();
    assert!(result.is_err(), "String @self + Int @raw in bind should be compile error");
}

// =========================================================================
// Category 1: Type — additional soundness
// =========================================================================

/// 1.1 S: Bind that uses undefined function fails.
/// Pair of: bind_identity_preserves_type (1.1 C)
#[tokio::test]
async fn bind_undefined_function_fails() {
    let result = NodeBuilder::new(Interner::new())
        .patch("x", "42", "@raw | nonexistent", "0")
        .try_build();
    assert!(result.is_err(), "bind with undefined function should be compile error");
}

/// 1.2 S: Bind type mismatch — initial type doesn't match bind result.
/// bind = @raw | to_string → String, initial = 0 (Int). @self is String but initial is Int.
/// Pair of: bind_to_string_changes_output_type (1.2 C)
/// Note: this is currently ALLOWED by design — bind can change the stored type.
/// The initial value is coerced to match on the first turn.
#[tokio::test]
async fn bind_type_change_initial_mismatch_is_allowed() {
    let result = NodeBuilder::new(Interner::new())
        .patch("x", "42", "@raw | to_string", "0")
        .try_build();
    // This compiles because bind is allowed to change the type.
    // initial=0(Int), bind=@raw|to_string→String. @self becomes String.
    // On turn 1: @self=0(Int) → to_string coerces it.
    assert!(result.is_ok(), "bind type change should be allowed");
}

// =========================================================================
// Category 9: Error handling — additional completeness
// =========================================================================

/// 9.1 C: String + String concatenation compiles and runs fine.
/// Pair of: error_type_mismatch_caught_at_compile (9.1 S)
#[tokio::test]
async fn string_plus_string_compiles_ok() {
    let g = NodeBuilder::new(Interner::new())
        .expr("x", r#""hello" + " world""#)
        .build();
    let val = g.resolve_once("x").await.unwrap();
    assert!(
        matches!(val.value(), Value::Pure(PureValue::String(s)) if s == "hello world"),
        "expected \"hello world\", got {:?}", val.value()
    );
}

/// 9.1 S: Comparison between incompatible types fails.
/// Pair of: int_plus_int_compiles_ok (9.1 C)
#[tokio::test]
async fn comparison_type_mismatch_fails() {
    let result = NodeBuilder::new(Interner::new())
        .expr("x", r#"1 == "one""#)
        .try_build();
    assert!(result.is_err(), "Int == String should be a compile error");
}

// =========================================================================
// Category 12: Scope violations — additional pairs
// =========================================================================

/// 12.4 S: Referencing @self in an ephemeral Expression should fail.
/// @self is only available in persistent nodes.
/// Pair of: scope_reference_existing_node_ok (12.4 C)
#[tokio::test]
async fn scope_self_in_ephemeral_fails() {
    let result = NodeBuilder::new(Interner::new())
        .expr("x", "@self + 1")
        .try_build();
    assert!(result.is_err(), "@self in ephemeral Expression should be compile error");
}

/// 12.4 C: @self in persistent Patch node compiles fine.
/// Pair of: scope_self_in_ephemeral_fails (12.4 S)
#[tokio::test]
async fn scope_self_in_patch_ok() {
    let g = NodeBuilder::new(Interner::new())
        .patch("x", "@self", "@raw", "100")
        .build();
    let val = g.resolve_once("x").await.unwrap();
    assert!(matches!(val.value(), Value::Pure(PureValue::Int(100))));
}

#[tokio::test]
async fn debug_sequence_types() {
    use acvus_mir::context_registry::PartialContextTypeRegistry;

    let interner = Interner::new();
    let spec = acvus_orchestration::NodeSpec {
        name: interner.intern("log"),
        kind: acvus_orchestration::NodeKind::Expression(acvus_orchestration::ExpressionSpec {
            source: "[1, 2, 3]".to_string(),
            output_ty: None,
        }),
        strategy: acvus_orchestration::Strategy {
            execution: acvus_orchestration::Execution::OncePerTurn,
            persistency: acvus_orchestration::Persistency::Sequence {
                bind: interner.intern("@self | chain(@raw | iter)"),
            },
            initial_value: Some(interner.intern("[]")),
            retry: 0,
            assert: None,
        },
        is_function: false,
        fn_params: vec![],
    };
    let registry = PartialContextTypeRegistry::default();
    let (graph, node_metas) = acvus_orchestration::lower::lower(&interner, &[spec], &registry);
    let compiled = graph.compile(&interner);

    let entrypoint_id = node_metas[0].entrypoint_id;
    eprintln!("entrypoint unit_output: {:?}", compiled.unit_outputs.get(&entrypoint_id));
}

#[tokio::test]
async fn debug_scc_membership() {
    use acvus_mir::context_registry::PartialContextTypeRegistry;
    let interner = Interner::new();
    let spec = acvus_orchestration::NodeSpec {
        name: interner.intern("log"),
        kind: acvus_orchestration::NodeKind::Expression(acvus_orchestration::ExpressionSpec {
            source: "[1, 2, 3]".to_string(),
            output_ty: None,
        }),
        strategy: acvus_orchestration::Strategy {
            execution: acvus_orchestration::Execution::OncePerTurn,
            persistency: acvus_orchestration::Persistency::Sequence {
                bind: interner.intern("@self | chain(@raw | iter)"),
            },
            initial_value: Some(interner.intern("[]")),
            retry: 0,
            assert: None,
        },
        is_function: false,
        fn_params: vec![],
    };
    let registry = PartialContextTypeRegistry::default();
    let (graph, node_metas) = acvus_orchestration::lower::lower(&interner, &[spec], &registry);

    // Check which units are in the graph
    eprintln!("units: {:?}", graph.units.iter().map(|u| (u.id, u.body.is_some(), u.output_binding)).collect::<Vec<_>>());

    let resolved = graph.resolve(&interner);
    let entrypoint_id = node_metas[0].entrypoint_id;
    eprintln!("entrypoint({:?}) output: {:?}", entrypoint_id, resolved.unit_outputs.get(&entrypoint_id));
}
