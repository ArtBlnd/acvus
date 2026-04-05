//! Inliner e2e tests.
//!
//! Tests compile multiple local functions, inline, and snapshot the resulting IR.
//! Organized by category with soundness and completeness coverage.

use acvus_mir::graph::{Constraint, FnConstraint, FnKind, Function, QualifiedRef, Signature};
use acvus_mir::ty::{Effect, Param, Ty};
use acvus_mir_test::*;
use acvus_utils::Interner;

fn sig(i: &Interner, params: &[(&str, Ty)]) -> Option<Signature> {
    Some(Signature {
        params: params
            .iter()
            .map(|(name, ty)| Param::new(i.intern(name), ty.clone()))
            .collect(),
    })
}

fn obj(i: &Interner, fields: &[(&str, Ty)]) -> Ty {
    Ty::Object(
        fields
            .iter()
            .map(|(k, v)| (i.intern(k), v.clone()))
            .collect(),
    )
}

// ═══════════════════════════════════════════════════════════════════════
//  1. Basic inline — local function calls become flat IR
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn inline_simple_call() {
    // double(x) = x + x; main calls double(5)
    let i = Interner::new();
    let ir = compile_inline_ir(
        &i,
        ("main", "double(5)"),
        &[("double", "$x + $x", sig(&i, &[("x", Ty::Int)]))],
        &[],
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

#[test]
fn inline_multi_arg() {
    // add(a, b) = a + b; main calls add(3, 4)
    let i = Interner::new();
    let ir = compile_inline_ir(
        &i,
        ("main", "add(3, 4)"),
        &[("add", "$a + $b", sig(&i, &[("a", Ty::Int), ("b", Ty::Int)]))],
        &[],
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

#[test]
fn inline_chain() {
    // inc(x) = x + 1; double(x) = x + x; main = double(inc(3))
    let i = Interner::new();
    let ir = compile_inline_ir(
        &i,
        ("main", "double(inc(3))"),
        &[
            ("inc", "$x + 1", sig(&i, &[("x", Ty::Int)])),
            ("double", "$x + $x", sig(&i, &[("x", Ty::Int)])),
        ],
        &[],
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

#[test]
fn inline_multiple_calls() {
    // inc(x) = x + 1; main = inc(1) + inc(2)
    let i = Interner::new();
    let ir = compile_inline_ir(
        &i,
        ("main", "inc(1) + inc(2)"),
        &[("inc", "$x + 1", sig(&i, &[("x", Ty::Int)]))],
        &[],
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

#[test]
fn inline_return_used_in_binop() {
    // square(x) = x * x; main = square(3) + square(4)
    let i = Interner::new();
    let ir = compile_inline_ir(
        &i,
        ("main", "square(3) + square(4)"),
        &[("square", "$x * $x", sig(&i, &[("x", Ty::Int)]))],
        &[],
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

#[test]
fn inline_no_arg_function() {
    // get_five() = 5; main = get_five()
    let i = Interner::new();
    let ir = compile_inline_ir(
        &i,
        ("main", "get_five()"),
        &[("get_five", "5", sig(&i, &[]))],
        &[],
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

#[test]
fn inline_pipe_syntax() {
    // double(x) = x + x; main = 5 | double
    let i = Interner::new();
    let ir = compile_inline_ir(
        &i,
        ("main", "5 | double"),
        &[("double", "$x + $x", sig(&i, &[("x", Ty::Int)]))],
        &[],
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

#[test]
fn inline_pipe_with_extra_args() {
    // add(a, b) = a + b; main = 3 | add(4)
    let i = Interner::new();
    let ir = compile_inline_ir(
        &i,
        ("main", "3 | add(4)"),
        &[("add", "$a + $b", sig(&i, &[("a", Ty::Int), ("b", Ty::Int)]))],
        &[],
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

// ═══════════════════════════════════════════════════════════════════════
//  2. ExternFn preservation — extern calls must NOT be inlined
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn inline_preserves_extern_call() {
    // main calls to_string (ExternFn) — should remain as FunctionCall
    let i = Interner::new();
    let ir = compile_inline_ir(&i, ("main", "42 | to_string"), &[], &[]).unwrap();
    insta::assert_snapshot!(ir);
}

#[test]
fn inline_local_around_extern() {
    // wrap(x) = x | to_string; main = wrap(42)
    // After inline: to_string call remains, wrap is inlined.
    let i = Interner::new();
    let ir = compile_inline_ir(
        &i,
        ("main", "wrap(42)"),
        &[("wrap", "$x | to_string", sig(&i, &[("x", Ty::Int)]))],
        &[],
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

#[test]
fn inline_extern_chain_preserved() {
    // process(s) = s | len_str | to_string; main = process("hello")
    // After inline: both len_str and to_string remain as calls.
    let i = Interner::new();
    let ir = compile_inline_ir(
        &i,
        ("main", r#"process("hello")"#),
        &[(
            "process",
            "$s | len_str | to_string",
            sig(&i, &[("s", Ty::String)]),
        )],
        &[],
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

#[test]
fn inline_mixed_local_extern() {
    // double(x) = x + x; main = double(3) | to_string
    // double is inlined, to_string remains.
    let i = Interner::new();
    let ir = compile_inline_ir(
        &i,
        ("main", "double(3) | to_string"),
        &[("double", "$x + $x", sig(&i, &[("x", Ty::Int)]))],
        &[],
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

// ═══════════════════════════════════════════════════════════════════════
//  3. Context propagation — inline + context read/write
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn inline_callee_reads_context() {
    // get_count() = @count; main = get_count()
    let i = Interner::new();
    let ir = compile_inline_ir(
        &i,
        ("main", "get_count()"),
        &[("get_count", "@count", sig(&i, &[]))],
        &[("count", Ty::Int)],
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

#[test]
fn inline_callee_writes_context() {
    // set_count(x) = { @count = x; x }; main = set_count(42)
    let i = Interner::new();
    let ir = compile_inline_ir(
        &i,
        ("main", "set_count(42)"),
        &[("set_count", "@count = $x; $x", sig(&i, &[("x", Ty::Int)]))],
        &[("count", Ty::Int)],
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

#[test]
fn inline_caller_and_callee_read_same_context() {
    // add_count(x) = x + @count; main = @count + add_count(1)
    let i = Interner::new();
    let ir = compile_inline_ir(
        &i,
        ("main", "@count + add_count(1)"),
        &[("add_count", "$x + @count", sig(&i, &[("x", Ty::Int)]))],
        &[("count", Ty::Int)],
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

#[test]
fn inline_callee_writes_caller_reads() {
    // bump() = { @count = @count + 1; @count }; main = { x = bump(); x + @count }
    let i = Interner::new();
    let ir = compile_inline_ir(
        &i,
        ("main", "x = bump(); x + @count"),
        &[("bump", "@count = @count + 1; @count", sig(&i, &[]))],
        &[("count", Ty::Int)],
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

#[test]
fn inline_multiple_context_writes() {
    // init() = { @a = 1; @b = 2; @a + @b }; main = init()
    let i = Interner::new();
    let ir = compile_inline_ir(
        &i,
        ("main", "init()"),
        &[("init", "@a = 1; @b = 2; @a + @b", sig(&i, &[]))],
        &[("a", Ty::Int), ("b", Ty::Int)],
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

// ═══════════════════════════════════════════════════════════════════════
//  4. Closure / Lambda — capture remap, lambda as argument
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn inline_callee_returns_closure_result() {
    // apply_double(xs) = xs | map(|x| -> x + x); main = apply_double([1, 2, 3]) | collect
    let i = Interner::new();
    let ir = compile_inline_ir(
        &i,
        ("main", "apply_double([1, 2, 3]) | collect"),
        &[(
            "apply_double",
            "$xs | map(|x| -> x + x)",
            sig(&i, &[("xs", Ty::List(Box::new(Ty::Int)))]),
        )],
        &[],
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

#[test]
fn inline_callee_takes_lambda_arg() {
    // apply(f, x) = f(x) — but f is indirect, so it won't inline further
    // Hmm, f would be Indirect. Let's do: transform(xs) = xs | map(|x| -> x * 2)
    // main = transform([1, 2])
    let i = Interner::new();
    let ir = compile_inline_ir(
        &i,
        ("main", "transform([1, 2]) | collect"),
        &[(
            "transform",
            "$xs | map(|x| -> x * 2)",
            sig(&i, &[("xs", Ty::List(Box::new(Ty::Int)))]),
        )],
        &[],
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

#[test]
fn inline_callee_with_filter_lambda() {
    // positives(xs) = xs | filter(|x| -> x > 0); main = positives([1, -2, 3]) | collect
    let i = Interner::new();
    let ir = compile_inline_ir(
        &i,
        ("main", "positives([1, -2, 3]) | collect"),
        &[(
            "positives",
            "$xs | filter(|x| -> x > 0)",
            sig(&i, &[("xs", Ty::List(Box::new(Ty::Int)))]),
        )],
        &[],
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

#[test]
fn inline_callee_lambda_captures_param() {
    // add_n(xs, n) = xs | map(|x| -> x + n); main = add_n(@items, 10) | collect
    let i = Interner::new();
    let ir = compile_inline_ir(
        &i,
        ("main", "add_n(@items, 10) | collect"),
        &[(
            "add_n",
            "$xs | map(|x| -> x + $n)",
            sig(&i, &[("xs", Ty::List(Box::new(Ty::Int))), ("n", Ty::Int)]),
        )],
        &[("items", Ty::List(Box::new(Ty::Int)))],
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

#[test]
fn inline_chain_with_lambda() {
    // double_all(xs) = xs | map(|x| -> x * 2)
    // sum_list(xs) = xs | fold(0, |a, b| -> a + b)
    // main = [1, 2, 3] | double_all | collect | sum_list
    let i = Interner::new();
    let ir = compile_inline_ir(
        &i,
        ("main", "[1, 2, 3] | double_all | collect | sum_list"),
        &[
            (
                "double_all",
                "$xs | map(|x| -> x * 2)",
                sig(&i, &[("xs", Ty::List(Box::new(Ty::Int)))]),
            ),
            (
                "sum_list",
                "$xs | fold(0, |a, b| -> a + b)",
                sig(&i, &[("xs", Ty::List(Box::new(Ty::Int)))]),
            ),
        ],
        &[],
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

// ═══════════════════════════════════════════════════════════════════════
//  5. Effect / Token — IO effect, token-based effect propagation
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn inline_pure_function() {
    // Pure function: no effects. add(a,b)=a+b; main=add(1,2)
    let i = Interner::new();
    let ir = compile_inline_ir(
        &i,
        ("main", "add(1, 2)"),
        &[("add", "$a + $b", sig(&i, &[("a", Ty::Int), ("b", Ty::Int)]))],
        &[],
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

#[test]
fn inline_io_effect_extern_inside() {
    // fetch(x) is ExternFn with IO effect; wrapper calls it.
    // wrapper(id) = fetch(id); main = wrapper(1)
    let i = Interner::new();
    let fetch = Function {
        qref: QualifiedRef::root(i.intern("fetch")),
        kind: FnKind::Extern,
        constraint: FnConstraint {
            signature: Some(Signature {
                params: vec![Param::new(i.intern("id"), Ty::Int)],
            }),
            output: Constraint::Exact(Ty::Fn {
                params: vec![Param::new(i.intern("id"), Ty::Int)],
                ret: Box::new(Ty::String),
                captures: vec![],
                effect: Effect::self_modifying(),
            }),
            effect: None,
            hint: None,
        },
    };
    let ir = compile_inline_ir_with(
        &i,
        ("main", "wrapper(1)"),
        &[("wrapper", "fetch($id)", sig(&i, &[("id", Ty::Int)]))],
        &[],
        &[fetch],
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

#[test]
fn inline_context_effect_propagation() {
    // Callee writes context (effect). After inline, effect is visible in caller.
    // inc_and_get() = { @counter = @counter + 1; @counter }; main = inc_and_get()
    let i = Interner::new();
    let ir = compile_inline_ir(
        &i,
        ("main", "inc_and_get()"),
        &[(
            "inc_and_get",
            "@counter = @counter + 1; @counter",
            sig(&i, &[]),
        )],
        &[("counter", Ty::Int)],
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

// ═══════════════════════════════════════════════════════════════════════
//  6. Soundness rejection — things that must NOT be inlined
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn inline_devirt_known_closure() {
    // Indirect call to a known closure (single MakeClosure def) gets devirtualized and inlined.
    // main = { f = |x| -> x + 1; f(5) }
    let i = Interner::new();
    let ir = compile_inline_ir(&i, ("main", "f = |x| -> x + 1; f(5)"), &[], &[]).unwrap();
    insta::assert_snapshot!(ir);
}

// ═══════════════════════════════════════════════════════════════════════
//  7. Complex / realistic scenarios
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn inline_nested_three_levels() {
    // a(x) = x + 1; b(x) = a(x) + a(x); c(x) = b(x) * 2; main = c(10)
    let i = Interner::new();
    let ir = compile_inline_ir(
        &i,
        ("main", "c(10)"),
        &[
            ("a", "$x + 1", sig(&i, &[("x", Ty::Int)])),
            ("b", "a($x) + a($x)", sig(&i, &[("x", Ty::Int)])),
            ("c", "b($x) * 2", sig(&i, &[("x", Ty::Int)])),
        ],
        &[],
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

#[test]
fn inline_string_operations() {
    // greet(name) = "Hello " + name; main = greet("world")
    let i = Interner::new();
    let ir = compile_inline_ir(
        &i,
        ("main", r#"greet("world")"#),
        &[(
            "greet",
            r#""Hello " + $name"#,
            sig(&i, &[("name", Ty::String)]),
        )],
        &[],
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

#[test]
fn inline_boolean_logic() {
    // both(a, b) = a && b; main = both(true, false)
    let i = Interner::new();
    let ir = compile_inline_ir(
        &i,
        ("main", "both(true, false)"),
        &[(
            "both",
            "$a && $b",
            sig(&i, &[("a", Ty::Bool), ("b", Ty::Bool)]),
        )],
        &[],
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

#[test]
fn inline_comparison() {
    // is_positive(x) = x > 0; main = is_positive(42)
    let i = Interner::new();
    let ir = compile_inline_ir(
        &i,
        ("main", "is_positive(42)"),
        &[("is_positive", "$x > 0", sig(&i, &[("x", Ty::Int)]))],
        &[],
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

#[test]
fn inline_with_local_binding() {
    // compute(x) = { y = x * 2; y + 1 }; main = compute(5)
    let i = Interner::new();
    let ir = compile_inline_ir(
        &i,
        ("main", "compute(5)"),
        &[("compute", "y = $x * 2; y + 1", sig(&i, &[("x", Ty::Int)]))],
        &[],
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

#[test]
fn inline_result_unused_intermediate() {
    // get_a() = 1; get_b() = 2; main = get_a() + get_b()
    let i = Interner::new();
    let ir = compile_inline_ir(
        &i,
        ("main", "get_a() + get_b()"),
        &[("get_a", "1", sig(&i, &[])), ("get_b", "2", sig(&i, &[]))],
        &[],
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

#[test]
fn inline_field_access_after_call() {
    // make_obj() = @user; main = make_obj().name
    let i = Interner::new();
    let ir = compile_inline_ir(
        &i,
        ("main", "make_obj().name"),
        &[("make_obj", "@user", sig(&i, &[]))],
        &[("user", obj(&i, &[("name", Ty::String), ("age", Ty::Int)]))],
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

#[test]
fn inline_list_collect_pattern() {
    // to_list(xs) = xs | map(|x| -> x * 2) | collect; main = to_list([1, 2, 3])
    let i = Interner::new();
    let ir = compile_inline_ir(
        &i,
        ("main", "to_list([1, 2, 3])"),
        &[(
            "to_list",
            "$xs | map(|x| -> x * 2) | collect",
            sig(&i, &[("xs", Ty::List(Box::new(Ty::Int)))]),
        )],
        &[],
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

#[test]
fn inline_multiple_context_different_callees() {
    // read_a() = @a; read_b() = @b; main = read_a() + read_b()
    let i = Interner::new();
    let ir = compile_inline_ir(
        &i,
        ("main", "read_a() + read_b()"),
        &[
            ("read_a", "@a", sig(&i, &[])),
            ("read_b", "@b", sig(&i, &[])),
        ],
        &[("a", Ty::Int), ("b", Ty::Int)],
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

#[test]
fn inline_callee_uses_builtin_len() {
    // count(xs) = xs | len; main = count([1, 2, 3])
    let i = Interner::new();
    let ir = compile_inline_ir(
        &i,
        ("main", "count([1, 2, 3])"),
        &[(
            "count",
            "$xs | len",
            sig(&i, &[("xs", Ty::List(Box::new(Ty::Int)))]),
        )],
        &[],
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}
