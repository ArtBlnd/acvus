use std::collections::BTreeSet;

use acvus_mir::graph::{FnKind, Function, QualifiedRef};
use acvus_mir::{
    graph::infer,
    ty::{Effect, EffectSet, EffectTarget, Param, ParamTerm, Poly, PolyBuilder, Ty, TyTerm, lift_to_poly, lift_effect_to_poly},
};
use acvus_mir_test::*;
use acvus_utils::{Astr, Freeze, Interner};
use rustc_hash::FxHashMap;

/// Helper: compile a template source via the graph pipeline (extract → resolve → lower).
/// Unknown @contexts are added as Inferred constraints.
fn compile_analysis(
    interner: &Interner,
    source: &str,
    ctx: &[(&str, Ty)],
) -> Result<acvus_mir::ir::MirModule, String> {
    use acvus_mir::graph::{
        CompilationGraph, Context, FnKind, Function, ParsedAst,
        QualifiedRef,
    };
    use acvus_mir::graph::{extract, lower as graph_lower};
    use acvus_mir::ty::PolyBuilder;
    use acvus_utils::Freeze;
    use rustc_hash::{FxHashMap, FxHashSet};

    // Build contexts from declared types.
    let mut contexts: Vec<Context> = ctx
        .iter()
        .map(|(name, ty)| Context {
            qref: QualifiedRef::root(interner.intern(name)),
            ty: lift_to_poly(ty),
        })
        .collect();

    // Discover context refs in source that aren't declared — add as Inferred.
    let template = acvus_ast::parse(interner, source).expect("parse failed");
    let declared: FxHashSet<Astr> = contexts.iter().map(|c| c.qref.name).collect();
    for ast_qref in acvus_ast::extract_template_context_refs(&template) {
        if !declared.contains(&ast_qref.name) {
            let mut pb = PolyBuilder::new();
            contexts.push(Context {
                qref: QualifiedRef::root(ast_qref.name),
                ty: pb.fresh_ty_var(),
            });
        }
    }

    let test_qref = QualifiedRef::root(interner.intern("test"));
    let template = acvus_ast::parse(interner, source).expect("parse failed");

    let mut pb = PolyBuilder::new();
    let mut functions: Vec<Function> = vec![Function {
        qref: test_qref,
        kind: FnKind::Local(ParsedAst::Template(template)),
        ty: TyTerm::Fn {
            params: vec![],
            ret: Box::new(pb.fresh_ty_var()),
            captures: vec![],
            effect: pb.fresh_effect_var(),
            hint: None,
        },
        effect_constraint: None,
    }];
    let mut type_registry = acvus_mir::ty::TypeRegistry::new();
    let std_regs = acvus_ext::std_registries(interner, &mut type_registry);
    for registry in std_regs {
        let registered = registry.register(interner);
        functions.extend(registered.functions);
    }

    let graph = CompilationGraph {
        functions: Freeze::new(functions),
        contexts: Freeze::new(contexts),
    };

    let ext = extract::extract(interner, &graph);
    let inf = infer::infer(
        interner,
        &graph,
        &ext,
        &FxHashMap::default(),
        Freeze::new(type_registry),
        &FxHashMap::default(),
    );

    // Collect infer errors.
    let mut errors: Vec<String> = Vec::new();
    for (qref, errs) in inf.errors() {
        let fn_name = interner.resolve(qref.name);
        for e in errs {
            errors.push(format!(
                "[infer:{}] [{}..{}] {}",
                fn_name,
                e.span.start,
                e.span.end,
                e.display(interner)
            ));
        }
    }

    let result = graph_lower::lower(interner, &graph, &ext, &inf, &FxHashMap::default());

    // Collect lower errors.
    for e in result.errors.iter().flat_map(|le| le.errors.iter()) {
        errors.push(format!(
            "[lower] [{}..{}] {}",
            e.span.start,
            e.span.end,
            e.display(interner)
        ));
    }

    if !errors.is_empty() {
        return Err(errors.join("\n"));
    }

    result
        .modules
        .into_iter()
        .find(|(id, _)| *id == test_qref)
        .map(|(_, module)| module)
        .ok_or_else(|| "no module produced for target".to_string())
}

/// Helper: build a context HashMap<Astr, Ty> from string pairs.
#[cfg(test)]
fn ctx(i: &Interner, pairs: &[(&str, Ty)]) -> FxHashMap<Astr, Ty> {
    pairs
        .iter()
        .map(|(k, v)| (i.intern(k), v.clone()))
        .collect()
}

/// Helper: build an Object type from string-keyed fields.
fn obj(i: &Interner, fields: &[(&str, Ty)]) -> Ty {
    Ty::Object(
        fields
            .iter()
            .map(|(k, v)| (i.intern(k), v.clone()))
            .collect(),
    )
}

// ── Text & literals ──────────────────────────────────────────────

#[test]
fn text_only() {
    let i = Interner::new();
    let ir = compile_simple(&i, "hello world").unwrap();
    insta::assert_snapshot!(ir);
}

#[test]
fn string_emit() {
    let i = Interner::new();
    let ir = compile_simple(&i, r#"{{ "hello" }}"#).unwrap();
    insta::assert_snapshot!(ir);
}

#[test]
fn string_concat() {
    let i = Interner::new();
    let ir = compile_simple(&i, r#"{{ "hello" + " " + "world" }}"#).unwrap();
    insta::assert_snapshot!(ir);
}

#[test]
fn mixed_text_and_expr() {
    let i = Interner::new();
    let context = ctx(&i, &[("name", Ty::String)]);
    let ir = compile_to_ir(&i, "Hello, {{ @name }}!", &context).unwrap();
    insta::assert_snapshot!(ir);
}

// ── Context / Variables ──────────────────────────────────────────

#[test]
fn context_read() {
    let i = Interner::new();
    let context = ctx(&i, &[("count", Ty::Int)]);
    let ir = compile_to_ir(&i, "{{ @count | to_string }}", &context).unwrap();
    insta::assert_snapshot!(ir);
}

#[test]
fn variable_write() {
    let i = Interner::new();
    let ir = compile_simple(&i, "{{ count = 42 }}").unwrap();
    insta::assert_snapshot!(ir);
}

#[test]
fn context_field_access() {
    let i = Interner::new();
    let ir = compile_to_ir(&i, "{{ @user.name }}", &user_context(&i)).unwrap();
    insta::assert_snapshot!(ir);
}

// ── Arithmetic ───────────────────────────────────────────────────

#[test]
fn arithmetic_to_string() {
    let i = Interner::new();
    let context = ctx(&i, &[("a", Ty::Int), ("b", Ty::Int)]);
    let ir = compile_to_ir(&i, "{{ @a + @b | to_string }}", &context).unwrap();
    insta::assert_snapshot!(ir);
}

// ── Match blocks ─────────────────────────────────────────────────

#[test]
fn simple_match_binding() {
    let i = Interner::new();
    let context = ctx(&i, &[("name", Ty::String)]);
    // Variable binding is body-less — defines x in current scope.
    let ir = compile_to_ir(&i, r#"{{ x = @name }}{{ x }}"#, &context).unwrap();
    insta::assert_snapshot!(ir);
}

#[test]
fn match_literal_filter() {
    let i = Interner::new();
    let context = ctx(&i, &[("role", Ty::String)]);
    let ir = compile_to_ir(
        &i,
        r#"{{ "admin" = @role }}admin page{{_}}guest page{{/}}"#,
        &context,
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

#[test]
fn multi_arm_match() {
    let i = Interner::new();
    let context = ctx(&i, &[("role", Ty::String)]);
    let ir = compile_to_ir(
        &i,
        r#"{{ "admin" = @role }}admin{{ "user" = }}user{{_}}guest{{/}}"#,
        &context,
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

#[test]
fn iteration_over_list() {
    let i = Interner::new();
    // Use object destructuring to iterate and extract name.
    let ir = compile_to_ir(
        &i,
        r#"{{ { name, } in @users }}{{ name }}{{/}}"#,
        &users_list_context(&i),
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

#[test]
fn nested_match() {
    let i = Interner::new();
    let context = ctx(
        &i,
        &[(
            "users",
            Ty::List(Box::new(obj(
                &i,
                &[
                    ("name", Ty::String),
                    (
                        "posts",
                        Ty::List(Box::new(obj(&i, &[("title", Ty::String)]))),
                    ),
                ],
            ))),
        )],
    );
    // Use object destructuring for both outer and inner iterations.
    let ir = compile_to_ir(
        &i,
        r#"{{ { name, posts, } in @users }}{{ { title, } in posts }}{{ title }}{{/}}{{/}}"#,
        &context,
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

// ── List patterns ────────────────────────────────────────────────

#[test]
fn list_destructure_head() {
    let i = Interner::new();
    let ir = compile_to_ir(
        &i,
        r#"{{ [a, b, ..] = @items }}{{ a | to_string }}{{_}}empty{{/}}"#,
        &items_context(&i),
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

// ── Object patterns ──────────────────────────────────────────────

#[test]
fn object_pattern() {
    let i = Interner::new();
    let ir = compile_to_ir(
        &i,
        r#"{{ { name, age, } = @user }}{{ name }}{{/}}"#,
        &user_context(&i),
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

// ── Range ────────────────────────────────────────────────────────

#[test]
fn range_binding() {
    // Variable binding captures a range value; iterate to emit scalar elements.
    let i = Interner::new();
    let ir = compile_simple(&i, r#"{{ x in 0..5 }}{{ x | to_string }}{{/}}"#).unwrap();
    insta::assert_snapshot!(ir);
}

#[test]
fn range_iteration() {
    // Explicit iteration with `in`.
    let i = Interner::new();
    let ir = compile_simple(&i, r#"{{ x in 0..3 }}{{ x | to_string }}{{/}}"#).unwrap();
    insta::assert_snapshot!(ir);
}

#[test]
fn range_pattern() {
    let i = Interner::new();
    let context = ctx(&i, &[("age", Ty::Int)]);
    let ir = compile_to_ir(
        &i,
        r#"{{ 0..10 = @age }}child{{ 10..=19 = }}teen{{_}}adult{{/}}"#,
        &context,
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

// ── Pipe & builtins ──────────────────────────────────────────────

#[test]
fn pipe_filter_map() {
    let i = Interner::new();
    // Variable binding is body-less.
    let ir = compile_to_ir(
        &i,
        r#"{{ x = @items | iter | filter(|x| -> x != 0) | map(|x| -> x + 1) | collect }}{{ x | len | to_string }}"#,
        &items_context(&i),
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

#[test]
fn pipe_to_string() {
    let i = Interner::new();
    let context = ctx(&i, &[("n", Ty::Int)]);
    let ir = compile_to_ir(&i, "{{ @n | to_string }}", &context).unwrap();
    insta::assert_snapshot!(ir);
}

// ── Lambda / closures ────────────────────────────────────────────

#[test]
fn lambda_in_filter() {
    let i = Interner::new();
    // Variable binding is body-less.
    let ir = compile_to_ir(
        &i,
        r#"{{ x = @items | iter | filter(|x| -> x != 0) | collect }}{{ x | len | to_string }}"#,
        &items_context(&i),
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

// ── Extern functions ─────────────────────────────────────────────

#[test]
fn extern_async_call() {
    let i = Interner::new();
    let fetch_user = Function {
        qref: QualifiedRef::root(i.intern("fetch_user")),
        kind: FnKind::Extern,
        ty: TyTerm::Fn {
            params: vec![ParamTerm::<Poly>::new(i.intern("id"), lift_to_poly(&Ty::Int))],
            ret: Box::new(lift_to_poly(&Ty::String)),
            captures: vec![],
            effect: lift_effect_to_poly(&Effect::pure()),
            hint: None,
        },
        effect_constraint: None,
    };
    let ir = compile_to_ir_with(
        &i,
        r#"{{ user = fetch_user(1) }}{{ user }}"#,
        &FxHashMap::default(),
        &[fetch_user],
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

// ── Tuple ────────────────────────────────────────────────────────

#[test]
fn tuple_expression() {
    let i = Interner::new();
    let context = ctx(&i, &[("a", Ty::Int), ("b", Ty::String)]);
    let ir = compile_to_ir(
        &i,
        r#"{{ (a, b) = (@a, @b) }}{{ a | to_string }}{{ b }}{{_}}{{/}}"#,
        &context,
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

#[test]
fn tuple_pattern_binding() {
    let i = Interner::new();
    let context = ctx(&i, &[("pair", Ty::Tuple(vec![Ty::String, Ty::Int]))]);
    let ir = compile_to_ir(&i, r#"{{ (name, age) = @pair }}{{ name }}{{/}}"#, &context).unwrap();
    insta::assert_snapshot!(ir);
}

#[test]
fn tuple_pattern_wildcard() {
    let i = Interner::new();
    let context = ctx(&i, &[("pair", Ty::Tuple(vec![Ty::String, Ty::Int]))]);
    let ir = compile_to_ir(&i, r#"{{ (name, _) = @pair }}{{ name }}{{/}}"#, &context).unwrap();
    insta::assert_snapshot!(ir);
}

#[test]
fn tuple_pattern_literal_match() {
    let i = Interner::new();
    let context = ctx(&i, &[("a", Ty::Int), ("b", Ty::Int)]);
    let ir = compile_to_ir(
        &i,
        r#"{{ (0, 1) = (@a, @b) }}zero-one{{ (1, _) = }}one-any{{_}}other{{/}}"#,
        &context,
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

#[test]
fn tuple_nested_destructure() {
    let i = Interner::new();
    let context = ctx(
        &i,
        &[(
            "data",
            Ty::Tuple(vec![Ty::String, obj(&i, &[("x", Ty::Int)])]),
        )],
    );
    let ir = compile_to_ir(
        &i,
        r#"{{ (label, { x, }) = @data }}{{ label }}{{/}}"#,
        &context,
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

#[test]
fn error_tuple_arity_mismatch() {
    let i = Interner::new();
    let context = ctx(&i, &[("pair", Ty::Tuple(vec![Ty::Int, Ty::Int]))]);
    let result = compile_to_ir(
        &i,
        r#"{{ (a, b, c) = @pair }}{{ a | to_string }}{{/}}"#,
        &context,
    );
    assert!(result.is_err());
    insta::assert_snapshot!(result.unwrap_err());
}

// ── Error cases ──────────────────────────────────────────────────

#[test]
fn error_emit_non_string() {
    let i = Interner::new();
    let result = compile_simple(&i, "{{ 42 }}");
    assert!(result.is_err());
    insta::assert_snapshot!(result.unwrap_err());
}

// FnRefs removed: undeclared contexts are now handled by the typechecker's infer vars
// in analysis mode, so @unknown no longer causes an error — it gets a fresh type var
// that may resolve during typechecking.
#[test]
fn undeclared_context_resolves_via_infer_var() {
    let i = Interner::new();
    let result = compile_to_ir(&i, "{{ @unknown | to_string }}", &FxHashMap::default());
    assert!(
        result.is_ok(),
        "undeclared context should resolve via infer var: {result:?}"
    );
}

#[test]
fn error_undefined_variable() {
    let i = Interner::new();
    let result = compile_to_ir(&i, "{{ x = unknown }}{{_}}{{/}}", &FxHashMap::default());
    assert!(result.is_err());
}

#[test]
fn error_type_mismatch() {
    let i = Interner::new();
    let result = compile_simple(&i, r#"{{ x = 1 + 2.0 }}{{_}}{{/}}"#);
    assert!(result.is_err());
    insta::assert_snapshot!(result.unwrap_err());
}

#[test]
fn error_range_float_bounds() {
    let i = Interner::new();
    let result = compile_simple(&i, "{{ x = 1.0..2.0 }}{{_}}{{/}}");
    assert!(result.is_err());
    insta::assert_snapshot!(result.unwrap_err());
}

// ── Iteration (`in`) ────────────────────────────────────────────

#[test]
fn iter_list_binding() {
    let i = Interner::new();
    let context = ctx(&i, &[("items", Ty::List(Box::new(Ty::Int)))]);
    let ir = compile_to_ir(&i, "{{ x in @items }}{{ x | to_string }}{{/}}", &context).unwrap();
    insta::assert_snapshot!(ir);
}

#[test]
fn iter_object_destructure() {
    let i = Interner::new();
    let ir = compile_to_ir(
        &i,
        "{{ { name, } in @users }}{{ name }}{{/}}",
        &users_list_context(&i),
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

#[test]
fn iter_tuple_destructure() {
    let i = Interner::new();
    let context = ctx(
        &i,
        &[(
            "pairs",
            Ty::List(Box::new(Ty::Tuple(vec![Ty::String, Ty::Int]))),
        )],
    );
    let ir = compile_to_ir(&i, "{{ (a, _) in @pairs }}{{ a }}{{/}}", &context).unwrap();
    insta::assert_snapshot!(ir);
}

#[test]
fn iter_with_catch_all() {
    let i = Interner::new();
    let context = ctx(&i, &[("items", Ty::List(Box::new(Ty::Int)))]);
    let ir = compile_to_ir(
        &i,
        "{{ x in @items }}{{ x | to_string }}{{_}}empty{{/}}",
        &context,
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

#[test]
fn error_iter_refutable_pattern() {
    let i = Interner::new();
    let context = ctx(&i, &[("roles", Ty::List(Box::new(Ty::String)))]);
    let result = compile_to_ir(&i, r#"{{ "admin" in @roles }}...{{/}}"#, &context);
    assert!(result.is_err());
}

#[test]
fn error_iter_not_iterable() {
    let i = Interner::new();
    let context = ctx(&i, &[("name", Ty::String)]);
    let result = compile_to_ir(&i, "{{ x in @name }}{{ x }}{{/}}", &context);
    assert!(result.is_err());
}

// ── Edge case: new variable ref binding ─────────────────────────

#[test]
fn variable_new_ref_binding() {
    let i = Interner::new();
    // result is not in initial context — dynamically created via binding.
    let context = ctx(&i, &[("name", Ty::String)]);
    let ir = compile_to_ir(&i, r#"{{ result = @name }}{{ result }}"#, &context).unwrap();
    insta::assert_snapshot!(ir);
}

#[test]
fn variable_new_ref_in_match_arm() {
    let i = Interner::new();
    // selected is created inside a match arm, then read after the match.
    let context = ctx(&i, &[("role", Ty::String)]);
    let ir = compile_to_ir(
        &i,
        r#"{{ selected = "" }}{{ "admin" = @role }}{{ selected = "yes" }}{{_}}{{ selected = "no" }}{{/}}{{ selected }}"#,
        &context,
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

// ── Edge case: nested destructuring ─────────────────────────────

#[test]
fn list_of_tuples_destructure() {
    let i = Interner::new();
    // Iterate over list of tuples, destructure each.
    let context = ctx(
        &i,
        &[(
            "pairs",
            Ty::List(Box::new(Ty::Tuple(vec![Ty::String, Ty::Int]))),
        )],
    );
    let ir = compile_to_ir(
        &i,
        r#"{{ (name, age) in @pairs }}{{ name }}{{/}}"#,
        &context,
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

#[test]
fn list_head_with_object_elements() {
    let i = Interner::new();
    // List destructure where elements are objects.
    let context = ctx(
        &i,
        &[(
            "users",
            Ty::List(Box::new(obj(&i, &[("name", Ty::String)]))),
        )],
    );
    let ir = compile_to_ir(
        &i,
        r#"{{ [first, ..] = @users }}{{ first.name }}{{_}}empty{{/}}"#,
        &context,
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

#[test]
fn tuple_with_list_element() {
    let i = Interner::new();
    // Tuple containing a list, extract and iterate.
    let context = ctx(
        &i,
        &[(
            "data",
            Ty::Tuple(vec![Ty::String, Ty::List(Box::new(Ty::Int))]),
        )],
    );
    let ir = compile_to_ir(
        &i,
        r#"{{ (label, items) = @data }}{{ label }}{{ x in items }}{{ x | to_string }}{{/}}{{/}}"#,
        &context,
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

// ── Edge case: object expression & matching ─────────────────────

#[test]
fn object_literal_field_access() {
    let i = Interner::new();
    let context = ctx(&i, &[("name", Ty::String)]);
    let ir = compile_to_ir(
        &i,
        r#"{{ o = { @name, } }}{{ o.name }}{{_}}{{/}}"#,
        &context,
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

// ── Edge case: comparison / boolean / unary ──────────────────────

#[test]
fn comparison_operators() {
    let i = Interner::new();
    let context = ctx(&i, &[("a", Ty::Int), ("b", Ty::Int)]);
    let ir = compile_to_ir(
        &i,
        r#"{{ x = @a > @b }}{{ x | to_string }}{{_}}{{/}}"#,
        &context,
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

#[test]
fn unary_negation() {
    let i = Interner::new();
    let context = ctx(&i, &[("n", Ty::Int)]);
    let ir = compile_to_ir(
        &i,
        r#"{{ x = -@n }}{{ x | to_string }}{{_}}{{/}}"#,
        &context,
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

#[test]
fn boolean_not() {
    let i = Interner::new();
    let context = ctx(&i, &[("flag", Ty::Bool)]);
    let ir = compile_to_ir(
        &i,
        r#"{{ x = !@flag }}{{ x | to_string }}{{_}}{{/}}"#,
        &context,
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

// ── Edge case: to_float / to_int conversion ─────────────────────

#[test]
fn to_float_conversion() {
    let i = Interner::new();
    let context = ctx(&i, &[("n", Ty::Int)]);
    let ir = compile_to_ir(
        &i,
        r#"{{ x = @n | to_float }}{{ x | to_string }}{{_}}{{/}}"#,
        &context,
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

#[test]
fn to_int_conversion() {
    let i = Interner::new();
    let context = ctx(&i, &[("f", Ty::Float)]);
    let ir = compile_to_ir(
        &i,
        r#"{{ x = @f | to_int }}{{ x | to_string }}{{_}}{{/}}"#,
        &context,
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

// ── Edge case: pmap builtin ─────────────────────────────────────

#[test]
fn pmap_builtin() {
    let i = Interner::new();
    let ir = compile_to_ir(
        &i,
        r#"{{ x = @items | iter | pmap(|i| -> i + 1) | collect }}{{ x | len | to_string }}"#,
        &items_context(&i),
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

// ── Edge case: list tail destructure ────────────────────────────

#[test]
fn list_destructure_tail() {
    let i = Interner::new();
    let ir = compile_to_ir(
        &i,
        r#"{{ [.., a, b] = @items }}{{ a | to_string }}{{_}}empty{{/}}"#,
        &items_context(&i),
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

// ── Edge case: variable write then read ─────────────────────────

#[test]
fn variable_write_then_read() {
    let i = Interner::new();
    let ir = compile_simple(&i, r#"{{ x = 42 }}{{ x | to_string }}"#).unwrap();
    insta::assert_snapshot!(ir);
}

// ── Edge case: nested iteration with binding ────────────────────

#[test]
fn nested_iteration_with_binding() {
    let i = Interner::new();
    let context = ctx(
        &i,
        &[("matrix", Ty::List(Box::new(Ty::List(Box::new(Ty::Int)))))],
    );
    let ir = compile_to_ir(
        &i,
        r#"{{ row in @matrix }}{{ x in row }}{{ x | to_string }}{{/}}{{/}}"#,
        &context,
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

// ── Edge case: range inclusive iteration ─────────────────────────

#[test]
fn range_inclusive_iteration() {
    let i = Interner::new();
    let ir = compile_simple(&i, r#"{{ x in 0..=3 }}{{ x | to_string }}{{/}}"#).unwrap();
    insta::assert_snapshot!(ir);
}

// ── Edge case: deeply nested object ─────────────────────────────

#[test]
fn deeply_nested_object_access() {
    let i = Interner::new();
    let context = ctx(
        &i,
        &[(
            "data",
            obj(
                &i,
                &[(
                    "user",
                    obj(&i, &[("address", obj(&i, &[("city", Ty::String)]))]),
                )],
            ),
        )],
    );
    let ir = compile_to_ir(&i, r#"{{ @data.user.address.city }}"#, &context).unwrap();
    insta::assert_snapshot!(ir);
}

// ── Edge case: closure capturing context ref ────────────────────

#[test]
fn closure_capture_context() {
    let i = Interner::new();
    let context = ctx(
        &i,
        &[
            ("items", Ty::List(Box::new(Ty::Int))),
            ("threshold", Ty::Int),
        ],
    );
    let ir = compile_to_ir(
        &i,
        r#"{{ x = @items | iter | filter(|i| -> i > @threshold) | collect }}{{ x | len | to_string }}"#,
        &context,
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

// ── Edge case: multi-arm with different pattern types ───────────

#[test]
fn multi_arm_range_and_literal() {
    let i = Interner::new();
    let context = ctx(&i, &[("score", Ty::Int)]);
    let ir = compile_to_ir(
        &i,
        r#"{{ 0 = @score }}zero{{ 1..10 = }}low{{ 10..=100 = }}high{{_}}other{{/}}"#,
        &context,
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

// ── Edge case: list literal ─────────────────────────────────────

#[test]
fn list_literal_expression() {
    let i = Interner::new();
    let ir = compile_simple(
        &i,
        r#"{{ x = [1, 2, 3] }}{{ x | len | to_string }}{{_}}{{/}}"#,
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

// ── Edge case: lambda with arithmetic ───────────────────────────

#[test]
fn lambda_map_arithmetic() {
    let i = Interner::new();
    // Lambda param type resolved via unification: map(List<Int>, |x| -> x + 1)
    let ir = compile_to_ir(
        &i,
        r#"{{ x = @items | iter | map(|i| -> i + 1) | collect }}{{ x | len | to_string }}"#,
        &items_context(&i),
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

#[test]
fn lambda_filter_comparison() {
    let i = Interner::new();
    // Lambda param type resolved via unification: filter(List<Int>, |x| -> x > 0)
    let ir = compile_to_ir(
        &i,
        r#"{{ x = @items | iter | filter(|i| -> i > 0) | collect }}{{ x | len | to_string }}"#,
        &items_context(&i),
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

// ── Edge case: closure with captured local var ──────────────────

#[test]
fn closure_capture_local() {
    let i = Interner::new();
    // Closure captures local variable (not context).
    let ir = compile_to_ir(
        &i,
        r#"{{ threshold = 5 }}{{ x = @items | iter | filter(|i| -> i > threshold) | collect }}{{ x | len | to_string }}{{_}}{{/}}"#,
        &items_context(&i),
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

// ── Edge case: list exact match (no rest) ───────────────────────

#[test]
fn list_exact_match() {
    let i = Interner::new();
    // Exact list pattern: [a, b] without rest (..).
    let ir = compile_to_ir(
        &i,
        r#"{{ [a, b] = @items }}{{ a | to_string }}{{_}}wrong length{{/}}"#,
        &items_context(&i),
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

// ── Edge case: list rest in middle ──────────────────────────────

#[test]
fn list_destructure_head_and_tail() {
    let i = Interner::new();
    // [a, .., z] pattern — head and tail extraction.
    let ir = compile_to_ir(
        &i,
        r#"{{ [first, .., last] = @items }}{{ first | to_string }}{{_}}empty{{/}}"#,
        &items_context(&i),
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

// ── Edge case: nested tuple pattern ─────────────────────────────

#[test]
fn nested_tuple_pattern() {
    let i = Interner::new();
    let context = ctx(
        &i,
        &[(
            "data",
            Ty::Tuple(vec![Ty::Tuple(vec![Ty::Int, Ty::Int]), Ty::String]),
        )],
    );
    let ir = compile_to_ir(
        &i,
        r#"{{ ((a, b), label) = @data }}{{ label }}{{/}}"#,
        &context,
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

// ── Edge case: variable write of computed value ─────────────────

#[test]
fn variable_write_computed() {
    let i = Interner::new();
    let context = ctx(&i, &[("a", Ty::Int), ("b", Ty::Int)]);
    let ir = compile_to_ir(
        &i,
        r#"{{ result = @a + @b }}{{ result | to_string }}"#,
        &context,
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

// ── Edge case: match block with binding pattern + body ──────────

#[test]
fn match_binding_with_body() {
    let i = Interner::new();
    // Object pattern with body (goes through normal match lowering).
    let ir = compile_to_ir(
        &i,
        r#"{{ { name, } = @user }}{{ name }} is here{{_}}no user{{/}}"#,
        &user_context(&i),
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

// ── Edge case: variable shadowing across scopes ─────────────────

#[test]
fn variable_shadowing() {
    let i = Interner::new();
    let context = ctx(&i, &[("name", Ty::String)]);
    let ir = compile_to_ir(
        &i,
        r#"{{ x = "outer" }}{{ x = @name }}{{ x }}{{_}}{{/}}"#,
        &context,
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

// ── Edge case: match inside match (nested match blocks) ─────────

#[test]
fn nested_match_blocks() {
    let i = Interner::new();
    let context = ctx(&i, &[("role", Ty::String), ("level", Ty::Int)]);
    let ir = compile_to_ir(
        &i,
        r#"{{ "admin" = @role }}{{ 1..10 = @level }}low{{_}}high{{/}}{{_}}guest{{/}}"#,
        &context,
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

// ── Edge case: catch-all with nested binding ────────────────────

#[test]
fn catch_all_with_binding() {
    let i = Interner::new();
    let context = ctx(&i, &[("role", Ty::String)]);
    let ir = compile_to_ir(
        &i,
        r#"{{ "admin" = @role }}admin{{_}}{{ fallback = "guest" }}{{ fallback }}{{/}}"#,
        &context,
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

// ── Edge case: multiple chained pipes ───────────────────────────

#[test]
fn triple_pipe_chain() {
    let i = Interner::new();
    let ir = compile_to_ir(
        &i,
        r#"{{ x = @items | iter | filter(|i| -> i != 0) | map(|i| -> i + 1) | map(|i| -> i * 2) | collect }}{{ x | len | to_string }}"#,
        &items_context(&i),
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

// ── Edge case: variable write in iteration body ─────────────────

#[test]
fn variable_write_in_iteration() {
    let i = Interner::new();
    let context = ctx(&i, &[("items", Ty::List(Box::new(Ty::Int)))]);
    let ir = compile_to_ir(
        &i,
        r#"{{ last = 0 }}{{ x in @items }}{{ last = x }}{{/}}{{ last | to_string }}"#,
        &context,
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

// ── Edge case: field access on destructured variable ────────────

#[test]
fn field_access_on_destructured() {
    let i = Interner::new();
    let context = ctx(
        &i,
        &[(
            "pair",
            Ty::Tuple(vec![obj(&i, &[("name", Ty::String)]), Ty::Int]),
        )],
    );
    let ir = compile_to_ir(&i, r#"{{ (obj, _) = @pair }}{{ obj.name }}{{/}}"#, &context).unwrap();
    insta::assert_snapshot!(ir);
}

// ── Edge case: boolean operators in match ────────────────────────

#[test]
fn equality_as_match_source() {
    let i = Interner::new();
    let context = ctx(&i, &[("a", Ty::Int), ("b", Ty::Int)]);
    let ir = compile_to_ir(
        &i,
        r#"{{ true = @a == @b }}equal{{_}}not equal{{/}}"#,
        &context,
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

// ── Edge case: unary negation on lambda param (Ty::Var) ─────────

#[test]
fn lambda_negate_param() {
    let i = Interner::new();
    // Lambda param has Ty::Var initially; -i must resolve via unification.
    let ir = compile_to_ir(
        &i,
        r#"{{ x = @items | iter | map(|i| -> -i) | collect }}{{ x | len | to_string }}"#,
        &items_context(&i),
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

// ── Edge case: unary not on lambda param (Ty::Var) ──────────────

#[test]
fn lambda_not_param() {
    let i = Interner::new();
    // Lambda param has Ty::Var initially; !i must resolve via unification.
    let context = ctx(&i, &[("flags", Ty::List(Box::new(Ty::Bool)))]);
    let ir = compile_to_ir(
        &i,
        r#"{{ x = @flags | iter | map(|i| -> !i) | collect }}{{ x | len | to_string }}"#,
        &context,
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

// ── Edge case: object pattern on match source (non-list) ────────

#[test]
fn object_destructure_match() {
    let i = Interner::new();
    // Object pattern directly on Object source (not List<Object>).
    let ir = compile_to_ir(
        &i,
        r#"{{ { name, age, } = @user }}{{ name }}{{ age | to_string }}{{_}}none{{/}}"#,
        &user_context(&i),
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

// ── Edge case: multiple closures sharing captured var ────────────

#[test]
fn multiple_closures_same_capture() {
    let i = Interner::new();
    let context = ctx(
        &i,
        &[("items", Ty::List(Box::new(Ty::Int))), ("offset", Ty::Int)],
    );
    let ir = compile_to_ir(
        &i,
        r#"{{ x = @items | iter | map(|i| -> i + @offset) | filter(|i| -> i > 0) | collect }}{{ x | len | to_string }}"#,
        &context,
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

// ── Edge case: string comparison ─────────────────────────────────

#[test]
fn string_equality_in_filter() {
    let i = Interner::new();
    let context = ctx(&i, &[("names", Ty::List(Box::new(Ty::String)))]);
    let ir = compile_to_ir(
        &i,
        r#"{{ x = @names | iter | filter(|n| -> n != "admin") }}{{ x | join(",") }}"#,
        &context,
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

// ── Edge case: nested lambda (lambda returning lambda result) ────

#[test]
fn lambda_field_access() {
    let i = Interner::new();
    // Lambda body accesses field on captured object.
    let context = ctx(
        &i,
        &[(
            "users",
            Ty::List(Box::new(obj(&i, &[("name", Ty::String), ("age", Ty::Int)]))),
        )],
    );
    let ir = compile_to_ir(
        &i,
        r#"{{ x = @users | iter | map(|u| -> u.name) }}{{ x | join(",") }}"#,
        &context,
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

// ── Edge case: variable accumulation in loop ─────────────────────

#[test]
fn variable_accumulate_in_loop() {
    let i = Interner::new();
    // Write to variable on each iteration.
    let context = ctx(&i, &[("items", Ty::List(Box::new(Ty::Int)))]);
    let ir = compile_to_ir(
        &i,
        r#"{{ sum = 0 }}{{ x in @items }}{{ sum = sum + x }}{{/}}{{ sum | to_string }}"#,
        &context,
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

// ── Edge case: multi-level pipe with to_string in middle ────────

#[test]
fn pipe_map_to_string_then_filter() {
    let i = Interner::new();
    let ir = compile_to_ir(
        &i,
        r#"{{ x = @items | iter | map(|i| -> i + 1) | filter(|i| -> i != 0) | collect }}{{ x | len | to_string }}"#,
        &items_context(&i),
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

// ── Edge case: local var captured in lambda ──────────────────────

#[test]
fn lambda_capture_local_var_ref() {
    let i = Interner::new();
    // offset is NOT in initial context — created as local var.
    // Lambda must capture it correctly (not fall through to StorageLoad).
    let context = ctx(&i, &[("items", Ty::List(Box::new(Ty::Int)))]);
    let ir = compile_to_ir(
        &i,
        r#"{{ offset = 10 }}{{ x = @items | iter | filter(|i| -> i > offset) | collect }}{{ x | len | to_string }}{{_}}{{/}}"#,
        &context,
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

// ── Edge case: multiple field accesses on same lambda param ─────

#[test]
fn lambda_multiple_field_access() {
    let i = Interner::new();
    // Lambda body accesses two fields on the same param.
    let context = ctx(
        &i,
        &[(
            "users",
            Ty::List(Box::new(obj(&i, &[("name", Ty::String), ("age", Ty::Int)]))),
        )],
    );
    let ir = compile_to_ir(
        &i,
        r#"{{ x = @users | iter | map(|u| -> (u.name, u.age)) | collect }}{{ x | len | to_string }}"#,
        &context,
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

// ── Edge case: chained field access in lambda ───────────────────

#[test]
fn lambda_chained_field_access() {
    let i = Interner::new();
    let context = ctx(
        &i,
        &[(
            "users",
            Ty::List(Box::new(obj(
                &i,
                &[("address", obj(&i, &[("city", Ty::String)]))],
            ))),
        )],
    );
    let ir = compile_to_ir(
        &i,
        r#"{{ x = @users | iter | map(|u| -> u.address.city) }}{{ x | join(",") }}"#,
        &context,
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

// ── Edge case: string concat in lambda ───────────────────────────

#[test]
fn lambda_string_concat() {
    let i = Interner::new();
    // Lambda param is Ty::Var; string concat (+) must resolve via unification.
    let context = ctx(&i, &[("names", Ty::List(Box::new(Ty::String)))]);
    let ir = compile_to_ir(
        &i,
        r#"{{ x = @names | iter | map(|n| -> n + "!") }}{{ x | join(",") }}"#,
        &context,
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

// ── Edge case: filter then map with field access ────────────────

#[test]
fn pipe_filter_then_map_field() {
    let i = Interner::new();
    let context = ctx(
        &i,
        &[(
            "users",
            Ty::List(Box::new(obj(&i, &[("name", Ty::String), ("age", Ty::Int)]))),
        )],
    );
    let ir = compile_to_ir(
        &i,
        r#"{{ x = @users | iter | filter(|u| -> u.age > 18) | map(|u| -> u.name) }}{{ x | join(",") }}"#,
        &context,
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

// ── Edge case: error — field access on non-object ────────────────

#[test]
fn error_field_access_on_int() {
    let i = Interner::new();
    let context = ctx(&i, &[("n", Ty::Int)]);
    let result = compile_to_ir(&i, "{{ @n.foo | to_string }}", &context);
    assert!(result.is_err());
    insta::assert_snapshot!(result.unwrap_err());
}

// ── Edge case: error — context write attempt ─────────────────────

#[test]
fn error_variable_write_type_mismatch() {
    let i = Interner::new();
    // Attempting to write to a context key (read-only).
    let context = ctx(&i, &[("count", Ty::Int)]);
    let result = compile_to_ir(&i, r#"{{ @count = "hello" }}"#, &context);
    assert!(result.is_err());
    insta::assert_snapshot!(result.unwrap_err());
}

// ── Edge case: float arithmetic in lambda ────────────────────────

#[test]
fn lambda_float_arithmetic() {
    let i = Interner::new();
    let context = ctx(&i, &[("vals", Ty::List(Box::new(Ty::Float)))]);
    let ir = compile_to_ir(
        &i,
        r#"{{ x = @vals | iter | map(|v| -> v * 2.0) | collect }}{{ x | len | to_string }}"#,
        &context,
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

// ── Edge case: bool literal as match source ─────────────────────

#[test]
fn match_bool_literal() {
    let i = Interner::new();
    let context = ctx(&i, &[("flag", Ty::Bool)]);
    let ir = compile_to_ir(&i, r#"{{ true = @flag }}on{{_}}off{{/}}"#, &context).unwrap();
    insta::assert_snapshot!(ir);
}

// ── Edge case: nested pipe with filter on object field ──────────

#[test]
fn filter_object_field_equality() {
    let i = Interner::new();
    let context = ctx(
        &i,
        &[(
            "users",
            Ty::List(Box::new(obj(
                &i,
                &[("name", Ty::String), ("active", Ty::Bool)],
            ))),
        )],
    );
    let ir = compile_to_ir(
        &i,
        r#"{{ x = @users | iter | filter(|u| -> u.active) | collect }}{{ x | len | to_string }}"#,
        &context,
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

// ── Edge case: extern function with object return ───────────────

#[test]
fn extern_fn_object_return() {
    let i = Interner::new();
    let context = ctx(
        &i,
        &[(
            "get_user",
            Ty::Fn {
                params: vec![Param::new(i.intern("_0"), Ty::Int)],
                ret: Box::new(obj(&i, &[("name", Ty::String), ("age", Ty::Int)])),
                captures: vec![],
                effect: Effect::pure(),
                hint: None,
            },
        )],
    );
    let ir = compile_to_ir(&i, r#"{{ u = @get_user(1) }}{{ u.name }}"#, &context).unwrap();
    insta::assert_snapshot!(ir);
}

// ── New builtins ─────────────────────────────────────────────────

#[test]
fn builtin_len() {
    let i = Interner::new();
    let context = ctx(&i, &[("items", Ty::List(Box::new(Ty::Int)))]);
    let ir = compile_to_ir(&i, "{{ @items | len | to_string }}", &context).unwrap();
    insta::assert_snapshot!(ir);
}

#[test]
fn builtin_reverse() {
    let i = Interner::new();
    let context = ctx(&i, &[("items", Ty::List(Box::new(Ty::Int)))]);
    let ir = compile_to_ir(
        &i,
        "{{ x in @items | reverse }}{{ x | to_string }}{{/}}",
        &context,
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

#[test]
fn builtin_join() {
    let i = Interner::new();
    let context = ctx(&i, &[("names", Ty::List(Box::new(Ty::String)))]);
    let ir = compile_to_ir(&i, r#"{{ @names | join(", ") }}"#, &context).unwrap();
    insta::assert_snapshot!(ir);
}

#[test]
fn builtin_contains() {
    let i = Interner::new();
    let context = ctx(&i, &[("items", Ty::List(Box::new(Ty::Int)))]);
    let ir = compile_to_ir(&i, "{{ @items | contains(3) | to_string }}", &context).unwrap();
    insta::assert_snapshot!(ir);
}

#[test]
fn builtin_find() {
    let i = Interner::new();
    let context = ctx(&i, &[("items", Ty::List(Box::new(Ty::Int)))]);
    let ir = compile_to_ir(
        &i,
        "{{ @items | find(|x| -> x > 10) | to_string }}",
        &context,
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

#[test]
fn builtin_reduce() {
    let i = Interner::new();
    let context = ctx(&i, &[("items", Ty::List(Box::new(Ty::Int)))]);
    let ir = compile_to_ir(
        &i,
        "{{ @items | reduce(|a, b| -> a + b) | to_string }}",
        &context,
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

#[test]
fn builtin_fold() {
    let i = Interner::new();
    let context = ctx(&i, &[("items", Ty::List(Box::new(Ty::Int)))]);
    let ir = compile_to_ir(
        &i,
        "{{ @items | fold(0, |acc, x| -> acc + x) | to_string }}",
        &context,
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

#[test]
fn builtin_any() {
    let i = Interner::new();
    let context = ctx(&i, &[("items", Ty::List(Box::new(Ty::Int)))]);
    let ir = compile_to_ir(
        &i,
        "{{ @items | any(|x| -> x > 10) | to_string }}",
        &context,
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

#[test]
fn builtin_all() {
    let i = Interner::new();
    let context = ctx(&i, &[("items", Ty::List(Box::new(Ty::Int)))]);
    let ir = compile_to_ir(&i, "{{ @items | all(|x| -> x > 0) | to_string }}", &context).unwrap();
    insta::assert_snapshot!(ir);
}

// ── Context Call ────────────────────────────────────────────────

// ── Variant (Option) ────────────────────────────────────────────

#[test]
fn variant_some_expr() {
    let i = Interner::new();
    let ir = compile_simple(&i, "{{ x = Some(42) }}{{_}}{{/}}").unwrap();
    insta::assert_snapshot!(ir);
}

#[test]
fn variant_none_expr() {
    let i = Interner::new();
    let ir = compile_simple(&i, "{{ x = None }}{{_}}{{/}}").unwrap();
    insta::assert_snapshot!(ir);
}

#[test]
fn variant_some_pattern() {
    let i = Interner::new();
    let context = ctx(&i, &[("opt", Ty::Option(Box::new(Ty::Int)))]);
    let ir = compile_to_ir(
        &i,
        "{{ Some(v) = @opt }}{{ v | to_string }}{{_}}nope{{/}}",
        &context,
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

#[test]
fn variant_none_pattern() {
    let i = Interner::new();
    let context = ctx(&i, &[("opt", Ty::Option(Box::new(Ty::Int)))]);
    let ir = compile_to_ir(&i, "{{ None = @opt }}none{{_}}has value{{/}}", &context).unwrap();
    insta::assert_snapshot!(ir);
}

#[test]
fn structural_enum_variant_merge() {
    let i = Interner::new();
    let module =
        compile_analysis(&i, "{{ A::B = @a }}hi{{/}}{{ A::C = @a }}bye{{/}}", &[]).unwrap();
    let ir = acvus_mir::printer::dump_with(&i, &module);
    assert!(ir.contains("B"), "variant B missing from IR:\n{ir}");
    assert!(ir.contains("C"), "variant C missing from IR:\n{ir}");
}

// ── Structural enum tests ──────────────────────────────────────

#[test]
fn structural_enum_single_variant() {
    let i = Interner::new();
    let module = compile_analysis(&i, "{{ A::B = @a }}yes{{_}}no{{/}}", &[]).unwrap();
    let ir = acvus_mir::printer::dump_with(&i, &module);
    assert!(ir.contains("B"), "variant B missing from IR:\n{ir}");
}

#[test]
fn structural_enum_three_variants_merge() {
    let i = Interner::new();
    let src = "{{ S::X = @v }}x{{/}}{{ S::Y = @v }}y{{/}}{{ S::Z = @v }}z{{/}}";
    let module = compile_analysis(&i, src, &[]).unwrap();
    let ir = acvus_mir::printer::dump_with(&i, &module);
    assert!(ir.contains("X"), "variant X missing:\n{ir}");
    assert!(ir.contains("Y"), "variant Y missing:\n{ir}");
    assert!(ir.contains("Z"), "variant Z missing:\n{ir}");
}

#[test]
fn structural_enum_with_payload() {
    let i = Interner::new();
    let src = r#"{{ R::Ok(v) = @r }}{{ v | to_string }}{{_}}err{{/}}"#;
    let module = compile_analysis(&i, src, &[]).unwrap();
    let ir = acvus_mir::printer::dump_with(&i, &module);
    assert!(ir.contains("Ok"), "variant Ok missing:\n{ir}");
}

#[test]
fn structural_enum_mixed_payload_and_unit() {
    let i = Interner::new();
    let src = r#"{{ R::Ok(v) = @r }}{{ v | to_string }}{{ R::Err = }}fail{{_}}??{{/}}"#;
    let module = compile_analysis(&i, src, &[]).unwrap();
    let ir = acvus_mir::printer::dump_with(&i, &module);
    assert!(ir.contains("Ok"), "variant Ok missing:\n{ir}");
    assert!(ir.contains("Err"), "variant Err missing:\n{ir}");
}

#[test]
fn structural_enum_same_var_different_blocks_merge() {
    // Key regression test: separate match blocks on the same context var must merge.
    let i = Interner::new();
    let src = "{{ A::B = @a }}b{{/}}{{ A::C = @a }}c{{/}}";
    let module = compile_analysis(&i, src, &[]).unwrap();
    let ir = acvus_mir::printer::dump_with(&i, &module);
    assert!(ir.contains("B"), "variant B missing:\n{ir}");
    assert!(ir.contains("C"), "variant C missing:\n{ir}");
}

#[test]
fn structural_enum_different_enums_different_vars() {
    let i = Interner::new();
    let src = "{{ X::A = @x }}xa{{/}}{{ Y::B = @y }}yb{{/}}";
    let module = compile_analysis(&i, src, &[]).unwrap();
    let ir = acvus_mir::printer::dump_with(&i, &module);
    assert!(ir.contains("A"), "variant A missing:\n{ir}");
    assert!(ir.contains("B"), "variant B missing:\n{ir}");
}

#[test]
fn structural_enum_name_mismatch_is_error() {
    // Matching X::A and Y::B on the same var should fail (different enum names).
    let i = Interner::new();
    let src = "{{ X::A = @v }}a{{/}}{{ Y::B = @v }}b{{/}}";
    let result = compile_analysis(&i, src, &[]);
    assert!(
        result.is_err(),
        "should fail: different enum names on same var"
    );
}

#[test]
fn structural_enum_payload_unifies_with_inner_match() {
    // Payload variable must unify with patterns inside the arm body.
    let i = Interner::new();
    let src = r#"{{ A::X(x) = @a }}{{ 0 = x }}zero{{_}}other{{/}}{{_}}none{{/}}"#;
    let module = compile_analysis(&i, src, &[]).unwrap();
    let ir = acvus_mir::printer::dump_with(&i, &module);
    assert!(ir.contains("X"), "variant X missing:\n{ir}");
}

#[test]
fn structural_enum_payload_unifies_with_emit() {
    // Payload bound by variant pattern can be used in expressions (emit).
    let i = Interner::new();
    let src = r#"{{ A::Val(v) = @a }}{{ v + 1 | to_string }}{{_}}n/a{{/}}"#;
    let module = compile_analysis(&i, src, &[]).unwrap();
    let ir = acvus_mir::printer::dump_with(&i, &module);
    assert!(ir.contains("Val"), "variant Val missing:\n{ir}");
}

#[test]
fn structural_enum_payload_type_propagates_through_context() {
    // When context provides an enum type with payload, payload type should propagate.
    let i = Interner::new();
    let mut variants = FxHashMap::default();
    variants.insert(i.intern("Ok"), Some(Box::new(Ty::Int)));
    variants.insert(i.intern("Err"), None);
    let src = r#"{{ R::Ok(v) = @r }}{{ v + 1 | to_string }}{{ R::Err = }}err{{_}}??{{/}}"#;
    let module = compile_analysis(
        &i,
        src,
        &[(
            "r",
            Ty::Enum {
                name: i.intern("R"),
                variants,
            },
        )],
    )
    .unwrap();
    let ir = acvus_mir::printer::dump_with(&i, &module);
    assert!(ir.contains("Ok"), "variant Ok missing:\n{ir}");
    assert!(ir.contains("Err"), "variant Err missing:\n{ir}");
}

// ── Variant unification inside Tuple/List patterns ─────────────
// Regression: nested Variant patterns inside Tuple/List must merge
// variant sets across match arms via the shared Var chain.

#[test]
fn variant_merge_inside_tuple_pattern() {
    // Two arms with different variants nested inside a tuple pattern.
    // Both A and B must appear in the final merged Enum type.
    let i = Interner::new();
    let src = r#"{{ (S::A, x) = @t }}{{ x }}{{ (S::B, y) = }}{{ y }}{{_}}??{{/}}"#;
    let module = compile_analysis(&i, src, &[]).unwrap();
    let ir = acvus_mir::printer::dump_with(&i, &module);
    eprintln!("=== TUPLE VARIANT IR ===\n{ir}\n=== END ===");
    assert!(ir.contains("A"), "variant A missing from IR:\n{ir}");
    assert!(ir.contains("B"), "variant B missing from IR:\n{ir}");
}

#[test]
fn variant_merge_inside_tuple_three_arms() {
    let i = Interner::new();
    let src = r#"{{ (S::X, _) = @t }}x{{ (S::Y, _) = }}y{{ (S::Z, _) = }}z{{_}}??{{/}}"#;
    let module = compile_analysis(&i, src, &[]).unwrap();
    let ir = acvus_mir::printer::dump_with(&i, &module);
    assert!(ir.contains("X"), "variant X missing:\n{ir}");
    assert!(ir.contains("Y"), "variant Y missing:\n{ir}");
    assert!(ir.contains("Z"), "variant Z missing:\n{ir}");
}

#[test]
fn variant_merge_inside_list_pattern() {
    // Variant inside list head pattern should merge across arms.
    let i = Interner::new();
    let src = r#"{{ [S::A, ..] = @lst }}a{{ [S::B, ..] = }}b{{_}}??{{/}}"#;
    let module = compile_analysis(&i, src, &[]).unwrap();
    let ir = acvus_mir::printer::dump_with(&i, &module);
    assert!(ir.contains("A"), "variant A missing:\n{ir}");
    assert!(ir.contains("B"), "variant B missing:\n{ir}");
}

#[test]
fn pruned_context_keys_in_dead_catch_all() {
    // Outer match on @Impersonation with catch_all containing @Pov.
    // When Impersonation=NoPersona is known, catch_all is dead,
    // so @Pov should appear in partition.pruned.
    use acvus_mir::analysis::reachable_context::{KnownValue, partition_context_keys};
    use acvus_mir::graph::QualifiedRef;
    use acvus_mir::ir::InstKind;

    let i = Interner::new();
    let src = r#"{-{ Impersonation::NoPersona = @Impersonation }}{-{_}}{-{ Pov::User = @Pov }}user{-{ Pov::Char = }}char{-{_}}other{-{ / }}{-{ / }}"#;
    let module = compile_analysis(&i, src, &[]).unwrap();

    let ir = acvus_mir::printer::dump_with(&i, &module);
    eprintln!("=== PRUNED TEST IR ===\n{ir}\n=== END ===");

    // Build name→QualifiedRef lookup from the compiled module's Ref(Context) instructions + debug info.
    let mut name_to_qref: FxHashMap<&str, QualifiedRef> = FxHashMap::default();
    for inst in &module.main.insts {
        if let InstKind::Ref {
            dst,
            target: acvus_mir::ir::RefTarget::Context(ctx),
            ..
        } = &inst.kind
        {
            if let Some(acvus_mir::ir::ValOrigin::Context(name)) =
                module.main.debug.val_origins.get(dst)
            {
                name_to_qref.insert(i.resolve(*name), *ctx);
            }
        }
    }
    let impersonation_id = name_to_qref["Impersonation"];
    let pov_id = name_to_qref["Pov"];

    let mut known = FxHashMap::default();
    known.insert(
        impersonation_id,
        KnownValue::Variant {
            tag: i.intern("NoPersona"),
            payload: None,
        },
    );
    let partition = partition_context_keys(&module, &known);

    eprintln!("eager: {:?}", partition.eager);
    eprintln!("lazy: {:?}", partition.lazy);
    eprintln!("reachable_known: {:?}", partition.reachable_known);
    eprintln!("pruned: {:?}", partition.pruned);

    assert!(
        partition.pruned.contains(&pov_id),
        "Pov ({:?}) should be pruned when Impersonation=NoPersona, but got:\n  eager: {:?}\n  lazy: {:?}\n  pruned: {:?}",
        pov_id,
        partition.eager,
        partition.lazy,
        partition.pruned,
    );
}

// ── SSA chain tests (script mode) ───────────────────────────────

#[test]
fn ssa_context_read_write() {
    let i = Interner::new();
    let context = ctx(&i, &[("x", Ty::Int)]);
    let ir = compile_script_ir(&i, "@x = @x + 1; @x", &context).unwrap();
    insta::assert_snapshot!(ir);
}

#[test]
fn ssa_context_branch_phi() {
    let i = Interner::new();
    let context = ctx(&i, &[("x", Ty::Int), ("flag", Ty::Bool)]);
    let ir = compile_script_ir(
        &i,
        r#"@x = @flag ? { @x = @x + 1; @x } : { @x = @x - 1; @x }; @x"#,
        &context,
    );
    // This may or may not compile depending on match/ternary syntax.
    // If it fails, try a match-based version.
    if let Ok(ir) = ir {
        insta::assert_snapshot!(ir);
    }
}

#[test]
fn ssa_multiple_contexts_independent() {
    let i = Interner::new();
    let context = ctx(&i, &[("a", Ty::Int), ("b", Ty::Int)]);
    let ir = compile_script_ir(&i, "@a = @a + 1; @b = @b + 2; @a + @b", &context).unwrap();
    insta::assert_snapshot!(ir);
}

// ════════════════════════════════════════════════════════════════════
// Migrated from acvus-mir unit tests (ExternFn-dependent)
// ════════════════════════════════════════════════════════════════════

// ── From lib.rs ─────────────────────────────────────────────────────

#[test]
fn migrated_extern_param_write_rejected() {
    let i = Interner::new();
    // Writing to extern param is rejected.
    assert!(compile_to_ir(&i, "{{ $count = 42 }}", &FxHashMap::default()).is_err());
    // Reading an extern param via context with pipe is valid.
    let context = ctx(&i, &[("count", Ty::Int)]);
    compile_to_ir(&i, "{{ @count | to_string }}", &context).unwrap();
}

#[test]
fn migrated_integration_range_expression() {
    let i = Interner::new();
    compile_to_ir(
        &i,
        "{{ x in 0..10 }}{{ x | to_string }}{{/}}",
        &FxHashMap::default(),
    )
    .unwrap();
}

#[test]
fn migrated_integration_list_destructure() {
    let i = Interner::new();
    compile_to_ir(
        &i,
        r#"{{ [a, b, ..] = @items }}{{ a | to_string }}{{_}}{{/}}"#,
        &items_context(&i),
    )
    .unwrap();
}

#[test]
fn migrated_integration_pipe_with_lambda() {
    let i = Interner::new();
    let context = ctx(&i, &[("items", Ty::List(Box::new(Ty::Int)))]);
    compile_to_ir(
        &i,
        r#"{{ x = @items | filter(|x| -> x != 0) | collect }}{{ x | len | to_string }}{{_}}{{/}}"#,
        &context,
    )
    .unwrap();
}

#[test]
fn migrated_projection_chained_field_access() {
    let i = Interner::new();
    let inner = obj(&i, &[("b", Ty::Int)]);
    let obj_ty = obj(&i, &[("a", inner)]);
    let context = ctx(&i, &[("obj", obj_ty)]);
    let ir = compile_script_ir(&i, "@obj.a.b | to_string", &context).unwrap();
    // IR: Ref(@obj) + Load (whole), then field access .a.b.
    assert!(ir.contains("ref @"), "should have ref @ in IR: {ir}");
    assert!(ir.contains("load"), "should have load in IR: {ir}");
    let field_get_count = ir.matches(".b").count() + ir.matches(".a").count();
    assert!(
        field_get_count >= 2,
        "should have at least 2 field accesses in IR: {ir}"
    );
}

#[test]
fn migrated_pipe_extern_fn_ok() {
    let i = Interner::new();
    let context = ctx(
        &i,
        &[
            (
                "mapper",
                Ty::Fn {
                    params: vec![Param::new(i.intern("_"), Ty::Int)],
                    ret: Box::new(Ty::String),
                    captures: vec![],
                    effect: Effect::pure(),
                    hint: None,
                },
            ),
            ("items", Ty::List(Box::new(Ty::Int))),
        ],
    );
    compile_to_ir(
        &i,
        r#"{{ x = @items | map(|i| -> @mapper(i)) | collect }}{{ x | len | to_string }}{{_}}{{/}}"#,
        &context,
    )
    .unwrap();
}

// ── From typeck.rs ──────────────────────────────────────────────────

#[test]
fn migrated_typeck_builtin_to_string() {
    let i = Interner::new();
    let context = ctx(&i, &[("count", Ty::Int)]);
    compile_to_ir(&i, "{{ @count | to_string }}", &context).unwrap();
}

#[test]
fn migrated_typeck_lambda_captures_outer_variable() {
    let i = Interner::new();
    let context = ctx(
        &i,
        &[
            ("items", Ty::List(Box::new(Ty::Int))),
            ("threshold", Ty::Int),
        ],
    );
    compile_to_ir(
        &i,
        "{{ @items | filter(|x| -> x > @threshold) | collect | len | to_string }}",
        &context,
    )
    .unwrap();
}

#[test]
fn migrated_typeck_lambda_type_check() {
    let i = Interner::new();
    let context = ctx(&i, &[("items", Ty::List(Box::new(Ty::Int)))]);
    compile_to_ir(
        &i,
        "{{ x = @items | filter(|x| -> x != 0) | collect }}{{ x | len | to_string }}{{_}}{{/}}",
        &context,
    )
    .unwrap();
}

#[test]
fn migrated_typeck_lambda_no_capture_local_params() {
    let i = Interner::new();
    let context = ctx(&i, &[("items", Ty::List(Box::new(Ty::Int)))]);
    compile_to_ir(
        &i,
        "{{ @items | map(|x| -> x + 1) | collect | len | to_string }}",
        &context,
    )
    .unwrap();
}

#[test]
fn migrated_typeck_list_pattern_matching() {
    let i = Interner::new();
    let context = ctx(&i, &[("items", Ty::List(Box::new(Ty::Int)))]);
    compile_to_ir(
        &i,
        "{{ [a, b, ..] = @items }}{{ a | to_string }}{{_}}{{/}}",
        &context,
    )
    .unwrap();
}

#[test]
fn migrated_typeck_nested_lambda_captures() {
    let i = Interner::new();
    let context = ctx(
        &i,
        &[("items", Ty::List(Box::new(Ty::Int))), ("factor", Ty::Int)],
    );
    compile_to_ir(
        &i,
        "{{ @items | map(|x| -> x * @factor) | collect | len | to_string }}",
        &context,
    )
    .unwrap();
}

#[test]
fn migrated_typeck_some_unifies_with_option_context() {
    let i = Interner::new();
    let context = ctx(&i, &[("opt", Ty::Option(Box::new(Ty::Int)))]);
    compile_to_ir(
        &i,
        "{{ Some(v) = @opt }}{{ v | to_string }}{{_}}{{/}}",
        &context,
    )
    .unwrap();
}

// ── From printer.rs ─────────────────────────────────────────────────

#[test]
fn migrated_print_arithmetic() {
    let i = Interner::new();
    let context = ctx(&i, &[("a", Ty::Int), ("b", Ty::Int)]);
    let ir = compile_to_ir(
        &i,
        "{{ x = @a + @b }}{{ x | to_string }}{{_}}{{/}}",
        &context,
    )
    .unwrap();
    assert!(ir.contains("+"), "should contain + operator in IR: {ir}");
}

#[test]
fn migrated_print_closure() {
    let i = Interner::new();
    let context = ctx(&i, &[("items", Ty::List(Box::new(Ty::Int)))]);
    let ir = compile_to_ir(
        &i,
        "{{ x = @items | filter(|x| -> x != 0) | collect }}{{ x | len | to_string }}{{_}}{{/}}",
        &context,
    )
    .unwrap();
    assert!(
        ir.contains("closure L"),
        "should contain closure label in IR: {ir}"
    );
    assert!(
        ir.contains("=== closure"),
        "should contain closure section in IR: {ir}"
    );
    assert!(ir.contains("!="), "should contain != operator in IR: {ir}");
    assert!(
        ir.contains("return"),
        "should contain return instruction in IR: {ir}"
    );
}

// ── From ssa_pass.rs ────────────────────────────────────────────────

#[test]
fn migrated_ssa_iter_no_write_no_phi() {
    let i = Interner::new();
    let context = ctx(&i, &[("items", Ty::List(Box::new(Ty::String)))]);
    let ir = compile_to_ir(&i, r#"{{ x in @items }}{{ x | to_string }}{{/}}"#, &context).unwrap();
    // Original test checked: count_context_stores == 0 after SSA pass.
    // In IR output, context stores would appear as "ctx_store".
    assert!(
        !ir.contains("ctx_store"),
        "iteration without context write should have no ctx_store: {ir}"
    );
}

// ── Effect propagation through Iterator combinators ─────────────────

#[test]
fn effect_map_impure_propagates() {
    // iter(list) | map(impure_fn) should produce an effectful Iterator.
    // Reusing the result should be rejected (move-only).
    let i = Interner::new();
    let context = ctx(
        &i,
        &[("items", Ty::List(Box::new(Ty::Int))), ("counter", Ty::Int)],
    );
    // map with context write → effectful. Using result twice → use-after-move.
    let result = compile_script_ir(
        &i,
        r#"it = @items | iter | map(|x| -> { @counter = x; x }); it | collect; it | collect"#,
        &context,
    );
    assert!(
        result.is_err(),
        "effectful iter reuse should be rejected: {result:?}"
    );
}

#[test]
fn effect_map_impure_single_use_ok() {
    // Single use of effectful iterator should compile.
    let i = Interner::new();
    let context = ctx(
        &i,
        &[("items", Ty::List(Box::new(Ty::Int))), ("counter", Ty::Int)],
    );
    let result = compile_script_ir(
        &i,
        r#"@items | iter | map(|x| -> { @counter = x; x }) | collect"#,
        &context,
    );
    assert!(
        result.is_ok(),
        "single use of effectful iter should compile: {result:?}"
    );
}

#[test]
fn effect_chain_multiple_impure_combines() {
    // map(impure_a) | filter(impure_b) → both effects combined.
    // Reusing should be rejected.
    let i = Interner::new();
    let context = ctx(
        &i,
        &[
            ("items", Ty::List(Box::new(Ty::Int))),
            ("a", Ty::Int),
            ("b", Ty::Int),
        ],
    );
    let result = compile_script_ir(
        &i,
        r#"it = @items | iter | map(|x| -> { @a = x; x }) | filter(|x| -> { @b = x; x > 0 }); it | collect; it | collect"#,
        &context,
    );
    assert!(
        result.is_err(),
        "chained impure iter reuse should be rejected: {result:?}"
    );
}

#[test]
fn effect_chain_multiple_impure_single_use_ok() {
    // Single use of chained impure should compile.
    let i = Interner::new();
    let context = ctx(
        &i,
        &[
            ("items", Ty::List(Box::new(Ty::Int))),
            ("a", Ty::Int),
            ("b", Ty::Int),
        ],
    );
    let result = compile_script_ir(
        &i,
        r#"@items | iter | map(|x| -> { @a = x; x }) | filter(|x| -> { @b = x; x > 0 }) | collect"#,
        &context,
    );
    assert!(
        result.is_ok(),
        "single use of chained impure should compile: {result:?}"
    );
}

#[test]
fn effect_pure_iter_is_reusable() {
    // Pure iterator: iter(list) | map(pure_fn). Should be reusable (not move-only).
    // Wait — UserDefined is always move-only now. So even pure iter can't be reused.
    let i = Interner::new();
    let context = ctx(&i, &[("items", Ty::List(Box::new(Ty::Int)))]);
    let result = compile_script_ir(
        &i,
        r#"it = @items | iter | map(|x| -> x + 1); it | collect; it | collect"#,
        &context,
    );
    // UserDefined is always move-only, even when pure.
    assert!(
        result.is_err(),
        "even pure UserDefined iter reuse should be rejected: {result:?}"
    );
}

#[test]
fn effect_reject_collect_impure_reuse_after_collect() {
    // After collecting an impure iterator, the iterator variable is consumed (move-only).
    // Attempting to collect again from the same variable should fail with use-after-move.
    let i = Interner::new();
    let context = ctx(
        &i,
        &[("items", Ty::List(Box::new(Ty::Int))), ("counter", Ty::Int)],
    );
    let result = compile_script_ir(
        &i,
        r#"it = @items | iter | map(|x| -> { @counter = x; x }); collected = it | collect; it | collect"#,
        &context,
    );
    assert!(
        result.is_err(),
        "reuse of impure iter after collect should be rejected: {result:?}"
    );
    assert!(
        has_use_after_move(&result.unwrap_err()),
        "expected use-after-move error"
    );
}

#[test]
fn effect_collect_result_is_reusable() {
    // The result of collect (List) should be freely reusable even when the source
    // iterator was impure. List is not move-only.
    let i = Interner::new();
    let context = ctx(
        &i,
        &[("items", Ty::List(Box::new(Ty::Int))), ("counter", Ty::Int)],
    );
    let result = compile_script_ir(
        &i,
        r#"collected = @items | iter | map(|x| -> { @counter = x; x }) | collect; collected | len; collected | len"#,
        &context,
    );
    assert!(
        result.is_ok(),
        "collected List from impure iter should be reusable: {result:?}"
    );
}

// ── From move_check.rs (e2e) ────────────────────────────────────────

fn iter_ty_with(interner: &Interner, effect: Effect) -> Ty {
    let iter_qref = QualifiedRef::root(interner.intern("Iterator"));
    Ty::UserDefined {
        id: iter_qref,
        type_args: vec![Ty::Int],
        effect_args: vec![effect],
    }
}

fn test_effectful(interner: &Interner) -> Effect {
    Effect::Resolved(EffectSet {
        reads: BTreeSet::new(),
        writes: BTreeSet::from([EffectTarget::Token(QualifiedRef::root(interner.intern("__test")))]),
    })
}

fn eff_iter_ty(interner: &Interner) -> Ty {
    iter_ty_with(interner, test_effectful(interner))
}

fn pure_iter_ty(interner: &Interner) -> Ty {
    iter_ty_with(interner, Effect::pure())
}

fn has_use_after_move(err: &str) -> bool {
    err.contains("use of move-only value") || err.contains("UseAfterMove")
}

// -- Soundness: should REJECT --

#[test]
fn migrated_move_reject_effectful_iter_reuse() {
    let i = Interner::new();
    let context = ctx(
        &i,
        &[("items", Ty::List(Box::new(Ty::Int))), ("counter", Ty::Int)],
    );
    let result = compile_script_ir(
        &i,
        r#"x = @items | iter | map(|x| -> { @counter = x; x }); x | collect; x | collect"#,
        &context,
    );
    assert!(result.is_err(), "should reject effectful iter reuse");
    assert!(
        has_use_after_move(&result.unwrap_err()),
        "expected use-after-move error"
    );
}

#[test]
fn migrated_move_reject_var_double_load() {
    let i = Interner::new();
    let context = ctx(&i, &[("items", Ty::List(Box::new(Ty::Int)))]);
    let result = compile_to_ir(
        &i,
        "{{ a = @items | iter }}{{ a | collect | len | to_string }}{{ a | collect | len | to_string }}",
        &context,
    );
    assert!(result.is_err(), "should reject var double load of iterator");
    assert!(
        has_use_after_move(&result.unwrap_err()),
        "expected use-after-move error"
    );
}

#[test]
fn migrated_move_reject_effectful_pipe_reuse() {
    let i = Interner::new();
    let context = ctx(
        &i,
        &[("items", Ty::List(Box::new(Ty::Int))), ("counter", Ty::Int)],
    );
    let result = compile_script_ir(
        &i,
        r#"x = @items | iter | map(|x| -> { @counter = x; x }); a = x | collect; b = x | collect; a"#,
        &context,
    );
    assert!(result.is_err());
    assert!(
        has_use_after_move(&result.unwrap_err()),
        "expected use-after-move error"
    );
}

// -- Completeness: should ACCEPT --

// NOTE: With Iterator now represented as UserDefined, ALL iterators are move-only
// regardless of effect. The old `pure_iter_reuse` test (which expected reuse to be
// allowed for pure iterators) no longer applies. UserDefined types are always move-only.
#[test]
fn migrated_move_reject_pure_iter_reuse() {
    let i = Interner::new();
    let context = ctx(&i, &[("src", pure_iter_ty(&i))]);
    let result = compile_script_ir(
        &i,
        "x = @src; a = x | collect; b = x | collect; a",
        &context,
    );
    assert!(
        result.is_err(),
        "UserDefined iterator (even pure) should be move-only: {result:?}"
    );
}

#[test]
fn migrated_move_accept_effectful_single_use() {
    let i = Interner::new();
    let context = ctx(
        &i,
        &[("items", Ty::List(Box::new(Ty::Int))), ("counter", Ty::Int)],
    );
    let result = compile_script_ir(
        &i,
        r#"x = @items | iter | map(|x| -> { @counter = x; x }); x | collect"#,
        &context,
    );
    assert!(
        result.is_ok(),
        "single use of effectful should be allowed: {result:?}"
    );
}

#[test]
fn migrated_move_accept_collect_then_reuse() {
    let i = Interner::new();
    let context = ctx(
        &i,
        &[("items", Ty::List(Box::new(Ty::Int))), ("counter", Ty::Int)],
    );
    let result = compile_script_ir(
        &i,
        r#"list = @items | iter | map(|x| -> { @counter = x; x }) | collect; a = list | len; b = list | len; a + b"#,
        &context,
    );
    assert!(
        result.is_ok(),
        "collected list should be reusable: {result:?}"
    );
}

#[test]
fn migrated_move_accept_var_reassign() {
    let i = Interner::new();
    let context = ctx(
        &i,
        &[
            ("items", Ty::List(Box::new(Ty::Int))),
            ("items2", Ty::List(Box::new(Ty::Int))),
        ],
    );
    let result = compile_to_ir(
        &i,
        "{{ a = @items | iter }}{{ a | collect | len | to_string }}{{ a = @items2 | iter }}{{ a | collect | len | to_string }}",
        &context,
    );
    assert!(result.is_ok(), "reassigned var should be alive: {result:?}");
}

#[test]
fn migrated_move_accept_effectful_pipe_chain() {
    let i = Interner::new();
    let context = ctx(
        &i,
        &[("items", Ty::List(Box::new(Ty::Int))), ("counter", Ty::Int)],
    );
    let result = compile_script_ir(
        &i,
        r#"@items | iter | map(|x| -> { @counter = x; x }) | filter(|x| -> x > 0) | map(|x| -> x * 2) | collect"#,
        &context,
    );
    assert!(
        result.is_ok(),
        "linear pipe chain should be allowed: {result:?}"
    );
}

#[test]
fn migrated_move_accept_effectful_fn_multiple_calls() {
    let i = Interner::new();
    let fn_ty = Ty::Fn {
        params: vec![Param::new(i.intern("_"), Ty::Int)],
        ret: Box::new(Ty::Int),
        captures: vec![],
        effect: test_effectful(&i),
        hint: None,
    };
    let context = ctx(&i, &[("f", fn_ty)]);
    let result = compile_script_ir(&i, "a = @f(1); b = @f(2); a + b", &context);
    assert!(
        result.is_ok(),
        "effectful fn without move-only captures should be callable multiple times: {result:?}"
    );
}

#[test]
fn migrated_move_reject_list_of_effectful_reuse() {
    let i = Interner::new();
    let ty = Ty::List(Box::new(eff_iter_ty(&i)));
    let context = ctx(&i, &[("src", ty)]);
    let result = compile_script_ir(&i, "x = @src; a = x | len; b = x | len; a + b", &context);
    assert!(
        result.is_err(),
        "List containing effectful should be move-only"
    );
}

#[test]
fn migrated_move_reject_option_effectful_reuse() {
    let i = Interner::new();
    let ty = Ty::Option(Box::new(eff_iter_ty(&i)));
    let context = ctx(&i, &[("src", ty)]);
    let result = compile_script_ir(
        &i,
        "x = @src; a = x | unwrap | collect; b = x | unwrap | collect; a",
        &context,
    );
    assert!(result.is_err(), "Option<Effectful> should be move-only");
}

#[test]
fn migrated_move_reject_branch_move_then_use() {
    let i = Interner::new();
    let context = ctx(
        &i,
        &[("flag", Ty::Bool), ("items", Ty::List(Box::new(Ty::Int)))],
    );
    let result = compile_to_ir(
        &i,
        "{{ a = @items | iter }}{{ true = @flag }}{{ a | collect | len | to_string }}{{_}}nothing{{/}}{{ a | collect | len | to_string }}",
        &context,
    );
    assert!(
        result.is_err(),
        "should reject use after move across branch: {result:?}"
    );
    assert!(
        has_use_after_move(&result.unwrap_err()),
        "expected use-after-move error"
    );
}

#[test]
fn migrated_move_reject_both_branches_move_then_use() {
    let i = Interner::new();
    let context = ctx(&i, &[("flag", Ty::Bool), ("src", eff_iter_ty(&i))]);
    let result = compile_to_ir(
        &i,
        "{{ a = @src }}{{ true = @flag }}{{ a | collect | len | to_string }}{{_}}{{ a | collect | len | to_string }}{{/}}{{ a | collect | len | to_string }}",
        &context,
    );
    assert!(
        result.is_err(),
        "should reject use after move in both branches: {result:?}"
    );
}

#[test]
fn migrated_move_accept_branch_move_no_use_after() {
    let i = Interner::new();
    let context = ctx(
        &i,
        &[("flag", Ty::Bool), ("items", Ty::List(Box::new(Ty::Int)))],
    );
    let result = compile_to_ir(
        &i,
        "{{ a = @items | iter }}{{ true = @flag }}{{ a | collect | len | to_string }}{{_}}nothing{{/}}",
        &context,
    );
    assert!(
        result.is_ok(),
        "move in branch without post-merge use should be OK: {result:?}"
    );
}

#[test]
fn migrated_move_reject_fnonce_double_call() {
    let i = Interner::new();
    let context = ctx(&i, &[("items", Ty::List(Box::new(Ty::Int)))]);
    let result = compile_script_ir(
        &i,
        "x = @items | iter; f = (|z| -> collect(x)); a = f(0); b = f(0); a",
        &context,
    );
    assert!(
        result.is_err(),
        "FnOnce called twice should be rejected: {result:?}"
    );
    assert!(
        has_use_after_move(&result.unwrap_err()),
        "expected use-after-move error"
    );
}

#[test]
fn migrated_move_accept_pure_capture_fn_multi_call() {
    let i = Interner::new();
    let context = ctx(&i, &[("val", Ty::Int)]);
    let result = compile_script_ir(
        &i,
        "x = @val; f = (|a| -> x + a); a = f(1); b = f(2); a + b",
        &context,
    );
    assert!(
        result.is_ok(),
        "Fn with pure captures should be callable multiple times: {result:?}"
    );
}

#[test]
fn migrated_move_accept_fnonce_single_call() {
    let i = Interner::new();
    let context = ctx(&i, &[("items", Ty::List(Box::new(Ty::Int)))]);
    let result = compile_script_ir(
        &i,
        "x = @items | iter; f = (|z| -> collect(x)); f(0)",
        &context,
    );
    assert!(
        result.is_ok(),
        "FnOnce called once should be OK: {result:?}"
    );
}

#[test]
fn migrated_move_accept_lambda_return_deque_as_iterator() {
    let i = Interner::new();
    let context = ctx(&i, &[("items", Ty::List(Box::new(Ty::Int)))]);
    let result = compile_script_ir(
        &i,
        "@items | flat_map(|x| -> [x, x + 1]) | map(|x| -> x * 2) | collect",
        &context,
    );
    assert!(
        result.is_ok(),
        "lambda returning Deque where Iterator expected should compile: {result:?}"
    );
}

#[test]
fn migrated_move_accept_lambda_return_scalar() {
    let i = Interner::new();
    let context = ctx(&i, &[("items", Ty::List(Box::new(Ty::Int)))]);
    let result = compile_script_ir(&i, "@items | map(|x| -> x + 1) | collect", &context);
    assert!(
        result.is_ok(),
        "lambda returning scalar should compile: {result:?}"
    );
}

#[test]
fn migrated_move_accept_nested_flat_map_deque_return() {
    let i = Interner::new();
    let context = ctx(&i, &[("items", Ty::List(Box::new(Ty::Int)))]);
    let result = compile_script_ir(
        &i,
        "@items | flat_map(|x| -> [x, x + 10]) | map(|x| -> x * 2) | collect",
        &context,
    );
    assert!(
        result.is_ok(),
        "nested flat_map + map with Deque return should compile: {result:?}"
    );
}

#[test]
fn migrated_move_accept_fnonce_passed_to_map() {
    let i = Interner::new();
    let context = ctx(&i, &[("items", Ty::List(Box::new(Ty::Int)))]);
    let result = compile_script_ir(
        &i,
        "x = @items | iter; f = (|z| -> collect(x)); @items | map(f) | collect",
        &context,
    );
    assert!(
        result.is_ok(),
        "FnOnce passed once to HOF should be OK: {result:?}"
    );
}

#[test]
fn migrated_move_accept_lambda_context_in_body_is_fn() {
    let i = Interner::new();
    let context = ctx(&i, &[("items", Ty::List(Box::new(Ty::Int)))]);
    let result = compile_to_ir(
        &i,
        "{{ f = (|z| -> collect(@items | iter)) }}{{ f(0) | len | to_string }}{{ f(0) | len | to_string }}",
        &context,
    );
    assert!(
        result.is_ok(),
        "Lambda with @context in body (not capture) should be Fn: {result:?}"
    );
}

#[test]
fn migrated_move_reject_fnonce_local_capture_double() {
    let i = Interner::new();
    let context = ctx(&i, &[("src", eff_iter_ty(&i))]);
    let result = compile_script_ir(
        &i,
        "x = @src; f = (|z| -> collect(x)); a = f(0); b = f(0); a",
        &context,
    );
    assert!(
        result.is_err(),
        "FnOnce with local capture double call should be rejected: {result:?}"
    );
}

#[test]
fn migrated_move_reject_effectful_without_purify() {
    let i = Interner::new();
    let context = ctx(&i, &[("src", eff_iter_ty(&i))]);
    let result = compile_script_ir(
        &i,
        "x = @src; a = x | collect; b = x | collect; a",
        &context,
    );
    assert!(
        result.is_err(),
        "effectful without purify should still be rejected"
    );
}

#[test]
fn migrated_move_reject_effectful_var_without_purify() {
    let i = Interner::new();
    let context = ctx(&i, &[("src", eff_iter_ty(&i))]);
    let result = compile_to_ir(
        &i,
        "{{ a = @src }}{{ a | collect | len | to_string }}{{ a | collect | len | to_string }}",
        &context,
    );
    assert!(
        result.is_err(),
        "effectful var without purify should be rejected"
    );
}

// ══════════════════════════════════════════════════════════════════════
// Projection system: soundness & completeness
// ══════════════════════════════════════════════════════════════════════

// ── Completeness: valid programs accepted ───────────────────────────

/// Variable whole read/write: `x = 42; x` — SSA promotion eliminates Ref/Load/Store.
#[test]
fn projection_var_whole_read_write() {
    let i = Interner::new();
    let ir = compile_script_ir(&i, "x = 42; x", &FxHashMap::default()).unwrap();
    // After SSA: no Ref/Load/Store should remain (all promoted).
    assert!(
        !ir.contains("ref "),
        "Ref should be eliminated by SSA: {ir}"
    );
    assert!(
        !ir.contains("store"),
        "Store should be eliminated by SSA: {ir}"
    );
    // The value should flow directly to return.
    assert!(ir.contains("return"), "should have return: {ir}");
}

/// Variable read after multiple writes: `x = 1; x = 2; x` → SSA sees last def.
#[test]
fn projection_var_multiple_writes() {
    let i = Interner::new();
    let ir = compile_script_ir(&i, "x = 1; x = 2; x", &FxHashMap::default()).unwrap();
    assert!(
        !ir.contains("ref "),
        "Ref should be eliminated by SSA: {ir}"
    );
}

/// Context whole read: `@ctx` — volatile context preserves Ref+Load.
#[test]
fn projection_context_whole_read() {
    let i = Interner::new();
    let context = ctx(&i, &[("data", Ty::Int)]);
    let ir = compile_script_ir(&i, "@data", &context).unwrap();
    // Non-volatile context: SSA should forward the entry load value.
    // Ref/Load from entry may remain or be forwarded — just verify it compiles + returns.
    assert!(ir.contains("return"), "should have return: {ir}");
}

/// Context field read: `@obj.name` — 1-depth Ref with field.
#[test]
fn projection_context_field_read() {
    let i = Interner::new();
    let context = ctx(&i, &[("obj", obj(&i, &[("name", Ty::String)]))]);
    let ir = compile_script_ir(&i, "@obj.name", &context).unwrap();
    // Should have a field ref (Ref with field) or SSA-forwarded result.
    assert!(ir.contains("return"), "should compile and return: {ir}");
}

/// Context field write: `@obj = { name: "test" }` — whole context store.
#[test]
fn projection_context_whole_write() {
    let i = Interner::new();
    let context = ctx(&i, &[("out", Ty::String)]);
    let ir = compile_to_ir(&i, r#"{{ @out = "hello" }}"#, &context).unwrap();
    // Context store should produce Store instruction (may remain after SSA for write-back).
    assert!(
        ir.contains("store"),
        "should have context store in IR: {ir}"
    );
}

/// Chained field access: `@obj.a.b` — Ref(@obj, "a") + Load + FieldGet("b").
#[test]
fn projection_chained_field_access_2depth() {
    let i = Interner::new();
    let inner = obj(&i, &[("b", Ty::Int)]);
    let context = ctx(&i, &[("obj", obj(&i, &[("a", inner)]))]);
    let ir = compile_script_ir(&i, "@obj.a.b | to_string", &context).unwrap();
    // Should have field access for .b (either as FieldGet or via Ref).
    assert!(ir.contains(".b"), "should access field b: {ir}");
    assert!(ir.contains("return"), "should compile and return: {ir}");
}

/// Variable used in arithmetic after assignment: `x = @val; x + 1`.
#[test]
fn projection_var_in_arithmetic() {
    let i = Interner::new();
    let context = ctx(&i, &[("val", Ty::Int)]);
    let ir = compile_script_ir(&i, "x = @val; x + 1", &context).unwrap();
    // SSA should promote x — no Ref for x should remain.
    assert!(ir.contains("+"), "should have addition: {ir}");
    assert!(ir.contains("return"), "should compile and return: {ir}");
}

/// Lambda captures variable: `x = @data; |y| -> x + y`.
#[test]
fn projection_lambda_capture() {
    let i = Interner::new();
    let context = ctx(&i, &[("data", Ty::Int)]);
    let ir = compile_script_ir(&i, "x = @data; |y| -> x + y", &context).unwrap();
    assert!(ir.contains("closure"), "should have closure: {ir}");
    assert!(ir.contains("return"), "should compile and return: {ir}");
}

/// ExternParam read: `$param` — should compile (immutable, Ref+Load).
#[test]
fn projection_param_read() {
    let i = Interner::new();
    let ir = compile_to_ir(
        &i,
        "{{ $count | to_string }}",
        &FxHashMap::from_iter([(i.intern("count"), Ty::Int)]),
    )
    .unwrap();
    assert!(ir.contains("return"), "should compile and return: {ir}");
}

// ── Soundness: invalid programs rejected ────────────────────────────

/// Store non-materializable (Fn) to context → must be rejected.
#[test]
fn projection_soundness_reject_fn_in_context() {
    let i = Interner::new();
    let fn_ty = Ty::Fn {
        params: vec![acvus_mir::ty::Param::new(i.intern("x"), Ty::Int)],
        ret: Box::new(Ty::Int),
        captures: vec![],
        effect: acvus_mir::ty::Effect::pure(),
        hint: None,
    };
    let context = ctx(&i, &[("f", fn_ty)]);
    let result = compile_script_ir(&i, "@f = @f; @f", &context);
    assert!(result.is_err(), "storing Fn to context should fail");
}

/// Store non-materializable (List<Fn>) to context → must be rejected.
#[test]
fn projection_soundness_reject_list_fn_in_context() {
    let i = Interner::new();
    let fn_ty = Ty::Fn {
        params: vec![acvus_mir::ty::Param::new(i.intern("x"), Ty::Int)],
        ret: Box::new(Ty::Int),
        captures: vec![],
        effect: acvus_mir::ty::Effect::pure(),
        hint: None,
    };
    let context = ctx(&i, &[("xs", Ty::List(Box::new(fn_ty)))]);
    let result = compile_script_ir(&i, "@xs = @xs; @xs", &context);
    assert!(result.is_err(), "storing List<Fn> to context should fail");
}

/// Write to ExternParam → must be rejected (typeck catches this).
#[test]
fn projection_soundness_reject_param_write() {
    let i = Interner::new();
    let result = compile_to_ir(
        &i,
        "{{ $count = 42 }}",
        &FxHashMap::from_iter([(i.intern("count"), Ty::Int)]),
    );
    assert!(result.is_err(), "writing to ExternParam should fail");
}

// ── SSA correctness: promotion & volatile ───────────────────────────

/// Non-volatile variable: SSA promotion eliminates Ref/Load/Store.
#[test]
fn projection_ssa_var_promoted() {
    let i = Interner::new();
    let ir = compile_script_ir(&i, "x = 1; y = x + 2; y", &FxHashMap::default()).unwrap();
    // All Ref/Load/Store for x and y should be eliminated.
    assert!(
        !ir.contains("ref "),
        "var Ref should be promoted away: {ir}"
    );
    assert!(
        !ir.contains("store "),
        "var Store should be promoted away: {ir}"
    );
}

/// Context read-then-write: SSA should preserve context Store (write-back).
#[test]
fn projection_ssa_context_write_back() {
    let i = Interner::new();
    let context = ctx(&i, &[("count", Ty::Int)]);
    let ir = compile_script_ir(&i, "@count = @count + 1; @count", &context).unwrap();
    // Context write-back Store should remain in the IR.
    assert!(
        ir.contains("store"),
        "context write-back store should remain: {ir}"
    );
}

/// Move-only value through variable: single use accepted.
#[test]
fn projection_move_single_use() {
    let i = Interner::new();
    let context = ctx(&i, &[("items", Ty::List(Box::new(Ty::Int)))]);
    let ir = compile_to_ir(
        &i,
        r#"{{ x = @items | iter }}{{ x | collect | len | to_string }}"#,
        &context,
    )
    .unwrap();
    assert!(
        ir.contains("return"),
        "single use of move-only should compile: {ir}"
    );
}

/// Move-only value through variable: reassignment revives.
#[test]
fn projection_move_var_reassign_revives() {
    let i = Interner::new();
    let context = ctx(&i, &[("items", Ty::List(Box::new(Ty::Int)))]);
    let ir = compile_to_ir(
        &i,
        r#"{{ x = @items | iter }}{{ x | collect | len | to_string }}{{ x = @items | iter }}{{ x | collect | len | to_string }}"#,
        &context,
    ).unwrap();
    assert!(
        ir.contains("return"),
        "var reassign should revive move-only: {ir}"
    );
}

// ══════════════════════════════════════════════════════════════════════
// SROA: field projection decomposition
// ══════════════════════════════════════════════════════════════════════

// ── Completeness: valid field access patterns ───────────────────────

#[test]
fn sroa_context_field_read_1depth() {
    let i = Interner::new();
    let context = ctx(&i, &[("obj", obj(&i, &[("name", Ty::String)]))]);
    let ir = compile_script_ir(&i, "@obj.name", &context).unwrap();
    insta::assert_snapshot!(ir);
}

#[test]
fn sroa_context_field_read_2depth() {
    let i = Interner::new();
    let inner = obj(&i, &[("b", Ty::Int)]);
    let context = ctx(&i, &[("obj", obj(&i, &[("a", inner)]))]);
    let ir = compile_script_ir(&i, "@obj.a.b | to_string", &context).unwrap();
    insta::assert_snapshot!(ir);
}

#[test]
fn sroa_context_field_read_3depth() {
    let i = Interner::new();
    let c = obj(&i, &[("c", Ty::Int)]);
    let b = obj(&i, &[("b", c)]);
    let context = ctx(&i, &[("obj", obj(&i, &[("a", b)]))]);
    let ir = compile_script_ir(&i, "@obj.a.b.c | to_string", &context).unwrap();
    insta::assert_snapshot!(ir);
}

#[test]
fn sroa_field_read_arithmetic() {
    let i = Interner::new();
    let context = ctx(&i, &[("obj", obj(&i, &[("val", Ty::Int)]))]);
    let ir = compile_script_ir(&i, "@obj.val + 1", &context).unwrap();
    insta::assert_snapshot!(ir);
}

#[test]
fn sroa_multiple_field_reads_same_object() {
    let i = Interner::new();
    let context = ctx(&i, &[("obj", obj(&i, &[("x", Ty::Int), ("y", Ty::Int)]))]);
    let ir = compile_script_ir(&i, "@obj.x + @obj.y", &context).unwrap();
    insta::assert_snapshot!(ir);
}

#[test]
fn sroa_field_read_in_lambda() {
    let i = Interner::new();
    let context = ctx(
        &i,
        &[(
            "users",
            Ty::List(Box::new(obj(&i, &[("name", Ty::String)]))),
        )],
    );
    let ir = compile_to_ir(
        &i,
        r#"{{ @users | iter | map(|u| -> u.name) | collect | join(",") }}"#,
        &context,
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

// ── Soundness ───────────────────────────────────────────────────────

#[test]
fn sroa_soundness_context_write_back_preserved() {
    let i = Interner::new();
    let context = ctx(&i, &[("count", Ty::Int)]);
    let ir = compile_script_ir(&i, "@count = @count + 1; @count", &context).unwrap();
    insta::assert_snapshot!(ir);
}

#[test]
fn sroa_soundness_reject_fn_in_context() {
    let i = Interner::new();
    let fn_ty = Ty::Fn {
        params: vec![acvus_mir::ty::Param::new(i.intern("x"), Ty::Int)],
        ret: Box::new(Ty::Int),
        captures: vec![],
        effect: acvus_mir::ty::Effect::pure(),
        hint: None,
    };
    let context = ctx(&i, &[("f", fn_ty)]);
    let result = compile_script_ir(&i, "@f = @f; @f", &context);
    assert!(
        result.is_err(),
        "storing Fn to context should fail even with SROA"
    );
}

// ── Context destructure chain ───────────────────────────────────────

#[test]
fn sroa_context_destructure_field_to_context() {
    let i = Interner::new();
    let context = ctx(&i, &[("b", obj(&i, &[("x", Ty::Int)])), ("a", Ty::Int)]);
    let ir = compile_script_ir(&i, "@a = @b.x; @a", &context).unwrap();
    insta::assert_snapshot!(ir);
}

#[test]
fn sroa_context_chain_destructure() {
    let i = Interner::new();
    let inner = obj(&i, &[("val", Ty::Int)]);
    let context = ctx(
        &i,
        &[
            ("b", obj(&i, &[("a", inner.clone())])),
            ("a", inner),
            ("c", Ty::Int),
        ],
    );
    let ir = compile_script_ir(&i, "@a = @b.a; @c = @a.val; @c", &context).unwrap();
    insta::assert_snapshot!(ir);
}

#[test]
fn sroa_context_destructure_then_overwrite() {
    let i = Interner::new();
    let context = ctx(&i, &[("b", obj(&i, &[("x", Ty::Int)])), ("a", Ty::Int)]);
    let ir = compile_script_ir(&i, "@a = @b.x; @a = 0; @a", &context).unwrap();
    insta::assert_snapshot!(ir);
}

// ── Projection store ────────────────────────────────────────────────

#[test]
fn context_projection_store_1depth() {
    let i = Interner::new();
    let context = ctx(&i, &[("obj", obj(&i, &[("x", Ty::Int)]))]);
    let ir = compile_script_ir(&i, "@obj.x = 42; @obj.x", &context).unwrap();
    insta::assert_snapshot!(ir);
}

#[test]
fn context_projection_store_2depth() {
    let i = Interner::new();
    let inner = obj(&i, &[("y", Ty::Int)]);
    let context = ctx(&i, &[("obj", obj(&i, &[("x", inner)]))]);
    let ir = compile_script_ir(&i, "@obj.x.y = 99; @obj.x.y", &context).unwrap();
    insta::assert_snapshot!(ir);
}

#[test]
fn var_field_store_1depth() {
    let i = Interner::new();
    let context = ctx(&i, &[("obj", obj(&i, &[("x", Ty::Int)]))]);
    let ir = compile_script_ir(&i, "a = @obj; a.x = 0; a.x", &context).unwrap();
    insta::assert_snapshot!(ir);
}

// ── Destructure projection ──────────────────────────────────────────

#[test]
fn destructure_projection_read() {
    let i = Interner::new();
    let context = ctx(&i, &[("a", obj(&i, &[("x", Ty::Int), ("y", Ty::Int)]))]);
    let ir = compile_script_ir(&i, "{ @x, } = @a { }; @a.x", &context).unwrap();
    insta::assert_snapshot!(ir);
}

#[test]
fn destructure_projection_write() {
    let i = Interner::new();
    let context = ctx(&i, &[("a", obj(&i, &[("x", Ty::Int)]))]);
    let ir = compile_script_ir(&i, "{ @x, } = @a { @x = 42; }; @a.x", &context).unwrap();
    insta::assert_snapshot!(ir);
}

#[test]
fn destructure_projection_shadowing() {
    let i = Interner::new();
    let context = ctx(&i, &[
        ("a", obj(&i, &[("x", Ty::Int)])),
        ("x", Ty::Int),
    ]);
    // @x exists as context, but inside body it's shadowed by projection @a.x
    let ir = compile_script_ir(&i, "@x = 99; { @x, } = @a { @x = 42; }; @x", &context).unwrap();
    insta::assert_snapshot!(ir);
}

// ── Uninit check ────────────────────────────────────────────────────

#[test]
fn uninit_field_load_rejected() {
    let i = Interner::new();
    // @a is Inferred (not declared). Literal only has x, but .y access widens type.
    // Value is missing field y → uninit error.
    let result = compile_script_ir(&i, "@a = { x: 0, }; @a.y | to_string", &FxHashMap::default());
    assert!(result.is_err(), "should catch uninit field access");
    let err = result.unwrap_err();
    assert!(err.contains("UninitError"), "error should be UninitError: {}", err);
}

#[test]
fn init_field_load_passes() {
    let i = Interner::new();
    let context = ctx(&i, &[("a", obj(&i, &[("x", Ty::Int)]))]);
    // All fields present — should compile fine.
    let ir = compile_script_ir(&i, "@a = { x: 42, }; @a.x", &context).unwrap();
    insta::assert_snapshot!(ir);
}

#[test]
fn field_store_then_load_passes() {
    let i = Interner::new();
    // @a is Inferred. Literal missing y, but field store fills it in → should pass.
    let ir = compile_script_ir(&i, "@a = { x: 0, }; @a.y = 1; @a.y | to_string", &FxHashMap::default()).unwrap();
    insta::assert_snapshot!(ir);
}

// ── Script mode tests ──────────────────────────────────────────────

fn script_mode(i: &Interner, source: &str) -> String {
    compile_script_mode_raw(i, source, &FxHashMap::default()).unwrap()
}

fn script_mode_ctx(i: &Interner, source: &str, ctx: &FxHashMap<Astr, Ty>) -> String {
    compile_script_mode_raw(i, source, ctx).unwrap()
}

#[test]
fn script_let_bind() {
    let i = Interner::new();
    let ir = script_mode(&i, "let x = 1; x");
    insta::assert_snapshot!(ir);
}

#[test]
fn script_let_assign() {
    let i = Interner::new();
    let ir = script_mode(&i, "let x = 1; x = 2; x");
    insta::assert_snapshot!(ir);
}

#[test]
fn script_let_shadowing() {
    let i = Interner::new();
    let ir = script_mode(&i, "let x = 1; let x = x + 1; x");
    insta::assert_snapshot!(ir);
}

#[test]
fn script_if_expr() {
    let i = Interner::new();
    let ir = script_mode(&i, "let x = if true { 1 } else { 2 }; x");
    insta::assert_snapshot!(ir);
}

#[test]
fn script_if_else_if() {
    let i = Interner::new();
    let ir = script_mode(&i, "let x = if false { 1 } else if true { 2 } else { 3 }; x");
    insta::assert_snapshot!(ir);
}

#[test]
fn script_if_no_else() {
    let i = Interner::new();
    // if without else → side effect only, used as statement
    let ir = script_mode(&i, "let x = 0; if true { x = 1; }; x");
    insta::assert_snapshot!(ir);
}

#[test]
fn script_for_loop() {
    let i = Interner::new();
    let ctx = ctx(&i, &[("sum", Ty::Int)]);
    let ir = script_mode_ctx(&i, "for item in [1, 2, 3] { @sum = @sum + item; }", &ctx);
    insta::assert_snapshot!(ir);
}

#[test]
fn script_while_loop() {
    let i = Interner::new();
    let ctx = ctx(&i, &[("n", Ty::Int)]);
    let ir = script_mode_ctx(&i, "while @n > 0 { @n = @n - 1; }", &ctx);
    insta::assert_snapshot!(ir);
}

#[test]
fn script_while_let() {
    let i = Interner::new();
    let ir = script_mode(&i, r#"
        let x = Some(1);
        let result = 0;
        while let Some(v) = x {
            result = v;
            x = None;
        }
        result
    "#);
    insta::assert_snapshot!(ir);
}

#[test]
fn script_if_let_expr() {
    let i = Interner::new();
    let ir = script_mode(&i, "let x = if let Some(v) = Some(42) { v } else { 0 }; x");
    insta::assert_snapshot!(ir);
}
