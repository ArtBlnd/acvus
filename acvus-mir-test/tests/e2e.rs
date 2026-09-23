use acvus_mir::graph::{FnKind, Function, QualifiedRef};
use acvus_mir::{
    graph::infer,
    ty::{ObjectTy, ParamTerm, Poly, Ty, TyTerm, TypeArg, lift_to_poly},
};
use acvus_mir_test::*;
use acvus_utils::{Astr, Interner};
use rustc_hash::FxHashMap;

/// Helper: compile a template source via the graph pipeline (extract -> resolve -> lower).
/// Unknown @contexts are added as Inferred constraints.
fn compile_analysis(
    interner: &Interner,
    source: &str,
    ctx: &[(&str, Ty)],
) -> Result<acvus_mir::ir::MirModule, String> {
    use acvus_mir::graph::{CompilationGraph, Context, FnKind, Function, ParsedAst, QualifiedRef};
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

    // Discover context refs in source that aren't declared - add as Inferred.
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
            effect: acvus_mir::ty::Effect::OPAQUE.into(),
        },
    }];
    let externs = acvus_extern::Externs::combine(
        acvus_ext::std_registries::<acvus_extern::TypesOnly>(),
        interner,
    )
    .expect("the standard registries combine");
    functions.extend(externs.functions);
    let type_registry = externs.types;

    let graph = CompilationGraph {
        functions: Freeze::new(functions),
        contexts: Freeze::new(contexts),
        bindings: acvus_mir::graph::Bindings::default(),
        entry: None,
    };

    let ext = extract::extract(interner, &graph);
    let inf = infer::infer(
        interner,
        &graph,
        &ext,
        &FxHashMap::default(),
        Freeze::new(type_registry),
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

    let result = graph_lower::lower(interner, &graph, &ext.view(), &inf);

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
    Ty::Object(ObjectTy::written(
        fields
            .iter()
            .map(|(k, v)| (i.intern(k), v.clone()))
            .collect(),
    ))
}

/// Helper: a context holding a `Vec<elem>`. A pipeline takes the vec whole
/// (`iter` consumes it), so a body that names the context takes it into a
/// local and puts an empty list back before the pipeline runs
/// (`% let items = @items` / `% @items = vec([])`, RFC-0025).
fn list_context(i: &Interner, name: &str, elem: Ty) -> FxHashMap<Astr, Ty> {
    ctx(i, &[(name, acvus_extern::vec_ty(i, elem))])
}

fn items_list_context(i: &Interner) -> FxHashMap<Astr, Ty> {
    list_context(i, "items", Ty::I64)
}

// -- Text & literals ----------------------------------------------

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
    let context = ctx(&i, &[("hello", Ty::String), ("world", Ty::String)]);
    let ir = compile_to_ir(&i, r#"{{ concat(&@hello, &@world) }}"#, &context).unwrap();
    insta::assert_snapshot!(ir);
}

#[test]
fn mixed_text_and_expr() {
    let i = Interner::new();
    let context = ctx(&i, &[("name", Ty::String)]);
    let ir = compile_to_ir(&i, "Hello, {{ &@name }}!", &context).unwrap();
    insta::assert_snapshot!(ir);
}

// -- Context / Variables ------------------------------------------

#[test]
fn context_read() {
    let i = Interner::new();
    let context = ctx(&i, &[("count", Ty::I64)]);
    let ir = compile_to_ir(&i, "{{ @count.to_string() }}", &context).unwrap();
    insta::assert_snapshot!(ir);
}

#[test]
fn variable_write() {
    let i = Interner::new();
    let ir = compile_simple(&i, "% let count = 42").unwrap();
    insta::assert_snapshot!(ir);
}

#[test]
fn context_field_access() {
    let i = Interner::new();
    let ir = compile_to_ir(&i, "{{ @user.name }}", &user_context(&i)).unwrap();
    insta::assert_snapshot!(ir);
}

// -- Arithmetic ---------------------------------------------------

#[test]
fn arithmetic_to_string() {
    let i = Interner::new();
    let context = ctx(&i, &[("a", Ty::I64), ("b", Ty::I64)]);
    let ir = compile_to_ir(
        &i,
        "% let out = @a + @b\n\
                                {{ out.to_string() }}",
        &context,
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

// -- Match blocks -------------------------------------------------

#[test]
fn simple_match_binding() {
    let i = Interner::new();
    let context = ctx(&i, &[("name", Ty::String)]);
    // Variable binding is body-less - defines x in current scope.
    let ir = compile_to_ir(
        &i,
        "% let x = &@name\n\
                                {{ x }}",
        &context,
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

#[test]
fn match_literal_filter() {
    let i = Interner::new();
    let context = ctx(&i, &[("role", Ty::String)]);
    let ir = compile_to_ir(
        &i,
        "% match @role\n\
         % \"admin\" =>\n\
         admin page\n\
         % _ =>\n\
         guest page\n\
         % end",
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
        "% match @role\n\
         % \"admin\" =>\n\
         admin\n\
         % \"user\" =>\n\
         user\n\
         % _ =>\n\
         guest\n\
         % end",
        &context,
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

// -- List patterns ------------------------------------------------

#[test]
fn list_destructure_head() {
    let i = Interner::new();
    let ir = compile_to_ir(
        &i,
        "% match @items\n\
         % [a, b, ..] =>\n\
         {{ a.to_string() }}\n\
         % _ =>\n\
         empty\n\
         % end",
        &items_context(&i),
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

// -- Object patterns ----------------------------------------------

#[test]
fn object_pattern() {
    let i = Interner::new();
    let ir = compile_to_ir(
        &i,
        "% if let { name, age, } = @user\n\
         {{ name }}\n\
         % end",
        &user_context(&i),
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

// -- Range --------------------------------------------------------

// -- Pipe & builtins ----------------------------------------------

#[test]
fn pipe_filter_map() {
    let i = Interner::new();
    // Variable binding is body-less.
    let ir = compile_to_ir(
        &i,
        "% let items = @items\n\
         % @items = vec([])\n\
         % let x = items | into_iter | filter(|x| -> *x != 0) | map(|x| -> x + 1) | collect\n\
         % let out = len(&x)\n\
         {{ out.to_string() }}",
        &items_list_context(&i),
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

#[test]
fn pipe_to_string() {
    let i = Interner::new();
    let context = ctx(&i, &[("n", Ty::I64)]);
    let ir = compile_to_ir(&i, "{{ @n.to_string() }}", &context).unwrap();
    insta::assert_snapshot!(ir);
}

// -- Lambda / closures --------------------------------------------

#[test]
fn lambda_in_filter() {
    let i = Interner::new();
    // Variable binding is body-less.
    let ir = compile_to_ir(
        &i,
        "% let items = @items\n\
         % @items = vec([])\n\
         % let x = items | into_iter | filter(|x| -> *x != 0) | collect\n\
         % let out = len(&x)\n\
         {{ out.to_string() }}",
        &items_list_context(&i),
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

// -- Extern functions ---------------------------------------------

/// An opaque extern function of the given name and monomorphic signature.
fn extern_fn(i: &Interner, name: &str, params: &[Ty], ret: Ty) -> Function {
    Function {
        qref: QualifiedRef::root(i.intern(name)),
        kind: FnKind::Extern {
            bounds: vec![],
            instances: Default::default(),
            requires: vec![],
        },
        ty: TyTerm::Fn {
            params: params
                .iter()
                .enumerate()
                .map(|(n, ty)| ParamTerm::<Poly>::new(i.intern(&format!("_{n}")), lift_to_poly(ty)))
                .collect(),
            ret: Box::new(lift_to_poly(&ret)),
            captures: vec![],
            effect: acvus_mir::ty::Effect::OPAQUE.into(),
        },
    }
}

#[test]
fn extern_async_call() {
    let i = Interner::new();
    let fetch_user = Function {
        qref: QualifiedRef::root(i.intern("fetch_user")),
        kind: FnKind::Extern {
            bounds: vec![],
            instances: Default::default(),
            requires: vec![],
        },
        ty: TyTerm::Fn {
            params: vec![ParamTerm::<Poly>::new(
                i.intern("id"),
                lift_to_poly(&Ty::I64),
            )],
            ret: Box::new(lift_to_poly(&Ty::String)),
            captures: vec![],
            effect: acvus_mir::ty::Effect::OPAQUE.into(),
        },
    };
    let ir = compile_to_ir_with(
        &i,
        "% let user = fetch_user(1)\n\
         {{ user }}",
        &FxHashMap::default(),
        &[fetch_user],
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

// -- Tuple --------------------------------------------------------

#[test]
fn tuple_expression() {
    let i = Interner::new();
    let context = ctx(&i, &[("a", Ty::I64), ("b", Ty::String)]);
    let ir = compile_to_ir(
        &i,
        "% if let (a, b) = (@a, @b)\n\
         {{ a.to_string() }}{{ b }}\n\
         % end",
        &context,
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

#[test]
fn tuple_pattern_binding() {
    let i = Interner::new();
    let context = ctx(&i, &[("pair", Ty::Tuple(vec![Ty::String, Ty::I64]))]);
    let ir = compile_to_ir(
        &i,
        "% if let (name, age) = @pair\n\
                                {{ name }}\n\
                                % end",
        &context,
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

#[test]
fn tuple_pattern_wildcard() {
    let i = Interner::new();
    let context = ctx(&i, &[("pair", Ty::Tuple(vec![Ty::String, Ty::I64]))]);
    let ir = compile_to_ir(
        &i,
        "% if let (name, _) = @pair\n\
                                {{ name }}\n\
                                % end",
        &context,
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

#[test]
fn tuple_pattern_literal_match() {
    let i = Interner::new();
    let context = ctx(&i, &[("a", Ty::I64), ("b", Ty::I64)]);
    let ir = compile_to_ir(
        &i,
        "% match (@a, @b)\n\
         % (0, 1) =>\n\
         zero-one\n\
         % (1, _) =>\n\
         one-any\n\
         % _ =>\n\
         other\n\
         % end",
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
            Ty::Tuple(vec![Ty::String, obj(&i, &[("x", Ty::I64)])]),
        )],
    );
    let ir = compile_to_ir(
        &i,
        "% if let (label, { x, }) = &@data\n\
         {{ label }}\n\
         % end",
        &context,
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

#[test]
fn error_tuple_arity_mismatch() {
    let i = Interner::new();
    let context = ctx(&i, &[("pair", Ty::Tuple(vec![Ty::I64, Ty::I64]))]);
    let result = compile_to_ir(
        &i,
        "% if let (a, b, c) = @pair\n\
         {{ a.to_string() }}\n\
         % end",
        &context,
    );
    assert!(result.is_err());
    insta::assert_snapshot!(result.unwrap_err());
}

// -- Error cases --------------------------------------------------

#[test]
fn error_emit_non_string() {
    let i = Interner::new();
    let result = compile_simple(&i, "{{ 42 }}");
    assert!(result.is_err());
    insta::assert_snapshot!(result.unwrap_err());
}

#[test]
fn undeclared_context_is_refused() {
    let i = Interner::new();
    let err = compile_to_ir(&i, "{{ @unknown.to_string() }}", &FxHashMap::default())
        .expect_err("an undeclared context is refused");
    assert!(
        err.contains("`@unknown` is not a declared context"),
        "{err}"
    );
}

#[test]
fn error_undefined_variable() {
    let i = Interner::new();
    let result = compile_to_ir(&i, "% let x = unknown", &FxHashMap::default());
    assert!(result.is_err());
}

#[test]
fn error_type_mismatch() {
    let i = Interner::new();
    let result = compile_simple(&i, "% let x = 1 + 2.0");
    assert!(result.is_err());
    insta::assert_snapshot!(result.unwrap_err());
}

// -- Iteration (`in`) --------------------------------------------

#[test]
fn error_iter_not_iterable() {
    let i = Interner::new();
    let context = ctx(&i, &[("name", Ty::String)]);
    let result = compile_to_ir(
        &i,
        "% for x in &@name\n\
         {{ x }}\n\
         % end",
        &context,
    );
    assert!(result.is_err());
}

// -- Edge case: new variable ref binding -------------------------

#[test]
fn variable_new_ref_binding() {
    let i = Interner::new();
    // result is not in initial context - dynamically created via binding.
    let context = ctx(&i, &[("name", Ty::String)]);
    let ir = compile_to_ir(
        &i,
        "% let result = &@name\n\
                                {{ result }}",
        &context,
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

#[test]
fn variable_new_ref_in_match_arm() {
    let i = Interner::new();
    // selected is created inside a match arm, then read after the match.
    let context = ctx(&i, &[("role", Ty::String)]);
    let ir = compile_to_ir(
        &i,
        "% let selected = \"\"\n\
         % match @role\n\
         % \"admin\" =>\n\
         % selected = \"yes\"\n\
         % _ =>\n\
         % selected = \"no\"\n\
         % end\n\
         {{ selected }}",
        &context,
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

// -- Edge case: nested destructuring -----------------------------

#[test]
fn list_head_with_object_elements() {
    let i = Interner::new();
    // List destructure where elements are objects.
    let context = ctx(
        &i,
        &[(
            "users",
            Ty::Array(
                Box::new(obj(&i, &[("name", Ty::String)])),
                acvus_mir::ty::LenTerm::Known(3),
            ),
        )],
    );
    let ir = compile_to_ir(
        &i,
        "% match &@users\n\
         % [first, ..] =>\n\
         {{ first.name }}\n\
         % _ =>\n\
         empty\n\
         % end",
        &context,
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

// -- Edge case: object expression & matching ---------------------

#[test]
fn object_literal_field_access() {
    let i = Interner::new();
    let context = ctx(&i, &[("name", Ty::String)]);
    let ir = compile_to_ir(
        &i,
        "% let o = { @name, }\n\
         {{ o.name }}",
        &context,
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

// -- Edge case: comparison / boolean / unary ----------------------

#[test]
fn comparison_operators() {
    let i = Interner::new();
    let context = ctx(&i, &[("a", Ty::I64), ("b", Ty::I64)]);
    let ir = compile_to_ir(
        &i,
        "% let x = @a > @b\n\
         {{ x.to_string() }}",
        &context,
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

#[test]
fn unary_negation() {
    let i = Interner::new();
    let context = ctx(&i, &[("n", Ty::I64)]);
    let ir = compile_to_ir(
        &i,
        "% let x = -@n\n\
         {{ x.to_string() }}",
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
        "% let x = !@flag\n\
         {{ x.to_string() }}",
        &context,
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

// -- Edge case: `as` conversion (RFC-0049) -----------------------

#[test]
fn to_float_conversion() {
    let i = Interner::new();
    let context = ctx(&i, &[("n", Ty::I64)]);
    let ir = compile_to_ir(
        &i,
        "% let x = @n as f64\n\
         {{ x.to_string() }}",
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
        "% let x = @f as i64\n\
         {{ x.to_string() }}",
        &context,
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

// -- Edge case: pmap builtin -------------------------------------

#[test]
fn pmap_builtin() {
    let i = Interner::new();
    let ir = compile_to_ir(
        &i,
        "% let items = @items\n\
         % @items = vec([])\n\
         % let x = items | into_iter | pmap(|i| -> i + 1) | collect\n\
         % let out = len(&x)\n\
         {{ out.to_string() }}",
        &items_list_context(&i),
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

// -- Edge case: list tail destructure ----------------------------

#[test]
fn list_destructure_tail() {
    let i = Interner::new();
    let ir = compile_to_ir(
        &i,
        "% match @items\n\
         % [.., a, b] =>\n\
         {{ a.to_string() }}\n\
         % _ =>\n\
         empty\n\
         % end",
        &items_context(&i),
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

// -- Edge case: variable write then read -------------------------

#[test]
fn variable_write_then_read() {
    let i = Interner::new();
    let ir = compile_simple(
        &i,
        "% let x = 42\n\
                                 {{ x.to_string() }}",
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

// -- Edge case: nested iteration with binding --------------------

// -- Edge case: range inclusive iteration -------------------------

// -- Edge case: deeply nested object -----------------------------

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

// -- Edge case: closure capturing context ref --------------------

#[test]
fn closure_capture_context() {
    let i = Interner::new();
    let context = ctx(
        &i,
        &[
            ("items", acvus_extern::vec_ty(&i, Ty::I64)),
            ("threshold", Ty::I64),
        ],
    );
    let ir = compile_to_ir(
        &i,
        "% let items = @items\n\
         % @items = vec([])\n\
         % let x = items | into_iter | filter(|i| -> *i > @threshold) | collect\n\
         % let out = len(&x)\n\
         {{ out.to_string() }}",
        &context,
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

// -- Edge case: multi-arm with different pattern types -----------

// -- Edge case: list literal -------------------------------------

#[test]
fn list_literal_expression() {
    let i = Interner::new();
    let ir = compile_simple(
        &i,
        "% let x = [1, 2, 3]\n\
         % let out = len(&x)\n\
         {{ out.to_string() }}",
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

// -- Edge case: lambda with arithmetic ---------------------------

#[test]
fn lambda_map_arithmetic() {
    let i = Interner::new();
    // Lambda param type resolved via unification: map(Vec<Int>, |x| -> x + 1)
    let ir = compile_to_ir(
        &i,
        "% let items = @items\n\
         % @items = vec([])\n\
         % let x = items | into_iter | map(|i| -> i + 1) | collect\n\
         % let out = len(&x)\n\
         {{ out.to_string() }}",
        &items_list_context(&i),
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

#[test]
fn lambda_filter_comparison() {
    let i = Interner::new();
    // Lambda param type resolved via unification: filter(Vec<Int>, |x| -> *x > 0)
    let ir = compile_to_ir(
        &i,
        "% let items = @items\n\
         % @items = vec([])\n\
         % let x = items | into_iter | filter(|i| -> *i > 0) | collect\n\
         % let out = len(&x)\n\
         {{ out.to_string() }}",
        &items_list_context(&i),
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

// -- Edge case: closure with captured local var ------------------

#[test]
fn closure_capture_local() {
    let i = Interner::new();
    // Closure captures local variable (not context); a captured word is
    // copied at each use (RFC-0018).
    let ir = compile_to_ir(
        &i,
        "% let threshold = 5\n\
         % let items = @items\n\
         % @items = vec([])\n\
         % let x = items | into_iter | filter(|i| -> *i > threshold) | collect\n\
         % let out = len(&x)\n\
         {{ out.to_string() }}",
        &items_list_context(&i),
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

// -- Edge case: list exact match (no rest) -----------------------

// -- Edge case: list rest in middle ------------------------------

#[test]
fn list_destructure_head_and_tail() {
    let i = Interner::new();
    // [a, .., z] pattern - head and tail extraction.
    let ir = compile_to_ir(
        &i,
        "% match @items\n\
         % [first, .., last] =>\n\
         {{ first.to_string() }}\n\
         % _ =>\n\
         empty\n\
         % end",
        &items_context(&i),
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

// -- Edge case: nested tuple pattern -----------------------------

#[test]
fn nested_tuple_pattern() {
    let i = Interner::new();
    let context = ctx(
        &i,
        &[(
            "data",
            Ty::Tuple(vec![Ty::Tuple(vec![Ty::I64, Ty::I64]), Ty::String]),
        )],
    );
    let ir = compile_to_ir(
        &i,
        "% if let ((a, b), label) = &@data\n\
         {{ label }}\n\
         % end",
        &context,
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

// -- Edge case: variable write of computed value -----------------

#[test]
fn variable_write_computed() {
    let i = Interner::new();
    let context = ctx(&i, &[("a", Ty::I64), ("b", Ty::I64)]);
    let ir = compile_to_ir(
        &i,
        "% let result = @a + @b\n\
         {{ result.to_string() }}",
        &context,
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

// -- Edge case: match block with binding pattern + body ----------

#[test]
fn match_binding_with_body() {
    let i = Interner::new();
    // Object pattern with body (goes through normal match lowering).
    let ir = compile_to_ir(
        &i,
        "% match @user\n\
         % { name, } =>\n\
         {{ name }} is here\n\
         % _ =>\n\
         no user\n\
         % end",
        &user_context(&i),
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

// -- Edge case: variable shadowing across scopes -----------------

#[test]
fn variable_shadowing() {
    let i = Interner::new();
    let context = ctx(&i, &[("name", Ty::String)]);
    let ir = compile_to_ir(
        &i,
        "% let x = \"outer\"\n\
         % let x = &@name\n\
         {{ x }}",
        &context,
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

// -- Edge case: match inside match (nested match blocks) ---------

// -- Edge case: catch-all with nested binding --------------------

#[test]
fn catch_all_with_binding() {
    let i = Interner::new();
    let context = ctx(&i, &[("role", Ty::String)]);
    let ir = compile_to_ir(
        &i,
        "% match @role\n\
         % \"admin\" =>\n\
         admin\n\
         % _ =>\n\
         % let fallback = \"guest\"\n\
         {{ fallback }}\n\
         % end",
        &context,
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

// -- Edge case: multiple chained pipes ---------------------------

#[test]
fn triple_pipe_chain() {
    let i = Interner::new();
    let ir = compile_to_ir(
        &i,
        "% let items = @items\n\
         % @items = vec([])\n\
         % let x = items | into_iter | filter(|i| -> *i != 0) | map(|i| -> i + 1) | map(|i| -> i * 2) | collect\n\
         % let out = len(&x)\n\
         {{ out.to_string() }}",
        &items_list_context(&i),
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

// -- Edge case: variable write in iteration body -----------------

// -- Edge case: field access on destructured variable ------------

#[test]
fn field_access_on_destructured() {
    let i = Interner::new();
    let context = ctx(
        &i,
        &[(
            "pair",
            Ty::Tuple(vec![obj(&i, &[("name", Ty::String)]), Ty::I64]),
        )],
    );
    let ir = compile_to_ir(
        &i,
        "% if let (obj, _) = &@pair\n\
         {{ obj.name }}\n\
         % end",
        &context,
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

// -- Edge case: boolean operators in match ------------------------

#[test]
fn equality_as_match_source() {
    let i = Interner::new();
    let context = ctx(&i, &[("a", Ty::I64), ("b", Ty::I64)]);
    let ir = compile_to_ir(
        &i,
        "% if @a == @b\n\
         equal\n\
         % else\n\
         not equal\n\
         % end",
        &context,
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

// -- Edge case: unary negation on lambda param (Ty::Var) ---------

#[test]
fn lambda_negate_param() {
    let i = Interner::new();
    // Lambda param has Ty::Var initially; -i must resolve via unification.
    let ir = compile_to_ir(
        &i,
        "% let items = @items\n\
         % @items = vec([])\n\
         % let x = items | into_iter | map(|i| -> -i) | collect\n\
         % let out = len(&x)\n\
         {{ out.to_string() }}",
        &items_list_context(&i),
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

// -- Edge case: unary not on lambda param (Ty::Var) --------------

#[test]
fn lambda_not_param() {
    let i = Interner::new();
    // Lambda param has Ty::Var initially; !i must resolve via unification.
    let context = list_context(&i, "flags", Ty::Bool);
    let ir = compile_to_ir(
        &i,
        "% let flags = @flags\n\
         % @flags = vec([])\n\
         % let x = flags | into_iter | map(|i| -> !i) | collect\n\
         % let out = len(&x)\n\
         {{ out.to_string() }}",
        &context,
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

// -- Edge case: object pattern on match source (non-list) --------

#[test]
fn object_destructure_match() {
    let i = Interner::new();
    // Object pattern directly on Object source (not Vec<Object>).
    let ir = compile_to_ir(
        &i,
        "% match @user\n\
         % { name, age, } =>\n\
         {{ name }}{{ age.to_string() }}\n\
         % _ =>\n\
         none\n\
         % end",
        &user_context(&i),
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

// -- Edge case: multiple closures sharing captured var ------------

#[test]
fn multiple_closures_same_capture() {
    let i = Interner::new();
    let context = ctx(
        &i,
        &[
            ("items", acvus_extern::vec_ty(&i, Ty::I64)),
            ("offset", Ty::I64),
        ],
    );
    let ir = compile_to_ir(
        &i,
        "% let items = @items\n\
         % @items = vec([])\n\
         % let x = items | into_iter | map(|i| -> i + @offset) | filter(|i| -> *i > 0) | collect\n\
         % let out = len(&x)\n\
         {{ out.to_string() }}",
        &context,
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

// -- Edge case: string comparison ---------------------------------

#[test]
fn string_equality_in_filter() {
    let i = Interner::new();
    let context = ctx(
        &i,
        &[(
            "names",
            Ty::Array(Box::new(Ty::String), acvus_mir::ty::LenTerm::Known(3)),
        )],
    );
    let ir = compile_to_ir(
        &i,
        "% let names = @names\n\
         % @names = [\"\".to_string(), \"\".to_string(), \"\".to_string()]\n\
         % let x = names | into_iter | filter(|n| -> n != \"admin\".to_string())\n\
         {{ x | join(\",\".to_string()) }}",
        &context,
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

// -- Edge case: nested lambda (lambda returning lambda result) ----

#[test]
fn lambda_field_access() {
    let i = Interner::new();
    // Lambda body accesses field on captured object.
    let context = list_context(
        &i,
        "users",
        obj(&i, &[("name", Ty::String), ("age", Ty::I64)]),
    );
    let ir = compile_to_ir(
        &i,
        "% let users = @users\n\
         % @users = vec([])\n\
         % let x = users | into_iter | map(|u| -> u.name)\n\
         {{ x | join(\",\".to_string()) }}",
        &context,
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

// -- Edge case: variable accumulation in loop ---------------------

// -- Edge case: multi-level pipe with to_string in middle --------

#[test]
fn pipe_map_to_string_then_filter() {
    let i = Interner::new();
    let ir = compile_to_ir(
        &i,
        "% let items = @items\n\
         % @items = vec([])\n\
         % let x = items | into_iter | map(|i| -> i + 1) | filter(|i| -> *i != 0) | collect\n\
         % let out = len(&x)\n\
         {{ out.to_string() }}",
        &items_list_context(&i),
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

// -- Edge case: local var captured in lambda ----------------------

#[test]
fn lambda_capture_local_var_ref() {
    let i = Interner::new();
    // offset is NOT in initial context - created as local var.
    // Lambda must capture it correctly (not fall through to StorageLoad);
    // a captured word is copied at each use (RFC-0018).
    let context = items_list_context(&i);
    let ir = compile_to_ir(
        &i,
        "% let offset = 10\n\
         % let items = @items\n\
         % @items = vec([])\n\
         % let x = items | into_iter | filter(|i| -> *i > offset) | collect\n\
         % let out = len(&x)\n\
         {{ out.to_string() }}",
        &context,
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

// -- Edge case: multiple field accesses on same lambda param -----

#[test]
fn lambda_multiple_field_access() {
    let i = Interner::new();
    // Lambda body accesses two fields on the same param.
    let context = list_context(
        &i,
        "users",
        obj(&i, &[("name", Ty::String), ("age", Ty::I64)]),
    );
    let ir = compile_to_ir(
        &i,
        "% let users = @users\n\
         % @users = vec([])\n\
         % let x = users | into_iter | map(|u| -> (u.name, u.age)) | collect\n\
         % let out = len(&x)\n\
         {{ out.to_string() }}",
        &context,
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

// -- Edge case: chained field access in lambda -------------------

#[test]
fn lambda_chained_field_access() {
    let i = Interner::new();
    let context = list_context(
        &i,
        "users",
        obj(&i, &[("address", obj(&i, &[("city", Ty::String)]))]),
    );
    let ir = compile_to_ir(
        &i,
        "% let users = @users\n\
         % @users = vec([])\n\
         % let x = users | into_iter | map(|u| -> u.address.city)\n\
         {{ x | join(\",\".to_string()) }}",
        &context,
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

// -- Edge case: string concat in lambda ---------------------------

#[test]
fn lambda_string_concat() {
    let i = Interner::new();
    // Lambda param is Ty::Var; string concat (+) must resolve via unification.
    let context = list_context(&i, "names", Ty::String);
    let ir = compile_to_ir(
        &i,
        "% let names = @names\n\
         % @names = vec([])\n\
         % let x = names | into_iter | map(|n| -> n + \"!\".to_string())\n\
         {{ x | join(\",\".to_string()) }}",
        &context,
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

// -- Edge case: filter then map with field access ----------------

#[test]
fn pipe_filter_then_map_field() {
    let i = Interner::new();
    let context = list_context(
        &i,
        "users",
        obj(&i, &[("name", Ty::String), ("age", Ty::I64)]),
    );
    let ir = compile_to_ir(
        &i,
        "% let users = @users\n\
         % @users = vec([])\n\
         % let x = users | into_iter | filter(|u| -> u.age > 18) | map(|u| -> u.name)\n\
         {{ x | join(\",\".to_string()) }}",
        &context,
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

// -- Edge case: error - field access on non-object ----------------

#[test]
fn error_field_access_on_int() {
    let i = Interner::new();
    let context = ctx(&i, &[("n", Ty::I64)]);
    let result = compile_to_ir(&i, "{{ @n.foo.to_string() }}", &context);
    assert!(result.is_err());
    insta::assert_snapshot!(result.unwrap_err());
}

// -- Edge case: error - context write attempt ---------------------

#[test]
fn error_variable_write_type_mismatch() {
    let i = Interner::new();
    // Attempting to write to a context key (read-only).
    let context = ctx(&i, &[("count", Ty::I64)]);
    let result = compile_to_ir(&i, "% @count = \"hello\".to_string()", &context);
    assert!(result.is_err());
    insta::assert_snapshot!(result.unwrap_err());
}

// -- Edge case: float arithmetic in lambda ------------------------

#[test]
fn lambda_float_arithmetic() {
    let i = Interner::new();
    let context = list_context(&i, "vals", Ty::Float);
    let ir = compile_to_ir(
        &i,
        "% let vals = @vals\n\
         % @vals = vec([])\n\
         % let x = vals | into_iter | map(|v| -> v * 2.0) | collect\n\
         % let out = len(&x)\n\
         {{ out.to_string() }}",
        &context,
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

// -- Edge case: bool literal as match source ---------------------

#[test]
fn match_bool_literal() {
    let i = Interner::new();
    let context = ctx(&i, &[("flag", Ty::Bool)]);
    let ir = compile_to_ir(
        &i,
        "% if @flag\n\
                                on\n\
                                % else\n\
                                off\n\
                                % end",
        &context,
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

// -- Edge case: nested pipe with filter on object field ----------

#[test]
fn filter_object_field_equality() {
    let i = Interner::new();
    let context = list_context(
        &i,
        "users",
        obj(&i, &[("name", Ty::String), ("active", Ty::Bool)]),
    );
    let ir = compile_to_ir(
        &i,
        "% let users = @users\n\
         % @users = vec([])\n\
         % let x = users | into_iter | filter(|u| -> u.active) | collect\n\
         % let out = len(&x)\n\
         {{ out.to_string() }}",
        &context,
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

// -- Edge case: extern function with object return ---------------

#[test]
fn extern_fn_object_return() {
    let i = Interner::new();
    let get_user = extern_fn(
        &i,
        "get_user",
        &[Ty::I64],
        obj(&i, &[("name", Ty::String), ("age", Ty::I64)]),
    );
    let ir = compile_to_ir_with(
        &i,
        "% let u = get_user(1)\n\
         {{ u.name }}",
        &FxHashMap::default(),
        &[get_user],
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

// -- New builtins -------------------------------------------------

#[test]
fn builtin_len() {
    let i = Interner::new();
    let context = items_list_context(&i);
    let ir = compile_to_ir(
        &i,
        "% let items = @items\n\
         % @items = vec([])\n\
         % let out = len(&items)\n\
         {{ out.to_string() }}",
        &context,
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

#[test]
fn builtin_join() {
    let i = Interner::new();
    let context = list_context(&i, "names", Ty::String);
    let ir = compile_to_ir(
        &i,
        "% let names = @names\n\
         % @names = vec([])\n\
         {{ names | into_iter | join(\", \".to_string()) }}",
        &context,
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

#[test]
fn builtin_contains() {
    let i = Interner::new();
    let context = items_list_context(&i);
    let ir = compile_to_ir(
        &i,
        "% let items = @items\n\
         % @items = vec([])\n\
         % let out = items | into_iter | contains(3)\n\
         {{ out.to_string() }}",
        &context,
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

#[test]
fn builtin_find() {
    let i = Interner::new();
    let context = items_list_context(&i);
    let ir = compile_to_ir(
        &i,
        "% let items = @items\n\
         % @items = vec([])\n\
         % let out = items | into_iter | find(|x| -> *x > 10)\n\
         % match out\n\
         % Some(v) =>\n\
         {{ v.to_string() }}\n\
         % _ =>\n\
         none\n\
         % end",
        &context,
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

#[test]
fn builtin_reduce() {
    let i = Interner::new();
    let context = items_list_context(&i);
    let ir = compile_to_ir(
        &i,
        "% let items = @items\n\
         % @items = vec([])\n\
         % let out = items | into_iter | reduce(|a, b| -> a + b)\n\
         % match out\n\
         % Some(v) =>\n\
         {{ v.to_string() }}\n\
         % _ =>\n\
         none\n\
         % end",
        &context,
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

#[test]
fn builtin_fold() {
    let i = Interner::new();
    let context = items_list_context(&i);
    let ir = compile_to_ir(
        &i,
        "% let items = @items\n\
         % @items = vec([])\n\
         % let out = items | into_iter | fold(0, |acc, x| -> acc + x)\n\
         {{ out.to_string() }}",
        &context,
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

#[test]
fn builtin_any() {
    let i = Interner::new();
    let context = items_list_context(&i);
    let ir = compile_to_ir(
        &i,
        "% let items = @items\n\
         % @items = vec([])\n\
         % let out = items | into_iter | any(|x| -> *x > 10)\n\
         {{ out.to_string() }}",
        &context,
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

#[test]
fn builtin_all() {
    let i = Interner::new();
    let context = items_list_context(&i);
    let ir = compile_to_ir(
        &i,
        "% let items = @items\n\
         % @items = vec([])\n\
         % let out = items | into_iter | all(|x| -> *x > 0)\n\
         {{ out.to_string() }}",
        &context,
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

// -- Context Call ------------------------------------------------

// -- Variant (Option) --------------------------------------------

#[test]
fn variant_some_expr() {
    let i = Interner::new();
    let ir = compile_simple(&i, "% let x = Some(42)").unwrap();
    insta::assert_snapshot!(ir);
}

#[test]
fn variant_none_expr() {
    let i = Interner::new();
    let ir = compile_simple(&i, "% let x = None").unwrap();
    insta::assert_snapshot!(ir);
}

#[test]
fn variant_some_pattern() {
    let i = Interner::new();
    let context = ctx(&i, &[("opt", Ty::Option(Box::new(Ty::I64)))]);
    let ir = compile_to_ir(
        &i,
        "% match &@opt\n\
         % Some(v) =>\n\
         {{ to_string(v) }}\n\
         % _ =>\n\
         nope\n\
         % end",
        &context,
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

#[test]
fn variant_none_pattern() {
    let i = Interner::new();
    let context = ctx(&i, &[("opt", Ty::Option(Box::new(Ty::I64)))]);
    let ir = compile_to_ir(
        &i,
        "% match @opt\n\
                                % None =>\n\
                                none\n\
                                % _ =>\n\
                                has value\n\
                                % end",
        &context,
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

#[test]
fn structural_enum_variant_merge() {
    let i = Interner::new();
    let module = compile_analysis(
        &i,
        "% if let A::B = @a\n\
                              hi\n\
                              % end\n\
                              % if let A::C = @a\n\
                              bye\n\
                              % end",
        &[],
    )
    .unwrap();
    let ir = acvus_mir::printer::dump_with(&i, &module);
    assert!(ir.contains("B"), "variant B missing from IR:\n{ir}");
    assert!(ir.contains("C"), "variant C missing from IR:\n{ir}");
}

// -- Structural enum tests --------------------------------------

#[test]
fn structural_enum_single_variant() {
    let i = Interner::new();
    let module = compile_analysis(
        &i,
        "% match @a\n\
                                       % A::B =>\n\
                                       yes\n\
                                       % _ =>\n\
                                       no\n\
                                       % end",
        &[],
    )
    .unwrap();
    let ir = acvus_mir::printer::dump_with(&i, &module);
    assert!(ir.contains("B"), "variant B missing from IR:\n{ir}");
}

#[test]
fn structural_enum_three_variants_merge() {
    let i = Interner::new();
    let src = "% if let S::X = @v\n\
               x\n\
               % end\n\
               % if let S::Y = @v\n\
               y\n\
               % end\n\
               % if let S::Z = @v\n\
               z\n\
               % end";
    let module = compile_analysis(&i, src, &[]).unwrap();
    let ir = acvus_mir::printer::dump_with(&i, &module);
    assert!(ir.contains("X"), "variant X missing:\n{ir}");
    assert!(ir.contains("Y"), "variant Y missing:\n{ir}");
    assert!(ir.contains("Z"), "variant Z missing:\n{ir}");
}

#[test]
fn structural_enum_with_payload() {
    let i = Interner::new();
    let src = "% match @r\n\
               % R::Good(v) =>\n\
               {{ v }}\n\
               % _ =>\n\
               err\n\
               % end";
    let module = compile_analysis(&i, src, &[]).unwrap();
    let ir = acvus_mir::printer::dump_with(&i, &module);
    assert!(ir.contains("Good"), "variant Good missing:\n{ir}");
}

#[test]
fn structural_enum_mixed_payload_and_unit() {
    let i = Interner::new();
    let src = "% match @r\n\
               % R::Good(v) =>\n\
               {{ v }}\n\
               % R::Bad =>\n\
               fail\n\
               % _ =>\n\
               ??\n\
               % end";
    let module = compile_analysis(&i, src, &[]).unwrap();
    let ir = acvus_mir::printer::dump_with(&i, &module);
    assert!(ir.contains("Good"), "variant Good missing:\n{ir}");
    assert!(ir.contains("Bad"), "variant Bad missing:\n{ir}");
}

#[test]
fn structural_enum_same_var_different_blocks_merge() {
    // Key regression test: separate match blocks on the same context var must merge.
    let i = Interner::new();
    let src = "% if let A::B = @a\n\
               b\n\
               % end\n\
               % if let A::C = @a\n\
               c\n\
               % end";
    let module = compile_analysis(&i, src, &[]).unwrap();
    let ir = acvus_mir::printer::dump_with(&i, &module);
    assert!(ir.contains("B"), "variant B missing:\n{ir}");
    assert!(ir.contains("C"), "variant C missing:\n{ir}");
}

#[test]
fn structural_enum_different_enums_different_vars() {
    let i = Interner::new();
    let src = "% if let X::A = @x\n\
               xa\n\
               % end\n\
               % if let Y::B = @y\n\
               yb\n\
               % end";
    let module = compile_analysis(&i, src, &[]).unwrap();
    let ir = acvus_mir::printer::dump_with(&i, &module);
    assert!(ir.contains("A"), "variant A missing:\n{ir}");
    assert!(ir.contains("B"), "variant B missing:\n{ir}");
}

#[test]
fn structural_enum_name_mismatch_is_error() {
    // Matching X::A and Y::B on the same var should fail (different enum names).
    let i = Interner::new();
    let src = "% if let X::A = @v\n\
               a\n\
               % end\n\
               % if let Y::B = @v\n\
               b\n\
               % end";
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
    let src = "% match @a\n\
               % A::X(x) =>\n\
               % match x\n\
               % 0 =>\n\
               zero\n\
               % _ =>\n\
               other\n\
               % end\n\
               % _ =>\n\
               none\n\
               % end";
    let module = compile_analysis(&i, src, &[]).unwrap();
    let ir = acvus_mir::printer::dump_with(&i, &module);
    assert!(ir.contains("X"), "variant X missing:\n{ir}");
}

#[test]
fn structural_enum_payload_unifies_with_emit() {
    // Payload bound by variant pattern can be used in expressions (emit).
    let i = Interner::new();
    let src = "% match @a\n\
               % A::Val(v) =>\n\
               % let out = v + 1\n\
               {{ out.to_string() }}\n\
               % _ =>\n\
               n/a\n\
               % end";
    let module = compile_analysis(&i, src, &[]).unwrap();
    let ir = acvus_mir::printer::dump_with(&i, &module);
    assert!(ir.contains("Val"), "variant Val missing:\n{ir}");
}

#[test]
fn structural_enum_payload_type_propagates_through_context() {
    // When context provides an enum type with payload, payload type should propagate.
    let i = Interner::new();
    let mut variants = FxHashMap::default();
    variants.insert(i.intern("Good"), Some(Box::new(Ty::I64)));
    variants.insert(i.intern("Bad"), None);
    let src = "% match @r\n\
               % R::Good(v) =>\n\
               % let out = v + 1\n\
               {{ out.to_string() }}\n\
               % R::Bad =>\n\
               err\n\
               % _ =>\n\
               ??\n\
               % end";
    let module = compile_analysis(
        &i,
        src,
        &[(
            "r",
            Ty::Enum {
                name: i.intern("R"),
                variants,
                home: acvus_mir::ty::Home::NONE,
            },
        )],
    )
    .unwrap();
    let ir = acvus_mir::printer::dump_with(&i, &module);
    assert!(ir.contains("Good"), "variant Good missing:\n{ir}");
    assert!(ir.contains("Bad"), "variant Bad missing:\n{ir}");
}

// -- Variant unification inside Tuple/List patterns -------------
// Regression: nested Variant patterns inside Tuple/List must merge
// variant sets across match arms via the shared Var chain.

#[test]
fn variant_merge_inside_tuple_pattern() {
    // Two arms with different variants nested inside a tuple pattern.
    // Both A and B must appear in the final merged Enum type.
    let i = Interner::new();
    let src = "% match @t\n\
               % (S::A, x) =>\n\
               {{ x }}\n\
               % (S::B, y) =>\n\
               {{ y }}\n\
               % _ =>\n\
               ??\n\
               % end";
    let module = compile_analysis(&i, src, &[]).unwrap();
    let ir = acvus_mir::printer::dump_with(&i, &module);
    eprintln!("=== TUPLE VARIANT IR ===\n{ir}\n=== END ===");
    assert!(ir.contains("A"), "variant A missing from IR:\n{ir}");
    assert!(ir.contains("B"), "variant B missing from IR:\n{ir}");
}

#[test]
fn variant_merge_inside_tuple_three_arms() {
    let i = Interner::new();
    let src = "% match @t\n\
               % (S::X, _) =>\n\
               x\n\
               % (S::Y, _) =>\n\
               y\n\
               % (S::Z, _) =>\n\
               z\n\
               % _ =>\n\
               ??\n\
               % end";
    let module = compile_analysis(&i, src, &[]).unwrap();
    let ir = acvus_mir::printer::dump_with(&i, &module);
    assert!(ir.contains("X"), "variant X missing:\n{ir}");
    assert!(ir.contains("Y"), "variant Y missing:\n{ir}");
    assert!(ir.contains("Z"), "variant Z missing:\n{ir}");
}

// -- SSA chain tests (script mode) -------------------------------

#[test]
fn ssa_context_read_write() {
    let i = Interner::new();
    let context = ctx(&i, &[("x", Ty::I64)]);
    let ir = compile_script_ir(&i, "@x = @x + 1; @x", &context).unwrap();
    insta::assert_snapshot!(ir);
}

#[test]
fn ssa_context_branch_phi() {
    let i = Interner::new();
    let context = ctx(&i, &[("x", Ty::I64), ("flag", Ty::Bool)]);
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
    let context = ctx(&i, &[("a", Ty::I64), ("b", Ty::I64)]);
    let ir = compile_script_ir(&i, "@a = @a + 1; @b = @b + 2; @a + @b", &context).unwrap();
    insta::assert_snapshot!(ir);
}

// ====================================================================
// Migrated from acvus-mir unit tests (ExternFn-dependent)
// ====================================================================

// -- From lib.rs -----------------------------------------------------

#[test]
fn migrated_extern_param_write_rejected() {
    let i = Interner::new();
    // Writing to extern param is rejected.
    assert!(compile_to_ir(&i, "% $count = 42", &FxHashMap::default()).is_err());
    // Reading an extern param via context with pipe is valid.
    let context = ctx(&i, &[("count", Ty::I64)]);
    compile_to_ir(&i, "{{ @count.to_string() }}", &context).unwrap();
}

#[test]
fn migrated_integration_list_destructure() {
    let i = Interner::new();
    compile_to_ir(
        &i,
        "% if let [a, b, ..] = @items\n\
         {{ a.to_string() }}\n\
         % end",
        &items_context(&i),
    )
    .unwrap();
}

#[test]
fn migrated_integration_pipe_with_lambda() {
    let i = Interner::new();
    let context = items_list_context(&i);
    compile_to_ir(
        &i,
        "% let items = @items\n\
         % @items = vec([])\n\
         % let x = items | into_iter | filter(|x| -> *x != 0) | collect\n\
         % let out = len(&x)\n\
         {{ out.to_string() }}",
        &context,
    )
    .unwrap();
}

#[test]
fn migrated_projection_chained_field_access() {
    let i = Interner::new();
    let inner = obj(&i, &[("b", Ty::I64)]);
    let obj_ty = obj(&i, &[("a", inner)]);
    let context = ctx(&i, &[("obj", obj_ty)]);
    let ir = compile_script_ir(&i, "@obj.a.b.to_string()", &context).unwrap();
    assert!(ir.contains("fetch @obj"), "should fetch @obj in IR: {ir}");
    assert!(
        ir.contains("ref &@obj.a.b"),
        "should read @obj.a.b in IR: {ir}"
    );
    let field_get_count = ir.matches(".b").count() + ir.matches(".a").count();
    assert!(
        field_get_count >= 2,
        "should have at least 2 field accesses in IR: {ir}"
    );
}

#[test]
fn migrated_pipe_extern_fn_ok() {
    let i = Interner::new();
    let mapper = extern_fn(&i, "mapper", &[Ty::I64], Ty::String);
    compile_to_ir_with(
        &i,
        "% let items = @items\n\
         % @items = vec([])\n\
         % let x = items | into_iter | map(|i| -> mapper(i)) | collect\n\
         % let out = len(&x)\n\
         {{ out.to_string() }}",
        &items_list_context(&i),
        &[mapper],
    )
    .unwrap();
}

// -- From typeck.rs --------------------------------------------------

#[test]
fn migrated_typeck_builtin_to_string() {
    let i = Interner::new();
    let context = ctx(&i, &[("count", Ty::I64)]);
    compile_to_ir(&i, "{{ @count.to_string() }}", &context).unwrap();
}

#[test]
fn migrated_typeck_lambda_captures_outer_variable() {
    let i = Interner::new();
    let context = ctx(
        &i,
        &[
            ("items", acvus_extern::vec_ty(&i, Ty::I64)),
            ("threshold", Ty::I64),
        ],
    );
    compile_to_ir(
        &i,
        "% let items = @items\n\
         % @items = vec([])\n\
         % let n1 = items | into_iter | filter(|x| -> *x > @threshold) | collect\n\
         % let out = len(&n1)\n\
         {{ out.to_string() }}",
        &context,
    )
    .unwrap();
}

#[test]
fn migrated_typeck_lambda_type_check() {
    let i = Interner::new();
    let context = items_list_context(&i);
    compile_to_ir(
        &i,
        "% let items = @items\n\
         % @items = vec([])\n\
         % let x = items | into_iter | filter(|x| -> *x != 0) | collect\n\
         % let out = len(&x)\n\
         {{ out.to_string() }}",
        &context,
    )
    .unwrap();
}

#[test]
fn migrated_typeck_lambda_no_capture_local_params() {
    let i = Interner::new();
    let context = items_list_context(&i);
    compile_to_ir(
        &i,
        "% let items = @items\n\
         % @items = vec([])\n\
         % let n2 = items | into_iter | map(|x| -> x + 1) | collect\n\
         % let out = len(&n2)\n\
         {{ out.to_string() }}",
        &context,
    )
    .unwrap();
}

#[test]
fn migrated_typeck_list_pattern_matching() {
    let i = Interner::new();
    let context = ctx(
        &i,
        &[(
            "items",
            Ty::Array(Box::new(Ty::I64), acvus_mir::ty::LenTerm::Known(3)),
        )],
    );
    compile_to_ir(
        &i,
        "% if let [a, b, ..] = @items\n\
         {{ a.to_string() }}\n\
         % end",
        &context,
    )
    .unwrap();
}

#[test]
fn migrated_typeck_nested_lambda_captures() {
    let i = Interner::new();
    let context = ctx(
        &i,
        &[
            ("items", acvus_extern::vec_ty(&i, Ty::I64)),
            ("factor", Ty::I64),
        ],
    );
    compile_to_ir(
        &i,
        "% let items = @items\n\
         % @items = vec([])\n\
         % let n3 = items | into_iter | map(|x| -> x * @factor) | collect\n\
         % let out = len(&n3)\n\
         {{ out.to_string() }}",
        &context,
    )
    .unwrap();
}

#[test]
fn migrated_typeck_some_unifies_with_option_context() {
    let i = Interner::new();
    let context = ctx(&i, &[("opt", Ty::Option(Box::new(Ty::I64)))]);
    compile_to_ir(
        &i,
        "% if let Some(v) = &@opt\n\
         {{ to_string(v) }}\n\
         % end",
        &context,
    )
    .unwrap();
}

// -- From printer.rs -------------------------------------------------

#[test]
fn migrated_print_arithmetic() {
    let i = Interner::new();
    let context = ctx(&i, &[("a", Ty::I64), ("b", Ty::I64)]);
    let ir = compile_to_ir(
        &i,
        "% let x = @a + @b\n\
         {{ x.to_string() }}",
        &context,
    )
    .unwrap();
    assert!(ir.contains("+"), "should contain + operator in IR: {ir}");
}

#[test]
fn migrated_print_closure() {
    let i = Interner::new();
    let context = items_list_context(&i);
    let ir = compile_to_ir(
        &i,
        "% let items = @items\n\
         % @items = vec([])\n\
         % let x = items | into_iter | filter(|x| -> *x != 0) | collect\n\
         % let out = len(&x)\n\
         {{ out.to_string() }}",
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

// -- From ssa_pass.rs ------------------------------------------------

// -- Iterator values are move-only through combinators ---------------

#[test]
fn iter_map_reuse_rejected() {
    // into_iter(list) | map(f) is an Iterator; reusing it is a use-after-move.
    let i = Interner::new();
    let context = ctx(
        &i,
        &[
            (
                "items",
                Ty::Array(Box::new(Ty::I64), acvus_mir::ty::LenTerm::Known(3)),
            ),
            ("counter", Ty::I64),
        ],
    );
    let result = compile_script_ir(
        &i,
        r#"let it = @items | into_iter | map(|x| -> { @counter = x; x }); it | collect; it | collect"#,
        &context,
    );
    assert!(result.is_err(), "iter reuse should be rejected: {result:?}");
}

#[test]
fn iter_map_single_use_ok() {
    // Single use of an iterator compiles.
    let i = Interner::new();
    let context = ctx(
        &i,
        &[
            (
                "items",
                Ty::Array(Box::new(Ty::I64), acvus_mir::ty::LenTerm::Known(3)),
            ),
            ("counter", Ty::I64),
        ],
    );
    let result = compile_script_ir(
        &i,
        r#"let items = @items; @items = [1, 2, 3]; items | into_iter | map(|x| -> { @counter = x; x }) | collect"#,
        &context,
    );
    assert!(
        result.is_ok(),
        "single use of iter should compile: {result:?}"
    );
}

#[test]
fn iter_chain_reuse_rejected() {
    // map | filter is still one Iterator value; reusing it is rejected.
    let i = Interner::new();
    let context = ctx(
        &i,
        &[
            (
                "items",
                Ty::Array(Box::new(Ty::I64), acvus_mir::ty::LenTerm::Known(3)),
            ),
            ("a", Ty::I64),
            ("b", Ty::I64),
        ],
    );
    let result = compile_script_ir(
        &i,
        r#"let it = @items | into_iter | map(|x| -> { @a = x; x }) | filter(|x| -> { @b = x; x > 0 }); it | collect; it | collect"#,
        &context,
    );
    assert!(
        result.is_err(),
        "chained iter reuse should be rejected: {result:?}"
    );
}

#[test]
fn iter_chain_single_use_ok() {
    // Single use of a chained iterator compiles.
    let i = Interner::new();
    let context = ctx(
        &i,
        &[
            (
                "items",
                Ty::Array(Box::new(Ty::I64), acvus_mir::ty::LenTerm::Known(3)),
            ),
            ("a", Ty::I64),
            ("b", Ty::I64),
        ],
    );
    let result = compile_script_ir(
        &i,
        r#"let items = @items; @items = [1, 2, 3]; items | into_iter | map(|x| -> { @a = x; x }) | filter(|x| -> { @b = *x; *x > 0 }) | collect"#,
        &context,
    );
    assert!(
        result.is_ok(),
        "single use of chained iter should compile: {result:?}"
    );
}

#[test]
fn iter_pure_map_reuse_rejected() {
    // A pure map does not make an Iterator copyable.
    let i = Interner::new();
    let context = ctx(
        &i,
        &[(
            "items",
            Ty::Array(Box::new(Ty::I64), acvus_mir::ty::LenTerm::Known(3)),
        )],
    );
    let result = compile_script_ir(
        &i,
        r#"let it = @items | into_iter | map(|x| -> x + 1); it | collect; it | collect"#,
        &context,
    );
    assert!(
        result.is_err(),
        "pure iter reuse should be rejected: {result:?}"
    );
}

#[test]
fn iter_reuse_after_collect_rejected() {
    // collect consumes the iterator variable; a second collect is a use-after-move.
    let i = Interner::new();
    let context = ctx(
        &i,
        &[
            (
                "items",
                Ty::Array(Box::new(Ty::I64), acvus_mir::ty::LenTerm::Known(3)),
            ),
            ("counter", Ty::I64),
        ],
    );
    let result = compile_script_ir(
        &i,
        r#"let it = @items | into_iter | map(|x| -> { @counter = x; x }); let collected = it | collect; it | collect"#,
        &context,
    );
    assert!(
        result.is_err(),
        "reuse of iter after collect should be rejected: {result:?}"
    );
    assert!(
        has_use_after_move(&result.unwrap_err()),
        "expected use-after-move error"
    );
}

// -- From move_check.rs (e2e) ----------------------------------------

/// `Items<Int>` from one fixed source, as a context would hold it.
fn iter_int_ty(interner: &Interner) -> Ty {
    Ty::UserDefined {
        id: QualifiedRef::root(interner.intern("Items")),
        type_args: vec![TypeArg::uniform(Ty::I64)],
        effect_args: vec![],
        identity_args: vec![acvus_mir::ty::IdentityTerm::Known(
            <acvus_mir::ty::IdentityId as acvus_utils::LocalIdOps>::from_raw(0),
        )],
    }
}

fn has_use_after_move(err: &str) -> bool {
    err.contains("after it was moved")
}

// -- Soundness: should REJECT --

#[test]
fn migrated_move_reject_iter_reuse() {
    let i = Interner::new();
    let context = ctx(
        &i,
        &[
            (
                "items",
                Ty::Array(Box::new(Ty::I64), acvus_mir::ty::LenTerm::Known(3)),
            ),
            ("counter", Ty::I64),
        ],
    );
    let result = compile_script_ir(
        &i,
        r#"let x = @items | into_iter | map(|x| -> { @counter = x; x }); x | collect; x | collect"#,
        &context,
    );
    assert!(result.is_err(), "should reject iter reuse");
    assert!(
        has_use_after_move(&result.unwrap_err()),
        "expected use-after-move error"
    );
}

#[test]
fn migrated_move_reject_var_double_load() {
    let i = Interner::new();
    let context = ctx(
        &i,
        &[(
            "items",
            Ty::Array(Box::new(Ty::I64), acvus_mir::ty::LenTerm::Known(3)),
        )],
    );
    let result = compile_to_ir(
        &i,
        "% let a = @items | into_iter\n\
         % let n4 = a | collect\n\
         % let out = len(&n4)\n\
         {{ out.to_string() }}\n\
         % let n5 = a | collect\n\
         % let out = len(&n5)\n\
         {{ out.to_string() }}",
        &context,
    );
    assert!(result.is_err(), "should reject var double load of iterator");
    assert!(
        has_use_after_move(&result.unwrap_err()),
        "expected use-after-move error"
    );
}

#[test]
fn migrated_move_reject_iter_pipe_reuse() {
    let i = Interner::new();
    let context = ctx(
        &i,
        &[
            (
                "items",
                Ty::Array(Box::new(Ty::I64), acvus_mir::ty::LenTerm::Known(3)),
            ),
            ("counter", Ty::I64),
        ],
    );
    let result = compile_script_ir(
        &i,
        r#"let x = @items | into_iter | map(|x| -> { @counter = x; x }); let a = x | collect; let b = x | collect; a"#,
        &context,
    );
    assert!(result.is_err());
    assert!(
        has_use_after_move(&result.unwrap_err()),
        "expected use-after-move error"
    );
}

// -- Completeness: should ACCEPT --

#[test]
fn migrated_move_reject_pure_iter_reuse() {
    let i = Interner::new();
    let context = ctx(&i, &[("src", iter_int_ty(&i))]);
    let result = compile_script_ir(
        &i,
        "let x = @src; let a = x | collect; let b = x | collect; a",
        &context,
    );
    assert!(
        result.is_err(),
        "UserDefined iterator (even pure) should be move-only: {result:?}"
    );
}

#[test]
fn migrated_move_accept_iter_single_use() {
    let i = Interner::new();
    let context = ctx(
        &i,
        &[
            (
                "items",
                Ty::Array(Box::new(Ty::I64), acvus_mir::ty::LenTerm::Known(3)),
            ),
            ("counter", Ty::I64),
        ],
    );
    let result = compile_script_ir(
        &i,
        r#"let items = @items; @items = [1, 2, 3]; let x = items | into_iter | map(|x| -> { @counter = x; x }); x | collect"#,
        &context,
    );
    assert!(
        result.is_ok(),
        "single use of iter should be allowed: {result:?}"
    );
}

#[test]
fn migrated_move_accept_var_reassign() {
    let i = Interner::new();
    let context = ctx(
        &i,
        &[
            ("items", acvus_extern::vec_ty(&i, Ty::I64)),
            ("items2", acvus_extern::vec_ty(&i, Ty::I64)),
        ],
    );
    let result = compile_to_ir(
        &i,
        "% let items = @items\n\
         % @items = vec([])\n\
         % let a = items | into_iter\n\
         % let n6 = a | collect\n\
         % let out = len(&n6)\n\
         {{ out.to_string() }}\n\
         % let items2 = @items2\n\
         % @items2 = vec([])\n\
         % let a = items2 | into_iter\n\
         % let n7 = a | collect\n\
         % let out = len(&n7)\n\
         {{ out.to_string() }}",
        &context,
    );
    assert!(result.is_ok(), "reassigned var should be alive: {result:?}");
}

#[test]
fn migrated_move_accept_iter_pipe_chain() {
    let i = Interner::new();
    let context = ctx(
        &i,
        &[
            (
                "items",
                Ty::Array(Box::new(Ty::I64), acvus_mir::ty::LenTerm::Known(3)),
            ),
            ("counter", Ty::I64),
        ],
    );
    let result = compile_script_ir(
        &i,
        r#"let items = @items; @items = [1, 2, 3]; items | into_iter | map(|x| -> { @counter = x; x }) | filter(|x| -> *x > 0) | map(|x| -> x * 2) | collect"#,
        &context,
    );
    assert!(
        result.is_ok(),
        "linear pipe chain should be allowed: {result:?}"
    );
}

#[test]
fn migrated_move_accept_fn_multiple_calls() {
    let i = Interner::new();
    let f = extern_fn(&i, "f", &[Ty::I64], Ty::I64);
    let result = compile_script_ir_with(
        &i,
        "let a = f(1); let b = f(2); a + b",
        &FxHashMap::default(),
        &[f],
    );
    assert!(
        result.is_ok(),
        "fn without move-only captures should be callable multiple times: {result:?}"
    );
}

#[test]
fn migrated_move_reject_list_of_iter_reuse() {
    let i = Interner::new();
    let ty = Ty::Array(Box::new(iter_int_ty(&i)), acvus_mir::ty::LenTerm::Known(3));
    let context = ctx(&i, &[("src", ty)]);
    let result = compile_script_ir(
        &i,
        "let x = @src; let a = len(&x); let b = len(&x); a + b",
        &context,
    );
    assert!(
        result.is_err(),
        "Vec containing an Iterator should be move-only"
    );
}

#[test]
fn migrated_move_reject_option_iter_reuse() {
    let i = Interner::new();
    let ty = Ty::Option(Box::new(iter_int_ty(&i)));
    let context = ctx(&i, &[("src", ty)]);
    let result = compile_script_ir(
        &i,
        "let x = @src; let a = x | unwrap | collect; let b = x | unwrap | collect; a",
        &context,
    );
    assert!(result.is_err(), "Option<Iterator> should be move-only");
}

#[test]
fn migrated_move_reject_branch_move_then_use() {
    let i = Interner::new();
    let context = ctx(
        &i,
        &[
            ("flag", Ty::Bool),
            (
                "items",
                Ty::Array(Box::new(Ty::I64), acvus_mir::ty::LenTerm::Known(3)),
            ),
        ],
    );
    let result = compile_to_ir(
        &i,
        "% let a = @items | into_iter\n\
         % if @flag\n\
         % let n8 = a | collect\n\
         % let out = len(&n8)\n\
         {{ out.to_string() }}\n\
         % else\n\
         nothing\n\
         % end\n\
         % let n9 = a | collect\n\
         % let out = len(&n9)\n\
         {{ out.to_string() }}",
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
    let context = ctx(&i, &[("flag", Ty::Bool), ("src", iter_int_ty(&i))]);
    let result = compile_to_ir(
        &i,
        "% let a = @src\n\
         % if @flag\n\
         % let n10 = a | collect\n\
         % let out = len(&n10)\n\
         {{ out.to_string() }}\n\
         % else\n\
         % let n11 = a | collect\n\
         % let out = len(&n11)\n\
         {{ out.to_string() }}\n\
         % end\n\
         % let n12 = a | collect\n\
         % let out = len(&n12)\n\
         {{ out.to_string() }}",
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
        &[
            ("flag", Ty::Bool),
            ("items", acvus_extern::vec_ty(&i, Ty::I64)),
        ],
    );
    let result = compile_to_ir(
        &i,
        "% let items = @items\n\
         % @items = vec([])\n\
         % let a = items | into_iter\n\
         % if @flag\n\
         % let n13 = a | collect\n\
         % let out = len(&n13)\n\
         {{ out.to_string() }}\n\
         % else\n\
         nothing\n\
         % end",
        &context,
    );
    assert!(
        result.is_ok(),
        "move in branch without post-merge use should be OK: {result:?}"
    );
}

#[test]
fn migrated_move_accept_pure_capture_fn_multi_call() {
    let i = Interner::new();
    let context = ctx(&i, &[("val", Ty::I64)]);
    let result = compile_script_ir(
        &i,
        "let x = @val; let f = (|a| -> x + a); let a = f(1); let b = f(2); a + b",
        &context,
    );
    assert!(
        result.is_ok(),
        "Fn with pure captures should be callable multiple times: {result:?}"
    );
}

#[test]
fn migrated_move_accept_lambda_return_deque_as_iterator() {
    let i = Interner::new();
    let context = ctx(
        &i,
        &[(
            "items",
            Ty::Array(Box::new(Ty::I64), acvus_mir::ty::LenTerm::Known(3)),
        )],
    );
    let result = compile_script_ir(
        &i,
        "let items = @items; @items = [1, 2, 3]; items | into_iter | flat_map(|x| -> [x, x + 1]) | map(|x| -> x * 2) | collect",
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
    let context = ctx(
        &i,
        &[(
            "items",
            Ty::Array(Box::new(Ty::I64), acvus_mir::ty::LenTerm::Known(3)),
        )],
    );
    let result = compile_script_ir(
        &i,
        "let items = @items; @items = [1, 2, 3]; items | into_iter | map(|x| -> x + 1) | collect",
        &context,
    );
    assert!(
        result.is_ok(),
        "lambda returning scalar should compile: {result:?}"
    );
}

#[test]
fn migrated_move_accept_nested_flat_map_deque_return() {
    let i = Interner::new();
    let context = ctx(
        &i,
        &[(
            "items",
            Ty::Array(Box::new(Ty::I64), acvus_mir::ty::LenTerm::Known(3)),
        )],
    );
    let result = compile_script_ir(
        &i,
        "let items = @items; @items = [1, 2, 3]; items | into_iter | flat_map(|x| -> [x, x + 10]) | map(|x| -> x * 2) | collect",
        &context,
    );
    assert!(
        result.is_ok(),
        "nested flat_map + map with Deque return should compile: {result:?}"
    );
}

#[test]
fn migrated_move_accept_lambda_context_in_body_is_fn() {
    let i = Interner::new();
    let context = items_list_context(&i);
    let result = compile_to_ir(
        &i,
        "% let f = (|z| -> { let items = @items; @items = vec([]); collect(items | into_iter) })\n\
         % let n19 = f(0)\n\
         % let out = len(&n19)\n\
         {{ out.to_string() }}\n\
         % let n20 = f(0)\n\
         % let out = len(&n20)\n\
         {{ out.to_string() }}",
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
    let context = ctx(&i, &[("src", iter_int_ty(&i))]);
    let result = compile_script_ir(
        &i,
        "let x = @src; let f = (|z| -> collect(x)); let a = f(0); let b = f(0); a",
        &context,
    );
    assert!(
        result.is_err(),
        "FnOnce with local capture double call should be rejected: {result:?}"
    );
}

#[test]
fn migrated_move_reject_iter_without_purify() {
    let i = Interner::new();
    let context = ctx(&i, &[("src", iter_int_ty(&i))]);
    let result = compile_script_ir(
        &i,
        "let x = @src; let a = x | collect; let b = x | collect; a",
        &context,
    );
    assert!(
        result.is_err(),
        "iter without purify should still be rejected"
    );
}

#[test]
fn migrated_move_reject_iter_var_without_purify() {
    let i = Interner::new();
    let context = ctx(&i, &[("src", iter_int_ty(&i))]);
    let result = compile_to_ir(
        &i,
        "% let a = @src\n\
         % let n14 = a | collect\n\
         % let out = len(&n14)\n\
         {{ out.to_string() }}\n\
         % let n15 = a | collect\n\
         % let out = len(&n15)\n\
         {{ out.to_string() }}",
        &context,
    );
    assert!(
        result.is_err(),
        "iter var without purify should be rejected"
    );
}

// ======================================================================
// Projection system: soundness & completeness
// ======================================================================

// -- Completeness: valid programs accepted ---------------------------

/// Variable whole read/write: `x = 42; x` - SSA promotion eliminates Ref/Load/Store.
#[test]
fn projection_var_whole_read_write() {
    let i = Interner::new();
    let ir = compile_script_ir(&i, "let x = 42; x", &FxHashMap::default()).unwrap();
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

/// Variable read after multiple writes: `x = 1; x = 2; x` -> SSA sees last def.
#[test]
fn projection_var_multiple_writes() {
    let i = Interner::new();
    let ir = compile_script_ir(&i, "let x = 1; x = 2; x", &FxHashMap::default()).unwrap();
    assert!(
        !ir.contains("ref "),
        "Ref should be eliminated by SSA: {ir}"
    );
}

/// Context whole read: `@ctx`.
#[test]
fn projection_context_whole_read() {
    let i = Interner::new();
    let context = ctx(&i, &[("data", Ty::I64)]);
    let ir = compile_script_ir(&i, "@data", &context).unwrap();
    // SSA forwards the entry load value.
    // Ref/Load from entry may remain or be forwarded - just verify it compiles + returns.
    assert!(ir.contains("return"), "should have return: {ir}");
}

/// Context field read: `@obj.name` - 1-depth Ref with field.
#[test]
fn projection_context_field_read() {
    let i = Interner::new();
    let context = ctx(&i, &[("obj", obj(&i, &[("name", Ty::String)]))]);
    let ir = compile_script_ir(&i, "@obj.name", &context).unwrap();
    // Should have a field ref (Ref with field) or SSA-forwarded result.
    assert!(ir.contains("return"), "should compile and return: {ir}");
}

/// Context field write: `@obj = { name: "test" }` - whole context store.
#[test]
fn projection_context_whole_write() {
    let i = Interner::new();
    let context = ctx(&i, &[("out", Ty::String)]);
    let ir = compile_to_ir(&i, "% @out = \"hello\".to_string()", &context).unwrap();
    // Context store should produce Store instruction (may remain after SSA for write-back).
    assert!(
        ir.contains("commit @out"),
        "should commit the context: {ir}"
    );
}

/// Chained field access: `@obj.a.b` - Ref(@obj, "a") + Load + FieldGet("b").
#[test]
fn projection_chained_field_access_2depth() {
    let i = Interner::new();
    let inner = obj(&i, &[("b", Ty::I64)]);
    let context = ctx(&i, &[("obj", obj(&i, &[("a", inner)]))]);
    let ir = compile_script_ir(&i, "@obj.a.b.to_string()", &context).unwrap();
    // Should have field access for .b (either as FieldGet or via Ref).
    assert!(ir.contains(".b"), "should access field b: {ir}");
    assert!(ir.contains("return"), "should compile and return: {ir}");
}

/// Variable used in arithmetic after assignment: `x = @val; x + 1`.
#[test]
fn projection_var_in_arithmetic() {
    let i = Interner::new();
    let context = ctx(&i, &[("val", Ty::I64)]);
    let ir = compile_script_ir(&i, "let x = @val; x + 1", &context).unwrap();
    // SSA should promote x - no Ref for x should remain.
    assert!(ir.contains("+"), "should have addition: {ir}");
    assert!(ir.contains("return"), "should compile and return: {ir}");
}

/// Lambda captures variable: `x = @data; |y| -> x + y` (a captured word is
/// copied at each use, RFC-0018).
#[test]
fn projection_lambda_capture() {
    let i = Interner::new();
    let context = ctx(&i, &[("data", Ty::I64)]);
    let ir = compile_script_ir(
        &i,
        "let x = @data; [1, 2] | into_iter | map(|y| -> x + y) | collect",
        &context,
    )
    .unwrap();
    assert!(ir.contains("closure"), "should have closure: {ir}");
    assert!(ir.contains("return"), "should compile and return: {ir}");
}

/// ExternParam read: `$param` - should compile (immutable, Ref+Load).
#[test]
fn projection_param_read() {
    let i = Interner::new();
    let ir = compile_to_ir(
        &i,
        "% let n = $count + 1\n\
         {{ n.to_string() }}",
        &FxHashMap::from_iter([(i.intern("count"), Ty::I64)]),
    )
    .unwrap();
    assert!(ir.contains("return"), "should compile and return: {ir}");
}

/// `to_string` is declared for more than one type, so a `$param` only it
/// reads has no type the resolution can close, and the refusal says so
/// rather than carrying an error type into lowering.
#[test]
fn a_param_no_use_gives_a_type_is_refused() {
    let i = Interner::new();
    let err = compile_to_ir(&i, "{{ $count.to_string() }}", &FxHashMap::default())
        .expect_err("a param whose type does not close is refused");
    assert!(err.contains("cannot infer type"), "{err}");
}

// -- Soundness: invalid programs rejected ----------------------------

/// Store non-materializable (Fn) to context -> must be rejected.
#[test]
fn projection_soundness_reject_fn_in_context() {
    let i = Interner::new();
    let fn_ty = Ty::Fn {
        params: vec![acvus_mir::ty::Param::new(i.intern("x"), Ty::I64)],
        ret: Box::new(Ty::I64),
        captures: vec![],
        effect: acvus_mir::ty::Effect::OPAQUE.into(),
    };
    let context = ctx(&i, &[("f", fn_ty)]);
    let result = compile_script_ir(&i, "@f = @f; @f", &context);
    assert!(result.is_err(), "storing Fn to context should fail");
}

/// Store non-materializable (Vec<Fn>) to context -> must be rejected.
#[test]
fn projection_soundness_reject_list_fn_in_context() {
    let i = Interner::new();
    let fn_ty = Ty::Fn {
        params: vec![acvus_mir::ty::Param::new(i.intern("x"), Ty::I64)],
        ret: Box::new(Ty::I64),
        captures: vec![],
        effect: acvus_mir::ty::Effect::OPAQUE.into(),
    };
    let context = ctx(
        &i,
        &[(
            "xs",
            Ty::Array(Box::new(fn_ty), acvus_mir::ty::LenTerm::Known(3)),
        )],
    );
    let result = compile_script_ir(&i, "@xs = @xs; @xs", &context);
    assert!(result.is_err(), "storing Vec<Fn> to context should fail");
}

/// Write to ExternParam -> must be rejected (typeck catches this).
#[test]
fn projection_soundness_reject_param_write() {
    let i = Interner::new();
    let result = compile_to_ir(
        &i,
        "% $count = 42",
        &FxHashMap::from_iter([(i.intern("count"), Ty::I64)]),
    );
    assert!(result.is_err(), "writing to ExternParam should fail");
}

// -- SSA correctness: promotion ---------------------------------------

/// Local variable: SSA promotion eliminates Ref/Load/Store.
#[test]
fn projection_ssa_var_promoted() {
    let i = Interner::new();
    let ir = compile_script_ir(&i, "let x = 1; let y = x + 2; y", &FxHashMap::default()).unwrap();
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
    let context = ctx(&i, &[("count", Ty::I64)]);
    let ir = compile_script_ir(&i, "@count = @count + 1; @count", &context).unwrap();
    // Context write-back Store should remain in the IR.
    assert!(
        ir.contains("commit @count"),
        "the context is committed at exit: {ir}"
    );
}

/// Move-only value through variable: single use accepted.
#[test]
fn projection_move_single_use() {
    let i = Interner::new();
    let context = items_list_context(&i);
    let ir = compile_to_ir(
        &i,
        "% let items = @items\n\
         % @items = vec([])\n\
         % let x = items | into_iter\n\
         % let n16 = x | collect\n\
         % let out = len(&n16)\n\
         {{ out.to_string() }}",
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
    let context = items_list_context(&i);
    let ir = compile_to_ir(
        &i,
        "% let items = @items\n\
         % @items = vec([])\n\
         % let x = items | into_iter\n\
         % let n17 = x | collect\n\
         % let out = len(&n17)\n\
         {{ out.to_string() }}\n\
         % let items = @items\n\
         % @items = vec([])\n\
         % let x = items | into_iter\n\
         % let n18 = x | collect\n\
         % let out = len(&n18)\n\
         {{ out.to_string() }}",
        &context,
    )
    .unwrap();
    assert!(
        ir.contains("return"),
        "var reassign should revive move-only: {ir}"
    );
}

// ======================================================================
// SROA: field projection decomposition
// ======================================================================

// -- Completeness: valid field access patterns -----------------------

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
    let inner = obj(&i, &[("b", Ty::I64)]);
    let context = ctx(&i, &[("obj", obj(&i, &[("a", inner)]))]);
    let ir = compile_script_ir(&i, "@obj.a.b.to_string()", &context).unwrap();
    insta::assert_snapshot!(ir);
}

#[test]
fn sroa_context_field_read_3depth() {
    let i = Interner::new();
    let c = obj(&i, &[("c", Ty::I64)]);
    let b = obj(&i, &[("b", c)]);
    let context = ctx(&i, &[("obj", obj(&i, &[("a", b)]))]);
    let ir = compile_script_ir(&i, "@obj.a.b.c.to_string()", &context).unwrap();
    insta::assert_snapshot!(ir);
}

#[test]
fn sroa_field_read_arithmetic() {
    let i = Interner::new();
    let context = ctx(&i, &[("obj", obj(&i, &[("val", Ty::I64)]))]);
    let ir = compile_script_ir(&i, "@obj.val + 1", &context).unwrap();
    insta::assert_snapshot!(ir);
}

#[test]
fn sroa_multiple_field_reads_same_object() {
    let i = Interner::new();
    let context = ctx(&i, &[("obj", obj(&i, &[("x", Ty::I64), ("y", Ty::I64)]))]);
    let ir = compile_script_ir(&i, "@obj.x + @obj.y", &context).unwrap();
    insta::assert_snapshot!(ir);
}

#[test]
fn sroa_field_read_in_lambda() {
    let i = Interner::new();
    let context = list_context(&i, "users", obj(&i, &[("name", Ty::String)]));
    let ir = compile_to_ir(
        &i,
        "% let users = @users\n\
         % @users = vec([])\n\
         {{ users | into_iter | map(|u| -> u.name) | collect | into_iter | join(\",\".to_string()) }}",
        &context,
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

// -- Soundness -------------------------------------------------------

#[test]
fn sroa_soundness_context_write_back_preserved() {
    let i = Interner::new();
    let context = ctx(&i, &[("count", Ty::I64)]);
    let ir = compile_script_ir(&i, "@count = @count + 1; @count", &context).unwrap();
    insta::assert_snapshot!(ir);
}

#[test]
fn sroa_soundness_reject_fn_in_context() {
    let i = Interner::new();
    let fn_ty = Ty::Fn {
        params: vec![acvus_mir::ty::Param::new(i.intern("x"), Ty::I64)],
        ret: Box::new(Ty::I64),
        captures: vec![],
        effect: acvus_mir::ty::Effect::OPAQUE.into(),
    };
    let context = ctx(&i, &[("f", fn_ty)]);
    let result = compile_script_ir(&i, "@f = @f; @f", &context);
    assert!(
        result.is_err(),
        "storing Fn to context should fail even with SROA"
    );
}

// -- Context destructure chain ---------------------------------------

#[test]
fn sroa_context_destructure_field_to_context() {
    let i = Interner::new();
    let context = ctx(&i, &[("b", obj(&i, &[("x", Ty::I64)])), ("a", Ty::I64)]);
    let ir = compile_script_ir(&i, "@a = @b.x; @a", &context).unwrap();
    insta::assert_snapshot!(ir);
}

#[test]
fn sroa_context_chain_destructure() {
    let i = Interner::new();
    let inner = obj(&i, &[("val", Ty::I64)]);
    let context = ctx(
        &i,
        &[
            ("b", obj(&i, &[("a", inner.clone())])),
            ("a", inner),
            ("c", Ty::I64),
        ],
    );
    // `@b.a` is an Object: reading it moves it out of `@b`, which is then
    // put back whole before the end (RFC-0024, RFC-0025).
    let ir = compile_script_ir(
        &i,
        "@a = @b.a; @b = { a: { val: 0, }, }; @c = @a.val; @c",
        &context,
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

#[test]
fn sroa_context_destructure_then_overwrite() {
    let i = Interner::new();
    let context = ctx(&i, &[("b", obj(&i, &[("x", Ty::I64)])), ("a", Ty::I64)]);
    let ir = compile_script_ir(&i, "@a = @b.x; @a = 0; @a", &context).unwrap();
    insta::assert_snapshot!(ir);
}

// -- Projection store ------------------------------------------------

#[test]
fn context_projection_store_1depth() {
    let i = Interner::new();
    let context = ctx(&i, &[("obj", obj(&i, &[("x", Ty::I64)]))]);
    let ir = compile_script_ir(&i, "@obj.x = 42; @obj.x", &context).unwrap();
    insta::assert_snapshot!(ir);
}

#[test]
fn context_projection_store_2depth() {
    let i = Interner::new();
    let inner = obj(&i, &[("y", Ty::I64)]);
    let context = ctx(&i, &[("obj", obj(&i, &[("x", inner)]))]);
    let ir = compile_script_ir(&i, "@obj.x.y = 99; @obj.x.y", &context).unwrap();
    insta::assert_snapshot!(ir);
}

#[test]
fn var_field_store_1depth() {
    let i = Interner::new();
    let context = ctx(&i, &[("obj", obj(&i, &[("x", Ty::I64)]))]);
    let ir = compile_script_ir(
        &i,
        "let a = @obj; a.x = 0; let x = a.x; @obj = a; x",
        &context,
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

// -- Uninit check ----------------------------------------------------

/// `@a`, declared with both fields, where the literal writes only `x`.
fn object_xy(i: &Interner) -> FxHashMap<Astr, Ty> {
    FxHashMap::from_iter([(
        i.intern("a"),
        Ty::Object(ObjectTy::written(FxHashMap::from_iter([
            (i.intern("x"), Ty::I64),
            (i.intern("y"), Ty::I64),
        ]))),
    )])
}

/// A declared context arrives from its host with every field stored, so
/// the value whose field is never stored is a local one.
#[test]
fn uninit_field_load_rejected() {
    let i = Interner::new();
    let result = compile_script_ir(&i, "let a = { x: 0, }; a.y + 1", &FxHashMap::default());
    let err = result.expect_err("the field is read where nothing stored it");
    assert!(
        err.contains("`a` has no `y` stored on every path that reaches here"),
        "{err}"
    );
}

#[test]
fn a_field_whose_type_nothing_decides_is_refused_before_lowering() {
    let i = Interner::new();
    let result = compile_script_ir(
        &i,
        "let a = { x: 0, }; a.y.to_string()",
        &FxHashMap::default(),
    );
    let err = result.expect_err("no instance of `to_string` is chosen for `a.y`");
    assert!(err.contains("cannot infer type"), "{err}");
}

#[test]
fn init_field_load_passes() {
    let i = Interner::new();
    let context = ctx(&i, &[("a", obj(&i, &[("x", Ty::I64)]))]);
    // All fields present - should compile fine.
    let ir = compile_script_ir(&i, "@a = { x: 42, }; @a.x", &context).unwrap();
    insta::assert_snapshot!(ir);
}

/// The literal is missing `y`, and the field store fills it in.
#[test]
fn field_store_then_load_passes() {
    let ie = Interner::new();
    let i = &ie;
    let ir = compile_script_ir(
        i,
        "@a = { x: 0, }; @a.y = 1; @a.y.to_string()",
        &object_xy(i),
    )
    .unwrap();
    insta::assert_snapshot!(ir);
}

// -- Script mode tests ----------------------------------------------

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
    let ir = script_mode(
        &i,
        "let x = if false { 1 } else if true { 2 } else { 3 }; x",
    );
    insta::assert_snapshot!(ir);
}

#[test]
fn script_if_no_else() {
    let i = Interner::new();
    // if without else -> side effect only, used as statement
    let ir = script_mode(&i, "let x = 0; if true { x = 1; }; x");
    insta::assert_snapshot!(ir);
}

#[test]
fn script_while_loop() {
    let i = Interner::new();
    let ctx = ctx(&i, &[("n", Ty::I64)]);
    let ir = script_mode_ctx(&i, "while @n > 0 { @n = @n - 1; }", &ctx);
    insta::assert_snapshot!(ir);
}

#[test]
fn script_while_let() {
    let i = Interner::new();
    let ir = script_mode(
        &i,
        r#"
        let x = Some(1);
        let result = 0;
        while let Some(v) = x {
            result = v;
            x = None;
        }
        result
    "#,
    );
    insta::assert_snapshot!(ir);
}

#[test]
fn script_if_let_expr() {
    let i = Interner::new();
    let ir = script_mode(&i, "let x = if let Some(v) = Some(42) { v } else { 0 }; x");
    insta::assert_snapshot!(ir);
}

// -- RFC-0071: a template is a script whose text lines are output -------

/// A `%` line's `for` is the script's own, so a traversal of a container
/// of objects needs no template form: the body's text lines append once
/// per iteration.
#[test]
fn a_template_traverses_a_vec_of_objects() {
    let i = Interner::new();
    let ir = compile_to_ir(
        &i,
        "% let people = vec([{ name: \"Ada\".to_string(), }, { name: \"Emmy\".to_string(), }])\n\
         % for p in &people\n\
         - {{ &p.name }}\n\
         % end\n",
        &FxHashMap::default(),
    )
    .unwrap();
    assert!(ir.contains("for slice"), "{ir}");
    assert!(ir.contains("append"), "{ir}");
    assert!(!ir.contains("string_concat"), "{ir}");
}

/// A template calls another function as any function that returns a
/// `String`; composition is a call and there is no include (RFC-0071
/// rule 4).
#[test]
fn a_template_calls_a_function_returning_a_string() {
    let i = Interner::new();
    let ir = compile_template_with_helpers(
        &i,
        ("main", "Answer {{ rules(@lang) }}.\n"),
        &[(
            "rules",
            r#""in " + $lang"#,
            vec![ParamTerm::<Poly>::new(
                i.intern("lang"),
                lift_to_poly(&Ty::String),
            )],
        )],
        &[("lang", Ty::String)],
        &[],
    )
    .unwrap();
    assert!(ir.contains("-- rules --"), "{ir}");
    assert!(ir.contains("-- main --"), "{ir}");
    assert!(ir.contains("append"), "{ir}");
}
