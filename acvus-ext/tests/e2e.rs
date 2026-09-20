//! E2E tests for ext functions: compile script with ext registry -> execute -> check result.

use std::collections::HashMap;
use std::sync::Arc;

use acvus_ext::*;
use acvus_extern::{
    Arr, ExternType, Externs, Owned, Registry, Var, extern_fn, extern_registry, kind,
};
use acvus_interpreter::AcvusRuntime;
use acvus_interpreter::*;
use acvus_mir::graph::*;
use acvus_mir::graph::{extract, infer, lower as graph_lower, optimize as graph_optimize};
use acvus_mir::ty::{Ty, lift_to_poly};
use acvus_utils::{Astr, Freeze, Interner};
use base64::Engine;
use rustc_hash::FxHashMap;

type TypedContext = FxHashMap<Astr, (Ty, Value)>;

/// Compile + execute a script with the std registries and `registries`.
async fn run_ext(
    interner: &Interner,
    source: &str,
    context: TypedContext,
    registries: Vec<Registry<AcvusRuntime>>,
) -> Value {
    let ast = ParsedAst::Script(acvus_ast::parse_script(interner, source).expect("parse"));
    run_parsed(interner, ast, context, registries).await
}

async fn run_ext_script_mode(
    interner: &Interner,
    source: &str,
    context: TypedContext,
    registries: Vec<Registry<AcvusRuntime>>,
) -> Value {
    let ast = ParsedAst::Script(acvus_ast::parse_script(interner, source).expect("parse"));
    run_parsed(interner, ast, context, registries).await
}

async fn run_ext_template(
    interner: &Interner,
    source: &str,
    context: TypedContext,
    registries: Vec<Registry<AcvusRuntime>>,
) -> Value {
    let ast = ParsedAst::Template(acvus_ast::parse(interner, source).expect("parse"));
    run_parsed(interner, ast, context, registries).await
}

async fn run_parsed(
    interner: &Interner,
    ast: ParsedAst,
    context: TypedContext,
    registries: Vec<Registry<AcvusRuntime>>,
) -> Value {
    let mut all_registries = std_registries::<AcvusRuntime>();
    all_registries.extend(registries);
    let Externs {
        mut functions,
        types: type_registry,
        handlers,
        instances,
        ..
    } = Externs::combine(all_registries, interner).expect("registries combine");

    let contexts: Vec<Context> = context
        .iter()
        .map(|(name, (ty, _))| Context {
            qref: QualifiedRef::root(*name),
            ty: lift_to_poly(ty),
        })
        .collect();

    let entry_qref = QualifiedRef::root(interner.intern("test"));
    {
        let mut pb = acvus_mir::ty::PolyBuilder::new();
        functions.push(Function {
            qref: entry_qref,
            kind: FnKind::Local(ast),
            ty: acvus_mir::ty::PolyTy::Fn {
                params: vec![],
                ret: Box::new(pb.fresh_ty_var()),
                captures: vec![],
                effect: acvus_mir::ty::Effect::OPAQUE.into(),
            },
        });
    }

    let graph = CompilationGraph {
        functions: Freeze::new(functions),
        contexts: Freeze::new(contexts),
        entry: Some(entry_qref),
    };

    let ext = extract::extract(interner, &graph);
    let inf = infer::infer(
        interner,
        &graph,
        &ext,
        &FxHashMap::default(),
        Freeze::new(type_registry),
    );
    let lowered = graph_lower::lower(interner, &graph, &ext, &inf);

    let errs: Vec<String> = inf
        .errors()
        .into_iter()
        .flat_map(|(_, errs)| errs.iter())
        .chain(lowered.errors.iter().flat_map(|e| e.errors.iter()))
        .map(|e| format!("{}", e.display(interner)))
        .collect();
    if !errs.is_empty() {
        panic!("compile failed: {}", errs.join("; "));
    }

    // Drops are inserted by the optimize pipeline and nowhere else, and a
    // value the frame does not drop trips `Machine::define_slot` when the
    // slot is written again. A run here is the run the CLI does.
    let result = graph_optimize::optimize(
        lowered.modules.into_iter().collect(),
        &FxHashMap::default(),
        graph_optimize::Opt::Full,
    );
    assert!(
        result.errors.is_empty(),
        "validation failed: {:?}",
        result.errors
    );

    let mut exec_fns: FxHashMap<QualifiedRef, Executable> = handlers
        .into_iter()
        .map(|(q, h)| (q, Executable::Extern(h)))
        .collect();
    let context_names: FxHashMap<QualifiedRef, Astr> = graph
        .contexts
        .iter()
        .map(|ctx| (ctx.qref, ctx.qref.name))
        .collect();
    let prepare_ctx = PrepareCtx {
        interner,
        externs: &exec_fns,
        context_names: &context_names,
        instances: &instances,
    };
    let prepared: Vec<(QualifiedRef, Executable)> = result
        .modules
        .iter()
        .map(|(qref, module)| {
            let prepared = prepare_module(module, &prepare_ctx);
            (*qref, Executable::Module(std::sync::Arc::new(prepared)))
        })
        .collect();
    exec_fns.extend(prepared);
    let snapshot: HashMap<String, Owned<AcvusRuntime>> = context
        .into_iter()
        .map(|(k, (_, v))| (interner.resolve(k).to_string(), Owned::from_value(v)))
        .collect();

    let executor = Arc::new(SequentialExecutor);
    let shared =
        InterpreterContext::new(interner, exec_fns, executor).with_context_names(context_names);
    let page = InMemoryContext::new(snapshot);
    let mut interp = Interpreter::new(shared, entry_qref, page);
    interp.execute().await
}

fn assert_str(v: &Value, expected: &str) {
    assert!(v.is_string(), "expected a String, got {v:?}");
    // SAFETY: the witness is String.
    assert_eq!(unsafe { v.as_str() }, expected);
}

fn strings_of(v: Value) -> Vec<String> {
    // SAFETY: `collect` returns a `Vec<T>`, whose store is the run of
    // `Owned` the element type erases to (RFC-0048 §1).
    let list: Vec<Owned<AcvusRuntime>> = unsafe { v.materialize() };
    list.iter()
        .map(|item| {
            assert!(item.is_string(), "expected a String, got {item:?}");
            // SAFETY: the witness is String.
            unsafe { item.as_str() }.to_owned()
        })
        .collect()
}

// =======================================================================
//  Regex
// =======================================================================

#[tokio::test]
async fn regex_match_true() {
    let i = Interner::new();
    let result = run_ext_script_mode(
        &i,
        r#"if let Ok(re) = regex("\\d+".to_string()) { let t = "abc123"; is_match(&re, &t) } else { false }"#,
        TypedContext::default(),
        vec![regex_registry::<AcvusRuntime>()],
    )
    .await;
    assert!(result.as_bool());
}

#[tokio::test]
async fn regex_match_false() {
    let i = Interner::new();
    let result = run_ext_script_mode(
        &i,
        r#"if let Ok(re) = regex("\\d+".to_string()) { let t = "abc"; is_match(&re, &t) } else { true }"#,
        TypedContext::default(),
        vec![regex_registry::<AcvusRuntime>()],
    )
    .await;
    assert!(!result.as_bool());
}

#[tokio::test]
async fn regex_find_all_collect() {
    let i = Interner::new();
    let result = run_ext_script_mode(
        &i,
        r#"if let Ok(re) = regex("\\d+".to_string()) { let t = "a1b22c333"; find_all(&re, &t) | map(|m| -> m.text) | collect } else { vec([]) }"#,
        TypedContext::default(),
        vec![regex_registry::<AcvusRuntime>()],
    )
    .await;
    assert_eq!(strings_of(result), vec!["1", "22", "333"]);
}

#[tokio::test]
async fn regex_replace() {
    let i = Interner::new();
    let result = run_ext_script_mode(
        &i,
        r#"if let Ok(re) = regex("\\s+".to_string()) { let t = "hello   world"; replace_all(&re, &t, " ".to_string()) } else { "?".to_string() }"#,
        TypedContext::default(),
        vec![regex_registry::<AcvusRuntime>()],
    )
    .await;
    assert_str(&result, "hello world");
}

#[tokio::test]
async fn regex_split_collect() {
    let i = Interner::new();
    let result = run_ext_script_mode(
        &i,
        r#"if let Ok(re) = regex("[,;]\\s*".to_string()) { let t = "a, b;c"; split(&re, &t) | collect } else { vec([]) }"#,
        TypedContext::default(),
        vec![regex_registry::<AcvusRuntime>()],
    )
    .await;
    assert_eq!(strings_of(result), vec!["a", "b", "c"]);
}

// =======================================================================
//  Encoding
// =======================================================================

#[tokio::test]
async fn base64_roundtrip() {
    let i = Interner::new();
    let result = run_ext_script_mode(
        &i,
        r#"if let Ok(s) = base64_decode(base64_encode("hello world".to_string())) { s } else { "not decoded".to_string() }"#,
        TypedContext::default(),
        vec![encoding_registry::<AcvusRuntime>()],
    )
    .await;
    assert_str(&result, "hello world");
}

#[tokio::test]
async fn base64_that_does_not_decode_names_which_decoding_failed() {
    let i = Interner::new();
    let src = |input: &str| {
        [
            &format!(r#"{{{{ r = base64_decode("{input}".to_string()) }}}}"#),
            "{{ Ok(s) = r }}{{ s }}",
            "{{ Err(Base64Error::InvalidBase64(e)) = }}not base64: {{ e.input }}",
            "{{ Err(Base64Error::InvalidUtf8(e)) = }}not utf8: {{ e.input }}{{/}}",
        ]
        .concat()
    };
    let regs = || vec![encoding_registry::<AcvusRuntime>()];
    let v = run_ext_template(&i, &src("!!!!"), TypedContext::default(), regs()).await;
    assert_str(&v, "not base64: !!!!");
    let not_utf8 = base64::engine::general_purpose::STANDARD.encode([0xFF]);
    let v = run_ext_template(&i, &src(&not_utf8), TypedContext::default(), regs()).await;
    assert_str(&v, &format!("not utf8: {not_utf8}"));
}

#[tokio::test]
async fn url_roundtrip() {
    let i = Interner::new();
    let result = run_ext(
        &i,
        r#"url_decode(url_encode("hello world&foo=bar".to_string()))"#,
        TypedContext::default(),
        vec![encoding_registry::<AcvusRuntime>()],
    )
    .await;
    assert_str(&result, "hello world&foo=bar");
}

// =======================================================================
//  DateTime
// =======================================================================

#[tokio::test]
async fn datetime_format_from_timestamp() {
    let i = Interner::new();
    // 2024-01-01 00:00:00 UTC = epoch 1704067200
    let result = run_ext_script_mode(
        &i,
        r#"if let Ok(dt) = from_timestamp(1704067200) { format_date(dt, "%Y-%m-%d".to_string()) } else { "?".to_string() }"#,
        TypedContext::default(),
        vec![datetime_registry::<AcvusRuntime>()],
    )
    .await;
    assert_str(&result, "2024-01-01");
}

#[tokio::test]
async fn datetime_timestamp_roundtrip() {
    let i = Interner::new();
    let result = run_ext_script_mode(
        &i,
        r#"if let Ok(dt) = from_timestamp(1704067200) { timestamp(dt) } else { -1 }"#,
        TypedContext::default(),
        vec![datetime_registry::<AcvusRuntime>()],
    )
    .await;
    assert_eq!(result.as_int(), 1704067200);
}

#[tokio::test]
async fn datetime_add_days() {
    let i = Interner::new();
    let result = run_ext_script_mode(
        &i,
        r#"if let Ok(dt) = from_timestamp(1704067200) { let dt2 = add_days(dt, 1); format_date(dt2, "%Y-%m-%d".to_string()) } else { "?".to_string() }"#,
        TypedContext::default(),
        vec![datetime_registry::<AcvusRuntime>()],
    )
    .await;
    assert_str(&result, "2024-01-02");
}

#[tokio::test]
async fn datetime_parse_and_format() {
    let i = Interner::new();
    let result = run_ext_script_mode(
        &i,
        r#"if let Ok(dt) = parse_date("2024-06-15 12:30:00".to_string(), "%Y-%m-%d %H:%M:%S".to_string()) { format_date(dt, "%m/%d/%Y".to_string()) } else { "?".to_string() }"#,
        TypedContext::default(),
        vec![datetime_registry::<AcvusRuntime>()],
    ).await;
    assert_str(&result, "06/15/2024");
}

// =======================================================================
//  Multiple registries
// =======================================================================

#[tokio::test]
async fn mixed_regex_and_encoding() {
    let i = Interner::new();
    let result = run_ext_script_mode(
        &i,
        r#"if let Ok(re) = regex("\\d+".to_string()) { let t = "abc123"; let m = is_match(&re, &t); base64_encode("hello".to_string()) + " " + m.to_string() } else { "?".to_string() }"#,
        TypedContext::default(),
        vec![
            regex_registry::<AcvusRuntime>(),
            encoding_registry::<AcvusRuntime>(),
        ],
    )
    .await;
    assert_str(&result, "aGVsbG8= true");
}

// =======================================================================
//  ExternCast - coercion via registered CastRule
// =======================================================================

#[derive(ExternType)]
#[repr(transparent)]
struct MyNum(i64);

#[extern_fn(effect = pure)]
fn make_num() -> MyNum {
    MyNum(42)
}

#[extern_fn(effect = pure)]
#[extern_cast]
fn num_to_int(n: MyNum) -> i64 {
    n.0
}

#[extern_fn(effect = pure)]
fn double(n: i64) -> i64 {
    n * 2
}

fn extern_cast_registry() -> Registry<AcvusRuntime> {
    extern_registry! {
        ns: "t",
        types: [MyNum],
        fns: [make_num, num_to_int, double],
    }
}

#[tokio::test]
async fn extern_cast_auto_coercion() {
    let i = Interner::new();
    let result = run_ext(
        &i,
        "double(make_num())",
        TypedContext::default(),
        vec![extern_cast_registry()],
    )
    .await;
    assert_eq!(result.as_int(), 84);
}

// =======================================================================
//  Objects across the boundary (RFC-0032)
// =======================================================================

#[derive(acvus_extern::TyArg)]
struct Pt {
    x: i64,
    label: String,
}

#[extern_fn(effect = pure)]
fn make_pt() -> Pt {
    Pt {
        x: 1,
        label: "one".to_owned(),
    }
}

#[extern_fn(effect = pure)]
fn shift(p: Pt) -> Pt {
    Pt {
        x: p.x + 1,
        label: format!("{}!", p.label),
    }
}

#[extern_fn(effect = pure)]
fn pts() -> Vec<Pt> {
    vec![
        Pt {
            x: 10,
            label: "a".to_owned(),
        },
        Pt {
            x: 20,
            label: "b".to_owned(),
        },
    ]
}

#[extern_fn(effect = pure)]
fn total(ps: Vec<Pt>) -> i64 {
    ps.iter().map(|p| p.x).sum()
}

#[extern_fn(effect = pure)]
fn maybe_pt(some: bool) -> Option<Pt> {
    some.then(|| Pt {
        x: 7,
        label: "seven".to_owned(),
    })
}

fn object_registry() -> Registry<AcvusRuntime> {
    extern_registry! {
        ns: "t",
        fns: [make_pt, shift, pts, total, maybe_pt],
    }
}

#[tokio::test]
async fn an_object_returned_by_an_extern_fn_is_the_script_s_object() {
    let i = Interner::new();
    let regs = || vec![object_registry()];
    let v = run_ext(&i, "make_pt().x", TypedContext::default(), regs()).await;
    assert_eq!(v.as_int(), 1);
    let v = run_ext(
        &i,
        "shift({ x: 41, label: \"a\".to_string(), }).x",
        TypedContext::default(),
        regs(),
    )
    .await;
    assert_eq!(v.as_int(), 42);
    let v = run_ext(
        &i,
        "shift(make_pt()).label",
        TypedContext::default(),
        regs(),
    )
    .await;
    assert_str(&v, "one!");
}

#[tokio::test]
async fn a_container_of_objects_converts_each_element() {
    let i = Interner::new();
    let regs = || vec![object_registry()];
    let v = run_ext(&i, "total(pts())", TypedContext::default(), regs()).await;
    assert_eq!(v.as_int(), 30);
    let v = run_ext(
        &i,
        "let ps = pts(); ps[1].x",
        TypedContext::default(),
        regs(),
    )
    .await;
    assert_eq!(v.as_int(), 20);
    let v = run_ext(
        &i,
        "let ps = pts(); ps.len()",
        TypedContext::default(),
        regs(),
    )
    .await;
    assert_eq!(v.as_int(), 2);
    let v = run_ext(
        &i,
        "let ps = pts(); ps.as_iter().map(|p| -> p.x).fold(0, |a, x| -> a + x)",
        TypedContext::default(),
        regs(),
    )
    .await;
    assert_eq!(v.as_int(), 30);
    let v = run_ext(
        &i,
        "unwrap(maybe_pt(true)).label",
        TypedContext::default(),
        regs(),
    )
    .await;
    assert_str(&v, "seven");
    let v = run_ext(
        &i,
        "unwrap_or(maybe_pt(false), make_pt()).x",
        TypedContext::default(),
        regs(),
    )
    .await;
    assert_eq!(v.as_int(), 1);
}

#[derive(acvus_extern::TyArg)]
enum Shape {
    Dot,
    Circle(i64),
    Rect { w: i64, h: i64 },
}

#[extern_fn(effect = pure)]
fn shape(kind: i64) -> Shape {
    match kind {
        0 => Shape::Dot,
        1 => Shape::Circle(5),
        _ => Shape::Rect { w: 2, h: 3 },
    }
}

#[extern_fn(effect = pure)]
fn area(s: Shape) -> i64 {
    match s {
        Shape::Dot => 0,
        Shape::Circle(r) => r * r,
        Shape::Rect { w, h } => w * h,
    }
}

fn enum_registry() -> Registry<AcvusRuntime> {
    extern_registry! {
        ns: "t",
        fns: [shape, area],
    }
}

#[tokio::test]
async fn an_enum_returned_by_an_extern_fn_is_matched_by_the_script() {
    let i = Interner::new();
    let regs = || vec![enum_registry()];
    let src = |kind: i64| {
        [
            &format!("{{{{ s = shape({kind}) }}}}"),
            "{{ Shape::Circle(r) = s }}{{ r.to_string() }}",
            "{{ Shape::Rect(d) = }}{{ d.w.to_string() }}x{{ d.h.to_string() }}",
            "{{ Shape::Dot = }}dot{{_}}?{{/}}",
        ]
        .concat()
    };
    let v = run_ext_template(&i, &src(0), TypedContext::default(), regs()).await;
    assert_str(&v, "dot");
    let v = run_ext_template(&i, &src(1), TypedContext::default(), regs()).await;
    assert_str(&v, "5");
    let v = run_ext_template(&i, &src(2), TypedContext::default(), regs()).await;
    assert_str(&v, "2x3");
}

#[tokio::test]
async fn an_enum_built_by_the_script_crosses_into_the_extern_fn() {
    let i = Interner::new();
    let regs = || vec![enum_registry()];
    let v = run_ext_template(
        &i,
        "{{ a = area(Shape::Circle(3)) }}{{ a.to_string() }}",
        TypedContext::default(),
        regs(),
    )
    .await;
    assert_str(&v, "9");
    let v = run_ext_template(
        &i,
        "{{ a = area(shape(2)) }}{{ a.to_string() }}",
        TypedContext::default(),
        regs(),
    )
    .await;
    assert_str(&v, "6");
}

#[derive(acvus_extern::TyArg)]
struct Price {
    amount: Decimal,
    currency: String,
}

#[extern_fn(effect = pure)]
fn price() -> Price {
    Price {
        amount: Decimal("9.99".parse().expect("a decimal literal")),
        currency: "USD".to_owned(),
    }
}

#[extern_fn(effect = pure)]
fn double_price(p: Price) -> Price {
    Price {
        amount: Decimal(p.amount.0 * rust_decimal::Decimal::TWO),
        currency: p.currency,
    }
}

fn price_registry() -> Registry<AcvusRuntime> {
    extern_registry! {
        ns: "t",
        fns: [price, double_price],
    }
}

#[tokio::test]
async fn a_decimal_is_exact_text_in_and_out() {
    let i = Interner::new();
    let v = run_ext_template(
        &i,
        r#"{{ Ok(d) = decimal("1.50".to_string()) }}{{ d.to_string() }}{{/}}"#,
        TypedContext::default(),
        vec![],
    )
    .await;
    assert_str(&v, "1.50");
    let v = run_ext_template(
        &i,
        r#"{{ Ok(a) = decimal("1.5".to_string()) }}{{ Ok(b) = decimal("1.50".to_string()) }}{{ same = a == b }}{{ same.to_string() }}{{/}}{{/}}"#,
        TypedContext::default(),
        vec![],
    )
    .await;
    assert_str(&v, "true");
    let v = run_ext_template(
        &i,
        r#"{{ Ok(d) = decimal("0.5".to_string()) }}{{ f = decimal_to_float(&d) }}{{ f.to_string() }}{{/}}"#,
        TypedContext::default(),
        vec![],
    )
    .await;
    assert_str(&v, "0.5");
}

#[tokio::test]
async fn an_extension_type_is_a_field_of_a_derived_object() {
    let i = Interner::new();
    let regs = || vec![price_registry()];
    let v = run_ext_template(
        &i,
        "{{ p = price() }}{{ p.amount.to_string() }} {{ p.currency }}",
        TypedContext::default(),
        regs(),
    )
    .await;
    assert_str(&v, "9.99 USD");
    let v = run_ext_template(
        &i,
        r#"{{ Ok(d) = decimal("0.05".to_string()) }}{{ p = double_price({ amount: d, currency: "KRW".to_string(), }) }}{{ p.amount.to_string() }}{{/}}"#,
        TypedContext::default(),
        regs(),
    )
    .await;
    assert_str(&v, "0.10");
}

#[extern_fn(effect = pure)]
fn or_zero(r: Result<i64, String>) -> i64 {
    r.unwrap_or(0)
}

#[extern_fn(effect = pure)]
fn describe(r: Result<i64, String>) -> String {
    match r {
        Ok(n) => format!("ok {n}"),
        Err(e) => format!("err {e}"),
    }
}

fn result_registry() -> Registry<AcvusRuntime> {
    extern_registry! {
        ns: "t",
        fns: [or_zero, describe],
    }
}

#[tokio::test]
async fn a_result_built_by_the_script_crosses_into_the_extern_fn() {
    let i = Interner::new();
    let regs = || vec![result_registry()];
    let v = run_ext_template(
        &i,
        "{{ n = or_zero(Ok(41)) }}{{ n.to_string() }}",
        TypedContext::default(),
        regs(),
    )
    .await;
    assert_str(&v, "41");
    let v = run_ext_template(
        &i,
        r#"{{ describe(Err("nope".to_string())) }}"#,
        TypedContext::default(),
        regs(),
    )
    .await;
    assert_str(&v, "err nope");
}

#[derive(acvus_extern::TyArg)]
enum ParseFail {
    Empty,
    NotANumber(String),
}

#[extern_fn(effect = pure)]
fn parse_int(text: String) -> Result<i64, ParseFail> {
    if text.is_empty() {
        return Err(ParseFail::Empty);
    }
    text.parse().map_err(|_| ParseFail::NotANumber(text))
}

#[extern_fn(effect = pure)]
fn must_be_even(n: i64) -> i64 {
    assert!(n % 2 == 0, "must_be_even: {n} is odd");
    n
}

fn fallible_registry() -> Registry<AcvusRuntime> {
    extern_registry! {
        ns: "t",
        fns: [parse_int, must_be_even],
    }
}

#[tokio::test]
async fn an_extern_fn_s_result_is_the_script_s_result() {
    let i = Interner::new();
    let regs = || vec![fallible_registry()];
    let src = |text: &str| {
        [
            &format!(r#"{{{{ r = parse_int("{text}".to_string()) }}}}"#),
            "{{ Ok(n) = r }}{{ n.to_string() }}",
            "{{ Err(ParseFail::NotANumber(t)) = }}not a number: {{ t }}",
            "{{ Err(ParseFail::Empty) = }}empty{{_}}?{{/}}",
        ]
        .concat()
    };
    let v = run_ext_template(&i, &src("42"), TypedContext::default(), regs()).await;
    assert_str(&v, "42");
    let v = run_ext_template(&i, &src("4x"), TypedContext::default(), regs()).await;
    assert_str(&v, "not a number: 4x");
    let v = run_ext_template(&i, &src(""), TypedContext::default(), regs()).await;
    assert_str(&v, "empty");
}

#[tokio::test]
async fn an_extern_that_can_panic_returns_its_value_when_it_does_not() {
    let i = Interner::new();
    let regs = || vec![fallible_registry()];
    let v = run_ext_template(
        &i,
        "{{ n = must_be_even(4) }}{{ n.to_string() }}",
        TypedContext::default(),
        regs(),
    )
    .await;
    assert_str(&v, "4");
}

#[tokio::test]
#[should_panic(expected = "must_be_even: 3 is odd")]
async fn an_extern_s_panic_carries_its_message() {
    let i = Interner::new();
    run_ext_template(
        &i,
        "{{ n = must_be_even(3) }}{{ n.to_string() }}",
        TypedContext::default(),
        vec![fallible_registry()],
    )
    .await;
}

#[derive(acvus_extern::TyArg)]
struct Line {
    pts: Vec<Pt>,
    origin: Pt,
}

#[extern_fn(effect = pure)]
fn line() -> Line {
    Line {
        pts: vec![
            Pt {
                x: 1,
                label: "a".to_owned(),
            },
            Pt {
                x: 2,
                label: "b".to_owned(),
            },
        ],
        origin: Pt {
            x: 9,
            label: "o".to_owned(),
        },
    }
}

fn line_registry() -> Registry<AcvusRuntime> {
    extern_registry! {
        ns: "t",
        fns: [line],
    }
}

#[tokio::test]
async fn a_derived_object_s_fields_cross_by_their_own_types() {
    let i = Interner::new();
    let v = run_ext_template(
        &i,
        "{{ l = line() }}{{ n = len(&l.pts) }}{{ n.to_string() }} {{ l.origin.label }}",
        TypedContext::default(),
        vec![line_registry()],
    )
    .await;
    assert_str(&v, "2 o");
}

#[tokio::test]
async fn a_refused_input_names_why_in_its_own_enum() {
    let i = Interner::new();
    let v = run_ext_template(
        &i,
        r#"{{ r = regex("(".to_string()) }}{{ Err(RegexError::Invalid(e)) = r }}{{ e.pattern }}{{_}}?{{/}}"#,
        TypedContext::default(),
        vec![regex_registry()],
    )
    .await;
    assert_str(&v, "(");
    let v = run_ext_template(
        &i,
        r#"{{ r = parse_date("yesterday".to_string(), "%Y".to_string()) }}{{ Err(DateError::Unparsable(e)) = r }}{{ e.input }} {{ e.format }}{{_}}?{{/}}"#,
        TypedContext::default(),
        vec![datetime_registry()],
    )
    .await;
    assert_str(&v, "yesterday %Y");
    let v = run_ext_template(
        &i,
        r#"{{ r = from_timestamp(9223372036854775807) }}{{ Err(DateError::OutOfRange(n)) = r }}{{ n.to_string() }}{{_}}?{{/}}"#,
        TypedContext::default(),
        vec![datetime_registry()],
    )
    .await;
    assert_str(&v, "9223372036854775807");
    let v = run_ext_template(
        &i,
        r#"{{ r = decimal("1.2.3".to_string()) }}{{ Err(DecimalError::Unparsable(t)) = r }}{{ t }}{{_}}?{{/}}"#,
        TypedContext::default(),
        vec![],
    )
    .await;
    assert_str(&v, "1.2.3");
}

#[tokio::test]
#[should_panic(expected = "the sky fell")]
async fn panic_stops_the_run_with_the_script_s_message() {
    let i = Interner::new();
    run_ext_script_mode(
        &i,
        r#"if @never { 1 } else { panic("the sky fell".to_string()) }"#,
        [(i.intern("never"), (Ty::Bool, Value::bool_(false)))]
            .into_iter()
            .collect(),
        vec![],
    )
    .await;
}

#[tokio::test]
async fn a_container_of_scalars_from_an_extern_fn_is_the_script_s_container() {
    let i = Interner::new();
    let v = run_ext(
        &i,
        "let s = \"ab\".to_string(); let b = to_bytes(s); b[0] as i64",
        TypedContext::default(),
        vec![],
    )
    .await;
    assert_eq!(v.as_int(), 97, "b\"ab\" is 97, 98");
    let v = run_ext(
        &i,
        "let s = \"ab\".to_string(); let b = to_bytes(s); b.len()",
        TypedContext::default(),
        vec![],
    )
    .await;
    assert_eq!(v.as_int(), 2);

    let v = run_ext(
        &i,
        "let s = \"héllo\".to_string(); to_utf8_lossy(to_bytes(s))",
        TypedContext::default(),
        vec![],
    )
    .await;
    assert_str(&v, "héllo");
    let v = run_ext_script_mode(
        &i,
        "if let Ok(re) = regex(\"[0-9]+\".to_string()) { let t = \"ab42cd\"; if let Some(m) = find(&re, &t) { m.text } else { \"none\".to_string() } } else { \"?\".to_string() }",
        TypedContext::default(),
        vec![regex_registry()],
    )
    .await;
    assert_str(&v, "42");
}

// =======================================================================
// The handler ABI, one declaration per arity (RFC-0044, stage 2c)
// =======================================================================

#[extern_fn(effect = pure)]
fn abi0() -> i64 {
    7
}

#[extern_fn(effect = pure)]
fn abi1(a: i64) -> i64 {
    a * 10
}

#[extern_fn(effect = pure)]
fn abi2(a: &i64, b: i64) -> i64 {
    *a * 10 + b
}

#[extern_fn(effect = pure)]
fn abi3(a: &i64, b: i64, c: i64) -> i64 {
    *a * 100 + b * 10 + c
}

/// Four arguments is past the by-value ABI: this one is lent the window.
#[extern_fn(effect = pure)]
fn abi4(a: &i64, b: i64, c: i64, d: i64) -> i64 {
    *a * 1000 + b * 100 + c * 10 + d
}

fn abi_registry() -> Registry<AcvusRuntime> {
    extern_registry! {
        ns: "t",
        fns: [abi0, abi1, abi2, abi3, abi4],
    }
}

#[tokio::test]
async fn an_extern_of_each_arity_gets_its_arguments() {
    let i = Interner::new();
    let regs = || vec![abi_registry()];

    let v = run_ext(&i, "abi0()", TypedContext::default(), regs()).await;
    assert_eq!(v.as_int(), 7);

    let v = run_ext(&i, "abi1(3)", TypedContext::default(), regs()).await;
    assert_eq!(v.as_int(), 30);

    let v = run_ext(
        &i,
        "let a = 4; abi2(&a, 5)",
        TypedContext::default(),
        regs(),
    )
    .await;
    assert_eq!(v.as_int(), 45);

    let v = run_ext(
        &i,
        "let a = 1; abi3(&a, 2, 3)",
        TypedContext::default(),
        regs(),
    )
    .await;
    assert_eq!(v.as_int(), 123);

    let v = run_ext(
        &i,
        "let a = 1; abi4(&a, 2, 3, 4)",
        TypedContext::default(),
        regs(),
    )
    .await;
    assert_eq!(v.as_int(), 1234);
}

#[tokio::test]
async fn a_borrowed_argument_survives_the_call_at_each_arity() {
    let i = Interner::new();
    let regs = || vec![abi_registry()];

    let v = run_ext(
        &i,
        "let a = 4; let r = abi2(&a, 5); r + a",
        TypedContext::default(),
        regs(),
    )
    .await;
    assert_eq!(v.as_int(), 49, "abi2 read 45 and `a` is still 4");

    let v = run_ext(
        &i,
        "let a = 1; let r = abi3(&a, 2, 3); r + a",
        TypedContext::default(),
        regs(),
    )
    .await;
    assert_eq!(v.as_int(), 124, "abi3 read 123 and `a` is still 1");

    let v = run_ext(
        &i,
        "let a = 1; let r = abi4(&a, 2, 3, 4); r + a",
        TypedContext::default(),
        regs(),
    )
    .await;
    assert_eq!(v.as_int(), 1235, "abi4 read 1234 and `a` is still 1");
}

// =======================================================================
//  An array crosses both ways (RFC-0048 §7)
// =======================================================================

/// A handler body cannot ask whether what crossed is an array: the macro
/// takes a runtime only as a generic parameter bounded by `Runtime`, so no
/// body names `acvus_interpreter::Value` and none reaches `is_array` or
/// `composite`. Those are asserted below, on the value the run returns.
#[extern_fn(effect = pure)]
fn reversed<N>(xs: Arr<i64, N>) -> Arr<i64, N>
where
    N: Var<kind::Length>,
{
    assert_eq!(
        xs.0,
        vec![1, 2, 3],
        "the handler reads the script's elements"
    );
    Arr::new(xs.0.into_iter().rev().collect())
}

/// The returned elements are allocated in Rust; the parameter is what
/// binds the length variable of the returned array's type.
#[extern_fn(effect = pure)]
fn labels<N>(xs: Arr<i64, N>) -> Arr<String, N>
where
    N: Var<kind::Length>,
{
    Arr::new(xs.0.into_iter().map(|n| format!("#{n}")).collect())
}

fn array_registry() -> Registry<AcvusRuntime> {
    extern_registry! {
        ns: "t",
        fns: [reversed, labels],
    }
}

#[tokio::test]
async fn an_array_the_script_built_crosses_into_a_handler_and_back() {
    let i = Interner::new();
    let regs = || vec![array_registry()];

    let v = run_ext(
        &i,
        "let xs = [1, 2, 3]; let ys = reversed(xs); ys[0]",
        TypedContext::default(),
        regs(),
    )
    .await;
    assert_eq!(v.as_int(), 3, "the script indexed the handler's array");

    let v = run_ext(&i, "reversed([1, 2, 3])", TypedContext::default(), regs()).await;
    assert!(v.is_array(), "the crossed value is an array, got {v:?}");
    assert_eq!(
        v.composite(),
        Some(Composite::Array),
        "its composite form is Some(Array)"
    );
    // SAFETY: `is_array` above read the vtable's composite, which only an
    // `Array` payload carries.
    let items: Vec<i64> = unsafe { v.as_array() }.iter().map(|e| e.as_int()).collect();
    assert_eq!(items, vec![3, 2, 1], "the elements read back reversed");
}

#[tokio::test]
async fn an_array_a_handler_builds_in_rust_is_indexed_by_the_script() {
    let i = Interner::new();
    let regs = || vec![array_registry()];

    let v = run_ext(
        &i,
        "let zs = labels([1, 2, 3]); zs[2].len()",
        TypedContext::default(),
        regs(),
    )
    .await;
    assert_eq!(v.as_int(), 2, "the script indexed to \"#3\"");

    let v = run_ext(&i, "labels([1, 2, 3])", TypedContext::default(), regs()).await;
    assert!(v.is_array(), "the Rust-built value is an array, got {v:?}");
    // SAFETY: as above.
    let items = unsafe { v.as_array() };
    // SAFETY: the handler returned a `String` at every element.
    let texts: Vec<&str> = items.iter().map(|e| unsafe { e.as_str() }).collect();
    assert_eq!(
        texts,
        vec!["#1", "#2", "#3"],
        "the elements the handler put there"
    );
}

#[tokio::test]
async fn an_argument_live_after_the_call_is_not_consumed_by_it() {
    let i = Interner::new();
    let v = run_ext(
        &i,
        "let b = 5; let a = 4; abi2(&a, b) + abi2(&a, b)",
        TypedContext::default(),
        vec![abi_registry()],
    )
    .await;
    assert_eq!(v.as_int(), 90, "each call read the same b = 5");
}

// -- A `&str` parameter (RFC-0062) --------------------------------------

#[extern_fn(effect = pure)]
fn byte_length(s: &str) -> u64 {
    s.len() as u64
}

#[extern_fn(effect = pure)]
fn first_byte(s: &str) -> u64 {
    u64::from(s.as_bytes()[0])
}

fn str_registry() -> Registry<AcvusRuntime> {
    extern_registry! {
        ns: "view",
        fns: [byte_length, first_byte],
    }
}

#[tokio::test]
async fn a_string_lent_to_a_str_parameter_is_the_pair_the_handler_reads() {
    let i = Interner::new();
    let v = run_ext(
        &i,
        "let s = \"héllo\".to_string(); byte_length(&s)",
        TypedContext::default(),
        vec![str_registry()],
    )
    .await;
    assert_eq!(v.as_int(), 6, "six bytes, five scalar values");
}

#[tokio::test]
async fn a_literal_at_a_str_parameter_reaches_the_handler_with_no_string_made() {
    let i = Interner::new();
    let v = run_ext(
        &i,
        "byte_length(\"héllo\")",
        TypedContext::default(),
        vec![str_registry()],
    )
    .await;
    assert_eq!(v.as_int(), 6, "six bytes, five scalar values");
}

#[tokio::test]
async fn a_str_parameter_reads_the_borrowed_string_s_own_bytes() {
    let i = Interner::new();
    let v = run_ext(
        &i,
        "let s = \"abc\".to_string(); first_byte(&s)",
        TypedContext::default(),
        vec![str_registry()],
    )
    .await;
    assert_eq!(
        v.as_int(),
        97,
        "the first byte of the string the script made"
    );
}
