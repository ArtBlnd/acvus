//! E2E tests for ext functions: compile script with ext registry -> execute -> check result.

use std::collections::HashMap;
use std::sync::Arc;

use acvus_ext::*;
use acvus_extern::{ExternType, Externs, Registry, Runtime, extern_fn, extern_registry};
use acvus_interpreter::AcvusRuntime;
use acvus_interpreter::*;
use acvus_mir::graph::*;
use acvus_mir::graph::{extract, infer, lower as graph_lower};
use acvus_mir::ty::{Ty, lift_to_poly};
use acvus_utils::{Astr, Freeze, Interner};
use rustc_hash::FxHashMap;

/// Compile + execute a script with the std registries and `registries`.
async fn run_ext(
    interner: &Interner,
    source: &str,
    context: FxHashMap<Astr, Value>,
    registries: Vec<Registry<AcvusRuntime>>,
) -> Value {
    let context_types: FxHashMap<Astr, Ty> = context
        .iter()
        .map(|(k, v)| (*k, infer_value_ty(interner, v)))
        .collect();

    let mut all_registries = std_registries::<AcvusRuntime>();
    all_registries.extend(registries);
    let Externs {
        functions: mut functions,
        types: type_registry,
        handlers,
    } = Externs::combine(all_registries, interner).expect("registries combine");

    let contexts: Vec<Context> = context_types
        .iter()
        .map(|(name, ty)| Context {
            qref: QualifiedRef::root(*name),
            ty: lift_to_poly(ty),
        })
        .collect();

    let entry_qref = QualifiedRef::root(interner.intern("test"));
    {
        let mut pb = acvus_mir::ty::PolyBuilder::new();
        functions.push(Function {
            qref: entry_qref,
            kind: FnKind::Local(ParsedAst::Script(
                acvus_ast::parse_script(interner, source).expect("parse"),
            )),
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
    };

    let ext = extract::extract(interner, &graph);
    let inf = infer::infer(
        interner,
        &graph,
        &ext,
        &FxHashMap::default(),
        Freeze::new(type_registry),
    );
    let result = graph_lower::lower(interner, &graph, &ext, &inf);

    if result.has_errors() {
        let errs: Vec<String> = result
            .errors
            .iter()
            .flat_map(|e| e.errors.iter())
            .map(|e| format!("{}", e.display(interner)))
            .collect();
        panic!("compile failed: {}", errs.join("; "));
    }

    let mut exec_fns: FxHashMap<QualifiedRef, Executable> = result
        .modules
        .into_iter()
        .map(|(qref, module)| (qref, Executable::Module(module)))
        .collect();
    exec_fns.extend(
        handlers
            .into_iter()
            .map(|(q, h)| (q, Executable::Extern(h))),
    );

    let context_names: FxHashMap<QualifiedRef, Astr> = graph
        .contexts
        .iter()
        .map(|ctx| (ctx.qref, ctx.qref.name))
        .collect();
    let snapshot: HashMap<String, Value> = context
        .into_iter()
        .map(|(k, v)| (interner.resolve(k).to_string(), v))
        .collect();

    let executor = Arc::new(SequentialExecutor);
    let shared =
        InterpreterContext::new(interner, exec_fns, executor).with_context_names(context_names);
    let page = InMemoryContext::new(snapshot);
    let mut interp = Interpreter::new(shared, entry_qref, page);
    interp.execute().await.expect("execution failed").value
}

/// Shallow type inference from Value.
fn infer_value_ty(interner: &Interner, v: &Value) -> Ty {
    match v {
        Value::Int(_) => Ty::Int,
        Value::Float(_) => Ty::Float,
        Value::Bool(_) => Ty::Bool,
        Value::String(_) => Ty::String,
        Value::Unit => Ty::Unit,
        Value::Byte(_) => Ty::Byte,
        Value::Array(items) => {
            let elem = items
                .first()
                .map(|v| infer_value_ty(interner, v))
                .unwrap_or(Ty::Int);
            Ty::Array(Box::new(elem), acvus_mir::ty::LenTerm::Known(items.len()))
        }
        Value::Object(fields) => Ty::Object(
            fields
                .iter()
                .map(|(k, v)| (*k, infer_value_ty(interner, v)))
                .collect(),
        ),
        Value::Extern(o) => Ty::UserDefined {
            id: o.type_name.qref(interner),
            type_args: vec![],
            effect_args: vec![],
            identity_args: vec![],
        },
        _ => Ty::Unit,
    }
}

// =======================================================================
//  Regex
// =======================================================================

#[tokio::test]
async fn regex_match_true() {
    let i = Interner::new();
    let c = FxHashMap::default();
    let result = run_ext(
        &i,
        r#"re = regex("\\d+"); regex_match(re, "abc123")"#,
        c,
        vec![regex_registry::<AcvusRuntime>()],
    )
    .await;
    assert_eq!(result, Value::Bool(true));
}

#[tokio::test]
async fn regex_match_false() {
    let i = Interner::new();
    let result = run_ext(
        &i,
        r#"re = regex("\\d+"); regex_match(re, "abc")"#,
        FxHashMap::default(),
        vec![regex_registry::<AcvusRuntime>()],
    )
    .await;
    assert_eq!(result, Value::Bool(false));
}

#[tokio::test]
async fn regex_find_all_collect() {
    let i = Interner::new();
    let result = run_ext(
        &i,
        r#"re = regex("\\d+"); regex_find_all(re, "a1b22c333") | collect"#,
        FxHashMap::default(),
        vec![regex_registry::<AcvusRuntime>()],
    )
    .await;
    let Value::Extern(list) = result else {
        panic!("expected List")
    };
    let items = list.downcast_ref::<Vec<Value>>().expect("List payload");
    assert_eq!(items.len(), 3);
    assert_eq!(items[0], Value::string("1"));
    assert_eq!(items[1], Value::string("22"));
    assert_eq!(items[2], Value::string("333"));
}

#[tokio::test]
async fn regex_replace() {
    let i = Interner::new();
    let result = run_ext(
        &i,
        r#"re = regex("\\s+"); regex_replace("hello   world", re, " ")"#,
        FxHashMap::default(),
        vec![regex_registry::<AcvusRuntime>()],
    )
    .await;
    assert_eq!(result, Value::string("hello world"));
}

#[tokio::test]
async fn regex_split_collect() {
    let i = Interner::new();
    let result = run_ext(
        &i,
        r#"re = regex("[,;]\\s*"); regex_split(re, "a, b;c") | collect"#,
        FxHashMap::default(),
        vec![regex_registry::<AcvusRuntime>()],
    )
    .await;
    let Value::Extern(list) = result else {
        panic!("expected List")
    };
    let items = list.downcast_ref::<Vec<Value>>().expect("List payload");
    assert_eq!(
        *items,
        vec![Value::string("a"), Value::string("b"), Value::string("c"),]
    );
}

// =======================================================================
//  Encoding
// =======================================================================

#[tokio::test]
async fn base64_roundtrip() {
    let i = Interner::new();
    let result = run_ext(
        &i,
        r#"base64_decode(base64_encode("hello world"))"#,
        FxHashMap::default(),
        vec![encoding_registry::<AcvusRuntime>()],
    )
    .await;
    assert_eq!(result, Value::string("hello world"));
}

#[tokio::test]
async fn url_roundtrip() {
    let i = Interner::new();
    let result = run_ext(
        &i,
        r#"url_decode(url_encode("hello world&foo=bar"))"#,
        FxHashMap::default(),
        vec![encoding_registry::<AcvusRuntime>()],
    )
    .await;
    assert_eq!(result, Value::string("hello world&foo=bar"));
}

// =======================================================================
//  DateTime
// =======================================================================

#[tokio::test]
async fn datetime_format_from_timestamp() {
    let i = Interner::new();
    // 2024-01-01 00:00:00 UTC = epoch 1704067200
    let result = run_ext(
        &i,
        r#"dt = from_timestamp(1704067200); format_date(dt, "%Y-%m-%d")"#,
        FxHashMap::default(),
        vec![datetime_registry::<AcvusRuntime>()],
    )
    .await;
    assert_eq!(result, Value::string("2024-01-01"));
}

#[tokio::test]
async fn datetime_timestamp_roundtrip() {
    let i = Interner::new();
    let result = run_ext(
        &i,
        r#"dt = from_timestamp(1704067200); timestamp(dt)"#,
        FxHashMap::default(),
        vec![datetime_registry::<AcvusRuntime>()],
    )
    .await;
    assert_eq!(result, Value::Int(1704067200));
}

#[tokio::test]
async fn datetime_add_days() {
    let i = Interner::new();
    let result = run_ext(
        &i,
        r#"dt = from_timestamp(1704067200); dt2 = add_days(dt, 1); format_date(dt2, "%Y-%m-%d")"#,
        FxHashMap::default(),
        vec![datetime_registry::<AcvusRuntime>()],
    )
    .await;
    assert_eq!(result, Value::string("2024-01-02"));
}

#[tokio::test]
async fn datetime_parse_and_format() {
    let i = Interner::new();
    let result = run_ext(
        &i,
        r#"dt = parse_date("2024-06-15 12:30:00", "%Y-%m-%d %H:%M:%S"); format_date(dt, "%m/%d/%Y")"#,
        FxHashMap::default(),
        vec![datetime_registry::<AcvusRuntime>()],
    ).await;
    assert_eq!(result, Value::string("06/15/2024"));
}

// =======================================================================
//  Multiple registries
// =======================================================================

#[tokio::test]
async fn mixed_regex_and_encoding() {
    let i = Interner::new();
    let result = run_ext(
        &i,
        r#"base64_encode("hello") + " " + to_string(regex_match(regex("\\d+"), "abc123"))"#,
        FxHashMap::default(),
        vec![
            regex_registry::<AcvusRuntime>(),
            encoding_registry::<AcvusRuntime>(),
        ],
    )
    .await;
    assert_eq!(result, Value::string("aGVsbG8= true"));
}

// =======================================================================
//  ExternCast - coercion via registered CastRule
// =======================================================================

#[derive(ExternType)]
struct MyNum(i64);

#[extern_fn(effect = pure)]
fn make_num<R>(_: &R) -> MyNum
where
    R: Runtime,
{
    MyNum(42)
}

#[extern_fn(effect = pure)]
#[extern_cast]
fn num_to_int<R>(_: &R, n: MyNum) -> i64
where
    R: Runtime,
{
    n.0
}

#[extern_fn(effect = pure)]
fn double<R>(_: &R, n: i64) -> i64
where
    R: Runtime,
{
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
        FxHashMap::default(),
        vec![extern_cast_registry()],
    )
    .await;
    assert_eq!(result, Value::Int(84));
}
