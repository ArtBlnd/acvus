//! E2E tests for ext functions: compile script with ext registry -> execute -> check result.

use std::collections::HashMap;
use std::sync::Arc;

use acvus_ext::*;
use acvus_extern::{ExternType, Externs, Registry, extern_fn, extern_registry};
use acvus_interpreter::AcvusRuntime;
use acvus_interpreter::*;
use acvus_mir::graph::*;
use acvus_mir::graph::{extract, infer, lower as graph_lower};
use acvus_mir::ty::{Ty, lift_to_poly};
use acvus_utils::{Astr, Freeze, Interner};
use rustc_hash::FxHashMap;

type TypedContext = FxHashMap<Astr, (Ty, Value)>;

/// Compile + execute a script with the std registries and `registries`.
async fn run_ext(
    interner: &Interner,
    source: &str,
    context: TypedContext,
    registries: Vec<Registry<AcvusRuntime>>,
) -> Value {
    let mut all_registries = std_registries::<AcvusRuntime>();
    all_registries.extend(registries);
    let Externs {
        mut functions,
        types: type_registry,
        handlers,
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

    let errs: Vec<String> = inf
        .errors()
        .into_iter()
        .flat_map(|(_, errs)| errs.iter())
        .chain(result.errors.iter().flat_map(|e| e.errors.iter()))
        .map(|e| format!("{}", e.display(interner)))
        .collect();
    if !errs.is_empty() {
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
        .map(|(k, (_, v))| (interner.resolve(k).to_string(), v))
        .collect();

    let executor = Arc::new(SequentialExecutor);
    let shared =
        InterpreterContext::new(interner, exec_fns, executor).with_context_names(context_names);
    let page = InMemoryContext::new(snapshot);
    let mut interp = Interpreter::new(shared, entry_qref, page);
    interp.execute().await.expect("execution failed")
}

fn assert_str(v: &Value, expected: &str) {
    assert!(v.is_string(), "expected a String, got {v:?}");
    // SAFETY: the witness is String.
    assert_eq!(unsafe { v.as_str() }, expected);
}

fn strings_of(v: Value) -> Vec<String> {
    // SAFETY: `collect` returns a `Vec<T>` and `T` is erased as `Value`.
    let list: Vec<Value> = unsafe { v.materialize() };
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
    let result = run_ext(
        &i,
        r#"re = regex("\\d+"); regex_match(re, "abc123")"#,
        TypedContext::default(),
        vec![regex_registry::<AcvusRuntime>()],
    )
    .await;
    assert!(result.as_bool());
}

#[tokio::test]
async fn regex_match_false() {
    let i = Interner::new();
    let result = run_ext(
        &i,
        r#"re = regex("\\d+"); regex_match(re, "abc")"#,
        TypedContext::default(),
        vec![regex_registry::<AcvusRuntime>()],
    )
    .await;
    assert!(!result.as_bool());
}

#[tokio::test]
async fn regex_find_all_collect() {
    let i = Interner::new();
    let result = run_ext(
        &i,
        r#"re = regex("\\d+"); regex_find_all(re, "a1b22c333") | collect"#,
        TypedContext::default(),
        vec![regex_registry::<AcvusRuntime>()],
    )
    .await;
    assert_eq!(strings_of(result), vec!["1", "22", "333"]);
}

#[tokio::test]
async fn regex_replace() {
    let i = Interner::new();
    let result = run_ext(
        &i,
        r#"re = regex("\\s+"); regex_replace("hello   world", re, " ")"#,
        TypedContext::default(),
        vec![regex_registry::<AcvusRuntime>()],
    )
    .await;
    assert_str(&result, "hello world");
}

#[tokio::test]
async fn regex_split_collect() {
    let i = Interner::new();
    let result = run_ext(
        &i,
        r#"re = regex("[,;]\\s*"); regex_split(re, "a, b;c") | collect"#,
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
    let result = run_ext(
        &i,
        r#"base64_decode(base64_encode("hello world"))"#,
        TypedContext::default(),
        vec![encoding_registry::<AcvusRuntime>()],
    )
    .await;
    assert_str(&result, "hello world");
}

#[tokio::test]
async fn url_roundtrip() {
    let i = Interner::new();
    let result = run_ext(
        &i,
        r#"url_decode(url_encode("hello world&foo=bar"))"#,
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
    let result = run_ext(
        &i,
        r#"dt = from_timestamp(1704067200); format_date(dt, "%Y-%m-%d")"#,
        TypedContext::default(),
        vec![datetime_registry::<AcvusRuntime>()],
    )
    .await;
    assert_str(&result, "2024-01-01");
}

#[tokio::test]
async fn datetime_timestamp_roundtrip() {
    let i = Interner::new();
    let result = run_ext(
        &i,
        r#"dt = from_timestamp(1704067200); timestamp(dt)"#,
        TypedContext::default(),
        vec![datetime_registry::<AcvusRuntime>()],
    )
    .await;
    assert_eq!(result.as_int(), 1704067200);
}

#[tokio::test]
async fn datetime_add_days() {
    let i = Interner::new();
    let result = run_ext(
        &i,
        r#"dt = from_timestamp(1704067200); dt2 = add_days(dt, 1); format_date(dt2, "%Y-%m-%d")"#,
        TypedContext::default(),
        vec![datetime_registry::<AcvusRuntime>()],
    )
    .await;
    assert_str(&result, "2024-01-02");
}

#[tokio::test]
async fn datetime_parse_and_format() {
    let i = Interner::new();
    let result = run_ext(
        &i,
        r#"dt = parse_date("2024-06-15 12:30:00", "%Y-%m-%d %H:%M:%S"); format_date(dt, "%m/%d/%Y")"#,
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
    let result = run_ext(
        &i,
        r#"m = regex_match(regex("\\d+"), "abc123"); base64_encode("hello") + " " + m.to_string()"#,
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
        "shift({ x: 41, label: \"a\", }).x",
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
        "ps = pts(); ps.get(1).x + ps.len()",
        TypedContext::default(),
        regs(),
    )
    .await;
    assert_eq!(v.as_int(), 22);
    let v = run_ext(
        &i,
        "ps = pts(); ps.as_iter().map(|p| -> p.x).fold(0, |a, x| -> a + x)",
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

#[tokio::test]
async fn a_container_of_scalars_from_an_extern_fn_is_the_script_s_container() {
    let i = Interner::new();
    let v = run_ext(
        &i,
        "s = \"ab\"; b = to_bytes(s); first = b.get(0); b.len() * 1000 + to_int(first)",
        TypedContext::default(),
        vec![],
    )
    .await;
    assert_eq!(v.as_int(), 2097);
    let v = run_ext(
        &i,
        "s = \"héllo\"; to_utf8_lossy(to_bytes(s))",
        TypedContext::default(),
        vec![],
    )
    .await;
    assert_str(&v, "héllo");
    let v = run_ext(
        &i,
        "re = regex(\"[0-9]+\"); unwrap_or(regex_find(re, \"ab42cd\"), \"none\")",
        TypedContext::default(),
        vec![regex_registry()],
    )
    .await;
    assert_str(&v, "42");
}
