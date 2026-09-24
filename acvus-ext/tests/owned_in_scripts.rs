//! RFC-0048 rule 7 where only the language reaches the holder: a value a
//! script builds is released once by the holder that took it, and the one
//! the script hands back is released by its host.
//!
//! The compile-and-run harness is the one `e2e.rs` uses; `Tracked` is an
//! extension type whose `Drop` counts, so the interpreter's `release` —
//! which reaches an erased extension type through its header's drop slot
//! — is what the count observes.

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use acvus_ext::*;
use acvus_extern::{ExternType, Externs, Owned, Registry, Release, extern_fn, extern_registry};
use acvus_interpreter::AcvusRuntime;
use acvus_interpreter::*;
use acvus_mir::graph::*;
use acvus_mir::graph::{extract, infer, lower as graph_lower, optimize as graph_optimize};
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
    let ast = ParsedAst::Script(acvus_ast::parse_script(interner, source).expect("parse"));
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
                flows: acvus_mir::ty::Flows::Every.into(),
            },
        });
    }

    let graph = CompilationGraph {
        functions: Freeze::new(functions),
        contexts: Freeze::new(contexts),
        types: Freeze::new(type_registry),
        bindings: acvus_mir::graph::Bindings::default(),
        entry: Some(entry_qref),
    };

    let ext = extract::extract(interner, &graph);
    let inf = infer::infer(interner, &graph, &ext);
    let lowered = graph_lower::lower(interner, &graph, &ext.view(), &inf);

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
        interner,
        &acvus_mir::laws::LawTable::of(graph.functions.iter()),
        lowered.modules.into_iter().collect(),
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
        instances: &acvus_extern::NoInstances,
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
        // SAFETY: the word was made for this holder and moved in; no other holder owns it.
        .map(|(k, (_, v))| (interner.resolve(k).to_string(), unsafe { Owned::from_value(v) }))
        .collect();

    let executor = Arc::new(SequentialExecutor);
    let shared =
        InterpreterContext::new(interner, exec_fns, executor).with_context_names(context_names);
    let page = InMemoryContext::new(snapshot);
    let mut interp = Interpreter::new(shared, entry_qref, page);
    interp.execute().await
}

// -- An extension type whose `Drop` counts -------------------------------

static DROPS: AtomicUsize = AtomicUsize::new(0);

/// The counter is process-wide because an extension type is declared once
/// for the whole registry, so each test takes the lock and measures its
/// own delta.
static ONE_TEST_AT_A_TIME: Mutex<()> = Mutex::new(());

struct Measured {
    _lock: std::sync::MutexGuard<'static, ()>,
    start: usize,
}

impl Measured {
    fn start() -> Self {
        let lock = ONE_TEST_AT_A_TIME
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        Self {
            _lock: lock,
            start: DROPS.load(Ordering::SeqCst),
        }
    }

    fn count(&self) -> usize {
        DROPS.load(Ordering::SeqCst) - self.start
    }
}

struct Counted(i64);

impl Drop for Counted {
    fn drop(&mut self) {
        DROPS.fetch_add(1, Ordering::SeqCst);
    }
}

#[derive(ExternType)]
#[repr(transparent)]
struct Tracked(Box<Counted>);

#[extern_fn(effect = pure)]
fn tracked(n: i64) -> Tracked {
    Tracked(Box::new(Counted(n)))
}

#[extern_fn(effect = pure)]
fn rank(t: &Tracked) -> i64 {
    t.0.0
}

fn tracked_registry() -> Registry<AcvusRuntime> {
    extern_registry! {
        ns: "t",
        types: [Tracked],
        fns: [tracked, rank],
    }
}

// -- `Machine.exit`: the value the script hands back ----------------------

#[tokio::test]
async fn a_script_result_is_released_once_by_its_host() {
    let measured = Measured::start();
    let i = Interner::new();
    let result = run_ext(
        &i,
        "tracked(7)",
        TypedContext::default(),
        vec![tracked_registry()],
    )
    .await;
    assert_eq!(measured.count(), 0, "the result is out of the machine");
    result.release();
    assert_eq!(measured.count(), 1, "its host released it once");
}

// -- `Vec<Owned<R>>` as the language's container -------------------------

#[tokio::test]
async fn a_container_the_script_drops_releases_its_elements_once() {
    let measured = Measured::start();
    let i = Interner::new();
    let result = run_ext(
        &i,
        "let xs = [tracked(1), tracked(2), tracked(3)]; len(&xs)",
        TypedContext::default(),
        vec![tracked_registry()],
    )
    .await;
    assert_eq!(result.as_int(), 3, "the script built three elements");
    assert_eq!(measured.count(), 3, "each element was released once");
}

// -- A path assignment: the old value and the new one --------------------

#[tokio::test]
async fn a_field_assignment_releases_the_value_it_replaced_and_no_other() {
    let measured = Measured::start();
    let i = Interner::new();
    let result = run_ext(
        &i,
        "let o = { t: tracked(1), }; o.t = tracked(2); rank(&o.t)",
        TypedContext::default(),
        vec![tracked_registry()],
    )
    .await;
    assert_eq!(result.as_int(), 2, "the field holds the value assigned");
    assert_eq!(
        measured.count(),
        2,
        "expected 2: tracked(1) at the assignment, tracked(2) at frame exit"
    );
}

#[tokio::test]
async fn an_element_assignment_releases_the_value_it_replaced_and_no_other() {
    let measured = Measured::start();
    let i = Interner::new();
    let result = run_ext(
        &i,
        "let v = [tracked(1)]; v[0] = tracked(2); rank(&v[0])",
        TypedContext::default(),
        vec![tracked_registry()],
    )
    .await;
    assert_eq!(result.as_int(), 2, "the element holds the value assigned");
    assert_eq!(
        measured.count(),
        2,
        "expected 2: tracked(1) at the assignment, tracked(2) at frame exit"
    );
}
