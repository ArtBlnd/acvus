//! A `Vec<T>` grown, shrunk, and made from a script. An index the
//! container does not hold is Rust's own panic, word for word.

use std::collections::HashMap;
use std::sync::Arc;

use acvus_ext::*;
use acvus_extern::{Externs, Registry};
use acvus_interpreter::AcvusRuntime;
use acvus_interpreter::*;
use acvus_mir::graph::*;
use acvus_mir::graph::{extract, infer, lower as graph_lower, optimize as graph_optimize};
use acvus_utils::{Astr, Freeze, Interner};
use rustc_hash::FxHashMap;

/// Compile and run `source` in script mode against the std registries.
async fn run(interner: &Interner, source: &str) -> Value {
    let ast = ParsedAst::Script(acvus_ast::parse_script(interner, source).expect("parse"));
    let registries: Vec<Registry<AcvusRuntime>> = std_registries();
    let Externs {
        mut functions,
        types: type_registry,
        handlers,
        ..
    } = Externs::combine(registries, interner).expect("registries combine");

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
        contexts: Freeze::new(vec![]),
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

    // Drops are inserted by the optimize pipeline, so a leaked element
    // trips the machine rather than passing quietly (RFC-0041).
    let result = graph_optimize::optimize(
        lowered.modules.into_iter().collect(),
        &FxHashMap::default(),
        &Default::default(),
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
    let context_names: FxHashMap<QualifiedRef, Astr> = FxHashMap::default();
    let prepare_ctx = PrepareCtx {
        interner,
        externs: &exec_fns,
        context_names: &context_names,
    };
    let prepared: Vec<(QualifiedRef, Executable)> = result
        .modules
        .iter()
        .map(|(qref, module)| {
            let prepared = prepare_module(module, &prepare_ctx);
            (*qref, Executable::Module(Arc::new(prepared)))
        })
        .collect();
    exec_fns.extend(prepared);

    let executor = Arc::new(SequentialExecutor);
    let shared = InterpreterContext::new(interner, exec_fns, executor);
    let page = InMemoryContext::new(HashMap::new());
    let mut interp = Interpreter::new(shared, entry_qref, page);
    interp.execute().await
}

async fn int_of(source: &str) -> i64 {
    let i = Interner::new();
    run(&i, source).await.as_int()
}

async fn float_of(source: &str) -> f64 {
    let i = Interner::new();
    run(&i, source).await.as_float()
}

async fn bool_of(source: &str) -> bool {
    let i = Interner::new();
    run(&i, source).await.as_bool()
}

async fn strings_of(source: &str) -> Vec<String> {
    let i = Interner::new();
    let v = run(&i, source).await;
    // SAFETY: the script's result is a `Vec<T>`, whose store is the run of
    // `Owned` the element type erases to (RFC-0048 §1).
    let items: Vec<acvus_extern::Owned<AcvusRuntime>> = unsafe { v.materialize() };
    items
        .iter()
        .map(|item| {
            assert!(item.is_string(), "expected a String, got {item:?}");
            // SAFETY: the witness is String.
            unsafe { item.as_str() }.to_owned()
        })
        .collect()
}

// -- grows --------------------------------------------------------------

#[tokio::test]
async fn a_vec_made_with_a_capacity_takes_pushes_and_gives_the_last_back() {
    assert_eq!(
        int_of("let v = with_capacity(4); v.push(1); v.push(2); if let Some(x) = v.pop() { x } else { -1 }")
            .await,
        2
    );
}

#[tokio::test]
async fn a_pop_of_an_empty_vec_is_none() {
    assert_eq!(
        int_of("let v = vec([7]); v.pop(); if let Some(x) = v.pop() { x } else { -1 }").await,
        -1
    );
}

#[tokio::test]
async fn push_then_pop_leaves_the_vec_empty() {
    assert!(bool_of("let v = with_capacity(4); v.push(1); v.pop(); v.is_empty()").await);
}

#[tokio::test]
async fn extend_appends_the_other_vec() {
    assert_eq!(
        int_of("let v = vec([1, 2]); v.extend(vec([3, 4])); v.len()").await,
        4
    );
    assert_eq!(
        int_of("let v = vec([1, 2]); v.extend(vec([3, 4])); v[3]").await,
        4
    );
}

// -- shrinks ------------------------------------------------------------

#[tokio::test]
async fn insert_puts_the_element_at_the_index_and_remove_gives_it_back() {
    assert_eq!(
        int_of("let v = vec([1, 2, 3]); v.insert(1, 9); v.remove(1)").await,
        9
    );
}

#[tokio::test]
#[should_panic(expected = "insertion index (is 5) should be <= len (is 3)")]
async fn insert_past_the_length_is_rust_s_message() {
    int_of("let v = vec([1, 2, 3]); v.insert(5, 9); 0").await;
}

#[tokio::test]
#[should_panic(expected = "removal index (is 5) should be < len (is 3)")]
async fn remove_past_the_last_element_is_rust_s_message() {
    int_of("let v = vec([1, 2, 3]); v.remove(5)").await;
}

#[tokio::test]
async fn clear_empties_the_vec() {
    assert_eq!(
        int_of("let v = vec([1, 2, 3]); v.clear(); v.len()").await,
        0
    );
}

#[tokio::test]
async fn truncate_keeps_the_first_n_and_a_longer_n_keeps_all() {
    assert_eq!(
        int_of("let v = vec([1, 2, 3]); v.truncate(1); v.len()").await,
        1
    );
    assert_eq!(
        int_of("let v = vec([1, 2, 3]); v.truncate(9); v.len()").await,
        3
    );
}

// -- swaps --------------------------------------------------------------

#[tokio::test]
async fn swap_exchanges_the_two_elements() {
    assert_eq!(
        int_of("let v = vec([1, 2, 3]); v.swap(0, 2); v[0] * 10 + v[2]").await,
        31
    );
}

#[tokio::test]
#[should_panic(expected = "index out of bounds: the len is 3 but the index is 7")]
async fn swap_past_the_last_element_is_rust_s_message() {
    int_of("let v = vec([1, 2, 3]); v.swap(0, 7); v[0]").await;
}

// -- is made ------------------------------------------------------------

#[tokio::test]
async fn filled_repeats_a_float() {
    assert_eq!(
        float_of("let v = filled(3, 0.5); v[0] + v[1] + v[2]").await,
        1.5
    );
}

#[tokio::test]
async fn filled_repeats_an_int_and_the_element_is_a_place() {
    assert_eq!(
        int_of("let v = filled(3, 0); v[1] = 7; v[0] + v[1] + v[2]").await,
        7
    );
}

#[tokio::test]
async fn filled_of_zero_is_empty() {
    assert!(bool_of("let v = filled(0, 1); v.is_empty()").await);
}

#[tokio::test]
async fn filled_clones_a_string_per_element() {
    assert_eq!(
        strings_of(r#"let s = "ab".to_string(); filled(3, s)"#).await,
        ["ab", "ab", "ab"]
    );
}
