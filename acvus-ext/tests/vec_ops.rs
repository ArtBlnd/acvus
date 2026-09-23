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
    run_at(interner, source, graph_optimize::Opt::Full).await
}

/// Compile and run `source` at one optimization level.
async fn run_at(interner: &Interner, source: &str, opt: graph_optimize::Opt) -> Value {
    let ast = ParsedAst::Script(acvus_ast::parse_script(interner, source).expect("parse"));
    let registries: Vec<Registry<AcvusRuntime>> = std_registries();
    let Externs {
        mut functions,
        types: type_registry,
        handlers,
        instances,
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

    // Drops are inserted by the optimize pipeline, so a leaked element
    // trips the machine rather than passing quietly (RFC-0048 rule 6).
    let result = graph_optimize::optimize(interner, &acvus_mir::laws::LawTable::of(graph.functions.iter()), lowered.modules.into_iter().collect(), opt);
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
        instances: &instances,
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

/// Every intent is asked of both levels: `None` runs only what a program
/// needs to reach the machine, `Full` runs every optimization, and an
/// answer that differs between them is an optimization defect.
async fn at_both<T, F>(source: &str, read: F) -> T
where
    T: PartialEq + std::fmt::Debug,
    F: Fn(Value) -> T,
{
    let none = read(run_at(&Interner::new(), source, graph_optimize::Opt::None).await);
    let full = read(run_at(&Interner::new(), source, graph_optimize::Opt::Full).await);
    assert_eq!(none, full, "Opt::None and Opt::Full disagree on: {source}");
    full
}

async fn int_of(source: &str) -> i64 {
    at_both(source, |v| v.as_int()).await
}

async fn float_of(source: &str) -> f64 {
    at_both(source, |v| v.as_float()).await
}

async fn bool_of(source: &str) -> bool {
    at_both(source, |v| v.as_bool()).await
}

async fn strings_of(source: &str) -> Vec<String> {
    let i = Interner::new();
    let v = run(&i, source).await;
    // SAFETY: the script's result is a `Vec<T>`, whose store is the run of
    // `Owned` the element type erases to (RFC-0048 rule 1).
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
async fn a_new_vec_is_empty_and_takes_a_push() {
    assert_eq!(int_of("let v = new(); v.push(1); v.len()").await, 1);
    assert_eq!(int_of("let v = vec::new(); v.push(1); v.len()").await, 1);
    assert!(bool_of("let v = new(); v.push(1); v.pop(); v.is_empty()").await);
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

// -- the view a script names itself -------------------------------------

#[tokio::test]
async fn as_slice_is_a_call_a_script_can_write() {
    assert_eq!(
        int_of("let v = vec([1, 2, 3]); let s = v.as_slice(); s.len() as i64").await,
        3
    );
}

#[tokio::test]
async fn an_array_names_its_own_view_too() {
    assert_eq!(
        int_of("let a = [1, 2, 3, 4]; a.as_slice().len() as i64").await,
        4
    );
}

#[tokio::test]
async fn an_exclusive_view_writes_through_to_the_container() {
    assert_eq!(
        int_of("let v = vec([1, 2, 3]); v.as_slice_mut().swap(0, 2); v[0]").await,
        3
    );
}

/// The coercion is what the target asks for, so where the target is a
/// `&Vec<T>` the argument stays a `Vec` and `vec::len` reads it.
#[tokio::test]
async fn a_vec_parameter_still_receives_the_vec() {
    assert_eq!(
        int_of("let v = vec([1, 2, 3, 4, 5]); v.len() as i64").await,
        5
    );
}

#[tokio::test]
async fn a_borrowed_container_still_walks_as_a_for_source() {
    assert_eq!(
        int_of("let v = vec([4, 5, 6]); let t = 0; for x in &v { t = t + *x; } t").await,
        15
    );
}

// -- ordered ------------------------------------------------------------

#[tokio::test]
async fn sort_puts_the_elements_in_order() {
    assert_eq!(
        int_of("let v = vec([3, 1, 2]); v.sort(); v[0] * 100 + v[1] * 10 + v[2]").await,
        123
    );
}

#[tokio::test]
async fn sort_orders_floats_by_total_cmp() {
    assert_eq!(
        float_of("let v = vec([2.5, 0.5, 1.5]); v.sort(); v[0]").await,
        0.5
    );
}

#[tokio::test]
async fn sort_orders_strings_by_their_bytes() {
    assert_eq!(
        strings_of(r#"let v = vec(["pear".to_string(), "apple".to_string()]); v.sort(); v"#).await,
        ["apple", "pear"]
    );
}

#[tokio::test]
async fn is_sorted_answers_for_the_order_sort_puts_them_in() {
    assert!(bool_of("let v = vec([3, 1, 2]); v.sort(); v.is_sorted()").await);
    assert!(!bool_of("let v = vec([3, 1, 2]); v.is_sorted()").await);
}

#[tokio::test]
async fn contains_finds_an_element_and_misses_one_that_is_not_there() {
    assert!(bool_of("let v = vec([1, 2, 3]); let x = 2; slice::contains(&v, &x)").await);
    assert!(!bool_of("let v = vec([1, 2, 3]); let x = 9; slice::contains(&v, &x)").await);
}

#[tokio::test]
async fn binary_search_answers_the_index_and_none_for_a_miss() {
    assert_eq!(
        int_of("let v = vec([1, 3, 5]); let x = 5; if let Some(at) = v.binary_search(&x) { at as i64 } else { 0 - 1 }").await,
        2
    );
    assert_eq!(
        int_of("let v = vec([1, 3, 5]); let x = 4; if let Some(at) = v.binary_search(&x) { at as i64 } else { 0 - 1 }").await,
        -1
    );
}

/// `vec::min` and `vec::max` are not declared: the name belongs to the
/// iterator, and a second declaration makes the bare call unresolvable.
#[tokio::test]
async fn the_least_element_is_the_iterators() {
    assert_eq!(
        int_of("let v = vec([7, 2, 9]); if let Some(x) = into_iter(v) | min { x } else { 0 - 1 }")
            .await,
        2
    );
}

#[tokio::test]
async fn to_vec_copies_the_elements_and_leaves_the_original() {
    assert_eq!(
        int_of("let v = vec([1, 2, 3]); let c = v.to_vec(); c[0] = 9; v[0] * 10 + c[0]").await,
        19
    );
}

// -- sorted by a closure ------------------------------------------------

#[tokio::test]
async fn sort_by_orders_by_what_the_closure_answers() {
    assert_eq!(
        int_of(
            "let v = vec([1, 2, 3]); \
             v.sort_by(|a, b| -> { if a > b { 0 - 1 } else if b > a { 1 } else { 0 } }); \
             v[0] * 100 + v[1] * 10 + v[2]"
        )
        .await,
        321
    );
}

/// Rust's `sort_by` is stable. Two keys that tie keep the order they came
/// in: sorting by the tens digit alone must leave 11 before 19.
#[tokio::test]
async fn sort_by_is_stable_where_the_closure_calls_a_tie() {
    assert_eq!(
        int_of(
            "let v = vec([19, 21, 11]); \
             v.sort_by(|a, b| -> { let x = *a / 10; let y = *b / 10; \
                                   if x < y { 0 - 1 } else if y < x { 1 } else { 0 } }); \
             v[0] * 10000 + v[1] * 100 + v[2]"
        )
        .await,
        19_11_21
    );
}

// -- the vec's own ------------------------------------------------------

#[tokio::test]
async fn split_off_leaves_the_head_and_gives_the_tail() {
    assert_eq!(
        int_of(
            "let v = vec([1, 2, 3, 4]); let tail = v.split_off(2); v.len() as i64 * 10 + tail[0]"
        )
        .await,
        23
    );
}

#[tokio::test]
#[should_panic(expected = "`at` split index (is 9) should be <= len (is 3)")]
async fn split_off_past_the_length_is_rust_s_message() {
    int_of("let v = vec([1, 2, 3]); let tail = v.split_off(9); tail[0]").await;
}

#[tokio::test]
async fn capacity_is_at_least_the_length_and_shrinks_to_it() {
    assert!(bool_of("let v = with_capacity(16); v.push(1); v.capacity() >= 16u64").await);
    assert!(
        bool_of("let v = with_capacity(16); v.push(1); v.shrink_to_fit(); v.capacity() >= 1u64")
            .await
    );
}

#[tokio::test]
async fn get_is_none_past_the_last_element() {
    assert_eq!(
        int_of("let v = vec([5, 6]); if let Some(x) = v.get(1u64) { *x } else { 0 - 1 }").await,
        6
    );
    assert_eq!(
        int_of("let v = vec([5, 6]); if let Some(x) = v.get(9u64) { *x } else { 0 - 1 }").await,
        -1
    );
}

// -- the view's own -----------------------------------------------------

/// A `for` traverses `&v`, `&mut v`, an array or a range, and `a[i]` takes
/// its own view of a container; a named `&[T]` is neither, so a view is
/// walked through `get`.
#[tokio::test]
async fn a_named_view_is_walked_through_get() {
    assert_eq!(
        int_of(
            "let v = vec([1, 2, 3]); let s = v.as_slice(); let t = 0; \
                for i in 0u64..s.len() { if let Some(x) = s.get(i) { t = t + *x; }; } t"
        )
        .await,
        6
    );
}

#[tokio::test]
async fn a_view_reports_its_own_length_and_emptiness() {
    assert!(bool_of("let v = with_capacity(2); v.as_slice().is_empty()").await);
    assert!(!bool_of("let v = vec([1]); v.as_slice().is_empty()").await);
}

#[tokio::test]
async fn a_view_swaps_two_of_its_containers_elements() {
    assert_eq!(
        int_of("let v = vec([1, 2, 3]); v.as_slice_mut().swap(0, 2); v[0] * 10 + v[2]").await,
        31
    );
}

#[tokio::test]
#[should_panic(expected = "index out of bounds: the len is 3 but the index is 7")]
async fn a_view_swap_past_the_last_element_is_rust_s_message() {
    int_of("let v = vec([1, 2, 3]); v.as_slice_mut().swap(0, 7); v[0]").await;
}

#[tokio::test]
async fn a_view_answers_its_first_last_and_indexed_element() {
    assert_eq!(
        int_of("let v = vec([4, 5, 6]); let s = v.as_slice(); if let Some(x) = s.first() { *x } else { 0 - 1 }")
            .await,
        4
    );
    assert_eq!(
        int_of("let v = vec([4, 5, 6]); let s = v.as_slice(); if let Some(x) = slice::last(s) { *x } else { 0 - 1 }")
            .await,
        6
    );
    assert_eq!(
        int_of("let v = vec([4, 5, 6]); let s = v.as_slice(); if let Some(x) = s.get(1u64) { *x } else { 0 - 1 }")
            .await,
        5
    );
}

// -- an array's read-only half ------------------------------------------

#[tokio::test]
async fn an_array_is_searched_for_an_element() {
    assert!(bool_of("let a = [1, 2, 3]; let x = 2; slice::contains(&a, &x)").await);
    assert!(!bool_of("let a = [1, 2, 3]; let x = 9; slice::contains(&a, &x)").await);
}

#[tokio::test]
async fn an_array_binary_searches_and_gets_by_index() {
    assert_eq!(
        int_of("let a = [1, 3, 5]; let x = 3; if let Some(at) = a.binary_search(&x) { at as i64 } else { 0 - 1 }")
            .await,
        1
    );
    assert_eq!(
        int_of("let a = [1, 3, 5]; if let Some(x) = a.get(2u64) { *x } else { 0 - 1 }").await,
        5
    );
}

// -- rotated, repeated, matched -----------------------------------------

#[tokio::test]
async fn rotate_left_and_right_move_the_run_by_the_offset() {
    assert_eq!(
        int_of("let v = vec([1, 2, 3, 4, 5]); v.as_slice_mut().rotate_left(2); v[0] * 10 + v[4]")
            .await,
        32
    );
    assert_eq!(
        int_of("let v = vec([1, 2, 3, 4, 5]); v.as_slice_mut().rotate_right(2); v[0] * 10 + v[4]")
            .await,
        43
    );
}

#[tokio::test]
async fn a_rotation_by_the_length_leaves_the_run_alone() {
    assert_eq!(
        int_of("let v = vec([1, 2, 3]); v.as_slice_mut().rotate_left(3); v[0] * 10 + v[2]").await,
        13
    );
}

#[tokio::test]
#[should_panic(expected = "rotation index (is 9) should be <= len (is 3)")]
async fn a_rotation_past_the_length_is_refused_at_the_bound_rust_states() {
    int_of("let v = vec([1, 2, 3]); v.as_slice_mut().rotate_left(9); v[0]").await;
}

#[tokio::test]
async fn starts_with_and_ends_with_match_a_prefix_and_a_suffix() {
    assert!(
        bool_of("let v = vec([1, 2, 3]); let p = vec([1, 2]); slice::starts_with(&v, &p)").await
    );
    assert!(
        !bool_of("let v = vec([1, 2, 3]); let p = vec([2, 3]); slice::starts_with(&v, &p)").await
    );
    assert!(bool_of("let v = vec([1, 2, 3]); let s = vec([2, 3]); slice::ends_with(&v, &s)").await);
    assert!(
        !bool_of("let v = vec([1, 2, 3]); let s = vec([1, 2]); slice::ends_with(&v, &s)").await
    );
}

/// A part longer than the whole matches neither end, and says so rather
/// than reading past the run.
#[tokio::test]
async fn a_part_longer_than_the_whole_matches_neither_end() {
    assert!(!bool_of("let v = vec([1]); let p = vec([1, 2]); slice::starts_with(&v, &p)").await);
    assert!(!bool_of("let v = vec([1]); let p = vec([1, 2]); slice::ends_with(&v, &p)").await);
}

#[tokio::test]
async fn an_empty_part_matches_both_ends() {
    assert!(
        bool_of("let v = vec([1, 2]); let p = with_capacity(1); slice::starts_with(&v, &p)").await
    );
    assert!(
        bool_of("let v = vec([1, 2]); let p = with_capacity(1); slice::ends_with(&v, &p)").await
    );
}

#[tokio::test]
async fn repeat_lays_the_run_down_that_many_times() {
    assert_eq!(
        int_of("let v = vec([1, 2]); let r = slice::repeat(&v, 3u64); r.len() as i64 * 10 + r[4]")
            .await,
        61
    );
    assert_eq!(
        int_of("let v = vec([1, 2]); let r = slice::repeat(&v, 0u64); r.len() as i64").await,
        0
    );
}

#[tokio::test]
async fn sort_by_key_orders_by_the_key_the_closure_answers() {
    assert_eq!(
        int_of(
            "let v = vec([30, 10, 20]); v.sort_by_key(|x| -> 0 - *x); \
             v[0] * 10000 + v[1] * 100 + v[2]"
        )
        .await,
        30_20_10
    );
}

/// The keys are read once per element, so a key that ties keeps the order
/// the elements came in.
#[tokio::test]
async fn sort_by_key_is_stable_where_the_key_ties() {
    assert_eq!(
        int_of(
            "let v = vec([19, 21, 11]); v.sort_by_key(|x| -> *x / 10); \
             v[0] * 10000 + v[1] * 100 + v[2]"
        )
        .await,
        19_11_21
    );
}

// -- min and max --------------------------------------------------------

/// Three declarations hold the bare name `max` — `num::max` over two
/// numbers, `iter::max` over a stage, `vec::max` over a container — and
/// one script reaches all three. The container's is written `slice::max(&v)`:
/// a method receiver does not settle between a lending and a consuming
/// candidate, the same bound `v.contains(&x)` has (`docs/std/vec.md`).
#[tokio::test]
async fn max_resolves_at_a_vec_at_two_numbers_and_over_an_iterator() {
    assert_eq!(
        int_of(
            "let v = vec([3, 9, 4]); \
             let a = if let Some(x) = slice::max(&v) { x } else { 0 - 1 }; \
             let w = vec([5, 2]); \
             let b = if let Some(x) = (w.into_iter() | max()) { x } else { 0 - 1 }; \
             a * 10000 + b * 100 + max(a, b)"
        )
        .await,
        9_05_09
    );
}

#[tokio::test]
async fn min_resolves_at_a_vec_at_two_numbers_and_over_an_iterator() {
    assert_eq!(
        int_of(
            "let v = vec([3, 9, 4]); \
             let a = if let Some(x) = slice::min(&v) { x } else { 0 - 1 }; \
             let w = vec([5, 2]); \
             let b = if let Some(x) = (w.into_iter() | min()) { x } else { 0 - 1 }; \
             a * 10000 + b * 100 + min(a, b)"
        )
        .await,
        3_02_02
    );
}

/// A `Vec` has no `iter::next`, so `iter::max` does not take it and the
/// method form is the slice's.
#[tokio::test]
async fn the_method_form_of_max_on_a_vec_is_the_slices() {
    assert_eq!(
        int_of("let v = vec([3, 9, 4]); if let Some(x) = v.max() { x } else { 0 - 1 }").await,
        9
    );
}

#[tokio::test]
async fn the_least_of_an_empty_vec_is_none() {
    assert_eq!(
        int_of(
            "let v = with_capacity(2); v.push(1); v.pop(); \
             if let Some(x) = slice::min(&v) { x } else { 0 - 1 }"
        )
        .await,
        -1
    );
}

/// `f64` orders by `total_cmp`, as `sort` does.
#[tokio::test]
async fn the_least_float_is_read_at_the_order_sort_uses() {
    assert_eq!(
        float_of(
            "let v = vec([2.5, 0.5, 1.5]); if let Some(x) = slice::min(&v) { x } else { 0.0 }"
        )
        .await,
        0.5
    );
}

#[tokio::test]
#[should_panic(expected = "compile failed")]
async fn max_of_a_vec_of_something_no_instance_orders_is_refused() {
    int_of("let v = vec([vec([1]), vec([2])]); if let Some(x) = slice::max(&v) { x[0] } else { 0 - 1 }")
        .await;
}

// -- an array ordered and copied ----------------------------------------

#[tokio::test]
async fn an_array_answers_whether_it_is_sorted() {
    assert!(bool_of("let a = [1, 2, 3]; a.is_sorted()").await);
    assert!(!bool_of("let a = [3, 1, 2]; a.is_sorted()").await);
}

/// `is_sorted` is declared for `Vec` and for `Array` alike, and one script
/// settles both by their receivers.
#[tokio::test]
async fn is_sorted_settles_at_an_array_and_at_a_vec_in_one_script() {
    assert!(
        bool_of(
            "let a = [1, 2, 3]; \
             let v = vec([3, 1, 2]); \
             a.is_sorted() && !v.is_sorted() && vec(a).is_sorted()"
        )
        .await
    );
}

#[tokio::test]
#[should_panic(expected = "compile failed")]
async fn is_sorted_over_an_array_of_something_no_instance_orders_is_refused() {
    bool_of("let a = [vec([1]), vec([2])]; a.is_sorted()").await;
}
