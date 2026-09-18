//! A lent iterator and the storage it borrows, at the script contract: an
//! iterator built from `&v` reads `v`'s elements for as long as the
//! iterator is used, and `v` is still there afterwards (RFC-0018).

use acvus_interpreter_test::*;
use acvus_utils::Interner;
use rustc_hash::FxHashMap;

async fn run(source: &str) -> i64 {
    let i = Interner::new();
    run_script(&i, source, FxHashMap::default()).await.as_int()
}

#[tokio::test]
async fn while_let_over_a_local_vector_reads_every_element() {
    let sum = run("let v = range(0, 10) | collect; \
         let it = as_iter(&v); \
         let acc = 0; \
         while let Some(x) = next(&mut it) { acc = acc + *x; } \
         acc")
    .await;
    assert_eq!(sum, 45, "0 + 1 + ... + 9");
}

#[tokio::test]
async fn the_vector_is_still_readable_after_the_loop_that_borrowed_it() {
    let n = run("let v = range(0, 10) | collect; \
         let it = as_iter(&v); \
         while let Some(x) = next(&mut it) { } \
         len(&v)")
    .await;
    assert_eq!(n, 10, "`v` outlives the iterator built from it");
}

#[tokio::test]
async fn two_iterators_over_one_vector_each_read_it_whole() {
    let sum = run("let v = range(0, 10) | collect; \
         let a = as_iter(&v); \
         let b = as_iter(&v); \
         let acc = 0; \
         while let Some(x) = next(&mut a) { acc = acc + *x; } \
         while let Some(y) = next(&mut b) { acc = acc + *y; } \
         acc")
    .await;
    assert_eq!(sum, 90, "45 twice");
}

#[tokio::test]
async fn a_vector_lent_into_a_lambda_is_iterated_there() {
    let sum = run("let v = range(0, 10) | collect; \
         let total = |k| -> as_iter(k) | map(|x| -> *x) | sum; \
         total(&v)")
    .await;
    assert_eq!(sum, 45);
}

#[tokio::test]
async fn while_let_over_a_local_deque_reads_every_element() {
    let sum = run("let d = deque(); \
         push_back(&mut d, 1); \
         push_back(&mut d, 2); \
         push_back(&mut d, 3); \
         let it = as_iter(&d); \
         let acc = 0; \
         while let Some(x) = next(&mut it) { acc = acc + *x; } \
         acc")
    .await;
    assert_eq!(sum, 6, "1 + 2 + 3");
}

#[tokio::test]
async fn a_reference_returned_by_get_keeps_the_vector_alive() {
    let n = run("let v = range(0, 10) | collect; let r = get(&v, 3); *r + len(&v)").await;
    assert_eq!(n, 13);
}

#[tokio::test]
async fn a_reference_returned_by_first_keeps_the_vector_alive() {
    let n = run("let v = range(5, 10) | collect; \
         let acc = if let Some(r) = first(&v) { *r } else { 0 }; \
         acc + len(&v)")
    .await;
    assert_eq!(n, 10, "5 + 5");
}
