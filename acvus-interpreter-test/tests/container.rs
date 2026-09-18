//! The container signatures (RFC-0028) at the contract: one name reads a
//! list, an array, or a deque through a reference to it, an element comes
//! out as a reference into it, and a write through `get_mut` lands in the
//! container.

use acvus_interpreter_test::*;
use acvus_mir::ty::Ty;
use acvus_utils::Interner;
use rustc_hash::FxHashMap;

async fn int(src: &str, ret: Ty) -> i64 {
    let i = Interner::new();
    run_script(&i, src, FxHashMap::default(), ret)
        .await
        .as_int()
}

#[tokio::test]
async fn len_reads_every_container_through_a_reference() {
    assert_eq!(int("let xs = [1, 2, 3]; len(&xs)", Ty::U64).await, 3);
    assert_eq!(
        int(
            "let xs = [1, 2, 3] | into_iter | collect; len(&xs)",
            Ty::U64
        )
        .await,
        3
    );
    assert_eq!(
        int(
            "let d = deque(); push_back(&mut d, 1); push_front(&mut d, 0); len(&d)",
            Ty::U64
        )
        .await,
        2
    );
}

#[tokio::test]
async fn get_yields_a_reference_into_the_container() {
    assert_eq!(int("let xs = [10, 20, 30]; xs[1]", Ty::I64).await, 20);
    assert_eq!(
        int(
            "let xs = [10, 20, 30] | into_iter | collect; xs[2]",
            Ty::I64
        )
        .await,
        30
    );
    assert_eq!(int("let d = deque(); push_back(&mut d, { x: 10, }); push_front(&mut d, { x: 5, }); get(&d, 0).x", Ty::I64).await, 5);
}

#[tokio::test]
async fn a_write_through_an_element_lands_in_the_container() {
    assert_eq!(
        int("let xs = [1, 2, 3]; xs[1] = 9; xs[1]", Ty::I64).await,
        9
    );
    assert_eq!(
        int("let d = deque(); push_back(&mut d, 1); push_back(&mut d, 2); *get_mut(&mut d, 0) = 7; *get(&d, 0) + *get(&d, 1)", Ty::I64).await,
        9
    );
}

#[tokio::test]
async fn first_and_last_are_none_on_an_empty_container_and_references_otherwise() {
    let i = Interner::new();
    let text = |src: &'static str| run(&i, src, FxHashMap::default());
    assert_eq!(
        text("{{ d = deque() }}{{ pushed = push_back(&mut d, 1) }}{{ popped = pop_back(&mut d) }}{{ Some(x) = first(&d) }}{{ to_string(x) }}{{_}}none{{/}}").await,
        "none"
    );
    assert_eq!(
        text("{{ xs = [4, 5, 6] }}{{ Some(x) = last(&xs) }}{{ to_string(x) }}{{_}}none{{/}}").await,
        "6"
    );
    assert_eq!(
        text("{{ xs = [4, 5, 6] }}{{ Some(x) = first(&xs) }}{{ to_string(x) }}{{_}}none{{/}}")
            .await,
        "4"
    );
}

#[tokio::test]
async fn a_method_chain_runs_as_the_calls_it_stands_for() {
    assert_eq!(
        int(
            "let xs = [1, 2, 3]; let ys = xs.as_iter().map(|x| -> *x * 10).collect(); ys.len() + ys[2]",
            Ty::U64
        )
        .await,
        33
    );
    assert_eq!(
        int("let d = deque(); d.push_back({ x: 4, }); d.push_front({ x: 3, }); d.get(0).x * 10 + deque::len(&d)", Ty::U64).await,
        32
    );
}
