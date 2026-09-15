//! The container signatures (RFC-0028) at the contract: one name reads a
//! list, an array, or a deque through a reference to it, an element comes
//! out as a reference into it, and a write through `get_mut` lands in the
//! container.

use acvus_interpreter_test::*;
use acvus_utils::Interner;
use rustc_hash::FxHashMap;

async fn int(src: &str) -> i64 {
    let i = Interner::new();
    run_script(&i, src, FxHashMap::default()).await.as_int()
}

#[tokio::test]
async fn len_reads_every_container_through_a_reference() {
    assert_eq!(int("xs = [1, 2, 3]; len(&xs)").await, 3);
    assert_eq!(
        int("xs = [1, 2, 3] | into_iter | collect; len(&xs)").await,
        3
    );
    assert_eq!(
        int("d = deque(); push_back(&mut d, 1); push_front(&mut d, 0); len(&d)").await,
        2
    );
}

#[tokio::test]
async fn get_yields_a_reference_into_the_container() {
    assert_eq!(int("xs = [10, 20, 30]; *get(&xs, 1)").await, 20);
    assert_eq!(
        int("xs = [10, 20, 30] | into_iter | collect; *get(&xs, 2)").await,
        30
    );
    assert_eq!(int("d = deque(); push_back(&mut d, { x: 10, }); push_front(&mut d, { x: 5, }); get(&d, 0).x").await, 5);
}

#[tokio::test]
async fn a_write_through_get_mut_lands_in_the_container() {
    assert_eq!(
        int("xs = [1, 2, 3]; *get_mut(&mut xs, 1) = 9; *get(&xs, 1)").await,
        9
    );
    assert_eq!(
        int("d = deque(); push_back(&mut d, 1); push_back(&mut d, 2); *get_mut(&mut d, 0) = 7; *get(&d, 0) + *get(&d, 1)").await,
        9
    );
}

#[tokio::test]
async fn first_and_last_are_none_on_an_empty_container_and_references_otherwise() {
    let i = Interner::new();
    let text = |src: &'static str| run(&i, src, FxHashMap::default());
    assert_eq!(
        text("{{ d = deque() }}{{ pushed = push_back(&mut d, 1) }}{{ popped = pop_back(&mut d) }}{{ Some(x) = first(&d) }}{{ *x | to_string }}{{_}}none{{/}}").await,
        "none"
    );
    assert_eq!(
        text("{{ xs = [4, 5, 6] }}{{ Some(x) = last(&xs) }}{{ *x | to_string }}{{_}}none{{/}}")
            .await,
        "6"
    );
    assert_eq!(
        text("{{ xs = [4, 5, 6] }}{{ Some(x) = first(&xs) }}{{ *x | to_string }}{{_}}none{{/}}")
            .await,
        "4"
    );
}
