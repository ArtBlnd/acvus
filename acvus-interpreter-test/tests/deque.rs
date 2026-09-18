//! The Deque instance of the context model: a sequence changed at its ends that
//! holds the runtime's own values, exercised over struct-shaped elements.

use acvus_interpreter_test::*;
use acvus_mir::ty::Ty;
use acvus_utils::Interner;
use rustc_hash::FxHashMap;

#[tokio::test]
async fn push_back_accumulates_over_structs() {
    let i = Interner::new();
    let src =
        "let d = deque(); push_back(&mut d, { x: 1, }); push_back(&mut d, { x: 2, }); len(&d)";
    assert_eq!(
        run_script(&i, src, FxHashMap::default(), Ty::U64)
            .await
            .as_int(),
        2
    );
}

#[tokio::test]
async fn get_reads_back_a_struct_field_in_order() {
    let i = Interner::new();
    let src = "let d = deque(); push_back(&mut d, { x: 10, }); push_back(&mut d, { x: 20, }); get(&d, 1).x";
    assert_eq!(
        run_script(&i, src, FxHashMap::default(), Ty::I64)
            .await
            .as_int(),
        20
    );
}

#[tokio::test]
async fn holds_enum_values() {
    let i = Interner::new();
    let src = "let d = deque(); push_back(&mut d, Some(7)); push_back(&mut d, None); len(&d)";
    assert_eq!(
        run_script(&i, src, FxHashMap::default(), Ty::U64)
            .await
            .as_int(),
        2
    );
}

#[tokio::test]
async fn a_deque_is_iterated_in_place_through_as_iter() {
    let i = Interner::new();
    let src = "let d = deque(); push_back(&mut d, 1); push_front(&mut d, 2); as_iter(&d) | fold(0, |acc, x| -> acc * 10 + *x)";
    assert_eq!(
        run_script(&i, src, FxHashMap::default(), Ty::I64)
            .await
            .as_int(),
        21
    );
}

#[tokio::test]
async fn a_deque_is_consumed_through_into_iter() {
    let i = Interner::new();
    let src = "let d = deque(); push_back(&mut d, 1); push_back(&mut d, 2); into_iter(d) | fold(0, |acc, x| -> acc * 10 + x)";
    assert_eq!(
        run_script(&i, src, FxHashMap::default(), Ty::I64)
            .await
            .as_int(),
        12
    );
}

#[tokio::test]
async fn a_deque_demotes_to_a_list_where_one_is_expected() {
    let i = Interner::new();
    let src = "let d = deque(); push_back(&mut d, 1); push_back(&mut d, 2); push_back(&mut d, 3); let xs = reverse(d); xs[0]";
    assert_eq!(
        run_script(&i, src, FxHashMap::default(), Ty::I64)
            .await
            .as_int(),
        3
    );
}
