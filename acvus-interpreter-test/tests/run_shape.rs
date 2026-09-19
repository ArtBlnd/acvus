//! Obligation across artifacts: `match_dispatch.rs` says the dispatch reaches
//! the arm its tag names, and `run_allocations.rs` says the aggregate left the
//! heap. Neither is re-asserted here.

use acvus_interpreter_test::listing::{BlockListing, ops_of_anywhere, script_listing_with_externs};
use acvus_interpreter_test::{int_context, run_script};
use acvus_mir::ty::Ty;
use acvus_utils::Interner;

const THREE_ARMED: &str = "\
let e = if @n == 0 { E::A(1) } else { if @n == 1 { E::B(2) } else { E::C(3) } }; \
let i = 0; while i < 1 { match e { E::A(v) => { i = i + v; }, E::B(v) => { i = i + v; }, \
E::C(v) => { i = i + v; } }; i = i + 1; } i";

const THREE_ARMED_ESCAPING: &str = "\
let e = if @n == 0 { E::A(1) } else { if @n == 1 { E::B(2) } else { E::C(3) } }; \
let v = [e]; let m = len(&v); let one = m / m; let i = m - m; let acc = 0; \
while i < m { match &v[i] { E::A(w) => { acc = acc + *w; }, E::B(w) => { acc = acc + *w; }, \
E::C(w) => { acc = acc + *w; } }; i = i + one; } acc";

fn ops(source: &str) -> Vec<String> {
    let interner = Interner::new();
    let blocks: Vec<BlockListing> = script_listing_with_externs(
        &interner,
        source,
        int_context(&interner, "n", 0),
        acvus_ext::std_registries(),
        Ty::I64,
    );
    let ends: Vec<String> = blocks.iter().map(|block| block.end.clone()).collect();
    ops_of_anywhere(&blocks).into_iter().chain(ends).collect()
}

fn holds(ops: &[String], name: &str) -> bool {
    ops.iter().any(|op| op.starts_with(name))
}

#[test]
fn a_three_armed_join_takes_one_run_and_no_heap_variant() {
    let ops = ops(THREE_ARMED);
    assert!(
        holds(&ops, "LayRun"),
        "the constructions write the run: {ops:?}"
    );
    assert!(holds(&ops, "Project"), "`&e` is a projection: {ops:?}");
    assert!(
        holds(&ops, "SwitchRun"),
        "the dispatch reads the tag register: {ops:?}"
    );
    assert!(
        holds(&ops, "DropRun"),
        "the run's Large registers are released: {ops:?}"
    );
    assert!(
        !holds(&ops, "MakeVariant"),
        "no `Make` exists for a value that stays in its body: {ops:?}"
    );
    assert!(
        !holds(&ops, "Switch<"),
        "no boxed variant is dispatched on: {ops:?}"
    );
}

#[test]
fn a_web_whose_member_escapes_keeps_its_heap_form() {
    let ops = ops(THREE_ARMED_ESCAPING);
    assert!(
        holds(&ops, "MakeVariant"),
        "an escaping aggregate is realized: {ops:?}"
    );
    assert!(!holds(&ops, "LayRun"), "no run is written: {ops:?}");
    assert!(!holds(&ops, "SwitchRun"), "no run is read: {ops:?}");
}

#[tokio::test]
async fn the_run_form_reaches_the_arm_its_tag_names() {
    let interner = Interner::new();
    for (n, expected) in [(0, 2), (1, 3), (2, 4)] {
        let answer = run_script(
            &interner,
            THREE_ARMED,
            int_context(&interner, "n", n),
            Ty::I64,
        )
        .await
        .as_int();
        assert_eq!(answer, expected, "n={n}");
    }
}
