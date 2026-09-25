//! `while let Some(x) = next(&mut it)` as one region whose source is the
//! call (RFC-0069): what the body reads, and what a `return` out of it does.
//!
//! The head no longer lands its verdict as a `some` for a `TestOption` to
//! take apart again, so the case that tells landing from not landing is a
//! payload that is itself an `Option`: over `Vec<Option<i64>>` a `Some(None)`
//! element and the `None` that ends the traversal are the same word, and only
//! the verdict beside it separates them.

use acvus_interpreter::listing::regions_named;
use acvus_interpreter_test::listing::script_listing;
use acvus_interpreter_test::*;
use acvus_mir::ty::Ty;
use acvus_utils::Interner;
use rustc_hash::FxHashMap;

async fn run(source: &str) -> i64 {
    let i = Interner::new();
    run_script(&i, source, FxHashMap::default(), Ty::I64)
        .await
        .as_int()
}

fn regions(source: &str) -> Vec<String> {
    let i = Interner::new();
    script_listing(&i, source, Context::default(), Ty::I64)
        .iter()
        .flat_map(|block| block.regions.iter())
        .map(|region| region.name.clone())
        .collect()
}

fn driven_by_a_call(source: &str) -> Vec<String> {
    regions(source)
        .into_iter()
        .filter(|name| name.starts_with("For<Call<"))
        .collect()
}

const OPTION_ELEMENTS: &str = "let v = vec([Some(1), None, Some(3)]); \
                               let it = as_iter(&v); \
                               let seen = 0; \
                               let acc = 0; \
                               while let Some(x) = next(&mut it) { \
                                   seen = seen + 1; \
                                   acc = acc + if let Some(k) = x { *k } else { 0 }; \
                               } \
                               seen * 100 + acc";

const BREAKS: &str = "let v = range(0, 10) | collect; \
                      let it = as_iter(&v); \
                      let acc = 0; \
                      while let Some(x) = next(&mut it) { \
                          if *x > 4 { break; }; \
                          acc = acc + *x; \
                      } \
                      acc";

const RETURNS: &str = "let v = range(0, 10) | collect; \
                       let it = as_iter(&v); \
                       let acc = 0; \
                       while let Some(x) = next(&mut it) { \
                           if *x > 4 { return acc; }; \
                           acc = acc + *x; \
                       } \
                       acc";

#[tokio::test]
async fn a_none_payload_is_an_element_and_not_the_end_of_the_traversal() {
    assert_eq!(
        run(OPTION_ELEMENTS).await,
        304,
        "three elements, of which `Some(1)` and `Some(3)` carry a number"
    );
}

#[tokio::test]
async fn an_option_payload_loop_is_driven_by_its_call() {
    assert_eq!(
        driven_by_a_call(OPTION_ELEMENTS).len(),
        1,
        "the regions of the option-payload loop are {:?}",
        regions(OPTION_ELEMENTS)
    );
}

#[tokio::test]
async fn a_break_leaves_the_loop_at_the_element_that_tripped_it() {
    assert_eq!(run(BREAKS).await, 10, "0 + 1 + 2 + 3 + 4");
}

#[tokio::test]
async fn a_return_inside_the_body_leaves_the_function() {
    assert_eq!(run(RETURNS).await, 10, "0 + 1 + 2 + 3 + 4");
}

#[tokio::test]
async fn a_returning_body_makes_the_call_loop_an_escaping_region() {
    let driven = driven_by_a_call(RETURNS);
    let [name] = driven.as_slice() else {
        panic!(
            "the regions of the returning loop are {:?}",
            regions(RETURNS)
        )
    };
    assert!(
        name.ends_with(", Escapes>"),
        "the returning loop prepared to {name}"
    );
}

/// The `break`'s exit edge carries the drops of the vector and the iterator,
/// which `acvus_mir::optimize::drop_insertion` gives a block of its own.
/// `prepare::recognize_loop` admits no block between the test and the body,
/// so this loop is not a region at all and never reaches the recognition
/// above — the same joints path `for_loop.rs` records for a `for` whose exit
/// edge carries a drop.
#[tokio::test]
async fn a_break_out_of_a_call_loop_stays_on_the_joints_path() {
    assert!(
        regions_named(
            &script_listing(&Interner::new(), BREAKS, Context::default(), Ty::I64),
            "Loop<Slot, Escapes>",
        )
        .is_empty(),
        "the regions of the breaking loop are {:?}",
        regions(BREAKS)
    );
    assert!(
        driven_by_a_call(BREAKS).is_empty(),
        "the regions of the breaking loop are {:?}",
        regions(BREAKS)
    );
}

/// A pull loop over an owning iterator ends its header in `While` with its
/// body cut into stages (RFC-0089 rule 1); it runs in place, the stages one
/// body, and is still the region its call drives.
const PULLED_AND_CUT: &str = "let it = into_iter(vec([5, 3, 8])); \
                              let out = vec([]); \
                              while let Some(x) = next(&mut it) { push(&mut out, x * 2); } \
                              len(&out) as i64";

#[tokio::test]
async fn a_pull_loop_cut_into_stages_runs_in_place() {
    assert_eq!(run(PULLED_AND_CUT).await, 3);
    assert_eq!(
        driven_by_a_call(PULLED_AND_CUT).len(),
        1,
        "{:?}",
        regions(PULLED_AND_CUT)
    );
}
