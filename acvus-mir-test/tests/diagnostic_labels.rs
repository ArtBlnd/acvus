//! A refusal names every place its story needs: the primary span where the
//! rule was broken, and a label at each other place. The label's span is
//! asserted as the source it covers, so a moved span shows up as a different
//! word rather than a different number.

use acvus_mir::ty::{LenTerm, Ty};
use acvus_mir_test::{Marked, Refusal, refuse_script_mode_optimized};
use acvus_utils::{Astr, Interner};
use rustc_hash::FxHashMap;

fn query(i: &Interner) -> FxHashMap<Astr, Ty> {
    FxHashMap::from_iter([(
        i.intern("query"),
        Ty::Array(Box::new(Ty::Float), LenTerm::Known(2)),
    )])
}

/// The refusals of `source`, with `context` declared.
fn refusals(source: &str, context: &dyn Fn(&Interner) -> FxHashMap<Astr, Ty>) -> Vec<Refusal> {
    let i = Interner::new();
    match refuse_script_mode_optimized(&i, source, &context(&i)) {
        Ok(ir) => panic!("expected a refusal, compiled:\n{ir}"),
        Err(refusals) => refusals,
    }
}

fn nothing(_: &Interner) -> FxHashMap<Astr, Ty> {
    FxHashMap::default()
}

/// The one refusal of `source`, as its message and its labels.
fn only(source: &str, context: &dyn Fn(&Interner) -> FxHashMap<Astr, Ty>) -> (String, Vec<Marked>) {
    let refusals = refusals(source, context);
    assert_eq!(refusals.len(), 1, "{source}: {:#?}", words(&refusals));
    (refusals[0].message.clone(), refusals[0].marked(source))
}

fn words(refusals: &[Refusal]) -> Vec<String> {
    refusals.iter().map(Refusal::to_string).collect()
}

fn at(source: &str, text: &str) -> Marked {
    Marked {
        source: Some(source.to_string()),
        text: text.to_string(),
    }
}

fn note(text: &str) -> Marked {
    Marked {
        source: None,
        text: text.to_string(),
    }
}

#[test]
fn a_use_after_move_labels_the_move() {
    let (message, labels) = only("let a = [1, 2]; let b = a; a", &nothing);
    assert_eq!(message, "`a` is used here after it was moved");
    assert_eq!(labels, [at("a", "moved here")]);
}

#[test]
fn a_use_of_a_moved_out_context_labels_the_move() {
    let (message, labels) = only("let q = @query; let r = @query;", &query);
    assert_eq!(message, "`@query` is used here after it was moved");
    assert_eq!(labels, [at("@query", "moved here")]);
}

#[test]
fn a_borrow_conflict_labels_the_borrow_and_the_use_that_keeps_it_live() {
    let (message, labels) = only(
        "let a = [1, 2]; let r = &mut a; let n = len(&a); r[0]",
        &nothing,
    );
    assert_eq!(message, "`a` is read here while a reference to it is live");
    assert_eq!(
        labels,
        [
            at("&mut a", "borrowed here"),
            at("r", "the reference is used here"),
        ]
    );
}

#[test]
fn a_write_while_a_reference_is_live_is_stated_as_a_write() {
    let refusals = refusals("let a = [1, 2]; let r = &mut a; a = [3, 4]; r[0]", &nothing);
    assert!(
        refusals
            .iter()
            .all(|r| r.message == "`a` is written here while a reference to it is live"),
        "{:#?}",
        words(&refusals)
    );
}

#[test]
fn a_move_out_of_an_index_carries_the_way_out_as_a_note() {
    let (message, labels) = only(
        "let v = [\"a\".to_string(), \"b\".to_string()]; let s = v[0]; s",
        &nothing,
    );
    assert_eq!(message, "cannot move out of index of `Array<String, 2>`");
    assert_eq!(
        labels,
        [note(
            "the element is String, which moves; take a reference with `&a[i]`"
        )]
    );
}

#[test]
fn an_assignment_to_a_capture_labels_the_lambda() {
    let (message, labels) = only(
        "let x = [1, 2]; let f = |k| -> { x = [3, 4]; 0 }; f(1)",
        &nothing,
    );
    assert_eq!(
        message,
        "cannot assign to `x`: it is captured by the lambda, not bound in it"
    );
    assert_eq!(labels, [at("{ x = [3, 4]; 0 }", "captured here")]);
}

#[test]
fn a_move_out_of_a_capture_labels_the_lambda_that_captured_it() {
    let refusals = refusals("let x = [1, 2]; let f = |k| -> |j| -> x; f(1)", &nothing);
    let moved = refusals
        .iter()
        .find(|r| {
            r.message
                .starts_with("cannot move `x` out of a closure's capture")
        })
        .unwrap_or_else(|| panic!("{:#?}", words(&refusals)));
    assert_eq!(
        moved.marked("let x = [1, 2]; let f = |k| -> |j| -> x; f(1)"),
        [at("|j| -> x", "captured here")]
    );
}

/// `typeck` learns a capture is a reference from the capture's resolved type
/// and not from a use of the name, so it holds the lambda's span alone and
/// there is no second place to label.
#[test]
fn a_captured_reference_names_one_place() {
    let refusals = refusals(
        "let a = [1, 2]; let r = &a; let f = |k| -> len(r); f(1)",
        &nothing,
    );
    let captured = refusals
        .iter()
        .find(|r| r.message == "a lambda cannot capture a reference")
        .unwrap_or_else(|| panic!("{:#?}", words(&refusals)));
    assert_eq!(captured.labels.len(), 0);
}

/// `Solver::lend` answers that the referent is shared without saying where
/// the shared reference was taken, so the refusal names one place.
#[test]
fn a_mutable_borrow_of_a_shared_reference_names_one_place() {
    let (message, labels) = only("let a = [1, 2]; let r = &a; let m = &mut r; 0", &nothing);
    assert_eq!(message, "a shared reference cannot be borrowed mutably");
    assert_eq!(labels, []);
}

/// The `Commit` that ends a run is emitted by lowering and carries the move's
/// own span, so the refusal has one place and points it at the move.
#[test]
fn a_context_moved_out_names_the_move_alone() {
    let (message, labels) = only("let q = @query; let dot = |k| -> k[0]; dot(&q)", &query);
    assert_eq!(
        message,
        "context @query is moved out here and not assigned again before the run ends"
    );
    assert_eq!(labels, []);
}

/// Two values of one type from two sources: the list, the `if` and the call
/// each name their own place where the two meet, and both origins.
mod one_type_two_sources {
    use super::*;

    const WORDS: &str = "`a` and `b` are values of one type from two different sources, \
                         and one place cannot hold both";

    fn begins_here(name: &str) -> Marked {
        at("into_iter", &format!("`{name}`'s source begins here"))
    }

    #[test]
    fn a_list_of_two_sources_labels_both() {
        let (message, labels) = only(
            "let a = [1, 2] | into_iter; let b = [3, 4] | into_iter; [a, b]; 0",
            &nothing,
        );
        assert_eq!(message, WORDS);
        assert_eq!(labels, [begins_here("a"), begins_here("b")]);
    }

    #[test]
    fn two_branches_of_two_sources_label_both() {
        let (message, labels) = only(
            "let a = [1, 2] | into_iter; let b = [3, 4] | into_iter; \
             let c = if true { a } else { b }; 0",
            &nothing,
        );
        assert_eq!(message, WORDS);
        assert_eq!(labels, [begins_here("a"), begins_here("b")]);
    }

    #[test]
    fn two_calls_of_one_lambda_label_both_sources() {
        let (message, labels) = only(
            "let a = [1, 2] | into_iter; let b = [3, 4] | into_iter; \
             let f = |x| -> x | collect; f(a); f(b); 0",
            &nothing,
        );
        assert_eq!(message, WORDS);
        assert_eq!(labels, [begins_here("a"), begins_here("b")]);
    }

    /// Rule 3: where the two types differ in something a reader can see, the
    /// refusal is about the types and names no source.
    #[test]
    fn two_sources_of_two_types_stay_a_type_mismatch() {
        let (message, labels) = only(
            "let a = [1, 2] | into_iter; let b = [\"x\".to_string(), \"y\".to_string()] | into_iter; [a, b]; 0",
            &nothing,
        );
        assert_eq!(
            message,
            "heterogeneous list: expected Iterator<i64, Pure>, got Iterator<String, Pure>"
        );
        assert_eq!(labels, []);
    }
}

/// A context no declaration names: the note lists what is declared, and
/// says nothing where nothing is.
mod undeclared_context {
    use super::*;

    #[test]
    fn the_declared_names_are_the_note() {
        let (message, labels) = only("@nope + 1", &query);
        assert_eq!(message, "`@nope` is not a declared context");
        assert_eq!(labels, [note("declared: `@query`")]);
    }

    #[test]
    fn nothing_declared_leaves_the_refusal_with_one_place() {
        let (message, labels) = only("@nope + 1", &nothing);
        assert_eq!(message, "`@nope` is not a declared context");
        assert_eq!(labels, []);
    }
}

/// A body's result is not a reference, nor data holding one: the `&str`
/// shape keeps the spelling that owns the text.
mod reference_returned_from_body {
    use super::*;

    const WORDS: &str = "a body does not return a reference";

    #[test]
    fn a_reference_to_a_word_is_refused() {
        let (message, labels) = only("let a = 1; &a", &nothing);
        assert_eq!(message, WORDS);
        assert_eq!(labels, []);
    }

    #[test]
    fn a_view_carries_the_way_to_own_the_text() {
        let (message, _) = only("let s = \"ab\"; &s", &nothing);
        assert_eq!(
            message,
            format!("{WORDS}; write `.to_string()` for the owned text")
        );
    }

    #[test]
    fn a_reference_inside_a_variant_payload_is_refused() {
        let (message, _) = only("let xs = [1]; nope::len(&xs)", &nothing);
        assert_eq!(message, WORDS);
    }

    #[test]
    fn a_reference_inside_an_option_payload_is_refused() {
        let (message, _) = only("let a = 1; Some(&a)", &nothing);
        assert_eq!(message, WORDS);
    }
}

/// A field the body reads where no path stored it.
#[test]
fn a_field_never_stored_names_the_value_that_lacks_it() {
    let (message, labels) = only("let x = { a: 1, }; x.b", &nothing);
    assert_eq!(
        message,
        "`x` has no `b` stored on every path that reaches here"
    );
    assert_eq!(labels, []);
}

/// `to_string` is declared for more than one type, so a `$param` only it
/// reads has no type the resolution can close.
#[test]
fn a_type_the_solve_leaves_open_is_refused_where_it_is_closed() {
    let (message, labels) = only("$count.to_string()", &nothing);
    assert_eq!(
        message,
        "cannot infer type: resolved to ! which contains unresolved type variables"
    );
    assert_eq!(labels, []);
}
