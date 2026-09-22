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

/// The one refusal of `source`, as the words its primary marker carries.
fn only_primary(source: &str) -> Option<String> {
    let refusals = refusals(source, &nothing);
    assert_eq!(refusals.len(), 1, "{source}: {:#?}", words(&refusals));
    refusals[0].primary.clone()
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

/// RFC-0064 "What it costs": a lambda called after the storage it borrows
/// was written is the one case worth its own words, and the two places it
/// names are the capture and the call.
#[test]
fn a_write_while_a_capturing_lambda_is_live_names_the_capture_and_the_call() {
    let (message, labels) = only(
        "let a = [1, 2]; let r = &a; let f = |k| -> len(r) + k; a = [3, 4]; f(1)",
        &nothing,
    );
    assert_eq!(
        message,
        "`a` is written here while a reference to it is live"
    );
    assert_eq!(
        labels,
        [
            at("|k| -> len(r) + k", "captured here"),
            at("f(1)", "the lambda is called here"),
        ]
    );
}

#[test]
fn a_lambda_whose_later_use_is_not_a_call_is_labelled_as_used() {
    let (message, labels) = only(
        "let a = [1, 2]; let r = &a; let f = |k| -> len(r) + k; \
         let g = |h, x| -> h(x); a = [3, 4]; g(f, 1)",
        &nothing,
    );
    assert_eq!(
        message,
        "`a` is written here while a reference to it is live"
    );
    assert_eq!(
        labels,
        [
            at("|k| -> len(r) + k", "captured here"),
            at("f", "the lambda is used here"),
        ]
    );
}

#[test]
fn a_write_while_two_references_are_live_is_one_refusal_naming_both() {
    let (message, labels) = only(
        "let a = [1, 2]; let r = &a; let s = &a; a = [3, 4]; len(r) + len(s)",
        &nothing,
    );
    assert_eq!(
        message,
        "`a` is written here while a reference to it is live"
    );
    assert_eq!(
        labels,
        [
            at("&a", "borrowed here"),
            at("r", "the reference is used here"),
            at("&a", "borrowed here"),
            at("s", "the reference is used here"),
        ]
    );
}

/// `Solver::lend` answers that the referent is shared without saying where
/// the shared reference was taken, so the refusal names the reference the
/// borrow was written on and the spelling that would admit it.
#[test]
fn a_mutable_borrow_of_a_shared_reference_names_one_place() {
    let (message, labels) = only("let a = [1, 2]; let r = &a; let m = &mut r; 0", &nothing);
    assert_eq!(
        message,
        "`r` is a shared reference and cannot be borrowed mutably; bind it with `&mut`"
    );
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

    /// The message names two places, so it says nothing at either of them;
    /// the primary marker carries the sentence that does.
    #[test]
    fn the_primary_marker_carries_its_own_words() {
        assert_eq!(
            only_primary("let a = [1, 2] | into_iter; let b = [3, 4] | into_iter; [a, b]; 0"),
            Some("`a` and `b` meet here".to_string())
        );
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
            "heterogeneous list: expected Items<i64>, got Items<String>"
        );
        assert_eq!(labels, []);
    }
}

/// Every refusal but the one above leaves its primary marker repeating the
/// message, which is `Report`'s rule where no primary text is set.
#[test]
fn a_refusal_that_says_what_it_points_at_carries_no_primary_text() {
    assert_eq!(only_primary("let a = [1, 2]; let b = a; a"), None);
}

/// An argument whose container the solve names only after checking is
/// reported by the two references, not by the two referents: `expected
/// [i64], got i64` named types no argument of the call has.
#[test]
fn a_deferred_argument_mismatch_names_the_two_references() {
    let (message, labels) = only("let f = |k| -> chars(&k); f(1)", &nothing);
    assert_eq!(message, "type mismatch: expected &str, got &i64");
    assert_eq!(labels, []);
}

/// `k`'s type is fixed by nothing, so the solve never binds it. `!` is the
/// type of an expression that does not return, and this is not one.
#[test]
fn a_type_the_solve_never_bound_prints_as_an_underscore() {
    let refusals = refusals("let f = |k, m| -> { let a = len(k); k < m }; 0", &nothing);
    let messages = words(&refusals);
    assert!(
        messages
            .iter()
            .any(|m| m.ends_with("type mismatch in `<`: _ vs _")),
        "{messages:#?}"
    );
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

/// A body's result is not a view, nor data holding a reference: the `&str`
/// shape keeps the spelling that owns the text. RFC-0064 rule 1 admits a
/// bare reference, and what it may name is the borrow check's to say.
mod reference_returned_from_body {
    use super::*;

    const WORDS: &str = "a body does not return a reference";

    #[test]
    fn a_reference_to_a_local_is_refused_where_the_local_was_borrowed() {
        let (message, labels) = only("let a = 1; &a", &nothing);
        assert_eq!(message, "a reference to `a` cannot leave the body");
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
    fn a_reference_to_a_temporary_is_refused_as_a_reference_to_a_local_is() {
        let (message, _) = only("let a = 1; &(a + 1)", &nothing);
        assert_eq!(message, "a reference to a local cannot leave the body");
    }
}

/// A reference is not an `Option`'s or a `Result`'s payload. `MakeSome`
/// moves one value where a view is two, so `Some("ab")` read a neighbouring
/// register for the length until this refusal existed.
mod reference_in_a_payload {
    use super::*;

    const WORDS: &str = "a reference cannot be stored in an Option or a Result";

    #[test]
    fn a_view_in_a_some_carries_the_way_to_own_the_text() {
        let (message, labels) = only(
            "let o = Some(\"ab\"); let n = 0; if let Some(s) = o { n = s.len(); }; n",
            &nothing,
        );
        assert_eq!(
            message,
            format!("{WORDS}; write `.to_string()` to store the text")
        );
        assert_eq!(labels, []);
    }

    #[test]
    fn a_reference_in_an_ok_is_refused() {
        let (message, labels) = only("let s = \"ab\".to_string(); let r = Ok(&s); 0", &nothing);
        assert_eq!(message, WORDS);
        assert_eq!(labels, []);
    }

    #[test]
    fn a_reference_in_an_err_is_refused() {
        let (message, labels) = only("let s = \"ab\".to_string(); let r = Err(&s); 0", &nothing);
        assert_eq!(message, WORDS);
        assert_eq!(labels, []);
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
/// reads has no type the resolution can close, and the call is refused as
/// the one decision left open.
#[test]
fn a_type_the_solve_leaves_open_is_refused_where_it_is_closed() {
    let (message, labels) = only("$count.to_string()", &nothing);
    assert_eq!(
        message,
        "cannot infer type: resolved to Fn(&_) -> String which contains unresolved type variables"
    );
    assert_eq!(labels, []);
}

/// A refusal over a name the environment does not have offers the names it
/// does: the functions a call of that spelling could have reached, the
/// bindings in scope, an object's fields, a type's variants.
mod did_you_mean {
    use super::*;

    #[test]
    fn a_misspelled_function_offers_the_one_that_is_near() {
        let (message, _) = only("let v = vec([1, 2]); v.pushh(2); v.len()", &nothing);
        assert_eq!(message, "undefined function `pushh`; did you mean `push`?");
    }

    #[test]
    fn a_name_with_nothing_near_it_keeps_the_sentence_it_had() {
        let (message, _) = only("frobnicate(1)", &nothing);
        assert_eq!(message, "undefined function `frobnicate`");
    }

    #[test]
    fn a_misspelled_binding_offers_the_binding() {
        let (message, _) = only("let total = 1; totl + 1", &nothing);
        assert_eq!(message, "undefined variable `totl`; did you mean `total`?");
    }

    #[test]
    fn a_misspelled_field_offers_the_field() {
        let (message, _) = only("let p = { name: 1, age: 3, }; p.nmae", &nothing);
        assert_eq!(
            message,
            "`p` has no `nmae` stored on every path that reaches here; did you mean `name`?"
        );
    }

    #[test]
    fn a_misspelled_variant_offers_the_variant() {
        let (message, _) = only(
            "let s = Shape::Circle(1); match s { Shape::Circl(r) => r, _ => 0, }",
            &nothing,
        );
        assert_eq!(
            message,
            "unreachable pattern: `Shape::Circl(_)` is not a variant of `Shape{Circle(i64)}`; \
             did you mean `Shape::Circle`?"
        );
    }
}

/// A parameter declared as a reference, given the value itself, carries the
/// call as it should have been written (RFC-0064's holder labels are what
/// names the places; this names the spelling).
#[test]
fn a_value_at_a_reference_parameter_carries_the_call_to_write() {
    let (_, labels) = only("let v = vec([1, 2]); len(v)", &nothing);
    assert_eq!(labels, [note("the parameter is a `&`; write `len(&v)`")]);
}

/// A rule is broken once. What the error type it leaves reaches refuses
/// again at a smaller piece of the same source, and that second refusal is
/// the first one's consequence.
mod a_consequence_is_not_reported {
    use super::*;

    #[test]
    fn an_arm_that_is_not_a_variant_refuses_once() {
        let (message, _) = only(
            "let s = Shape::Circle(1); match s { Shape::Circle(r) => r, Shape::Square(q) => q, }",
            &nothing,
        );
        assert_eq!(
            message,
            "unreachable pattern: `Shape::Square(_)` is not a variant of `Shape{Circle(i64)}`"
        );
    }

    #[test]
    fn a_store_to_a_name_no_binding_has_refuses_once() {
        let (message, _) = only("total = 1; total", &nothing);
        assert_eq!(
            message,
            "cannot assign to `total`: no binding named `total` is in scope; \
             `let total = ...;` binds it"
        );
    }
}
