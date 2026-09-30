//! RFC-0064 rule 5 on the machine: a lambda that captured a reference is
//! compiled and run, and the number it reads through the capture is the
//! number the program names.
//!
//! Every program runs at both optimization levels. At `Opt::Full` a small
//! pure closure called where it was made is spliced into its caller
//! (RFC-0060) and the capture becomes the reference the caller already held;
//! at `Opt::None` the closure stays a closure and the `Kind::Ref` word
//! travels in its capture list. The two must read one referent.

use acvus_interpreter::Value;
use acvus_interpreter_test::{Helper, Refusal, check_graph, execute_compiled};
use acvus_mir::graph::ParsedAst;
use acvus_mir::graph::optimize::Opt;
use acvus_mir::ty::{LenTerm, Mutability, ParamTerm, Poly, PolyParam, Ty, TypeArg, lift_to_poly};
use acvus_utils::Interner;
use rustc_hash::FxHashMap;
use std::sync::Arc;

fn numbers() -> Ty {
    Ty::Array(Box::new(Ty::I64), LenTerm::Known(3))
}

fn borrowing(i: &Interner, name: &str, inner: Ty) -> Vec<PolyParam> {
    let ty = Ty::Ref(Mutability::Shared, Box::new(TypeArg::uniform(inner)));
    vec![ParamTerm::<Poly>::new(i.intern(name), lift_to_poly(&ty))]
}

fn compile_and_run(
    i: &Interner,
    helpers: &[Helper<'_>],
    main: &str,
    ret: Ty,
    opt: Opt,
) -> Result<Value, Refusal> {
    let parsed = ParsedAst::Script(acvus_ast::parse_script(i, main).expect("main parses"));
    let cr = check_graph(
        i,
        parsed,
        helpers,
        &FxHashMap::default(),
        acvus_ext::std_registries::<acvus_interpreter::AcvusRuntime>(),
        ret,
        opt,
        |_| {},
    )?;
    let (_, mut interp) = execute_compiled(
        i,
        cr,
        std::collections::HashMap::new(),
        Arc::new(acvus_interpreter::SequentialExecutor),
    );
    let runtime = tokio::runtime::Builder::new_current_thread()
        .build()
        .expect("a current-thread runtime");
    Ok(runtime.block_on(interp.execute()).expect("the seeds hold every context the run fetches"))
}

/// The integer a program yields at both optimization levels. Disagreement is
/// the differential's own contract, so it fails here rather than being
/// reported as one number.
fn integer_at_both_levels(main: &str) -> i64 {
    integer_at_both_levels_with(&Interner::new(), &[], main)
}

/// A helper's declared parameter names are `Astr`s of `i`, which is why the
/// interner the program compiles in is the caller's rather than one minted
/// here.
fn integer_at_both_levels_with(i: &Interner, helpers: &[Helper<'_>], main: &str) -> i64 {
    let at = |opt| {
        compile_and_run(i, helpers, main, Ty::I64, opt)
            .unwrap_or_else(|r| panic!("{opt:?} refused:\n  {}", r.messages.join("\n  ")))
            .as_int()
    };
    let full = at(Opt::Full);
    let none = at(Opt::None);
    assert_eq!(full, none, "the two optimization levels read one referent");
    full
}

fn refusal(i: &Interner, helpers: &[Helper<'_>], main: &str) -> String {
    match compile_and_run(i, helpers, main, Ty::I64, Opt::Full) {
        Ok(value) => panic!("expected a refusal, ran to {value:?}"),
        Err(r) => r.messages.join("\n"),
    }
}

const CAPTURING: &str = "let v = [1, 2, 3];\nlet r = &v;\nlet f = |k| -> len(r) + k;\n";

// -- The capture is read at the call -----------------------------------

#[test]
fn a_length_read_through_a_captured_reference_is_the_borrowed_arrays() {
    assert_eq!(
        integer_at_both_levels(&format!("{CAPTURING}f(1) as i64\n")),
        4
    );
}

#[test]
fn reading_the_borrowed_storage_again_leaves_the_lambda_callable() {
    assert_eq!(
        integer_at_both_levels(&format!("{CAPTURING}let n = v[1];\n(f(1) as i64) + n\n")),
        6
    );
}

/// The reference is captured by an inner lambda through an outer one, so the
/// `Kind::Ref` word is copied out of one capture list into another.
#[test]
fn a_reference_captured_through_two_lambdas_is_read_at_the_inner_call() {
    assert_eq!(
        integer_at_both_levels(
            "let v = [1, 2, 3];\nlet r = &v;\n\
             let outer = |k| -> { let inner = |j| -> len(r) + j; inner(k) };\nouter(1) as i64\n"
        ),
        4
    );
}

/// Inside `apply`, `h` is a parameter that holds a loan, so what `h(1)`
/// borrows is what the argument borrowed.
#[test]
fn a_capturing_lambda_passed_to_another_lambda_is_called_there() {
    assert_eq!(
        integer_at_both_levels(
            "let v = [1, 2, 3];\nlet r = &v;\nlet g = |k| -> len(r) + k;\n\
             let apply = |h, x| -> h(x);\napply(g, 1) as i64\n"
        ),
        4
    );
}

// -- What a lambda leaves with ----------------------------------------

#[test]
fn a_lambda_returning_a_reference_to_its_parameter_is_read_by_the_caller() {
    assert_eq!(
        integer_at_both_levels("let v = [1, 2, 3];\nlet f = |xs| -> &xs[0];\n*f(&v) + 1\n"),
        2
    );
}

/// A lambda that captured an owned `String` reads it through a reference into
/// the closure's own capture, so letting that reference out would outlive the
/// closure. It ran and printed a freed string before the result rule reached a
/// closure body; the storage the reference names is a local of the closure and
/// the refusal is step 1's.
#[test]
fn a_lambda_returning_a_reference_into_its_own_capture_is_refused() {
    let refused = refusal(
        &Interner::new(),
        &[],
        "let s = \"ab\".to_string();\nlet f = |y| -> s;\n(f(1).len() as i64)\n",
    );
    assert!(
        refused.contains("a reference to `s` cannot leave the body"),
        "{refused}"
    );
}

/// The counterfactual for the test below, one variable apart: the lambda a
/// named body returns captures nothing, so nothing RFC-0064 rule 5 changed
/// is in it.
#[test]
fn a_lambda_returned_from_a_body_capturing_nothing_is_called_by_the_caller() {
    assert_eq!(
        integer_at_both_levels_with(
            &Interner::new(),
            std::slice::from_ref(&Helper {
                name: "make",
                source: "|k| -> k + 1\n",
                params: vec![],
            }),
            "let g = make();\ng(1)\n"
        ),
        2
    );
}

/// The body's summary carries the loan the returned lambda captured, which
/// is how the caller may hold it.
#[test]
fn a_lambda_returned_from_a_body_keeps_the_parameter_it_captured() {
    let i = Interner::new();
    let helper = Helper {
        name: "make",
        source: "|k| -> len($xs) + k\n",
        params: borrowing(&i, "xs", numbers()),
    };
    assert_eq!(
        integer_at_both_levels_with(
            &i,
            std::slice::from_ref(&helper),
            "let v = [1, 2, 3];\nlet g = make(&v);\ng(1) as i64\n"
        ),
        4
    );
}

// -- The refusals ------------------------------------------------------

/// RFC-0064 rule 5: the two places the refusal names are the
/// capture and the write.
#[test]
fn writing_the_borrowed_storage_while_the_lambda_is_live_is_refused() {
    let refused = refusal(
        &Interner::new(),
        &[],
        &format!("{CAPTURING}v = [4, 5, 6];\nf(1) as i64\n"),
    );
    assert!(
        refused.contains("`v` is written here while a reference to it is live"),
        "{refused}"
    );
}

#[test]
fn a_lambda_whose_result_borrows_its_own_local_is_refused_at_the_local() {
    let refused = refusal(
        &Interner::new(),
        &[],
        "let f = |k| -> { let l = [k, k, k]; &l[0] };\n*f(1) + 1\n",
    );
    assert!(
        refused.contains("a reference to `l` cannot leave the body"),
        "{refused}"
    );
}

/// Step 1's rule reaching a closure value: what leaves a body may name the
/// body's parameters and no local of it.
#[test]
fn a_lambda_returned_from_a_body_may_not_hold_a_local_of_it() {
    let i = Interner::new();
    let refused = refusal(
        &i,
        std::slice::from_ref(&Helper {
            name: "make",
            source: "let l = [$xs[0], $xs[1], $xs[2]];\nlet p = &l;\n|k| -> len(p) + k\n",
            params: borrowing(&i, "xs", numbers()),
        }),
        "let v = [1, 2, 3];\nlet g = make(&v);\ng(1) as i64\n",
    );
    assert!(
        refused.contains("a reference to `l` cannot leave the body"),
        "{refused}"
    );
}

#[test]
fn a_capturing_lambda_is_not_stored_in_a_list() {
    let refused = refusal(
        &Interner::new(),
        &[],
        &format!("{CAPTURING}let l = [f];\nlen(&l) as i64\n"),
    );
    assert!(
        refused.contains("a reference cannot be stored in a list, object, or tuple"),
        "{refused}"
    );
}

/// A view is the register pair of RFC-0047 rule 6 and a capture is one
/// word, so this capture stays refused where a bare reference is admitted.
#[test]
fn a_lambda_capturing_a_view_is_refused() {
    let refused = refusal(
        &Interner::new(),
        &[],
        "let s = \"abc\";\nlet f = |k| -> (s.len() as i64) + k;\nf(1)\n",
    );
    assert!(
        refused.contains("a lambda cannot capture a string or slice view"),
        "{refused}"
    );
}
