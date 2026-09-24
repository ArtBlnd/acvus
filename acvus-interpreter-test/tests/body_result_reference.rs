//! RFC-0064 rule 2 on the machine: a body whose result is a bare reference
//! is compiled, run, and the number the caller reads through it is the
//! number the program names.

use acvus_interpreter::Value;
use acvus_interpreter_test::{Helper, Refusal, check_graph, execute_compiled};
use acvus_mir::graph::ParsedAst;
use acvus_mir::graph::optimize::Opt;
use acvus_mir::ty::{LenTerm, Mutability, ParamTerm, Poly, PolyParam, Ty, TypeArg, lift_to_poly};
use acvus_utils::Interner;
use rustc_hash::FxHashMap;
use std::sync::Arc;

fn borrowing(i: &Interner, name: &str, m: Mutability, inner: Ty) -> Vec<PolyParam> {
    let ty = Ty::Ref(m, Box::new(TypeArg::uniform(inner)));
    vec![ParamTerm::<Poly>::new(i.intern(name), lift_to_poly(&ty))]
}

fn numbers() -> Ty {
    Ty::Array(Box::new(Ty::I64), LenTerm::Known(3))
}

fn strings() -> Ty {
    Ty::Array(Box::new(Ty::String), LenTerm::Known(2))
}

fn compile_and_run(
    i: &Interner,
    helper: Helper<'_>,
    main: &str,
    ret: Ty,
    opt: Opt,
) -> Result<Value, Refusal> {
    let parsed = ParsedAst::Script(acvus_ast::parse_script(i, main).expect("main parses"));
    let cr = check_graph(
        i,
        parsed,
        std::slice::from_ref(&helper),
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
    Ok(runtime.block_on(interp.execute()).expect("the page holds every context the run fetches first"))
}

/// The integer a program yields at both optimization levels. Disagreement is
/// the differential's own contract, so it fails here rather than being
/// reported as one number.
fn integer_at_both_levels(i: &Interner, helper: fn(&Interner) -> Helper<'_>, main: &str) -> i64 {
    let full = compile_and_run(i, helper(i), main, Ty::I64, Opt::Full)
        .unwrap_or_else(|r| panic!("Opt::Full refused:\n  {}", r.messages.join("\n  ")))
        .as_int();
    let none = compile_and_run(i, helper(i), main, Ty::I64, Opt::None)
        .unwrap_or_else(|r| panic!("Opt::None refused:\n  {}", r.messages.join("\n  ")))
        .as_int();
    assert_eq!(full, none, "the two optimization levels read one referent");
    full
}

fn refusal(i: &Interner, helper: Helper<'_>, main: &str) -> String {
    match compile_and_run(i, helper, main, Ty::I64, Opt::Full) {
        Ok(value) => panic!("expected a refusal, ran to {value:?}"),
        Err(r) => r.messages.join("\n"),
    }
}

fn first_number(i: &Interner) -> Helper<'_> {
    Helper {
        name: "first",
        source: "&$xs[0]",
        params: borrowing(i, "xs", Mutability::Shared, numbers()),
    }
}

fn first_string(i: &Interner) -> Helper<'_> {
    Helper {
        name: "first",
        source: "&$xs[0]",
        params: borrowing(i, "xs", Mutability::Shared, strings()),
    }
}

// -- The referent is a word in the caller's frame ----------------------

#[test]
fn a_number_read_through_the_result_is_the_element_lent() {
    let i = Interner::new();
    let value = integer_at_both_levels(
        &i,
        first_number,
        "let v = [10, 20, 30];\nlet r = first(&v);\n*r + 1\n",
    );
    assert_eq!(value, 11);
}

#[test]
fn reading_the_argument_again_leaves_the_result_readable() {
    let i = Interner::new();
    let value = integer_at_both_levels(
        &i,
        first_number,
        "let v = [10, 20, 30];\nlet r = first(&v);\nlet s = v[1];\n*r + s\n",
    );
    assert_eq!(value, 30);
}

// -- The referent's value is on the heap -------------------------------

/// A `&String` result: the one `Kind::Ref` word names an element of the
/// caller's array, and the bytes whose length the caller reads are the
/// heap allocation that element owns.
#[test]
fn a_length_read_through_the_result_is_the_lent_strings() {
    let i = Interner::new();
    let value = integer_at_both_levels(
        &i,
        first_string,
        "let v = [\"hello\".to_string(), \"wide world\".to_string()];\n\
         let r = first(&v);\n(r.len() as i64) + 1\n",
    );
    assert_eq!(value, 6);
}

/// `.len()` needs no `*`, so the length above would be the same number if
/// the result were a copy of the string rather than a reference to it. A
/// loan is what separates the two, and only a loan makes this refusal.
#[test]
fn the_string_result_holds_a_loan_on_the_argument() {
    let i = Interner::new();
    let refused = refusal(
        &i,
        first_string(&i),
        "let v = [\"hello\".to_string(), \"wide world\".to_string()];\n\
         let r = first(&v);\nv = [\"x\".to_string(), \"y\".to_string()];\n\
         (r.len() as i64)\n",
    );
    assert!(
        refused.contains("`v` is written here while a reference to it is live"),
        "{refused}"
    );
}

// -- What the caller may no longer do ----------------------------------

#[test]
fn writing_the_argument_while_the_result_lives_is_refused() {
    let i = Interner::new();
    let refused = refusal(
        &i,
        first_number(&i),
        "let v = [10, 20, 30];\nlet r = first(&v);\nv = [4, 5, 6];\n*r\n",
    );
    assert!(
        refused.contains("`v` is written here while a reference to it is live"),
        "{refused}"
    );
}

#[test]
fn a_result_borrowing_a_local_is_refused_at_the_local() {
    let i = Interner::new();
    let refused = refusal(
        &i,
        Helper {
            name: "first",
            source: "let l = [$xs[0], $xs[1], $xs[2]];\n&l[0]\n",
            params: borrowing(&i, "xs", Mutability::Shared, numbers()),
        },
        "let v = [10, 20, 30];\nfirst(&v) + 0\n",
    );
    assert!(
        refused.contains("a reference to `l` cannot leave the body"),
        "{refused}"
    );
}

// -- A named function in a cycle of the call graph (RFC-0079 rule 5) ------

fn lends_and_stops(i: &Interner, names: &[&str]) -> Vec<PolyParam> {
    let lent = Ty::Ref(Mutability::Shared, Box::new(TypeArg::uniform(numbers())));
    names
        .iter()
        .map(|name| ParamTerm::<Poly>::new(i.intern(name), lift_to_poly(&lent)))
        .chain([ParamTerm::<Poly>::new(
            i.intern("stop"),
            lift_to_poly(&Ty::Bool),
        )])
        .collect()
}

/// A body whose result is its parameter's element in one arm and its own
/// recursive call in the other.
fn deep(i: &Interner) -> Helper<'_> {
    Helper {
        name: "deep",
        source: "if $stop { &$xs[0] } else { deep(&$xs, true) }\n",
        params: lends_and_stops(i, &["xs"]),
    }
}

/// A body that passes both parameters to itself and returns only the
/// first's element: the least fixpoint of its flows names `a` alone.
fn keep_first(i: &Interner) -> Helper<'_> {
    Helper {
        name: "keep",
        source: "if $stop { &$a[0] } else { keep(&$a, &$b, true) }\n",
        params: lends_and_stops(i, &["a", "b"]),
    }
}

#[test]
fn a_recursive_result_borrowing_its_parameter_runs() {
    let i = Interner::new();
    let value = integer_at_both_levels(&i, deep, "let v = [10, 20, 30];\n*deep(&v, false) + 1\n");
    assert_eq!(value, 11);
}

#[test]
fn a_recursive_result_holds_its_lender() {
    let i = Interner::new();
    let refused = refusal(
        &i,
        deep(&i),
        "let v = [10, 20, 30];\nlet r = deep(&v, false);\nv = [4, 5, 6];\n*r\n",
    );
    assert!(
        refused.contains("`v` is written here while a reference to it is live"),
        "{refused}"
    );
}

/// The flows start at none and climb through the recursive call, so the
/// argument the recursion only passes along is not among them: the second
/// lender is free once the call returns.
#[test]
fn a_recursive_call_passing_a_parameter_along_leaves_its_lender_free() {
    let i = Interner::new();
    let value = integer_at_both_levels(
        &i,
        keep_first,
        "let v = [10, 20, 30];\nlet w = [1, 2, 3];\nlet r = keep(&v, &w, false);\n\
         w = [4, 5, 6];\n*r + w[0]\n",
    );
    assert_eq!(value, 14);
}

#[test]
fn a_recursive_result_that_could_hold_a_local_is_refused() {
    let i = Interner::new();
    let refused = refusal(
        &i,
        Helper {
            name: "deep",
            source: "let l = [$xs[0], $xs[1], $xs[2]];\n\
                     if $stop { &l[0] } else { deep(&$xs, true) }\n",
            params: lends_and_stops(&i, &["xs"]),
        },
        "let v = [10, 20, 30];\n*deep(&v, false)\n",
    );
    assert!(
        refused.contains("a reference to `l` cannot leave the body"),
        "{refused}"
    );
}
