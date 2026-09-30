//! RFC-0064 rule 4 on the machine: a body in a cycle of the call graph
//! returns a reference to its parameter, and the number the caller reads
//! through it is the number the program names.

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

fn lent_numbers() -> Ty {
    Ty::Ref(Mutability::Shared, Box::new(TypeArg::uniform(numbers())))
}

fn param(i: &Interner, name: &str, ty: Ty) -> PolyParam {
    ParamTerm::<Poly>::new(i.intern(name), lift_to_poly(&ty))
}

fn compile_and_run(
    i: &Interner,
    helpers: &[Helper<'_>],
    main: &str,
    opt: Opt,
) -> Result<Value, Refusal> {
    let parsed = ParsedAst::Script(acvus_ast::parse_script(i, main).expect("main parses"));
    let cr = check_graph(
        i,
        parsed,
        helpers,
        &FxHashMap::default(),
        acvus_ext::std_registries::<acvus_interpreter::AcvusRuntime>(),
        Ty::I64,
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
fn integer_at_both_levels(
    i: &Interner,
    helpers: fn(&Interner) -> Vec<Helper<'_>>,
    main: &str,
) -> i64 {
    let full = compile_and_run(i, &helpers(i), main, Opt::Full)
        .unwrap_or_else(|r| panic!("Opt::Full refused:\n  {}", r.messages.join("\n  ")))
        .as_int();
    let none = compile_and_run(i, &helpers(i), main, Opt::None)
        .unwrap_or_else(|r| panic!("Opt::None refused:\n  {}", r.messages.join("\n  ")))
        .as_int();
    assert_eq!(full, none, "the two optimization levels read one referent");
    full
}

fn refusal(i: &Interner, helpers: &[Helper<'_>], main: &str) -> String {
    match compile_and_run(i, helpers, main, Opt::Full) {
        Ok(value) => panic!("expected a refusal, ran to {value:?}"),
        Err(r) => r.messages.join("\n"),
    }
}

fn first_nonzero(i: &Interner) -> Vec<Helper<'_>> {
    vec![Helper {
        name: "first_nonzero",
        source: "let here = &$xs[$i];\n\
                 if *here != 0 { here } else {\n\
                 if $i >= 2 { here } else { first_nonzero(&$xs, $i + 1) } }\n",
        params: vec![param(i, "xs", lent_numbers()), param(i, "i", Ty::U64)],
    }]
}

fn ping_pong(i: &Interner) -> Vec<Helper<'_>> {
    vec![
        Helper {
            name: "ping",
            source: "let here = &$xs[0];\nif $stop { here } else { pong(&$xs, true) }\n",
            params: vec![param(i, "xs", lent_numbers()), param(i, "stop", Ty::Bool)],
        },
        Helper {
            name: "pong",
            source: "let here = &$xs[1];\nif $stop { here } else { ping(&$xs, true) }\n",
            params: vec![param(i, "xs", lent_numbers()), param(i, "stop", Ty::Bool)],
        },
    ]
}

// -- What the recursion lends ------------------------------------------

#[test]
fn a_number_read_through_a_recursive_result_is_the_element_lent() {
    let i = Interner::new();
    let value = integer_at_both_levels(
        &i,
        first_nonzero,
        "let v = [0, 0, 7];\nlet r = first_nonzero(&v, 0);\n*r + 1\n",
    );
    assert_eq!(value, 8);
}

/// The counterfactual for the test above, one element apart: the walk stops
/// at the first non-zero, so a different element is the referent.
#[test]
fn the_recursion_stops_at_the_element_the_program_names() {
    let i = Interner::new();
    let value = integer_at_both_levels(
        &i,
        first_nonzero,
        "let v = [0, 5, 7];\nlet r = first_nonzero(&v, 0);\n*r + 1\n",
    );
    assert_eq!(value, 6);
}

#[test]
fn a_mutually_recursive_pair_lends_through_both_bodies() {
    let i = Interner::new();
    let value = integer_at_both_levels(
        &i,
        ping_pong,
        "let v = [10, 20, 30];\nlet r = ping(&v, false);\n*r + 1\n",
    );
    assert_eq!(value, 21);
}

// -- What the caller may no longer do -----------------------------------

#[test]
fn writing_the_argument_while_a_recursive_result_lives_is_refused() {
    let i = Interner::new();
    let refused = refusal(
        &i,
        &first_nonzero(&i),
        "let v = [0, 0, 7];\nlet r = first_nonzero(&v, 0);\nv = [1, 2, 3];\n*r\n",
    );
    assert!(
        refused.contains("`v` is written here while a reference to it is live"),
        "{refused}"
    );
}

#[test]
fn a_recursive_result_borrowing_a_local_is_refused_at_the_local() {
    let i = Interner::new();
    let refused = refusal(
        &i,
        &[Helper {
            name: "deep",
            source: "let l = [$xs[0], $xs[1], $xs[2]];\n\
                     if $stop { &l[0] } else { deep(&$xs, true) }\n",
            params: vec![param(&i, "xs", lent_numbers()), param(&i, "stop", Ty::Bool)],
        }],
        "let v = [10, 20, 30];\ndeep(&v, false) + 0\n",
    );
    assert!(
        refused.contains("a reference to `l` cannot leave the body"),
        "{refused}"
    );
}

// -- What the inliner does with a cycle ---------------------------------

/// Before `graph::optimize` handed `inliner::inline` the functions of every
/// cyclic component, every caller passed it the empty set, and this program
/// did not finish compiling at `Opt::Full`: `inline_body` loops until no call
/// was spliced, and splicing `countdown`'s call to itself always produces
/// another of the same. The step-1 report stopped it after 300 seconds.
#[test]
fn a_self_recursive_body_compiles_and_runs_at_both_levels() {
    let i = Interner::new();
    let value = integer_at_both_levels(
        &i,
        |i| {
            vec![Helper {
                name: "countdown",
                source: "if $n <= 0 { 0 } else { countdown($n - 1) + 1 }\n",
                params: vec![param(i, "n", Ty::I64)],
            }]
        },
        "countdown(3)\n",
    );
    assert_eq!(value, 3);
}
