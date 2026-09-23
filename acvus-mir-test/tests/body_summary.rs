//! RFC-0064 steps 1 and 3 at their contract: what a body's result may
//! borrow, what a caller holds once it has one, and what a body in a cycle of
//! the call graph leaves with.

use acvus_mir::ty::{LenTerm, Mutability, ParamTerm, Poly, PolyParam, Ty, TypeArg, lift_to_poly};
use acvus_mir_test::compile_multi_fn_optimized;
use acvus_utils::Interner;

/// A program of local functions whose results are the ones under test, and a
/// `main` that calls them.
struct Program<'a> {
    helpers: Vec<Helper<'a>>,
    main: &'a str,
}

struct Helper<'a> {
    name: &'a str,
    body: &'a str,
    takes: Vec<PolyParam>,
}

fn sig(i: &Interner, params: &[(&str, Ty)]) -> Vec<PolyParam> {
    params
        .iter()
        .map(|(name, ty)| ParamTerm::<Poly>::new(i.intern(name), lift_to_poly(ty)))
        .collect()
}

fn borrow(m: Mutability, inner: Ty) -> Ty {
    Ty::Ref(m, Box::new(TypeArg::uniform(inner)))
}

fn numbers() -> Ty {
    Ty::Array(Box::new(Ty::I64), LenTerm::Known(3))
}

fn lends(i: &Interner, m: Mutability) -> Vec<PolyParam> {
    sig(i, &[("xs", borrow(m, numbers()))])
}

fn compile(i: &Interner, program: Program<'_>) -> Result<String, String> {
    let helpers: Vec<(&str, &str, Vec<PolyParam>)> = program
        .helpers
        .into_iter()
        .map(|Helper { name, body, takes }| (name, body, takes))
        .collect();
    compile_multi_fn_optimized(i, ("main", program.main), &helpers, &[], &[])
}

fn first(body: &str, takes: Vec<PolyParam>) -> Helper<'_> {
    Helper {
        name: "first",
        body,
        takes,
    }
}

fn admits(i: &Interner, program: Program<'_>) {
    if let Err(refusal) = compile(i, program) {
        panic!("expected the program to compile:\n{refusal}");
    }
}

fn refuses(i: &Interner, program: Program<'_>) -> String {
    match compile(i, program) {
        Ok(ir) => panic!("expected a refusal, compiled:\n{ir}"),
        Err(refusal) => refusal,
    }
}

// -- The result a body may leave with ---------------------------------

#[test]
fn a_result_borrowing_only_a_parameter_is_admitted_and_used() {
    let i = Interner::new();
    admits(
        &i,
        Program {
            helpers: vec![first("&$xs[0]", lends(&i, Mutability::Shared))],
            main: "let v = [1, 2, 3];\nlet r = first(&v);\nr + 0\n",
        },
    );
}

#[test]
fn a_result_borrowing_a_local_is_refused_at_the_local() {
    let i = Interner::new();
    let refusal = refuses(
        &i,
        Program {
            helpers: vec![first(
                "let l = [$xs[0], $xs[1], $xs[2]];\n&l[0]\n",
                lends(&i, Mutability::Shared),
            )],
            main: "let v = [1, 2, 3];\nfirst(&v) + 0\n",
        },
    );
    assert!(
        refusal.contains("a reference to `l` cannot leave the body"),
        "{refusal}"
    );
}

// -- What the caller holds --------------------------------------------

#[test]
fn writing_the_argument_while_the_result_lives_is_refused() {
    let i = Interner::new();
    let refusal = refuses(
        &i,
        Program {
            helpers: vec![first("&$xs[0]", lends(&i, Mutability::Shared))],
            main: "let v = [1, 2, 3];\nlet r = first(&v);\nv = [4, 5, 6];\nr + 0\n",
        },
    );
    assert!(
        refusal.contains("`v` is written here while a reference to it is live"),
        "{refusal}"
    );
}

#[test]
fn a_mutable_summary_excludes_a_read_of_the_argument() {
    let i = Interner::new();
    let refusal = refuses(
        &i,
        Program {
            helpers: vec![first("&$xs[0]", lends(&i, Mutability::Mut))],
            main: "let v = [1, 2, 3];\nlet r = first(&mut v);\nlet s = v[0];\nr + s\n",
        },
    );
    assert!(
        refusal.contains("while a reference to it is live"),
        "{refusal}"
    );
}

// -- A summary over two parameters ------------------------------------

/// A `match` of two arms holding two different parameters, whose summary
/// is the union of both (RFC-0064 rule 2).
fn pick(i: &Interner) -> Helper<'static> {
    Helper {
        name: "pick",
        body: "let l = &$a[0];\nlet r = &$b[0];\nif $flag { l } else { r }\n",
        takes: sig(
            i,
            &[
                ("a", borrow(Mutability::Shared, numbers())),
                ("b", borrow(Mutability::Shared, numbers())),
                ("flag", Ty::Bool),
            ],
        ),
    }
}

#[test]
fn a_union_summary_refuses_a_write_to_either_argument() {
    for written in ["x", "y"] {
        let i = Interner::new();
        let refusal = refuses(
            &i,
            Program {
                helpers: vec![pick(&i)],
                main: &format!(
                    "let x = [1, 2, 3];\nlet y = [4, 5, 6];\n\
                     let r = pick(&x, &y, true);\n{written} = [7, 8, 9];\nr + 0\n"
                ),
            },
        );
        assert!(
            refusal.contains(&format!(
                "`{written}` is written here while a reference to it is live"
            )),
            "{refusal}"
        );
    }
}

// -- Recursion is a fixpoint (RFC-0064 rule 4) --------------------

fn lends_and_stops(i: &Interner) -> Vec<PolyParam> {
    sig(
        i,
        &[
            ("xs", borrow(Mutability::Shared, numbers())),
            ("stop", Ty::Bool),
        ],
    )
}

/// One body in a cycle with itself, whose result is a reference to its
/// parameter in one arm and its own recursive call in the other.
fn deep(i: &Interner) -> Helper<'static> {
    Helper {
        name: "deep",
        body: "let here = &$xs[0];\nif $stop { here } else { deep(&$xs, true) }\n",
        takes: lends_and_stops(i),
    }
}

/// Two bodies in one cycle, each returning a reference to its own parameter
/// in one arm and the other's call in the second.
fn ping_pong(i: &Interner) -> Vec<Helper<'static>> {
    vec![
        Helper {
            name: "ping",
            body: "let here = &$xs[0];\nif $stop { here } else { pong(&$xs, true) }\n",
            takes: lends_and_stops(i),
        },
        Helper {
            name: "pong",
            body: "let here = &$xs[1];\nif $stop { here } else { ping(&$xs, true) }\n",
            takes: lends_and_stops(i),
        },
    ]
}

#[test]
fn a_recursive_result_borrowing_only_a_parameter_is_admitted() {
    let i = Interner::new();
    admits(
        &i,
        Program {
            helpers: vec![deep(&i)],
            main: "let v = [1, 2, 3];\ndeep(&v, false) + 0\n",
        },
    );
}

/// The first round of the fixpoint gives `deep` the empty summary, under
/// which this write is legal: the caller's result would hold no loan at all.
/// Only the summary the iteration settles on refuses it.
#[test]
fn the_settled_summary_refuses_the_write_the_first_round_would_admit() {
    let i = Interner::new();
    let refusal = refuses(
        &i,
        Program {
            helpers: vec![deep(&i)],
            main: "let v = [1, 2, 3];\nlet r = deep(&v, false);\nv = [4, 5, 6];\nr + 0\n",
        },
    );
    assert!(
        refusal.contains("`v` is written here while a reference to it is live"),
        "{refusal}"
    );
}

#[test]
fn a_mutually_recursive_pair_borrowing_their_parameters_is_admitted() {
    let i = Interner::new();
    admits(
        &i,
        Program {
            helpers: ping_pong(&i),
            main: "let v = [1, 2, 3];\nping(&v, false) + 0\n",
        },
    );
}

#[test]
fn a_mutually_recursive_pair_lends_the_argument_to_the_caller() {
    let i = Interner::new();
    let refusal = refuses(
        &i,
        Program {
            helpers: ping_pong(&i),
            main: "let v = [1, 2, 3];\nlet r = ping(&v, false);\nv = [4, 5, 6];\nr + 0\n",
        },
    );
    assert!(
        refusal.contains("`v` is written here while a reference to it is live"),
        "{refusal}"
    );
}

#[test]
fn a_recursive_result_borrowing_a_local_is_refused_at_the_local() {
    let i = Interner::new();
    let refusal = refuses(
        &i,
        Program {
            helpers: vec![Helper {
                name: "deep",
                body: "let l = [$xs[0], $xs[1], $xs[2]];\n\
                       if $stop { &l[0] } else { deep(&$xs, true) }\n",
                takes: lends_and_stops(&i),
            }],
            main: "let v = [1, 2, 3];\ndeep(&v, false) + 0\n",
        },
    );
    assert!(
        refusal.contains("a reference to `l` cannot leave the body"),
        "{refusal}"
    );
}

/// A lambda cannot call itself or a sibling bound after it — `let f = |k| ->
/// f(k - 1)` is "undefined function `f`" — so the makes-and-calls relation
/// over a module's closures has no cycle of its own, and `Label` summaries
/// enter the fixpoint only this way: they are rebuilt, inner closures first,
/// on every round of the component the module belongs to.
#[test]
fn a_closure_inside_a_cycle_carries_the_recursive_call() {
    let i = Interner::new();
    admits(
        &i,
        Program {
            helpers: vec![Helper {
                name: "deep",
                body: "let here = &$xs[0];\n\
                       let again = |b| -> deep(&$xs, b);\n\
                       if $stop { here } else { again(true) }\n",
                takes: lends_and_stops(&i),
            }],
            main: "let v = [1, 2, 3];\ndeep(&v, false) + 0\n",
        },
    );
}

/// The inliner is handed the cyclic components pass 0 already computed, so a
/// call inside its own callee is left standing. Without that set this program
/// does not finish compiling: `inline_body` loops until no call was spliced,
/// and splicing this one produces another of the same.
#[test]
fn a_self_recursive_body_compiles_through_the_inliner() {
    let i = Interner::new();
    admits(
        &i,
        Program {
            helpers: vec![Helper {
                name: "countdown",
                body: "if $n <= 0 { 0 } else { countdown($n - 1) }\n",
                takes: sig(&i, &[("n", Ty::I64)]),
            }],
            main: "countdown(3)\n",
        },
    );
}
