//! RFC-0064 step 1 at its contract: what a body's result may borrow, and
//! what a caller holds once it has one.

use acvus_mir::ty::{LenTerm, Mutability, ParamTerm, Poly, PolyParam, Ty, TypeArg, lift_to_poly};
use acvus_mir_test::{compile_multi_fn_optimized, compile_multi_fn_required};
use acvus_utils::Interner;

/// A program of two local functions: `helper`, whose result is the one under
/// test, and `main`, which calls it.
struct Program<'a> {
    helper: Helper<'a>,
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
    let Helper { name, body, takes } = program.helper;
    compile_multi_fn_optimized(i, ("main", program.main), &[(name, body, takes)], &[], &[])
}

fn compile_unoptimized(i: &Interner, program: Program<'_>) -> Result<String, String> {
    let Helper { name, body, takes } = program.helper;
    compile_multi_fn_required(i, ("main", program.main), &[(name, body, takes)], &[], &[])
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
            helper: first("&$xs[0]", lends(&i, Mutability::Shared)),
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
            helper: first(
                "let l = [$xs[0], $xs[1], $xs[2]];\n&l[0]\n",
                lends(&i, Mutability::Shared),
            ),
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
            helper: first("&$xs[0]", lends(&i, Mutability::Shared)),
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
            helper: first("&$xs[0]", lends(&i, Mutability::Mut)),
            main: "let v = [1, 2, 3];\nlet r = first(&mut v);\nlet s = v[0];\nr + s\n",
        },
    );
    assert!(
        refusal.contains("while a reference to it is live"),
        "{refusal}"
    );
}

// -- A summary over two parameters ------------------------------------

/// The `match` of RFC-0064's "where this is hard": two arms holding two
/// different parameters, whose summary is the union of both.
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
                helper: pick(&i),
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

// -- Recursion -------------------------------------------------------

#[test]
fn a_recursive_body_whose_result_is_a_reference_is_refused_by_name() {
    let i = Interner::new();
    let program = Program {
        helper: Helper {
            name: "deep",
            body: "let here = &$xs[0];\nif $stop { here } else { deep(&$xs, true) }\n",
            takes: sig(
                &i,
                &[
                    ("xs", borrow(Mutability::Shared, numbers())),
                    ("stop", Ty::Bool),
                ],
            ),
        },
        main: "let v = [1, 2, 3];\ndeep(&v, false) + 0\n",
    };
    let refusal = match compile_unoptimized(&i, program) {
        Ok(ir) => panic!("expected a refusal, compiled:\n{ir}"),
        Err(refusal) => refusal,
    };
    assert!(
        refusal.contains("a recursive body's result may not be a reference"),
        "{refusal}"
    );
}
