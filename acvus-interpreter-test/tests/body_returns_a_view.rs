//! RFC-0062's waiting item on the machine: a body whose result is a view
//! lands it in the two adjacent registers of Decision 1, so `trim` and
//! `substring` reach a caller through a body and not only through an extern
//! call's `CallShape::Pair*`.

use acvus_extern::{Erased, Registry, Runtime, extern_fn, extern_registry};
use acvus_interpreter::{AcvusRuntime, Value};
use acvus_interpreter_test::{Helper, Refusal, check_graph, execute_compiled};
use acvus_mir::graph::ParsedAst;
use acvus_mir::graph::optimize::Opt;
use acvus_mir::ty::{Mutability, ParamTerm, Poly, PolyParam, Ty, TypeArg, lift_to_poly};
use acvus_utils::Interner;
use rustc_hash::FxHashMap;
use std::sync::Arc;

/// The sum of a view, as `slice_param.rs` reads one: the evidence that both
/// words of the pair arrived, since a wrong `ptr` reads other storage and a
/// wrong `len` reads another count of elements.
#[extern_fn(effect = pure)]
fn total<Rt>(a: &[Erased<Rt, i64>]) -> i64
where
    Rt: Runtime,
{
    a.iter().map(Erased::get).sum()
}

fn registries() -> Vec<Registry<AcvusRuntime>> {
    let mut regs = acvus_ext::std_registries::<AcvusRuntime>();
    regs.push(extern_registry! {
        ns: "t",
        fns: [total],
    });
    regs
}

fn borrowing(i: &Interner, name: &str, inner: Ty) -> Vec<PolyParam> {
    let ty = Ty::Ref(Mutability::Shared, Box::new(TypeArg::uniform(inner)));
    vec![ParamTerm::<Poly>::new(i.intern(name), lift_to_poly(&ty))]
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
        registries(),
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
    Ok(runtime.block_on(interp.execute()))
}

/// The number a program yields at both optimization levels, which is one
/// number or the differential's own contract is broken.
fn integer_at_both_levels(i: &Interner, helpers: &[Helper<'_>], main: &str) -> i64 {
    let full = compile_and_run(i, helpers, main, Opt::Full)
        .unwrap_or_else(|r| panic!("Opt::Full refused:\n  {}", r.messages.join("\n  ")))
        .as_int();
    let none = compile_and_run(i, helpers, main, Opt::None)
        .unwrap_or_else(|r| panic!("Opt::None refused:\n  {}", r.messages.join("\n  ")))
        .as_int();
    assert_eq!(full, none, "the two optimization levels read one view");
    full
}

fn refusal(i: &Interner, helpers: &[Helper<'_>], main: &str) -> String {
    match compile_and_run(i, helpers, main, Opt::Full) {
        Ok(value) => panic!("expected a refusal, ran to {value:?}"),
        Err(r) => r.messages.join("\n"),
    }
}

fn my_trim(i: &Interner) -> Helper<'_> {
    Helper {
        name: "my_trim",
        source: "trim($s)\n",
        params: borrowing(i, "s", Ty::Str),
    }
}

fn head(i: &Interner) -> Helper<'_> {
    Helper {
        name: "head",
        source: "$xs\n",
        params: borrowing(i, "xs", Ty::Slice(Box::new(Ty::I64))),
    }
}

// -- A view of a parameter leaves the body -----------------------------

#[test]
fn a_text_view_read_through_the_result_is_the_run_the_body_named() {
    let i = Interner::new();
    let value = integer_at_both_levels(
        &i,
        &[my_trim(&i)],
        "let s = \"  ab  \".to_string();\nlet v = my_trim(&s);\n\
         (len(&v) as i64) * 1000 + (char_at(&v, 0) as i64)\n",
    );
    assert_eq!(value, 2097);
}

#[test]
fn a_slice_result_reads_the_callers_own_elements() {
    let i = Interner::new();
    let value = integer_at_both_levels(
        &i,
        &[head(&i)],
        "let v = [10, 20, 30];\nt::total(head(&v))\n",
    );
    assert_eq!(value, 60);
}

#[test]
fn a_view_crosses_two_body_calls() {
    let i = Interner::new();
    let outer = Helper {
        name: "outer",
        source: "my_trim($s)\n",
        params: borrowing(&i, "s", Ty::Str),
    };
    let value = integer_at_both_levels(
        &i,
        &[my_trim(&i), outer],
        "let s = \"  abc  \".to_string();\nlet v = outer(&s);\nlen(&v) as i64\n",
    );
    assert_eq!(value, 3);
}

#[test]
fn a_view_result_is_an_externs_argument_where_it_stands() {
    let i = Interner::new();
    let value = integer_at_both_levels(
        &i,
        &[my_trim(&i)],
        "let s = \"  abcd  \".to_string();\nlen(my_trim(&s)) as i64\n",
    );
    assert_eq!(value, 4);
}

// -- The loans of the result -------------------------------------------

/// The summary of a body returning a view of its parameter is that
/// parameter's loan, so the caller writing the storage while the result is
/// live is one conflict — the same refusal a bare reference result raises.
#[test]
fn writing_the_argument_while_the_view_lives_is_refused() {
    let i = Interner::new();
    let refused = refusal(
        &i,
        &[my_trim(&i)],
        "let s = \"  ab  \".to_string();\nlet v = my_trim(&s);\n\
         s = \"xy\".to_string();\nlen(&v) as i64\n",
    );
    assert!(
        refused.contains("`s` is written here while a reference to it is live"),
        "{refused}"
    );
}

#[test]
fn a_view_of_a_local_is_refused_at_the_local() {
    let i = Interner::new();
    let refused = refusal(
        &i,
        &[Helper {
            name: "inner",
            source: "let t = $s.to_string();\ntrim(&t)\n",
            params: borrowing(&i, "s", Ty::Str),
        }],
        "let s = \"ab\".to_string();\nlen(inner(&s)) as i64\n",
    );
    assert!(
        refused.contains("a reference to `t` cannot leave the body"),
        "{refused}"
    );
}
