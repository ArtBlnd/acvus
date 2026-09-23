//! An argument the remaining candidates of a bare name take only by weaker
//! admissions of different kinds (RFC-0043): one takes it as a view of what
//! it lends, another only through a declared conversion. `Bag` converts to
//! and from a `Vec<i64>` by one rule each way, so `t::peek` takes a `&Vec`
//! through the reference, and `u::peek` takes the same argument as the
//! slice it lends; the second argument tells the two apart.

use acvus_extern::{Erased, ExternType, Registry, Runtime, extern_fn, extern_registry};
use acvus_interpreter::AcvusRuntime;
use acvus_interpreter_test::*;
use acvus_mir::graph::ParsedAst;
use acvus_mir::graph::optimize::Opt;
use acvus_mir::ty::Ty;
use acvus_utils::Interner;
use rustc_hash::FxHashMap;

#[derive(ExternType)]
#[extern_type(name = "Bag")]
#[repr(transparent)]
struct Bag(Vec<i64>);

#[extern_fn(effect = pure)]
#[extern_cast]
fn bag(items: Vec<i64>) -> Bag {
    Bag(items)
}

#[extern_fn(effect = pure)]
#[extern_cast]
fn unbag(b: Bag) -> Vec<i64> {
    b.0
}

#[extern_fn(name = "peek", effect = pure)]
fn peek_bag(b: &Bag, n: i64) -> i64 {
    b.0.len() as i64 * 100 + n
}

#[extern_fn(name = "peek", effect = pure)]
fn peek_slice<Rt>(s: &[Erased<Rt, i64>], wide: bool) -> i64
where
    Rt: Runtime,
{
    let sum: i64 = s.iter().map(|x| **x).sum();
    if wide { sum * 10 } else { sum }
}

fn registries() -> Vec<Registry<AcvusRuntime>> {
    let mut regs = acvus_ext::std_registries::<AcvusRuntime>();
    regs.push(extern_registry! {
        ns: "t",
        types: [Bag],
        fns: [bag, unbag, peek_bag],
    });
    regs.push(extern_registry! {
        ns: "u",
        fns: [peek_slice],
    });
    regs
}

/// RFC-0043: no conversion decision is opened at an argument some remaining
/// candidate takes by view, so the argument is held and settled after the
/// solve as rule 2 settles one, where no conversion is asked. Where the rest
/// of the call settles on `u::peek`, the view is the checker's at the
/// argument (rule 5); where it leaves only `t::peek`, which takes the
/// argument only through the conversion, that candidate leaves the set too.
#[tokio::test]
async fn a_held_argument_reaches_a_viewing_candidate_and_not_a_converting_one() {
    let i = Interner::new();
    let viewed = "let v = vec([1, 2]); peek(&v, true)";
    let got = run_script_with_externs(&i, viewed, Context::default(), registries(), Ty::I64)
        .await
        .value
        .as_int();
    assert_eq!(got, 30);
    let converted = "let v = vec([1, 2]); peek(&v, 1)";
    for opt in [Opt::None, Opt::Full] {
        let ast = ParsedAst::Script(acvus_ast::parse_script(&i, converted).expect("parse error"));
        let Err(refusal) = check_source(
            &i,
            ast,
            &FxHashMap::default(),
            registries(),
            Ty::I64,
            opt,
            |_| {},
        ) else {
            panic!("at {opt:?}, expected a refusal: {converted}");
        };
        let why = refusal.messages.join("\n");
        assert!(
            why.contains("no `peek` takes a call of type Fn(&Vec<i64>, i64)"),
            "at {opt:?}: {why}"
        );
    }
}
