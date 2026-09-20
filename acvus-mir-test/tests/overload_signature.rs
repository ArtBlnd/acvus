//! RFC-0043 where the candidate set mixes kinds: a plain declaration of a
//! name beside a shared signature (RFC-0019) of the same name, settled by
//! the receiver's ground type — and by the receiver's type alone, whenever
//! the rest of the call has already emptied the set.
//!
//! `fx_p::f(&str, &str)` is the plain one, `fx_q::f<T>(&Vec<T>, &Vec<T>)`
//! the signature, with an `i64` instance. `fx_q::max<T>(&Vec<T>)` stands
//! beside `iter::max` for the same question over a consumer, and
//! `fx_p::map` beside `iter::map` for one over a closure argument.

use acvus_extern::{
    Closure, Externs, Ref, Registry, Runtime, Shared, TypesOnly, Var, extern_fn, extern_registry,
    extern_signature, kind,
};
use acvus_mir::graph::{
    CompilationGraph, FnKind, Function, ParsedAst, QualifiedRef, extract, infer,
};
use acvus_mir::ir::Callee;
use acvus_mir::ty::{PolyBuilder, Ty, TyTerm, TypeRegistry};
use acvus_utils::{Freeze, Interner};
use rustc_hash::FxHashMap;

mod fx_p {
    use super::*;

    #[extern_fn(effect = pure)]
    pub fn f(s: &str, pat: &str) -> bool {
        let _ = (s, pat);
        unreachable!("a type-only fixture is never run")
    }

    fn map_now<T, U, E, Rt>(
        rt: &Rt,
        frame: &mut Rt::Frame<'_>,
        val: Option<T>,
        f: Closure<(T,), U, E, Rt>,
    ) -> Option<U>
    where
        T: Var<kind::Type> + acvus_extern::OneValue<Rt> + acvus_extern::Cross<Rt>,
        U: Var<kind::Type> + acvus_extern::OneValue<Rt>,
        E: Var<kind::Effect>,
        Rt: Runtime,
    {
        let _ = (rt, frame, val, f);
        unreachable!("a type-only fixture is never run")
    }

    #[extern_fn(effect = E, sync = map_now)]
    pub async fn map<T, U, E, Rt>(
        rt: &Rt,
        frame: &mut Rt::Frame<'_>,
        val: Option<T>,
        f: Closure<(T,), U, E, Rt>,
    ) -> Option<U>
    where
        T: Var<kind::Type> + acvus_extern::OneValue<Rt> + acvus_extern::Cross<Rt>,
        U: Var<kind::Type> + acvus_extern::OneValue<Rt>,
        E: Var<kind::Effect>,
        Rt: Runtime,
    {
        let _ = (rt, frame, val, f);
        unreachable!("a type-only fixture is never run")
    }

    pub fn registry() -> Registry<TypesOnly> {
        extern_registry! {
            ns: "fx_p",
            fns: [f, map],
        }
    }
}

mod fx_q {
    use super::*;

    extern_signature! {
        ns: "fx_q",
        fn f<T>(c: &Vec<T>, other: &Vec<T>) -> bool
        where
            T: Var<kind::Type>;
    }

    extern_signature! {
        ns: "fx_q",
        fn max<T>(c: &Vec<T>) -> Option<T>
        where
            T: Var<kind::Type>;
    }

    #[extern_fn(instance_of = f, effect = pure)]
    fn f_int<Rt>(rt: &Rt, c: Ref<Vec<i64>, Shared, Rt>, other: Ref<Vec<i64>, Shared, Rt>) -> bool
    where
        Rt: Runtime,
    {
        let _ = (rt, c, other);
        unreachable!("a type-only fixture is never run")
    }

    #[extern_fn(instance_of = max, effect = pure)]
    fn max_int<Rt>(rt: &Rt, c: Ref<Vec<i64>, Shared, Rt>) -> Option<i64>
    where
        Rt: Runtime,
    {
        let _ = (rt, c);
        unreachable!("a type-only fixture is never run")
    }

    pub fn registry() -> Registry<TypesOnly> {
        extern_registry! {
            ns: "fx_q",
            signatures: [f, max],
            fns: [f_int, max_int],
        }
    }
}

// -- Harness ----------------------------------------------------------------

fn script_fn(i: &Interner, source: &str) -> Function {
    let mut pb = PolyBuilder::new();
    Function {
        qref: QualifiedRef::root(i.intern("script")),
        kind: FnKind::Local(ParsedAst::Script(
            acvus_ast::parse_script(i, source).expect("parse"),
        )),
        ty: TyTerm::Fn {
            params: vec![],
            ret: Box::new(pb.fresh_ty_var()),
            captures: vec![],
            effect: pb.fresh_effect_var(),
        },
    }
}

struct Checked {
    ret: Ty,
    callees: Vec<String>,
}

fn check_functions(
    i: &Interner,
    reg: TypeRegistry,
    mut functions: Vec<Function>,
    source: &str,
) -> Result<Checked, Vec<String>> {
    let script = script_fn(i, source);
    let qref = script.qref;
    functions.push(script);
    let graph = CompilationGraph {
        functions: Freeze::new(functions),
        contexts: Freeze::new(vec![]),
        entry: None,
    };
    let ext = extract::extract(i, &graph);
    let inf = infer::infer(i, &graph, &ext, &FxHashMap::default(), Freeze::new(reg));
    if inf.has_errors() {
        return Err(inf
            .errors()
            .into_iter()
            .flat_map(|(_, errs)| errs.iter().map(|e| e.display(i).to_string()))
            .collect());
    }
    let outcome = &inf.outcomes[&qref];
    let Ty::Fn { ret, .. } = &outcome.meta().ty else {
        panic!("expected Fn, got {:?}", outcome.meta().ty)
    };
    let resolution = outcome.resolution().expect("complete");
    let mut callees: Vec<String> = resolution
        .direct_calls
        .values()
        .map(|callee| {
            let id = match callee {
                Callee::Extern { id, .. } | Callee::Direct(id) => id,
                Callee::Indirect(_) => unreachable!("direct_calls holds named callees only"),
            };
            let name = i.resolve(id.name);
            match id.namespace {
                Some(ns) => format!("{}::{name}", i.resolve(ns)),
                None => name.to_string(),
            }
        })
        .collect();
    callees.sort();
    Ok(Checked {
        ret: (**ret).clone(),
        callees,
    })
}

fn check(i: &Interner, source: &str) -> Result<Checked, Vec<String>> {
    let mut registries = acvus_ext::std_registries::<TypesOnly>();
    registries.extend([fx_p::registry(), fx_q::registry()]);
    let Externs {
        functions, types, ..
    } = Externs::combine(registries, i).expect("registries combine");
    check_functions(i, types, functions, source)
}

fn checked(i: &Interner, source: &str) -> Checked {
    check(i, source).unwrap_or_else(|e| panic!("{e:?}"))
}

fn calls(checked: &Checked, callee: &str) -> usize {
    checked
        .callees
        .iter()
        .filter(|c| c.as_str() == callee)
        .count()
}

// -- 1: a ground receiver picks its kind ------------------------------------

#[test]
fn a_string_receiver_settles_the_plain_f_beside_the_signature() {
    let i = Interner::new();
    let c = checked(&i, r#"let s = "ab".to_string(); s.f("a")"#);
    assert_eq!(c.ret, Ty::Bool);
    assert_eq!(calls(&c, "fx_p::f"), 1, "{:?}", c.callees);
    assert_eq!(calls(&c, "fx_q::f"), 0, "{:?}", c.callees);
}

#[test]
fn a_vec_receiver_settles_the_signature_f_beside_the_plain_one() {
    let i = Interner::new();
    let c = checked(&i, "let v = vec([1, 2]); let w = vec([1]); v.f(&w)");
    assert_eq!(c.ret, Ty::Bool);
    assert_eq!(calls(&c, "fx_q::f"), 1, "{:?}", c.callees);
    assert_eq!(calls(&c, "fx_p::f"), 0, "{:?}", c.callees);
}

// -- 2: a receiver the call site grounds ------------------------------------

#[test]
fn a_closure_parameter_the_call_site_makes_a_string_settles_the_plain_f() {
    let i = Interner::new();
    let c = checked(&i, r#"let g = |t| -> t.f("a"); g("ab".to_string())"#);
    assert_eq!(c.ret, Ty::Bool);
    assert_eq!(calls(&c, "fx_p::f"), 1, "{:?}", c.callees);
    assert_eq!(calls(&c, "fx_q::f"), 0, "{:?}", c.callees);
}

#[test]
fn a_closure_parameter_the_call_site_makes_a_vec_settles_the_signature_f() {
    let i = Interner::new();
    let c = checked(
        &i,
        "let g = |c, d| -> c.f(d); let v = vec([1, 2]); let w = vec([1]); g(&v, &w)",
    );
    assert_eq!(c.ret, Ty::Bool);
    assert_eq!(calls(&c, "fx_q::f"), 1, "{:?}", c.callees);
    assert_eq!(calls(&c, "fx_p::f"), 0, "{:?}", c.callees);
}

/// The origin, one variable apart. The set empties to `fx_p::f` at the
/// second argument, whose `&str` no `&Vec<T>` takes, while the receiver is
/// still the closure's own variable. The extra `String` call is what orders
/// the solve so that the settle happens first; before this file's fix it
/// bound the parameter to `str` and the call site's `String` was reported
/// as "expected str, got String".
#[test]
fn the_settle_leaves_an_open_receiver_for_the_call_site_to_ground() {
    let i = Interner::new();
    let c = checked(
        &i,
        r#"let g = |t| -> t.f("a");
           let s = "x".to_string();
           let n = s.len();
           g("ab".to_string())"#,
    );
    assert_eq!(c.ret, Ty::Bool);
    assert_eq!(calls(&c, "fx_p::f"), 1, "{:?}", c.callees);
    assert_eq!(calls(&c, "string::len"), 1, "{:?}", c.callees);
}

// -- 3: the real registries -------------------------------------------------

#[test]
fn contains_of_a_lent_vec_settles_vec_contains_beside_iter_contains() {
    let i = Interner::new();
    let c = checked(&i, "let v = vec([1, 2]); let x = 2; contains(&v, &x)");
    assert_eq!(c.ret, Ty::Bool);
    assert_eq!(calls(&c, "vec::contains"), 1, "{:?}", c.callees);
    assert_eq!(calls(&c, "iter::contains"), 0, "{:?}", c.callees);
}

#[test]
fn max_of_an_iterator_settles_iter_max_beside_a_vec_max_signature() {
    let i = Interner::new();
    let c = checked(&i, "into_iter(vec([3, 1, 2])) | max");
    assert_eq!(c.ret, Ty::Option(Box::new(Ty::I64)));
    assert_eq!(calls(&c, "iter::max"), 1, "{:?}", c.callees);
    assert_eq!(calls(&c, "fx_q::max"), 0, "{:?}", c.callees);
}

#[test]
fn max_of_a_lent_vec_settles_the_vec_max_signature() {
    let i = Interner::new();
    let c = checked(&i, "let v = vec([3, 1, 2]); max(&v)");
    assert_eq!(c.ret, Ty::Option(Box::new(Ty::I64)));
    assert_eq!(calls(&c, "fx_q::max"), 1, "{:?}", c.callees);
    assert_eq!(calls(&c, "iter::max"), 0, "{:?}", c.callees);
}

// -- 4: a closure argument, measured ----------------------------------------

/// `fx_p::map` has `option::map`'s shape — `effect = E` over a `Closure`,
/// with a `_now` twin — beside `iter::map`. The option batch withdrew
/// `option::map` on the finding that a closure argument is typed from its
/// body before the call's candidates pin its parameter; this measures the
/// same question on this tree, where it does not happen.
#[test]
fn an_option_shaped_map_settles_beside_iter_map() {
    let i = Interner::new();
    for (source, callee) in [
        (
            "into_iter([1, 2]) | map(|s| -> s + 1) | count()",
            "iter::map",
        ),
        ("let o = Some(1); map(o, |s| -> s + 1)", "fx_p::map"),
        ("let o = Some(1); o.map(|s| -> s + 1)", "fx_p::map"),
        (
            "into_iter([{ score: 1, }]) | map(|s| -> s.score) | count()",
            "iter::map",
        ),
        (
            "let o = Some({ score: 1, }); map(o, |s| -> s.score)",
            "fx_p::map",
        ),
    ] {
        let c = checked(&i, source);
        assert_eq!(calls(&c, callee), 1, "{source}: {:?}", c.callees);
    }
}
