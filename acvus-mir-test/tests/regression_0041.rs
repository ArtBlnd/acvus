//! Regression tests for RFC-0041/0042 at the checker's contract (R7–R15):
//! the instance a call settles, the casts the IR carries, and the type a
//! script returns, with the standard registries beside a `Monomorphize`
//! member registry. A test that fails is a finding, kept as it fails.

use acvus_extern::{Externs, Monomorphize, TypesOnly, extern_fn, extern_registry};
use acvus_mir::graph::{
    CompilationGraph, FnKind, Function, ParsedAst, QualifiedRef, extract, infer,
};
use acvus_mir::ir::{Callee, CastKind};
use acvus_mir::ty::{PolyBuilder, Ty, TyTerm, TypeArg, TypeRegistry};
use acvus_utils::{Freeze, Interner};
use rustc_hash::FxHashMap;

// -- The member functions -------------------------------------------------

trait Float: Copy + Send + Sync + 'static {
    const ZERO: Self;
    fn mul_add(self, a: Self, b: Self) -> Self;
}

impl Float for f64 {
    const ZERO: Self = 0.0;

    fn mul_add(self, a: Self, b: Self) -> Self {
        self + a * b
    }
}

#[extern_fn(effect = pure)]
fn dot<T>(a: Vec<T>, b: Vec<T>) -> T
where
    T: Monomorphize<(f64,)> + Float,
{
    a.iter()
        .zip(&b)
        .fold(T::ZERO, |acc, (x, y)| acc.mul_add(*x, *y))
}

#[extern_fn(effect = pure)]
fn zeros<T>(n: i64) -> Vec<T>
where
    T: Monomorphize<(f64,)> + Float,
{
    vec![T::ZERO; usize::try_from(n).expect("a count is not negative")]
}

#[extern_fn(effect = pure)]
fn norm<T>(v: &Vec<T>) -> T
where
    T: Monomorphize<(f64,)> + Float,
{
    v.iter().fold(T::ZERO, |acc, x| acc.mul_add(*x, *x))
}

#[extern_fn(effect = pure)]
fn norm2<T>(a: &Vec<T>, b: &Vec<T>) -> T
where
    T: Monomorphize<(f64,)> + Float,
{
    a.iter()
        .zip(b.iter())
        .fold(T::ZERO, |acc, (x, y)| acc.mul_add(*x, *y))
}

/// `T` has no bound but the member list, so the macro emits the generic
/// instance beside `#f64`.
#[extern_fn(effect = pure)]
fn ident<T>(v: Vec<T>) -> Vec<T>
where
    T: Monomorphize<(f64,)>,
{
    v
}

const IDENT_AT_F64: usize = 0;
const IDENT_GENERIC: usize = 1;

/// A reference parameter beside a value parameter of the element type, so
/// a call can lend a place and nest a call that lends it again.
#[extern_fn(effect = pure)]
fn scale<T>(v: &Vec<T>, k: T) -> T
where
    T: Monomorphize<(f64,)> + Float,
{
    v.iter().fold(T::ZERO, |acc, x| acc.mul_add(*x, k))
}

/// Two reference parameters that demand two representations of one
/// element type: a member slot beside a plain concrete one.
#[extern_fn(effect = pure)]
fn mixed<T>(a: &Vec<T>, b: &Vec<f64>) -> T
where
    T: Monomorphize<(f64,)> + Float,
{
    a.iter()
        .zip(b.iter())
        .fold(T::ZERO, |acc, (x, _)| acc.mul_add(*x, *x))
}

fn member_registry() -> acvus_extern::Registry<TypesOnly> {
    extern_registry! {
        ns: "t",
        fns: [dot, zeros, norm, norm2, ident, scale, mixed],
    }
}

// -- Harness ----------------------------------------------------------------

fn vec_of(i: &Interner, arg: TypeArg<acvus_mir::ty::Concrete>) -> Ty {
    TyTerm::UserDefined {
        id: QualifiedRef::root(i.intern("Vec")),
        type_args: vec![arg],
        effect_args: vec![],
        identity_args: vec![],
    }
}

fn specialized_vec(i: &Interner, elem: Ty) -> Ty {
    vec_of(i, TypeArg::specialized(elem))
}

fn uniform_vec(i: &Interner, elem: Ty) -> Ty {
    vec_of(i, TypeArg::uniform(elem))
}

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

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct SettledCall {
    callee: String,
    instance: usize,
}

struct Checked {
    ret: Ty,
    calls: Vec<SettledCall>,
    /// The conversions settled on a cast, by the cast's name; a cast
    /// through a reference is `&cast/back`.
    casts: Vec<String>,
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
    let mut casts: Vec<String> = resolution
        .coercion_map
        .iter()
        .map(|(_, kind)| match kind {
            CastKind::Extern(cast) => i.resolve(cast.fn_ref.name).to_string(),
            CastKind::ThroughRef { cast, back, .. } => format!(
                "&{}/{}",
                i.resolve(cast.fn_ref.name),
                i.resolve(back.fn_ref.name)
            ),
            CastKind::Slice { as_slice, .. } => {
                format!("[{}]", i.resolve(as_slice.fn_ref.name))
            }
            CastKind::Str { as_str } => format!("str[{}]", i.resolve(as_str.fn_ref.name)),
        })
        .collect();
    casts.sort();
    let mut calls: Vec<SettledCall> = resolution
        .direct_calls
        .values()
        .filter_map(|callee| match callee {
            Callee::Extern { id, instance, .. } => Some(SettledCall {
                callee: i.resolve(id.name).to_string(),
                instance: *instance,
            }),
            _ => None,
        })
        .collect();
    calls.sort();
    Ok(Checked {
        ret: (**ret).clone(),
        calls,
        casts,
    })
}

fn check(i: &Interner, source: &str) -> Result<Checked, Vec<String>> {
    let mut registries = acvus_ext::std_registries::<TypesOnly>();
    registries.push(member_registry());
    let Externs {
        functions, types, ..
    } = Externs::combine(registries, i).expect("registries combine");
    check_functions(i, types, functions, source)
}

fn checked(i: &Interner, source: &str) -> Checked {
    check(i, source).unwrap_or_else(|e| panic!("{e:?}"))
}

fn instances_of(checked: &Checked, name: &str) -> Vec<usize> {
    checked
        .calls
        .iter()
        .filter(|c| c.callee == name)
        .map(|c| c.instance)
        .collect()
}

fn instance_of(checked: &Checked, name: &str) -> usize {
    let instances = instances_of(checked, name);
    let [instance] = instances.as_slice() else {
        panic!("one call to {name}: {:?}", checked.calls)
    };
    *instance
}

const THROUGH_REF: &str = "&materialize/erase";

// -- R7 -----------------------------------------------------------------------

#[test]
fn a_specialized_local_pays_one_erase_per_generic_consumer_and_none_at_its_member() {
    let i = Interner::new();
    let c = checked(&i, "let x = zeros(3); reverse(x); dot(x, x)");
    assert_eq!(c.ret, Ty::Float);
    assert_eq!(instance_of(&c, "zeros"), 0);
    assert_eq!(instance_of(&c, "dot"), 0);
    assert_eq!(instance_of(&c, "reverse"), 0);
    assert_eq!(c.casts, vec!["erase".to_string()]);
}

// -- R8 -----------------------------------------------------------------------

#[test]
fn a_uniform_else_branch_is_materialized_to_a_specialized_then_branch() {
    let i = Interner::new();
    let c = checked(&i, "let x = if true { zeros(3) } else { vec([1.0]) }; x");
    assert_eq!(c.ret, specialized_vec(&i, Ty::Float));
    assert_eq!(instance_of(&c, "zeros"), 0);
    assert_eq!(
        c.casts,
        vec!["materialize".to_string()],
        "the else branch is the conversion site and its target is the then type"
    );
}

#[test]
fn a_specialized_then_branch_stored_to_a_uniform_place_pays_a_cast_at_the_else_and_at_the_store() {
    let i = Interner::new();
    let c = checked(
        &i,
        "let x = vec([1.0]); x = if true { zeros(3) } else { vec([2.0]) }; x",
    );
    assert_eq!(c.ret, uniform_vec(&i, Ty::Float));
    assert_eq!(instance_of(&c, "zeros"), 0);
    assert_eq!(
        c.casts,
        vec!["erase".to_string(), "materialize".to_string()],
        "greedy per site: the else materializes to the then type, then the store erases the whole; RFC-0042 leaves a global choice unbuilt"
    );
}

// -- R9 -----------------------------------------------------------------------

#[test]
fn a_member_with_a_generic_fallback_takes_the_member_for_a_specialized_argument_with_no_cast() {
    let i = Interner::new();
    let c = checked(&i, "ident(zeros(3))");
    assert_eq!(c.ret, specialized_vec(&i, Ty::Float));
    assert_eq!(instance_of(&c, "ident"), IDENT_AT_F64);
    assert!(c.casts.is_empty(), "{:?}", c.casts);
}

#[test]
fn a_member_with_a_generic_fallback_takes_the_generic_for_a_uniform_argument_with_no_cast() {
    let i = Interner::new();
    let c = checked(&i, "ident(vec([1.0]))");
    assert_eq!(c.ret, uniform_vec(&i, Ty::Float));
    assert_eq!(instance_of(&c, "ident"), IDENT_GENERIC);
    assert!(c.casts.is_empty(), "{:?}", c.casts);
}

// -- R10 ----------------------------------------------------------------------

#[test]
fn one_place_lent_twice_to_one_specialized_call_is_cast_in_place_once() {
    let i = Interner::new();
    let c = checked(&i, "let x = vec([1.0, 2.0]); norm2(&x, &x)");
    assert_eq!(c.ret, Ty::Float);
    assert_eq!(instance_of(&c, "norm2"), 0);
    assert_eq!(
        c.casts,
        vec![THROUGH_REF.to_string()],
        "one place is materialized once before the call and erased once after it"
    );
}

// -- R11 ----------------------------------------------------------------------

#[test]
fn a_field_place_lent_to_a_specialized_parameter_is_cast_in_place() {
    let i = Interner::new();
    let c = checked(&i, "let o = { v: vec([3.0, 4.0]), }; norm(&o.v)");
    assert_eq!(c.ret, Ty::Float);
    assert_eq!(instance_of(&c, "norm"), 0);
    assert_eq!(c.casts, vec![THROUGH_REF.to_string()]);
}

// -- R12 ----------------------------------------------------------------------

#[test]
fn a_lend_inside_a_loop_is_one_cast_pair_in_the_ir_at_its_one_site() {
    let i = Interner::new();
    let c = checked(
        &i,
        "let x = vec([1.0, 2.0]); let n = 0; let acc = 0.0; while n < 3 { acc = acc + norm(&x); n = n + 1; } acc",
    );
    assert_eq!(c.ret, Ty::Float);
    assert_eq!(instance_of(&c, "norm"), 0);
    assert_eq!(
        c.casts,
        vec![THROUGH_REF.to_string()],
        "the IR carries one cast pair per site; the loop pays it per iteration at run time"
    );
}

// -- R13 ----------------------------------------------------------------------

#[test]
fn contains_over_an_iter_of_a_uniform_int_vec_checks_as_bool_with_no_cast() {
    let i = Interner::new();
    let c = checked(&i, "into_iter(vec([1, 2, 3])) | contains(2)");
    assert_eq!(c.ret, Ty::Bool);
    assert_eq!(instance_of(&c, "contains"), 0, "i64 is the first member");
    assert!(c.casts.is_empty(), "{:?}", c.casts);
}

// -- R14: a nested call lends a place the outer call holds ---------------------

#[test]
fn a_place_lent_to_a_call_and_again_inside_a_nested_argument_is_cast_in_place_once() {
    let i = Interner::new();
    let c = checked(&i, "let x = vec([1.0, 2.0]); scale(&x, norm(&x))");
    assert_eq!(c.ret, Ty::Float);
    assert_eq!(instance_of(&c, "scale"), 0);
    assert_eq!(instance_of(&c, "norm"), 0);
    assert_eq!(
        c.casts,
        vec![THROUGH_REF.to_string()],
        "the outer call's hold reaches the nested call's lend"
    );
}

// -- R15: one place, two representations demanded by one call -----------------

#[test]
fn one_place_lent_twice_to_a_call_that_demands_two_representations_is_a_type_mismatch() {
    let i = Interner::new();
    let errors = check(&i, "let x = vec([1.0, 2.0]); mixed(&x, &x)")
        .err()
        .expect("the held place cannot be both Vec<#Float> and Vec<Float>");
    assert!(
        errors
            .iter()
            .any(|e| e.contains("Vec<#Float>") && e.contains("Vec<Float>")),
        "{errors:?}"
    );
}
