//! A declared call whose result holds a part of a type variable that a
//! function value it is given, or an impl it requires, makes from another
//! part (extern-fn §4, §6, ownership §5): the result holds the loans that
//! part is made from, as the same call does in Rust, where the instantiated
//! types carry the lifetimes.

use acvus_utils::QualifiedRef;

use super::borrowck::one_conflict;
use super::{Fixture, covers, span_of};
use crate::decl::{
    DeclaredFacts, EndLabels, GenericFunction, ImplKind, ImplOf, LabelledSignature, Lifetime, Requirement, Scheme, SegmentLabel,
    TyVarBound, UserTypeDecl, declared_fn_flows,
};
use crate::ty::{Calls, Effect, EffectTerm, FlowEnd, FlowTerm, Holds, Mark, MarkTerm, Mutability, Poly, TyId, TyKind};

const T: SegmentLabel = SegmentLabel::TypeVar(0);
const U: SegmentLabel = SegmentLabel::TypeVar(1);
const L0: SegmentLabel = SegmentLabel::Lifetime(Lifetime(0));
const L1: SegmentLabel = SegmentLabel::Lifetime(Lifetime(1));

fn signature(ends: &[(FlowEnd, &[SegmentLabel])]) -> LabelledSignature {
    let ends = ends.iter().map(|(end, labels)| EndLabels { end: *end, labels: (*labels).into() }).collect();
    LabelledSignature { ends, outlives: Box::new([]) }
}

/// A pure function type of these parameters and result, of every flow.
fn function(fx: &mut Fixture, params: &[TyId<Poly>], ret: TyId<Poly>, calls: Calls, holds: Holds) -> TyId<Poly> {
    fx.ty(TyKind::Fn {
        params: params.into(),
        ret,
        effect: EffectTerm::Known(Effect::pure()),
        flows: FlowTerm::Known(declared_fn_flows()),
        calls,
        holds,
        values: (),
    })
}

/// The declaration `ty` of `variables` type variables, any type each.
fn scheme(fx: &Fixture, ty: TyId<Poly>, variables: usize, lifetimes: Option<LabelledSignature>) -> Scheme {
    let TyKind::Fn { params, .. } = fx.raw.types.kind(ty) else { panic!("a declaration's type is a function type") };
    let param_names = (0..params.len()).map(|at| fx.name(&format!("p{at}"))).collect();
    let facts = DeclaredFacts { lifetimes, ..DeclaredFacts::default() };
    Scheme {
        ty,
        param_names,
        bounds: vec![TyVarBound::Any; variables].into(),
        impl_of: None,
        requires: Box::new([]),
        facts,
        runs: Box::new([]),
        specialized: Box::new([]),
        awaits: false,
    }
}

/// The one `LoanConflict` of `source`'s refusals lies within `within`.
fn refused_within(fx: Fixture, source: &str, within: &str) {
    let refusals = fx.refusals(source);
    let (span, _) = one_conflict(&refusals);
    assert!(covers(span_of(source, within, 0), span), "within {within:?}: {span:?}");
}

/// `map_opt<T, U>(x: Option<T>, f: fn(T) -> U) -> Option<U>`, `f` holding
/// shared loans: with `labelled`, `map_opt<'f, T, U>`, `f` of `'f` and each
/// type variable's part its own, else stating no lifetimes.
fn declare_map_opt(fx: &mut Fixture, labelled: bool) {
    let (t, u) = (fx.ty(TyKind::Var(0)), fx.ty(TyKind::Var(1)));
    let (option_t, option_u) = (fx.ty(TyKind::Option(t, ())), fx.ty(TyKind::Option(u, ())));
    let f = function(fx, &[t], u, Calls::Once, Holds::Shared);
    let ty = function(fx, &[option_t, f], option_u, Calls::Many, Holds::Nothing);
    let lifetimes = labelled.then(|| signature(&[(FlowEnd::Result, &[U]), (FlowEnd::Param(0), &[T]), (FlowEnd::Param(1), &[L0])]));
    let declaration = scheme(fx, ty, 2, lifetimes);
    fx.raw.functions.push((QualifiedRef::root(fx.name("map_opt")), declaration));
}

/// `id_opt<T>(x: Option<T>) -> Option<T>`, its type variable's part its own.
fn declare_id_opt(fx: &mut Fixture) {
    let t = fx.ty(TyKind::Var(0));
    let option_t = fx.ty(TyKind::Option(t, ()));
    let ty = function(fx, &[option_t], option_t, Calls::Many, Holds::Nothing);
    let declaration = scheme(fx, ty, 1, Some(signature(&[(FlowEnd::Result, &[T]), (FlowEnd::Param(0), &[T])])));
    fx.raw.functions.push((QualifiedRef::root(fx.name("id_opt")), declaration));
}

const READ_R: &str = "match r { Some(p) => *p, None => 0, }";

/// The callback gives back the argument's `&b`, which `r` then holds.
#[test]
fn a_write_meets_a_loan_a_callback_carries_from_an_argument_into_a_labelled_result() {
    let mut fx = Fixture::new();
    declare_map_opt(&mut fx, true);
    refused_within(fx, &format!("let b = 2; let r = map_opt(Some(&b), |x| -> x); b = 3; {READ_R}"), "b = 3;");
}

/// The callback gives back the `&b` it captured, which `r` then holds.
#[test]
fn a_write_meets_a_loan_a_callback_captures_into_a_labelled_result() {
    let mut fx = Fixture::new();
    declare_map_opt(&mut fx, true);
    refused_within(fx, &format!("let b = 2; let r = map_opt(Some(1), |x| -> &b); b = 3; {READ_R}"), "b = 3;");
}

/// extern-fn §4: stating no lifetimes, `map_opt` has every flow.
#[test]
fn a_write_meets_a_loan_a_callback_carries_into_an_unlabelled_result() {
    let mut fx = Fixture::new();
    declare_map_opt(&mut fx, false);
    refused_within(fx, &format!("let b = 2; let r = map_opt(Some(&b), |x| -> x); b = 3; {READ_R}"), "b = 3;");
}

/// A type variable's part carried from an argument to the result with
/// nothing between them.
#[test]
fn a_write_meets_a_loan_a_labelled_type_variable_carries_into_the_result() {
    let mut fx = Fixture::new();
    declare_id_opt(&mut fx);
    refused_within(fx, &format!("let b = 2; let r = id_opt(Some(&b)); b = 3; {READ_R}"), "b = 3;");
}

/// An extern `Src<'a>`; `src_of<'a>(a: &'a i64) -> Src<'a>`; the generic
/// `next<It, T>(it: &mut It) -> Option<T>` and its impl
/// `next<'a, 'b>(it: &'b mut Src<'a>) -> Option<&'a i64>`; and
/// `first_of<It, T>(src: It) -> Option<T>` requiring `next` at `(It, T)`:
/// with `labelled`, each type variable's part its own, else stating no
/// lifetimes.
fn declare_first_of(fx: &mut Fixture, labelled: bool) {
    let src_id = QualifiedRef::root(fx.name("Src"));
    fx.raw.user_types.push(UserTypeDecl { id: src_id, type_params: Box::new([]), carries_effect: false, carries_identity: false, lifetimes: 1 });
    let src = fx.ty(TyKind::UserDefined { id: src_id, type_args: Box::new([]), effect: None, identity: None, marked: MarkTerm::Known(Mark::Unmarked) });
    let int = fx.int();
    let shared_int = fx.ty(TyKind::Ref(Mutability::Shared, int));
    let src_of = function(fx, &[shared_int], src, Calls::Many, Holds::Nothing);
    let src_of = scheme(fx, src_of, 0, Some(signature(&[(FlowEnd::Result, &[L0]), (FlowEnd::Param(0), &[L0])])));
    fx.raw.functions.push((QualifiedRef::root(fx.name("src_of")), src_of));

    let (it, t) = (fx.ty(TyKind::Var(0)), fx.ty(TyKind::Var(1)));
    let (mut_it, option_t) = (fx.ty(TyKind::Ref(Mutability::Mut, it)), fx.ty(TyKind::Option(t, ())));
    let next_ty = function(fx, &[mut_it], option_t, Calls::Many, Holds::Nothing);
    let next = QualifiedRef::root(fx.name("next"));
    let generic = GenericFunction { ty: next_ty, param_names: Box::new([fx.name("it")]), bounds: Box::new([TyVarBound::Any, TyVarBound::Any]), specialized: Box::new([]) };
    fx.raw.generic_functions.push((next, generic));

    let (mut_src, option_int_ref) = (fx.ty(TyKind::Ref(Mutability::Mut, src)), fx.ty(TyKind::Option(shared_int, ())));
    let next_at_src = function(fx, &[mut_src], option_int_ref, Calls::Many, Holds::Nothing);
    let mut next_at_src = scheme(fx, next_at_src, 0, Some(signature(&[(FlowEnd::Result, &[L0]), (FlowEnd::Param(0), &[L1, L0])])));
    next_at_src.impl_of = Some(ImplOf { generic_function: next, kind: ImplKind::Concrete });
    fx.raw.functions.push((QualifiedRef::root(fx.name("next#0")), next_at_src));

    let first_of = function(fx, &[it], option_t, Calls::Many, Holds::Nothing);
    let lifetimes = labelled.then(|| signature(&[(FlowEnd::Result, &[U]), (FlowEnd::Param(0), &[T])]));
    let mut first_of = scheme(fx, first_of, 2, lifetimes);
    first_of.requires = Box::new([Requirement { generic_function: next, pattern: next_ty, fact: None }]);
    fx.raw.functions.push((QualifiedRef::root(fx.name("first_of")), first_of));
}

/// `next` at `Src<'a>` gives back a `&'a i64` from the source `first_of`
/// is given, which `r` then holds.
#[test]
fn a_write_meets_a_loan_a_required_impl_carries_into_a_labelled_result() {
    let mut fx = Fixture::new();
    declare_first_of(&mut fx, true);
    refused_within(fx, &format!("let a = 1; let r = first_of(src_of(&a)); a = 3; {READ_R}"), "a = 3;");
}

/// extern-fn §4: stating no lifetimes, `first_of` has every flow.
#[test]
fn a_write_meets_a_loan_a_required_impl_carries_into_an_unlabelled_result() {
    let mut fx = Fixture::new();
    declare_first_of(&mut fx, false);
    refused_within(fx, &format!("let a = 1; let r = first_of(src_of(&a)); a = 3; {READ_R}"), "a = 3;");
}

/// The impl called by name: its own lifetimes give its result the loan.
#[test]
fn a_write_meets_a_loan_the_impl_called_directly_carries_into_its_result() {
    let mut fx = Fixture::new();
    declare_first_of(&mut fx, true);
    refused_within(fx, &format!("let a = 1; let s = src_of(&a); let r = next(&mut s); a = 3; {READ_R}"), "a = 3;");
}
