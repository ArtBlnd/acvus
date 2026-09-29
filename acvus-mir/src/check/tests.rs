//! Each judgment over lowered programs: parsed, checked by typeck, lowered,
//! and taken through the required steps that run before the checks.

use acvus_ast::Script;
use acvus_utils::{Interner, QualifiedRef};

use super::{Refusal, RefusalKind};
use crate::decl::{Core, Declarations, DeclaredFacts, LoopMarkers, RawDeclarations, Scheme};
use crate::ir::cfg::{BodyId, Module};
use crate::lower::lower_script;
use crate::lower::required::{clone_strings, remove_dead_context_ops};
use crate::passes::analysis::AnalysisManager;
use crate::ty::{Calls, Effect, EffectTerm, FlowTerm, Flows, Holds, IntTy, Mutability, Poly, TyId, TyKind, TypeArg, Types};
use crate::typeck::check_script;

struct Fixture {
    names: Interner,
    raw: RawDeclarations,
}

type Judgment = fn(BodyId, &Module, &mut AnalysisManager<'_>, &mut Vec<Refusal>);

impl Fixture {
    fn new() -> Self {
        let names = Interner::new();
        let core = Core::of(&names);
        let raw = RawDeclarations {
            types: Types::new(),
            functions: Vec::new(),
            lang_impls: Vec::new(),
            user_types: Vec::new(),
            casts: Vec::new(),
            conversions: Vec::new(),
            contexts: Vec::new(),
            inputs: Vec::new(),
            loop_markers: LoopMarkers::default(),
            core,
        };
        Fixture { names, raw }
    }

    fn ty(&mut self, kind: TyKind<Poly>) -> TyId<Poly> {
        self.raw.types.intern(kind)
    }

    fn int(&mut self) -> TyId<Poly> {
        self.ty(TyKind::Int(IntTy::I64))
    }

    fn reference(&mut self, mutability: Mutability, target: TyId<Poly>) -> TyId<Poly> {
        self.ty(TyKind::Ref(mutability, TypeArg::Uniform(target)))
    }

    fn input(&mut self, name: &str, ty: TyId<Poly>) {
        let name = self.names.intern(name);
        self.raw.inputs.push((name, ty));
    }

    /// A pure extern fn `name` of type `(params) -> ret`.
    fn declare(&mut self, name: &str, params: &[TyId<Poly>], ret: TyId<Poly>) {
        let ty = self.ty(TyKind::Fn {
            params: params.into(),
            ret,
            effect: EffectTerm::Known(Effect::pure()),
            flows: FlowTerm::Known(Flows::none()),
            calls: Calls::Many,
            holds: Holds::Nothing,
            values: (),
        });
        let param_names = (0..params.len()).map(|at| self.names.intern(&format!("p{at}"))).collect();
        let scheme = Scheme { ty, param_names, bounds: Box::new([]), impls: None, requires: Box::new([]), facts: DeclaredFacts::default() };
        self.raw.functions.push((QualifiedRef::root(self.names.intern(name)), scheme));
    }

    /// `source` admitted by typeck, lowered, taken through the steps before
    /// the checks, and judged by `judgment` over every body.
    fn judged(self, source: &str, judgment: Judgment) -> Vec<RefusalKind> {
        let script: Script = acvus_ast::parse_script(&self.names, source).expect("the test's source parses");
        let decls = Declarations::verify(self.raw).expect("the test's declarations verify");
        let mut types = Types::new();
        let typeck = check_script(&decls, &mut types, &script);
        assert!(typeck.refusals.is_empty(), "{source}: typeck refused {:?}", typeck.refusals);
        let mut module = lower_script(&decls, &typeck, types, &script);
        let mut am = AnalysisManager::new(&decls);
        clone_strings(&mut module, &mut am);
        remove_dead_context_ops(&mut module, &mut am);
        let mut out = Vec::new();
        for body in module.body_ids() {
            judgment(body, &module, &mut am, &mut out);
        }
        out.into_iter().map(|refusal| refusal.kind).collect()
    }
}

/// A fixture with `$c: bool`.
fn with_condition() -> Fixture {
    let mut fx = Fixture::new();
    let bool_ty = fx.ty(TyKind::Bool);
    fx.input("c", bool_ty);
    fx
}

// -- matchck (patterns §3) ---------------------------------------------------

fn matchck(fx: Fixture, source: &str) -> Vec<RefusalKind> {
    fx.judged(source, super::matchck::check)
}

/// patterns §3: a program's enum made in the body has every variant it may
/// hold seen, so arms covering them need no `_`.
#[test]
fn a_local_enum_covered_without_a_wildcard_is_admitted() {
    let source = "let other = Color::Blue; let c = Color::Red; match c { Color::Red => 1, Color::Blue => 2, }";
    assert_eq!(matchck(Fixture::new(), source), vec![]);
}

/// patterns §3: the variants a value made in the body holds are the ones
/// it may hold, not every variant of its type.
#[test]
fn a_local_enum_covered_where_it_may_be_is_admitted() {
    let source = "let other = Color::Blue; let c = Color::Red; match c { Color::Red => 1, }";
    assert_eq!(matchck(Fixture::new(), source), vec![]);
}

/// patterns §3: a variant a local value may hold and no arm answers.
#[test]
fn a_local_value_missing_a_variant_it_may_hold_is_refused() {
    let source = "let v = if $c { Some(1) } else { None }; match v { Some(x) => x, }";
    assert_eq!(matchck(with_condition(), source), vec![RefusalKind::NotExhaustive]);
}

/// patterns §3: a program's enum from a function's argument is not seen,
/// so its `match` needs `_`.
#[test]
fn a_parameters_program_enum_without_a_wildcard_is_refused() {
    let source = "let f = |c| -> match c { Color::Red => 1, Color::Blue => 2, }; f(Color::Red)";
    assert_eq!(matchck(Fixture::new(), source), vec![RefusalKind::NotExhaustive]);
    let source = "let f = |c| -> match c { Color::Red => 1, Color::Blue => 2, _ => 3, }; f(Color::Red)";
    assert_eq!(matchck(Fixture::new(), source), vec![]);
}

/// types §2, patterns §2-3: an `Option` from outside the body has its
/// variants by its type, so `Some` and `None` cover it; one of them alone
/// does not.
#[test]
fn an_extern_option_covered_by_some_and_none_is_admitted() {
    let mut fx = Fixture::new();
    let int = fx.int();
    let option = fx.ty(TyKind::Option(int, ()));
    fx.input("o", option);
    assert_eq!(matchck(fx, "match $o { Some(x) => x, None => 0, }"), vec![]);
    let mut fx = Fixture::new();
    let int = fx.int();
    let option = fx.ty(TyKind::Option(int, ()));
    fx.input("o", option);
    assert_eq!(matchck(fx, "match $o { Some(x) => x, }"), vec![RefusalKind::NotExhaustive]);
}

/// patterns §3: nested patterns cover an `Option` of an `Option` from
/// outside part by part; leaving out `Some(None)` leaves a value no arm
/// answers.
#[test]
fn nested_arms_cover_an_extern_option_part_by_part() {
    let covered = "match $o { Some(Some(x)) => x, Some(None) => 1, None => 0, }";
    let uncovered = "match $o { Some(Some(x)) => x, None => 0, }";
    for (source, expected) in [(covered, vec![]), (uncovered, vec![RefusalKind::NotExhaustive])] {
        let mut fx = Fixture::new();
        let int = fx.int();
        let inner = fx.ty(TyKind::Option(int, ()));
        let option = fx.ty(TyKind::Option(inner, ()));
        fx.input("o", option);
        assert_eq!(matchck(fx, source), expected, "{source}");
    }
}

/// patterns §3: a local value's parts are followed into the arms: a value
/// that is never `Some(None)` needs no arm for it.
#[test]
fn nested_arms_cover_a_local_value_where_it_may_be() {
    let source = "let v = if $c { Some(Some(1)) } else { None }; match v { Some(Some(x)) => x, None => 0, }";
    assert_eq!(matchck(with_condition(), source), vec![]);
}

/// patterns §3: a `bool` has two values, and an integer more than any list
/// of literals names.
#[test]
fn literals_cover_a_bool_and_never_an_integer() {
    assert_eq!(matchck(with_condition(), "match $c { true => 1, false => 0, }"), vec![]);
    let mut fx = Fixture::new();
    let int = fx.int();
    fx.input("n", int);
    assert_eq!(matchck(fx, "match $n { 1 => 1, 2 => 0, }"), vec![RefusalKind::NotExhaustive]);
}
