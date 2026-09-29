//! matchck over programs parsed, checked by typeck and lowered, each held
//! to patterns §2-4 and types §2.

use acvus_ast::Script;
use acvus_utils::{Astr, Interner, QualifiedRef};

use crate::check::{RefusalKind, check};
use crate::decl::{Core, Declarations, DeclaredFacts, LoopMarkers, RawDeclarations, Scheme};
use crate::lower::lower_script;
use crate::passes::analysis::AnalysisManager;
use crate::ty::{Calls, Effect, EffectTerm, FlowTerm, Flows, Holds, IntTy, Poly, TyId, TyKind, Types};
use crate::typeck::check_script;

struct Fixture {
    names: Interner,
    raw: RawDeclarations,
    /// Each input `$x` with its type, at which a run's inputs are fixed.
    inputs: Vec<(Astr, TyId<Poly>)>,
}

impl Fixture {
    fn new() -> Self {
        let names = Interner::new();
        let core = Core::of(&names);
        Fixture {
            names,
            raw: RawDeclarations {
                types: Types::new(),
                functions: Vec::new(),
                lang_impls: Vec::new(),
                user_types: Vec::new(),
                casts: Vec::new(),
                conversions: Vec::new(),
                contexts: Vec::new(),
                entries: Vec::new(),
                loop_markers: LoopMarkers::default(),
                core,
            },
            inputs: Vec::new(),
        }
    }

    fn ty(&mut self, kind: TyKind<Poly>) -> TyId<Poly> {
        self.raw.types.intern(kind)
    }

    fn int(&mut self) -> TyId<Poly> {
        self.ty(TyKind::Int(IntTy::I64))
    }

    fn input(&mut self, name: &str, ty: TyId<Poly>) {
        let name = self.names.intern(name);
        self.inputs.push((name, ty));
    }

    /// A pure extern fn `name() -> ret`.
    fn nullary(&mut self, name: &str, ret: TyId<Poly>) {
        let ty = self.ty(TyKind::Fn {
            params: Box::new([]),
            ret,
            effect: EffectTerm::Known(Effect::pure()),
            flows: FlowTerm::Known(Flows::none()),
            calls: Calls::Many,
            holds: Holds::Nothing,
            values: (),
        });
        let id = QualifiedRef::root(self.names.intern(name));
        let scheme =
            Scheme { ty, param_names: Box::new([]), bounds: Box::new([]), impls: None, requires: Box::new([]), facts: DeclaredFacts::default() };
        self.raw.functions.push((id, scheme));
    }

    /// The refusals `check` makes over `source`, each with the text of the
    /// source at its span.
    fn refusals(self, source: &str) -> Vec<(RefusalKind, String)> {
        let script: Script = acvus_ast::parse_script(&self.names, source).expect("the test's source parses");
        let decls = Declarations::verify(self.raw).expect("the test's declarations verify");
        let mut types = Types::new();
        let inputs = crate::typeck::fixed_inputs(&decls, &mut types, &self.inputs);
        let typeck = check_script(&self.names, &decls, &crate::typeck::Interfaces::new(), crate::typeck::Inputs::Fixed(&inputs), &mut types, &script);
        assert!(typeck.refusals.is_empty(), "typeck refused {source}: {:?}", typeck.refusals);
        let module = lower_script(&decls, &typeck, types, &script);
        let mut am = AnalysisManager::new(&decls);
        check(&module, &mut am)
            .into_iter()
            .map(|refusal| (refusal.kind, source[refusal.span.start..refusal.span.end].to_string()))
            .collect()
    }
}

fn admitted(fx: Fixture, source: &str) {
    let refusals = fx.refusals(source);
    assert!(refusals.is_empty(), "{source}: {refusals:?}");
}

/// Refused once, `NotExhaustive`, at the `match`.
fn refused_at_the_match(fx: Fixture, source: &str) {
    let refusals = fx.refusals(source);
    let [(RefusalKind::NotExhaustive, at)] = &refusals[..] else {
        panic!("{source}: one NotExhaustive, not {refusals:?}")
    };
    assert!(at.starts_with("match"), "{source}: refused at `{at}`, not at its match");
}

/// patterns §2, §3: a program's enum made in the body is transparent, so
/// its type holds every variant it is made with and matched by, and a
/// `match` naming them all covers it.
#[test]
fn a_local_enum_covered_by_its_variants_is_admitted() {
    admitted(Fixture::new(), "let c = Color::Red; match c { Color::Red => 1, Color::Blue => 2, }");
}

/// patterns §3: the same enum, the variant it is made with left unnamed
/// and no `_`.
#[test]
fn a_local_enum_missing_a_variant_is_refused() {
    refused_at_the_match(Fixture::new(), "let c = Color::Red; match c { Color::Blue => 1, }");
}

/// types §2: `Option` is closed like an extern enum, so an `Option` from an
/// extern call is covered by `Some` and `None`.
#[test]
fn an_option_from_an_extern_call_covered_by_some_and_none_is_admitted() {
    let mut fx = Fixture::new();
    let int = fx.int();
    let option = fx.ty(TyKind::Option(int, ()));
    fx.nullary("get", option);
    admitted(fx, "match get() { Some(x) => x, None => 0, }");
}

/// patterns §3: `true` and `false` are every `bool`.
#[test]
fn a_bool_covered_by_true_and_false_is_admitted() {
    let mut fx = Fixture::new();
    let bool_ty = fx.ty(TyKind::Bool);
    fx.input("b", bool_ty);
    admitted(fx, "match $b { true => 1, false => 0, }");
}

/// patterns §3: an integer's keys leave values no arm answers.
#[test]
fn an_int_match_without_a_catch_all_is_refused() {
    let mut fx = Fixture::new();
    let int = fx.int();
    fx.input("n", int);
    refused_at_the_match(fx, "match $n { 0 => 1, 1 => 2, }");
}

/// The known limit: a test's false edge does not narrow what it tested, so
/// a nested pattern's arms are each tested against every variant, and even
/// a `match` whose arms cover every value is refused without `_`.
#[test]
fn a_nested_pattern_without_a_catch_all_is_refused() {
    let mut fx = Fixture::new();
    let int = fx.int();
    let option = fx.ty(TyKind::Option(int, ()));
    fx.input("o", option);
    refused_at_the_match(fx, "match $o { Some(0) => 1, Some(x) => x, None => 3, }");
}

/// patterns §4: a scrutinee of type `!` reaches no arm, and no
/// exhaustiveness question arises.
#[test]
fn a_scrutinee_of_type_never_is_not_refused() {
    let mut fx = Fixture::new();
    let never = fx.ty(TyKind::Never);
    fx.nullary("abort", never);
    admitted(fx, "match abort() { 0 => 1, }");
}

/// patterns §3: `_` answers whatever the other arms leave.
#[test]
fn a_match_with_a_catch_all_is_admitted() {
    for source in ["match $n { 0 => 1, _ => 2, }", "match $o { Some(0) => 1, _ => 2, }"] {
        let mut fx = Fixture::new();
        let int = fx.int();
        let option = fx.ty(TyKind::Option(int, ()));
        fx.input("n", int);
        fx.input("o", option);
        admitted(fx, source);
    }
}
