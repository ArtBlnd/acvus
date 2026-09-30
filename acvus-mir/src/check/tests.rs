//! The judgments over programs built from source: parsed, checked by
//! typeck, lowered, and taken through the required steps up to
//! `clone_strings`, as the checks see them.

mod borrowck;
mod initialization;
mod matchck;
mod refill;

use acvus_ast::{Script, Span};
use acvus_utils::{Astr, Interner, QualifiedRef};

use crate::check::{Refusal, check};
use crate::decl::{ContextDecl, Core, Declarations, DeclaredFacts, LabelledSignature, RawDeclarations, Scheme, TyVarBound, UserTypeDecl};
use crate::lower::{lower_script, required};
use crate::passes::analysis::AnalysisManager;
use crate::ty::{Mark, MarkTerm};
use crate::ty::{Calls, Effect, EffectTerm, FlowTerm, Holds, IntTy, ObjectTy, Poly, TyId, TyKind, Types};
use crate::typeck::check_script;

pub(super) struct Fixture {
    names: Interner,
    raw: RawDeclarations,
    /// Each input `$x` with its type, at which a run's inputs are fixed.
    inputs: Vec<(Astr, TyId<Poly>)>,
}

impl Fixture {
    pub(super) fn new() -> Self {
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
                loop_markers: Default::default(),
                core,
            },
            inputs: Vec::new(),
        }
    }

    pub(super) fn name(&self, name: &str) -> Astr {
        self.names.intern(name)
    }

    pub(super) fn ty(&mut self, kind: TyKind<Poly>) -> TyId<Poly> {
        self.raw.types.intern(kind)
    }

    pub(super) fn int(&mut self) -> TyId<Poly> {
        self.ty(TyKind::Int(IntTy::I64))
    }

    pub(super) fn object(&mut self, fields: &[(&str, TyId<Poly>)]) -> TyId<Poly> {
        let fields = fields.iter().map(|(name, ty)| (self.name(name), *ty)).collect();
        self.ty(TyKind::Object(ObjectTy::Structural(fields, ())))
    }

    /// A pure extern fn `name` of these parameters and result.
    pub(super) fn declare(&mut self, name: &str, params: &[TyId<Poly>], ret: TyId<Poly>) -> QualifiedRef {
        self.declare_with_effect(name, params, ret, Effect::pure())
    }

    /// An extern fn `name` of these parameters, result and effect.
    pub(super) fn declare_with_effect(&mut self, name: &str, params: &[TyId<Poly>], ret: TyId<Poly>, effect: Effect) -> QualifiedRef {
        let ty = self.ty(TyKind::Fn {
            params: params.into(),
            ret,
            effect: EffectTerm::Known(effect),
            flows: FlowTerm::Known(crate::decl::declared_fn_flows()),
            calls: Calls::Many,
            holds: Holds::Nothing,
            values: (),
        });
        let id = QualifiedRef::root(self.name(name));
        let param_names = (0..params.len()).map(|at| self.name(&format!("p{at}"))).collect();
        let scheme = Scheme { ty, param_names, bounds: Box::new([]), impls: None, requires: Box::new([]), facts: DeclaredFacts::default(), runs: Box::new([]), specialized: Box::new([]) };
        self.raw.functions.push((id, scheme));
        id
    }

    /// A pure extern fn `name` of these parameters and result, stating
    /// `lifetimes` (extern-fn §4).
    pub(super) fn declare_labelled(&mut self, name: &str, params: &[TyId<Poly>], ret: TyId<Poly>, lifetimes: LabelledSignature) -> QualifiedRef {
        let id = self.declare(name, params, ret);
        let (_, scheme) = self.raw.functions.last_mut().expect("the declaration just pushed");
        scheme.facts.lifetimes = Some(lifetimes);
        id
    }

    /// A context `@name` of this type, not volatile.
    pub(super) fn context(&mut self, name: &str, ty: TyId<Poly>) {
        let id = QualifiedRef::root(self.name(name));
        self.raw.contexts.push((id, ContextDecl { ty, volatile: false }));
    }

    pub(super) fn input(&mut self, name: &str, ty: TyId<Poly>) {
        let name = self.name(name);
        self.inputs.push((name, ty));
    }

    /// An extern `Vec<i64>`, which does not copy (ownership §1).
    pub(super) fn vec_of_ints(&mut self) -> TyId<Poly> {
        let id = QualifiedRef::root(self.name("Vec"));
        self.raw.user_types.push(UserTypeDecl {
            id,
            type_params: Box::new([TyVarBound::Any]),
            carries_effect: false,
            carries_identity: false,
            lifetimes: 0,
        });
        let int = self.int();
        self.ty(TyKind::UserDefined {
            id,
            type_args: Box::new([int]),
            effect: None,
            identity: None,
            marked: MarkTerm::Known(Mark::Unmarked),
        })
    }

    /// `make() -> Vec<i64>` and `consume(Vec<i64>) -> i64`; the type
    /// `Vec<i64>`.
    pub(super) fn vecs(&mut self) -> TyId<Poly> {
        let (vec, int) = (self.vec_of_ints(), self.int());
        self.declare("make", &[], vec);
        self.declare("consume", &[vec], int);
        vec
    }

    /// The refusals of `check` over `source`, which typeck admits.
    pub(super) fn refusals(self, source: &str) -> Vec<Refusal> {
        let script: Script = acvus_ast::parse_script(&self.names, source).expect("the test's source parses");
        let decls = Declarations::verify(self.raw).expect("the test's declarations verify");
        let mut types = Types::new();
        let inputs = crate::typeck::fixed_inputs(&decls, &mut types, &self.inputs);
        let typeck = check_script(&self.names, &decls, &crate::typeck::Interfaces::new(), crate::typeck::Inputs::Fixed(&inputs), &mut types, &script);
        assert!(typeck.refusals.is_empty(), "typeck refused: {:?}", typeck.refusals);
        let mut module = lower_script(&decls, &typeck, types, &script);
        let mut am = AnalysisManager::new(&decls);
        required::clone_strings(&mut module, &mut am);
        check(&module, &mut am)
    }
}

/// The span of the `nth` occurrence of `text` in `source`, from 0.
pub(super) fn span_of(source: &str, text: &str, nth: usize) -> Span {
    let start = source.match_indices(text).nth(nth).unwrap_or_else(|| panic!("{text:?} occurs {nth} times before in {source:?}")).0;
    Span { start, end: start + text.len() }
}

pub(super) fn covers(outer: Span, inner: Span) -> bool {
    outer.start <= inner.start && inner.end <= outer.end
}



