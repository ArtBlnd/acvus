//! A declaration's scheme at one use (extern-fn §6), and a type written
//! elsewhere, declared or concrete, at one use.

use std::convert::Infallible;

use acvus_utils::QualifiedRef;
use rustc_hash::FxHashMap;

use super::{Bound, FlowId, Infer, Level, Term};
use crate::decl::{Declarations, NamedFact, Scheme, TyVarBound, effect_variables, identity_variables};
use crate::ty::{
    Calls, CallsTerm, Concrete, Effect, EffectTerm, FlowTerm, Flows, FnValues, Holds, HoldsTerm, IdentityId, IdentityTerm,
    Infer as InferPhase, Label, LenTerm, LenVarId, Mark, MarkTerm, Phase, Poly, Ty, TyId, TyKind, Types,
};

/// A requirement at the use's own terms: an impl of `generic_function` whose
/// type joins `pattern`, stating `fact` where named.
#[derive(Debug, Clone, PartialEq)]
pub struct RequiredAt {
    pub generic_function: QualifiedRef,
    pub pattern: Term,
    pub fact: Option<NamedFact>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Instantiated {
    /// The function type at this use.
    pub ty: Term,
    pub requirements: Vec<RequiredAt>,
    /// The flows into the call's effect this use made (extern-fn §2), each
    /// refused where it would raise a fixed effect, as any flow is.
    pub flows: Vec<FlowId>,
    /// The result as declared, where it holds a mark the call's result
    /// starts in a fresh cell of its own (types §8): the call gives its
    /// value at this type, and takes it at the type its uses settle.
    pub declared_result: Option<Term>,
    /// For a declaration with `specialized` types, the one mark its
    /// positions of those types share (types §8).
    pub pair: Option<PairAt>,
}

/// The mark a declaration's positions of its `specialized` types share at
/// one use, unset until those positions' types are known: it starts
/// marked where each is one of `specialized`, unmarked otherwise.
#[derive(Debug, Clone, PartialEq)]
pub struct PairAt {
    pub mark: MarkTerm<InferPhase>,
    pub positions: Vec<Term>,
    pub specialized: Box<[TyId<Poly>]>,
}

/// A variable of a written type: a declaration's positional placeholder,
/// or none, in a concrete type.
pub(super) trait Placeholder: Copy {
    fn index(self) -> u32;
}

impl Placeholder for u32 {
    fn index(self) -> u32 {
        self
    }
}

impl Placeholder for Infallible {
    fn index(self) -> u32 {
        match self {}
    }
}

/// A phase whose types a use instantiates: written elsewhere, so each has
/// its flows, holds and calls fixed and no label.
pub(super) trait Instantiable:
    Phase<
        TyVar: Placeholder,
        EffectVar: Placeholder,
        LenVar: Placeholder,
        IdentityVar: Placeholder,
        FlowVar = Infallible,
        MarkVar = Infallible,
        HoldsTerm = Holds,
        CallsTerm = Calls,
        ValuesTerm = (),
        Label = (),
    >
{
    fn flows(flows: &FlowTerm<Self>) -> &Flows;
}

impl Instantiable for Poly {
    fn flows(flows: &FlowTerm<Poly>) -> &Flows {
        flows.get()
    }
}

impl Instantiable for Concrete {
    fn flows(flows: &FlowTerm<Concrete>) -> &Flows {
        flows.get()
    }
}

/// Where a placeholder's variable comes from.
#[derive(Clone, Copy)]
enum Placeholders {
    /// A scheme's: each made up front with its declared bound.
    Declared,
    /// A cast's patterns': declared with no bound, so any type.
    Unbounded,
    /// A context's type, or a concrete type, which has none.
    Absent,
}

/// One use's fresh variables, one per placeholder, and each written type
/// already instantiated, so a shared subterm is built once.
struct AtUse<'a, V>
where
    V: Instantiable,
{
    written: &'a Types<V>,
    placeholders: Placeholders,
    built: FxHashMap<TyId<V>, Term>,
    ty_vars: FxHashMap<u32, Term>,
    effect_vars: FxHashMap<u32, EffectTerm<InferPhase>>,
    len_vars: FxHashMap<u32, LenVarId>,
    identity_vars: FxHashMap<u32, IdentityTerm<InferPhase>>,
}

impl<'a, V> AtUse<'a, V>
where
    V: Instantiable,
{
    fn new(written: &'a Types<V>, placeholders: Placeholders) -> Self {
        Self {
            written,
            placeholders,
            built: FxHashMap::default(),
            ty_vars: FxHashMap::default(),
            effect_vars: FxHashMap::default(),
            len_vars: FxHashMap::default(),
            identity_vars: FxHashMap::default(),
        }
    }
}

impl Infer {
    /// `scheme` at fresh variables, each with its declared bound. Every
    /// structural node it builds is `Label::Elsewhere`, and every function
    /// type it builds may be `Any` function value; a type parameter's place
    /// takes the program's term as it is (types §1-2).
    pub fn instantiate(&mut self, decls: &Declarations, scheme: &Scheme) -> Instantiated {
        self.instantiate_in(decls.types(), scheme)
    }

    /// `scheme`, whose types are in `poly`, at fresh variables, as
    /// `instantiate` makes a declaration's.
    pub fn instantiate_in(&mut self, poly: &Types<Poly>, scheme: &Scheme) -> Instantiated {
        let mut at = AtUse::new(poly, Placeholders::Declared);
        let vars: Vec<Term> = scheme.bounds.iter().map(|declared| self.fresh(bound_of(declared))).collect();
        // A declaration's type variable never stands for a marked type
        // (types §8): each is kept unmarked for as long as it is solved.
        for var in &vars {
            if let TyKind::Var(var) = self.types.kind(*var) {
                self.unmarked.insert(*var);
            }
        }
        at.ty_vars.extend((0u32..).zip(vars));
        self.mint_result_identities(&mut at, scheme.ty);
        let (ty, flows) = match scheme.impls {
            // A generic function's call reaches one of its impls, and the
            // impl's own signature states the work the call runs
            // (extern-fn §6), so the generic function's effect is the one
            // its impl's instantiation makes, related to it by the caller.
            Some(_) => (self.instantiate_ty(&mut at, scheme.ty), Vec::new()),
            None => self.instantiate_running(&mut at, scheme.ty, &scheme.runs),
        };
        let (ty, pair) = self.with_pair_mark(poly, ty, &scheme.specialized);
        let (ty, declared_result) = self.with_origin_marks(ty);
        let requirements = scheme
            .requires
            .iter()
            .map(|required| RequiredAt {
                generic_function: required.generic_function,
                pattern: self.instantiate_ty(&mut at, required.pattern),
                fact: required.fact,
            })
            .collect();
        Instantiated { ty, requirements, flows, declared_result, pair }
    }

    /// Each identity variable `ty`'s result holds and none of its parameters
    /// gives, a fresh identity made at this use (types §3): a call makes
    /// it, one per call site in each instantiation of the body the call is
    /// in.
    fn mint_result_identities(&mut self, at: &mut AtUse<'_, Poly>, ty: TyId<Poly>) {
        let written = at.written;
        let TyKind::Fn { params, ret, .. } = written.kind(ty) else {
            return;
        };
        let given = identity_variables(written, params.iter().copied());
        for var in identity_variables(written, [*ret]).difference(&given) {
            let made = self.make_identity(self.level);
            at.identity_vars.insert(*var, IdentityTerm::Known(made));
        }
    }

    /// The function type `ty` with each of its parameters' and its result's
    /// positions of a `specialized` type's constructor, behind no reference,
    /// in no function type and in no extern-defined type's argument, which
    /// no marked type is (types §8), carrying one fresh mark, unset.
    fn with_pair_mark(&mut self, poly: &Types<Poly>, ty: Term, specialized: &[TyId<Poly>]) -> (Term, Option<PairAt>) {
        if specialized.is_empty() {
            return (ty, None);
        }
        let TyKind::Fn { params, ret, effect, flows, calls, holds, values } = self.types.kind(ty).clone() else {
            return (ty, None);
        };
        let named: Vec<QualifiedRef> = specialized
            .iter()
            .map(|named| match poly.kind(*named) {
                TyKind::UserDefined { id, .. } => *id,
                other => panic!("a specialized type is an extern-defined type, which alone is marked, not {other:?}"),
            })
            .collect();
        let mark = MarkTerm::Var(self.mark_vars.fresh("Infer::with_pair_mark: u32::MAX variables"));
        let mut positions = Vec::new();
        let params = params.iter().map(|param| self.pair_positions(*param, &named, mark, &mut positions)).collect();
        let ret = self.pair_positions(ret, &named, mark, &mut positions);
        let ty = self.intern(TyKind::Fn { params, ret, effect, flows, calls, holds, values });
        (ty, Some(PairAt { mark, positions, specialized: specialized.into() }))
    }

    fn pair_positions(&mut self, term: Term, named: &[QualifiedRef], mark: MarkTerm<InferPhase>, positions: &mut Vec<Term>) -> Term {
        acvus_utils::grow(|| {
            let kind = self.types.kind(term).clone();
            match kind {
                TyKind::Ref(..) | TyKind::Fn { .. } | TyKind::Var(_) => term,
                TyKind::UserDefined { id, type_args, effect, identity, .. } if named.contains(&id) => {
                    let position = self.intern(TyKind::UserDefined { id, type_args, effect, identity, marked: mark });
                    positions.push(position);
                    position
                }
                TyKind::UserDefined { .. } => term,
                kind => {
                    let mut parts = Vec::new();
                    kind.for_each_child(|part| parts.push(part));
                    let mut given = parts
                        .into_iter()
                        .map(|part| self.pair_positions(part, named, mark, positions))
                        .collect::<Vec<_>>()
                        .into_iter();
                    let rebuilt = kind.with_children(&mut |_| given.next().expect("one part given for each part read"));
                    match rebuilt == kind {
                        true => term,
                        false => self.intern(rebuilt),
                    }
                }
            }
        })
    }

    /// The function type `ty` with each mark its result holds as declared
    /// marked, as itself or a part but behind no reference and in no
    /// function type, a fresh mark of its own (types §8): the value the
    /// call gives starts marked, and each use taking it unmarked raises it.
    /// Its parameters keep theirs, fixing what enters them marked. The
    /// result as declared is returned beside, where any mark was made
    /// fresh.
    fn with_origin_marks(&mut self, ty: Term) -> (Term, Option<Term>) {
        let TyKind::Fn { params, ret, effect, flows, calls, holds, values } = self.types.kind(ty).clone() else {
            return (ty, None);
        };
        let given = self.fresh_origin_marks(ret);
        match given == ret {
            true => (ty, None),
            false => (self.intern(TyKind::Fn { params, ret: given, effect, flows, calls, holds, values }), Some(ret)),
        }
    }

    /// `term` with each mark it holds written marked, as itself or a part
    /// but behind no reference and in no function type, a fresh mark of
    /// its own: a value given at it starts marked (types §8).
    pub fn fresh_origin_marks(&mut self, term: Term) -> Term {
        acvus_utils::grow(|| {
            let kind = self.types.kind(term).clone();
            if matches!(kind, TyKind::Ref(..) | TyKind::Fn { .. } | TyKind::Var(_)) {
                return term;
            }
            let mut parts = Vec::new();
            kind.for_each_child(|part| parts.push(part));
            let mut given = parts.into_iter().map(|part| self.fresh_origin_marks(part)).collect::<Vec<_>>().into_iter();
            let mut rebuilt = kind.with_children(&mut |_| given.next().expect("one part given for each part read"));
            if let TyKind::UserDefined { marked: marked @ MarkTerm::Known(Mark::Marked), .. } = &mut rebuilt {
                *marked = self.fresh_mark();
            }
            match rebuilt == kind {
                true => term,
                false => self.intern(rebuilt),
            }
        })
    }

    /// A declared function type at this use, its effect the call's
    /// (extern-fn §2): a fresh effect into which the type's own effect and
    /// each effect variable of `runs` flow. An effect variable no parameter
    /// gives names no work, and is pure. A type that is no function type
    /// is no call's, and is instantiated as it is written.
    fn instantiate_running(&mut self, at: &mut AtUse<'_, Poly>, ty: TyId<Poly>, runs: &[u32]) -> (Term, Vec<FlowId>) {
        let written = at.written;
        let TyKind::Fn { params, .. } = written.kind(ty) else {
            return (self.instantiate_ty(at, ty), Vec::new());
        };
        let given = effect_variables(written, params.iter().copied());
        for var in effect_variables(written, [ty]).difference(&given) {
            at.effect_vars.insert(*var, EffectTerm::Known(Effect::pure()));
        }
        let instantiated = self.instantiate_ty(at, ty);
        let TyKind::Fn { params, ret, effect: own, flows, calls, holds, values } = self.types.kind(instantiated).clone() else {
            unreachable!("instantiate_running: a function type instantiates as a function type")
        };
        let call = EffectTerm::Var(self.fresh_effect());
        let mut flows_in = vec![self.flows_into(&own, &call)];
        for var in runs {
            let ran = at
                .effect_vars
                .get(var)
                .cloned()
                .expect("verify refuses a `runs` entry that is no effect variable of its signature (RunsNoEffectVariable)");
            flows_in.push(self.flows_into(&ran, &call));
        }
        let ty = self.intern(TyKind::Fn { params, ret, effect: call, flows, calls, holds, values });
        (ty, flows_in)
    }

    /// A declared type that is not a function's, such as a context's or an
    /// input's, at this use: every structural node it builds is
    /// `Label::Elsewhere`, and every function type it builds may be `Any`
    /// function value (types §1-2).
    ///
    /// # Panics
    /// `ty` holds a type variable: `verify` refuses a context or an input
    /// whose type holds one (`TypeVariableInValueType`).
    pub fn instantiate_type(&mut self, decls: &Declarations, ty: TyId<Poly>) -> Term {
        let mut at = AtUse::new(decls.types(), Placeholders::Absent);
        self.instantiate_ty(&mut at, ty)
    }

    /// `context`'s declared type `ty` at this use, as `instantiate_type`
    /// makes one, each identity it holds the one the declarations made for
    /// it (context §3), outside every body.
    pub fn instantiate_context(&mut self, decls: &Declarations, context: QualifiedRef, ty: TyId<Poly>) -> Term {
        let mut at = AtUse::new(decls.types(), Placeholders::Absent);
        for placeholder in identity_variables(decls.types(), [ty]) {
            let identity = decls
                .context_identity(context, placeholder)
                .expect("verify makes an identity for each identity placeholder of a context's type");
            at.identity_vars.insert(placeholder, self.written_identity(identity));
        }
        self.instantiate_ty(&mut at, ty)
    }

    /// A concrete type, such as a fixed input's, at this use, as
    /// `instantiate_type` makes a declared one: written elsewhere.
    pub fn instantiate_concrete(&mut self, types: &Types<Concrete>, ty: Ty) -> Term {
        let mut at = AtUse::new(types, Placeholders::Absent);
        self.instantiate_ty(&mut at, ty)
    }

    /// A cast's or a conversion's two patterns at one use, sharing their
    /// fresh variables (extern-fn §7).
    pub(super) fn instantiate_ends(&mut self, poly: &Types<Poly>, from: TyId<Poly>, to: TyId<Poly>) -> (Term, Term) {
        let mut at = AtUse::new(poly, Placeholders::Unbounded);
        let from = self.instantiate_ty(&mut at, from);
        let to = self.instantiate_ty(&mut at, to);
        (from, to)
    }

    fn instantiate_ty<V>(&mut self, at: &mut AtUse<'_, V>, ty: TyId<V>) -> Term
    where
        V: Instantiable,
    {
        acvus_utils::grow(|| self.instantiate_level(at, ty))
    }

    fn instantiate_level<V>(&mut self, at: &mut AtUse<'_, V>, ty: TyId<V>) -> Term
    where
        V: Instantiable,
    {
        if let Some(built) = at.built.get(&ty) {
            return *built;
        }
        let written = at.written;
        let kind = match written.kind(ty) {
            TyKind::Var(index) => {
                let var = self.ty_var_at(at, index.index());
                at.built.insert(ty, var);
                return var;
            }
            TyKind::Int(width) => TyKind::Int(*width),
            TyKind::Float => TyKind::Float,
            TyKind::Char => TyKind::Char,
            TyKind::String => TyKind::String,
            TyKind::Bool => TyKind::Bool,
            TyKind::Unit => TyKind::Unit,
            TyKind::Never => TyKind::Never,
            TyKind::Order => TyKind::Order,
            TyKind::Str => TyKind::Str,
            TyKind::Handle(result) => TyKind::Handle(self.instantiate_ty(at, *result)),
            TyKind::Closures(signatures) => TyKind::Closures(signatures.iter().map(|signature| self.instantiate_ty(at, *signature)).collect()),
            TyKind::Array(element, len, _) => {
                let element = self.instantiate_ty(at, *element);
                let len = match len {
                    LenTerm::Known(len) => LenTerm::Known(*len),
                    LenTerm::Var(index) => {
                        LenTerm::Var(*at.len_vars.entry(index.index()).or_insert_with(|| self.fresh_len()))
                    }
                };
                TyKind::Array(element, len, Label::Elsewhere)
            }
            TyKind::Slice(element) => TyKind::Slice(self.instantiate_ty(at, *element)),
            TyKind::Option(inner, _) => TyKind::Option(self.instantiate_ty(at, *inner), Label::Elsewhere),
            TyKind::Result(ok, err, _) => TyKind::Result(self.instantiate_ty(at, *ok), self.instantiate_ty(at, *err), Label::Elsewhere),
            TyKind::Tuple(items, _) => TyKind::Tuple(items.iter().map(|item| self.instantiate_ty(at, *item)).collect(), Label::Elsewhere),
            TyKind::Object { fields, label: () } => TyKind::Object {
                fields: fields
                    .iter()
                    .map(|(name, field)| (*name, self.instantiate_ty(at, *field)))
                    .collect(),
                label: Label::Elsewhere,
            },
            TyKind::Enum { name, variants, .. } => TyKind::Enum {
                name: *name,
                variants: variants
                    .iter()
                    .map(|(tag, payload)| (*tag, payload.map(|payload| self.instantiate_ty(at, payload))))
                    .collect(),
                label: Label::Elsewhere,
            },
            TyKind::Fn {
                params,
                ret,
                effect,
                flows,
                calls,
                holds,
                values: (),
            } => TyKind::Fn {
                params: params
                    .iter()
                    .map(|param| self.instantiate_ty(at, *param))
                    .collect(),
                ret: self.instantiate_ty(at, *ret),
                effect: self.effect_at(at, effect),
                flows: FlowTerm::Known(V::flows(flows).clone()),
                calls: CallsTerm::Known(*calls),
                holds: HoldsTerm::Known(*holds),
                values: self.fresh_values_of(FnValues::Any),
            },
            TyKind::UserDefined {
                id,
                type_args,
                effect,
                identity,
                marked,
            } => TyKind::UserDefined {
                id: *id,
                type_args: type_args.iter().map(|arg| self.instantiate_ty(at, *arg)).collect(),
                effect: effect.as_ref().map(|effect| self.effect_at(at, effect)),
                identity: identity.map(|identity| match identity {
                    IdentityTerm::Known(id) => self.written_identity(id),
                    IdentityTerm::Var(index) => {
                        *at.identity_vars.entry(index.index()).or_insert_with(|| IdentityTerm::Var(self.fresh_identity()))
                    }
                }),
                marked: match marked {
                    MarkTerm::Known(mark) => MarkTerm::Known(*mark),
                    MarkTerm::Var(never) => match *never {},
                },
            },
            TyKind::Ref(mutability, target) => TyKind::Ref(*mutability, self.instantiate_ty(at, *target)),
        };
        let term = self.intern(kind);
        at.built.insert(ty, term);
        term
    }

    /// An identity a written type names: made outside every body.
    fn written_identity(&mut self, identity: IdentityId) -> IdentityTerm<InferPhase> {
        self.made_at.insert(identity, Level::OUTSIDE);
        IdentityTerm::Known(identity)
    }

    /// The variable of the placeholder `index`.
    ///
    /// # Panics
    /// A scheme's placeholder has no variable made up front: `verify`
    /// refuses a scheme whose bounds are not one per type variable
    /// (`BoundsDoNotFit`), so a verified scheme's variables are all made.
    /// A context's or an input's type holds a placeholder: `verify` refuses
    /// one (`TypeVariableInValueType`).
    fn ty_var_at<V>(&mut self, at: &mut AtUse<'_, V>, index: u32) -> Term
    where
        V: Instantiable,
    {
        match at.placeholders {
            Placeholders::Declared => *at
                .ty_vars
                .get(&index)
                .expect("a verified scheme gives each of its type variables one bound (BoundsDoNotFit)"),
            Placeholders::Unbounded => *at.ty_vars.entry(index).or_insert_with(|| self.fresh(Bound::Any)),
            Placeholders::Absent => panic!(
                "a verified context's type holds no type variable (TypeVariableInValueType), \
                 and a concrete type none, but {index} is written in one"
            ),
        }
    }

    fn effect_at<V>(&mut self, at: &mut AtUse<'_, V>, effect: &EffectTerm<V>) -> EffectTerm<InferPhase>
    where
        V: Instantiable,
    {
        match effect {
            EffectTerm::Known(effect) => EffectTerm::Known(effect.clone()),
            EffectTerm::Var(index) => at
                .effect_vars
                .entry(index.index())
                .or_insert_with(|| EffectTerm::Var(self.fresh_effect()))
                .clone(),
        }
    }
}

/// A declaration's bound as a variable's.
fn bound_of(bound: &TyVarBound) -> Bound {
    match bound {
        TyVarBound::Any => Bound::Any,
        TyVarBound::Settled => Bound::Settled,
        TyVarBound::Integer { signed, among } => Bound::Integer {
            signed: *signed,
            among: among.clone(),
            literal: false,
        },
    }
}
