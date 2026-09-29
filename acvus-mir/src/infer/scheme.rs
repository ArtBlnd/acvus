//! A declaration's scheme at one use (extern-fn §6), and a type written
//! elsewhere, declared or concrete, at one use.

use std::convert::Infallible;

use acvus_utils::QualifiedRef;
use rustc_hash::FxHashMap;

use super::{Bound, Infer, Term};
use crate::decl::{Declarations, NamedFact, Scheme, TyVarBound};
use crate::ty::{
    Calls, CallsTerm, Concrete, EffectTerm, EffectVarId, FlowTerm, Flows, FnValues, Holds, HoldsTerm, IdentityTerm, IdentityVarId,
    Infer as InferPhase, Label, LenTerm, LenVarId, ObjectTy, Phase, Poly, ReprVarId, Ty, TyId, TyKind, TypeArg, Types,
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
        ReprVar: Placeholder,
        FlowVar = Infallible,
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
    effect_vars: FxHashMap<u32, EffectVarId>,
    len_vars: FxHashMap<u32, LenVarId>,
    identity_vars: FxHashMap<u32, IdentityVarId>,
    repr_vars: FxHashMap<u32, ReprVarId>,
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
            repr_vars: FxHashMap::default(),
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
        at.ty_vars.extend((0u32..).zip(vars));
        let ty = self.instantiate_ty(&mut at, scheme.ty);
        let requirements = scheme
            .requires
            .iter()
            .map(|required| RequiredAt {
                generic_function: required.generic_function,
                pattern: self.instantiate_ty(&mut at, required.pattern),
                fact: required.fact,
            })
            .collect();
        Instantiated { ty, requirements }
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
            TyKind::Object(ObjectTy::Structural(fields, ())) => TyKind::Object(ObjectTy::Structural(
                fields
                    .iter()
                    .map(|(name, field)| (*name, self.instantiate_ty(at, *field)))
                    .collect(),
                Label::Elsewhere,
            )),
            TyKind::Object(ObjectTy::Declared(id, fields)) => TyKind::Object(ObjectTy::Declared(
                *id,
                fields
                    .iter()
                    .map(|(name, field)| (*name, self.instantiate_ty(at, *field)))
                    .collect(),
            )),
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
            } => TyKind::UserDefined {
                id: *id,
                type_args: type_args.iter().map(|arg| self.type_arg_at(at, arg)).collect(),
                effect: effect.as_ref().map(|effect| self.effect_at(at, effect)),
                identity: identity.map(|identity| match identity {
                    IdentityTerm::Known(id) => IdentityTerm::Known(id),
                    IdentityTerm::Var(index) => {
                        IdentityTerm::Var(*at.identity_vars.entry(index.index()).or_insert_with(|| self.fresh_identity()))
                    }
                }),
            },
            TyKind::Ref(mutability, target) => TyKind::Ref(*mutability, self.type_arg_at(at, target)),
        };
        let term = self.intern(kind);
        at.built.insert(ty, term);
        term
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
            EffectTerm::Var(index) => {
                EffectTerm::Var(*at.effect_vars.entry(index.index()).or_insert_with(|| self.fresh_effect()))
            }
        }
    }

    fn type_arg_at<V>(&mut self, at: &mut AtUse<'_, V>, arg: &TypeArg<V>) -> TypeArg<InferPhase>
    where
        V: Instantiable,
    {
        match arg {
            TypeArg::Uniform(ty) => TypeArg::Uniform(self.instantiate_ty(at, *ty)),
            TypeArg::Specialized(ty) => TypeArg::Specialized(self.instantiate_ty(at, *ty)),
            TypeArg::Open(index, ty) => {
                let mark = *at.repr_vars.entry(index.index()).or_insert_with(|| self.fresh_repr());
                TypeArg::Open(mark, self.instantiate_ty(at, *ty))
            }
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
