//! A declaration's scheme at one use (extern-fn §6).

use acvus_utils::QualifiedRef;
use rustc_hash::FxHashMap;

use super::{Bound, Infer, Term};
use crate::decl::{Declarations, NamedLaw, Scheme, TyVarBound};
use crate::ty::{
    CallsTerm, EffectArg, EffectTerm, HoldsTerm, EffectVarId, FlowTerm, IdentityTerm, IdentityVarId, Infer as InferPhase, Label, LenTerm,
    LenVarId, ObjectTy, Poly, ReprVarId, TyId, TyKind, TypeArg, Types,
};

/// A requirement at the use's own terms: an impl of `generic_function` whose
/// type joins `pattern`, stating `law` where named.
#[derive(Debug, Clone, PartialEq)]
pub struct RequiredAt {
    pub generic_function: QualifiedRef,
    pub pattern: Term,
    pub law: Option<NamedLaw>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Instantiated {
    /// The function type at this use.
    pub ty: Term,
    pub requirements: Vec<RequiredAt>,
}

/// Where a placeholder's variable comes from.
#[derive(Clone, Copy)]
enum Placeholders {
    /// A scheme's: each made up front with its declared bound.
    Declared,
    /// A cast's patterns': declared with no bound, so any type.
    Unbounded,
    /// A context's or an input's type, which has none.
    Absent,
}

/// One use's fresh variables, one per placeholder, and each declared type
/// already instantiated, so a shared subterm is built once.
struct AtUse<'a> {
    poly: &'a Types<Poly>,
    placeholders: Placeholders,
    built: FxHashMap<TyId<Poly>, Term>,
    ty_vars: FxHashMap<u32, Term>,
    effect_vars: FxHashMap<u32, EffectVarId>,
    len_vars: FxHashMap<u32, LenVarId>,
    identity_vars: FxHashMap<u32, IdentityVarId>,
    repr_vars: FxHashMap<u32, ReprVarId>,
}

impl<'a> AtUse<'a> {
    fn new(poly: &'a Types<Poly>, placeholders: Placeholders) -> Self {
        Self {
            poly,
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
    /// structural node it builds is `Label::Elsewhere`; a type parameter's
    /// place takes the program's term as it is (types §1).
    pub fn instantiate(&mut self, decls: &Declarations, scheme: &Scheme) -> Instantiated {
        let mut at = AtUse::new(decls.types(), Placeholders::Declared);
        let vars: Vec<Term> = scheme.bounds.iter().map(|declared| self.fresh(bound_of(declared))).collect();
        at.ty_vars.extend((0u32..).zip(vars));
        let ty = self.instantiate_ty(&mut at, scheme.ty);
        let requirements = scheme
            .requires
            .iter()
            .map(|required| RequiredAt {
                generic_function: required.generic_function,
                pattern: self.instantiate_ty(&mut at, required.pattern),
                law: required.law,
            })
            .collect();
        Instantiated { ty, requirements }
    }

    /// A declared type that is not a function's, such as a context's or an
    /// input's, at this use: every structural node it builds is
    /// `Label::Elsewhere` (types §1).
    ///
    /// # Panics
    /// `ty` holds a type variable: `verify` refuses a context or an input
    /// whose type holds one (`TypeVariableInValueType`).
    pub fn instantiate_type(&mut self, decls: &Declarations, ty: TyId<Poly>) -> Term {
        let mut at = AtUse::new(decls.types(), Placeholders::Absent);
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

    fn instantiate_ty(&mut self, at: &mut AtUse<'_>, ty: TyId<Poly>) -> Term {
        acvus_utils::grow(|| self.instantiate_level(at, ty))
    }

    fn instantiate_level(&mut self, at: &mut AtUse<'_>, ty: TyId<Poly>) -> Term {
        if let Some(built) = at.built.get(&ty) {
            return *built;
        }
        let poly = at.poly;
        let kind = match poly.kind(ty) {
            TyKind::Var(index) => {
                let var = self.ty_var_at(at, *index);
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
            TyKind::Array(element, len, _) => {
                let element = self.instantiate_ty(at, *element);
                let len = match len {
                    LenTerm::Known(len) => LenTerm::Known(*len),
                    LenTerm::Var(index) => {
                        LenTerm::Var(*at.len_vars.entry(*index).or_insert_with(|| self.fresh_len()))
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
            } => TyKind::Fn {
                params: params
                    .iter()
                    .map(|param| self.instantiate_ty(at, *param))
                    .collect(),
                ret: self.instantiate_ty(at, *ret),
                effect: self.effect_at(at, effect),
                flows: FlowTerm::Known(flows.get().clone()),
                calls: CallsTerm::Known(*calls),
                holds: HoldsTerm::Known(*holds),
            },
            TyKind::UserDefined {
                id,
                type_args,
                effect_args,
                identity_args,
            } => TyKind::UserDefined {
                id: *id,
                type_args: type_args.iter().map(|arg| self.type_arg_at(at, arg)).collect(),
                effect_args: effect_args
                    .iter()
                    .map(|arg| match arg {
                        EffectArg::Uniform(effect) => EffectArg::Uniform(self.effect_at(at, effect)),
                        EffectArg::Specialized(effect) => EffectArg::Specialized(self.effect_at(at, effect)),
                    })
                    .collect(),
                identity_args: identity_args
                    .iter()
                    .map(|identity| match identity {
                        IdentityTerm::Known(id) => IdentityTerm::Known(*id),
                        IdentityTerm::Var(index) => {
                            IdentityTerm::Var(*at.identity_vars.entry(*index).or_insert_with(|| self.fresh_identity()))
                        }
                    })
                    .collect(),
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
    fn ty_var_at(&mut self, at: &mut AtUse<'_>, index: u32) -> Term {
        match at.placeholders {
            Placeholders::Declared => *at
                .ty_vars
                .get(&index)
                .expect("a verified scheme gives each of its type variables one bound (BoundsDoNotFit)"),
            Placeholders::Unbounded => *at.ty_vars.entry(index).or_insert_with(|| self.fresh(Bound::Any)),
            Placeholders::Absent => panic!(
                "a verified context's or input's type holds no type variable (TypeVariableInValueType), \
                 but {index} is written in one"
            ),
        }
    }

    fn effect_at(&mut self, at: &mut AtUse<'_>, effect: &EffectTerm<Poly>) -> EffectTerm<InferPhase> {
        match effect {
            EffectTerm::Known(effect) => EffectTerm::Known(effect.clone()),
            EffectTerm::Var(index) => {
                EffectTerm::Var(*at.effect_vars.entry(*index).or_insert_with(|| self.fresh_effect()))
            }
        }
    }

    fn type_arg_at(&mut self, at: &mut AtUse<'_>, arg: &TypeArg<Poly>) -> TypeArg<InferPhase> {
        match arg {
            TypeArg::Uniform(ty) => TypeArg::Uniform(self.instantiate_ty(at, *ty)),
            TypeArg::Specialized(ty) => TypeArg::Specialized(self.instantiate_ty(at, *ty)),
            TypeArg::Open(index, ty) => {
                let mark = *at.repr_vars.entry(*index).or_insert_with(|| self.fresh_repr());
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
