//! A declaration's scheme at one use (extern-fn §6).

use acvus_utils::QualifiedRef;
use rustc_hash::FxHashMap;

use super::{Infer, Term};
use crate::decl::{Declarations, Scheme, TyVarBound};
use crate::flows::FlowTerm;
use crate::laws::NamedLaw;
use crate::ty::{
    EffectArg, EffectTerm, EffectVarId, IdentityTerm, IdentityVarId, Infer as InferPhase, Label, LenTerm, LenVarId,
    ObjectTy, ParamTerm, Poly, ReprVarId, TyId, TyKind, TypeArg, Types,
};

/// A requirement at the use's own terms: an instance of `signature` whose
/// type joins `pattern`, stating `law` where named.
#[derive(Debug, Clone, PartialEq)]
pub struct RequiredAt {
    pub signature: QualifiedRef,
    pub pattern: Term,
    pub law: Option<NamedLaw>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Instantiated {
    /// The function type at this use.
    pub ty: Term,
    pub requirements: Vec<RequiredAt>,
}

/// One use's fresh variables, one per placeholder of the scheme, and each
/// declared type already instantiated, so a shared subterm is built once.
struct AtUse<'a> {
    poly: &'a Types<Poly>,
    bounds: &'a [TyVarBound],
    built: FxHashMap<TyId<Poly>, Term>,
    ty_vars: FxHashMap<u32, Term>,
    effect_vars: FxHashMap<u32, EffectVarId>,
    len_vars: FxHashMap<u32, LenVarId>,
    identity_vars: FxHashMap<u32, IdentityVarId>,
    repr_vars: FxHashMap<u32, ReprVarId>,
}

impl Infer {
    /// `scheme` at fresh variables, each with its declared bound. Every
    /// structural node it builds is `Label::Opaque`; a type parameter's
    /// place takes the program's term as it is (types §1).
    pub fn instantiate(&mut self, decls: &Declarations, scheme: &Scheme) -> Instantiated {
        let mut at = AtUse {
            poly: decls.types(),
            bounds: &scheme.bounds,
            built: FxHashMap::default(),
            ty_vars: FxHashMap::default(),
            effect_vars: FxHashMap::default(),
            len_vars: FxHashMap::default(),
            identity_vars: FxHashMap::default(),
            repr_vars: FxHashMap::default(),
        };
        let ty = self.instantiate_ty(&mut at, scheme.ty);
        let requirements = scheme
            .requires
            .iter()
            .map(|required| RequiredAt {
                signature: required.signature,
                pattern: self.instantiate_ty(&mut at, required.pattern),
                law: required.law,
            })
            .collect();
        Instantiated { ty, requirements }
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
            TyKind::Array(element, len) => {
                let element = self.instantiate_ty(at, *element);
                let len = match len {
                    LenTerm::Known(len) => LenTerm::Known(*len),
                    LenTerm::Var(index) => {
                        LenTerm::Var(*at.len_vars.entry(*index).or_insert_with(|| self.fresh_len()))
                    }
                };
                TyKind::Array(element, len)
            }
            TyKind::Slice(element) => TyKind::Slice(self.instantiate_ty(at, *element)),
            TyKind::Option(inner) => TyKind::Option(self.instantiate_ty(at, *inner)),
            TyKind::Result(ok, err) => TyKind::Result(self.instantiate_ty(at, *ok), self.instantiate_ty(at, *err)),
            TyKind::Tuple(items) => TyKind::Tuple(items.iter().map(|item| self.instantiate_ty(at, *item)).collect()),
            TyKind::Object(ObjectTy::Structural(fields, ())) => TyKind::Object(ObjectTy::Structural(
                fields
                    .iter()
                    .map(|(name, field)| (*name, self.instantiate_ty(at, *field)))
                    .collect(),
                Label::Opaque,
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
                label: Label::Opaque,
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
                    .map(|param| ParamTerm {
                        name: param.name,
                        ty: self.instantiate_ty(at, param.ty),
                    })
                    .collect(),
                ret: self.instantiate_ty(at, *ret),
                effect: self.effect_at(at, effect),
                flows: FlowTerm::Known(flows.get().clone()),
                calls: *calls,
                holds: *holds,
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

    /// The fresh variable of the scheme's type variable `index`, with its
    /// declared bound.
    ///
    /// # Panics
    /// The scheme declares no bound at `index`: every type parameter
    /// carries one (extern-fn §1).
    fn ty_var_at(&mut self, at: &mut AtUse<'_>, index: u32) -> Term {
        if let Some(var) = at.ty_vars.get(&index) {
            return *var;
        }
        let declared = at
            .bounds
            .get(index as usize)
            .expect("every type parameter of a scheme carries one bound (extern-fn §1)");
        let bound = self.bound_of(declared);
        let var = self.fresh(bound);
        at.ty_vars.insert(index, var);
        var
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
