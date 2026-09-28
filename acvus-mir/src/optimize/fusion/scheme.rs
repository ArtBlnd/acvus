use rustc_hash::FxHashMap;

use crate::flows::FlowTerm;
use crate::ir::{known_effect, known_len};
use crate::ty::{
    Concrete, EffectArg, EffectTerm, IdentityTerm, LenTerm, ObjectTy, ParamTerm, Poly, Ty, TyId, TyKind, TypeArg,
    Types,
};

/// Whether an argument carries `#` (types §8), as a concrete argument
/// settles an open mark of a declared one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Mark {
    Uniform,
    Specialized,
}

/// A declared type's variables, bound where it is matched against a
/// concrete type.
#[derive(Debug, Clone, Default)]
pub(super) struct Bindings {
    tys: FxHashMap<u32, Ty>,
    effects: FxHashMap<u32, EffectTerm<Concrete>>,
    identities: FxHashMap<u32, IdentityTerm<Concrete>>,
    lens: FxHashMap<u32, LenTerm<Concrete>>,
    marks: FxHashMap<u32, Mark>,
}

/// Which value a type `apply` builds is for. A value the program observes,
/// a consumer's result or anything that leaves the fused loop, is typed by
/// the IR's recorded type and never built here; every type `apply` builds is
/// a value's that lives inside the fused loop.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Built {
    /// A value inside the fused loop, which nothing outside it observes:
    /// an argument whose `#` nothing decides is not specialized (types §5).
    InsideTheLoop,
}

impl Bindings {
    pub(super) fn bind(&mut self, poly: &Types<Poly>, pattern: TyId<Poly>, types: &Types<Concrete>, ty: Ty) -> bool {
        acvus_utils::grow(|| self.bind_level(poly, pattern, types, ty))
    }

    fn bind_all(
        &mut self,
        poly: &Types<Poly>,
        types: &Types<Concrete>,
        pairs: impl IntoIterator<Item = (TyId<Poly>, Ty)>,
    ) -> bool {
        pairs.into_iter().all(|(pattern, ty)| self.bind(poly, pattern, types, ty))
    }

    fn bind_level(&mut self, poly: &Types<Poly>, pattern: TyId<Poly>, types: &Types<Concrete>, ty: Ty) -> bool {
        match (poly.kind(pattern), types.kind(ty)) {
            (TyKind::Var(v), _) => match self.tys.get(v) {
                Some(bound) => *bound == ty,
                None => {
                    self.tys.insert(*v, ty);
                    true
                }
            },
            (TyKind::Int(a), TyKind::Int(b)) => a == b,
            (TyKind::Float, TyKind::Float)
            | (TyKind::Char, TyKind::Char)
            | (TyKind::String, TyKind::String)
            | (TyKind::Bool, TyKind::Bool)
            | (TyKind::Unit, TyKind::Unit)
            | (TyKind::Never, TyKind::Never)
            | (TyKind::Order, TyKind::Order)
            | (TyKind::Str, TyKind::Str) => true,
            (TyKind::Array(p, p_len), TyKind::Array(t, t_len)) => {
                self.bind_len(p_len, t_len) && self.bind(poly, *p, types, *t)
            }
            (TyKind::Tuple(ps), TyKind::Tuple(ts)) => {
                ps.len() == ts.len() && self.bind_all(poly, types, ps.iter().copied().zip(ts.iter().copied()))
            }
            (TyKind::Option(p), TyKind::Option(t))
            | (TyKind::Slice(p), TyKind::Slice(t))
            | (TyKind::Handle(p), TyKind::Handle(t)) => self.bind(poly, *p, types, *t),
            (TyKind::Result(p_ok, p_err), TyKind::Result(t_ok, t_err)) => {
                self.bind(poly, *p_ok, types, *t_ok) && self.bind(poly, *p_err, types, *t_err)
            }
            (
                TyKind::Fn {
                    params: p_params,
                    ret: p_ret,
                    effect: p_effect,
                    calls: p_calls,
                    holds: p_holds,
                    ..
                },
                TyKind::Fn {
                    params: t_params,
                    ret: t_ret,
                    effect: t_effect,
                    calls: t_calls,
                    holds: t_holds,
                    ..
                },
            ) => {
                p_params.len() == t_params.len()
                    && p_calls == t_calls
                    && p_holds == t_holds
                    && self.bind_all(
                        poly,
                        types,
                        p_params.iter().map(|p| p.ty).zip(t_params.iter().map(|t| t.ty)),
                    )
                    && self.bind(poly, *p_ret, types, *t_ret)
                    && self.bind_effect(p_effect, t_effect)
            }
            (
                TyKind::UserDefined {
                    id: p_id,
                    type_args: p_args,
                    effect_args: p_effects,
                    identity_args: p_identities,
                },
                TyKind::UserDefined {
                    id: t_id,
                    type_args: t_args,
                    effect_args: t_effects,
                    identity_args: t_identities,
                },
            ) => {
                p_id == t_id
                    && p_args.len() == t_args.len()
                    && p_effects.len() == t_effects.len()
                    && p_identities.len() == t_identities.len()
                    && p_args
                        .iter()
                        .zip(t_args.iter())
                        .all(|(p, t)| self.bind_arg(poly, p, types, t))
                    && p_effects
                        .iter()
                        .zip(t_effects.iter())
                        .all(|(p, t)| self.bind_effect(p.effect(), t.effect()))
                    && p_identities
                        .iter()
                        .zip(t_identities.iter())
                        .all(|(p, t)| self.bind_identity(p, t))
            }
            (TyKind::Ref(p_mut, p_arg), TyKind::Ref(t_mut, t_arg)) => {
                p_mut == t_mut && self.bind_arg(poly, p_arg, types, t_arg)
            }
            (TyKind::Object(p_object), TyKind::Object(t_object)) => {
                let same_kind = match (p_object, t_object) {
                    (ObjectTy::Structural(..), ObjectTy::Structural(..)) => true,
                    (ObjectTy::Declared(p_id, _), ObjectTy::Declared(t_id, _)) => p_id == t_id,
                    (ObjectTy::Structural(..), ObjectTy::Declared(..))
                    | (ObjectTy::Declared(..), ObjectTy::Structural(..)) => false,
                };
                let (p_fields, t_fields) = (p_object.fields(), t_object.fields());
                same_kind
                    && p_fields.len() == t_fields.len()
                    && p_fields.iter().zip(t_fields.iter()).all(|((p, _), (t, _))| p == t)
                    && self.bind_all(
                        poly,
                        types,
                        p_fields.iter().map(|(_, p)| *p).zip(t_fields.iter().map(|(_, t)| *t)),
                    )
            }
            (
                TyKind::Enum {
                    name: p_name,
                    variants: p_variants,
                    ..
                },
                TyKind::Enum {
                    name: t_name,
                    variants: t_variants,
                    ..
                },
            ) => {
                p_name == t_name
                    && p_variants.len() == t_variants.len()
                    && p_variants
                        .iter()
                        .zip(t_variants.iter())
                        .all(|((p_tag, p_payload), (t_tag, t_payload))| {
                            p_tag == t_tag
                                && match (p_payload, t_payload) {
                                    (Some(p), Some(t)) => self.bind(poly, *p, types, *t),
                                    (None, None) => true,
                                    (Some(_), None) | (None, Some(_)) => false,
                                }
                        })
            }
            _ => false,
        }
    }

    fn bind_arg(&mut self, poly: &Types<Poly>, pattern: &TypeArg<Poly>, types: &Types<Concrete>, arg: &TypeArg<Concrete>) -> bool {
        let held = match arg {
            TypeArg::Uniform(_) => Mark::Uniform,
            TypeArg::Specialized(_) => Mark::Specialized,
            TypeArg::Open(var, _) => match *var {},
        };
        let fits = match pattern {
            TypeArg::Uniform(_) => held == Mark::Uniform,
            TypeArg::Specialized(_) => held == Mark::Specialized,
            TypeArg::Open(mark, _) => match self.marks.get(mark) {
                Some(bound) => *bound == held,
                None => {
                    self.marks.insert(*mark, held);
                    true
                }
            },
        };
        fits && self.bind(poly, pattern.ty(), types, arg.ty())
    }

    fn bind_effect(&mut self, pattern: &EffectTerm<Poly>, effect: &EffectTerm<Concrete>) -> bool {
        match pattern {
            EffectTerm::Known(known) => known == known_effect(effect),
            EffectTerm::Var(v) => match self.effects.get(v) {
                Some(bound) => bound == effect,
                None => {
                    self.effects.insert(*v, effect.clone());
                    true
                }
            },
        }
    }

    fn bind_identity(&mut self, pattern: &IdentityTerm<Poly>, identity: &IdentityTerm<Concrete>) -> bool {
        let id = match identity {
            IdentityTerm::Known(id) => *id,
            IdentityTerm::Var(var) => match *var {},
        };
        match pattern {
            IdentityTerm::Known(known) => *known == id,
            IdentityTerm::Var(v) => match self.identities.get(v) {
                Some(bound) => bound == identity,
                None => {
                    self.identities.insert(*v, *identity);
                    true
                }
            },
        }
    }

    fn bind_len(&mut self, pattern: &LenTerm<Poly>, len: &LenTerm<Concrete>) -> bool {
        match pattern {
            LenTerm::Known(known) => *known == known_len(len),
            LenTerm::Var(v) => match self.lens.get(v) {
                Some(bound) => bound == len,
                None => {
                    self.lens.insert(*v, *len);
                    true
                }
            },
        }
    }

    /// `pattern` with its variables replaced by what they are bound to, in
    /// `types`; `None` where a variable is unbound. `built` says which value
    /// the type is for, and so what a mark nothing fixed is.
    pub(super) fn apply(
        &self,
        poly: &Types<Poly>,
        pattern: TyId<Poly>,
        types: &mut Types<Concrete>,
        built: Built,
    ) -> Option<Ty> {
        acvus_utils::grow(|| self.apply_level(poly, pattern, types, built))
    }

    fn apply_level(
        &self,
        poly: &Types<Poly>,
        pattern: TyId<Poly>,
        types: &mut Types<Concrete>,
        built: Built,
    ) -> Option<Ty> {
        let kind = match poly.kind(pattern) {
            TyKind::Var(v) => return self.tys.get(v).copied(),
            TyKind::Int(k) => TyKind::Int(*k),
            TyKind::Float => TyKind::Float,
            TyKind::Char => TyKind::Char,
            TyKind::String => TyKind::String,
            TyKind::Bool => TyKind::Bool,
            TyKind::Unit => TyKind::Unit,
            TyKind::Never => TyKind::Never,
            TyKind::Order => TyKind::Order,
            TyKind::Str => TyKind::Str,
            TyKind::Handle(inner) => TyKind::Handle(self.apply(poly, *inner, types, built)?),
            TyKind::Slice(inner) => TyKind::Slice(self.apply(poly, *inner, types, built)?),
            TyKind::Option(inner) => TyKind::Option(self.apply(poly, *inner, types, built)?),
            TyKind::Array(inner, len) => {
                TyKind::Array(self.apply(poly, *inner, types, built)?, self.apply_len(len)?)
            }
            TyKind::Result(ok, err) => {
                TyKind::Result(self.apply(poly, *ok, types, built)?, self.apply(poly, *err, types, built)?)
            }
            TyKind::Tuple(items) => TyKind::Tuple(
                items
                    .iter()
                    .map(|item| self.apply(poly, *item, types, built))
                    .collect::<Option<_>>()?,
            ),
            TyKind::Object(object) => {
                let fields = object
                    .fields()
                    .iter()
                    .map(|(name, field)| Some((*name, self.apply(poly, *field, types, built)?)))
                    .collect::<Option<_>>()?;
                TyKind::Object(match object {
                    ObjectTy::Structural(_, ()) => ObjectTy::Structural(fields, ()),
                    ObjectTy::Declared(id, _) => ObjectTy::Declared(*id, fields),
                })
            }
            TyKind::Enum { name, variants, .. } => TyKind::Enum {
                name: *name,
                variants: variants
                    .iter()
                    .map(|(tag, payload)| {
                        Some((
                            *tag,
                            match payload {
                                Some(payload) => Some(self.apply(poly, *payload, types, built)?),
                                None => None,
                            },
                        ))
                    })
                    .collect::<Option<_>>()?,
                label: (),
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
                    .map(|param| {
                        Some(ParamTerm {
                            name: param.name,
                            ty: self.apply(poly, param.ty, types, built)?,
                        })
                    })
                    .collect::<Option<_>>()?,
                ret: self.apply(poly, *ret, types, built)?,
                effect: self.apply_effect(effect)?,
                flows: match flows {
                    FlowTerm::Known(flows) => FlowTerm::Known(flows.clone()),
                    FlowTerm::Var(var) => match *var {},
                },
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
                type_args: type_args
                    .iter()
                    .map(|arg| self.apply_arg(poly, arg, types, built))
                    .collect::<Option<_>>()?,
                effect_args: effect_args
                    .iter()
                    .map(|arg| {
                        let effect = self.apply_effect(arg.effect())?;
                        Some(match arg {
                            EffectArg::Uniform(_) => EffectArg::Uniform(effect),
                            EffectArg::Specialized(_) => EffectArg::Specialized(effect),
                        })
                    })
                    .collect::<Option<_>>()?,
                identity_args: identity_args
                    .iter()
                    .map(|identity| match identity {
                        IdentityTerm::Known(id) => Some(IdentityTerm::Known(*id)),
                        IdentityTerm::Var(v) => self.identities.get(v).copied(),
                    })
                    .collect::<Option<_>>()?,
            },
            TyKind::Ref(mutability, arg) => TyKind::Ref(*mutability, self.apply_arg(poly, arg, types, built)?),
        };
        Some(types.intern(kind))
    }

    fn apply_arg(
        &self,
        poly: &Types<Poly>,
        arg: &TypeArg<Poly>,
        types: &mut Types<Concrete>,
        built: Built,
    ) -> Option<TypeArg<Concrete>> {
        let ty = self.apply(poly, arg.ty(), types, built)?;
        let mark = match arg {
            TypeArg::Uniform(_) => Mark::Uniform,
            TypeArg::Specialized(_) => Mark::Specialized,
            TypeArg::Open(mark, _) => match (self.marks.get(mark), built) {
                (Some(fixed), _) => *fixed,
                (None, Built::InsideTheLoop) => Mark::Uniform,
            },
        };
        Some(match mark {
            Mark::Uniform => TypeArg::Uniform(ty),
            Mark::Specialized => TypeArg::Specialized(ty),
        })
    }

    fn apply_effect(&self, effect: &EffectTerm<Poly>) -> Option<EffectTerm<Concrete>> {
        match effect {
            EffectTerm::Known(effect) => Some(EffectTerm::Known(effect.clone())),
            EffectTerm::Var(v) => self.effects.get(v).cloned(),
        }
    }

    fn apply_len(&self, len: &LenTerm<Poly>) -> Option<LenTerm<Concrete>> {
        match len {
            LenTerm::Known(len) => Some(LenTerm::Known(*len)),
            LenTerm::Var(v) => self.lens.get(v).copied(),
        }
    }
}
