use std::convert::Infallible;

use rustc_hash::FxHashMap;

use crate::flows::FlowTerm;
use crate::ty::{
    Concrete, EffectTerm, IdentityTerm, LenTerm, PolyTy, Repr, Ty, TyTerm, TypeArg,
};

#[derive(Debug, Clone, Default)]
pub(super) struct Bindings {
    tys: FxHashMap<u32, Ty>,
    effects: FxHashMap<u32, EffectTerm<Concrete>>,
    identities: FxHashMap<u32, IdentityTerm<Concrete>>,
    lens: FxHashMap<u32, LenTerm<Concrete>>,
    reprs: FxHashMap<u32, Repr<Concrete>>,
}

/// The representation an open `ρ` no position fixed takes: the solve takes
/// the least element, `Uniform`, for it (RFC-0042 rule 3), as
/// `Solver::freeze_repr` does.
fn unfixed_repr() -> Repr<Concrete> {
    Repr::Uniform
}

impl Bindings {
    pub(super) fn bind(&mut self, pattern: &PolyTy, ty: &Ty) -> bool {
        acvus_utils::grow(|| self.bind_level(pattern, ty))
    }

    fn bind_level(&mut self, pattern: &PolyTy, ty: &Ty) -> bool {
        match (pattern, ty) {
            (TyTerm::Var(v), _) => match self.tys.get(v) {
                Some(bound) => bound == ty,
                None => {
                    self.tys.insert(*v, ty.clone());
                    true
                }
            },
            (TyTerm::Int(a), TyTerm::Int(b)) => a == b,
            (TyTerm::Float, TyTerm::Float)
            | (TyTerm::Char, TyTerm::Char)
            | (TyTerm::String, TyTerm::String)
            | (TyTerm::Bool, TyTerm::Bool)
            | (TyTerm::Unit, TyTerm::Unit)
            | (TyTerm::Never, TyTerm::Never)
            | (TyTerm::Order, TyTerm::Order)
            | (TyTerm::Str, TyTerm::Str) => true,
            (TyTerm::Array(p, p_len), TyTerm::Array(t, t_len)) => {
                self.bind(p, t) && self.bind_len(p_len, t_len)
            }
            (TyTerm::Tuple(ps), TyTerm::Tuple(ts)) => {
                ps.len() == ts.len() && ps.iter().zip(ts).all(|(p, t)| self.bind(p, t))
            }
            (TyTerm::Option(p), TyTerm::Option(t))
            | (TyTerm::Slice(p), TyTerm::Slice(t))
            | (TyTerm::Handle(p), TyTerm::Handle(t)) => self.bind(p, t),
            (TyTerm::Result(p_ok, p_err), TyTerm::Result(t_ok, t_err)) => {
                self.bind(p_ok, t_ok) && self.bind(p_err, t_err)
            }
            (
                TyTerm::Fn {
                    params: p_params,
                    ret: p_ret,
                    captures: p_captures,
                    effect: p_effect,
                    ..
                },
                TyTerm::Fn {
                    params: t_params,
                    ret: t_ret,
                    captures: t_captures,
                    effect: t_effect,
                    ..
                },
            ) => {
                p_params.len() == t_params.len()
                    && p_captures.len() == t_captures.len()
                    && p_params
                        .iter()
                        .zip(t_params)
                        .all(|(p, t)| self.bind(&p.ty, &t.ty))
                    && p_captures.iter().zip(t_captures).all(|(p, t)| self.bind(p, t))
                    && self.bind(p_ret, t_ret)
                    && self.bind_effect(p_effect, t_effect)
            }
            (
                TyTerm::UserDefined {
                    id: p_id,
                    type_args: p_args,
                    effect_args: p_effects,
                    identity_args: p_identities,
                    region_params: p_regions,
                },
                TyTerm::UserDefined {
                    id: t_id,
                    type_args: t_args,
                    effect_args: t_effects,
                    identity_args: t_identities,
                    region_params: t_regions,
                },
            ) => {
                p_id == t_id
                    && p_regions == t_regions
                    && p_args.len() == t_args.len()
                    && p_effects.len() == t_effects.len()
                    && p_identities.len() == t_identities.len()
                    && p_args.iter().zip(t_args).all(|(p, t)| self.bind_arg(p, t))
                    && p_effects
                        .iter()
                        .zip(t_effects)
                        .all(|(p, t)| self.bind_effect(p.effect(), t.effect()))
                    && p_identities
                        .iter()
                        .zip(t_identities)
                        .all(|(p, t)| self.bind_identity(p, t))
            }
            (TyTerm::Ref(p_mut, p_arg), TyTerm::Ref(t_mut, t_arg)) => {
                p_mut == t_mut && self.bind_arg(p_arg, t_arg)
            }
            (TyTerm::Object(_) | TyTerm::Enum { .. }, _) => {
                Bindings::default().apply(pattern).is_some_and(|closed| closed == *ty)
            }
            _ => false,
        }
    }

    fn bind_arg(&mut self, pattern: &TypeArg<crate::ty::Poly>, arg: &TypeArg<Concrete>) -> bool {
        if let TypeArg::Open(repr, _) = pattern {
            let held = arg.repr();
            match self.reprs.get(repr) {
                Some(bound) if *bound != held => return false,
                Some(_) => {}
                None => {
                    self.reprs.insert(*repr, held);
                }
            }
        }
        self.bind(&pattern.ty(), &arg.ty())
    }

    fn bind_effect(&mut self, pattern: &EffectTerm<crate::ty::Poly>, effect: &EffectTerm<Concrete>) -> bool {
        match pattern {
            EffectTerm::Known(known) => known == effect.get(),
            EffectTerm::Var(v) => match self.effects.get(v) {
                Some(bound) => bound == effect,
                None => {
                    self.effects.insert(*v, effect.clone());
                    true
                }
            },
        }
    }

    fn bind_identity(
        &mut self,
        pattern: &IdentityTerm<crate::ty::Poly>,
        identity: &IdentityTerm<Concrete>,
    ) -> bool {
        match pattern {
            IdentityTerm::Known(known) => *known == identity.get(),
            IdentityTerm::Var(v) => match self.identities.get(v) {
                Some(bound) => bound == identity,
                None => {
                    self.identities.insert(*v, *identity);
                    true
                }
            },
        }
    }

    fn bind_len(&mut self, pattern: &LenTerm<crate::ty::Poly>, len: &LenTerm<Concrete>) -> bool {
        match pattern {
            LenTerm::Known(known) => *known == len.get(),
            LenTerm::Var(v) => match self.lens.get(v) {
                Some(bound) => bound == len,
                None => {
                    self.lens.insert(*v, *len);
                    true
                }
            },
        }
    }

    pub(super) fn apply(&self, pattern: &PolyTy) -> Option<Ty> {
        pattern
            .try_map::<Concrete, ()>(
                &mut |v| self.tys.get(&v).cloned().ok_or(()),
                &mut |v| self.identities.get(&v).copied().ok_or(()),
                &mut |v| self.effects.get(&v).cloned().ok_or(()),
                &mut |v| self.lens.get(&v).copied().ok_or(()),
                &mut |v| Ok(self.reprs.get(&v).cloned().unwrap_or_else(unfixed_repr)),
                &mut |v: Infallible| -> Result<FlowTerm<Concrete>, ()> { match v {} },
            )
            .ok()
    }
}
