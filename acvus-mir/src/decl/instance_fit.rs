//! Whether an instance's type is its shared signature's type at some choice
//! of the signature's variables (extern-fn §6).

use rustc_hash::{FxHashMap, FxHashSet};

use crate::ty::{EffectArg, EffectTerm, IdentityTerm, LenTerm, ObjectTy, Poly, TyId, TyKind, TypeArg, Types};

/// Whether `instance` is `signature` with each of the signature's type,
/// effect, length, identity and mark variables replaced by one choice.
///
/// Both are declared function types. Their parameter names are not part of
/// the comparison, and neither are their own flows: `verify` writes each
/// declared signature's flows from that signature's own lifetimes. The
/// flows of a function type inside either one are compared.
pub(super) fn is_instance_of(types: &Types<Poly>, signature: TyId<Poly>, instance: TyId<Poly>) -> bool {
    let mut choice = Choice {
        types,
        tys: FxHashMap::default(),
        effects: FxHashMap::default(),
        lens: FxHashMap::default(),
        identities: FxHashMap::default(),
        marks: FxHashMap::default(),
        matched: FxHashSet::default(),
    };
    match (types.kind(signature), types.kind(instance)) {
        (
            TyKind::Fn {
                params: signature_params,
                ret: signature_ret,
                effect: signature_effect,
                calls: signature_calls,
                holds: signature_holds,
                flows: _,
            },
            TyKind::Fn {
                params,
                ret,
                effect,
                calls,
                holds,
                flows: _,
            },
        ) => {
            signature_params.len() == params.len()
                && signature_calls == calls
                && signature_holds == holds
                && choice.effect(signature_effect, effect)
                && signature_params
                    .iter()
                    .zip(params.iter())
                    .all(|(signature_param, param)| choice.ty(*signature_param, *param))
                && choice.ty(*signature_ret, *ret)
        }
        _ => false,
    }
}

/// The mark an argument carries, as a signature's mark variable is chosen.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Mark {
    Uniform,
    Specialized,
    Open(u32),
}

/// One choice of the signature's variables, fixed as matching finds them.
struct Choice<'t> {
    types: &'t Types<Poly>,
    tys: FxHashMap<u32, TyId<Poly>>,
    effects: FxHashMap<u32, EffectTerm<Poly>>,
    lens: FxHashMap<u32, LenTerm<Poly>>,
    identities: FxHashMap<u32, IdentityTerm<Poly>>,
    marks: FxHashMap<u32, Mark>,
    /// Pairs already matched. A choice only grows, so a pair that matched
    /// once matches again, and a type read as a tree is matched once per
    /// shared part.
    matched: FxHashSet<(TyId<Poly>, TyId<Poly>)>,
}

/// Whether `var` is chosen as `value`: it is, if nothing chose it yet.
fn choose<V>(chosen: &mut FxHashMap<u32, V>, var: u32, value: V) -> bool
where
    V: PartialEq,
{
    match chosen.get(&var) {
        Some(earlier) => *earlier == value,
        None => {
            chosen.insert(var, value);
            true
        }
    }
}

fn all_pairs<T, F>(signature: &[T], instance: &[T], mut each: F) -> bool
where
    F: FnMut(&T, &T) -> bool,
{
    signature.len() == instance.len() && signature.iter().zip(instance).all(|(a, b)| each(a, b))
}

impl Choice<'_> {
    fn ty(&mut self, signature: TyId<Poly>, instance: TyId<Poly>) -> bool {
        acvus_utils::grow(|| {
            if self.matched.contains(&(signature, instance)) {
                return true;
            }
            let fits = self.kind(signature, instance);
            if fits {
                self.matched.insert((signature, instance));
            }
            fits
        })
    }

    fn kind(&mut self, signature: TyId<Poly>, instance: TyId<Poly>) -> bool {
        let types = self.types;
        match (types.kind(signature), types.kind(instance)) {
            (TyKind::Var(var), _) => choose(&mut self.tys, *var, instance),
            (
                TyKind::Int(_)
                | TyKind::Float
                | TyKind::Char
                | TyKind::String
                | TyKind::Bool
                | TyKind::Unit
                | TyKind::Never
                | TyKind::Order
                | TyKind::Str,
                other,
            ) => types.kind(signature) == other,
            (TyKind::Handle(a), TyKind::Handle(b))
            | (TyKind::Slice(a), TyKind::Slice(b))
            | (TyKind::Option(a), TyKind::Option(b)) => self.ty(*a, *b),
            (TyKind::Array(a, a_len), TyKind::Array(b, b_len)) => self.len(a_len, b_len) && self.ty(*a, *b),
            (TyKind::Tuple(a), TyKind::Tuple(b)) => all_pairs(a, b, |a, b| self.ty(*a, *b)),
            (TyKind::Result(a_ok, a_err), TyKind::Result(b_ok, b_err)) => {
                self.ty(*a_ok, *b_ok) && self.ty(*a_err, *b_err)
            }
            (TyKind::Object(a), TyKind::Object(b)) => {
                let same_object = match (a, b) {
                    (ObjectTy::Structural(_, a_label), ObjectTy::Structural(_, b_label)) => a_label == b_label,
                    (ObjectTy::Declared(a_id, _), ObjectTy::Declared(b_id, _)) => a_id == b_id,
                    (ObjectTy::Structural(..), ObjectTy::Declared(..))
                    | (ObjectTy::Declared(..), ObjectTy::Structural(..)) => false,
                };
                same_object
                    && all_pairs(a.fields(), b.fields(), |(a_name, a), (b_name, b)| {
                        a_name == b_name && self.ty(*a, *b)
                    })
            }
            (
                TyKind::Enum {
                    name: a_name,
                    variants: a_variants,
                    label: a_label,
                },
                TyKind::Enum {
                    name: b_name,
                    variants: b_variants,
                    label: b_label,
                },
            ) => {
                a_name == b_name
                    && a_label == b_label
                    && all_pairs(a_variants, b_variants, |(a_tag, a), (b_tag, b)| {
                        a_tag == b_tag
                            && match (a, b) {
                                (Some(a), Some(b)) => self.ty(*a, *b),
                                (None, None) => true,
                                (Some(_), None) | (None, Some(_)) => false,
                            }
                    })
            }
            (
                TyKind::Fn {
                    params: a_params,
                    ret: a_ret,
                    effect: a_effect,
                    flows: a_flows,
                    calls: a_calls,
                    holds: a_holds,
                },
                TyKind::Fn {
                    params: b_params,
                    ret: b_ret,
                    effect: b_effect,
                    flows: b_flows,
                    calls: b_calls,
                    holds: b_holds,
                },
            ) => {
                a_flows == b_flows
                    && a_calls == b_calls
                    && a_holds == b_holds
                    && self.effect(a_effect, b_effect)
                    && all_pairs(a_params, b_params, |a, b| self.ty(*a, *b))
                    && self.ty(*a_ret, *b_ret)
            }
            (
                TyKind::UserDefined {
                    id: a_id,
                    type_args: a_types,
                    effect_args: a_effects,
                    identity_args: a_identities,
                },
                TyKind::UserDefined {
                    id: b_id,
                    type_args: b_types,
                    effect_args: b_effects,
                    identity_args: b_identities,
                },
            ) => {
                a_id == b_id
                    && all_pairs(a_types, b_types, |a, b| self.type_arg(a, b))
                    && all_pairs(a_effects, b_effects, |a, b| self.effect_arg(a, b))
                    && all_pairs(a_identities, b_identities, |a, b| self.identity(a, b))
            }
            (TyKind::Ref(a_mutability, a), TyKind::Ref(b_mutability, b)) => {
                a_mutability == b_mutability && self.type_arg(a, b)
            }
            (
                TyKind::Handle(_)
                | TyKind::Array(..)
                | TyKind::Slice(_)
                | TyKind::Tuple(_)
                | TyKind::Option(_)
                | TyKind::Result(..)
                | TyKind::Object(_)
                | TyKind::Enum { .. }
                | TyKind::Fn { .. }
                | TyKind::UserDefined { .. }
                | TyKind::Ref(..),
                _,
            ) => false,
        }
    }

    fn type_arg(&mut self, signature: &TypeArg<Poly>, instance: &TypeArg<Poly>) -> bool {
        let mark = match instance {
            TypeArg::Uniform(_) => Mark::Uniform,
            TypeArg::Specialized(_) => Mark::Specialized,
            TypeArg::Open(var, _) => Mark::Open(*var),
        };
        let marks_fit = match signature {
            TypeArg::Open(var, _) => choose(&mut self.marks, *var, mark),
            TypeArg::Uniform(_) => mark == Mark::Uniform,
            TypeArg::Specialized(_) => mark == Mark::Specialized,
        };
        marks_fit && self.ty(signature.ty(), instance.ty())
    }

    fn effect_arg(&mut self, signature: &EffectArg<Poly>, instance: &EffectArg<Poly>) -> bool {
        let marks_fit = matches!(
            (signature, instance),
            (EffectArg::Uniform(_), EffectArg::Uniform(_)) | (EffectArg::Specialized(_), EffectArg::Specialized(_))
        );
        marks_fit && self.effect(signature.effect(), instance.effect())
    }

    fn effect(&mut self, signature: &EffectTerm<Poly>, instance: &EffectTerm<Poly>) -> bool {
        match signature {
            EffectTerm::Var(var) => choose(&mut self.effects, *var, instance.clone()),
            EffectTerm::Known(_) => signature == instance,
        }
    }

    fn len(&mut self, signature: &LenTerm<Poly>, instance: &LenTerm<Poly>) -> bool {
        match signature {
            LenTerm::Var(var) => choose(&mut self.lens, *var, *instance),
            LenTerm::Known(_) => signature == instance,
        }
    }

    fn identity(&mut self, signature: &IdentityTerm<Poly>, instance: &IdentityTerm<Poly>) -> bool {
        match signature {
            IdentityTerm::Var(var) => choose(&mut self.identities, *var, *instance),
            IdentityTerm::Known(_) => signature == instance,
        }
    }
}
