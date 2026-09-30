//! Whether an impl's type is its generic function's type at some choice
//! of the generic function's variables (extern-fn §6).

use rustc_hash::{FxHashMap, FxHashSet};

use crate::ty::{EffectTerm, IdentityTerm, LenTerm, Poly, TyId, TyKind, Types};

/// Whether `impl` is `generic` with each of the generic function's type,
/// effect, length, identity and mark variables replaced by one choice.
///
/// Both are declared function types. Their own flows are not part of the
/// comparison: `verify` writes each declared signature's flows from that
/// signature's own lifetimes. The
/// flows of a function type inside either one are compared.
pub(super) fn is_impl_of(types: &Types<Poly>, generic: TyId<Poly>, r#impl: TyId<Poly>) -> bool {
    let mut choice = Choice {
        types,
        tys: FxHashMap::default(),
        effects: FxHashMap::default(),
        lens: FxHashMap::default(),
        identities: FxHashMap::default(),
        matched: FxHashSet::default(),
    };
    match (types.kind(generic), types.kind(r#impl)) {
        (
            TyKind::Fn {
                params: generic_params,
                ret: generic_ret,
                effect: generic_effect,
                calls: generic_calls,
                holds: generic_holds,
                values: (),
                flows: _,
            },
            TyKind::Fn {
                params,
                ret,
                effect,
                calls,
                holds,
                values: (),
                flows: _,
            },
        ) => {
            generic_params.len() == params.len()
                && generic_calls == calls
                && generic_holds == holds
                && choice.effect(generic_effect, effect)
                && generic_params
                    .iter()
                    .zip(params.iter())
                    .all(|(generic_param, param)| choice.ty(*generic_param, *param))
                && choice.ty(*generic_ret, *ret)
        }
        _ => false,
    }
}

/// One choice of the generic function's variables, fixed as matching finds them.
struct Choice<'t> {
    types: &'t Types<Poly>,
    tys: FxHashMap<u32, TyId<Poly>>,
    effects: FxHashMap<u32, EffectTerm<Poly>>,
    lens: FxHashMap<u32, LenTerm<Poly>>,
    identities: FxHashMap<u32, IdentityTerm<Poly>>,
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

fn all_pairs<T, F>(generic: &[T], r#impl: &[T], mut each: F) -> bool
where
    F: FnMut(&T, &T) -> bool,
{
    generic.len() == r#impl.len() && generic.iter().zip(r#impl).all(|(a, b)| each(a, b))
}

impl Choice<'_> {
    fn ty(&mut self, generic: TyId<Poly>, r#impl: TyId<Poly>) -> bool {
        acvus_utils::grow(|| {
            if self.matched.contains(&(generic, r#impl)) {
                return true;
            }
            let fits = self.kind(generic, r#impl);
            if fits {
                self.matched.insert((generic, r#impl));
            }
            fits
        })
    }

    fn kind(&mut self, generic: TyId<Poly>, r#impl: TyId<Poly>) -> bool {
        let types = self.types;
        match (types.kind(generic), types.kind(r#impl)) {
            (TyKind::Var(var), _) => choose(&mut self.tys, *var, r#impl),
            (
                TyKind::Int(_)
                | TyKind::Float
                | TyKind::Char
                | TyKind::String
                | TyKind::Bool
                | TyKind::Unit
                | TyKind::Never
                | TyKind::Order
                | TyKind::Closures(_)
                | TyKind::Str,
                other,
            ) => types.kind(generic) == other,
            (TyKind::Handle(a), TyKind::Handle(b))
            | (TyKind::Slice(a), TyKind::Slice(b))
            | (TyKind::Option(a, _), TyKind::Option(b, _)) => self.ty(*a, *b),
            (TyKind::Array(a, a_len, _), TyKind::Array(b, b_len, _)) => self.len(a_len, b_len) && self.ty(*a, *b),
            (TyKind::Tuple(a, _), TyKind::Tuple(b, _)) => all_pairs(a, b, |a, b| self.ty(*a, *b)),
            (TyKind::Result(a_ok, a_err, _), TyKind::Result(b_ok, b_err, _)) => {
                self.ty(*a_ok, *b_ok) && self.ty(*a_err, *b_err)
            }
            (TyKind::Object { fields: a, label: a_label }, TyKind::Object { fields: b, label: b_label }) => {
                let same_object = a_label == b_label;
                same_object
                    && all_pairs(a, b, |(a_name, a), (b_name, b)| {
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
                    values: (),
                },
                TyKind::Fn {
                    params: b_params,
                    ret: b_ret,
                    effect: b_effect,
                    flows: b_flows,
                    calls: b_calls,
                    holds: b_holds,
                    values: (),
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
                    effect: a_effect,
                    identity: a_identity,
                    marked: a_marked,
                },
                TyKind::UserDefined {
                    id: b_id,
                    type_args: b_types,
                    effect: b_effect,
                    identity: b_identity,
                    marked: b_marked,
                },
            ) => {
                a_id == b_id
                    && a_marked == b_marked
                    && all_pairs(a_types, b_types, |a, b| self.ty(*a, *b))
                    && all_pairs(a_effect.as_slice(), b_effect.as_slice(), |a, b| self.effect(a, b))
                    && all_pairs(a_identity.as_slice(), b_identity.as_slice(), |a, b| self.identity(a, b))
            }
            (TyKind::Ref(a_mutability, a), TyKind::Ref(b_mutability, b)) => {
                a_mutability == b_mutability && self.ty(*a, *b)
            }
            (
                TyKind::Handle(_)
                | TyKind::Array(..)
                | TyKind::Slice(_)
                | TyKind::Tuple(.., _)
                | TyKind::Option(.., _)
                | TyKind::Result(..)
                | TyKind::Object { .. }
                | TyKind::Enum { .. }
                | TyKind::Fn { .. }
                | TyKind::UserDefined { .. }
                | TyKind::Ref(..),
                _,
            ) => false,
        }
    }

    fn effect(&mut self, generic: &EffectTerm<Poly>, r#impl: &EffectTerm<Poly>) -> bool {
        match generic {
            EffectTerm::Var(var) => choose(&mut self.effects, *var, r#impl.clone()),
            EffectTerm::Known(_) => generic == r#impl,
        }
    }

    fn len(&mut self, generic: &LenTerm<Poly>, r#impl: &LenTerm<Poly>) -> bool {
        match generic {
            LenTerm::Var(var) => choose(&mut self.lens, *var, *r#impl),
            LenTerm::Known(_) => generic == r#impl,
        }
    }

    fn identity(&mut self, generic: &IdentityTerm<Poly>, r#impl: &IdentityTerm<Poly>) -> bool {
        match generic {
            IdentityTerm::Var(var) => choose(&mut self.identities, *var, *r#impl),
            IdentityTerm::Known(_) => generic == r#impl,
        }
    }
}
