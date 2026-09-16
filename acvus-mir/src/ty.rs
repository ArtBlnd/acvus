use std::collections::BTreeSet;
use std::convert::Infallible;
use std::fmt;

use crate::graph::types::QualifiedRef;
use acvus_utils::{Astr, Interner};
use rustc_hash::FxHashMap;

// -- UserDefined type system ------------------------------------------

/// Declaration of a user-defined type - the **single source of truth**
/// for parameter count and constraints. Registered once, referenced by QualifiedRef everywhere.
#[derive(Debug, Clone)]
pub struct UserDefinedDecl {
    pub qref: QualifiedRef,
    pub type_params: Vec<TyVarBound>,
    pub effect_params: usize,
    /// Identity parameters, at most one: a value of a type with one is a
    /// distinct source, and the type is move-only.
    pub identity_params: usize,
}

/// The width and signedness of an integer type: `i8` to `i64`, `u8` to
/// `u64`. An integer literal takes the width its use demands and is `i64`
/// where nothing demands one (RFC-0037).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum IntTy {
    I8,
    I16,
    I32,
    I64,
    U8,
    U16,
    U32,
    U64,
}

impl IntTy {
    pub const ALL: [IntTy; 8] = [
        IntTy::I8,
        IntTy::I16,
        IntTy::I32,
        IntTy::I64,
        IntTy::U8,
        IntTy::U16,
        IntTy::U32,
        IntTy::U64,
    ];

    pub fn bits(self) -> u32 {
        match self {
            IntTy::I8 | IntTy::U8 => 8,
            IntTy::I16 | IntTy::U16 => 16,
            IntTy::I32 | IntTy::U32 => 32,
            IntTy::I64 | IntTy::U64 => 64,
        }
    }

    pub fn bytes(self) -> usize {
        self.bits() as usize / 8
    }

    pub fn signed(self) -> bool {
        matches!(self, IntTy::I8 | IntTy::I16 | IntTy::I32 | IntTy::I64)
    }

    pub fn name(self) -> &'static str {
        match self {
            IntTy::I8 => "i8",
            IntTy::I16 => "i16",
            IntTy::I32 => "i32",
            IntTy::I64 => "i64",
            IntTy::U8 => "u8",
            IntTy::U16 => "u16",
            IntTy::U32 => "u32",
            IntTy::U64 => "u64",
        }
    }

    pub fn min(self) -> i128 {
        if self.signed() {
            -(1i128 << (self.bits() - 1))
        } else {
            0
        }
    }

    pub fn max(self) -> i128 {
        if self.signed() {
            (1i128 << (self.bits() - 1)) - 1
        } else {
            (1i128 << self.bits()) - 1
        }
    }

    /// Whether `value` is representable in this type.
    pub fn holds(self, value: i128) -> bool {
        (self.min()..=self.max()).contains(&value)
    }

    /// The value the low `bits()` of `bits` spell, sign- or zero-extended
    /// to `i128`.
    pub fn read(self, bits: u64) -> i128 {
        match self {
            IntTy::I8 => bits as u8 as i8 as i128,
            IntTy::I16 => bits as u16 as i16 as i128,
            IntTy::I32 => bits as u32 as i32 as i128,
            IntTy::I64 => bits as i64 as i128,
            IntTy::U8 => bits as u8 as i128,
            IntTy::U16 => bits as u16 as i128,
            IntTy::U32 => bits as u32 as i128,
            IntTy::U64 => bits as i128,
        }
    }
}

/// What a declaration says about one of its type variables. The solver
/// carries it on the variable and verifies it when the variable freezes.
#[derive(Debug, Clone, PartialEq)]
pub enum TyVarBound {
    Any,
    /// The variable resolves to a type of one of these shapes (RFC-0027).
    OneOf(Vec<PolyTy>),
    /// The variable is an integer literal's type: one of `among`, which a
    /// use narrows, signed where the literal is negated (RFC-0037).
    Integer {
        signed: bool,
        among: Vec<IntTy>,
    },
}

impl TyVarBound {
    /// An integer bound over the widths `among` yields, signed-only where
    /// `signed`; `None` when no width remains.
    pub fn integer(signed: bool, among: impl Iterator<Item = IntTy>) -> Option<Self> {
        let among: Vec<IntTy> = among.filter(|k| !signed || k.signed()).collect();
        (!among.is_empty()).then_some(Self::Integer { signed, among })
    }

    /// The width an integer literal takes where its uses left more than one:
    /// `i64` when it remains, the only one when one remains, none otherwise.
    pub fn integer_default(&self) -> Option<IntTy> {
        let Self::Integer { among, .. } = self else {
            return None;
        };
        if among.contains(&IntTy::I64) {
            Some(IntTy::I64)
        } else if let [only] = among.as_slice() {
            Some(*only)
        } else {
            None
        }
    }

    pub fn admits(&self, ty: &Ty) -> bool {
        match self {
            Self::Any => true,
            Self::OneOf(shapes) => shapes.iter().any(|s| matches_poly(ty, s)),
            Self::Integer { signed, among } => {
                matches!(ty, Ty::Int(k) if among.contains(k) && (!*signed || k.signed()))
            }
        }
    }

    /// The bound both sides satisfy: the pairwise unifiers of their shapes.
    /// `None` when no type does.
    pub fn meet(&self, other: &Self) -> Option<Self> {
        match (self, other) {
            (Self::Any, b) | (b, Self::Any) => Some(b.clone()),
            (
                Self::Integer {
                    signed: sa,
                    among: aa,
                },
                Self::Integer {
                    signed: sb,
                    among: ab,
                },
            ) => Self::integer(*sa || *sb, aa.iter().filter(|k| ab.contains(k)).copied()),
            (Self::Integer { signed, among }, Self::OneOf(shapes))
            | (Self::OneOf(shapes), Self::Integer { signed, among }) => Self::integer(
                *signed,
                shapes.iter().filter_map(|s| match s {
                    TyTerm::Int(k) if among.contains(k) => Some(*k),
                    _ => None,
                }),
            ),
            (Self::OneOf(a), Self::OneOf(b)) => {
                let both: Vec<PolyTy> = a
                    .iter()
                    .flat_map(|x| b.iter().filter_map(move |y| unify_patterns(x, y)))
                    .collect();
                if both.is_empty() {
                    None
                } else {
                    Some(Self::OneOf(both))
                }
            }
        }
    }
}

/// A polymorphic type with the bounds its variables were declared with.
/// Variable `i` of `ty` has bound `bounds[i]`; a missing entry is `Any`.
/// A shared signature also carries the types of its instances; the solver
/// settles each instantiation on one of them (RFC-0027).
#[derive(Debug, Clone, PartialEq)]
pub struct Scheme {
    pub ty: PolyTy,
    pub bounds: Vec<TyVarBound>,
    pub instances: Vec<PolyTy>,
}

impl Scheme {
    pub fn unbounded(ty: PolyTy) -> Self {
        Self {
            ty,
            bounds: Vec::new(),
            instances: Vec::new(),
        }
    }

    pub fn bound_of(&self, var: u32) -> TyVarBound {
        self.bounds
            .get(var as usize)
            .cloned()
            .unwrap_or(TyVarBound::Any)
    }
}

/// Whether a concrete type has the shape of a polymorphic pattern.
pub fn matches_poly(ty: &Ty, pattern: &PolyTy) -> bool {
    matches_pattern(ty, pattern)
}

/// Whether a type has the shape of a polymorphic pattern. A pattern
/// variable stands for any one term, the same term wherever it recurs;
/// effect, length, and identity variables of the pattern stand for any
/// effect, length, or identity. A variable of `ty` matches only a pattern
/// variable: the pattern is the more general of the two.
pub fn matches_pattern<P>(ty: &TyTerm<P>, pattern: &PolyTy) -> bool
where
    P: Phase + PartialEq,
{
    matches_pattern_with(ty, pattern, Unknowns::Fixed)
}

/// Whether a type whose variables are still open could have the shape of
/// a pattern.
pub fn could_match_pattern<P>(ty: &TyTerm<P>, pattern: &PolyTy) -> bool
where
    P: Phase + PartialEq,
{
    matches_pattern_with(ty, pattern, Unknowns::Open)
}

/// What a variable of the matched type stands for.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Unknowns {
    /// One term, not yet named: it matches only a pattern variable.
    Fixed,
    /// Anything: it matches every pattern.
    Open,
}

fn matches_pattern_with<P>(ty: &TyTerm<P>, pattern: &PolyTy, unknowns: Unknowns) -> bool
where
    P: Phase + PartialEq,
{
    fn effect_matches<P>(
        effect: &EffectTerm<P>,
        pattern: &EffectTerm<Poly>,
        unknowns: Unknowns,
    ) -> bool
    where
        P: Phase + PartialEq,
    {
        match (effect, pattern) {
            (_, EffectTerm::Var(_)) => true,
            (EffectTerm::Var(_), _) => unknowns == Unknowns::Open,
            (EffectTerm::Known(e), EffectTerm::Known(p)) => e == p,
        }
    }
    fn go<P>(
        ty: &TyTerm<P>,
        pat: &PolyTy,
        seen: &mut FxHashMap<u32, TyTerm<P>>,
        unknowns: Unknowns,
    ) -> bool
    where
        P: Phase + PartialEq,
    {
        let open = |t: &TyTerm<P>| unknowns == Unknowns::Open && matches!(t, TyTerm::Var(_));
        match (ty, pat) {
            (_, TyTerm::Var(v)) => match seen.get(v) {
                Some(bound) => bound == ty || open(bound) || open(ty),
                None => {
                    seen.insert(*v, ty.clone());
                    true
                }
            },
            (TyTerm::Var(_), _) => unknowns == Unknowns::Open,
            (TyTerm::Int(a), TyTerm::Int(b)) => a == b,
            (TyTerm::Float, TyTerm::Float)
            | (TyTerm::String, TyTerm::String)
            | (TyTerm::Bool, TyTerm::Bool)
            | (TyTerm::Unit, TyTerm::Unit)
            | (TyTerm::Order, TyTerm::Order) => true,
            (TyTerm::Array(e, n), TyTerm::Array(pe, pn)) => {
                let len_ok = match (n, pn) {
                    (_, LenTerm::Var(_)) => true,
                    (LenTerm::Var(_), _) => unknowns == Unknowns::Open,
                    (LenTerm::Known(n), LenTerm::Known(k)) => n == k,
                };
                len_ok && go(e, pe, seen, unknowns)
            }
            (TyTerm::Option(i), TyTerm::Option(pi)) => go(i, pi, seen, unknowns),
            (TyTerm::Result(t, e), TyTerm::Result(pt, pe)) => {
                go(t, pt, seen, unknowns) && go(e, pe, seen, unknowns)
            }
            (TyTerm::Handle(i), TyTerm::Handle(pi)) => go(i, pi, seen, unknowns),
            (TyTerm::Ref(m, i), TyTerm::Ref(pm, pi)) => m == pm && go(i, pi, seen, unknowns),
            (TyTerm::Tuple(es), TyTerm::Tuple(ps)) => {
                es.len() == ps.len() && es.iter().zip(ps).all(|(e, p)| go(e, p, seen, unknowns))
            }
            (TyTerm::Object(fs), TyTerm::Object(pfs)) => {
                fs.len() == pfs.len()
                    && fs
                        .iter()
                        .all(|(k, v)| pfs.get(k).is_some_and(|pv| go(v, pv, seen, unknowns)))
            }
            (
                TyTerm::Fn {
                    params,
                    ret,
                    effect,
                    ..
                },
                TyTerm::Fn {
                    params: pp,
                    ret: pr,
                    effect: pe,
                    ..
                },
            ) => {
                params.len() == pp.len()
                    && effect_matches(effect, pe, unknowns)
                    && params
                        .iter()
                        .zip(pp)
                        .all(|(a, b)| go(&a.ty, &b.ty, seen, unknowns))
                    && go(ret, pr, seen, unknowns)
            }
            (
                TyTerm::UserDefined {
                    id,
                    type_args,
                    effect_args,
                    identity_args,
                },
                TyTerm::UserDefined {
                    id: pid,
                    type_args: pargs,
                    effect_args: peffects,
                    identity_args: pidentities,
                },
            ) => {
                id == pid
                    && type_args.len() == pargs.len()
                    && effect_args.len() == peffects.len()
                    && identity_args.len() == pidentities.len()
                    && effect_args
                        .iter()
                        .zip(peffects)
                        .all(|(a, b)| effect_matches(a, b, unknowns))
                    && identity_args
                        .iter()
                        .zip(pidentities)
                        .all(|(a, b)| match (a, b) {
                            (_, IdentityTerm::Var(_)) => true,
                            (IdentityTerm::Var(_), _) => unknowns == Unknowns::Open,
                            (IdentityTerm::Known(a), IdentityTerm::Known(k)) => a == k,
                        })
                    && type_args
                        .iter()
                        .zip(pargs)
                        .all(|(a, b)| go(a, b, seen, unknowns))
            }
            (
                TyTerm::Enum { name, variants },
                TyTerm::Enum {
                    name: pn,
                    variants: pv,
                },
            ) => {
                name == pn
                    && variants.len() == pv.len()
                    && variants.iter().all(|(k, v)| match (v, pv.get(k)) {
                        (Some(v), Some(Some(p))) => go(v, p, seen, unknowns),
                        (None, Some(None)) => true,
                        _ => false,
                    })
            }
            _ => false,
        }
    }
    go(ty, pattern, &mut FxHashMap::default(), unknowns)
}

// -- Pattern unification ----------------------------------------------

/// The variables of a pattern, numbered per kind, as the one past the
/// highest of each.
#[derive(Default, Clone, Copy)]
struct VarSpan {
    ty: u32,
    identity: u32,
    effect: u32,
    len: u32,
}

fn var_span(pattern: &PolyTy) -> VarSpan {
    let mut span = VarSpan::default();
    let _ = pattern.map::<Poly>(
        &mut |v| {
            span.ty = span.ty.max(v + 1);
            TyTerm::Var(v)
        },
        &mut |v| {
            span.identity = span.identity.max(v + 1);
            IdentityTerm::Var(v)
        },
        &mut |v| {
            span.effect = span.effect.max(v + 1);
            EffectTerm::Var(v)
        },
        &mut |v| {
            span.len = span.len.max(v + 1);
            LenTerm::Var(v)
        },
    );
    span
}

fn shift_vars(pattern: &PolyTy, by: VarSpan) -> PolyTy {
    pattern.map::<Poly>(
        &mut |v| TyTerm::Var(v + by.ty),
        &mut |v| IdentityTerm::Var(v + by.identity),
        &mut |v| EffectTerm::Var(v + by.effect),
        &mut |v| LenTerm::Var(v + by.len),
    )
}

#[derive(Default)]
struct PatternSubst {
    ty: FxHashMap<u32, PolyTy>,
    identity: FxHashMap<u32, IdentityTerm<Poly>>,
    effect: FxHashMap<u32, EffectTerm<Poly>>,
    len: FxHashMap<u32, LenTerm<Poly>>,
}

impl PatternSubst {
    fn walk(&self, t: &PolyTy) -> PolyTy {
        match t {
            TyTerm::Var(v) => match self.ty.get(v) {
                Some(bound) => self.walk(bound),
                None => t.clone(),
            },
            other => other.clone(),
        }
    }

    fn occurs(&self, v: u32, t: &PolyTy) -> bool {
        let mut found = false;
        let _ = t.map::<Poly>(
            &mut |x| {
                if x == v || self.ty.get(&x).is_some_and(|b| self.occurs(v, b)) {
                    found = true;
                }
                TyTerm::Var(x)
            },
            &mut IdentityTerm::Var,
            &mut EffectTerm::Var,
            &mut LenTerm::Var,
        );
        found
    }

    fn unify_effect(&mut self, a: &EffectTerm<Poly>, b: &EffectTerm<Poly>) -> bool {
        let resolve = |s: &Self, e: &EffectTerm<Poly>| match e {
            EffectTerm::Var(v) => s.effect.get(v).cloned().unwrap_or(e.clone()),
            known => known.clone(),
        };
        match (resolve(self, a), resolve(self, b)) {
            (EffectTerm::Known(x), EffectTerm::Known(y)) => x == y,
            (EffectTerm::Var(x), EffectTerm::Var(y)) if x == y => true,
            (EffectTerm::Var(v), other) | (other, EffectTerm::Var(v)) => {
                self.effect.insert(v, other);
                true
            }
        }
    }

    fn unify_len(&mut self, a: &LenTerm<Poly>, b: &LenTerm<Poly>) -> bool {
        let resolve = |s: &Self, l: &LenTerm<Poly>| match l {
            LenTerm::Var(v) => s.len.get(v).copied().unwrap_or(*l),
            known => *known,
        };
        match (resolve(self, a), resolve(self, b)) {
            (LenTerm::Known(x), LenTerm::Known(y)) => x == y,
            (LenTerm::Var(x), LenTerm::Var(y)) if x == y => true,
            (LenTerm::Var(v), other) | (other, LenTerm::Var(v)) => {
                self.len.insert(v, other);
                true
            }
        }
    }

    fn unify_identity(&mut self, a: &IdentityTerm<Poly>, b: &IdentityTerm<Poly>) -> bool {
        let resolve = |s: &Self, i: &IdentityTerm<Poly>| match i {
            IdentityTerm::Var(v) => s.identity.get(v).copied().unwrap_or(*i),
            known => *known,
        };
        match (resolve(self, a), resolve(self, b)) {
            (IdentityTerm::Known(x), IdentityTerm::Known(y)) => x == y,
            (IdentityTerm::Var(x), IdentityTerm::Var(y)) if x == y => true,
            (IdentityTerm::Var(v), other) | (other, IdentityTerm::Var(v)) => {
                self.identity.insert(v, other);
                true
            }
        }
    }

    fn unify(&mut self, a: &PolyTy, b: &PolyTy) -> bool {
        let (a, b) = (self.walk(a), self.walk(b));
        match (&a, &b) {
            (TyTerm::Var(x), TyTerm::Var(y)) if x == y => true,
            (TyTerm::Var(v), other) | (other, TyTerm::Var(v)) => {
                if self.occurs(*v, other) {
                    return false;
                }
                self.ty.insert(*v, other.clone());
                true
            }
            (TyTerm::Int(a), TyTerm::Int(b)) => a == b,
            (TyTerm::Float, TyTerm::Float)
            | (TyTerm::String, TyTerm::String)
            | (TyTerm::Bool, TyTerm::Bool)
            | (TyTerm::Unit, TyTerm::Unit)
            | (TyTerm::Order, TyTerm::Order) => true,
            (TyTerm::Error(_), _) | (_, TyTerm::Error(_)) => false,
            (TyTerm::Array(ea, la), TyTerm::Array(eb, lb)) => {
                self.unify_len(la, lb) && self.unify(ea, eb)
            }
            (TyTerm::Option(ia), TyTerm::Option(ib)) | (TyTerm::Handle(ia), TyTerm::Handle(ib)) => {
                self.unify(ia, ib)
            }
            (TyTerm::Result(ta, ea), TyTerm::Result(tb, eb)) => {
                self.unify(ta, tb) && self.unify(ea, eb)
            }
            (TyTerm::Ref(ma, ia), TyTerm::Ref(mb, ib)) => ma == mb && self.unify(ia, ib),
            (TyTerm::Tuple(ea), TyTerm::Tuple(eb)) => {
                ea.len() == eb.len() && ea.iter().zip(eb).all(|(x, y)| self.unify(x, y))
            }
            (TyTerm::Object(fa), TyTerm::Object(fb)) => {
                fa.len() == fb.len()
                    && fa
                        .iter()
                        .all(|(k, v)| fb.get(k).is_some_and(|w| self.unify(v, w)))
            }
            (
                TyTerm::Fn {
                    params: pa,
                    ret: ra,
                    captures: ca,
                    effect: ea,
                },
                TyTerm::Fn {
                    params: pb,
                    ret: rb,
                    captures: cb,
                    effect: eb,
                },
            ) => {
                pa.len() == pb.len()
                    && ca.len() == cb.len()
                    && self.unify_effect(ea, eb)
                    && pa.iter().zip(pb).all(|(x, y)| self.unify(&x.ty, &y.ty))
                    && ca.iter().zip(cb).all(|(x, y)| self.unify(x, y))
                    && self.unify(ra, rb)
            }
            (
                TyTerm::UserDefined {
                    id: ia,
                    type_args: ta,
                    effect_args: ea,
                    identity_args: na,
                },
                TyTerm::UserDefined {
                    id: ib,
                    type_args: tb,
                    effect_args: eb,
                    identity_args: nb,
                },
            ) => {
                ia == ib
                    && ta.len() == tb.len()
                    && ea.len() == eb.len()
                    && na.len() == nb.len()
                    && ea.iter().zip(eb).all(|(x, y)| self.unify_effect(x, y))
                    && na.iter().zip(nb).all(|(x, y)| self.unify_identity(x, y))
                    && ta.iter().zip(tb).all(|(x, y)| self.unify(x, y))
            }
            (
                TyTerm::Enum {
                    name: na,
                    variants: va,
                },
                TyTerm::Enum {
                    name: nb,
                    variants: vb,
                },
            ) => {
                na == nb
                    && va.len() == vb.len()
                    && va.iter().all(|(k, v)| match (v, vb.get(k)) {
                        (Some(v), Some(Some(w))) => self.unify(v, w),
                        (None, Some(None)) => true,
                        _ => false,
                    })
            }
            _ => false,
        }
    }

    fn apply(&self, t: &PolyTy) -> PolyTy {
        t.map::<Poly>(
            &mut |v| match self.ty.get(&v) {
                Some(bound) => self.apply(bound),
                None => TyTerm::Var(v),
            },
            &mut |v| {
                self.identity
                    .get(&v)
                    .copied()
                    .unwrap_or(IdentityTerm::Var(v))
            },
            &mut |v| self.effect.get(&v).cloned().unwrap_or(EffectTerm::Var(v)),
            &mut |v| self.len.get(&v).copied().unwrap_or(LenTerm::Var(v)),
        )
    }
}

/// The most general pattern that has the shape of both, with the two
/// patterns' variables kept apart; `None` when no type has.
pub fn unify_patterns(a: &PolyTy, b: &PolyTy) -> Option<PolyTy> {
    let b = shift_vars(b, var_span(a));
    let mut subst = PatternSubst::default();
    subst.unify(a, &b).then(|| subst.apply(a))
}

/// Immutable registry of all UserDefined type declarations and ExternCast rules.
/// Built once at setup, then frozen via `Freeze<TypeRegistry>` and shared everywhere.
///
/// Contains:
/// - `decls`: UserDefined type declarations (source of truth for params/constraints).
/// - `cast_rules`: ExternCast coercion rules (UserDefined -> other type).
#[derive(Debug, Clone, Default)]
pub struct TypeRegistry {
    decls: FxHashMap<QualifiedRef, UserDefinedDecl>,
    /// ExternCast coercion rules, indexed by source UserDefined QualifiedRef.
    /// `from_rules`: keyed by the `from` type's QualifiedRef.
    /// `to_rules`: keyed by the `to` type's QualifiedRef (when target is UserDefined).
    // pub(crate) for test access.
    pub(crate) from_rules: FxHashMap<QualifiedRef, Vec<CastRule>>,
    pub(crate) to_rules: FxHashMap<QualifiedRef, Vec<CastRule>>,
}

/// A coercion rule: `from` can be implicitly converted to `to`.
/// Both `from` and `to` share positional Var placeholders (Poly phase),
/// so instantiating them together links corresponding parameters.
#[derive(Debug, Clone)]
pub struct CastRule {
    /// Source type pattern (must be UserDefined). May contain positional Var placeholders.
    pub from: PolyTy,
    /// Target type pattern (Var placeholders shared with `from`).
    pub to: PolyTy,
    /// The pure ExternFn that performs the conversion.
    pub fn_ref: QualifiedRef,
}

/// Head constructor of a type - used for duplicate cast rule detection.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum TyHead {
    Int(IntTy),
    Float,
    String,
    Bool,
    Unit,
    Order,
    Array,
    Object,
    Tuple,
    Fn,
    Option,
    Result,
    Enum,
    Handle,
    Ref,
    UserDefined(QualifiedRef),
    Error,
}

fn ty_head<V: Phase>(ty: &TyTerm<V>) -> TyHead {
    match ty {
        TyTerm::Int(k) => TyHead::Int(*k),
        TyTerm::Float => TyHead::Float,
        TyTerm::String => TyHead::String,
        TyTerm::Bool => TyHead::Bool,
        TyTerm::Unit => TyHead::Unit,
        TyTerm::Order => TyHead::Order,
        TyTerm::Array(..) => TyHead::Array,
        TyTerm::Object(_) => TyHead::Object,
        TyTerm::Tuple(_) => TyHead::Tuple,
        TyTerm::Fn { .. } => TyHead::Fn,
        TyTerm::Option(_) => TyHead::Option,
        TyTerm::Result(..) => TyHead::Result,
        TyTerm::Enum { .. } => TyHead::Enum,
        TyTerm::Handle(..) => TyHead::Handle,
        TyTerm::Ref(..) => TyHead::Ref,
        TyTerm::UserDefined { id, .. } => TyHead::UserDefined(*id),
        TyTerm::Error(_) => TyHead::Error,
        TyTerm::Var(_) => TyHead::Error,
    }
}

impl TypeRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    // -- Type declarations -------------------------------------------

    /// Register a declaration. Panics on duplicate qref.
    pub fn register(&mut self, decl: UserDefinedDecl) {
        let qref = decl.qref;
        assert!(
            decl.identity_params <= 1,
            "{qref:?}: a type has at most one identity parameter; a value is one source"
        );
        let prev = self.decls.insert(qref, decl);
        assert!(prev.is_none(), "duplicate UserDefined type: {qref:?}");
    }

    /// Look up a declaration by qref. Panics if not found - missing decl is a bug.
    pub fn get(&self, qref: QualifiedRef) -> &UserDefinedDecl {
        self.decls
            .get(&qref)
            .unwrap_or_else(|| panic!("unknown UserDefined type: {qref:?}"))
    }

    pub fn iter(&self) -> impl Iterator<Item = (&QualifiedRef, &UserDefinedDecl)> {
        self.decls.iter()
    }

    // -- Cast rules --------------------------------------------------

    /// Register a cast rule. Indexes by `from`'s QualifiedRef (if UserDefined)
    /// and by `to`'s QualifiedRef (if UserDefined). At least one side must be UserDefined.
    pub fn register_cast(&mut self, rule: CastRule) {
        let from_qref = match &rule.from {
            TyTerm::UserDefined { id, .. } => Some(*id),
            _ => None,
        };
        let to_qref = match &rule.to {
            TyTerm::UserDefined { id, .. } => Some(*id),
            _ => None,
        };
        assert!(
            from_qref.is_some() || to_qref.is_some(),
            "CastRule: at least one side must be UserDefined"
        );

        // Duplicate check (same from head + same to head).
        let from_head = ty_head(&rule.from);
        let to_head = ty_head(&rule.to);
        if let Some(fq) = from_qref {
            for existing in self.from_rules.get(&fq).into_iter().flatten() {
                assert!(
                    !(ty_head(&existing.from) == from_head && ty_head(&existing.to) == to_head),
                    "duplicate CastRule: same from and to head constructor"
                );
            }
        }

        // Index by from (if UserDefined).
        if let Some(fq) = from_qref {
            self.from_rules.entry(fq).or_default().push(rule.clone());
        }
        // Index by to (if UserDefined).
        if let Some(tq) = to_qref {
            self.to_rules.entry(tq).or_default().push(rule);
        }
    }

    /// Get all cast rules where `from` matches the given QualifiedRef.
    pub fn rules_from(&self, qref: QualifiedRef) -> &[CastRule] {
        self.from_rules.get(&qref).map_or(&[], |v| v.as_slice())
    }

    /// Get all cast rules where `to` is a UserDefined matching the given QualifiedRef.
    pub fn rules_to(&self, qref: QualifiedRef) -> &[CastRule] {
        self.to_rules.get(&qref).map_or(&[], |v| v.as_slice())
    }
}

/// A named, typed function parameter.
pub type Param = ParamTerm<Concrete>;

/// Token for `Ty::Error` construction.
///
/// `Ty::Error` is a **poison type** - it suppresses cascading errors by unifying
/// with anything. Permitted uses:
///
/// - **Type checker / compiler**: After reporting a type error, return `Ty::error()`
///   so compilation continues and collects all errors (not just the first one).
/// - **Deserialization recovery**: When loading a persisted type that can't be parsed.
///
/// **Forbidden uses**:
///
/// - As a "don't know" placeholder (use the actual type instead).
/// - As a default/fallback when you're too lazy to propagate the real type.
/// - In runtime code paths - Error must never appear in a running program's types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ErrorToken(());

impl ErrorToken {
    pub(crate) fn new() -> Self {
        Self(())
    }
}

/// Polarity for subtyping direction in unification.
///
/// - `Covariant`: `a <= b` - `a` may be a subtype of `b`.
/// - `Contravariant`: `b <= a` - reversed direction (e.g. function parameters).
/// - `Invariant`: `a = b` - no subtyping allowed, must be exactly equal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Polarity {
    Covariant,
    Contravariant,
    Invariant,
}

impl Polarity {
    /// Flip polarity: Covariant <-> Contravariant, Invariant stays.
    pub fn flip(self) -> Self {
        match self {
            Polarity::Covariant => Polarity::Contravariant,
            Polarity::Contravariant => Polarity::Covariant,
            Polarity::Invariant => Polarity::Invariant,
        }
    }
}

/// Whether a call may be issued again (RFC-0014). The
/// derived order is the chain `Pure < Idempotent < Opaque`.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize, serde::Deserialize,
)]
pub enum Reissue {
    Pure,
    Idempotent,
    Opaque,
}

/// The contexts a call may touch (RFC-0017), as a set of context names.
pub type Contexts = BTreeSet<QualifiedRef>;

/// The effect of a call (RFC-0013, RFC-0017): the reissue chain, whether
/// two calls commute, and the contexts the call may read and write. A
/// Pure call that writes nothing commutes by definition; a call that
/// writes a context never commutes, since a second call may read it.
/// Every constructor keeps both invariants.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Effect {
    pub reissue: Reissue,
    pub commutes: bool,
    pub reads: Contexts,
    pub writes: Contexts,
}

impl Effect {
    pub const PURE: Effect = Effect {
        reissue: Reissue::Pure,
        commutes: true,
        reads: BTreeSet::new(),
        writes: BTreeSet::new(),
    };
    pub const IDEMPOTENT: Effect = Effect {
        reissue: Reissue::Idempotent,
        commutes: false,
        reads: BTreeSet::new(),
        writes: BTreeSet::new(),
    };
    pub const OPAQUE: Effect = Effect {
        reissue: Reissue::Opaque,
        commutes: false,
        reads: BTreeSet::new(),
        writes: BTreeSet::new(),
    };

    pub fn new(reissue: Reissue, commutes: bool) -> Effect {
        Effect::with_contexts(reissue, commutes, Contexts::new(), Contexts::new())
    }

    pub fn with_contexts(
        reissue: Reissue,
        commutes: bool,
        reads: Contexts,
        writes: Contexts,
    ) -> Effect {
        Effect {
            reissue,
            commutes: (commutes || reissue == Reissue::Pure) && writes.is_empty(),
            reads,
            writes,
        }
    }

    /// The effect of reading one context: Pure, and a read of it.
    pub fn read(context: QualifiedRef) -> Effect {
        Effect::with_contexts(
            Reissue::Pure,
            true,
            Contexts::from([context]),
            Contexts::new(),
        )
    }

    /// The effect of writing one context: Pure on the chain, and a write
    /// of it, which does not commute.
    pub fn write(context: QualifiedRef) -> Effect {
        Effect::with_contexts(
            Reissue::Pure,
            true,
            Contexts::new(),
            Contexts::from([context]),
        )
    }

    /// The same level, declared to commute.
    pub fn commutative(&self) -> Effect {
        Effect::with_contexts(self.reissue, true, self.reads.clone(), self.writes.clone())
    }

    pub fn is_pure(&self) -> bool {
        self.reissue == Reissue::Pure
    }

    /// No level above Pure and no context touched: the effect a type
    /// display leaves out.
    pub fn is_empty(&self) -> bool {
        self.is_pure() && self.reads.is_empty() && self.writes.is_empty()
    }

    /// Whether the call may touch `context` at all.
    pub fn touches(&self, context: QualifiedRef) -> bool {
        self.reads.contains(&context) || self.writes.contains(&context)
    }

    /// The product order on the chain and commutativity: `self` is no
    /// more effectful than `other` when it is no higher on the chain and
    /// commutes whenever `other` does. The context sets are not part of
    /// this order: a bound on an effect bounds its level, and the sets
    /// only accumulate through `join`. `touches_at_most` orders the sets.
    pub fn at_most(&self, other: &Effect) -> bool {
        self.reissue <= other.reissue && (self.commutes || !other.commutes)
    }

    /// Whether `self` touches no context `other` does not.
    pub fn touches_at_most(&self, other: &Effect) -> bool {
        self.reads.is_subset(&other.reads) && self.writes.is_subset(&other.writes)
    }

    /// The least effect above both: the higher level, commutative only if
    /// both are, touching what either does.
    pub fn join(&self, other: &Effect) -> Effect {
        Effect::with_contexts(
            self.reissue.max(other.reissue),
            self.commutes && other.commutes,
            self.reads.union(&other.reads).copied().collect(),
            self.writes.union(&other.writes).copied().collect(),
        )
    }

    /// The greatest effect below both.
    pub fn meet(&self, other: &Effect) -> Effect {
        Effect::with_contexts(
            self.reissue.min(other.reissue),
            self.commutes || other.commutes,
            self.reads.intersection(&other.reads).copied().collect(),
            self.writes.intersection(&other.writes).copied().collect(),
        )
    }
}

impl PartialOrd for Effect {
    fn partial_cmp(&self, other: &Effect) -> Option<std::cmp::Ordering> {
        use std::cmp::Ordering;
        match (self.at_most(other), other.at_most(self)) {
            (true, true) => Some(Ordering::Equal),
            (true, false) => Some(Ordering::Less),
            (false, true) => Some(Ordering::Greater),
            (false, false) => None,
        }
    }
}

impl fmt::Display for Effect {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self.reissue)?;
        if self.commutes && !self.is_pure() {
            write!(f, "+commutative")?;
        }
        Ok(())
    }
}

/// A required effect level that exceeds the level allowed at that point.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EffectConflict {
    pub required: Effect,
    pub allowed: Effect,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum EffectTerm<V: Phase> {
    Known(Effect),
    Var(V::EffectVar),
}

impl<V: Phase> From<Effect> for EffectTerm<V> {
    fn from(effect: Effect) -> Self {
        EffectTerm::Known(effect)
    }
}

impl<V: Phase> EffectTerm<V> {
    pub fn map<W: Phase>(
        &self,
        on_effect: &mut impl FnMut(V::EffectVar) -> EffectTerm<W>,
    ) -> EffectTerm<W> {
        match self {
            EffectTerm::Known(e) => EffectTerm::Known(e.clone()),
            EffectTerm::Var(v) => on_effect(*v),
        }
    }

    pub fn try_map<W: Phase, E>(
        &self,
        on_effect: &mut impl FnMut(V::EffectVar) -> Result<EffectTerm<W>, E>,
    ) -> Result<EffectTerm<W>, E> {
        match self {
            EffectTerm::Known(e) => Ok(EffectTerm::Known(e.clone())),
            EffectTerm::Var(v) => on_effect(*v),
        }
    }
}

impl EffectTerm<Concrete> {
    pub fn get(&self) -> &Effect {
        match self {
            EffectTerm::Known(e) => e,
            EffectTerm::Var(v) => match *v {},
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LenTerm<V: Phase> {
    Known(usize),
    Var(V::LenVar),
}

impl<V: Phase> LenTerm<V> {
    pub fn map<W: Phase>(&self, on_len: &mut impl FnMut(V::LenVar) -> LenTerm<W>) -> LenTerm<W> {
        match self {
            LenTerm::Known(n) => LenTerm::Known(*n),
            LenTerm::Var(v) => on_len(*v),
        }
    }

    pub fn try_map<W: Phase, E>(
        &self,
        on_len: &mut impl FnMut(V::LenVar) -> Result<LenTerm<W>, E>,
    ) -> Result<LenTerm<W>, E> {
        match self {
            LenTerm::Known(n) => Ok(LenTerm::Known(*n)),
            LenTerm::Var(v) => on_len(*v),
        }
    }
}

impl LenTerm<Concrete> {
    pub fn get(&self) -> usize {
        match self {
            LenTerm::Known(n) => *n,
            LenTerm::Var(v) => match *v {},
        }
    }
}

// -- Identity system --------------------------------------------------

acvus_utils::declare_local_id!(pub IdentityId);

/// An identity argument of a user-defined type: the source a value came
/// from. Two values unify only when their identities are the same, so a
/// value from one source never mixes with a value from another.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum IdentityTerm<V: Phase> {
    Known(IdentityId),
    Var(V::IdentityVar),
}

impl<V: Phase> IdentityTerm<V> {
    pub fn map<W: Phase>(
        &self,
        on_identity: &mut impl FnMut(V::IdentityVar) -> IdentityTerm<W>,
    ) -> IdentityTerm<W> {
        match self {
            IdentityTerm::Known(id) => IdentityTerm::Known(*id),
            IdentityTerm::Var(v) => on_identity(*v),
        }
    }

    pub fn try_map<W: Phase, E>(
        &self,
        on_identity: &mut impl FnMut(V::IdentityVar) -> Result<IdentityTerm<W>, E>,
    ) -> Result<IdentityTerm<W>, E> {
        match self {
            IdentityTerm::Known(id) => Ok(IdentityTerm::Known(*id)),
            IdentityTerm::Var(v) => on_identity(*v),
        }
    }
}

impl IdentityTerm<Concrete> {
    pub fn get(&self) -> IdentityId {
        match self {
            IdentityTerm::Known(id) => *id,
            IdentityTerm::Var(v) => match *v {},
        }
    }
}

impl std::fmt::Display for IdentityId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Identity({self:?})")
    }
}

/// Concrete type - always fully resolved. `Var(Infallible)` is uninhabitable.
pub type Ty = TyTerm<Concrete>;

impl TyTerm<Concrete> {
    /// Create an `Error` (poison) type. See [`ErrorToken`] for permitted uses.
    pub fn error() -> Self {
        Ty::Error(ErrorToken::new())
    }

    pub fn is_error(&self) -> bool {
        matches!(self, Ty::Error(_))
    }

    pub fn effect(&self) -> Option<Effect> {
        match self {
            Ty::Fn { effect, .. } => Some(effect.get().clone()),
            _ => None,
        }
    }

    /// Extract the element type from a collection type.
    pub fn elem_of(&self) -> Option<&Ty> {
        match self {
            Ty::Array(elem, _) => Some(elem),
            _ => None,
        }
    }

    /// Whether a value of this type is data: something a host can keep
    /// from one run to the next (RFC-0014). A function, a spawn handle, an
    /// order, or a reference is not data, and neither is a type that holds
    /// one. An extension type is data; how it is written down is the
    /// host's declaration, not the checker's.
    pub fn is_data(&self) -> bool {
        match self {
            Ty::Int(_) | Ty::Float | Ty::String | Ty::Bool | Ty::Unit => true,
            Ty::Array(inner, _) | Ty::Option(inner) => inner.is_data(),
            Ty::Result(ok, err) => ok.is_data() && err.is_data(),
            Ty::Tuple(elems) => elems.iter().all(Ty::is_data),
            Ty::Object(fields) => fields.values().all(Ty::is_data),
            Ty::Enum { variants, .. } => variants
                .values()
                .all(|p| p.as_ref().is_none_or(|ty| ty.is_data())),
            Ty::UserDefined { type_args, .. } => type_args.iter().all(Ty::is_data),
            Ty::Fn { .. } | Ty::Handle(..) | Ty::Order | Ty::Ref(..) | Ty::Error(_) => false,
            Ty::Var(v) => match *v {},
        }
    }
}

impl<V> TyTerm<V>
where
    V: Phase,
{
    pub fn display<'a>(&'a self, interner: &'a Interner) -> TyDisplay<'a, V> {
        TyDisplay { ty: self, interner }
    }
}

pub struct TyDisplay<'a, V>
where
    V: Phase,
{
    ty: &'a TyTerm<V>,
    interner: &'a Interner,
}

/// A variable of a displayed type: `'n` for the n-th of its kind.
struct VarDisplay<T>(T);

impl<T> fmt::Display for VarDisplay<T>
where
    T: fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "'{}", self.0)
    }
}

impl<'a, V> fmt::Display for TyDisplay<'a, V>
where
    V: Phase,
    V::TyVar: fmt::Display,
    V::EffectVar: fmt::Display,
    V::LenVar: fmt::Display,
    V::IdentityVar: fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.ty {
            TyTerm::Int(k) => write!(f, "{}", k.name()),
            TyTerm::Order => write!(f, "Order"),
            TyTerm::Float => write!(f, "Float"),
            TyTerm::String => write!(f, "String"),
            TyTerm::Bool => write!(f, "Bool"),
            TyTerm::Unit => write!(f, "Unit"),
            TyTerm::Object(fields) => {
                let mut sorted: Vec<_> = fields.iter().collect();
                sorted.sort_by_key(|(k, _)| self.interner.resolve(**k).to_string());
                write!(f, "{{")?;
                for (i, (k, v)) in sorted.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(
                        f,
                        "{}: {}",
                        self.interner.resolve(**k),
                        v.display(self.interner)
                    )?;
                }
                write!(f, "}}")
            }
            TyTerm::Tuple(elems) => {
                write!(f, "(")?;
                for (i, e) in elems.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", e.display(self.interner))?;
                }
                write!(f, ")")
            }
            TyTerm::Fn {
                params,
                ret,
                captures: _,
                effect,
            } => {
                write!(f, "Fn(")?;
                for (i, p) in params.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", p.ty.display(self.interner))?;
                }
                write!(f, ") -> {}", ret.display(self.interner))?;
                let effect = match effect {
                    EffectTerm::Known(effect) => effect,
                    EffectTerm::Var(v) => return write!(f, " with {}", VarDisplay(v)),
                };
                if effect.is_empty() {
                    return Ok(());
                }
                write!(f, " with")?;
                if !effect.is_pure() {
                    write!(f, " {effect}")?;
                }
                for (label, set) in [("reads", &effect.reads), ("writes", &effect.writes)] {
                    if set.is_empty() {
                        continue;
                    }
                    write!(f, " {label} {{")?;
                    for (i, ctx) in set.iter().enumerate() {
                        if i > 0 {
                            write!(f, ", ")?;
                        }
                        write!(f, "@{}", self.interner.resolve(ctx.name))?;
                    }
                    write!(f, "}}")?;
                }
                Ok(())
            }
            TyTerm::Array(inner, len) => {
                write!(f, "Array<{}, ", inner.display(self.interner))?;
                match len {
                    LenTerm::Known(n) => write!(f, "{n}>"),
                    LenTerm::Var(v) => write!(f, "{}>", VarDisplay(v)),
                }
            }
            TyTerm::Handle(inner) => {
                write!(f, "Handle<{}>", inner.display(self.interner))
            }
            TyTerm::Option(inner) => write!(f, "Option<{}>", inner.display(self.interner)),
            TyTerm::Result(ok, err) => write!(
                f,
                "Result<{}, {}>",
                ok.display(self.interner),
                err.display(self.interner)
            ),
            TyTerm::UserDefined {
                id,
                type_args,
                effect_args,
                identity_args,
            } => {
                let name = self.interner.resolve(id.name);
                write!(f, "{name}")?;
                if !type_args.is_empty() || !effect_args.is_empty() || !identity_args.is_empty() {
                    write!(f, "<")?;
                    let mut first = true;
                    for arg in type_args {
                        if !first {
                            write!(f, ", ")?;
                        }
                        first = false;
                        write!(f, "{}", arg.display(self.interner))?;
                    }
                    for arg in effect_args {
                        if !first {
                            write!(f, ", ")?;
                        }
                        first = false;
                        match arg {
                            EffectTerm::Known(e) => write!(f, "{e}")?,
                            EffectTerm::Var(v) => write!(f, "{}", VarDisplay(v))?,
                        }
                    }
                    for arg in identity_args {
                        if !first {
                            write!(f, ", ")?;
                        }
                        first = false;
                        match arg {
                            IdentityTerm::Known(id) => write!(f, "#{id:?}")?,
                            IdentityTerm::Var(v) => write!(f, "#{}", VarDisplay(v))?,
                        }
                    }
                    write!(f, ">")?;
                }
                Ok(())
            }
            TyTerm::Enum { name, .. } => write!(f, "{}", self.interner.resolve(*name)),
            TyTerm::Ref(m, inner) => write!(f, "{}{}", m.prefix(), inner.display(self.interner)),
            TyTerm::Error(_) => write!(f, "<error>"),
            TyTerm::Var(v) => write!(f, "{}", VarDisplay(v)),
        }
    }
}

impl<'a, V> fmt::Debug for TyDisplay<'a, V>
where
    V: Phase,
    TyDisplay<'a, V>: fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

// -- TypeEnv ----------------------------------------------------------

/// Unified type environment for the type checker.
///
/// Replaces `ContextTypeRegistry` + internal `BuiltinRegistry`.
/// The type checker receives this as its sole external input -
/// it does not know whether a function is a builtin, extern, or user-defined.
/// All keys are QualifiedRef - the canonical identifier.
#[derive(Debug, Clone)]
pub struct TypeEnv {
    /// Context variable types - may contain inference variables (Solver-scoped).
    pub contexts: FxHashMap<QualifiedRef, InferTy>,
    /// Function type schemes - polymorphic, instantiated per call site.
    pub functions: FxHashMap<QualifiedRef, Scheme>,
}

/// What a name resolves to among the environment's functions.
pub enum FnLookup<'a> {
    Found(QualifiedRef, &'a Scheme),
    Ambiguous(Vec<QualifiedRef>),
    Missing,
}

impl TypeEnv {
    /// A script's bare name is its own function if it has one, else the one
    /// function of that name under any namespace (RFC-0021).
    pub fn resolve_fn(&self, name: QualifiedRef) -> FnLookup<'_> {
        if let Some(scheme) = self.functions.get(&name) {
            return FnLookup::Found(name, scheme);
        }
        if name.namespace.is_some() {
            return FnLookup::Missing;
        }
        let mut candidates: Vec<QualifiedRef> = self
            .functions
            .keys()
            .filter(|q| q.name == name.name && q.namespace.is_some())
            .copied()
            .collect();
        candidates.sort();
        match candidates.as_slice() {
            [] => FnLookup::Missing,
            [one] => FnLookup::Found(*one, &self.functions[one]),
            _ => FnLookup::Ambiguous(candidates),
        }
    }

    pub fn new() -> Self {
        Self {
            contexts: FxHashMap::default(),
            functions: FxHashMap::default(),
        }
    }
}

impl Default for TypeEnv {
    fn default() -> Self {
        Self::new()
    }
}

// -- Phase-parameterized type system ---------------------------------
//
// `TyTerm<V>` is a type term parameterized over inference variables.
// Two phases:
//   - `Concrete`: no inference variables (Var = Infallible). Post-inference.
//   - `Infer`:    may contain inference variables (Var = TypeBoundId). During inference.
//
// `type Ty = TyTerm<Concrete>` - always fully resolved. Compiler enforces this.
// `type InferTy = TyTerm<Infer>` - may have holes. Solver fills them in.

/// Phase marker trait - determines what can appear in inference variable slots.
pub trait Phase: 'static + Clone {
    /// Type inference variable. `Infallible` for concrete (uninhabitable).
    type TyVar: fmt::Debug + Clone + PartialEq + Eq + std::hash::Hash + Copy;
    /// Effect inference variable. `Infallible` for concrete (uninhabitable).
    type EffectVar: fmt::Debug + Clone + PartialEq + Eq + std::hash::Hash + Copy;
    /// Array length inference variable. `Infallible` for concrete (uninhabitable).
    type LenVar: fmt::Debug + Clone + PartialEq + Eq + std::hash::Hash + Copy;
    /// Identity inference variable. `Infallible` for concrete (uninhabitable).
    type IdentityVar: fmt::Debug + Clone + PartialEq + Eq + std::hash::Hash + Copy;
}

/// Post-inference phase - all types fully resolved.
/// `TyVar = Infallible` makes `TyTerm::Var` uninhabitable at type level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Concrete;

impl Phase for Concrete {
    type TyVar = Infallible;
    type EffectVar = Infallible;
    type LenVar = Infallible;
    type IdentityVar = Infallible;
}

/// Polymorphic declaration phase - type templates stored in the graph.
/// `TyVar = u32` is a positional placeholder, not tied to any Solver instance.
/// Instantiated to `Infer` per call site during type checking.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Poly;

impl Phase for Poly {
    type TyVar = u32;
    type EffectVar = u32;
    type LenVar = u32;
    type IdentityVar = u32;
}

/// During-inference phase - types may contain unresolved variables.
/// `TyVar = TypeBoundId` is scoped to a specific `Solver` instance.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Infer;

impl Phase for Infer {
    type TyVar = TypeBoundId;
    type EffectVar = EffectVarId;
    type LenVar = LenVarId;
    type IdentityVar = IdentityVarId;
}

/// Polymorphic type - template with positional placeholders.
pub type PolyTy = TyTerm<Poly>;
/// Polymorphic function parameter.
pub type PolyParam = ParamTerm<Poly>;

// -- Solver types ----------------------------------------------------

/// Index into `Solver::ty_bounds`. Identifies a type inference variable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TypeBoundId(pub u32);

/// Index into `Solver::effect_vars`. Identifies an effect inference variable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct EffectVarId(pub u32);

/// Index into `Solver::len_vars`. Identifies an array length inference variable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct LenVarId(pub u32);

/// Index into `Solver::identity_vars`. Identifies an identity inference variable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct IdentityVarId(pub u32);

// Re-export solver types - these were historically in ty.rs.
pub use crate::solver::{FreezeError, Solver, SolverSnapshot, Sources, TypeBound};

/// Type alias - always concrete, no inference variables.
pub type InferTy = TyTerm<Infer>;

/// A type term parameterized over inference phase.
///
/// When `V = Concrete`: `Var(Infallible)` is uninhabitable - type is always concrete.
/// When `V = Infer`: `Var(TypeBoundId)` references the solver's bound table.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TyTerm<V: Phase> {
    // Primitives
    Int(IntTy),
    Float,
    String,
    Bool,
    Unit,
    /// A dependency between effectful calls (RFC-0007). IR-only: no script
    /// names it, and no value of it exists at runtime.
    Order,
    // Containers
    Array(Box<TyTerm<V>>, LenTerm<V>),
    Object(FxHashMap<Astr, TyTerm<V>>),
    Tuple(Vec<TyTerm<V>>),
    Option(Box<TyTerm<V>>),
    Result(Box<TyTerm<V>>, Box<TyTerm<V>>),
    // Functions
    Fn {
        params: Vec<ParamTerm<V>>,
        ret: Box<TyTerm<V>>,
        captures: Vec<TyTerm<V>>,
        effect: EffectTerm<V>,
    },
    // Nominal
    UserDefined {
        id: QualifiedRef,
        type_args: Vec<TyTerm<V>>,
        effect_args: Vec<EffectTerm<V>>,
        identity_args: Vec<IdentityTerm<V>>,
    },
    Enum {
        name: Astr,
        variants: FxHashMap<Astr, Option<Box<TyTerm<V>>>>,
    },
    // Resources
    Handle(Box<TyTerm<V>>),
    /// `&T` or `&mut T`: a second name for a storage holding a `T`
    /// (RFC-0018). A word at runtime; never data.
    Ref(Mutability, Box<TyTerm<V>>),
    // Special
    Error(ErrorToken),
    /// Inference variable - only inhabitable when `V = Infer`.
    /// For `V = Concrete`, this is `Var(Infallible)` which cannot be constructed.
    Var(V::TyVar),
}

/// Whether a reference may write through to its storage (RFC-0018).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum Mutability {
    Shared,
    Mut,
}

impl Mutability {
    pub fn prefix(self) -> &'static str {
        match self {
            Mutability::Shared => "&",
            Mutability::Mut => "&mut ",
        }
    }
}

/// Named, typed function parameter - parameterized over phase. A parameter
/// that borrows has a reference type; there is no mode beside the type.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParamTerm<V: Phase> {
    pub name: Astr,
    pub ty: TyTerm<V>,
}

impl<V: Phase> ParamTerm<V> {
    pub fn new(name: Astr, ty: TyTerm<V>) -> Self {
        Self { name, ty }
    }

    /// The same parameter with another type.
    pub fn retyped<W: Phase>(&self, ty: TyTerm<W>) -> ParamTerm<W> {
        ParamTerm {
            name: self.name,
            ty,
        }
    }
}

// -- Generic phase traversal ----------------------------------------
//
// `map` and `try_map` provide the single recursive traversal over
// `TyTerm<V>`. All phase-to-phase transformations (lift, freeze,
// resolve, instantiate) are specializations of these two operations.

impl<V: Phase> TyTerm<V> {
    pub const I8: Self = TyTerm::Int(IntTy::I8);
    pub const I16: Self = TyTerm::Int(IntTy::I16);
    pub const I32: Self = TyTerm::Int(IntTy::I32);
    pub const I64: Self = TyTerm::Int(IntTy::I64);
    pub const U8: Self = TyTerm::Int(IntTy::U8);
    pub const U16: Self = TyTerm::Int(IntTy::U16);
    pub const U32: Self = TyTerm::Int(IntTy::U32);
    pub const U64: Self = TyTerm::Int(IntTy::U64);

    /// A word-sized type the runtime copies (RFC-0018); everything else moves.
    pub fn is_primitive(&self) -> bool {
        matches!(
            self,
            TyTerm::Int(_) | TyTerm::Float | TyTerm::Bool | TyTerm::Unit | TyTerm::Order
        )
    }

    /// Map this type term from phase `V` to phase `W`.
    ///
    /// Structural recursion is automatic - only variable slots and
    /// identity slots need custom handling via the provided closures.
    pub fn map<W: Phase>(
        &self,
        on_var: &mut impl FnMut(V::TyVar) -> TyTerm<W>,
        on_identity: &mut impl FnMut(V::IdentityVar) -> IdentityTerm<W>,
        on_effect: &mut impl FnMut(V::EffectVar) -> EffectTerm<W>,
        on_len: &mut impl FnMut(V::LenVar) -> LenTerm<W>,
    ) -> TyTerm<W> {
        match self {
            TyTerm::Int(k) => TyTerm::Int(*k),
            TyTerm::Float => TyTerm::Float,
            TyTerm::String => TyTerm::String,
            TyTerm::Bool => TyTerm::Bool,
            TyTerm::Unit => TyTerm::Unit,
            TyTerm::Order => TyTerm::Order,
            TyTerm::Array(inner, len) => TyTerm::Array(
                Box::new(inner.map(on_var, on_identity, on_effect, on_len)),
                len.map(on_len),
            ),
            TyTerm::Object(fields) => TyTerm::Object(
                fields
                    .iter()
                    .map(|(k, v)| (*k, v.map(on_var, on_identity, on_effect, on_len)))
                    .collect(),
            ),
            TyTerm::Tuple(elems) => TyTerm::Tuple(
                elems
                    .iter()
                    .map(|e| e.map(on_var, on_identity, on_effect, on_len))
                    .collect(),
            ),
            TyTerm::Option(inner) => {
                TyTerm::Option(Box::new(inner.map(on_var, on_identity, on_effect, on_len)))
            }
            TyTerm::Result(ok, err) => TyTerm::Result(
                Box::new(ok.map(on_var, on_identity, on_effect, on_len)),
                Box::new(err.map(on_var, on_identity, on_effect, on_len)),
            ),
            TyTerm::Fn {
                params,
                ret,
                captures,
                effect,
            } => TyTerm::Fn {
                params: params
                    .iter()
                    .map(|p| p.retyped(p.ty.map(on_var, on_identity, on_effect, on_len)))
                    .collect(),
                ret: Box::new(ret.map(on_var, on_identity, on_effect, on_len)),
                captures: captures
                    .iter()
                    .map(|c| c.map(on_var, on_identity, on_effect, on_len))
                    .collect(),
                effect: effect.map(on_effect),
            },
            TyTerm::UserDefined {
                id,
                type_args,
                effect_args,
                identity_args,
            } => TyTerm::UserDefined {
                id: *id,
                type_args: type_args
                    .iter()
                    .map(|t| t.map(on_var, on_identity, on_effect, on_len))
                    .collect(),
                effect_args: effect_args.iter().map(|e| e.map(on_effect)).collect(),
                identity_args: identity_args.iter().map(|i| i.map(on_identity)).collect(),
            },
            TyTerm::Enum { name, variants } => TyTerm::Enum {
                name: *name,
                variants: variants
                    .iter()
                    .map(|(tag, payload)| {
                        (
                            *tag,
                            payload
                                .as_ref()
                                .map(|ty| Box::new(ty.map(on_var, on_identity, on_effect, on_len))),
                        )
                    })
                    .collect(),
            },
            TyTerm::Handle(inner) => {
                TyTerm::Handle(Box::new(inner.map(on_var, on_identity, on_effect, on_len)))
            }
            TyTerm::Ref(m, inner) => TyTerm::Ref(
                *m,
                Box::new(inner.map(on_var, on_identity, on_effect, on_len)),
            ),
            TyTerm::Error(token) => TyTerm::Error(*token),
            TyTerm::Var(v) => on_var(*v),
        }
    }

    /// Fallible version of `map` - short-circuits on first error.
    pub fn try_map<W: Phase, E>(
        &self,
        on_var: &mut impl FnMut(V::TyVar) -> Result<TyTerm<W>, E>,
        on_identity: &mut impl FnMut(V::IdentityVar) -> Result<IdentityTerm<W>, E>,
        on_effect: &mut impl FnMut(V::EffectVar) -> Result<EffectTerm<W>, E>,
        on_len: &mut impl FnMut(V::LenVar) -> Result<LenTerm<W>, E>,
    ) -> Result<TyTerm<W>, E> {
        match self {
            TyTerm::Int(k) => Ok(TyTerm::Int(*k)),
            TyTerm::Float => Ok(TyTerm::Float),
            TyTerm::String => Ok(TyTerm::String),
            TyTerm::Bool => Ok(TyTerm::Bool),
            TyTerm::Unit => Ok(TyTerm::Unit),
            TyTerm::Order => Ok(TyTerm::Order),
            TyTerm::Array(inner, len) => Ok(TyTerm::Array(
                Box::new(inner.try_map(on_var, on_identity, on_effect, on_len)?),
                len.try_map(on_len)?,
            )),
            TyTerm::Object(fields) => {
                let mapped: Result<FxHashMap<_, _>, E> = fields
                    .iter()
                    .map(|(k, v)| {
                        v.try_map(on_var, on_identity, on_effect, on_len)
                            .map(|mv| (*k, mv))
                    })
                    .collect();
                Ok(TyTerm::Object(mapped?))
            }
            TyTerm::Tuple(elems) => Ok(TyTerm::Tuple(
                elems
                    .iter()
                    .map(|e| e.try_map(on_var, on_identity, on_effect, on_len))
                    .collect::<Result<_, _>>()?,
            )),
            TyTerm::Option(inner) => Ok(TyTerm::Option(Box::new(inner.try_map(
                on_var,
                on_identity,
                on_effect,
                on_len,
            )?))),
            TyTerm::Result(ok, err) => Ok(TyTerm::Result(
                Box::new(ok.try_map(on_var, on_identity, on_effect, on_len)?),
                Box::new(err.try_map(on_var, on_identity, on_effect, on_len)?),
            )),
            TyTerm::Fn {
                params,
                ret,
                captures,
                effect,
            } => Ok(TyTerm::Fn {
                params: params
                    .iter()
                    .map(|p| {
                        p.ty.try_map(on_var, on_identity, on_effect, on_len)
                            .map(|ty| p.retyped(ty))
                    })
                    .collect::<Result<_, _>>()?,
                ret: Box::new(ret.try_map(on_var, on_identity, on_effect, on_len)?),
                captures: captures
                    .iter()
                    .map(|c| c.try_map(on_var, on_identity, on_effect, on_len))
                    .collect::<Result<_, _>>()?,
                effect: effect.try_map(on_effect)?,
            }),
            TyTerm::UserDefined {
                id,
                type_args,
                effect_args,
                identity_args,
            } => Ok(TyTerm::UserDefined {
                id: *id,
                type_args: type_args
                    .iter()
                    .map(|t| t.try_map(on_var, on_identity, on_effect, on_len))
                    .collect::<Result<_, _>>()?,
                effect_args: effect_args
                    .iter()
                    .map(|e| e.try_map(on_effect))
                    .collect::<Result<_, _>>()?,
                identity_args: identity_args
                    .iter()
                    .map(|i| i.try_map(on_identity))
                    .collect::<Result<_, _>>()?,
            }),
            TyTerm::Enum { name, variants } => {
                let mapped: Result<FxHashMap<_, _>, E> = variants
                    .iter()
                    .map(|(tag, payload)| {
                        let mp = match payload {
                            Some(ty) => Some(Box::new(ty.try_map(
                                on_var,
                                on_identity,
                                on_effect,
                                on_len,
                            )?)),
                            None => None,
                        };
                        Ok((*tag, mp))
                    })
                    .collect();
                Ok(TyTerm::Enum {
                    name: *name,
                    variants: mapped?,
                })
            }
            TyTerm::Handle(inner) => Ok(TyTerm::Handle(Box::new(inner.try_map(
                on_var,
                on_identity,
                on_effect,
                on_len,
            )?))),
            TyTerm::Ref(m, inner) => Ok(TyTerm::Ref(
                *m,
                Box::new(inner.try_map(on_var, on_identity, on_effect, on_len)?),
            )),
            TyTerm::Error(token) => Ok(TyTerm::Error(*token)),
            TyTerm::Var(v) => on_var(*v),
        }
    }
}

// -- Lift: Concrete -> any Phase --------------------------------------

/// Lift a concrete `Ty` into any phase (mechanical, zero information change).
/// Infallible because `Concrete` has `TyVar = Infallible` (uninhabitable).
pub fn lift_ty<W: Phase>(ty: &Ty) -> TyTerm<W> {
    ty.map(
        &mut |v: Infallible| match v {},
        &mut |v: Infallible| match v {},
        &mut |v: Infallible| match v {},
        &mut |v: Infallible| match v {},
    )
}

/// Alias: lift `Ty` to `PolyTy`. Kept for call-site compatibility.
pub fn lift_to_poly(ty: &Ty) -> PolyTy {
    lift_ty(ty)
}

/// Lift a concrete type as a declaration: a host that declares a context
/// or a parameter names no source, so every identity argument becomes a
/// variable the compilation mints a source for (RFC-0012). A source
/// number in a `Ty` is meaningful only inside the compilation that
/// minted it, never in a declaration.
pub fn lift_declaration(ty: &Ty, builder: &mut PolyBuilder) -> PolyTy {
    fn go(ty: &Ty, builder: &mut PolyBuilder) -> PolyTy {
        match ty {
            Ty::Int(k) => TyTerm::Int(*k),
            Ty::Float => TyTerm::Float,
            Ty::String => TyTerm::String,
            Ty::Bool => TyTerm::Bool,
            Ty::Unit => TyTerm::Unit,
            Ty::Order => TyTerm::Order,
            Ty::Array(inner, len) => TyTerm::Array(Box::new(go(inner, builder)), lift_ty_len(len)),
            Ty::Object(fields) => {
                TyTerm::Object(fields.iter().map(|(k, v)| (*k, go(v, builder))).collect())
            }
            Ty::Tuple(elems) => TyTerm::Tuple(elems.iter().map(|e| go(e, builder)).collect()),
            Ty::Option(inner) => TyTerm::Option(Box::new(go(inner, builder))),
            Ty::Result(ok, err) => {
                TyTerm::Result(Box::new(go(ok, builder)), Box::new(go(err, builder)))
            }
            Ty::Fn {
                params,
                ret,
                captures,
                effect,
            } => TyTerm::Fn {
                params: params
                    .iter()
                    .map(|p| ParamTerm {
                        name: p.name,
                        ty: go(&p.ty, builder),
                    })
                    .collect(),
                ret: Box::new(go(ret, builder)),
                captures: captures.iter().map(|c| go(c, builder)).collect(),
                effect: lift_ty_effect(effect),
            },
            Ty::UserDefined {
                id,
                type_args,
                effect_args,
                identity_args,
            } => TyTerm::UserDefined {
                id: *id,
                type_args: type_args.iter().map(|t| go(t, builder)).collect(),
                effect_args: effect_args.iter().map(lift_ty_effect).collect(),
                identity_args: identity_args
                    .iter()
                    .map(|_| builder.fresh_identity_var())
                    .collect(),
            },
            Ty::Enum { name, variants } => TyTerm::Enum {
                name: *name,
                variants: variants
                    .iter()
                    .map(|(k, v)| (*k, v.as_ref().map(|t| Box::new(go(t, builder)))))
                    .collect(),
            },
            Ty::Handle(inner) => TyTerm::Handle(Box::new(go(inner, builder))),
            Ty::Ref(m, inner) => TyTerm::Ref(*m, Box::new(go(inner, builder))),
            Ty::Error(token) => TyTerm::Error(*token),
            Ty::Var(v) => match *v {},
        }
    }
    go(ty, builder)
}

fn lift_ty_len(len: &LenTerm<Concrete>) -> LenTerm<Poly> {
    len.map(&mut |v: Infallible| match v {})
}

fn lift_ty_effect(effect: &EffectTerm<Concrete>) -> EffectTerm<Poly> {
    effect.map(&mut |v: Infallible| match v {})
}

/// Try to convert a `PolyTy` to a concrete `Ty`.
/// Returns `None` if the poly type contains any Var placeholders.
pub fn try_freeze_poly(ty: &PolyTy) -> Option<Ty> {
    ty.try_map(
        &mut |_: u32| Err(()),
        &mut |_: u32| Err(()),
        &mut |_: u32| Err(()),
        &mut |_: u32| Err(()),
    )
    .ok()
}

// -- PolyBuilder -----------------------------------------------------

/// Builder for polymorphic type templates. No Solver dependency.
/// Creates positional placeholders (Var(0), Var(1), ...) for type variables.
pub struct PolyBuilder {
    next_ty: u32,
    next_effect: u32,
    next_len: u32,
    next_identity: u32,
}

impl PolyBuilder {
    pub fn new() -> Self {
        Self {
            next_ty: 0,
            next_effect: 0,
            next_len: 0,
            next_identity: 0,
        }
    }

    pub fn fresh_len_var(&mut self) -> LenTerm<Poly> {
        let id = self.next_len;
        self.next_len += 1;
        LenTerm::Var(id)
    }

    pub fn fresh_effect_var(&mut self) -> EffectTerm<Poly> {
        let id = self.next_effect;
        self.next_effect += 1;
        EffectTerm::Var(id)
    }

    /// Create a fresh type placeholder.
    pub fn fresh_ty_var(&mut self) -> PolyTy {
        let id = self.next_ty;
        self.next_ty += 1;
        TyTerm::Var(id)
    }

    pub fn fresh_identity_var(&mut self) -> IdentityTerm<Poly> {
        let id = self.next_identity;
        self.next_identity += 1;
        IdentityTerm::Var(id)
    }
}

impl Default for PolyBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::types::QualifiedRef;
    use acvus_utils::Interner;

    use Polarity::*;

    #[test]
    fn a_pure_effect_commutes_by_definition() {
        assert_eq!(Effect::new(Reissue::Pure, false), Effect::PURE);
        assert!(Effect::PURE.commutes);
        assert_eq!(Effect::PURE.to_string(), "Pure");
        assert_eq!(
            Effect::IDEMPOTENT.commutative().to_string(),
            "Idempotent+commutative"
        );
    }

    #[test]
    fn effects_join_on_both_axes() {
        let c = Effect::IDEMPOTENT.commutative();
        assert_eq!(c.join(&Effect::IDEMPOTENT), Effect::IDEMPOTENT);
        assert_eq!(c.join(&Effect::PURE), c);
        assert_eq!(
            c.join(&Effect::OPAQUE.commutative()),
            Effect::OPAQUE.commutative()
        );
        assert_eq!(Effect::OPAQUE.meet(&c), c);
    }

    #[test]
    fn the_product_order_leaves_the_axes_incomparable() {
        let c = Effect::OPAQUE.commutative();
        assert!(Effect::PURE.at_most(&c));
        assert!(c.at_most(&Effect::OPAQUE));
        assert!(!Effect::IDEMPOTENT.at_most(&c));
        assert!(!c.at_most(&Effect::IDEMPOTENT));
        assert_eq!(Effect::IDEMPOTENT.partial_cmp(&c), None);
    }

    fn arr<V: Phase>(elem: TyTerm<V>, n: usize) -> TyTerm<V> {
        TyTerm::Array(Box::new(elem), LenTerm::Known(n))
    }

    /// Test helper: create a unique `QualifiedRef` for each call.
    /// Uses a thread-local counter to ensure uniqueness across tests.
    fn fresh_qref() -> QualifiedRef {
        use std::sync::atomic::{AtomicU32, Ordering};
        static COUNTER: AtomicU32 = AtomicU32::new(0);
        thread_local! {
            static INTERNER: Interner = Interner::new();
        }
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        INTERNER.with(|i| QualifiedRef::root(i.intern(&format!("TestType{n}"))))
    }

    /// Test helper: create a `TyTerm::UserDefined` with a fresh id and no type args.
    fn test_user_defined() -> Ty {
        TyTerm::UserDefined {
            id: fresh_qref(),
            type_args: vec![],
            effect_args: vec![],
            identity_args: vec![],
        }
    }

    #[test]
    fn unify_same_concrete() {
        let mut sources = Sources::new();
        let mut s = Solver::new(&mut sources);
        let registry = TypeRegistry::new();
        assert!(
            s.unify_ty(&TyTerm::I64, &TyTerm::I64, Invariant, &registry)
                .is_ok()
        );
        assert!(
            s.unify_ty(&TyTerm::Float, &TyTerm::Float, Invariant, &registry)
                .is_ok()
        );
        assert!(
            s.unify_ty(&TyTerm::String, &TyTerm::String, Invariant, &registry)
                .is_ok()
        );
        assert!(
            s.unify_ty(&TyTerm::Bool, &TyTerm::Bool, Invariant, &registry)
                .is_ok()
        );
        assert!(
            s.unify_ty(&TyTerm::Unit, &TyTerm::Unit, Invariant, &registry)
                .is_ok()
        );
    }

    #[test]
    fn unify_different_concrete_fails() {
        let mut sources = Sources::new();
        let mut s = Solver::new(&mut sources);
        let registry = TypeRegistry::new();
        assert!(
            s.unify_ty(&TyTerm::I64, &TyTerm::Float, Invariant, &registry)
                .is_err()
        );
        assert!(
            s.unify_ty(&TyTerm::String, &TyTerm::Bool, Invariant, &registry)
                .is_err()
        );
    }

    #[test]
    fn unify_var_with_concrete() {
        let mut sources = Sources::new();
        let mut s = Solver::new(&mut sources);
        let registry = TypeRegistry::new();
        let t = s.fresh_ty_var();
        assert!(s.unify_ty(&t, &TyTerm::I64, Invariant, &registry).is_ok());
        assert_eq!(s.resolve_ty(&t), TyTerm::I64);
    }

    #[test]
    fn unify_object() {
        let mut sources = Sources::new();
        let mut s = Solver::new(&mut sources);
        let registry = TypeRegistry::new();
        let interner = Interner::new();
        let t = s.fresh_ty_var();
        let obj1 = TyTerm::Object(FxHashMap::from_iter([
            (interner.intern("name"), TyTerm::String),
            (interner.intern("age"), t.clone()),
        ]));
        let obj2 = TyTerm::Object(FxHashMap::from_iter([
            (interner.intern("name"), TyTerm::String),
            (interner.intern("age"), TyTerm::I64),
        ]));
        assert!(s.unify_ty(&obj1, &obj2, Invariant, &registry).is_ok());
        assert_eq!(s.resolve_ty(&t), TyTerm::I64);
    }

    #[test]
    fn unify_object_key_mismatch() {
        let mut sources = Sources::new();
        let mut s = Solver::new(&mut sources);
        let registry = TypeRegistry::new();
        let interner = Interner::new();
        let obj1 = TyTerm::Object(FxHashMap::from_iter([(
            interner.intern("name"),
            TyTerm::String,
        )]));
        let obj2 = TyTerm::Object(FxHashMap::from_iter([(
            interner.intern("age"),
            TyTerm::I64,
        )]));
        assert!(s.unify_ty(&obj1, &obj2, Invariant, &registry).is_err());
    }

    #[test]
    fn transitive_resolution() {
        let mut sources = Sources::new();
        let mut s = Solver::new(&mut sources);
        let registry = TypeRegistry::new();
        let t1 = s.fresh_ty_var();
        let t2 = s.fresh_ty_var();
        assert!(s.unify_ty(&t1, &t2, Invariant, &registry).is_ok());
        assert!(
            s.unify_ty(&t2, &TyTerm::String, Invariant, &registry)
                .is_ok()
        );
        assert_eq!(s.resolve_ty(&t1), TyTerm::String);
    }

    // -- Object merge tests --

    #[test]
    fn unify_object_disjoint_via_var() {
        // Var -> {a} then Var -> {b} should merge to {a, b}
        let mut sources = Sources::new();
        let mut s = Solver::new(&mut sources);
        let registry = TypeRegistry::new();
        let i = Interner::new();
        let v = s.fresh_ty_var();
        let obj_a = TyTerm::Object(FxHashMap::from_iter([(i.intern("a"), TyTerm::I64)]));
        let obj_b = TyTerm::Object(FxHashMap::from_iter([(i.intern("b"), TyTerm::String)]));
        assert!(s.unify_ty(&v, &obj_a, Invariant, &registry).is_ok());
        assert!(s.unify_ty(&v, &obj_b, Invariant, &registry).is_ok());
        let resolved = s.resolve_ty(&v);
        match &resolved {
            TyTerm::Object(fields) => {
                assert_eq!(fields.len(), 2, "expected {{a, b}}, got {fields:?}");
                assert_eq!(fields.get(&i.intern("a")), Some(&TyTerm::I64));
                assert_eq!(fields.get(&i.intern("b")), Some(&TyTerm::String));
            }
            other => panic!("expected Object, got {other:?}"),
        }
    }

    #[test]
    fn unify_object_overlapping_via_var() {
        // Var -> {a, b} then Var -> {b, c} should merge to {a, b, c}
        let mut sources = Sources::new();
        let mut s = Solver::new(&mut sources);
        let registry = TypeRegistry::new();
        let i = Interner::new();
        let v = s.fresh_ty_var();
        let obj_ab = TyTerm::Object(FxHashMap::from_iter([
            (i.intern("a"), TyTerm::I64),
            (i.intern("b"), TyTerm::String),
        ]));
        let obj_bc = TyTerm::Object(FxHashMap::from_iter([
            (i.intern("b"), TyTerm::String),
            (i.intern("c"), TyTerm::Bool),
        ]));
        assert!(s.unify_ty(&v, &obj_ab, Invariant, &registry).is_ok());
        assert!(s.unify_ty(&v, &obj_bc, Invariant, &registry).is_ok());
        let resolved = s.resolve_ty(&v);
        match &resolved {
            TyTerm::Object(fields) => {
                assert_eq!(fields.len(), 3, "expected {{a, b, c}}, got {fields:?}");
                assert_eq!(fields.get(&i.intern("a")), Some(&TyTerm::I64));
                assert_eq!(fields.get(&i.intern("b")), Some(&TyTerm::String));
                assert_eq!(fields.get(&i.intern("c")), Some(&TyTerm::Bool));
            }
            other => panic!("expected Object, got {other:?}"),
        }
    }

    #[test]
    fn unify_object_overlap_type_conflict_fails() {
        // {b: Int} and {b: String} via same Var should fail
        let mut sources = Sources::new();
        let mut s = Solver::new(&mut sources);
        let registry = TypeRegistry::new();
        let i = Interner::new();
        let v = s.fresh_ty_var();
        let obj1 = TyTerm::Object(FxHashMap::from_iter([(i.intern("b"), TyTerm::I64)]));
        let obj2 = TyTerm::Object(FxHashMap::from_iter([(i.intern("b"), TyTerm::String)]));
        assert!(s.unify_ty(&v, &obj1, Invariant, &registry).is_ok());
        assert!(s.unify_ty(&v, &obj2, Invariant, &registry).is_err());
    }

    #[test]
    fn fresh_param_produces_unique_ids() {
        let mut sources = Sources::new();
        let mut s = Solver::new(&mut sources);
        let registry = TypeRegistry::new();
        let o1 = s.fresh_ty_var();
        let o2 = s.fresh_ty_var();
        let o3 = s.fresh_ty_var();
        assert_ne!(o1, o2);
        assert_ne!(o2, o3);
        assert_ne!(o1, o3);
    }

    // -- Polarity-based subtyping tests --

    #[test]
    fn polarity_flip() {
        assert_eq!(Covariant.flip(), Contravariant);
        assert_eq!(Contravariant.flip(), Covariant);
        assert_eq!(Invariant.flip(), Invariant);
    }

    // -- Variance unsoundness edge case tests --

    // ================================================================
    // Var chain + coercion interaction
    // ================================================================

    // ================================================================
    // Occurs check + polarity
    // ================================================================

    #[test]
    fn occurs_check_through_list_covariant() {
        // Var = List<Var> should fail (occurs) regardless of polarity.
        let mut sources = Sources::new();
        let mut s = Solver::new(&mut sources);
        let registry = TypeRegistry::new();
        let v = s.fresh_ty_var();
        let cyclic = arr(v.clone(), 3);
        assert!(s.unify_ty(&v, &cyclic, Covariant, &registry).is_err());
    }

    // ================================================================
    // Deep nesting coercion
    // ================================================================

    // ================================================================
    // Object merge + coercion at the same time
    // ================================================================

    // ================================================================
    // Snapshot/rollback isolation
    // ================================================================

    // ================================================================
    // Polarity symmetry / duality
    // ================================================================

    #[test]
    fn invariant_same_types_both_directions() {
        // Same concrete type: Invariant must succeed regardless of order.
        let mut sources = Sources::new();
        let mut s = Solver::new(&mut sources);
        let registry = TypeRegistry::new();
        let l1 = arr(TyTerm::I64, 3);
        let l2 = arr(TyTerm::I64, 3);
        assert!(s.unify_ty(&l1, &l2, Invariant, &registry).is_ok());
        assert!(s.unify_ty(&l2, &l1, Invariant, &registry).is_ok());
    }

    // ================================================================
    // Unresolved Var containers + coercion
    // ================================================================

    // ================================================================
    // Bidirectional Var binding + coercion
    // ================================================================

    // ================================================================
    // N-way demotion (large fan-out)
    // ================================================================

    // ================================================================
    // Mixed concrete/param identities
    // ================================================================

    // ================================================================
    // Error / Param + polarity (poison / unification absorption)
    // ================================================================

    // ================================================================
    // Transitive coercion chains
    // ================================================================

    // ================================================================
    // Inner type mismatch under coercion (must not be masked)
    // ================================================================

    // ================================================================
    // Coercion does NOT propagate across unrelated type constructors
    // ================================================================

    #[test]
    fn list_vs_tuple_fails_any_polarity() {
        let mut sources = Sources::new();
        let mut s = Solver::new(&mut sources);
        let registry = TypeRegistry::new();
        let l = arr(TyTerm::I64, 3);
        let t = TyTerm::Tuple(vec![TyTerm::I64]);
        assert!(s.unify_ty(&l, &t, Covariant, &registry).is_err());
        assert!(s.unify_ty(&l, &t, Invariant, &registry).is_err());
    }

    // ================================================================
    // Triple flip (Fn<Fn<Fn<...>>>)
    // ================================================================

    // ================================================================
    // Regression: same identity must not trigger demotion
    // ================================================================

    // -- Sequence identity tracking ---------------------------------

    // -- UserDefined unification tests -------------------------------

    fn ud(id: QualifiedRef, type_args: Vec<InferTy>) -> InferTy {
        TyTerm::UserDefined {
            id,
            type_args,
            effect_args: vec![],
            identity_args: vec![],
        }
    }

    // -- Completeness: valid UserDefined unifications --

    #[test]
    fn user_defined_same_id_empty_args_unifies() {
        let mut sources = Sources::new();
        let mut s = Solver::new(&mut sources);
        let registry = TypeRegistry::new();
        let id = fresh_qref();
        assert!(
            s.unify_ty(&ud(id, vec![]), &ud(id, vec![]), Invariant, &registry)
                .is_ok()
        );
    }

    #[test]
    fn user_defined_same_id_concrete_type_args_unifies() {
        let mut sources = Sources::new();
        let mut s = Solver::new(&mut sources);
        let registry = TypeRegistry::new();
        let id = fresh_qref();
        assert!(
            s.unify_ty(
                &ud(id, vec![TyTerm::I64]),
                &ud(id, vec![TyTerm::I64]),
                Invariant,
                &registry
            )
            .is_ok()
        );
    }

    #[test]
    fn user_defined_param_type_arg_resolved_via_unify() {
        let mut sources = Sources::new();
        let mut s = Solver::new(&mut sources);
        let registry = TypeRegistry::new();
        let id = fresh_qref();
        let p = s.fresh_ty_var();
        assert!(
            s.unify_ty(
                &ud(id, vec![p.clone()]),
                &ud(id, vec![TyTerm::I64]),
                Invariant,
                &registry
            )
            .is_ok()
        );
        assert_eq!(s.resolve_ty(&p), TyTerm::I64);
    }

    #[test]
    fn user_defined_nested_type_arg_unifies() {
        // UserDefined<List<Param>> vs UserDefined<List<Int>> -> resolves Param to Int
        let mut sources = Sources::new();
        let mut s = Solver::new(&mut sources);
        let registry = TypeRegistry::new();
        let id = fresh_qref();
        let p = s.fresh_ty_var();
        assert!(
            s.unify_ty(
                &ud(id, vec![arr(p.clone(), 3)]),
                &ud(id, vec![arr(TyTerm::I64, 3)]),
                Invariant,
                &registry
            )
            .is_ok()
        );
        assert_eq!(s.resolve_ty(&p), TyTerm::I64);
    }

    // -- Soundness: invalid UserDefined unifications --

    #[test]
    fn user_defined_different_id_fails() {
        let mut sources = Sources::new();
        let mut s = Solver::new(&mut sources);
        let registry = TypeRegistry::new();
        let id_a = fresh_qref();
        let id_b = fresh_qref();
        assert!(
            s.unify_ty(&ud(id_a, vec![]), &ud(id_b, vec![]), Invariant, &registry)
                .is_err()
        );
    }

    #[test]
    fn user_defined_type_arg_mismatch_fails() {
        let mut sources = Sources::new();
        let mut s = Solver::new(&mut sources);
        let registry = TypeRegistry::new();
        let id = fresh_qref();
        assert!(
            s.unify_ty(
                &ud(id, vec![TyTerm::I64]),
                &ud(id, vec![TyTerm::String]),
                Invariant,
                &registry
            )
            .is_err()
        );
    }

    #[test]
    fn user_defined_vs_other_ty_fails() {
        let mut sources = Sources::new();
        let mut s = Solver::new(&mut sources);
        let registry = TypeRegistry::new();
        let id = fresh_qref();
        assert!(
            s.unify_ty(&ud(id, vec![]), &TyTerm::I64, Invariant, &registry)
                .is_err()
        );
        assert!(
            s.unify_ty(&TyTerm::String, &ud(id, vec![]), Invariant, &registry)
                .is_err()
        );
    }

    // -- Resolve --

    #[test]
    fn user_defined_inside_list_resolves() {
        let mut sources = Sources::new();
        let mut s = Solver::new(&mut sources);
        let registry = TypeRegistry::new();
        let id = fresh_qref();
        let p = s.fresh_ty_var();
        let ty = arr(ud(id, vec![p.clone()]), 3);
        assert!(s.unify_ty(&p, &TyTerm::I64, Invariant, &registry).is_ok());
        match s.resolve_ty(&ty) {
            TyTerm::Array(inner, _) => match *inner {
                TyTerm::UserDefined { type_args, .. } => {
                    assert_eq!(type_args, vec![TyTerm::I64])
                }
                other => panic!("expected UserDefined, got {other:?}"),
            },
            other => panic!("expected Array, got {other:?}"),
        }
    }

    // -- TypeRegistry --

    #[test]
    fn type_registry_register_and_get() {
        let mut reg = TypeRegistry::new();
        let id = fresh_qref();
        reg.register(UserDefinedDecl {
            qref: id,
            type_params: vec![TyVarBound::Any],
            effect_params: 0,
            identity_params: 0,
        });
        let decl = reg.get(id);
        assert_eq!(decl.qref, id);
        assert_eq!(decl.type_params.len(), 1);
    }

    #[test]
    #[should_panic(expected = "duplicate")]
    fn type_registry_duplicate_panics() {
        let mut reg = TypeRegistry::new();
        let id = fresh_qref();
        reg.register(UserDefinedDecl {
            qref: id,
            type_params: vec![],
            effect_params: 0,
            identity_params: 0,
        });
        reg.register(UserDefinedDecl {
            qref: id,
            type_params: vec![],
            effect_params: 0,
            identity_params: 0,
        });
    }

    #[test]
    #[should_panic(expected = "unknown")]
    fn type_registry_unknown_id_panics() {
        let reg = TypeRegistry::new();
        let id = fresh_qref();
        reg.get(id);
    }

    // -- ExternCast tests --------------------------------------------

    /// A registry with one cast rule `UserDefined(A, [T..]) -> to`.
    struct CastSetup {
        from_id: QualifiedRef,
        registry: TypeRegistry,
    }

    fn make_cast_registry(
        type_param_count: usize,
        build_to: impl FnOnce(&[PolyTy]) -> PolyTy,
    ) -> CastSetup {
        let id = fresh_qref();
        let i = acvus_utils::Interner::new();
        let fn_id = QualifiedRef::root(i.intern("cast_fn"));
        let mut builder = PolyBuilder::new();
        let params: Vec<PolyTy> = (0..type_param_count)
            .map(|_| builder.fresh_ty_var())
            .collect();
        let from = TyTerm::UserDefined {
            id,
            type_args: params.clone(),
            effect_args: vec![],
            identity_args: vec![],
        };
        let to = build_to(&params);
        let mut reg = TypeRegistry::new();
        reg.register(UserDefinedDecl {
            qref: id,
            type_params: vec![TyVarBound::Any; type_param_count],
            effect_params: 0,
            identity_params: 0,
        });
        reg.register_cast(CastRule {
            from,
            to,
            fn_ref: fn_id,
        });
        CastSetup {
            from_id: id,
            registry: reg,
        }
    }

    // -- Completeness: valid ExternCast coercions --

    #[test]
    fn extern_cast_basic_coercion() {
        // UserDefined(A, [T]) -> List<T>
        let CastSetup {
            from_id: id,
            registry,
        } = make_cast_registry(1, |p| arr(p[0].clone(), 3));
        let mut sources = Sources::new();
        let mut s = Solver::new(&mut sources);

        let from = TyTerm::UserDefined {
            id,
            type_args: vec![TyTerm::I64],
            effect_args: vec![],
            identity_args: vec![],
        };
        let to = arr(TyTerm::I64, 3);
        assert!(s.unify_ty(&from, &to, Covariant, &registry).is_ok());
    }

    #[test]
    fn extern_cast_with_param_resolution() {
        // UserDefined(A, [T]) -> List<T>, where T is a fresh param on the consumer side
        let CastSetup {
            from_id: id,
            registry,
        } = make_cast_registry(1, |p| arr(p[0].clone(), 3));
        let mut sources = Sources::new();
        let mut s = Solver::new(&mut sources);

        let from = TyTerm::UserDefined {
            id,
            type_args: vec![TyTerm::I64],
            effect_args: vec![],
            identity_args: vec![],
        };
        let consumer_param = s.fresh_ty_var();
        let to = arr(consumer_param.clone(), 3);
        assert!(s.unify_ty(&from, &to, Covariant, &registry).is_ok());
        assert_eq!(s.resolve_ty(&consumer_param), TyTerm::I64);
    }

    #[test]
    fn extern_cast_no_type_params() {
        // UserDefined(A, []) -> Int
        let CastSetup {
            from_id: id,
            registry,
        } = make_cast_registry(0, |_| TyTerm::I64);
        let mut sources = Sources::new();
        let mut s = Solver::new(&mut sources);

        let from = TyTerm::UserDefined {
            id,
            type_args: vec![],
            effect_args: vec![],
            identity_args: vec![],
        };
        assert!(
            s.unify_ty(&from, &TyTerm::I64, Covariant, &registry)
                .is_ok()
        );
    }

    // -- Soundness: invalid ExternCast --

    #[test]
    fn extern_cast_wrong_target_fails() {
        // Rule: A -> List<T>, but expected String
        let CastSetup {
            from_id: id,
            registry,
        } = make_cast_registry(1, |p| arr(p[0].clone(), 3));
        let mut sources = Sources::new();
        let mut s = Solver::new(&mut sources);

        let from = TyTerm::UserDefined {
            id,
            type_args: vec![TyTerm::I64],
            effect_args: vec![],
            identity_args: vec![],
        };
        assert!(
            s.unify_ty(&from, &TyTerm::String, Covariant, &registry)
                .is_err()
        );
    }

    #[test]
    fn extern_cast_no_rule_fails() {
        // No cast rules registered
        let id = fresh_qref();
        let mut sources = Sources::new();
        let mut s = Solver::new(&mut sources);
        let registry = TypeRegistry::new();

        let from = TyTerm::UserDefined {
            id,
            type_args: vec![],
            effect_args: vec![],
            identity_args: vec![],
        };
        assert!(
            s.unify_ty(&from, &TyTerm::I64, Covariant, &registry)
                .is_err()
        );
    }

    #[test]
    fn extern_cast_invariant_not_attempted() {
        // ExternCast only works in covariant/contravariant, not invariant
        let CastSetup {
            from_id: id,
            registry,
        } = make_cast_registry(0, |_| TyTerm::I64);
        let mut sources = Sources::new();
        let mut s = Solver::new(&mut sources);

        let from = TyTerm::UserDefined {
            id,
            type_args: vec![],
            effect_args: vec![],
            identity_args: vec![],
        };
        assert!(
            s.unify_ty(&from, &TyTerm::I64, Invariant, &registry)
                .is_err()
        );
    }

    // -- Ambiguity --

    #[test]
    fn extern_cast_ambiguity_rejected() {
        // Bypass TypeRegistry duplicate check - inject two rules with same to head
        // directly into the type_registry to test try_extern_cast ambiguity detection.
        let i = acvus_utils::Interner::new();
        let id = fresh_qref();
        let fn_id_a = QualifiedRef::root(i.intern("cast_a"));
        let fn_id_b = QualifiedRef::root(i.intern("cast_b"));

        // Use a PolyBuilder for the CastRule variables, separate Solver for unification.
        let mut sources = Sources::new();
        let mut s = Solver::new(&mut sources);
        let mut builder = PolyBuilder::new();
        let t1 = builder.fresh_ty_var();
        let rule_a = CastRule {
            from: TyTerm::UserDefined {
                id,
                type_args: vec![t1.clone()],
                effect_args: vec![],
                identity_args: vec![],
            },
            to: arr(t1, 3),
            fn_ref: fn_id_a,
        };
        let t2 = builder.fresh_ty_var();
        let rule_b = CastRule {
            from: TyTerm::UserDefined {
                id,
                type_args: vec![t2.clone()],
                effect_args: vec![],
                identity_args: vec![],
            },
            to: arr(t2, 3),
            fn_ref: fn_id_b,
        };

        // Build registry manually (bypassing register_cast duplicate check)
        let mut reg = TypeRegistry::new();
        reg.register(UserDefinedDecl {
            qref: id,
            type_params: vec![TyVarBound::Any],
            effect_params: 0,
            identity_params: 0,
        });
        reg.from_rules.entry(id).or_default().push(rule_a);
        reg.from_rules.entry(id).or_default().push(rule_b);

        let from = TyTerm::UserDefined {
            id,
            type_args: vec![TyTerm::I64],
            effect_args: vec![],
            identity_args: vec![],
        };
        assert!(
            s.unify_ty(&from, &arr(TyTerm::I64, 3), Covariant, &reg)
                .is_err()
        );
    }

    // -- TypeRegistry cast rules --

    #[test]
    #[should_panic(expected = "duplicate")]
    fn cast_registry_duplicate_panics() {
        let i = acvus_utils::Interner::new();
        let id = fresh_qref();
        let fn_id_a = QualifiedRef::root(i.intern("cast_a"));
        let fn_id_b = QualifiedRef::root(i.intern("cast_b"));
        let mut builder = PolyBuilder::new();
        let t = builder.fresh_ty_var();

        let mut reg = TypeRegistry::new();
        reg.register(UserDefinedDecl {
            qref: id,
            type_params: vec![TyVarBound::Any],
            effect_params: 0,
            identity_params: 0,
        });
        reg.register_cast(CastRule {
            from: TyTerm::UserDefined {
                id,
                type_args: vec![t.clone()],
                effect_args: vec![],
                identity_args: vec![],
            },
            to: arr(t.clone(), 3),
            fn_ref: fn_id_a,
        });
        // Same from_id + same to head (List) -> panic
        let mut builder2 = PolyBuilder::new();
        let t2 = builder2.fresh_ty_var();
        reg.register_cast(CastRule {
            from: TyTerm::UserDefined {
                id,
                type_args: vec![t2.clone()],
                effect_args: vec![],
                identity_args: vec![],
            },
            to: arr(t2, 3),
            fn_ref: fn_id_b,
        });
    }

    #[test]
    fn cast_registry_different_to_head_ok() {
        let i = acvus_utils::Interner::new();
        let id = fresh_qref();
        let fn_id_a = QualifiedRef::root(i.intern("cast_a"));
        let fn_id_b = QualifiedRef::root(i.intern("cast_b"));

        let mut reg = TypeRegistry::new();
        reg.register(UserDefinedDecl {
            qref: id,
            type_params: vec![TyVarBound::Any],
            effect_params: 0,
            identity_params: 0,
        });
        let mut builder1 = PolyBuilder::new();
        let t1 = builder1.fresh_ty_var();
        reg.register_cast(CastRule {
            from: TyTerm::UserDefined {
                id,
                type_args: vec![t1.clone()],
                effect_args: vec![],
                identity_args: vec![],
            },
            to: arr(t1, 3),
            fn_ref: fn_id_a,
        });
        // Different to head (Option vs List) -> ok
        let mut builder2 = PolyBuilder::new();
        let t2 = builder2.fresh_ty_var();
        reg.register_cast(CastRule {
            from: TyTerm::UserDefined {
                id,
                type_args: vec![t2.clone()],
                effect_args: vec![],
                identity_args: vec![],
            },
            to: TyTerm::Option(Box::new(t2)),
            fn_ref: fn_id_b,
        });
        assert_eq!(reg.rules_from(id).len(), 2);
    }

    // -- is_data ---------------------------------------------------------

    #[test]
    fn data_scalars_and_containers_of_scalars() {
        assert!(Ty::I64.is_data());
        assert!(Ty::String.is_data());
        assert!(arr(Ty::I64, 3).is_data());
        assert!(Ty::Option(Box::new(Ty::String)).is_data());
        assert!(Ty::Tuple(vec![Ty::I64, Ty::String]).is_data());
        assert!(arr(Ty::Option(Box::new(arr(Ty::I64, 2))), 3).is_data());
    }

    #[test]
    fn data_user_defined_is_data() {
        assert!(test_user_defined().is_data());
        assert!(arr(test_user_defined(), 3).is_data());
    }

    #[test]
    fn data_user_defined_over_fn_is_not_data() {
        let fn_ty = Ty::Fn {
            params: vec![],
            ret: Box::new(Ty::I64),
            captures: vec![],
            effect: Effect::OPAQUE.into(),
        };
        let Ty::UserDefined {
            id,
            effect_args,
            identity_args,
            ..
        } = test_user_defined()
        else {
            unreachable!()
        };
        let over_fn = Ty::UserDefined {
            id,
            type_args: vec![fn_ty],
            effect_args,
            identity_args,
        };
        assert!(!over_fn.is_data());
    }

    #[test]
    fn data_fn_handle_order_ref_are_not_data() {
        let fn_ty = Ty::Fn {
            params: vec![],
            ret: Box::new(Ty::I64),
            captures: vec![],
            effect: Effect::PURE.into(),
        };
        assert!(!fn_ty.is_data());
        assert!(!Ty::Handle(Box::new(Ty::I64)).is_data());
        assert!(!Ty::Order.is_data());
        assert!(!Ty::Ref(Mutability::Shared, Box::new(Ty::I64)).is_data());
        assert!(!Ty::error().is_data());
    }

    #[test]
    fn data_container_holding_fn_is_not_data() {
        let fn_ty = Ty::Fn {
            params: vec![],
            ret: Box::new(Ty::I64),
            captures: vec![],
            effect: Effect::PURE.into(),
        };
        assert!(!arr(fn_ty.clone(), 3).is_data());
        assert!(!Ty::Option(Box::new(fn_ty.clone())).is_data());
        assert!(!Ty::Tuple(vec![Ty::I64, fn_ty.clone()]).is_data());
        let interner = Interner::new();
        let obj = Ty::Object(FxHashMap::from_iter([(interner.intern("cb"), fn_ty)]));
        assert!(!obj.is_data());
    }
}
