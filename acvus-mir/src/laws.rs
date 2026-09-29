//! The algebraic laws and the postconditions an extern declares (RFC-0082
//! rules 2 to 4), read by a MIR pass through a call's callee.
//!
//! A law or a postcondition is the author's promise, trusted as an effect
//! is. Nothing here checks that a function is associative or that its
//! result is its argument's length, and neither is inferred from a
//! handler's body, which is Rust and opaque to the compiler (RFC-0082 rule
//! 5). A debug build of the extension evaluates each postcondition at the
//! function's return; that check is `#[extern_fn]`'s, not this module's.

use acvus_ast::Literal;
use rustc_hash::FxHashMap;

use acvus_utils::QualifiedRef;

use crate::decl::{
    BinaryLaws, Declarations, FoldLaw, Identity, Laws, Postcondition, Reaches, Returns, Scheme,
};
use crate::ir::{Callee, Chosen};
use crate::means::{EntryReading, Means, Named, Stmt as MeansStmt, Term as MeansTerm};
use crate::step::Step;
use crate::ty::{
    EffectArg, EffectTerm, IdentityTerm, IntTy, LenTerm, Mutability, ObjectTy, Poly, TyId, TyKind,
    TypeArg, Types,
};

impl Returns {
    fn meet(self, other: Returns) -> Returns {
        match (self, other) {
            (Returns::Unstated, _) | (_, Returns::Unstated) => Returns::Unstated,
            (Returns::Stated, _) | (_, Returns::Stated) => Returns::Stated,
            (Returns::Total, Returns::Total) => Returns::Total,
        }
    }
}

/// An extern and one of its impls, numbered as `Callee::Extern`
/// numbers them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ExternImpl {
    pub id: QualifiedRef,
    pub r#impl: usize,
}

/// An impl's [`Laws`] with each extern they name taken at the impl
/// whose types are the declaring impl's.
#[derive(Debug, Clone, PartialEq, Default)]
pub enum ResolvedLaws {
    #[default]
    None,
    Binary(ResolvedBinary),
    /// A law stated on `f(a: T, b: T) -> Option<T>`: the law of `f`'s
    /// extension to `Option<T>`, `Some(a) · Some(b)` being `f(a, b)` and
    /// `None` absorbing from either side, its identity `Some` of the one
    /// stated at `T` (RFC-0082 rule 2). No reader reads it as a law over
    /// `T`.
    Extension(ResolvedBinary),
    Fold(ResolvedFold),
    TotalOrder,
    /// `inverse = g`, with `g` taken at the impl whose state and payload
    /// are the declaring impl's.
    Inverse(ExternImpl),
    Payload,
    Equivalence,
    Step(Step),
}

#[derive(Debug, Clone, PartialEq)]
pub struct ResolvedBinary {
    pub associative: bool,
    pub commutative: bool,
    pub identity: Option<ResolvedIdentity>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ResolvedIdentity {
    Const(Literal),
    Extern(ExternImpl),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResolvedFold {
    pub combine: ExternImpl,
    pub identity: ExternImpl,
    pub commutative: bool,
}

/// Which extern of a law [`resolve`] looks for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LawRole {
    /// A binary law's `identity = e`: `e() -> T` at the declaration's `T`.
    Identity,
    /// A fold's `combine = g`: `g(s: &mut S, part: S)` at the declaration's
    /// `S`.
    FoldCombine,
    /// A fold's `identity = e`: `e() -> S` at the declaration's `S`.
    FoldIdentity,
    /// An inverse's `g`: `g(s: &mut S, x: X)` at the declaration's `S` and
    /// `X`.
    Inverse,
    /// An extern a step's term calls, whose impl the call's argument
    /// types choose where the step runs.
    Step,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Unresolved {
    /// The declaring impl's type is not the signature its law is
    /// stated over.
    UnfitDeclaration,
    /// `named` is no extern, or none of its impls, or more than one,
    /// has the type `role` asks at the declaring impl's types.
    NoFittingImpl { role: LawRole, named: QualifiedRef },
}

/// `laws`, declared on an impl of type `declaring`, with each extern
/// they name taken at the impl a call at `declaring`'s types chooses:
/// its one concrete impl those types have the shape of, and otherwise
/// its generic impl. A name picks the extern and the types pick its
/// impl: `num::min` over `i32` names `i32::MAX`, and a fold on `push`
/// at `Vec<T>` combines through `extend` at `Vec<T>`.
pub fn resolve(laws: &Laws, declaring: TyId<Poly>, decls: &Declarations) -> Result<ResolvedLaws, Unresolved> {
    let poly = decls.types();
    let TyKind::Fn { params, ret, .. } = poly.kind(declaring) else {
        return match laws {
            Laws::None => Ok(ResolvedLaws::None),
            Laws::Binary(_)
            | Laws::Fold(_)
            | Laws::TotalOrder
            | Laws::Inverse(_)
            | Laws::Payload
            | Laws::Equivalence
            | Laws::Step(_) => Err(Unresolved::UnfitDeclaration),
        };
    };
    let ret = *ret;
    let impl_of = |role: LawRole, named: QualifiedRef, wanted: &Wanted| {
        let unfit = Unresolved::NoFittingImpl { role, named };
        let scheme = decls.scheme(named).ok_or(unfit)?;
        let at = |r#impl: usize| ExternImpl { id: named, r#impl };
        let Some(impls) = &scheme.impls else {
            return match wanted.fits(poly, scheme.ty) {
                true => Ok(at(0)),
                false => Err(unfit),
            };
        };
        let concrete: Vec<usize> = impls
            .concrete
            .iter()
            .enumerate()
            .filter(|(_, r#impl)| wanted.fits(poly, r#impl.ty))
            .map(|(r#impl, _)| r#impl)
            .collect();
        match concrete[..] {
            [r#impl] => Ok(at(r#impl)),
            [] if wanted.fits(poly, scheme.ty) => match &impls.generic {
                Some(_) => Ok(at(impls.concrete.len())),
                None if impls.concrete.is_empty() => Ok(at(0)),
                None => Err(unfit),
            },
            _ => Err(unfit),
        }
    };
    match laws {
        Laws::None => Ok(ResolvedLaws::None),
        Laws::Binary(BinaryLaws {
            associative,
            commutative,
            identity,
        }) => {
            let (value, extension) = match (binary_fits(poly, params, ret), extension_of(poly, params, ret)) {
                (true, _) => (ret, false),
                (false, Some(value)) => (value, true),
                (false, None) => return Err(Unresolved::UnfitDeclaration),
            };
            let resolved = ResolvedBinary {
                associative: *associative,
                commutative: *commutative,
                identity: match identity {
                    None => None,
                    Some(Identity::Const(constant)) => Some(ResolvedIdentity::Const(constant.clone())),
                    Some(Identity::Extern(named)) => Some(ResolvedIdentity::Extern(impl_of(
                        LawRole::Identity,
                        *named,
                        &Wanted::Returning(value),
                    )?)),
                },
            };
            Ok(match extension {
                false => ResolvedLaws::Binary(resolved),
                true => ResolvedLaws::Extension(resolved),
            })
        }
        Laws::Fold(FoldLaw {
            combine,
            identity,
            commutative,
        }) => {
            let Some(TyKind::Ref(Mutability::Mut, state)) = params.first().map(|param| poly.kind(*param)) else {
                return Err(Unresolved::UnfitDeclaration);
            };
            let state = state.ty();
            Ok(ResolvedLaws::Fold(ResolvedFold {
                combine: impl_of(LawRole::FoldCombine, *combine, &Wanted::Combining(state))?,
                identity: impl_of(LawRole::FoldIdentity, *identity, &Wanted::Returning(state))?,
                commutative: *commutative,
            }))
        }
        Laws::TotalOrder => {
            match compares_two_shared(poly, params) && matches!(poly.kind(ret), TyKind::Int(IntTy::I64)) {
                true => Ok(ResolvedLaws::TotalOrder),
                false => Err(Unresolved::UnfitDeclaration),
            }
        }
        Laws::Inverse(restore) => {
            let ([state], TyKind::Option(payload)) = (&params[..], poly.kind(ret)) else {
                return Err(Unresolved::UnfitDeclaration);
            };
            let TyKind::Ref(Mutability::Mut, state) = poly.kind(*state) else {
                return Err(Unresolved::UnfitDeclaration);
            };
            Ok(ResolvedLaws::Inverse(impl_of(
                LawRole::Inverse,
                *restore,
                &Wanted::Restoring {
                    state: state.ty(),
                    payload: *payload,
                },
            )?))
        }
        Laws::Payload => match &params[..] {
            [option] if poly.kind(*option) == &TyKind::Option(ret) => Ok(ResolvedLaws::Payload),
            _ => Err(Unresolved::UnfitDeclaration),
        },
        Laws::Equivalence => match compares_two_shared(poly, params) && poly.kind(ret) == &TyKind::Bool {
            true => Ok(ResolvedLaws::Equivalence),
            false => Err(Unresolved::UnfitDeclaration),
        },
        Laws::Step(step) => {
            if !step_fits(poly, step, params) {
                return Err(Unresolved::UnfitDeclaration);
            }
            for named in step.named_externs() {
                if decls.scheme(named).is_none() {
                    return Err(Unresolved::NoFittingImpl {
                        role: LawRole::Step,
                        named,
                    });
                }
            }
            Ok(ResolvedLaws::Step(step.clone()))
        }
    }
}

/// `f(a: &T, b: &T)`: two shared references to one type.
fn compares_two_shared(poly: &Types<Poly>, params: &[TyId<Poly>]) -> bool {
    match params {
        [a, b] => match (poly.kind(*a), poly.kind(*b)) {
            (TyKind::Ref(Mutability::Shared, a), TyKind::Ref(Mutability::Shared, b)) => a.ty() == b.ty(),
            _ => false,
        },
        _ => false,
    }
}

/// RFC-0099 rule 1, over the declaration's type: the stream is a parameter,
/// a closure a step calls is a parameter of function type, and a value
/// parameter it reads is one of neither.
fn step_fits(poly: &Types<Poly>, step: &Step, params: &[TyId<Poly>]) -> bool {
    let stream = step.stream().param;
    let is_fn = |at: usize| matches!(params.get(at).map(|param| poly.kind(*param)), Some(TyKind::Fn { .. }));
    let mut values = Vec::new();
    for term in step.terms() {
        term.visit(&mut |term| {
            if let crate::step::Term::ValueParam(at) = term {
                values.push(*at);
            }
        });
    }
    stream < params.len()
        && step
            .called_closures()
            .into_iter()
            .all(|at| at != stream && is_fn(at))
        && values
            .into_iter()
            .all(|at| at != stream && at < params.len() && !is_fn(at))
}

/// RFC-0082 rule 2.
fn binary_fits(poly: &Types<Poly>, params: &[TyId<Poly>], ret: TyId<Poly>) -> bool {
    let [a, b] = params else {
        return false;
    };
    let str_view = |ty: TyId<Poly>| match poly.kind(ty) {
        TyKind::Ref(Mutability::Shared, lent) => poly.kind(lent.ty()) == &TyKind::Str,
        _ => false,
    };
    let over_values = a == b && *a == ret;
    let over_string_views = poly.kind(ret) == &TyKind::String && str_view(*a) && str_view(*b);
    over_values || over_string_views
}

/// RFC-0082 rule 2's `f(a: T, b: T) -> Option<T>`: `T`, the type a law
/// stated there is `f`'s extension's over.
fn extension_of(poly: &Types<Poly>, params: &[TyId<Poly>], ret: TyId<Poly>) -> Option<TyId<Poly>> {
    let [a, b] = params else {
        return None;
    };
    let TyKind::Option(payload) = poly.kind(ret) else {
        return None;
    };
    (a == b && a == payload).then_some(*a)
}

/// Whether `means`, declared on an impl of type `declaring`, reads each
/// parameter as its type lends it (RFC-0104 rule 1): `*x` of a reference
/// parameter, a parameter by value, and the entry `x[k]` of a `&` or `&mut`
/// table at a parameter by value or by `&`. What else the term computes is
/// typed where the model checker runs it.
pub fn means_fits(poly: &Types<Poly>, means: &Means, declaring: TyId<Poly>) -> bool {
    let TyKind::Fn { params, .. } = poly.kind(declaring) else {
        return false;
    };
    let is_ref = |at: usize| matches!(params.get(at).map(|param| poly.kind(*param)), Some(TyKind::Ref(..)));
    let is_value = |at: usize| {
        params
            .get(at)
            .is_some_and(|param| !matches!(poly.kind(*param), TyKind::Ref(..)))
    };
    let entry_fits = means
        .entry
        .is_none_or(|entry| is_ref(entry.table) && entry.key < params.len() && entry.key != entry.table);
    let mut reads_fit = true;
    let terms = means
        .stmts
        .iter()
        .map(|stmt| match stmt {
            MeansStmt::Let { value, .. } | MeansStmt::Store(value) => value,
        })
        .chain([&means.result]);
    for term in terms {
        term.visit(&mut |term| {
            reads_fit &= match term {
                MeansTerm::Lent(at) => is_ref(*at),
                MeansTerm::Param(at) => is_value(*at),
                MeansTerm::Entry | MeansTerm::LendEntry(_) => means.entry.is_some(),
                _ => true,
            };
        });
    }
    let stores_fit = means.entry.is_some()
        || !means.stmts.iter().any(|stmt| matches!(stmt, MeansStmt::Store(_)));
    entry_fits && reads_fit && stores_fit
}

/// The type a law asks of an extern it names, at the declaring impl's
/// types.
enum Wanted {
    /// `e() -> T`.
    Returning(TyId<Poly>),
    /// `g(s: &mut S, part: S)`.
    Combining(TyId<Poly>),
    /// `g(s: &mut S, x: X)`.
    Restoring { state: TyId<Poly>, payload: TyId<Poly> },
}

impl Wanted {
    /// Whether a call at the wanted types has the shape of an impl of
    /// type `candidate`.
    fn fits(&self, poly: &Types<Poly>, candidate: TyId<Poly>) -> bool {
        let TyKind::Fn { params, ret, .. } = poly.kind(candidate) else {
            return false;
        };
        let mut pattern = PatternMatch::new(poly);
        let returns_unit = poly.kind(*ret) == &TyKind::Unit;
        match (self, &params[..]) {
            (Self::Returning(value), []) => pattern.matches(*value, *ret),
            (Self::Combining(state), [into, part]) if returns_unit => match poly.kind(*into) {
                TyKind::Ref(Mutability::Mut, into) => {
                    pattern.matches(*state, into.ty()) && pattern.matches(*state, *part)
                }
                _ => false,
            },
            (Self::Restoring { state, payload }, [into, x]) if returns_unit => match poly.kind(*into) {
                TyKind::Ref(Mutability::Mut, into) => {
                    pattern.matches(*state, into.ty()) && pattern.matches(*payload, *x)
                }
                _ => false,
            },
            _ => false,
        }
    }
}

/// Whether a declared type has the shape of a declared pattern. A pattern
/// variable stands for any one type, the same wherever it recurs; effect,
/// length and identity variables of the pattern stand for any effect,
/// length or identity. A variable of the matched type matches only a
/// pattern variable. A decided `#` mark matches only itself, and a
/// pattern's undecided mark takes either (types §8).
struct PatternMatch<'a> {
    poly: &'a Types<Poly>,
    bound: FxHashMap<u32, TyId<Poly>>,
}

impl<'a> PatternMatch<'a> {
    fn new(poly: &'a Types<Poly>) -> Self {
        PatternMatch {
            poly,
            bound: FxHashMap::default(),
        }
    }

    fn matches(&mut self, ty: TyId<Poly>, pattern: TyId<Poly>) -> bool {
        acvus_utils::grow(|| self.matches_level(ty, pattern))
    }

    fn all(&mut self, pairs: impl IntoIterator<Item = (TyId<Poly>, TyId<Poly>)>) -> bool {
        pairs.into_iter().all(|(ty, pattern)| self.matches(ty, pattern))
    }

    fn arg_matches(&mut self, arg: &TypeArg<Poly>, pattern: &TypeArg<Poly>) -> bool {
        match (arg, pattern) {
            (TypeArg::Uniform(ty) | TypeArg::Specialized(ty) | TypeArg::Open(_, ty), TypeArg::Open(_, p))
            | (TypeArg::Uniform(ty), TypeArg::Uniform(p))
            | (TypeArg::Specialized(ty), TypeArg::Specialized(p)) => self.matches(*ty, *p),
            (TypeArg::Uniform(_) | TypeArg::Specialized(_) | TypeArg::Open(..), _) => false,
        }
    }

    fn matches_level(&mut self, ty: TyId<Poly>, pattern: TyId<Poly>) -> bool {
        let poly = self.poly;
        match (poly.kind(ty), poly.kind(pattern)) {
            (_, TyKind::Var(var)) => match self.bound.get(var) {
                Some(bound) => *bound == ty,
                None => {
                    self.bound.insert(*var, ty);
                    true
                }
            },
            (TyKind::Var(_), _) => false,
            (TyKind::Int(a), TyKind::Int(b)) => a == b,
            (TyKind::Float, TyKind::Float)
            | (TyKind::Char, TyKind::Char)
            | (TyKind::String, TyKind::String)
            | (TyKind::Str, TyKind::Str)
            | (TyKind::Bool, TyKind::Bool)
            | (TyKind::Unit, TyKind::Unit)
            | (TyKind::Never, TyKind::Never)
            | (TyKind::Order, TyKind::Order) => true,
            (TyKind::Array(element, len), TyKind::Array(p_element, p_len)) => {
                len_matches(len, p_len) && self.matches(*element, *p_element)
            }
            (TyKind::Option(inner), TyKind::Option(p_inner))
            | (TyKind::Handle(inner), TyKind::Handle(p_inner))
            | (TyKind::Slice(inner), TyKind::Slice(p_inner)) => self.matches(*inner, *p_inner),
            (TyKind::Result(ok, err), TyKind::Result(p_ok, p_err)) => {
                self.matches(*ok, *p_ok) && self.matches(*err, *p_err)
            }
            (TyKind::Ref(m, target), TyKind::Ref(p_m, p_target)) => m == p_m && self.arg_matches(target, p_target),
            (TyKind::Tuple(items), TyKind::Tuple(p_items)) => {
                items.len() == p_items.len() && self.all(items.iter().copied().zip(p_items.iter().copied()))
            }
            (TyKind::Object(object), TyKind::Object(p_object)) => {
                let same_kind = match (object, p_object) {
                    (ObjectTy::Structural(..), ObjectTy::Structural(..)) => true,
                    (ObjectTy::Declared(id, _), ObjectTy::Declared(p_id, _)) => id == p_id,
                    (ObjectTy::Structural(..), ObjectTy::Declared(..))
                    | (ObjectTy::Declared(..), ObjectTy::Structural(..)) => false,
                };
                let (fields, p_fields) = (object.fields(), p_object.fields());
                same_kind
                    && fields.len() == p_fields.len()
                    && fields.iter().zip(p_fields.iter()).all(|((a, _), (b, _))| a == b)
                    && self.all(fields.iter().map(|(_, ty)| *ty).zip(p_fields.iter().map(|(_, ty)| *ty)))
            }
            (
                TyKind::Fn {
                    params, ret, effect, ..
                },
                TyKind::Fn {
                    params: p_params,
                    ret: p_ret,
                    effect: p_effect,
                    ..
                },
            ) => {
                params.len() == p_params.len()
                    && effect_matches(effect, p_effect)
                    && self.all(params.iter().copied().zip(p_params.iter().copied()))
                    && self.matches(*ret, *p_ret)
            }
            (
                TyKind::UserDefined {
                    id,
                    type_args,
                    effect_args,
                    identity_args,
                },
                TyKind::UserDefined {
                    id: p_id,
                    type_args: p_type_args,
                    effect_args: p_effect_args,
                    identity_args: p_identity_args,
                },
            ) => {
                id == p_id
                    && type_args.len() == p_type_args.len()
                    && effect_args.len() == p_effect_args.len()
                    && identity_args.len() == p_identity_args.len()
                    && effect_args.iter().zip(p_effect_args.iter()).all(|(a, p)| {
                        matches!(
                            (a, p),
                            (EffectArg::Uniform(_), EffectArg::Uniform(_))
                                | (EffectArg::Specialized(_), EffectArg::Specialized(_))
                        ) && effect_matches(a.effect(), p.effect())
                    })
                    && identity_args.iter().zip(p_identity_args.iter()).all(|(a, p)| match (a, p) {
                        (_, IdentityTerm::Var(_)) => true,
                        (IdentityTerm::Var(_), IdentityTerm::Known(_)) => false,
                        (IdentityTerm::Known(a), IdentityTerm::Known(p)) => a == p,
                    })
                    && type_args
                        .iter()
                        .zip(p_type_args.iter())
                        .all(|(a, p)| self.arg_matches(a, p))
            }
            (
                TyKind::Enum { name, variants, .. },
                TyKind::Enum {
                    name: p_name,
                    variants: p_variants,
                    ..
                },
            ) => {
                name == p_name
                    && variants.len() == p_variants.len()
                    && variants
                        .iter()
                        .zip(p_variants.iter())
                        .all(|((tag, payload), (p_tag, p_payload))| {
                            tag == p_tag
                                && match (payload, p_payload) {
                                    (Some(payload), Some(p_payload)) => self.matches(*payload, *p_payload),
                                    (None, None) => true,
                                    (Some(_), None) | (None, Some(_)) => false,
                                }
                        })
            }
            _ => false,
        }
    }
}

fn effect_matches(effect: &EffectTerm<Poly>, pattern: &EffectTerm<Poly>) -> bool {
    match (effect, pattern) {
        (_, EffectTerm::Var(_)) => true,
        (EffectTerm::Var(_), EffectTerm::Known(_)) => false,
        (EffectTerm::Known(e), EffectTerm::Known(p)) => e == p,
    }
}

fn len_matches(len: &LenTerm<Poly>, pattern: &LenTerm<Poly>) -> bool {
    match (len, pattern) {
        (_, LenTerm::Var(_)) => true,
        (LenTerm::Var(_), LenTerm::Known(_)) => false,
        (LenTerm::Known(n), LenTerm::Known(k)) => n == k,
    }
}

/// What one impl of an extern declares.
#[derive(Debug, Default)]
struct Declared {
    laws: ResolvedLaws,
    ensures: Vec<Postcondition>,
    reaches: Reaches,
    returns: Returns,
    means: Option<Means>,
}

impl Declared {
    /// # Panics
    /// A law or a `means` of `id` does not fit its declaration, which
    /// `acvus_extern::Externs::combine` refuses.
    fn of(id: QualifiedRef, ty: TyId<Poly>, facts: &crate::decl::DeclaredFacts, decls: &Declarations) -> Self {
        Declared {
            laws: resolve(&facts.laws, ty, decls).unwrap_or_else(|unresolved| {
                panic!(
                    "a law of {id:?} does not resolve ({unresolved:?}), and combining the \
                     registries refuses it"
                )
            }),
            ensures: facts.ensures.to_vec(),
            reaches: facts.reaches.clone(),
            returns: facts.returns,
            means: facts
                .means
                .as_ref()
                .inspect(|means| {
                    assert!(
                        means_fits(decls.types(), means, ty),
                        "the `means` of {id:?} reads a parameter as its type does not lend it, \
                         and combining the registries refuses it"
                    )
                })
                .cloned(),
        }
    }
}

/// One impl of an extern: its type, and whether its declaration
/// requires an impl of a generic function (extern-fn §6), which a call names
/// in `Callee::Extern::required`.
#[derive(Debug, Clone)]
pub struct ImplType {
    pub ty: TyId<Poly>,
    pub requires: bool,
    pub generic: bool,
}

#[derive(Debug)]
pub struct LawTable {
    by_impl: FxHashMap<QualifiedRef, Vec<Declared>>,
    /// Numbered as `Callee::Extern` numbers an extern's impls.
    impl_types: FxHashMap<QualifiedRef, Vec<ImplType>>,
    /// The declaration of the `Equiv` keying marker (RFC-0098 rule 2),
    /// where the declarations name it.
    equiv: Option<QualifiedRef>,
    pull_generic_function: Option<QualifiedRef>,
}

impl LawTable {
    /// The laws of every impl of every declaration, numbered as
    /// `Callee::Extern` numbers them: the concrete impls in order, then
    /// the generic one; a declaration that lists no impls is impl 0.
    ///
    /// # Panics
    /// A law names an extern with no impl of the types it asks:
    /// `acvus_extern::Externs::combine` refuses such a registry by the same
    /// [`resolve`].
    pub fn of(decls: &Declarations) -> Self {
        let mut by_impl = FxHashMap::default();
        let mut impl_types = FxHashMap::default();
        for (id, scheme) in decls.functions() {
            by_impl.insert(id, declared_impls(id, scheme, decls));
            impl_types.insert(id, typed_impls(scheme));
        }
        let markers = decls.loop_markers();
        Self {
            by_impl,
            impl_types,
            equiv: markers.keying.map(|keying| keying.equiv),
            pull_generic_function: markers.pull,
        }
    }

    /// The step the impl a call names states (RFC-0099 rule 1); a call
    /// of a local function or through a value states none.
    pub fn step_of(&self, callee: &Callee) -> Option<&Step> {
        match &self.declared(callee)?.laws {
            ResolvedLaws::Step(step) => Some(step),
            _ => None,
        }
    }

    /// Each impl of the extern `id`, or none where `id` is no extern.
    pub fn impl_types(&self, id: QualifiedRef) -> &[ImplType] {
        self.impl_types.get(&id).map_or(&[], Vec::as_slice)
    }

    /// Whether `id` is the declaration of the `Equiv` keying marker
    /// (RFC-0098 rule 2): the marker is known by the declaration the
    /// registries named from its Rust type, not by how a script spells it.
    pub fn is_equiv(&self, id: QualifiedRef) -> bool {
        self.equiv == Some(id)
    }

    /// Whether a call names an impl of `iter::next`, the generic function a
    /// pull loop's header calls (RFC-0089 rule 1).
    pub fn pulls(&self, callee: &Callee) -> bool {
        matches!(callee, Callee::Extern { id, .. } if Some(*id) == self.pull_generic_function)
    }

    pub fn of_callee(&self, callee: &Callee) -> &ResolvedLaws {
        static NONE: ResolvedLaws = ResolvedLaws::None;
        match self.declared(callee) {
            Some(declared) => &declared.laws,
            None => &NONE,
        }
    }

    /// The postconditions the impl a call names declares; a call of a
    /// local function or through a value has none.
    pub fn postconditions_of(&self, callee: &Callee) -> &[Postcondition] {
        match self.declared(callee) {
            Some(declared) => &declared.ensures,
            None => &[],
        }
    }

    /// What a call of the impl `callee` names reaches through its
    /// reference arguments; a call of a local function or through a value
    /// states nothing, and reaches all they lend.
    pub fn reaches_of(&self, callee: &Callee) -> &Reaches {
        static LENT: Reaches = Reaches::Lent;
        match self.declared(callee) {
            Some(declared) => &declared.reaches,
            None => &LENT,
        }
    }

    /// The term the impl a call names states (RFC-0104); a call of a
    /// local function or through a value states none.
    pub fn means_of(&self, callee: &Callee) -> Option<&Means> {
        self.declared(callee)?.means.as_ref()
    }

    pub fn means_of_impl(&self, r#impl: ExternImpl) -> Option<&Means> {
        self.by_impl
            .get(&r#impl.id)?
            .get(r#impl.r#impl)?
            .means
            .as_ref()
    }

    /// The argument whose lent value the result of the impl a call
    /// names is, where its term is `*x` alone (RFC-0104 rule 5).
    pub fn lent_value_of(&self, callee: &Callee) -> Option<usize> {
        self.means_of(callee)?.lent_value()
    }

    /// What a call of the impl does to the entry its term names
    /// (RFC-0098 rule 5).
    pub fn entry_access(&self, callee: &Callee) -> Option<EntryReading> {
        let means = self.means_of(callee)?;
        crate::means::entry_access(means, &|named| self.named_reading(named))
    }

    /// How a term reads a call of `named` it holds: by the one term every
    /// impl of `named` states, or by the `payload` law every impl
    /// states. Impls that state different terms leave the call opaque,
    /// since a term names an extern and not its impl.
    fn named_reading(&self, named: QualifiedRef) -> Option<Named<'_>> {
        let impls = self.by_impl.get(&named)?;
        let (first, rest) = impls.split_first()?;
        if let Some(means) = &first.means
            && rest.iter().all(|other| other.means.as_ref() == Some(means))
        {
            return Some(Named::Means(means));
        }
        impls
            .iter()
            .all(|declared| declared.means.is_none() && declared.laws == ResolvedLaws::Payload)
            .then_some(Named::Payload)
    }

    /// How a call ends, read over the impl tree it chose (RFC-0082
    /// rule 9). A `returns` impl is not descended into: RFC-0082 rule 8
    /// makes `returns` a promise of the whole call, read by the impl the
    /// call names.
    pub fn returns_of(&self, callee: &Callee) -> Returns {
        let Callee::Extern {
            id,
            r#impl,
            required,
        } = callee
        else {
            return Returns::Unstated;
        };
        self.returns_over(*id, *r#impl, required)
    }

    fn returns_over(&self, id: QualifiedRef, r#impl: usize, required: &[Chosen]) -> Returns {
        match self.declared_at(id, r#impl).returns {
            Returns::Unstated => Returns::Unstated,
            Returns::Stated => Returns::Stated,
            Returns::Total => required
                .iter()
                .map(|chosen| self.returns_over(chosen.generic_function, chosen.r#impl, &chosen.required))
                .fold(Returns::Total, Returns::meet),
        }
    }

    fn declared(&self, callee: &Callee) -> Option<&Declared> {
        let Callee::Extern { id, r#impl, .. } = callee else {
            return None;
        };
        Some(self.declared_at(*id, *r#impl))
    }

    fn declared_at(&self, id: QualifiedRef, r#impl: usize) -> &Declared {
        let Some(impls) = self.by_impl.get(&id) else {
            panic!("the law table was built without the extern {id:?} a call names")
        };
        let Some(declared) = impls.get(r#impl) else {
            panic!(
                "the extern {id:?} has {} impls, and a call names impl {}",
                impls.len(),
                r#impl
            )
        };
        declared
    }
}

/// Each impl's declared facts, numbered as `LawTable` numbers them.
fn declared_impls(id: QualifiedRef, scheme: &Scheme, decls: &Declarations) -> Vec<Declared> {
    let Some(impls) = &scheme.impls else {
        return vec![Declared::of(id, scheme.ty, &scheme.facts, decls)];
    };
    let mut declared: Vec<Declared> = impls
        .concrete
        .iter()
        .map(|r#impl| Declared::of(id, r#impl.ty, &r#impl.facts, decls))
        .chain(
            impls
                .generic
                .as_ref()
                .map(|generic| Declared::of(id, scheme.ty, &generic.facts, decls)),
        )
        .collect();
    if declared.is_empty() {
        declared.push(Declared::default());
    }
    declared
}

fn typed_impls(scheme: &Scheme) -> Vec<ImplType> {
    let Some(impls) = &scheme.impls else {
        return vec![ImplType {
            ty: scheme.ty,
            requires: !scheme.requires.is_empty(),
            generic: true,
        }];
    };
    let mut typed: Vec<ImplType> = impls
        .concrete
        .iter()
        .map(|r#impl| ImplType {
            ty: r#impl.ty,
            requires: !r#impl.requires.is_empty(),
            generic: false,
        })
        .collect();
    if impls.generic.is_some() || typed.is_empty() {
        typed.push(ImplType {
            ty: scheme.ty,
            requires: !scheme.requires.is_empty(),
            generic: true,
        });
    }
    typed
}
