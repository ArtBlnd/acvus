//! The algebraic laws and the postconditions an extern declares (RFC-0082
//! rules 2 to 4), read by a MIR pass through a call's callee, and the
//! weight it states (RFC-0066 rule 8), which only `analysis::cost` reads
//! and no pass does.
//!
//! A law or a postcondition is the author's promise, trusted as an effect
//! is. Nothing here checks that a function is associative or that its
//! result is its argument's length, and neither is inferred from a
//! handler's body, which is Rust and opaque to the compiler (RFC-0082 rule
//! 5). A debug build of the extension evaluates each postcondition at the
//! function's return; that check is `#[extern_fn]`'s, not this module's.

use acvus_ast::Literal;
use rustc_hash::FxHashMap;

use crate::graph::{FnKind, Function, QualifiedRef};
use crate::ir::Callee;
use crate::ty::{Mutability, PolyTy, matches_pattern};

#[derive(Debug, Clone, PartialEq, Default)]
pub enum Laws {
    #[default]
    None,
    /// `#[extern_fn(law(associative, commutative, identity = e))]` on
    /// `f(a: T, b: T) -> T`.
    Binary(BinaryLaws),
    /// `#[extern_fn(law(fold(combine = g, identity = e)))]` on
    /// `f(s: &mut S, x: X)`.
    Fold(FoldLaw),
    /// `#[extern_fn(law(total_order))]` on `f(a: &T, b: &T) -> i64`: `f`'s
    /// sign is a total order's comparison of `a` with `b`, under which equal
    /// values are one value (RFC-0082 rule 10).
    TotalOrder,
}

#[derive(Debug, Clone, PartialEq)]
pub struct BinaryLaws {
    pub associative: bool,
    pub commutative: bool,
    pub identity: Option<Identity>,
}

/// `acvus_extern::Externs::combine` holds a constant to the declaration's
/// result type and an extern to a registered one of no argument.
#[derive(Debug, Clone, PartialEq)]
pub enum Identity {
    Const(Literal),
    Extern(QualifiedRef),
}

/// The promise: a run of `f` over `s` equals `combine` applied to the
/// states that runs over its parts reach, each part started from what
/// `identity` returns. Without `commutative`, `combine` keeps the parts in
/// order. `acvus_extern::Externs::combine` holds `combine` to a registered
/// `g(s: &mut S, part: S)` and `identity` to a registered `e() -> S`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FoldLaw {
    pub combine: QualifiedRef,
    pub identity: QualifiedRef,
    pub commutative: bool,
}

/// `#[extern_fn(ensures(left relation right))]` (RFC-0082 rule 4).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Postcondition {
    pub left: PostTerm,
    pub relation: Relation,
    pub right: PostTerm,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Relation {
    /// `=`
    Eq,
    /// `≤`, written `<=`
    Le,
    /// `<`
    Lt,
}

/// A term of RFC-0066 rule 3 over one call: a constant, a parameter, the
/// result, the length of either, `+`, `−`, `×` and `max` of terms, and
/// `old(t)`, `t` as it stood when the call began (RFC-0082 rule 4). It
/// denotes an integer. A parameter is numbered as the declaration's acvus
/// parameters are, which is the order of a call's arguments.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PostTerm {
    Const(i128),
    Param(usize),
    Ret,
    /// The element count of a slice or a container.
    Len(Subject),
    Add(Box<PostTerm>, Box<PostTerm>),
    Sub(Box<PostTerm>, Box<PostTerm>),
    Mul(Box<PostTerm>, Box<PostTerm>),
    Max(Box<PostTerm>, Box<PostTerm>),
    /// The term as it stood when the call began, over the state of `&mut`
    /// parameters; `#[extern_fn]` refuses one that reads anything else.
    Old(Box<PostTerm>),
}

/// What `len(x)` reads: a parameter or the result.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Subject {
    Param(usize),
    Ret,
}

/// `#[extern_fn(copies(x))]` (RFC-0082 rule 10): `ret` is a value equal to
/// what reference parameter `param` lends, numbered as [`PostTerm::Param`]
/// numbers it, so a reader may read the result as that value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Copies {
    pub param: usize,
}

/// `#[extern_fn(reaches(p, ..))]` (RFC-0082 rule 7): what a call reaches
/// of the storages its reference arguments lend, which `analysis::loop_deps`
/// reads (RFC-0089 rule 4).
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum Reaches {
    /// No declaration: a call reaches all each reference argument lends it.
    #[default]
    Lent,
    /// A call reaches these places and nothing else through its reference
    /// arguments.
    Places(Vec<ReachedPlace>),
}

/// How a call of the instance ends, as its declaration states it (RFC-0082
/// rules 8 and 9): the author's promise, read by the instance a call names.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Returns {
    #[default]
    Unstated,
    /// `#[extern_fn(returns)]`: every call returns or traps. RFC-0089 rule 5
    /// reads it to run the call ahead of its iteration's control token.
    Stated,
    /// `#[extern_fn(total)]`: every call returns a value and never traps.
    /// `analysis::raise` reads it to let a call whose value nothing reads go
    /// (RFC-0048 rule 8).
    Total,
}

impl Returns {
    pub fn returns_or_traps(self) -> bool {
        match self {
            Returns::Unstated => false,
            Returns::Stated | Returns::Total => true,
        }
    }

    pub fn never_traps(self) -> bool {
        match self {
            Returns::Unstated | Returns::Stated => false,
            Returns::Total => true,
        }
    }
}

/// `x`, the whole of what reference parameter `x` lends, or `x[i]`, its
/// element at the `u64` parameter `i`. A parameter is numbered as
/// [`PostTerm::Param`] numbers it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReachedPlace {
    pub param: usize,
    pub element: Option<usize>,
}

/// An extern and one of its instances, numbered as `Callee::Extern`
/// numbers them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ExternInstance {
    pub id: QualifiedRef,
    pub instance: usize,
}

/// An instance's [`Laws`] with each extern they name taken at the instance
/// whose types are the declaring instance's.
#[derive(Debug, Clone, PartialEq, Default)]
pub enum ResolvedLaws {
    #[default]
    None,
    Binary(ResolvedBinary),
    Fold(ResolvedFold),
    TotalOrder,
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
    Extern(ExternInstance),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResolvedFold {
    pub combine: ExternInstance,
    pub identity: ExternInstance,
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
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Unresolved {
    /// The declaring instance's type is not the signature its law is
    /// stated over.
    UnfitDeclaration,
    /// `named` is no extern, or none of its instances, or more than one,
    /// has the type `role` asks at the declaring instance's types.
    NoFittingInstance { role: LawRole, named: QualifiedRef },
}

/// `laws`, declared on an instance of type `declaring`, with each extern
/// they name taken at the instance a call at `declaring`'s types chooses:
/// its one concrete instance those types have the shape of, and otherwise
/// its generic instance. A name picks the extern and the types pick its
/// instance: `num::min` over `i32` names `i32::MAX`, and a fold on `push`
/// at `Vec<T>` combines through `extend` at `Vec<T>`.
pub fn resolve<'a>(
    laws: &Laws,
    declaring: &PolyTy,
    function: impl Fn(QualifiedRef) -> Option<&'a Function>,
) -> Result<ResolvedLaws, Unresolved> {
    let PolyTy::Fn { params, ret, .. } = declaring else {
        return match laws {
            Laws::None => Ok(ResolvedLaws::None),
            Laws::Binary(_) | Laws::Fold(_) | Laws::TotalOrder => {
                Err(Unresolved::UnfitDeclaration)
            }
        };
    };
    let instance_of = |role: LawRole, named: QualifiedRef, wanted: &Wanted| {
        let unfit = Unresolved::NoFittingInstance { role, named };
        let found = function(named).ok_or(unfit)?;
        let FnKind::Extern { instances, .. } = &found.kind else {
            return Err(unfit);
        };
        let at = |instance: usize| ExternInstance {
            id: named,
            instance,
        };
        let concrete: Vec<usize> = instances
            .concrete
            .iter()
            .enumerate()
            .filter(|(_, instance)| wanted.fits(&instance.ty))
            .map(|(instance, _)| instance)
            .collect();
        match concrete[..] {
            [instance] => Ok(at(instance)),
            [] if wanted.fits(&found.ty) => match &instances.generic {
                Some(_) => Ok(at(instances.generic_index())),
                // The solver calls an extern that lists no instance at all
                // through its scheme, as instance 0 (`fixed_generic` in
                // `solver.rs`).
                None if instances.concrete.is_empty() => Ok(at(0)),
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
        }) => Ok(ResolvedLaws::Binary(ResolvedBinary {
            associative: *associative,
            commutative: *commutative,
            identity: match identity {
                None => None,
                Some(Identity::Const(constant)) => Some(ResolvedIdentity::Const(constant.clone())),
                Some(Identity::Extern(named)) => Some(ResolvedIdentity::Extern(instance_of(
                    LawRole::Identity,
                    *named,
                    &Wanted::Returning((**ret).clone()),
                )?)),
            },
        })),
        Laws::Fold(FoldLaw {
            combine,
            identity,
            commutative,
        }) => {
            let Some(PolyTy::Ref(Mutability::Mut, state)) = params.first().map(|p| &p.ty) else {
                return Err(Unresolved::UnfitDeclaration);
            };
            let state = state.ty().into_owned();
            Ok(ResolvedLaws::Fold(ResolvedFold {
                combine: instance_of(
                    LawRole::FoldCombine,
                    *combine,
                    &Wanted::Combining(state.clone()),
                )?,
                identity: instance_of(LawRole::FoldIdentity, *identity, &Wanted::Returning(state))?,
                commutative: *commutative,
            }))
        }
        Laws::TotalOrder => {
            let compares = match params.as_slice() {
                [a, b] => match (&a.ty, &b.ty) {
                    (PolyTy::Ref(Mutability::Shared, a), PolyTy::Ref(Mutability::Shared, b)) => {
                        a.ty() == b.ty()
                    }
                    _ => false,
                },
                _ => false,
            };
            match compares && matches!(**ret, PolyTy::Int(crate::ty::IntTy::I64)) {
                true => Ok(ResolvedLaws::TotalOrder),
                false => Err(Unresolved::UnfitDeclaration),
            }
        }
    }
}

/// Whether `copies`, declared on an instance of type `declaring`, names a
/// shared reference parameter lending a value of the result's type
/// (RFC-0082 rule 10).
pub fn copies_fits(copies: Copies, declaring: &PolyTy) -> bool {
    let PolyTy::Fn { params, ret, .. } = declaring else {
        return false;
    };
    match params.get(copies.param).map(|param| &param.ty) {
        Some(PolyTy::Ref(Mutability::Shared, lent)) => *lent.ty() == **ret,
        _ => false,
    }
}

/// The type a law asks of an extern it names, at the declaring instance's
/// types.
enum Wanted {
    /// `e() -> T`.
    Returning(PolyTy),
    /// `g(s: &mut S, part: S)`.
    Combining(PolyTy),
}

impl Wanted {
    /// Whether a call at the wanted types has the shape of an instance of
    /// type `candidate`.
    fn fits(&self, candidate: &PolyTy) -> bool {
        let PolyTy::Fn { params, ret, .. } = candidate else {
            return false;
        };
        match (self, params.as_slice()) {
            (Self::Returning(value), []) => matches_pattern(value, ret),
            (Self::Combining(state), [into, part]) if **ret == PolyTy::Unit => match &into.ty {
                PolyTy::Ref(Mutability::Mut, into) => matches_pattern(
                    &PolyTy::Tuple(vec![state.clone(), state.clone()]),
                    &PolyTy::Tuple(vec![into.ty().into_owned(), part.ty.clone()]),
                ),
                _ => false,
            },
            _ => false,
        }
    }
}

/// One instance's type and what its declaration states.
struct DeclaredAt<'a> {
    ty: &'a PolyTy,
    laws: &'a Laws,
    ensures: &'a [Postcondition],
    reaches: &'a Reaches,
    returns: Returns,
    copies: Option<Copies>,
    cost: Option<u64>,
}

/// What one instance of an extern declares.
#[derive(Debug, Default)]
struct Declared {
    laws: ResolvedLaws,
    ensures: Vec<Postcondition>,
    reaches: Reaches,
    returns: Returns,
    copies: Option<Copies>,
    cost: Option<u64>,
}

#[derive(Debug, Default)]
pub struct LawTable {
    by_instance: FxHashMap<QualifiedRef, Vec<Declared>>,
}

impl LawTable {
    /// # Panics
    /// If a law names an extern with no instance of the types it asks:
    /// `acvus_extern::Externs::combine` refuses such a registry by the same
    /// [`resolve`].
    pub fn of<'a>(functions: impl IntoIterator<Item = &'a Function>) -> Self {
        let functions: FxHashMap<QualifiedRef, &Function> = functions
            .into_iter()
            .map(|function| (function.qref, function))
            .collect();
        let by_instance = functions
            .values()
            .filter_map(|function| {
                let FnKind::Extern { instances, .. } = &function.kind else {
                    return None;
                };
                let declared_at = instances
                    .concrete
                    .iter()
                    .map(|instance| DeclaredAt {
                        ty: &instance.ty,
                        laws: &instance.laws,
                        ensures: &instance.ensures,
                        reaches: &instance.reaches,
                        returns: instance.returns,
                        copies: instance.copies,
                        cost: instance.cost,
                    })
                    .chain(instances.generic.as_ref().map(|generic| DeclaredAt {
                        ty: &function.ty,
                        laws: &generic.laws,
                        ensures: &generic.ensures,
                        reaches: &generic.reaches,
                        returns: generic.returns,
                        copies: generic.copies,
                        cost: generic.cost,
                    }));
                let mut declared: Vec<Declared> = declared_at
                    .map(|DeclaredAt {
                             ty,
                             laws,
                             ensures,
                             reaches,
                             returns,
                             copies,
                             cost,
                         }| Declared {
                        laws: resolve(laws, ty, |named| functions.get(&named).copied())
                            .unwrap_or_else(|unresolved| {
                                panic!(
                                    "a law of {:?} does not resolve ({unresolved:?}), and \
                                     combining the registries refuses it",
                                    function.qref
                                )
                            }),
                        ensures: ensures.to_vec(),
                        reaches: reaches.clone(),
                        returns,
                        copies: copies.inspect(|copies| {
                            assert!(
                                copies_fits(*copies, ty),
                                "`copies` of {:?} names no shared reference parameter lending \
                                 its result's type, and combining the registries refuses it",
                                function.qref
                            )
                        }),
                        cost,
                    })
                    .collect();
                if declared.is_empty() {
                    declared.push(Declared::default());
                }
                Some((function.qref, declared))
            })
            .collect();
        Self { by_instance }
    }

    pub fn of_callee(&self, callee: &Callee) -> &ResolvedLaws {
        static NONE: ResolvedLaws = ResolvedLaws::None;
        match self.declared(callee) {
            Some(declared) => &declared.laws,
            None => &NONE,
        }
    }

    /// The postconditions the instance a call names declares; a call of a
    /// local function or through a value has none.
    pub fn postconditions_of(&self, callee: &Callee) -> &[Postcondition] {
        match self.declared(callee) {
            Some(declared) => &declared.ensures,
            None => &[],
        }
    }

    /// What a call of the instance `callee` names reaches through its
    /// reference arguments; a call of a local function or through a value
    /// states nothing, and reaches all they lend.
    pub fn reaches_of(&self, callee: &Callee) -> &Reaches {
        static LENT: Reaches = Reaches::Lent;
        match self.declared(callee) {
            Some(declared) => &declared.reaches,
            None => &LENT,
        }
    }

    /// The argument whose lent value the result of the instance a call names
    /// equals (`copies(x)`, RFC-0082 rule 10), numbered as a call's
    /// arguments are; `None` where it states none, and for a call of a local
    /// function or through a value.
    pub fn copies_of(&self, callee: &Callee) -> Option<usize> {
        self.declared(callee)
            .and_then(|declared| declared.copies)
            .map(|copies| copies.param)
    }

    pub fn returns_of(&self, callee: &Callee) -> Returns {
        self.declared(callee)
            .map_or(Returns::Unstated, |declared| declared.returns)
    }

    /// The weight in ticks the instance a call names states of one call
    /// (`cost = N`, RFC-0066 rule 8), read by instance as a law is; `None`
    /// where it states none, and for a call of a local function or through
    /// a value, which no declaration weighs.
    pub fn cost_of(&self, callee: &Callee) -> Option<u64> {
        self.declared(callee).and_then(|declared| declared.cost)
    }

    fn declared(&self, callee: &Callee) -> Option<&Declared> {
        let Callee::Extern { id, instance, .. } = callee else {
            return None;
        };
        let Some(instances) = self.by_instance.get(id) else {
            panic!("the law table was built without the extern {id:?} a call names")
        };
        let Some(declared) = instances.get(*instance) else {
            panic!(
                "the extern {id:?} has {} instances, and a call names instance {instance}",
                instances.len()
            )
        };
        Some(declared)
    }
}
