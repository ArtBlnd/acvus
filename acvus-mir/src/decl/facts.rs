//! What an extern declares beyond its signature (extern-fn §3-4): its laws,
//! its postconditions, what a call reaches and how it ends. Each is the
//! author's promise, trusted as an effect is (RFC-0082 rules 2 to 9).

use acvus_ast::Literal;

use acvus_utils::QualifiedRef;

use crate::step::Step;

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
    /// `#[extern_fn(law(inverse = g))]` on `f(s: &mut S) -> Option<X>`:
    /// after `g(s, x)`, `f(s)` gives `Some(x)` and leaves `s` as before `g`;
    /// `g(s, x)` after `f(s)` gave `Some(x)` leaves `s` as before `f`; and
    /// `f(s)` giving `None` leaves `s` as it was (RFC-0082 rule 3).
    /// `acvus_extern::Externs::combine` holds `g` to a registered
    /// `g(s: &mut S, x: X)`.
    Inverse(QualifiedRef),
    /// `#[extern_fn(payload(o))]` on `f(o: Option<T>) -> T`: `f(Some(x))`
    /// is `x` and `f(None)` traps (RFC-0082 rule 3). The signature it is
    /// stated over states no other law, so it is one of this enum's forms.
    Payload,
    /// `#[extern_fn(law(equivalence))]` on a `core::eq` impl: it is
    /// reflexive, symmetric and transitive, and the `core::hash` impl at
    /// its type hashes values it calls equal alike (RFC-0082 rule 11). A
    /// requirement naming [`NamedLaw::Equivalence`] is resolved only by an
    /// impl stating it (RFC-0070 rule 6).
    Equivalence,
    /// `#[extern_fn(step(..))]` on a stream adaptor or consumer (RFC-0099
    /// rule 1).
    Step(Step),
}

/// A law a requirement names at its type (RFC-0070 rule 6), which only an
/// impl stating it resolves.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NamedLaw {
    /// `law::Equivalence`: a `core::eq` impl declared an equivalence
    /// (`Access::Equivalence`).
    Equivalence,
    /// `law::TotalOrder`: a `core::cmp` impl declared a total order
    /// (`Action::TotalOrder`).
    TotalOrder,
}

impl NamedLaw {
    /// Whether an impl declaring `facts` states this law, by the declared
    /// fact of its kind (D82; extern-fn §3).
    pub fn stated_by(self, facts: &crate::decl::DeclaredFacts) -> bool {
        match self {
            NamedLaw::Equivalence => facts.access.iter().any(|access| matches!(access, crate::decl::declared::Access::Equivalence)),
            NamedLaw::TotalOrder => facts.action.iter().any(|action| matches!(action, crate::decl::declared::Action::TotalOrder)),
        }
    }

    /// The law as a declaration names it.
    pub fn written(self) -> &'static str {
        match self {
            NamedLaw::Equivalence => "law::Equivalence",
            NamedLaw::TotalOrder => "law::TotalOrder",
        }
    }
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

/// `#[extern_fn(reaches(p, ..))]` (RFC-0082 rule 7): what a call reaches
/// of the storages its reference arguments lend.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum Reaches {
    /// No declaration: a call reaches all each reference argument lends it.
    #[default]
    Lent,
    /// A call reaches these places and nothing else through its reference
    /// arguments.
    Places(Vec<ReachedPlace>),
}

/// How a call of the impl ends, as its declaration states it (RFC-0082
/// rules 8 and 9): the author's promise, which a call reads over the
/// impl tree it chose.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Returns {
    #[default]
    Unstated,
    /// `#[extern_fn(returns)]`: every call returns or traps. RFC-0089 rule 5
    /// reads it to run the call ahead of its iteration's control token.
    Stated,
    /// `#[extern_fn(total)]`: the impl's own body returns a value and
    /// never traps. An impl with requirements states it of that body
    /// alone, and the impls its requirements choose answer for
    /// themselves.
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

/// `x`, the whole of what reference parameter `x` lends, `x[i]`, its
/// element at the `u64` parameter `i`, or `x[k]`, a map or set's entry at
/// the key parameter `k`. A parameter is numbered as [`PostTerm::Param`]
/// numbers it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReachedPlace {
    pub param: usize,
    pub element: Option<ReachedElement>,
}

/// What names the part of a reached place (RFC-0082 rule 7).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReachedElement {
    /// The element at the `u64` parameter this numbers.
    Index(usize),
    /// The entry at the key parameter this numbers, which meets other keys
    /// by the map's equivalence where its type carries `Equiv` (RFC-0098
    /// rule 2).
    Key(usize),
}
