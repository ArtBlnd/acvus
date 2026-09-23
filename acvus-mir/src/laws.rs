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

use crate::graph::{FnKind, Function, QualifiedRef};
use crate::ir::Callee;

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
/// result, the length of either, and `+`, `−`, `×` and `max` of terms. It
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
}

/// What `len(x)` reads: a parameter or the result.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Subject {
    Param(usize),
    Ret,
}

/// What one instance of an extern declares.
#[derive(Debug, Default)]
struct Declared {
    laws: Laws,
    ensures: Vec<Postcondition>,
}

#[derive(Debug, Default)]
pub struct LawTable {
    by_instance: FxHashMap<QualifiedRef, Vec<Declared>>,
}

impl LawTable {
    pub fn of<'a>(functions: impl IntoIterator<Item = &'a Function>) -> Self {
        let by_instance = functions
            .into_iter()
            .filter_map(|function| match &function.kind {
                // The solver calls an extern that lists no instance at all
                // through its scheme, as instance 0 (`fixed_generic` in
                // `solver.rs`), and such a function has nowhere to state a
                // law.
                FnKind::Extern { instances, .. }
                    if instances.concrete.is_empty() && instances.generic.is_none() =>
                {
                    Some((function.qref, vec![Declared::default()]))
                }
                FnKind::Extern { instances, .. } => {
                    let declared = instances
                        .concrete
                        .iter()
                        .map(|instance| Declared {
                            laws: instance.laws.clone(),
                            ensures: instance.ensures.clone(),
                        })
                        .chain(instances.generic.as_ref().map(|g| Declared {
                            laws: g.laws.clone(),
                            ensures: g.ensures.clone(),
                        }))
                        .collect();
                    Some((function.qref, declared))
                }
                FnKind::Local(_) => None,
            })
            .collect();
        Self { by_instance }
    }

    pub fn of_callee(&self, callee: &Callee) -> &Laws {
        static NONE: Laws = Laws::None;
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
