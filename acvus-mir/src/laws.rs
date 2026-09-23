//! The algebraic laws an extern declares (RFC-0082 rules 2 and 3), read by
//! a MIR pass through a call's callee.
//!
//! A law is the author's promise, trusted as an effect is. Nothing here
//! checks that a function is associative, and no law is inferred from a
//! handler's body, which is Rust and opaque to the compiler (RFC-0082 rule
//! 5).

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

#[derive(Debug, Default)]
pub struct LawTable {
    by_instance: FxHashMap<QualifiedRef, Vec<Laws>>,
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
                    Some((function.qref, vec![Laws::None]))
                }
                FnKind::Extern { instances, .. } => {
                    let laws = instances
                        .concrete
                        .iter()
                        .map(|instance| instance.laws.clone())
                        .chain(instances.generic.as_ref().map(|g| g.laws.clone()))
                        .collect();
                    Some((function.qref, laws))
                }
                FnKind::Local(_) => None,
            })
            .collect();
        Self { by_instance }
    }

    pub fn of_callee(&self, callee: &Callee) -> &Laws {
        static NONE: Laws = Laws::None;
        let Callee::Extern { id, instance, .. } = callee else {
            return &NONE;
        };
        let Some(instances) = self.by_instance.get(id) else {
            panic!("the law table was built without the extern {id:?} a call names")
        };
        let Some(laws) = instances.get(*instance) else {
            panic!(
                "the extern {id:?} has {} instances, and a call names instance {instance}",
                instances.len()
            )
        };
        laws
    }
}
