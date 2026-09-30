//! The analysis manager (passes-structure §2): an analysis is computed on
//! first request and kept until an edit changes an aspect it reads. No pass
//! declares what it preserves; the Rewriter's record of what changed
//! decides (ir::cfg::Changes).

use std::any::{Any, TypeId};
use std::rc::Rc;

use rustc_hash::{FxHashMap, FxHashSet};

use crate::decl::Declarations;
use crate::ir::cfg::{Aspect, BodyId, Changes, Module};

/// An analysis the manager keeps. `READS` lists every aspect of a body its
/// facts depend on; its doc states its cost and its fixpoint's bound.
pub trait Analysis: 'static {
    type Facts: 'static;

    const READS: &'static [Aspect];

    fn compute(am: &mut AnalysisManager<'_>, module: &Module, body: BodyId) -> Self::Facts;
}

pub struct AnalysisManager<'d> {
    decls: &'d Declarations,
    kept: FxHashMap<(TypeId, BodyId), (Rc<dyn Any>, &'static [Aspect])>,
    /// The analyses being computed, each until its `compute` returns.
    computing: FxHashSet<(TypeId, BodyId)>,
}

impl<'d> AnalysisManager<'d> {
    pub fn new(decls: &'d Declarations) -> Self {
        Self { decls, kept: FxHashMap::default(), computing: FxHashSet::default() }
    }

    pub fn decls(&self) -> &'d Declarations {
        self.decls
    }

    /// The facts of `A` for `body`, computed on first request. Cost: O(1)
    /// when kept, else `A`'s own.
    ///
    /// # Panics
    /// `A` for `body` is requested while it is being computed: an analysis
    /// that depends on itself, directly or through others, is a compiler
    /// defect.
    pub fn get<A>(&mut self, module: &Module, body: BodyId) -> Rc<A::Facts>
    where
        A: Analysis,
    {
        let key = (TypeId::of::<A>(), body);
        if let Some((facts, _)) = self.kept.get(&key) {
            return Rc::clone(facts)
                .downcast::<A::Facts>()
                .unwrap_or_else(|_| panic!("the facts kept under an analysis's TypeId are its own"));
        }
        assert!(
            self.computing.insert(key),
            "{} for {body:?} was requested while it is being computed; an analysis that depends on itself is a compiler defect",
            std::any::type_name::<A>()
        );
        let facts = Rc::new(A::compute(self, module, body));
        self.computing.remove(&key);
        self.kept.insert(key, (Rc::clone(&facts) as Rc<dyn Any>, A::READS));
        facts
    }

    /// Drops every kept analysis of `body` that reads an aspect `changes`
    /// touched. Cost: O(kept analyses · aspects).
    pub fn invalidate(&mut self, body: BodyId, changes: &Changes) {
        self.kept.retain(|(_, of), (_, reads)| {
            *of != body || !reads.iter().any(|aspect| changes.touched(*aspect))
        });
    }
}
