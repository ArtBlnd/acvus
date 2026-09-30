//! The analysis manager (passes-structure §2): an analysis is computed on
//! first request and kept until an edit changes an aspect it reads. No pass
//! declares what it preserves; the Rewriter's record of what changed
//! decides (ir::cfg::Changes). A body analysis is kept per body and dropped
//! by an edit of that body; a module analysis is kept once per module and
//! dropped by an edit of any body.

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

/// An analysis of the whole module the manager keeps once. `READS` lists
/// every aspect of any body its facts depend on; its doc states its cost.
pub trait ModuleAnalysis: 'static {
    type Facts: 'static;

    const READS: &'static [Aspect];

    fn compute(am: &mut AnalysisManager<'_>, module: &Module) -> Self::Facts;
}

pub struct AnalysisManager<'d> {
    decls: &'d Declarations,
    kept: FxHashMap<(TypeId, BodyId), (Rc<dyn Any>, &'static [Aspect])>,
    /// The analyses being computed, each until its `compute` returns.
    computing: FxHashSet<(TypeId, BodyId)>,
    kept_of_module: FxHashMap<TypeId, (Rc<dyn Any>, &'static [Aspect])>,
    /// The module analyses being computed, each until its `compute` returns.
    computing_of_module: FxHashSet<TypeId>,
}

impl<'d> AnalysisManager<'d> {
    pub fn new(decls: &'d Declarations) -> Self {
        Self {
            decls,
            kept: FxHashMap::default(),
            computing: FxHashSet::default(),
            kept_of_module: FxHashMap::default(),
            computing_of_module: FxHashSet::default(),
        }
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

    /// The facts of the module analysis `A`, computed on first request
    /// and kept once for the module. Cost: O(1) when kept, else `A`'s own.
    ///
    /// # Panics
    /// `A` is requested while it is being computed, as `get`.
    pub fn get_of_module<A>(&mut self, module: &Module) -> Rc<A::Facts>
    where
        A: ModuleAnalysis,
    {
        let key = TypeId::of::<A>();
        if let Some((facts, _)) = self.kept_of_module.get(&key) {
            return Rc::clone(facts)
                .downcast::<A::Facts>()
                .unwrap_or_else(|_| panic!("the facts kept under a module analysis's TypeId are its own"));
        }
        assert!(
            self.computing_of_module.insert(key),
            "{} was requested while it is being computed; an analysis that depends on itself is a compiler defect",
            std::any::type_name::<A>()
        );
        let facts = Rc::new(A::compute(self, module));
        self.computing_of_module.remove(&key);
        self.kept_of_module.insert(key, (Rc::clone(&facts) as Rc<dyn Any>, A::READS));
        facts
    }

    /// Drops every kept analysis of `body`, and every kept module
    /// analysis, that reads an aspect `changes` touched. Cost: O((kept
    /// analyses + kept module analyses) · aspects).
    pub fn invalidate(&mut self, body: BodyId, changes: &Changes) {
        let stale = |reads: &[Aspect]| reads.iter().any(|aspect| changes.touched(*aspect));
        self.kept.retain(|(_, of), (_, reads)| *of != body || !stale(reads));
        self.kept_of_module.retain(|_, (_, reads)| !stale(reads));
    }
}
