//! The analysis manager (passes-structure §2), a query system in the manner
//! of rustc's queries and salsa: an analysis is computed on first request
//! and kept until something it read changes. No pass declares what it
//! preserves; the Rewriter's record of what changed decides
//! (ir::cfg::Changes).
//!
//! A kept result reads two things. It reads IR aspects directly, which its
//! `READS` states: a body analysis those of its own body, a module analysis
//! those of every body. And it reads other kept results, which the manager
//! records itself: every request made while a result is being computed is
//! an edge from that result to the requested one, whether the request was
//! answered from the kept results or computed. Dropping a result drops,
//! transitively, every kept result that read it, so no kept result depends
//! on anything that changed since it was computed.
//!
//! A module analysis also ranges over the module's body set, so a change
//! of that set drops every module analysis and what read it.

use std::any::{Any, TypeId};
use std::rc::Rc;

use rustc_hash::{FxHashMap, FxHashSet};

use crate::decl::Declarations;
use crate::ir::cfg::{Aspect, BodyId, BodySetChanged, Changes, Module};

/// An analysis the manager keeps per body. `READS` lists every aspect of
/// its body it reads directly; its doc states its cost and its fixpoint's
/// bound.
pub trait Analysis: 'static {
    type Facts: 'static;

    const READS: &'static [Aspect];

    fn compute(am: &mut AnalysisManager<'_>, module: &Module, body: BodyId) -> Self::Facts;
}

/// An analysis of the whole module the manager keeps once. `READS` lists
/// every aspect of any body it reads directly; its doc states its cost.
pub trait ModuleAnalysis: 'static {
    type Facts: 'static;

    const READS: &'static [Aspect];

    fn compute(am: &mut AnalysisManager<'_>, module: &Module) -> Self::Facts;
}

/// A kept result, or one being computed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum Key {
    Body(TypeId, BodyId),
    Module(TypeId),
}

struct Kept {
    facts: Rc<dyn Any>,
    reads: &'static [Aspect],
    /// The kept results this one read while it was computed.
    dependencies: FxHashSet<Key>,
    /// The kept results that read this one while they were computed.
    dependents: FxHashSet<Key>,
}

pub struct AnalysisManager<'d> {
    decls: &'d Declarations,
    kept: FxHashMap<Key, Kept>,
    /// The results being computed, innermost last, each until its
    /// `compute` returns, with the results each has read so far.
    computing: Vec<(Key, FxHashSet<Key>)>,
}

impl<'d> AnalysisManager<'d> {
    pub fn new(decls: &'d Declarations) -> Self {
        Self { decls, kept: FxHashMap::default(), computing: Vec::new() }
    }

    pub fn decls(&self) -> &'d Declarations {
        self.decls
    }

    /// The facts of `A` for `body`, computed on first request. Cost: O(1)
    /// when kept, else `A`'s own and O(the requests being computed).
    ///
    /// # Panics
    /// `A` for `body` is requested while it is being computed: an analysis
    /// that depends on itself, directly or through others, is a compiler
    /// defect.
    pub fn get<A>(&mut self, module: &Module, body: BodyId) -> Rc<A::Facts>
    where
        A: Analysis,
    {
        let key = Key::Body(TypeId::of::<A>(), body);
        let facts = match self.kept_facts(key) {
            Some(facts) => facts,
            None => {
                self.begin(key, || format!("{} for {body:?}", std::any::type_name::<A>()));
                let facts: Rc<dyn Any> = Rc::new(A::compute(self, module, body));
                self.end(key, Rc::clone(&facts), A::READS);
                facts
            }
        };
        self.record_read(key);
        facts.downcast::<A::Facts>().unwrap_or_else(|_| panic!("the facts kept under an analysis's TypeId are its own"))
    }

    /// The facts of the module analysis `A`, computed on first request
    /// and kept once for the module. Cost: as `get`.
    ///
    /// # Panics
    /// `A` is requested while it is being computed, as `get`.
    pub fn get_of_module<A>(&mut self, module: &Module) -> Rc<A::Facts>
    where
        A: ModuleAnalysis,
    {
        let key = Key::Module(TypeId::of::<A>());
        let facts = match self.kept_facts(key) {
            Some(facts) => facts,
            None => {
                self.begin(key, || std::any::type_name::<A>().to_owned());
                let facts: Rc<dyn Any> = Rc::new(A::compute(self, module));
                self.end(key, Rc::clone(&facts), A::READS);
                facts
            }
        };
        self.record_read(key);
        facts.downcast::<A::Facts>().unwrap_or_else(|_| panic!("the facts kept under a module analysis's TypeId are its own"))
    }

    /// Drops every kept analysis of `body`, and every kept module
    /// analysis, that reads an aspect `changes` touched, and every kept
    /// result that read a dropped one, transitively. Cost: O(kept results
    /// · aspects) for the direct scan, and O(dropped results + their edges)
    /// beyond it.
    pub fn invalidate(&mut self, body: BodyId, changes: &Changes) {
        let stale: Vec<Key> = self
            .kept
            .iter()
            .filter(|(key, _)| match key {
                Key::Body(_, of) => *of == body,
                Key::Module(_) => true,
            })
            .filter(|(_, kept)| kept.reads.iter().any(|aspect| changes.touched(*aspect)))
            .map(|(key, _)| *key)
            .collect();
        self.drop_with_dependents(stale);
    }

    /// Drops every kept module analysis, since each ranges over the body
    /// set `changed` records a change of, and every kept result that read
    /// a dropped one, transitively. Body analyses no dropped result reached
    /// are kept. Cost: O(kept results) for the scan, and O(dropped results
    /// + their edges) beyond it.
    pub fn invalidate_body_set(&mut self, changed: BodySetChanged) {
        let BodySetChanged { .. } = changed;
        let stale: Vec<Key> = self.kept.keys().filter(|key| matches!(key, Key::Module(_))).copied().collect();
        self.drop_with_dependents(stale);
    }

    fn kept_facts(&self, key: Key) -> Option<Rc<dyn Any>> {
        self.kept.get(&key).map(|kept| Rc::clone(&kept.facts))
    }

    /// # Panics
    /// `key` is being computed; `name` names it.
    fn begin<N>(&mut self, key: Key, name: N)
    where
        N: FnOnce() -> String,
    {
        assert!(
            self.computing.iter().all(|(computing, _)| *computing != key),
            "{} was requested while it is being computed; an analysis that depends on itself is a compiler defect",
            name()
        );
        self.computing.push((key, FxHashSet::default()));
    }

    fn end(&mut self, key: Key, facts: Rc<dyn Any>, reads: &'static [Aspect]) {
        let (computed, dependencies) = self.computing.pop().expect("a result that ends was begun");
        assert_eq!(computed, key, "requests end innermost first");
        for dependency in &dependencies {
            let kept = self.kept.get_mut(dependency).expect("a result read during a compute is kept until the compute ends");
            kept.dependents.insert(key);
        }
        self.kept.insert(key, Kept { facts, reads, dependencies, dependents: FxHashSet::default() });
    }

    /// Records that the result being computed, if any, read `key`.
    fn record_read(&mut self, key: Key) {
        if let Some((_, dependencies)) = self.computing.last_mut() {
            dependencies.insert(key);
        }
    }

    /// Drops `stale` and every kept result that read a dropped one,
    /// transitively, and the edges of each dropped result.
    fn drop_with_dependents(&mut self, stale: Vec<Key>) {
        let mut pending = stale;
        while let Some(key) = pending.pop() {
            let Some(dropped) = self.kept.remove(&key) else {
                continue;
            };
            for dependency in &dropped.dependencies {
                if let Some(kept) = self.kept.get_mut(dependency) {
                    kept.dependents.remove(&key);
                }
            }
            pending.extend(dropped.dependents);
        }
    }
}
