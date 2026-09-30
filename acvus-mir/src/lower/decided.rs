//! What typeck decided, as one body being lowered reads it (functions §2):
//! the run's own decisions, or, for a lambda body lowered at an instance of
//! a generalized binding, that instance's, then the instances it was made
//! within, outward, then the run's. Every form reads its site's type and
//! decisions through this view; none reads `Typeck`'s site tables directly.

use acvus_ast::AstId;
use rustc_hash::FxHashMap;

use crate::decl::PairMember;
use crate::ty::{Concrete, Ty, TyKind, Types};
use crate::typeck::{Adjustment, AsCast, Chosen, Coercion, InstanceDecided, InstanceId, Iteration, StructureTree, TryExit, Typeck, UnitCall};

/// A view of `Typeck` at one instance, or at none.
#[derive(Clone, Copy)]
pub(super) struct Decided<'a> {
    pub typeck: &'a Typeck,
    pub instance: Option<InstanceId>,
    /// The type a generalized binding's value has at each body it is
    /// lowered in (`closure_types`).
    pub closures: &'a ClosureTypes,
}

/// The type of each generalized binding's value in each body it may be
/// written in, by binder and by the instance the body is at: the one
/// signature's function type where its instances there have one, and the
/// `Closures` type of their set otherwise, none included (functions §2).
pub(super) type ClosureTypes = FxHashMap<(AstId, Option<InstanceId>), Ty>;

/// Every `ClosureTypes` entry, interned into `types`: each binder at the
/// run, and at each instance a body may be lowered at.
pub(super) fn closure_types(typeck: &Typeck, types: &mut Types<Concrete>) -> ClosureTypes {
    let views: Vec<Option<InstanceId>> = std::iter::once(None).chain(typeck.instances.keys().copied().map(Some)).collect();
    let mut out = FxHashMap::default();
    for binder in typeck.generalized.keys() {
        for view in &views {
            let mut signatures: Vec<Ty> = typeck
                .instances
                .values()
                .filter(|instance| instance.binder == Some(*binder) && within(typeck, instance, *view))
                .map(|instance| {
                    *instance.types.get(binder).unwrap_or_else(|| panic!("typeck records each instance's signature at its binder"))
                })
                .collect();
            signatures.sort_by_key(|signature| signature.canonical_key());
            signatures.dedup();
            let ty = match signatures[..] {
                [one] => one,
                _ => types.intern(TyKind::Closures(signatures.into())),
            };
            out.insert((*binder, *view), ty);
        }
    }
    out
}

/// Whether `instance` is one a body at `view` makes code for: made within
/// `view`, at any depth, or any at the run's level.
pub(super) fn within(typeck: &Typeck, instance: &InstanceDecided, view: Option<InstanceId>) -> bool {
    let Some(view) = view else {
        return true;
    };
    let mut at = instance.within;
    while let Some(within) = at {
        if within == view {
            return true;
        }
        at = typeck.instances.get(&within).and_then(|instance| instance.within);
    }
    false
}

impl<'a> Decided<'a> {
    /// The run's own view.
    pub fn run(typeck: &'a Typeck, closures: &'a ClosureTypes) -> Decided<'a> {
        Decided { typeck, instance: None, closures }
    }

    /// The type a value of the generalized binding `binder` has at this
    /// view's body.
    pub fn value_type(&self, binder: AstId) -> Ty {
        *self
            .closures
            .get(&(binder, self.instance))
            .unwrap_or_else(|| panic!("every generalized binding has a type at every view, and not {binder:?}"))
    }

    /// `get` of the view's instance, then of each instance it was made
    /// within, outward, then `run` of the run's tables.
    fn read<T>(&self, get: impl Fn(&'a InstanceDecided) -> Option<T>, run: impl FnOnce(&'a Typeck) -> Option<T>) -> Option<T> {
        let mut at = self.instance;
        while let Some(instance) = at {
            let decided = self
                .typeck
                .instances
                .get(&instance)
                .unwrap_or_else(|| panic!("typeck records every instance lower is at, and not {instance:?}"));
            if let Some(found) = get(decided) {
                return Some(found);
            }
            at = decided.within;
        }
        run(self.typeck)
    }

    /// The type at `site`.
    ///
    /// # Panics
    /// No table the view reads types `site`: every site of an admitted
    /// program is typed at every instance it is lowered at.
    pub fn ty(&self, site: AstId) -> Ty {
        // A site a generalized binding's value passes through where it is
        // written holds every signature it has at this body.
        if let Some(binder) = self.typeck.value_sites.get(&site) {
            return self.value_type(*binder);
        }
        self.read(|at| at.types.get(&site).copied(), |run| run.types.get(&site).copied())
            .unwrap_or_else(|| panic!("typeck types every site of an admitted program, and not {site:?}"))
    }

    pub fn reached(&self, site: AstId) -> Option<&'a Chosen> {
        self.read(|at| at.reached.get(&site), |run| run.reached.get(&site))
    }

    /// The call of a unit at `site`, the same at every instance.
    pub fn unit_call(&self, site: AstId) -> Option<&'a UnitCall> {
        self.typeck.units.get(&site)
    }

    pub fn callee(&self, site: AstId) -> Option<Ty> {
        self.read(|at| at.callees.get(&site).copied(), |run| run.callees.get(&site).copied())
    }

    /// The type the value of the call at `site` is taken at, where its uses
    /// settled it other than its declared result (types §8).
    pub fn unmark(&self, site: AstId) -> Option<Ty> {
        self.read(|at| at.unmarks.get(&site).copied(), |run| run.unmarks.get(&site).copied())
    }

    /// Which of a `#` pair the call at `site` is, where it calls a pair's
    /// declaration (types §8).
    pub fn pair(&self, site: AstId) -> Option<PairMember> {
        self.read(|at| at.pairs.get(&site).copied(), |run| run.pairs.get(&site).copied())
    }

    pub fn coercion(&self, site: AstId) -> Option<&'a Coercion> {
        self.read(|at| at.coercions.get(&site), |run| run.coercions.get(&site))
    }

    pub fn structure_tree(&self, site: AstId) -> Option<&'a StructureTree> {
        self.read(|at| at.structure_trees.get(&site), |run| run.structure_trees.get(&site))
    }

    pub fn adjustment(&self, site: AstId) -> Option<Adjustment> {
        self.read(|at| at.adjustments.get(&site).copied(), |run| run.adjustments.get(&site).copied())
    }

    pub fn iteration(&self, site: AstId) -> Option<Iteration> {
        self.read(|at| at.iteration.get(&site).copied(), |run| run.iteration.get(&site).copied())
    }

    pub fn try_exit(&self, site: AstId) -> Option<TryExit<Ty>> {
        self.read(|at| at.tries.get(&site).copied(), |run| run.tries.get(&site).copied())
    }

    pub fn cast(&self, site: AstId) -> Option<AsCast> {
        self.read(|at| at.casts.get(&site).copied(), |run| run.casts.get(&site).copied())
    }

    /// The same view at `instance`, a lambda body's instance within this
    /// one.
    pub fn at(&self, instance: InstanceId) -> Decided<'a> {
        Decided { typeck: self.typeck, instance: Some(instance), closures: self.closures }
    }
}
