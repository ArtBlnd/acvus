//! For each slot, the first use a promotion cannot rewrite: any use but a
//! whole-slot `Take` or `Assign` of it, such as a `Ref`, an access to a
//! part, or a terminator naming it. No such use is exactly what
//! `Editor::promote` requires (D56). "First" is in block order, a block's
//! instructions before its terminator.
//!
//! Cost: O(the slots' uses · log of their number), to order them.
//!
//! passes-analyses.md.

use rustc_hash::FxHashMap;

use crate::ir::cfg::{Aspect, Body, BodyId, Module, Op, Site};
use crate::ir::{RefTarget, ValueId};
use crate::passes::analysis::{Analysis, AnalysisManager};

pub struct Escapes {
    first: FxHashMap<ValueId, Option<Site>>,
}

impl Escapes {
    /// The first use of `slot` a promotion cannot rewrite. Cost: O(1).
    ///
    /// # Panics
    /// `slot` is not a slot of the body.
    pub fn escape(&self, slot: ValueId) -> Option<Site> {
        *self.first.get(&slot).unwrap_or_else(|| panic!("{slot:?} is a slot of the body the facts cover"))
    }
}

impl Analysis for Escapes {
    type Facts = Escapes;

    const READS: &'static [Aspect] = &[Aspect::CfgShape, Aspect::Instructions, Aspect::Storage];

    fn compute(_: &mut AnalysisManager<'_>, module: &Module, body: BodyId) -> Escapes {
        let body = module.body(body);
        let first = body.slots().iter().map(|slot| (*slot, first_escape(body, *slot))).collect();
        Escapes { first }
    }
}

fn first_escape(body: &Body, slot: ValueId) -> Option<Site> {
    let mut sites: Vec<Site> = body.users(slot).collect();
    sites.sort_by(|a, b| body.site_order(*a, *b));
    sites.into_iter().find(|site| !promotable(body, slot, *site))
}

/// Whether the use at `site` is a whole-slot `Take` or `Assign` of `slot`
/// that names it nowhere else.
fn promotable(body: &Body, slot: ValueId, site: Site) -> bool {
    let Site::Inst(inst) = site else {
        return false;
    };
    match body.inst(inst).op() {
        Op::Take { target: RefTarget::Var(named), path, .. } => *named == slot && path.is_empty(),
        Op::Assign { target: RefTarget::Var(named), path, value, .. } => {
            *named == slot && path.is_empty() && *value != slot
        }
        _ => false,
    }
}
