//! Bounds-check elimination (RFC-0047 rule 7): an `Index` or `IndexSet`
//! whose index `analysis::interval` puts below its slice's length runs
//! without the comparison.

use crate::analysis::interval::{InstAt, bounded_indices};
use crate::cfg::{BlockIdx, CfgBody};
use crate::ir::{IndexBound, InstKind};
use crate::laws::LawTable;

pub fn run(cfg: &mut CfgBody, laws: &LawTable) {
    let bounded = bounded_indices(cfg, laws);
    for (b, block) in cfg.blocks.iter_mut().enumerate() {
        for (at, inst) in block.insts.iter_mut().enumerate() {
            if !bounded.contains(&InstAt {
                block: BlockIdx(b),
                at,
            }) {
                continue;
            }
            match &mut inst.kind {
                InstKind::Index { bound, .. } | InstKind::IndexSet { bound, .. } => {
                    *bound = IndexBound::Proven;
                }
                other => unreachable!("the interval domain bounds an index, not {other:?}"),
            }
        }
    }
}
