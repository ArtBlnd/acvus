//! Which context a `Load` or `Store` reaches, for passes that order
//! calls against a run's own context accesses (RFC-0017).

use rustc_hash::FxHashMap;

use crate::cfg::CfgBody;
use crate::graph::QualifiedRef;
use crate::ir::{InstKind, RefTarget, ValueId};

/// Every `Ref` of a whole context, by its value: the context it names.
pub(crate) fn ref_to_ctx(cfg: &CfgBody) -> FxHashMap<ValueId, QualifiedRef> {
    let mut map = FxHashMap::default();
    for block in &cfg.blocks {
        for inst in &block.insts {
            if let InstKind::Ref {
                dst,
                target: RefTarget::Context(qref),
                path,
            } = &inst.kind
                && path.is_empty()
            {
                map.insert(*dst, *qref);
            }
        }
    }
    map
}

pub(crate) fn context_of_load(
    kind: &InstKind,
    ref_to_ctx: &FxHashMap<ValueId, QualifiedRef>,
) -> Option<QualifiedRef> {
    match kind {
        InstKind::Load { src, .. } => ref_to_ctx.get(src).copied(),
        _ => None,
    }
}

pub(crate) fn context_of_store(
    kind: &InstKind,
    ref_to_ctx: &FxHashMap<ValueId, QualifiedRef>,
) -> Option<QualifiedRef> {
    match kind {
        InstKind::Store { dst, .. } => ref_to_ctx.get(dst).copied(),
        _ => None,
    }
}
