//! moveck (ownership §2, §6): a value that does not copy is not used after
//! a use moved it, whole or in the part moved; a value a part of which moved
//! out is not used whole again. It reads `Initialization`: at each `Take` or
//! `Ref` whose target is a slot or a by-value parameter's storage, a unit its
//! path overlaps that some path moved out refuses the read `UsedAfterMove`
//! at its span, naming the move. A read one of whose units is only never
//! written is `initck`'s.

use super::{Refusal, RefusalKind};
use crate::ir::RefTarget;
use crate::ir::cfg::{BodyId, Module, Op};
use crate::passes::analysis::{AnalysisManager, Initialization};

/// The judgment over `body`, its refusals pushed to `out`.
pub(super) fn check(module: &Module, body: BodyId, am: &mut AnalysisManager<'_>, out: &mut Vec<Refusal>) {
    let initialization = am.get::<Initialization>(module, body);
    for (_, block) in module.body(body).blocks() {
        for inst in block.insts() {
            let (Op::Take { target: RefTarget::Var(place) | RefTarget::Param(place), path, .. }
            | Op::Ref { target: RefTarget::Var(place) | RefTarget::Param(place), path, .. }) = inst.op()
            else {
                continue;
            };
            let read = initialization.before(inst.id(), *place, path);
            let Some(moved) = read.iter().find_map(|(_, fact)| fact.moved.first()) else {
                continue;
            };
            out.push(Refusal { kind: RefusalKind::UsedAfterMove { moved: *moved }, span: inst.span() });
        }
    }
}
