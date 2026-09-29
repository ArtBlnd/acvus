//! initck (ownership §4): a place is read by name only where every path to
//! the read has written what it reads: a field where that field is written,
//! the value whole where every field of its type is. It reads
//! `Initialization`: at each `Take` or `Ref` whose target is a slot or a
//! by-value parameter's storage, each unit its path overlaps must be written
//! and nothing else (`Written::is_written`). Where one is not and none of
//! them was moved out, the read is refused `NotWritten` at its span; a unit
//! moved out is `moveck`'s. What a read through a reference reaches is not
//! located by name, and is not this check's.

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
            let unwritten = read.iter().any(|(_, fact)| !fact.is_written());
            let moved = read.iter().any(|(_, fact)| !fact.moved.is_empty());
            if unwritten && !moved {
                out.push(Refusal { kind: RefusalKind::NotWritten, span: inst.span() });
            }
        }
    }
}
