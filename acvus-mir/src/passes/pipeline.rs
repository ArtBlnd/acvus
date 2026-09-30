//! The one entry for transforms (passes-pipeline.md, D70): P1-P7 at
//! `Level::Full`, each step's required form asserted in debug and test
//! builds, and the scalar fixpoint P3* bounded by the lexicographic measure
//! of D64, which every round that changes something strictly decreases.
//!
//! Cost: P1-P7 run once per body and P3* at most three times per body, each
//! run at most O(n²) rounds in the body's size (passes-pipeline §4); every
//! step is polynomial, so the whole is (compilation §2).

use crate::ir::cfg::edit::{
    DefDominatesUses, Invariants, NeverDiverges, OrderChainWhole, SplitTakenOnce, StructureKept, TypesAgree,
};
use crate::ir::cfg::{BodyId, Changes, Module, Op};
use crate::passes::analysis::{AnalysisManager, Escapes, LoopTree};
use crate::passes::optimize::{
    Bce, BodyTransform, Commute, Dce, Dse, Gvn, Inline, IvCanon, Licm, LoopDeletion, Lsr, Mem2Reg, ModuleTransform,
    Reborrow, Schedule, Sccp, SimplifyCfg, Sink, SplitCalls, Sroa, WhileToFor, union_of,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Level {
    /// No transform; the required steps are the entry's, not the pipeline's.
    None,
    Full,
}

pub fn optimize(module: &mut Module, am: &mut AnalysisManager<'_>, level: Level) {
    match level {
        Level::None => {}
        Level::Full => optimize_fully(module, am),
    }
}

fn optimize_fully(module: &mut Module, am: &mut AnalysisManager<'_>) {
    let bodies: Vec<BodyId> = module.body_ids().collect();
    for body in bodies.iter().copied() {
        canonical(module, body, am);
    }
    for (body, changes) in module_step::<Inline>(module, am) {
        if recorded_any(&changes) {
            canonical(module, body, am);
        }
    }
    for body in bodies {
        scalar_fixpoint(module, body, am, scalar_round);
        storage(module, body, am);
        proofs(module, body, am);
        motion(module, body, am);
        loops(module, body, am);
        effects(module, body, am);
    }
}

/// P1.
fn canonical(module: &mut Module, body: BodyId, am: &mut AnalysisManager<'_>) {
    step::<Mem2Reg>(module, body, am);
    step::<SimplifyCfg>(module, body, am);
}

/// One round of P3*, and the union of what its steps changed.
fn scalar_round(module: &mut Module, body: BodyId, am: &mut AnalysisManager<'_>) -> Changes {
    union_of([
        step::<Sccp>(module, body, am),
        step::<Gvn>(module, body, am),
        step::<Reborrow>(module, body, am),
        step::<Dce>(module, body, am),
        step::<SimplifyCfg>(module, body, am),
    ])
}

/// P4.
fn storage(module: &mut Module, body: BodyId, am: &mut AnalysisManager<'_>) {
    step::<Sroa>(module, body, am);
    step::<Mem2Reg>(module, body, am);
    scalar_fixpoint(module, body, am, scalar_round);
}

/// P5.
fn proofs(module: &mut Module, body: BodyId, am: &mut AnalysisManager<'_>) {
    step::<Dse>(module, body, am);
    step::<Bce>(module, body, am);
}

/// P6. `licm` runs last because it makes every loop's preheader, and
/// `simplify_cfg`, which P3* runs, bypasses a preheader left empty; the loop
/// group and P7 keep F2, so it holds from here into P7 (D73).
fn motion(module: &mut Module, body: BodyId, am: &mut AnalysisManager<'_>) {
    step::<Sink>(module, body, am);
    scalar_fixpoint(module, body, am, scalar_round);
    step::<Licm>(module, body, am);
}

/// The loop group (compilation §4), in the order of the standard loop
/// pipeline: promotion to a counted loop, induction-variable simplification,
/// deletion, then strength reduction, which reads the loops the others left.
/// `bce` runs again since a promoted loop's counter proves indices; `dce`
/// removes what the rewrites left unread. None edits a block's preheader away,
/// so F2 holds.
fn loops(module: &mut Module, body: BodyId, am: &mut AnalysisManager<'_>) {
    step::<WhileToFor>(module, body, am);
    step::<IvCanon>(module, body, am);
    step::<LoopDeletion>(module, body, am);
    // lsr takes its place here as it is filled.
    step::<Bce>(module, body, am);
    step::<Dce>(module, body, am);
}

/// P7. Nothing after `schedule` moves an instruction.
fn effects(module: &mut Module, body: BodyId, am: &mut AnalysisManager<'_>) {
    step::<Commute>(module, body, am);
    step::<SplitCalls>(module, body, am);
    step::<Schedule>(module, body, am);
}

/// The form a body is in (passes-pipeline §2). Each is the one before it
/// with one more property: `Ssa` is `Lowered` with no slot but those
/// `Escapes` names, and `LoopsCanonical` is `Ssa` with a preheader on every
/// natural loop.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Form {
    /// F0. Every body the IR's invariants admit is lowered, an SSA one
    /// included, since P2 runs `mem2reg` again on the bodies `inline`
    /// changed; so this form adds nothing to those invariants.
    Lowered,
    /// F1.
    Ssa,
    /// F2.
    LoopsCanonical,
}

/// The form a transform requires, as it states it.
trait Requires {
    const FORM: Form;
}

impl Requires for Mem2Reg {
    const FORM: Form = Form::Lowered;
}
impl Requires for SimplifyCfg {
    const FORM: Form = Form::Lowered;
}
impl Requires for Dce {
    const FORM: Form = Form::Lowered;
}
impl Requires for Inline {
    const FORM: Form = Form::Ssa;
}
impl Requires for Sccp {
    const FORM: Form = Form::Ssa;
}
impl Requires for Gvn {
    const FORM: Form = Form::Ssa;
}
impl Requires for Reborrow {
    const FORM: Form = Form::Ssa;
}
impl Requires for Sroa {
    const FORM: Form = Form::Ssa;
}
impl Requires for Dse {
    const FORM: Form = Form::Ssa;
}
impl Requires for Bce {
    const FORM: Form = Form::Ssa;
}
impl Requires for Licm {
    const FORM: Form = Form::Ssa;
}
impl Requires for Sink {
    const FORM: Form = Form::Ssa;
}
impl Requires for WhileToFor {
    const FORM: Form = Form::LoopsCanonical;
}
impl Requires for IvCanon {
    const FORM: Form = Form::LoopsCanonical;
}
impl Requires for LoopDeletion {
    const FORM: Form = Form::LoopsCanonical;
}
impl Requires for Lsr {
    const FORM: Form = Form::LoopsCanonical;
}
impl Requires for Commute {
    const FORM: Form = Form::LoopsCanonical;
}
impl Requires for SplitCalls {
    const FORM: Form = Form::LoopsCanonical;
}
impl Requires for Schedule {
    const FORM: Form = Form::LoopsCanonical;
}

/// Whether this build asserts forms, the IR's invariants and P3*'s measure.
const CHECKED: bool = cfg!(any(debug_assertions, test));

/// Every invariant an editor keeps, over the whole body.
type Kept = (((TypesAgree, DefDominatesUses, OrderChainWhole, StructureKept), NeverDiverges), SplitTakenOnce);

/// Runs `T` over `body` with its required form asserted before, and the
/// analyses it made stale dropped and the IR's invariants checked after. A
/// body a seal added drops the module analyses, which range over every
/// body.
fn step<T>(module: &mut Module, body: BodyId, am: &mut AnalysisManager<'_>) -> Changes
where
    T: BodyTransform + Requires,
{
    if CHECKED {
        assert_form(module, body, am, T::FORM, T::NAME);
    }
    let changes = T::run(module, body, am);
    if let Some(body_set_change) = module.take_body_set_change() {
        am.invalidate_body_set(body_set_change);
    }
    am.invalidate(body, &changes);
    if CHECKED {
        assert_kept(module, body, T::NAME);
    }
    changes
}

/// `step` for a transform over the module; every body is asserted in its
/// form before, and every body it changed is checked after. A body a seal
/// added drops the module analyses, which range over every body.
fn module_step<T>(module: &mut Module, am: &mut AnalysisManager<'_>) -> Vec<(BodyId, Changes)>
where
    T: ModuleTransform + Requires,
{
    if CHECKED {
        let bodies: Vec<BodyId> = module.body_ids().collect();
        for body in bodies {
            assert_form(module, body, am, T::FORM, T::NAME);
        }
    }
    let changed = T::run(module, am);
    if let Some(body_set_change) = module.take_body_set_change() {
        am.invalidate_body_set(body_set_change);
    }
    for (body, changes) in &changed {
        am.invalidate(*body, changes);
        if CHECKED && recorded_any(changes) {
            assert_kept(module, *body, T::NAME);
        }
    }
    changed
}

/// # Panics
/// `body` is not in `form`: the step before `step` left it in a form the
/// pipeline's order does not give it, a compiler defect.
fn assert_form(module: &Module, body: BodyId, am: &mut AnalysisManager<'_>, form: Form, step: &str) {
    match form {
        Form::Lowered => {}
        Form::Ssa => assert_ssa(module, body, am, step),
        Form::LoopsCanonical => {
            assert_ssa(module, body, am, step);
            assert_loops_canonical(module, body, am, step);
        }
    }
}

fn assert_ssa(module: &Module, body: BodyId, am: &mut AnalysisManager<'_>, step: &str) {
    let escapes = am.get::<Escapes>(module, body);
    for slot in module.body(body).slots() {
        assert!(
            escapes.escape(*slot).is_some(),
            "{step} requires F1 of {body:?}, and its slot {slot:?} has no use that escapes a promotion"
        );
    }
}

fn assert_loops_canonical(module: &Module, body: BodyId, am: &mut AnalysisManager<'_>, step: &str) {
    let loops = am.get::<LoopTree>(module, body);
    for of in loops.loops() {
        assert!(
            loops.preheader(of).is_some(),
            "{step} requires F2 of {body:?}, and the loop at {:?} has no preheader",
            loops.header(of)
        );
    }
}

/// # Panics
/// `step` left `body` breaking an invariant every editor keeps.
fn assert_kept(module: &Module, body: BodyId, step: &str) {
    if let Err(violation) = Kept::check(module.body(body), module.types()) {
        panic!("{step} left {body:?} breaking {}: {}", violation.invariant, violation.detail);
    }
}

/// P3*'s measure μ (D64), compared lexicographically: first the
/// instructions, other than `Const`, whose result has a use; then all
/// instructions plus blocks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct Measure {
    used_non_const: usize,
    size: usize,
}

impl Measure {
    /// Cost: O(blocks + instructions).
    fn of(module: &Module, body: BodyId) -> Measure {
        let body = module.body(body);
        let mut measure = Measure { used_non_const: 0, size: 0 };
        for (_, block) in body.blocks() {
            measure.size += 1 + block.insts().len();
            let used_non_const = block
                .insts()
                .filter(|inst| !matches!(inst.op(), Op::Const { .. }))
                .filter(|inst| inst.result().is_some_and(|result| body.use_count(result) > 0))
                .count();
            measure.used_non_const += used_non_const;
        }
        measure
    }
}

/// P3*: `round` repeated while it records any change. Returns μ before the
/// first round and after each round that changed something.
///
/// # Panics
/// In debug and test builds, a round that changed something and did not
/// strictly lower μ: a compiler defect, since the bound rests on it.
fn scalar_fixpoint<R>(module: &mut Module, body: BodyId, am: &mut AnalysisManager<'_>, round: R) -> Vec<Measure>
where
    R: Fn(&mut Module, BodyId, &mut AnalysisManager<'_>) -> Changes,
{
    let mut measures = vec![Measure::of(module, body)];
    loop {
        let changes = round(module, body, am);
        if !recorded_any(&changes) {
            return measures;
        }
        let before = *measures.last().expect("the measure before the first round is recorded");
        let after = Measure::of(module, body);
        if CHECKED {
            assert!(
                after < before,
                "P3* round {} over {body:?} recorded {:?} and did not lower its measure: {before:?} became {after:?}",
                measures.len(),
                changes.aspects().collect::<Vec<_>>()
            );
        }
        measures.push(after);
    }
}

fn recorded_any(changes: &Changes) -> bool {
    changes.aspects().next().is_some()
}

#[cfg(test)]
mod tests;
