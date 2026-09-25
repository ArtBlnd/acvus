//! Dead Code Elimination (DCE) - mark-sweep on CfgBody.
//!
//! Removes instructions that don't contribute to observable behavior.
//! Observable = Return value, context Store, IO (Eval), effectful FunctionCall.
//!
//! Algorithm:
//! 1. **Root**: instructions with side effects are unconditionally live.
//! 2. **Backward walk**: trace operands of live instructions -> mark their
//!    definitions as live -> trace their operands -> fixpoint.
//! 3. **Sweep**: remove non-live instructions.
//!
//! A store into a local slot is live only where a live instruction reads
//! that slot before the next store into it (`Stores`).
//!
//! Runs post-SSA, post-DSE. Removing a store leaves the value it stored
//! with no reader, and the release of that value comes from
//! `drop_insertion`, which `graph::optimize::run_pass2` runs after this
//! pass.

use rustc_hash::{FxHashMap, FxHashSet};

use crate::analysis::inst_info;
use crate::analysis::loans::Loans;
use crate::cfg::{BlockIdx, CfgBody, Terminator};
use crate::ir::{Inst, InstKind, Label, ValueId};
use crate::ty::Ty;
use crate::validate::move_check::is_move_only;

/// An instruction of a body, or a block's terminator at the index one past
/// that block's last instruction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct Point {
    block: usize,
    inst: usize,
}

impl Point {
    fn next(self) -> Self {
        Self {
            block: self.block,
            inst: self.inst + 1,
        }
    }
}

// -- Def location ----------------------------------------------------

/// Where a ValueId is defined.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum DefLoc {
    /// Defined by an instruction.
    Inst(Point),
    /// Defined as a block parameter.
    BlockParam { block: usize, param: usize },
    /// Function parameter or capture - always live.
    EntryParam,
}

/// Build ValueId -> DefLoc mapping.
fn build_def_map(cfg: &CfgBody) -> FxHashMap<ValueId, DefLoc> {
    let mut map = FxHashMap::default();

    // Entry params and captures are always live.
    for v in cfg.entry_defs() {
        map.insert(v, DefLoc::EntryParam);
    }

    for (bi, block) in cfg.blocks.iter().enumerate() {
        // Block params.
        for (pi, &param) in block.params.iter().enumerate() {
            map.insert(
                param,
                DefLoc::BlockParam {
                    block: bi,
                    param: pi,
                },
            );
        }

        // Instructions.
        for (ii, inst) in block.insts.iter().enumerate() {
            for d in inst_info::defs(&inst.kind) {
                map.insert(
                    d,
                    DefLoc::Inst(Point {
                        block: bi,
                        inst: ii,
                    }),
                );
            }
        }
    }

    map
}

// -- Stores into a slot ----------------------------------------------

/// The slot an `Assign` fills whole, which is the store whose liveness a
/// reader decides. A store through a reference or into a path writes
/// storage this body does not own alone, and stays a root.
fn filled_slot(kind: &InstKind) -> Option<ValueId> {
    let InstKind::Assign { target, path, .. } = kind else {
        return None;
    };
    path.is_empty()
        .then(|| inst_info::storage(target))
        .flatten()
}

/// Whether a slot's type has a release: a word owns nothing, so a store
/// into it releases nothing whatever the slot held (RFC-0048 rule 4). A type
/// the classification cannot read is taken to have one.
fn releases(slot: ValueId, cfg: &CfgBody) -> bool {
    cfg.val_types.get(&slot).and_then(is_move_only) != Some(false)
}

/// Which slots may hold a value at a point: a store lands on an occupant
/// there, and removing the store would move that occupant's release to the
/// next store or to the body's end (RFC-0045).
struct Occupancy {
    entry: Vec<FxHashSet<ValueId>>,
}

impl Occupancy {
    fn of(cfg: &CfgBody) -> Self {
        let mut entry = vec![FxHashSet::default(); cfg.blocks.len()];
        // A parameter's and a capture's slot arrives full.
        entry[0] = cfg.entry_defs().collect();

        let mut changed = true;
        while changed {
            changed = false;
            for bi in 0..cfg.blocks.len() {
                let exit = Self::through(cfg, &entry[bi], &cfg.blocks[bi].insts);
                for succ in cfg.successors(BlockIdx(bi)) {
                    for slot in &exit {
                        changed |= entry[succ.0].insert(*slot);
                    }
                }
            }
        }

        Self { entry }
    }

    fn may_hold(&self, cfg: &CfgBody, at: Point, slot: ValueId) -> bool {
        let before = &cfg.blocks[at.block].insts[..at.inst];
        Self::through(cfg, &self.entry[at.block], before).contains(&slot)
    }

    fn through(cfg: &CfgBody, entry: &FxHashSet<ValueId>, insts: &[Inst]) -> FxHashSet<ValueId> {
        let mut held = entry.clone();
        for inst in insts {
            if let Some(slot) = filled_slot(&inst.kind) {
                held.insert(slot);
                continue;
            }
            // A take moves the value out of a slot that owns one; a word
            // is copied and its slot keeps it.
            if let InstKind::Take { target, path, .. } = &inst.kind
                && path.is_empty()
                && let Some(slot) = inst_info::storage(target)
                && releases(slot, cfg)
            {
                held.remove(&slot);
            }
        }
        held
    }
}

/// The stores whose liveness is a reader's, and the readers that decide it.
struct Stores {
    /// Every store rule 2 admits removing, so `is_root` does not hold it.
    conditional: FxHashSet<Point>,
    /// For each point, the conditional stores whose slot that point reads:
    /// live wherever that point is live.
    read_at: FxHashMap<Point, Vec<Point>>,
    /// Conditional stores a terminator reads, which is live unconditionally.
    read_at_terminator: Vec<Point>,
}

impl Stores {
    fn of(loans: &Loans<'_>) -> Self {
        let cfg = loans.cfg();
        let occupancy = Occupancy::of(cfg);
        let mut this = Self {
            conditional: FxHashSet::default(),
            read_at: FxHashMap::default(),
            read_at_terminator: Vec::new(),
        };

        for (bi, block) in cfg.blocks.iter().enumerate() {
            for (ii, inst) in block.insts.iter().enumerate() {
                let at = Point {
                    block: bi,
                    inst: ii,
                };
                let Some(slot) = filled_slot(&inst.kind) else {
                    continue;
                };
                if releases(slot, cfg) && occupancy.may_hold(cfg, at, slot) {
                    continue;
                }
                this.conditional.insert(at);
                for reader in readers_from(loans, at, slot) {
                    if reader.inst == cfg.blocks[reader.block].insts.len() {
                        this.read_at_terminator.push(at);
                    } else {
                        this.read_at.entry(reader).or_default().push(at);
                    }
                }
            }
        }

        this
    }

    fn read_at(&self, point: Point) -> &[Point] {
        self.read_at.get(&point).map_or(&[], Vec::as_slice)
    }
}

/// Every point that may read `slot` on a path out of the store at `from`,
/// up to the next store into `slot` or the body's end.
fn readers_from(loans: &Loans<'_>, from: Point, slot: ValueId) -> Vec<Point> {
    let cfg = loans.cfg();
    let mut seen: FxHashSet<Point> = FxHashSet::default();
    let mut work = vec![from.next()];
    let mut readers = Vec::new();

    while let Some(at) = work.pop() {
        if !seen.insert(at) {
            continue;
        }
        let block = &cfg.blocks[at.block];
        if at.inst == block.insts.len() {
            if storage_reached(loans, terminator_values(&block.terminator)).contains(&slot) {
                readers.push(at);
            }
            work.extend(
                cfg.successors(BlockIdx(at.block))
                    .into_iter()
                    .map(|s| Point {
                        block: s.0,
                        inst: 0,
                    }),
            );
            continue;
        }
        let kind = &block.insts[at.inst].kind;
        if loans.uses_with_storage(kind).contains(&slot) {
            readers.push(at);
        }
        if filled_slot(kind) == Some(slot) {
            continue;
        }
        work.push(at.next());
    }

    readers
}

/// The storage a set of values reaches through the loans they hold: the
/// slots a use of one of them touches.
fn storage_reached(
    loans: &Loans<'_>,
    values: impl IntoIterator<Item = ValueId>,
) -> FxHashSet<ValueId> {
    let mut reached = FxHashSet::default();
    let mut work: Vec<ValueId> = values.into_iter().collect();
    while let Some(v) = work.pop() {
        if !reached.insert(v) {
            continue;
        }
        work.extend(loans.holds(v).filter_map(|l| l.storage.slot()));
    }
    reached
}

// -- Root identification ---------------------------------------------

/// Is this instruction a root (has side effects, unconditionally live)?
///
/// An instruction with ANY effect (read, write, IO) must not
/// be removed. Only provably pure instructions can be dead.
///
/// A store `Stores::conditional` holds is asked of its readers instead, and
/// this answer does not apply to it.
fn is_root(kind: &InstKind, loans: &Loans<'_>) -> bool {
    let val_types = &loans.cfg().val_types;
    if !loans.storage_effect(kind).writes.is_empty() {
        return true;
    }
    // A call handed a `&mut` may write what it names, wherever that storage
    // is: a capture's word, a parameter's referent, a local. A lambda called
    // may write through what it captured, which a body calling a lambda it
    // was handed cannot see.
    if let InstKind::FunctionCall { args, callee, .. } | InstKind::Spawn { args, callee, .. } = kind
        && (matches!(callee, crate::ir::Callee::Indirect(_))
            || args.iter().any(|arg| {
                matches!(
                    val_types.get(arg),
                    Some(Ty::Ref(crate::ty::Mutability::Mut, _))
                )
            }))
    {
        return true;
    }
    match kind {
        // A write to storage is observable; a take leaves its storage empty.
        // An element store and an append write through the `&mut` their
        // type demands, wherever the storage it names is.
        InstKind::Assign { .. }
        | InstKind::Take { .. }
        | InstKind::Fetch { .. }
        | InstKind::Commit { .. }
        | InstKind::IndexSet { .. }
        | InstKind::StringAppend { .. } => true,

        // Eval - IO execution point.
        InstKind::Eval { .. } => true,

        InstKind::Check { .. } | InstKind::CheckSteps { .. } => true,

        // A call typed `!` ends the run (RFC-0038): observable whatever its
        // effect says.
        InstKind::FunctionCall { callee_ty, .. } if matches!(callee_ty, Ty::Fn { ret, .. } if matches!(**ret, Ty::Never)) => {
            true
        }
        // A Pure call that writes no context has no effect (RFC-0007,
        // RFC-0025 rule 4): dead if its result is unused. A call whose effect is
        // unknown stays.
        InstKind::FunctionCall { callee_ty, .. } => !callee_ty
            .effect()
            .is_some_and(|e| e.is_pure() && e.writes.is_empty()),

        // Spawn: pure (deferred execution). The actual effect happens at Eval.
        // Dead if handle is unused (no Eval consumes it).
        InstKind::Spawn { .. } => false,

        // Everything else: pure computation, dead if result unused.
        _ => false,
    }
}

// -- Mark phase ------------------------------------------------------

/// The values a terminator needs regardless of any block param.
fn terminator_roots(term: &Terminator) -> Vec<ValueId> {
    match term {
        Terminator::Return { value, order, .. } => std::iter::once(*value).chain(*order).collect(),
        Terminator::JumpIf { cond, .. } | Terminator::Diamond { cond, .. } => vec![*cond],
        // The tag a `Switch` reads is live wherever the dispatch is.
        Terminator::Switch { tag, .. } => vec![*tag],
        // A `For` reads its source on every iteration, so the source is a
        // root of the traversal (RFC-0057).
        Terminator::For { source, .. } => source.uses().to_vec(),
        Terminator::Jump { .. } | Terminator::Fallthrough | Terminator::Diverge => vec![],
    }
}

/// Every value a terminator names, block arguments included: what it may
/// carry a slot's loan out through.
fn terminator_values(term: &Terminator) -> Vec<ValueId> {
    let mut values = terminator_roots(term);
    match term {
        Terminator::Jump { args, .. } => values.extend(args),
        Terminator::JumpIf {
            then_args,
            else_args,
            ..
        }
        | Terminator::Diamond {
            then_args,
            else_args,
            ..
        } => values.extend(then_args.iter().chain(else_args)),
        Terminator::For { exit_args, .. } => values.extend(exit_args),
        Terminator::Switch { arms, default, .. } => {
            for (_, _, args) in arms {
                values.extend(args);
            }
            if let Some((_, args)) = default {
                values.extend(args);
            }
        }
        Terminator::Return { .. } | Terminator::Fallthrough | Terminator::Diverge => {}
    }
    values
}

// -- Public API ------------------------------------------------------

/// Run DCE on a CfgBody. Removes all instructions that don't contribute
/// to observable behavior (Return, Store, Eval, effectful calls).
pub fn run(cfg: &mut CfgBody) {
    let def_map = build_def_map(cfg);
    let loans = Loans::build(cfg);
    let stores = Stores::of(&loans);

    // Live instruction set.
    let mut live_insts: FxHashSet<Point> = FxHashSet::default();
    // Live terminators (always live, but track for block param tracing).
    let mut live_terminators: FxHashSet<usize> = FxHashSet::default();
    // Worklist of ValueIds to trace.
    let mut worklist: Vec<ValueId> = Vec::new();

    // Phase 1: seed roots.
    for (bi, block) in cfg.blocks.iter().enumerate() {
        for (ii, inst) in block.insts.iter().enumerate() {
            let at = Point {
                block: bi,
                inst: ii,
            };
            if !stores.conditional.contains(&at) && is_root(&inst.kind, &loans) {
                live_insts.insert(at);
                worklist.extend(inst_info::uses(&inst.kind));
            }
        }

        // Terminators are always live. A returned value and a branch
        // condition are roots; a jump argument is live only when the block
        // param it feeds is, and BlockParam tracing pulls it in then.
        live_terminators.insert(bi);
        worklist.extend(terminator_roots(&block.terminator));
    }

    // Phase 2: backward walk, alternating with the stores the live
    // instructions read.
    let mut live_values: FxHashSet<ValueId> = FxHashSet::default();
    let mut pending: Vec<Point> = stores.read_at_terminator.clone();

    loop {
        while let Some(at) = pending.pop() {
            if live_insts.insert(at) {
                worklist.extend(inst_info::uses(&cfg.blocks[at.block].insts[at.inst].kind));
            }
        }

        while let Some(val) = worklist.pop() {
            if !live_values.insert(val) {
                continue; // Already processed.
            }

            let Some(&def_loc) = def_map.get(&val) else {
                continue; // External value (not defined in this body).
            };

            match def_loc {
                DefLoc::Inst(at) => {
                    if live_insts.insert(at) {
                        // Newly live - trace its operands.
                        worklist.extend(inst_info::uses(&cfg.blocks[at.block].insts[at.inst].kind));
                    }
                }
                DefLoc::BlockParam { block, param } => {
                    // Block param is live -> trace corresponding jump args from predecessors.
                    let block_label = cfg.blocks[block].label;
                    for pred_block in cfg.blocks.iter() {
                        // A `Switch` can reach one block through several arms, so
                        // an edge list, not one edge (RFC-0051). The `usize` is
                        // the first parameter the edge carries an argument for:
                        // a `For`'s body edge starts after the parameters the
                        // terminator fills itself (RFC-0057).
                        let pred_args: Vec<(usize, &[ValueId])> = match &pred_block.terminator {
                            Terminator::Jump { label, args } if *label == block_label => {
                                vec![(0, args.as_slice())]
                            }
                            Terminator::JumpIf {
                                then_label,
                                then_args,
                                else_label,
                                else_args,
                                ..
                            }
                            | Terminator::Diamond {
                                then_label,
                                then_args,
                                else_label,
                                else_args,
                                ..
                            } => [(then_label, then_args), (else_label, else_args)]
                                .into_iter()
                                .filter(|(label, _)| **label == block_label)
                                .map(|(_, args)| (0, args.as_slice()))
                                .collect(),
                            Terminator::Switch { arms, default, .. } => arms
                                .iter()
                                .map(|(_, label, args)| (label, args))
                                .chain(default.iter().map(|(label, args)| (label, args)))
                                .filter(|(label, _)| **label == block_label)
                                .map(|(_, args)| (0, args.as_slice()))
                                .collect(),
                            Terminator::For {
                                exit,
                                exit_trip,
                                exit_args,
                                ..
                            } => [(*exit, exit_trip.supplied_params(), exit_args.as_slice())]
                                .into_iter()
                                .filter(|(label, _, _)| *label == block_label)
                                .map(|(_, first, args)| (first, args))
                                .collect(),
                            _ => Vec::new(),
                        };
                        for (first, args) in pred_args {
                            if let Some(&arg) = param.checked_sub(first).and_then(|i| args.get(i)) {
                                worklist.push(arg);
                            }
                        }
                    }
                }
                DefLoc::EntryParam => {
                    // Function param/capture - always live, nothing to trace.
                }
            }
        }

        pending.extend(
            live_insts
                .iter()
                .flat_map(|at| stores.read_at(*at))
                .filter(|store| !live_insts.contains(store))
                .copied(),
        );
        if pending.is_empty() {
            break;
        }
    }

    // Phase 3: sweep - remove dead instructions.
    for (bi, block) in cfg.blocks.iter_mut().enumerate() {
        let mut ii = 0;
        block.insts.retain(|_| {
            let keep = live_insts.contains(&Point {
                block: bi,
                inst: ii,
            });
            ii += 1;
            keep
        });
    }

    // Phase 4: sweep - remove dead block params and the jump args that
    // fed them. A dead param is a phi nothing reads; left in place it
    // would carry a second copy of a move-only value.
    //
    // A `For`'s body takes its element and its counter by position from the
    // terminator, and its exit the trip count where the edge defines one,
    // so those parameters stay whether anything reads them: the machine's
    // `For` writes them there (RFC-0057 rules 2 and 9).
    let supplied: FxHashMap<Label, usize> = cfg
        .blocks
        .iter()
        .filter_map(|block| {
            let Terminator::For {
                source,
                stages,
                exit,
                exit_trip,
                ..
            } = &block.terminator
            else {
                return None;
            };
            Some([
                (stages.body(), source.supplied_params()),
                (*exit, exit_trip.supplied_params()),
            ])
        })
        .flatten()
        .filter(|(_, pinned)| *pinned > 0)
        .collect();
    let dead_params: Vec<(Label, Vec<usize>)> = cfg
        .blocks
        .iter()
        .map(|block| {
            let pinned = supplied.get(&block.label).copied().unwrap_or(0);
            let dead = block
                .params
                .iter()
                .enumerate()
                .skip(pinned)
                .filter(|(_, p)| !live_values.contains(p))
                .map(|(pi, _)| pi)
                .collect();
            (block.label, dead)
        })
        .filter(|(_, dead): &(Label, Vec<usize>)| !dead.is_empty())
        .collect();
    if dead_params.is_empty() {
        return;
    }
    let dead_of = |label: Label| -> Option<&Vec<usize>> {
        dead_params
            .iter()
            .find(|(l, _)| *l == label)
            .map(|(_, d)| d)
    };
    let prune = |args: &mut Vec<ValueId>, dead: &[usize]| {
        let mut pi = 0;
        args.retain(|_| {
            let keep = !dead.contains(&pi);
            pi += 1;
            keep
        });
    };
    for block in &mut cfg.blocks {
        match &mut block.terminator {
            Terminator::Jump { label, args } => {
                if let Some(dead) = dead_of(*label) {
                    prune(args, dead);
                }
            }
            Terminator::JumpIf {
                then_label,
                then_args,
                else_label,
                else_args,
                ..
            }
            | Terminator::Diamond {
                then_label,
                then_args,
                else_label,
                else_args,
                ..
            } => {
                if let Some(dead) = dead_of(*then_label) {
                    prune(then_args, dead);
                }
                if let Some(dead) = dead_of(*else_label) {
                    prune(else_args, dead);
                }
            }
            Terminator::For {
                exit,
                exit_trip,
                exit_args,
                ..
            } => {
                if let Some(dead) = dead_of(*exit) {
                    let first = exit_trip.supplied_params();
                    let shifted: Vec<usize> = dead.iter().map(|pi| pi - first).collect();
                    prune(exit_args, &shifted);
                }
            }
            Terminator::Switch { arms, default, .. } => {
                for (_, label, args) in arms.iter_mut() {
                    if let Some(dead) = dead_of(*label) {
                        prune(args, dead);
                    }
                }
                if let Some((label, args)) = default
                    && let Some(dead) = dead_of(*label)
                {
                    prune(args, dead);
                }
            }
            Terminator::Return { .. } | Terminator::Fallthrough | Terminator::Diverge => {}
        }
        if let Some(dead) = dead_of(block.label) {
            prune(&mut block.params, dead);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cfg;
    use crate::graph::QualifiedRef;
    use crate::ir::{Callee, DebugInfo, Inst, MirBody, RefTarget};
    use crate::ty::Effect;
    use crate::ty::Ty;
    use acvus_utils::{Interner, LocalFactory, LocalIdOps};
    use rustc_hash::FxHashMap;

    fn v(n: usize) -> ValueId {
        ValueId::from_raw(n)
    }

    /// `dst = name()` with the given effect; its result is never used.
    fn unused_call(i: &Interner, name: &str, effect: Effect, dst: usize) -> InstKind {
        returning_call(i, name, effect, dst, Ty::I64)
    }

    /// `dst = name() -> ret` with the given effect.
    fn returning_call(i: &Interner, name: &str, effect: Effect, dst: usize, ret: Ty) -> InstKind {
        InstKind::FunctionCall {
            dst: v(dst),
            callee: Callee::Direct(QualifiedRef::root(i.intern(name))),
            callee_ty: Ty::Fn {
                params: vec![],
                ret: Box::new(ret),
                captures: vec![],
                effect: effect.into(),
                flows: crate::ty::Flows::Every.into(),
            },
            args: vec![],
            order: None,
        }
    }

    /// `assign slot = value`, the whole slot.
    fn assign(slot: usize, value: usize) -> InstKind {
        InstKind::Assign {
            target: RefTarget::Var(v(slot)),
            path: Vec::new(),
            value: v(value),
            restores: false,
        }
    }

    fn body(insts: Vec<InstKind>, val_count: usize) -> CfgBody {
        typed_body(insts, &vec![Ty::I64; val_count])
    }

    fn typed_body(insts: Vec<InstKind>, types: &[Ty]) -> CfgBody {
        let mut factory = LocalFactory::<ValueId>::new();
        let mut val_types = FxHashMap::default();
        for ty in types {
            val_types.insert(factory.next(), ty.clone());
        }
        cfg::promote(MirBody {
            demoted_diamonds: Default::default(),
            insts: insts
                .into_iter()
                .map(|kind| Inst {
                    span: acvus_ast::Span::ZERO,
                    kind,
                })
                .collect(),
            val_types,
            params: Vec::new(),
            captures: Vec::new(),
            order_param: None,
            task: crate::ty::Task::Sync,
            debug: DebugInfo::new(),
            val_factory: factory,
            label_count: 0,
        })
    }

    fn calls(cfg: &CfgBody) -> usize {
        cfg.blocks
            .iter()
            .flat_map(|b| b.insts.iter())
            .filter(|i| matches!(i.kind, InstKind::FunctionCall { .. }))
            .count()
    }

    /// The value each surviving `Assign` stores, in program order.
    fn stored(cfg: &CfgBody) -> Vec<ValueId> {
        cfg.blocks
            .iter()
            .flat_map(|b| b.insts.iter())
            .filter_map(|i| match i.kind {
                InstKind::Assign { value, .. } => Some(value),
                _ => None,
            })
            .collect()
    }

    fn refs(cfg: &CfgBody) -> usize {
        cfg.blocks
            .iter()
            .flat_map(|b| b.insts.iter())
            .filter(|i| matches!(i.kind, InstKind::Ref { .. }))
            .count()
    }

    /// `v0 = opaque() -> String` into a slot `v1` of the same type, and a
    /// word returned: the store has no reader.
    fn opaque_store_never_read(i: &Interner, tail: Vec<InstKind>) -> CfgBody {
        let mut insts = vec![
            returning_call(i, "make", Effect::OPAQUE, 0, Ty::String),
            assign(1, 0),
        ];
        insts.extend(tail);
        insts.push(InstKind::Const {
            dst: v(3),
            value: acvus_ast::Literal::Int(1),
        });
        insts.push(InstKind::Return {
            value: v(3),
            order: None,
        });
        typed_body(
            insts,
            &[
                Ty::String,
                Ty::String,
                Ty::Ref(
                    crate::ty::Mutability::Shared,
                    Box::new(crate::ty::TypeArg::uniform(Ty::String)),
                ),
                Ty::I64,
            ],
        )
    }

    #[test]
    fn a_store_no_one_reads_is_dead_and_its_producer_stays() {
        let i = Interner::new();
        let mut cfg = opaque_store_never_read(&i, Vec::new());
        run(&mut cfg);
        assert_eq!(stored(&cfg), Vec::new(), "a store with no reader is dead");
        assert_eq!(
            calls(&cfg),
            1,
            "the effectful call that produced the stored value stays"
        );
    }

    #[test]
    fn a_dead_ref_of_the_slot_does_not_keep_the_store() {
        let i = Interner::new();
        let mut cfg = opaque_store_never_read(
            &i,
            vec![InstKind::Ref {
                dst: v(2),
                target: RefTarget::Var(v(1)),
                path: Vec::new(),
                mutability: crate::ty::Mutability::Shared,
            }],
        );
        run(&mut cfg);
        assert_eq!(refs(&cfg), 0, "a reference no one reads is dead");
        assert_eq!(
            stored(&cfg),
            Vec::new(),
            "a read from a dead instruction does not keep the store alive"
        );
    }

    #[test]
    fn a_store_onto_an_occupant_stays() {
        let i = Interner::new();
        let mut cfg = typed_body(
            vec![
                returning_call(&i, "make", Effect::OPAQUE, 0, Ty::String),
                assign(2, 0),
                returning_call(&i, "make", Effect::OPAQUE, 1, Ty::String),
                assign(2, 1),
                InstKind::Const {
                    dst: v(3),
                    value: acvus_ast::Literal::Int(1),
                },
                InstKind::Return {
                    value: v(3),
                    order: None,
                },
            ],
            &[Ty::String, Ty::String, Ty::String, Ty::I64],
        );
        run(&mut cfg);
        assert_eq!(
            stored(&cfg),
            vec![v(1)],
            "the store that lands on an occupant releases it and stays; \
             the store into the empty slot goes"
        );
    }

    #[test]
    fn a_store_into_a_word_slot_is_dead_whatever_the_slot_holds() {
        let mut cfg = body(
            vec![
                InstKind::Const {
                    dst: v(0),
                    value: acvus_ast::Literal::Int(1),
                },
                assign(2, 0),
                InstKind::Const {
                    dst: v(1),
                    value: acvus_ast::Literal::Int(2),
                },
                assign(2, 1),
                InstKind::Return {
                    value: v(0),
                    order: None,
                },
            ],
            4,
        );
        run(&mut cfg);
        assert_eq!(
            stored(&cfg),
            Vec::new(),
            "a word slot has no release, so neither store has to stand"
        );
    }

    #[test]
    fn an_unused_pure_call_is_dead() {
        let i = Interner::new();
        let mut cfg = body(
            vec![
                InstKind::Const {
                    dst: v(0),
                    value: acvus_ast::Literal::Int(1),
                },
                unused_call(&i, "len", Effect::PURE, 1),
                InstKind::Return {
                    value: v(0),
                    order: None,
                },
            ],
            2,
        );
        run(&mut cfg);
        assert_eq!(
            calls(&cfg),
            0,
            "a Pure call with an unused result is removed"
        );
    }

    #[test]
    fn an_unused_effectful_call_stays() {
        let i = Interner::new();
        for effect in [Effect::IDEMPOTENT.commutative(), Effect::OPAQUE] {
            let mut cfg = body(
                vec![
                    InstKind::Const {
                        dst: v(0),
                        value: acvus_ast::Literal::Int(1),
                    },
                    unused_call(&i, "put", effect, 1),
                    InstKind::Return {
                        value: v(0),
                        order: None,
                    },
                ],
                2,
            );
            run(&mut cfg);
            assert_eq!(
                calls(&cfg),
                1,
                "an effectful call stays whether or not it commutes"
            );
        }
    }
}
