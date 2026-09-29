//! The required steps after lowering (D40, D55, D71): meaning, not
//! optimization. They run on every program, at every level, in the order
//! `clone_strings`, `remove_dead_context_ops`, the checks, `insert_drops`.
//! Each edits every body of the module through its own scopes' editors,
//! with `EditFacts` as their needs where they read facts, and invalidates
//! the manager for what it changed.
//! - `clone_strings`, scopes `Clones` and `Value`: a `String` use that is
//!   not its last is a clone, and its last use moves it (ownership §3),
//!   through `clone_read` (S4) for a place's content and `insert` and
//!   `replace_operand` (G3) for a value.
//! - `remove_dead_context_ops`, scopes `Discharge` and `Remove`: a load
//!   whose value is never read and a commit of a context the body never
//!   writes do not exist (context §3), through `discharge` (G1).
//! - `insert_drops`, scope `Cfg` for an edge split and then `Drops`: what
//!   the program holds that does not copy is dropped where its liveness
//!   ends, and what moves to another holder is not (ownership §6), through
//!   `split_edge` (G2), `insert_drop` and `drop_content` (S3). It runs after
//!   the checks, which admit the moves and the initializations.

use std::collections::BTreeSet;

use acvus_ast::Span;
use rustc_hash::{FxHashMap, FxHashSet};
use smallvec::SmallVec;

use crate::ir::cfg::edit::{
    CannotDischarge, CannotRemoveParam, Cfg, Clones, Discharge, Drops, Editor, InstRef, NoFacts, PartOf, Position,
    Remove, Value,
};
use crate::ir::cfg::takings::{Named, Taken, by_value, inst_takings, terminator_takings};
use crate::ir::cfg::{BlockId, Body, BodyId, Changes, Definition, Inst, InstId, Module, Op, Site, Terminator};
use crate::ir::parts::{self, Path, covers, overlap};
use crate::ir::{Callee, PathSeg, RefTarget, ValueId};
use crate::passes::analysis::{AnalysisManager, ContentLiveness, Liveness, Loans, Presence, Reach, Storage};
use crate::passes::edit_facts::EditFacts;
use crate::ty::{Calls, Concrete, Mutability, Ty, TyKind, Types, copies};

#[cfg(test)]
mod tests;

/// Every required step over `module`, in the module doc's order, the checks
/// among them: the refusals where the checks refuse the program, which
/// then takes no step after them (refusals: the checker is the only place
/// a program is refused).
pub fn run(module: &mut Module, am: &mut AnalysisManager<'_>) -> Result<(), Vec<crate::check::Refusal>> {
    clone_strings(module, am);
    remove_dead_context_ops(module, am);
    let refusals = crate::check::check(module, am);
    if !refusals.is_empty() {
        return Err(refusals);
    }
    insert_drops(module, am);
    Ok(())
}

/// Every body: a `String` a use moves while the same content is read again
/// on some path after it is cloned just before that use, with the use's
/// span, and the use takes the clone (ownership §3). A `String` part of a
/// place read by a `Take`, the whole included, becomes a clone of a shared
/// `Ref` of it where that same part is read again, per part as
/// `ContentLiveness` keeps it (D89); a value's use takes a `StringClone` of
/// the value. Of the
/// operands of one site that move one value, all but the last are clones,
/// and the last is one too where the value is read after the site. A clone
/// before a terminator takes the terminator's span.
///
/// Cost per body: `ContentLiveness`' and `Liveness`', one `clone_read` per
/// place read, and one insertion and one `replace_operand` per value use
/// cloned, each an edit that drops the facts it made stale; a dominance
/// check computes `Dominators` again only after an edit of the CFG's shape.
pub fn clone_strings(module: &mut Module, am: &mut AnalysisManager<'_>) {
    for body in module.body_ids() {
        clone_place_reads(module, body, am);
        let changes = clone_value_uses(module, body, am);
        am.invalidate(body, &changes);
    }
}

/// Every body: each commit of a context the body never writes, then each
/// load whose stored value no read reaches, is offered to `discharge`,
/// which removes it where `EditFacts` proves it `NeverWritten` or
/// `NeverRead`; a volatile context's are never proven (D61). A commit is
/// the `Take` of the context's slot and the `Commit` of what it took, so a
/// discharged commit's `Take` goes with it, and the content stays the
/// page's (context §3). A load is the `Fetch` and the `Assign` of its value
/// into the context's slot, so that store is offered before the `Fetch`.
///
/// Cost per body: O(instructions) discharges and removals, each judged on
/// `Loans` and `IntervalValues` as the edits before it left them.
pub fn remove_dead_context_ops(module: &mut Module, am: &mut AnalysisManager<'_>) {
    for body in module.body_ids() {
        remove_dead_context_ops_in(module, body, am);
    }
}

/// Every body: what the program holds that does not copy (ownership §1) is
/// dropped where its liveness ends (ownership §6).
/// - A value: just after its last use, unless that use moves it to another
///   holder; on an edge where it is live out of the block and not into the
///   target, the edge is split and it is dropped at the new block's start,
///   with the span of the edge's source terminator, where it ends.
/// - A slot's content, or a by-value parameter's storage content: the
///   same, by `ContentLiveness`, through `drop_content`; a store that no read
///   follows ends the content it stores. A `Take` through a reference to
///   the place alone moves out of it as one by name does.
/// - A context's slot: its content is the page's until the body overwrites
///   it; the old content of an overwrite is the body's and is dropped just
///   before the store, and the body never drops the content at its end
///   (context §3, D71).
/// - A value or a place's content some part of which that does not copy is
///   moved out, or stored into: the part moved is ended by its holder, each
///   other unit (`ir::parts`) ends where its own liveness ends, dropped
///   through `drop_part`, and the whole is never dropped (ownership §6). A
///   place's unit is dropped only where it is present; where it is present
///   on some paths only, its liveness there is a read `Loans` could not
///   locate, which no path that moved it reaches, so it is dropped on each
///   edge into the join where it is present.
///
/// A capture is the closure's, an order token ends in no drop, and a
/// parameter the body names as a storage is a place, not a value. A
/// variant's payload moved out moves all the variant owns, since its tag
/// copies and it has one payload, so an unwrap moves the variant whole.
///
/// Cost per body: `Liveness`', `ContentLiveness`', O(values · blocks) for
/// the plan, O(units · (blocks + edges)) for each cut value's units'
/// liveness, one `split_edge` per edge that ends something, and one
/// `insert_drop`, `drop_content` or `drop_part` per drop, each judged on
/// `Dominators` and `ContentLiveness` as the drops before it left them.
///
/// # Panics
/// A parameter is dead where the body or its block starts, where no edge of
/// its own carries the drop; a context slot's part is read or taken on some
/// paths only before an overwrite; a parameter is both a value and a
/// storage. Each message states the question.
pub fn insert_drops(module: &mut Module, am: &mut AnalysisManager<'_>) {
    for body in module.body_ids() {
        insert_drops_in(module, body, am);
    }
}

/// For each edge of `terminator`, in `Terminator::edges`' order, the index
/// among `Terminator::operands` of its first argument.
fn edge_arg_offsets(terminator: &Terminator) -> SmallVec<[usize; 2]> {
    match terminator {
        Terminator::Jump { .. } => smallvec::smallvec![0],
        Terminator::JumpIf { then_args, .. } | Terminator::Diamond { then_args, .. } => {
            smallvec::smallvec![1, 1 + then_args.len()]
        }
        Terminator::Switch { arms, default, .. } => {
            let mut offsets = SmallVec::new();
            let mut next = 1;
            for args in arms.iter().map(|(_, _, args)| args).chain(default.iter().map(|(_, args)| args)) {
                offsets.push(next);
                next += args.len();
            }
            offsets
        }
        Terminator::For { source, .. } => {
            let exit = source.uses().len();
            smallvec::smallvec![exit, exit]
        }
        Terminator::While { .. } => smallvec::smallvec![1, 1],
        Terminator::Return { .. } | Terminator::Diverge | Terminator::NoArmMatched { .. } => SmallVec::new(),
    }
}

fn is_string(types: &Types<Concrete>, ty: Ty) -> bool {
    matches!(types.kind(ty), TyKind::String)
}

/// Whether `site` stands after the instruction `inst`, in its block.
fn later_in_block(body: &Body, site: Site, inst: InstId) -> bool {
    let block = body.block_of(inst);
    match site {
        Site::Inst(later) => body.block_of(later) == block && body.comes_before(inst, later),
        Site::Term(at) => at == block,
    }
}

/// The place a direct `Take`, `Ref` or `Assign` names: a slot or a by-value
/// parameter's own storage.
fn named_place(target: &RefTarget) -> Option<ValueId> {
    match target {
        RefTarget::Var(place) | RefTarget::Param(place) => Some(*place),
        RefTarget::Through(_) => None,
    }
}

// -- clone_strings -----------------------------------------------------------

/// Each `Take` of a `String`, whole or a field, whose part of the place is
/// read again after it becomes a clone of that part (S4).
fn clone_place_reads(module: &mut Module, body: BodyId, am: &mut AnalysisManager<'_>) {
    let content = am.get::<ContentLiveness>(module, body);
    let (of, types) = (module.body(body), module.types());
    let mut read_again: Vec<ValueId> = Vec::new();
    for (_, record) in of.blocks() {
        for inst in record.insts() {
            let Op::Take { target, path, .. } = inst.op() else {
                continue;
            };
            let result = inst.result().expect("a Take yields a value");
            let Some(place) = named_place(target) else {
                continue;
            };
            if is_string(types, of.ty_of(result)) && content.part_live_after(Site::Inst(inst.id()), place, path) {
                read_again.push(result);
            }
        }
    }
    let mut no_facts = NoFacts;
    let mut editor = Editor::<Clones>::new(module, body, &mut no_facts);
    for read in read_again {
        let take = editor.defining_inst(read).expect("a Take's result is defined by it");
        editor.clone_read(take).expect("a Take whose result is a String reads a String place");
    }
    let changes = editor.finish();
    am.invalidate(body, &changes);
}

fn clone_value_uses(module: &mut Module, body: BodyId, am: &mut AnalysisManager<'_>) -> Changes {
    let liveness = am.get::<Liveness>(module, body);
    let mut facts = EditFacts::new(am, body);
    let mut editor = Editor::<Value>::new(module, body, &mut facts);
    let mut cursor = editor.cursor();
    while let Some(inst) = cursor.next() {
        clone_before_inst(cursor.editor(), &liveness, inst);
    }
    let blocks: Vec<BlockId> = editor.body().blocks().map(|(block, _)| block).collect();
    for block in blocks {
        clone_before_terminator(&mut editor, &liveness, block);
    }
    editor.finish()
}

/// A `String` value one site moves, and the operand indices that move it.
struct MovedString {
    value: ValueId,
    ty: Ty,
    operands: SmallVec<[usize; 2]>,
}

/// Clones each `String` value the instruction at `inst` moves through an
/// operand that is not its last use, and gives that operand the clone.
fn clone_before_inst(editor: &mut Editor<'_, Value>, liveness: &Liveness, inst: InstRef) {
    let block = editor.block_of(inst);
    let record = editor.inst_of(inst);
    let span = record.span();
    let mut moved: Vec<MovedString> = Vec::new();
    for (position, operand) in inst_takings(editor.body(), editor.types(), record).into_iter().enumerate() {
        let ty = editor.body().ty_of(operand.value);
        if operand.names_place || operand.taken != Taken::Moved || !is_string(editor.types(), ty) {
            continue;
        }
        match moved.iter_mut().find(|string| string.value == operand.value) {
            Some(string) => string.operands.push(position),
            None => moved.push(MovedString { value: operand.value, ty, operands: smallvec::smallvec![position] }),
        }
    }
    for MovedString { value, ty, operands } in moved {
        let body = editor.body();
        let read_after = body.users(value).any(|site| later_in_block(body, site, inst.id()))
            || liveness.live_at_end(block).contains(&value);
        let cloned = if read_after { &operands[..] } else { &operands[..operands.len() - 1] };
        for position in cloned {
            let clone = editor.insert(Position::Before(inst), Op::StringClone { src: value }, ty, span);
            editor
                .replace_operand(Site::Inst(inst.id()), *position, value, clone)
                .expect("a clone inserted just before its use dominates it");
        }
    }
}

/// An edge's argument that takes a clone, by its index among the
/// terminator's operands.
struct ClonedArg {
    value: ValueId,
    operand: usize,
}

/// On each edge of `block`'s terminator, a `String` passed through more
/// than one argument is cloned for all but the last, and for the last too
/// where the target reads it by name; each clone stands before the
/// terminator with its span. Only one edge is taken, so each edge is
/// judged alone.
fn clone_before_terminator(editor: &mut Editor<'_, Value>, liveness: &Liveness, block: BlockId) {
    let record = editor.body().block(block);
    let span = record.terminator_span();
    let terminator = record.terminator();
    let mut cloned: Vec<ClonedArg> = Vec::new();
    for (edge, first) in terminator.edges().iter().zip(edge_arg_offsets(terminator)) {
        for (position, arg) in edge.args.iter().enumerate() {
            if !is_string(editor.types(), editor.body().ty_of(*arg)) {
                continue;
            }
            let passed_again = edge.args[position + 1..].contains(arg);
            if passed_again || liveness.live_at_start(edge.to).contains(arg) {
                cloned.push(ClonedArg { value: *arg, operand: first + position });
            }
        }
    }
    for ClonedArg { value, operand } in cloned {
        let ty = editor.body().ty_of(value);
        let clone = editor.insert(Position::End(block), Op::StringClone { src: value }, ty, span);
        editor
            .replace_operand(Site::Term(block), operand, value, clone)
            .expect("a clone inserted at its block's end dominates the terminator");
    }
}

// -- remove_dead_context_ops -------------------------------------------------

fn remove_dead_context_ops_in(module: &mut Module, body: BodyId, am: &mut AnalysisManager<'_>) {
    let mut taken_for_commits: Vec<ValueId> = Vec::new();
    {
        let mut facts = EditFacts::new(am, body);
        let mut editor = Editor::<Discharge>::new(module, body, &mut facts);
        let mut cursor = editor.cursor();
        while let Some(inst) = cursor.next() {
            let editor = cursor.editor();
            let Op::Commit { value, .. } = editor.body().inst(inst.id()).op() else {
                continue;
            };
            let value = *value;
            if offer(editor, inst) && commits_its_slot(editor.body(), editor.defining_inst(value), value) {
                taken_for_commits.push(value);
            }
        }
        let changes = editor.finish();
        am.invalidate(body, &changes);
    }
    {
        let mut no_facts = NoFacts;
        let mut editor = Editor::<Remove>::new(module, body, &mut no_facts);
        for taken in taken_for_commits {
            let take = editor.defining_inst(taken).expect("a commit's Take defines what it took");
            editor.remove(take).expect("a Take whose only use was a discharged commit is unused and pure");
        }
        let changes = editor.finish();
        am.invalidate(body, &changes);
    }
    let mut facts = EditFacts::new(am, body);
    let mut editor = Editor::<Discharge>::new(module, body, &mut facts);
    let mut cursor = editor.cursor();
    while let Some(inst) = cursor.next() {
        let editor = cursor.editor();
        if stores_a_load(editor, inst) {
            offer(editor, inst);
        }
    }
    let mut cursor = editor.cursor();
    while let Some(inst) = cursor.next() {
        let editor = cursor.editor();
        if matches!(editor.body().inst(inst.id()).op(), Op::Fetch { .. }) {
            offer(editor, inst);
        }
    }
    let changes = editor.finish();
    am.invalidate(body, &changes);
}

/// Offers `inst` to `discharge`; whether it went.
fn offer(editor: &mut Editor<'_, Discharge>, inst: InstRef) -> bool {
    match editor.discharge(inst) {
        Ok(()) => true,
        Err(CannotDischarge::NotProven | CannotDischarge::WrongFact) => false,
    }
}

/// Whether `value`, which a discharged commit passed, is a whole `Take` of a
/// context's slot that nothing else uses: the first half of the commit.
fn commits_its_slot(body: &Body, take: Option<InstRef>, value: ValueId) -> bool {
    let Some(take) = take else {
        return false;
    };
    let Op::Take { target: RefTarget::Var(slot), path, .. } = body.inst(take.id()).op() else {
        return false;
    };
    path.is_empty() && body.context_of_slot(*slot).is_some() && body.use_count(value) == 0
}

/// Whether `inst` stores into a context's slot, whole, the value a `Fetch`
/// of that context produced: the second half of the load (context §3).
fn stores_a_load(editor: &Editor<'_, Discharge>, inst: InstRef) -> bool {
    let body = editor.body();
    let Op::Assign { target: RefTarget::Var(slot), path, value, .. } = body.inst(inst.id()).op()
    else {
        return false;
    };
    path.is_empty() && body.context_of_slot(*slot).is_some() && is_load_of(body, editor.defining_inst(*value), *slot)
}

/// Whether `loaded` is a `Fetch` of the context `slot` holds.
fn is_load_of(body: &Body, loaded: Option<InstRef>, slot: ValueId) -> bool {
    let Some(loaded) = loaded else {
        return false;
    };
    matches!(
        body.inst(loaded.id()).op(),
        Op::Fetch { context } if Some(*context) == body.context_of_slot(slot)
    )
}

// -- insert_drops ------------------------------------------------------------

/// What a drop ends: a value, the content of a place, or one unit of
/// either.
#[derive(Debug, Clone)]
enum Ended {
    Value(ValueId),
    Content(ValueId),
    Part { of: PartOf, unit: Path },
}

/// A drop the plan places, with the span it stands at.
#[derive(Debug, Clone)]
struct PlannedDrop {
    ended: Ended,
    span: Span,
}

/// The drops on one edge, which is split so they stand at the start of the
/// new block.
struct EdgeDrops {
    from: BlockId,
    to: BlockId,
    drops: Vec<PlannedDrop>,
}

/// Where every drop of a body stands: just before or just after an
/// instruction, named by its identity, which the drops put in do not
/// change, or on an edge.
#[derive(Default)]
struct DropPlan {
    /// At a block's start: the entry's, for a parameter dead there.
    at_start: FxHashMap<BlockId, Vec<PlannedDrop>>,
    before: FxHashMap<Site, Vec<PlannedDrop>>,
    after: FxHashMap<Site, Vec<PlannedDrop>>,
    on_edges: Vec<EdgeDrops>,
}


fn insert_drops_in(module: &mut Module, body: BodyId, am: &mut AnalysisManager<'_>) {
    remove_dead_block_params(module, body, am);
    let liveness = am.get::<Liveness>(module, body);
    let content = am.get::<ContentLiveness>(module, body);
    let loans = am.get::<Loans>(module, body);
    let facts = PlanFacts { body: module.body(body), types: module.types(), liveness: &liveness, content: &content, loans: &loans };
    let DropPlan { mut at_start, mut before, mut after, on_edges } = facts.plan();

    let mut no_facts = NoFacts;
    let mut editor = Editor::<Cfg>::new(module, body, &mut no_facts);
    let mut by_edge: Vec<EdgeDrops> = Vec::new();
    for EdgeDrops { from, to, drops } in on_edges {
        match by_edge.iter_mut().find(|edge| edge.from == from && edge.to == to) {
            Some(edge) => edge.drops.extend(drops),
            None => by_edge.push(EdgeDrops { from, to, drops }),
        }
    }
    let split: Vec<EdgeDrops> = by_edge
        .into_iter()
        .map(|EdgeDrops { from, to, drops }| EdgeDrops { from, to: editor.split_edge(from, to), drops })
        .collect();
    let changes = editor.finish();
    am.invalidate(body, &changes);

    let mut facts = EditFacts::new(am, body);
    let mut editor = Editor::<Drops>::new(module, body, &mut facts);
    let mut owed: Vec<PlannedDrop> = Vec::new();
    let mut walking: Option<BlockId> = None;
    let mut cursor = editor.cursor();
    while let Some(inst) = cursor.next() {
        let editor = cursor.editor();
        let block = editor.block_of(inst);
        if walking != Some(block) {
            if let Some(walked) = walking {
                drop_all(editor, owed.drain(..), Position::End(walked));
            }
            owed.extend(at_start.remove(&block).into_iter().flatten());
        }
        let original = Site::Inst(inst.id());
        owed.extend(before.remove(&original).into_iter().flatten());
        drop_all(editor, owed.drain(..), Position::Before(inst));
        owed.extend(after.remove(&original).into_iter().flatten());
        walking = Some(block);
    }
    if let Some(walked) = walking {
        drop_all(&mut editor, owed.drain(..), Position::End(walked));
    }
    for EdgeDrops { to: middle, drops, .. } in split {
        drop_all(&mut editor, drops.into_iter(), Position::End(middle));
    }
    for (block, drops) in at_start {
        drop_all(&mut editor, drops.into_iter(), Position::End(block));
    }
    assert!(
        before.is_empty() && after.is_empty(),
        "every instruction a drop stands by is walked, and {before:?} and {after:?} are left"
    );
    let changes = editor.finish();
    am.invalidate(body, &changes);
}

/// A block parameter that does not copy and that nothing reads has no
/// reason to exist: it goes with its argument on every edge, and each
/// argument then ends on its edge unless read beyond (D74). Removing one can
/// leave the parameter that fed it unread, so this repeats until none is
/// left; each round removes at least one parameter.
fn remove_dead_block_params(module: &mut Module, body: BodyId, am: &mut AnalysisManager<'_>) {
    let mut no_facts = NoFacts;
    let mut editor = Editor::<Cfg>::new(module, body, &mut no_facts);
    loop {
        let (of, types) = (editor.body(), editor.types());
        let dead: Vec<ValueId> = of
            .blocks()
            .flat_map(|(_, block)| block.params().iter().copied())
            .filter(|param| of.use_count(*param) == 0 && !of.is_token(*param) && !copies(types, of.ty_of(*param)))
            .collect();
        let mut removed = false;
        for param in dead {
            match editor.remove_param(param) {
                Ok(()) => removed = true,
                Err(CannotRemoveParam::StillUsed | CannotRemoveParam::Supplied) => {}
            }
        }
        if !removed {
            break;
        }
    }
    let changes = editor.finish();
    am.invalidate(body, &changes);
}

fn drop_all(editor: &mut Editor<'_, Drops>, drops: impl Iterator<Item = PlannedDrop>, at: Position) {
    for PlannedDrop { ended, span } in drops {
        let placed = match &ended {
            Ended::Value(value) => editor.insert_drop(*value, at, span),
            Ended::Content(place) => editor.drop_content(*place, at, span),
            Ended::Part { of, unit } => editor.drop_part(*of, unit, at, span),
        };
        placed.unwrap_or_else(|refused| panic!("the plan drops {ended:?} at {at:?}, where it is still read: {refused:?}"));
    }
}

/// What the plan reads, for one body as the step found it.
struct PlanFacts<'f> {
    body: &'f Body,
    types: &'f Types<Concrete>,
    liveness: &'f Liveness,
    content: &'f ContentLiveness,
    loans: &'f Loans,
}

impl PlanFacts<'_> {
    /// Cost: O(the body's operands · the uses of each), O((values +
    /// places + units) · edges) for the edges, and each cut value's units'
    /// liveness.
    fn plan(&self) -> DropPlan {
        let body = self.body;
        let storages = self.storage_params();
        let cuts = self.cuts(&storages);
        let mut plan = DropPlan::default();
        let mut walked: FxHashSet<WalkedEdge> = FxHashSet::default();
        for (block, _) in body.blocks() {
            self.values_in_block(&storages, &cuts, block, &mut plan);
            self.places_in_block(&cuts, block, &mut plan);
            self.edges_of(&storages, &cuts, block, &mut plan);
            for (value, units) in &cuts.values {
                for unit in units {
                    self.value_unit_in_block(*value, unit, block, &mut plan);
                    self.value_unit_on_edges(*value, unit, block, &mut plan);
                }
            }
            for (place, units) in &cuts.places {
                for unit in units {
                    self.place_unit_in_block(*place, unit, block, &mut plan, &mut walked);
                    self.place_unit_on_edges(*place, unit, block, &mut plan, &mut walked);
                }
            }
        }
        self.context_overwrites(&mut plan, &mut walked);
        plan
    }

    /// The parameters the body names as a storage, which are places.
    ///
    /// # Panics
    /// One of them is also passed as a value.
    /// Whether the body names `param` as a storage.
    fn names_storage(&self, param: ValueId) -> bool {
        self.body.users(param).any(|site| match site {
            Site::Inst(inst) => {
                matches!(place_of(self.body.inst(inst).op()), Some((RefTarget::Param(named), _)) if *named == param)
            }
            Site::Term(_) => false,
        })
    }

    fn storage_params(&self) -> FxHashSet<ValueId> {
        let body = self.body;
        let mut storages = FxHashSet::default();
        for (_, record) in body.blocks() {
            for inst in record.insts() {
                if let Some((RefTarget::Param(param), _)) = place_of(inst.op()) {
                    storages.insert(*param);
                }
            }
        }
        for param in &storages {
            let as_value = body.users(*param).any(|site| match site {
                Site::Inst(inst) => {
                    let inst = body.inst(inst);
                    inst_takings(body, self.types, inst).iter().any(|operand| operand.value == *param && !operand.names_place)
                }
                Site::Term(_) => true,
            });
            assert!(
                !as_value,
                "the parameter {param:?} is named both as a storage and as a value; a by-value parameter lives in \
                 one form, its storage where anything lends or writes it and a value otherwise, and two are lower's \
                 defect (D74)"
            );
        }
        storages
    }

    /// Whether the program holds `value` as a value and must end it: it
    /// does not copy, and it is neither an order token, nor a slot or a
    /// parameter the body names as a storage, nor a capture, which the
    /// closure holds and a call of it only reads.
    fn holds(&self, storages: &FxHashSet<ValueId>, value: ValueId) -> bool {
        let body = self.body;
        let holdable = match body.definition(value) {
            Definition::Param(_) => !storages.contains(&value),
            Definition::BlockParam(..) | Definition::Inst(..) => true,
            Definition::Capture(_) | Definition::OrderParam | Definition::Slot(_) => false,
        };
        holdable && !body.is_token(value) && !copies(self.types, body.ty_of(value))
    }

    /// The drops of values whose liveness ends inside `block`: those it
    /// defines or reads that are neither live after it nor named by its
    /// terminator.
    fn values_in_block(&self, storages: &FxHashSet<ValueId>, cuts: &Cuts, block: BlockId, plan: &mut DropPlan) {
        let (body, record) = (self.body, self.body.block(block));
        let mut candidates: BTreeSet<ValueId> = record.params().iter().copied().collect();
        if block == body.entry() {
            candidates.extend(body.params().iter().map(|(_, param)| *param));
        }
        for inst in record.insts() {
            candidates.extend(inst.result());
            candidates.extend(inst.operands());
        }
        let named_by_terminator = record.terminator().operands();
        for value in candidates {
            if !self.holds(storages, value)
                || cuts.values.contains_key(&value)
                || self.liveness.live_at_end(block).contains(&value)
                || named_by_terminator.contains(&value)
            {
                continue;
            }
            let last_use = body
                .users(value)
                .filter_map(|site| match site {
                    Site::Inst(at) if body.block_of(at) == block => Some(at),
                    Site::Inst(_) | Site::Term(_) => None,
                })
                .reduce(|last, other| if body.comes_before(last, other) { other } else { last });
            let (site, span) = match last_use {
                Some(at) => {
                    let inst = body.inst(at);
                    let moved = inst_takings(body, self.types, inst)
                        .iter()
                        .any(|operand| operand.value == value && operand.taken == Taken::Moved);
                    if moved {
                        continue;
                    }
                    (Site::Inst(at), inst.span())
                }
                None => match body.definition(value) {
                    Definition::Inst(at) => (Site::Inst(at), body.inst(at).span()),
                    Definition::Param(_) => {
                        let drop = PlannedDrop { ended: Ended::Value(value), span: body.binder_span(value) };
                        plan.at_start.entry(block).or_default().push(drop);
                        continue;
                    }
                    definition => panic!(
                        "{value:?}, defined at {definition:?}, is dead where {block:?} starts; a block parameter \
                         nothing reads was removed before the plan, unless a loop's terminator fills it (D74)"
                    ),
                },
            };
            plan.after.entry(site).or_default().push(PlannedDrop { ended: Ended::Value(value), span });
        }
    }

    /// Whether the body owns `place`'s content and drops it where its
    /// liveness ends: a slot that is not a context's, or a parameter named
    /// as a storage, whose type does not copy. A context's slot is
    /// `context_overwrites`'.
    fn liveness_ends_content(&self, place: ValueId) -> bool {
        let body = self.body;
        let owned = match body.definition(place) {
            Definition::Slot(_) => body.context_of_slot(place).is_none(),
            Definition::Param(_) => self.names_storage(place),
            Definition::Capture(_) | Definition::OrderParam | Definition::BlockParam(..) | Definition::Inst(..) => false,
        };
        owned && !copies(self.types, body.ty_of(place))
    }

    /// The places the site reads, by `Loans`, and the place it stores into
    /// whole, which starts a new content there.
    fn content_events(&self, site: Site, inst: &Inst) -> SmallVec<[ValueId; 2]> {
        let mut places: SmallVec<[ValueId; 2]> = self
            .loans
            .reads(site)
            .iter()
            .filter_map(|storage| match storage {
                Storage::Slot(place) | Storage::Param(place) => Some(*place),
                Storage::Outside { .. } => None,
            })
            .collect();
        if let Op::Assign { target, path, .. } = inst.op()
            && path.is_empty()
            && let Some(place) = named_place(target)
        {
            places.push(place);
        }
        places
    }

    /// The drops of place contents whose liveness ends inside `block`: after
    /// each site that reads the content or stores it where the content is
    /// not read after it, unless that site takes the whole content out.
    fn places_in_block(&self, cuts: &Cuts, block: BlockId, plan: &mut DropPlan) {
        let (body, record) = (self.body, self.body.block(block));
        for inst in record.insts() {
            let site = Site::Inst(inst.id());
            let ended: BTreeSet<ValueId> = self
                .content_events(site, inst)
                .into_iter()
                .filter(|place| self.ends_whole(cuts, *place) && !self.content.live_after(site).contains(place))
                .collect();
            for place in ended {
                let takes_all = inst_takings(body, self.types, inst)
                    .iter()
                    .any(|operand| operand.value == place && operand.names_place && operand.taken == Taken::Moved)
                    || self.moved_through(inst).is_some_and(|(moved, region)| moved == place && region.is_empty());
                if !takes_all {
                    let drop = PlannedDrop { ended: Ended::Content(place), span: inst.span() };
                    plan.after.entry(site).or_default().push(drop);
                }
            }
        }
        if block == body.entry() {
            for (_, param) in body.params() {
                if self.ends_whole(cuts, *param) && !self.content.live_at_start(block).contains(param) {
                    let drop = PlannedDrop { ended: Ended::Content(*param), span: body.binder_span(*param) };
                    plan.at_start.entry(block).or_default().push(drop);
                }
            }
        }
    }

    /// The values and place contents live out of `block` and not into one
    /// of its successors, which end on that edge, where control leaves the
    /// block: they take the span of its terminator.
    fn edges_of(&self, storages: &FxHashSet<ValueId>, cuts: &Cuts, block: BlockId, plan: &mut DropPlan) {
        let (body, record) = (self.body, self.body.block(block));
        let terminator = record.terminator();
        let span = record.terminator_span();
        let named = terminator_takings(body, self.types, terminator);
        let mut values: BTreeSet<ValueId> = self.liveness.live_at_end(block).iter().copied().collect();
        values.extend(named.iter().filter(|named| named.taken == Taken::Read).map(|named| named.value));
        values.retain(|value| self.holds(storages, *value) && !cuts.values.contains_key(value));
        for named in named.iter().filter(|named| named.taken == Taken::Moved) {
            values.remove(&named.value);
        }
        let mut places: BTreeSet<ValueId> = self.content.live_at_end(block).iter().copied().collect();
        places.extend(self.loans.reads(Site::Term(block)).iter().filter_map(|storage| storage_place(*storage)));
        places.retain(|place| self.ends_whole(cuts, *place));
        for edge in terminator.edges() {
            let moved_on_edge = |value: &ValueId| {
                edge.args.iter().any(|arg| arg == value && by_value(self.types, body.ty_of(*arg)) == Taken::Moved)
            };
            let ended_values = values
                .iter()
                .filter(|value| !moved_on_edge(value) && !self.liveness.live_at_start(edge.to).contains(*value))
                .map(|value| PlannedDrop { ended: Ended::Value(*value), span });
            let ended_places = places
                .iter()
                .filter(|place| !self.content.live_at_start(edge.to).contains(*place))
                .map(|place| PlannedDrop { ended: Ended::Content(*place), span });
            let drops: Vec<PlannedDrop> = ended_values.chain(ended_places).collect();
            if !drops.is_empty() {
                plan.on_edges.push(EdgeDrops { from: block, to: edge.to, drops });
            }
        }
    }

    /// Before each store that overwrites a context's slot, whole or in
    /// part, other than the store of its load, each unit it replaces is the
    /// body's where the slot holds it there, and is dropped (context §3,
    /// D71). Where the slot holds it on some paths only, nothing reads it
    /// between the join and the store, so its liveness ended on the edges
    /// into the join: it is dropped on each such edge where it is present
    /// (ownership §6, D74).
    fn context_overwrites(&self, plan: &mut DropPlan, walked: &mut FxHashSet<WalkedEdge>) {
        let body = self.body;
        for (block, record) in body.blocks() {
            for inst in record.insts() {
                let Some((slot, path)) = self.context_overwritten(inst) else {
                    continue;
                };
                let units = self.content.units(slot).expect("a context slot that does not copy has units");
                for unit in units.iter().filter(|unit| covers(path, unit) && !self.part_copies(slot, unit)) {
                    match self.content.presence_before(inst.id(), slot, unit) {
                        Presence::Present => {
                            let drop = PlannedDrop { ended: self.place_part(slot, unit), span: inst.span() };
                            plan.before.entry(Site::Inst(inst.id())).or_default().push(drop);
                        }
                        Presence::Absent | Presence::Unreached => {}
                        Presence::Maybe => {
                            assert!(
                                self.untouched(slot, unit, block, Some(inst.id())),
                                "the context slot {slot:?}'s part {unit:?} is read or taken on some paths only before \
                                 {:?} of {block:?} overwrites it; check refuses a read of content moved out on some \
                                 path, and insert_drops runs after check",
                                inst.id()
                            );
                            self.drop_where_present(slot, unit, block, true, plan, walked);
                        }
                    }
                }
            }
        }
    }

    /// The context slot `inst` stores into by name, whole or in part, and
    /// the path, other than by storing its own load.
    fn context_overwritten<'i>(&self, inst: &'i Inst) -> Option<(ValueId, &'i [PathSeg])> {
        let Op::Assign { target: RefTarget::Var(slot), path, value, .. } = inst.op() else {
            return None;
        };
        let body = self.body;
        let is_context = body.context_of_slot(*slot).is_some() && !copies(self.types, body.ty_of(*slot));
        let stores_its_load = match body.definition(*value) {
            Definition::Inst(loaded) => matches!(
                body.inst(loaded).op(),
                Op::Fetch { context } if Some(*context) == body.context_of_slot(*slot)
            ),
            _ => false,
        };
        (is_context && !(path.is_empty() && stores_its_load)).then_some((*slot, path.as_slice()))
    }

    /// Whether nothing in `block` before the instruction `upto`, or before
    /// the block's end where that is none, reads, takes, lends or stores
    /// `unit` of `place`.
    fn untouched(&self, place: ValueId, unit: &[PathSeg], block: BlockId, upto: Option<InstId>) -> bool {
        self.body.block(block).insts().take_while(|inst| Some(inst.id()) != upto).all(|inst| {
            let site = Site::Inst(inst.id());
            match place_of(inst.op()) {
                Some((RefTarget::Var(named) | RefTarget::Param(named), path)) if *named == place => !overlap(path, unit),
                _ => !self.loans.reads(site).iter().any(|storage| storage_place(*storage) == Some(place)),
            }
        })
    }

    /// `unit` of `place` is present on some paths only where `block`
    /// starts: it is dropped on each edge into the join where it is
    /// present, walking back through blocks where it is present on some
    /// paths only. `assert_untouched` asserts, for a context's slot, that
    /// nothing touches it on the way. An edge already given this unit's
    /// drop is not given it again.
    fn drop_where_present(
        &self,
        place: ValueId,
        unit: &[PathSeg],
        block: BlockId,
        assert_untouched: bool,
        plan: &mut DropPlan,
        walked: &mut FxHashSet<WalkedEdge>,
    ) {
        let body = self.body;
        let preds = body.predecessors();
        let mut visited: FxHashSet<BlockId> = FxHashSet::from_iter([block]);
        let mut pending = vec![block];
        while let Some(join) = pending.pop() {
            for pred in &preds[&join] {
                match self.content.presence_at_end(*pred, place, unit) {
                    Presence::Present => {
                        if walked.insert((*pred, join, place, unit.to_vec())) {
                            let drop = PlannedDrop { ended: self.place_part(place, unit), span: body.block(*pred).terminator_span() };
                            plan.on_edges.push(EdgeDrops { from: *pred, to: join, drops: vec![drop] });
                        }
                    }
                    Presence::Absent | Presence::Unreached => {}
                    Presence::Maybe => {
                        assert!(
                            !assert_untouched || self.untouched(place, unit, *pred, None),
                            "the context slot {place:?}'s part {unit:?} is read or taken on some paths only in \
                             {pred:?}, before an overwrite; check refuses such a read, and insert_drops runs after check"
                        );
                        if visited.insert(*pred) {
                            pending.push(*pred);
                        }
                    }
                }
            }
        }
    }

    /// What ends `unit` of `place`: its whole content where the unit is the
    /// whole, and the part otherwise.
    fn place_part(&self, place: ValueId, unit: &[PathSeg]) -> Ended {
        match unit.is_empty() {
            true => Ended::Content(place),
            false => Ended::Part { of: PartOf::Place(place), unit: unit.to_vec() },
        }
    }

    fn takes_a_copy(&self, inst: &Inst) -> bool {
        inst.result().is_some_and(|taken| copies(self.types, self.body.ty_of(taken)))
    }

    fn part_copies(&self, holder: ValueId, unit: &[PathSeg]) -> bool {
        copies(self.types, parts::part_ty(self.types, self.body.ty_of(holder), unit))
    }

    /// Whether the body drops `place`'s content whole where its liveness
    /// ends: it owns the content, and no part of it is cut.
    /// The slot or parameter, and the region in it, that a `Take` through a
    /// reference moves out of, where the takings move what it lends and it
    /// reaches that place alone (`Loans::through_one`).
    fn moved_through(&self, inst: &Inst) -> Option<(ValueId, &[PathSeg])> {
        if !matches!(inst.op(), Op::Take { target: RefTarget::Through(_), .. }) {
            return None;
        }
        let place = self.loans.through_one(Site::Inst(inst.id()), Reach::Move)?;
        storage_place(place.storage).map(|value| (value, parts::owned_region(&place.path)))
    }

    fn ends_whole(&self, cuts: &Cuts, place: ValueId) -> bool {
        self.liveness_ends_content(place) && !cuts.places.contains_key(&place)
    }
}

// -- Parts of a cut holder (ownership §6) ------------------------------------

/// An edge a walk back from a join gave a unit's drop: its source, its
/// target, the place, and the unit.
type WalkedEdge = (BlockId, BlockId, ValueId, Path);

/// The holders a body cuts, each with its units that do not copy: a value,
/// with where each unit is live, and a place the body owns.
struct Cuts {
    values: FxHashMap<ValueId, Vec<ValueUnit>>,
    places: FxHashMap<ValueId, Vec<Path>>,
}

/// One unit of a cut value and the blocks it is live at the start and at
/// the end of: SSA liveness over the uses that touch it, an edge's argument
/// live at its source's end.
struct ValueUnit {
    path: Path,
    at_start: FxHashSet<BlockId>,
    at_end: FxHashSet<BlockId>,
}

/// How an instruction or a terminator uses one unit of a cut value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum UnitUse {
    Read,
    Moved,
}

/// The path of the part an extraction of `value` names, where `op` is one.
fn extracted_path(op: &Op, value: ValueId) -> Option<Path> {
    let fields = |field: &acvus_utils::Astr, rest: &[acvus_utils::Astr]| {
        std::iter::once(field).chain(rest).map(|name| PathSeg::Field(*name)).collect()
    };
    match op {
        Op::FieldGet { object, field, rest } if *object == value => Some(fields(field, rest)),
        Op::ObjectGet { object, key } if *object == value => Some(vec![PathSeg::Field(*key)]),
        Op::TupleIndex { tuple: whole, index } | Op::ArrayIndex { array: whole, index } if *whole == value => {
            Some(vec![PathSeg::Index(*index)])
        }
        _ => None,
    }
}

/// The path of the part a `FieldSet` of `value` replaces, where `op` is one.
fn replaced_path(op: &Op, value: ValueId) -> Option<Path> {
    match op {
        Op::FieldSet { object, field, rest, .. } if *object == value => {
            Some(std::iter::once(field).chain(rest).map(|name| PathSeg::Field(*name)).collect())
        }
        _ => None,
    }
}

impl PlanFacts<'_> {
    /// Every holder the body cuts. A value is cut where an operand takes
    /// it as `PartMoved`: an extraction moves a part out of it, or a
    /// `FieldSet` replaces one. A place the body owns is cut where
    /// `ContentLiveness` finds more than its one whole unit.
    fn cuts(&self, storages: &FxHashSet<ValueId>) -> Cuts {
        let body = self.body;
        let mut value_cuts: FxHashMap<ValueId, Vec<Path>> = FxHashMap::default();
        for (_, record) in body.blocks() {
            for inst in record.insts() {
                for operand in inst_takings(body, self.types, inst) {
                    if operand.taken != Taken::PartMoved || operand.names_place || !self.holds(storages, operand.value) {
                        continue;
                    }
                    let path = extracted_path(inst.op(), operand.value)
                        .or_else(|| replaced_path(inst.op(), operand.value))
                        .unwrap_or_else(|| panic!("a value is taken as a part only by an extraction or a FieldSet, not {:?}", inst.op()));
                    let cuts = value_cuts.entry(operand.value).or_default();
                    if !cuts.contains(&path) {
                        cuts.push(path);
                    }
                }
            }
        }
        let preds = body.predecessors();
        let values = value_cuts
            .into_iter()
            .map(|(value, cuts)| {
                let units = parts::units(self.types, body.ty_of(value), &cuts)
                    .into_iter()
                    .filter(|unit| !self.part_copies(value, unit))
                    .map(|unit| self.value_unit(&preds, value, unit))
                    .collect();
                (value, units)
            })
            .collect();
        let places = body
            .slots()
            .iter()
            .chain(body.params().iter().map(|(_, param)| param))
            .filter(|place| self.liveness_ends_content(**place))
            .filter_map(|place| {
                let units = self.content.units(*place)?;
                let cut = !matches!(units, [whole] if whole.is_empty());
                cut.then(|| (*place, units.iter().filter(|unit| !self.part_copies(*place, unit)).cloned().collect()))
            })
            .collect();
        Cuts { values, places }
    }

    /// `path` of `value` and where it is live. Cost: O(blocks + edges).
    fn value_unit(&self, preds: &FxHashMap<BlockId, SmallVec<[BlockId; 2]>>, value: ValueId, path: Path) -> ValueUnit {
        let body = self.body;
        let defined = self.defining_block(value);
        let mut unit = ValueUnit { path, at_start: FxHashSet::default(), at_end: FxHashSet::default() };
        let mut pending: Vec<BlockId> = Vec::new();
        for site in body.users(value) {
            let block = match site {
                Site::Inst(inst) => {
                    if self.unit_use(body.inst(inst), value, &unit.path).is_none() {
                        continue;
                    }
                    body.block_of(inst)
                }
                Site::Term(block) => {
                    if body.block(block).terminator().edges().iter().any(|edge| edge.args.contains(&value)) {
                        unit.at_end.insert(block);
                    }
                    block
                }
            };
            if block != defined && unit.at_start.insert(block) {
                pending.push(block);
            }
        }
        while let Some(block) = pending.pop() {
            for pred in &preds[&block] {
                unit.at_end.insert(*pred);
                if *pred != defined && unit.at_start.insert(*pred) {
                    pending.push(*pred);
                }
            }
        }
        unit
    }

    /// The block whose start or instructions define `value`.
    ///
    /// # Panics
    /// `value` is not one the body holds: a capture, a slot or the order
    /// parameter.
    fn defining_block(&self, value: ValueId) -> BlockId {
        match self.body.definition(value) {
            Definition::Inst(inst) => self.body.block_of(inst),
            Definition::BlockParam(block, _) => block,
            Definition::Param(_) => self.body.entry(),
            definition => panic!("{value:?}, defined at {definition:?}, is not a value the body holds"),
        }
    }

    /// How `inst` uses `unit` of the cut value `value`, if at all. An
    /// extraction reads an overlapping unit where what it takes copies, and
    /// otherwise moves the units under it; a `FieldSet` moves every unit
    /// but those under the part it replaces; any other use uses the value
    /// whole.
    ///
    /// # Panics
    /// An extraction that moves a part overlaps `unit` without covering it:
    /// units refine every cut.
    fn unit_use(&self, inst: &Inst, value: ValueId, unit: &[PathSeg]) -> Option<UnitUse> {
        if let Some(path) = extracted_path(inst.op(), value) {
            let result_copies = inst.result().is_some_and(|result| copies(self.types, self.body.ty_of(result)));
            return match (result_copies, overlap(&path, unit), covers(&path, unit)) {
                (_, false, _) => None,
                (true, true, _) => Some(UnitUse::Read),
                (false, true, true) => Some(UnitUse::Moved),
                (false, true, false) => panic!("{path:?} of {value:?} moves out part of the unit {unit:?}; units refine every cut"),
            };
        }
        if let Some(path) = replaced_path(inst.op(), value) {
            return (!covers(&path, unit)).then_some(UnitUse::Moved);
        }
        let takings: SmallVec<[Taken; 2]> = inst_takings(self.body, self.types, inst)
            .into_iter()
            .filter(|operand| operand.value == value)
            .map(|operand| operand.taken)
            .collect();
        match takings.is_empty() {
            true => None,
            false if takings.contains(&Taken::Moved) => Some(UnitUse::Moved),
            false => Some(UnitUse::Read),
        }
    }

    /// The drop of `unit` of `value` where its liveness ends inside
    /// `block`: after the last instruction that touches it, unless that
    /// moves it; where nothing in the block touches it, after its
    /// definition, or at the block's start for a parameter.
    fn value_unit_in_block(&self, value: ValueId, unit: &ValueUnit, block: BlockId, plan: &mut DropPlan) {
        let (body, record) = (self.body, self.body.block(block));
        let defined_here = self.defining_block(value) == block;
        if !(defined_here || unit.at_start.contains(&block))
            || unit.at_end.contains(&block)
            || record.terminator().operands().contains(&value)
        {
            return;
        }
        let ended = Ended::Part { of: PartOf::Value(value), unit: unit.path.clone() };
        let last = record.insts().rev().find_map(|inst| self.unit_use(inst, value, &unit.path).map(|used| (inst, used)));
        match last {
            Some((_, UnitUse::Moved)) => {}
            Some((inst, UnitUse::Read)) => {
                plan.after.entry(Site::Inst(inst.id())).or_default().push(PlannedDrop { ended, span: inst.span() });
            }
            None => match body.definition(value) {
                Definition::Inst(at) => {
                    let span = body.inst(at).span();
                    plan.after.entry(Site::Inst(at)).or_default().push(PlannedDrop { ended, span });
                }
                Definition::Param(_) => {
                    let drop = PlannedDrop { ended, span: body.binder_span(value) };
                    plan.at_start.entry(block).or_default().push(drop);
                }
                Definition::BlockParam(..) => {
                    let drop = PlannedDrop { ended, span: record.terminator_span() };
                    plan.at_start.entry(block).or_default().push(drop);
                }
                definition => panic!("{value:?}, defined at {definition:?}, is not a value the body holds"),
            },
        }
    }

    /// The drops of `unit` of `value` on the edges out of `block` it is
    /// live out of, or read by the terminator, and not live into, unless
    /// the edge passes the value on or the terminator moves it.
    fn value_unit_on_edges(&self, value: ValueId, unit: &ValueUnit, block: BlockId, plan: &mut DropPlan) {
        let (body, record) = (self.body, self.body.block(block));
        let terminator = record.terminator();
        let named: SmallVec<[Named; 2]> =
            terminator_takings(body, self.types, terminator).into_iter().filter(|named| named.value == value).collect();
        if named.iter().any(|named| named.taken == Taken::Moved) {
            return;
        }
        let read_by_terminator = !named.is_empty();
        if !(unit.at_end.contains(&block) || read_by_terminator) {
            return;
        }
        let span = record.terminator_span();
        for edge in terminator.edges() {
            let passed = edge.args.iter().any(|arg| *arg == value && by_value(self.types, body.ty_of(*arg)) == Taken::Moved);
            if passed || unit.at_start.contains(&edge.to) {
                continue;
            }
            let drop = PlannedDrop { ended: Ended::Part { of: PartOf::Value(value), unit: unit.path.clone() }, span };
            plan.on_edges.push(EdgeDrops { from: block, to: edge.to, drops: vec![drop] });
        }
    }

    /// The drop of `unit` of the cut place `place` after each site in
    /// `block` that reads or stores it where it is not read after, unless
    /// that site moves it out: where it is present, or on the edges into the
    /// join where it is present on some paths only. A by-value parameter's
    /// unit not read from the entry's start is dropped there.
    fn place_unit_in_block(
        &self,
        place: ValueId,
        unit: &[PathSeg],
        block: BlockId,
        plan: &mut DropPlan,
        walked: &mut FxHashSet<WalkedEdge>,
    ) {
        let (body, record) = (self.body, self.body.block(block));
        let ended = || Ended::Part { of: PartOf::Place(place), unit: unit.to_vec() };
        for inst in record.insts() {
            let site = Site::Inst(inst.id());
            let event = match place_of(inst.op()) {
                Some((RefTarget::Var(named) | RefTarget::Param(named), path)) if *named == place => match inst.op() {
                    Op::Take { .. } if covers(parts::owned_region(path), unit) && !self.takes_a_copy(inst) => {
                        Some(PlaceEvent::Moved)
                    }
                    Op::Take { .. } if overlap(path, unit) => Some(PlaceEvent::Read),
                    Op::Assign { .. } if covers(parts::owned_region(path), unit) => Some(PlaceEvent::Stored),
                    _ => None,
                },
                _ => match self.moved_through(inst) {
                    Some((moved, region)) if moved == place && covers(region, unit) => Some(PlaceEvent::Moved),
                    _ => self.loans.reads(site).iter().any(|storage| storage_place(*storage) == Some(place)).then_some(PlaceEvent::Read),
                },
            };
            let Some(event) = event else {
                continue;
            };
            if self.content.part_live_after(site, place, unit) {
                continue;
            }
            let presence = match event {
                PlaceEvent::Moved => continue,
                PlaceEvent::Stored => Presence::Present,
                PlaceEvent::Read => self.content.presence_before(inst.id(), place, unit),
            };
            match presence {
                Presence::Present => plan.after.entry(site).or_default().push(PlannedDrop { ended: ended(), span: inst.span() }),
                Presence::Absent | Presence::Unreached => {}
                Presence::Maybe => self.drop_where_present(place, unit, block, false, plan, walked),
            }
        }
        let is_param = matches!(body.definition(place), Definition::Param(_));
        if is_param && block == body.entry() && !self.content.part_live_at_start(block, place, unit) {
            let drop = PlannedDrop { ended: ended(), span: body.binder_span(place) };
            plan.at_start.entry(block).or_default().push(drop);
        }
    }

    /// The drops of `unit` of the cut place `place` on the edges out of
    /// `block` it is live out of, or read by the terminator, and not live
    /// into: where it is present at the block's end, or on the edges into
    /// the join where it is present on some paths only.
    fn place_unit_on_edges(
        &self,
        place: ValueId,
        unit: &[PathSeg],
        block: BlockId,
        plan: &mut DropPlan,
        walked: &mut FxHashSet<WalkedEdge>,
    ) {
        let record = self.body.block(block);
        let read_by_terminator =
            self.loans.reads(Site::Term(block)).iter().any(|storage| storage_place(*storage) == Some(place));
        if !(self.content.part_live_at_end(block, place, unit) || read_by_terminator) {
            return;
        }
        let span = record.terminator_span();
        for edge in record.terminator().edges() {
            if self.content.part_live_at_start(edge.to, place, unit) {
                continue;
            }
            match self.content.presence_at_end(block, place, unit) {
                Presence::Present => {
                    let drop = PlannedDrop { ended: Ended::Part { of: PartOf::Place(place), unit: unit.to_vec() }, span };
                    plan.on_edges.push(EdgeDrops { from: block, to: edge.to, drops: vec![drop] });
                }
                Presence::Absent | Presence::Unreached => {}
                Presence::Maybe => self.drop_where_present(place, unit, block, false, plan, walked),
            }
        }
    }
}

/// What a site does to one unit of a cut place.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PlaceEvent {
    Read,
    Stored,
    Moved,
}

fn storage_place(storage: Storage) -> Option<ValueId> {
    match storage {
        Storage::Slot(place) | Storage::Param(place) => Some(place),
        Storage::Outside { .. } => None,
    }
}

fn place_of(op: &Op) -> Option<(&RefTarget, &[PathSeg])> {
    match op {
        Op::Take { target, path, .. } | Op::Assign { target, path, .. } | Op::Ref { target, path, .. } => {
            Some((target, path))
        }
        _ => None,
    }
}
