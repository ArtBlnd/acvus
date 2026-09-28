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
    CannotDischarge, Cfg, Clones, Discharge, Drops, Editor, InstRef, NoFacts, Position, Remove, Value,
};
use crate::ir::cfg::{BlockId, Body, BodyId, Changes, Definition, Edge, Inst, Module, Op, Site, Terminator};
use crate::ir::{Callee, ForSource, PathSeg, RefTarget, ValueId};
use crate::passes::analysis::{
    AnalysisManager, ContentLiveness, DenseAnalysis, Direction, Liveness, Loans, Storage, solve_dense,
};
use crate::passes::domain::{JoinSemiLattice, Widen};
use crate::passes::edit_facts::EditFacts;
use crate::ty::{Calls, Concrete, Mutability, Ty, TyKind, Types, copies};

#[cfg(test)]
mod tests;

/// Every body: a `String` a use moves while the same content is read again
/// on some path after it is cloned just before that use, with the use's
/// span, and the use takes the clone (ownership §3). A place's content read
/// by a `Take`, a field of it included, becomes a clone of a shared `Ref`
/// of the place; a value's use takes a `StringClone` of the value. Of the
/// operands of one site that move one value, all but the last are clones,
/// and the last is one too where the value is read after the site. A clone
/// before a terminator takes the terminator's span.
///
/// Cost per body: `ContentLiveness`' and `Liveness`', one `clone_read` per
/// place read, and one insertion and one `replace_operand` per value use
/// cloned, each an edit and, in scope `Value`, a refresh of `EditFacts`.
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
/// Cost per body: O(instructions) discharges and removals, each a refresh
/// of `EditFacts` in scope `Discharge`.
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
///   follows ends the content it stores.
/// - A context's slot: its content is the page's until the body overwrites
///   it; the old content of an overwrite is the body's and is dropped just
///   before the store, and the body never drops the content at its end
///   (context §3, D71).
///
/// A capture is the closure's, an order token ends in no drop, and a
/// parameter the body names as a storage is a place, not a value.
///
/// Cost per body: `Liveness`', `ContentLiveness`', a forward pass over the
/// context slots, O(values · blocks) for the plan, one `split_edge` per edge
/// that ends something, and one `insert_drop` or `drop_content` per drop,
/// each a refresh of `EditFacts`.
///
/// # Panics
/// A part that does not copy moves out of a whole that does not copy
/// (PARKED P10); a parameter is dead where the body or its block starts,
/// where no edge of its own carries the drop; a context slot holds its
/// content on some paths only at an overwrite; a parameter is both a value
/// and a storage. Each message states the question.
pub fn insert_drops(module: &mut Module, am: &mut AnalysisManager<'_>) {
    for body in module.body_ids() {
        insert_drops_in(module, body, am);
    }
}

// -- Ownership of operands ---------------------------------------------------

/// How an operation takes one of its operands (ownership §1-2, §5).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Taken {
    /// The value goes to another holder, or ends here.
    Moved,
    /// The value is read, and its holder keeps it.
    Read,
    /// A part that does not copy leaves a whole that does not copy, or a
    /// part is replaced in one: where the rest ends is `parked_part_moved`'s.
    PartMoved,
}

/// PARKED(P10): the ledger does not say how the rest of a value that does
/// not copy ends once a part of it that does not copy has moved out, or
/// where a part a `FieldSet` or a store into part of a place replaces ends.
/// Until it does, drop insertion stops here with the question.
///
/// # Panics
/// Always.
fn parked_part_moved(site: Site, whole: ValueId) -> ! {
    panic!(
        "PARKED(P10): {site:?} moves a part that does not copy out of {whole:?}, which does not copy, or replaces \
         such a part in it; how the rest of {whole:?} ends is not settled"
    )
}

/// One operand of an instruction and how it is taken. `names_place` marks
/// an operand that names a storage, as a `Take`'s or an `Assign`'s target
/// does, rather than passing a value.
#[derive(Debug, Clone, Copy)]
struct Operand {
    value: ValueId,
    taken: Taken,
    names_place: bool,
}

/// How a by-value position takes `ty`: a `&mut` is reborrowed by each use
/// (ownership §5), and any other value is moved.
fn by_value(types: &Types<Concrete>, ty: Ty) -> Taken {
    match types.kind(ty) {
        TyKind::Ref(Mutability::Mut, _) => Taken::Read,
        _ => Taken::Moved,
    }
}

/// Every operand of `inst` but the order token, in the order
/// `Inst::operands` gives them, with how each is taken. Cost: O(its
/// operands).
fn inst_operands(body: &Body, types: &Types<Concrete>, inst: &Inst) -> SmallVec<[Operand; 4]> {
    let ty = |value: ValueId| body.ty_of(value);
    let copying = |value: ValueId| copies(types, ty(value));
    let taken = |value: ValueId, taken: Taken| Operand { value, taken, names_place: false };
    let passed = |value: ValueId| taken(value, by_value(types, ty(value)));
    let read = |value: ValueId| taken(value, Taken::Read);
    let result_copies = || inst.result().is_some_and(copying);
    let extracted = |src: ValueId| match copying(src) || result_copies() {
        true => read(src),
        false => taken(src, Taken::PartMoved),
    };
    let place = |target: &RefTarget, taken: Taken| match target {
        RefTarget::Var(named) | RefTarget::Param(named) => Operand { value: *named, taken, names_place: true },
        RefTarget::Through(reference) => read(*reference),
    };
    let callee = |callee: &Callee| match callee {
        Callee::Indirect(function) => match types.kind(ty(*function)) {
            TyKind::Fn { calls: Calls::Once, .. } => Some(taken(*function, Taken::Moved)),
            TyKind::Fn { calls: Calls::Many, .. } => Some(read(*function)),
            other => panic!("an indirect callee {function:?} is a function value, not {other:?}"),
        },
        Callee::Direct(_) | Callee::Extern { .. } => None,
    };
    let mut operands: SmallVec<[Operand; 4]> = SmallVec::new();
    match inst.op() {
        Op::Const { .. }
        | Op::ConstStr { .. }
        | Op::Fetch { .. }
        | Op::LoadFunction { .. }
        | Op::ArrayBegin { .. }
        | Op::Undef => {}
        Op::StringConcat { parts } => operands.extend(parts.iter().map(|part| passed(*part))),
        Op::StringAppend { target, part } => operands.extend([read(*target), passed(*part)]),
        Op::StringEq { a, b } | Op::StructuralEq { a, b, .. } => operands.extend([read(*a), read(*b)]),
        Op::StringClone { src } | Op::StructuralClone { src, .. } => operands.push(read(*src)),
        Op::Ref { target, .. } => operands.push(place(target, Taken::Read)),
        Op::Take { target, path, .. } => {
            let whole = match target {
                RefTarget::Var(named) | RefTarget::Param(named) => copying(*named),
                RefTarget::Through(_) => true,
            };
            let taking = match (path.is_empty(), whole || result_copies()) {
                (_, true) => Taken::Read,
                (true, false) => Taken::Moved,
                (false, false) => Taken::PartMoved,
            };
            operands.push(place(target, taking));
        }
        Op::Assign { target, path, value, .. } => {
            let replaces_a_part = !path.is_empty() && !copying(*value);
            let taking = if replaces_a_part { Taken::PartMoved } else { Taken::Read };
            operands.extend([place(target, taking), passed(*value)]);
        }
        Op::AsSlice { container: lent, .. } | Op::ArrayView { src: lent, .. } => operands.push(read(*lent)),
        Op::Index { slice, index, .. } => operands.extend([read(*slice), read(*index)]),
        Op::IndexSet { slice, index, value, .. } => operands.extend([read(*slice), read(*index), passed(*value)]),
        Op::Commit { value, .. } => operands.push(passed(*value)),
        Op::FieldGet { object: src, .. }
        | Op::ObjectGet { object: src, .. }
        | Op::TupleIndex { tuple: src, .. }
        | Op::ArrayIndex { array: src, .. }
        | Op::UnwrapVariant { src } => operands.push(extracted(*src)),
        Op::FieldSet { object, value, .. } => {
            let object = if copying(*object) { read(*object) } else { taken(*object, Taken::PartMoved) };
            operands.extend([object, passed(*value)]);
        }
        Op::BinOp { left, right, .. } | Op::Check { left, right, .. } => operands.extend([read(*left), read(*right)]),
        Op::UnaryOp { operand, .. } => operands.push(read(*operand)),
        Op::Cast { src, .. } => operands.push(read(*src)),
        Op::CheckSteps { from, step, count } => operands.extend([read(*from), read(*step), read(*count)]),
        Op::FunctionCall { callee: called, args, .. } | Op::Spawn { callee: called, args, .. } => {
            operands.extend(callee(called));
            operands.extend(args.iter().map(|arg| passed(*arg)));
        }
        Op::TakeEffect { src } => operands.push(taken(*src, Taken::Moved)),
        Op::Merge { orders } => operands.extend(orders.iter().map(|order| read(*order))),
        Op::ArrayPush { array, value } => operands.extend([passed(*array), passed(*value)]),
        Op::MakeObject { fields } => operands.extend(fields.iter().map(|(_, field)| passed(*field))),
        Op::MakeTuple { elements } => operands.extend(elements.iter().map(|element| passed(*element))),
        Op::MakeVariant { payload, .. } => operands.extend(payload.iter().map(|payload| passed(*payload))),
        Op::MakeClosure { captures, .. } => operands.extend(captures.iter().map(|capture| passed(*capture))),
        Op::TestLiteral { src, .. } | Op::TestObjectKey { src, .. } | Op::TestVariant { src, .. } => {
            operands.push(read(*src))
        }
        Op::Drop { src } => operands.push(taken(*src, Taken::Moved)),
    }
    assert!(
        operands.iter().map(|operand| operand.value).eq(inst.operands().into_iter().take(operands.len())),
        "the takings of {:?} follow its operands in order",
        inst.op()
    );
    operands
}

/// A value a terminator names apart from its edges' arguments, and how it
/// is taken.
#[derive(Debug, Clone, Copy)]
struct Named {
    value: ValueId,
    taken: Taken,
}

/// What a terminator does with the values it names apart from its edges'
/// arguments: a condition or a tag is read, a `For` moves an array it
/// traverses and reads any other source, and a `Return` passes its value
/// out. An edge's argument is passed by value into its target's parameter.
fn terminator_operands(body: &Body, types: &Types<Concrete>, terminator: &Terminator) -> SmallVec<[Named; 2]> {
    let read = |value: ValueId| Named { value, taken: Taken::Read };
    match terminator {
        Terminator::Jump { .. } | Terminator::Diverge => SmallVec::new(),
        Terminator::JumpIf { cond, .. } | Terminator::Diamond { cond, .. } | Terminator::While { cond, .. } => {
            smallvec::smallvec![read(*cond)]
        }
        Terminator::Switch { tag, .. } => smallvec::smallvec![read(*tag)],
        Terminator::For { source, .. } => match source {
            ForSource::Array(array) => smallvec::smallvec![Named { value: *array, taken: Taken::Moved }],
            ForSource::Slice(slice) | ForSource::SliceMut(slice) => smallvec::smallvec![read(*slice)],
            ForSource::Range { at, hi } => smallvec::smallvec![read(*at), read(*hi)],
        },
        Terminator::Return { value, .. } => {
            smallvec::smallvec![Named { value: *value, taken: by_value(types, body.ty_of(*value)) }]
        }
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
        Terminator::Return { .. } | Terminator::Diverge => SmallVec::new(),
    }
}

fn is_string(types: &Types<Concrete>, ty: Ty) -> bool {
    matches!(types.kind(ty), TyKind::String)
}

/// Whether `site` stands in `block` after the instruction at `index`.
fn later_in_block(site: Site, block: BlockId, index: usize) -> bool {
    match site {
        Site::Inst(at, later) => at == block && later > index,
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

/// Each `Take` of a `String`, whole or a field, out of a place whose content
/// is read again after it becomes a clone of the place (S4).
fn clone_place_reads(module: &mut Module, body: BodyId, am: &mut AnalysisManager<'_>) {
    let content = am.get::<ContentLiveness>(module, body);
    let (of, types) = (module.body(body), module.types());
    let mut read_again: Vec<ValueId> = Vec::new();
    for (block, record) in of.blocks() {
        for (index, inst) in record.insts().iter().enumerate() {
            let Op::Take { target, .. } = inst.op() else {
                continue;
            };
            let result = inst.result().expect("a Take yields a value");
            let Some(place) = named_place(target) else {
                continue;
            };
            if is_string(types, of.ty_of(result)) && content.live_after(Site::Inst(block, index)).contains(&place) {
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
    let mut facts = EditFacts::new(am, module, body);
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
    let (block, mut index) = (inst.block(), inst.index());
    let record = &editor.body().block(block).insts()[index];
    let span = record.span();
    let mut moved: Vec<MovedString> = Vec::new();
    for (position, operand) in inst_operands(editor.body(), editor.types(), record).into_iter().enumerate() {
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
        let read_after = body.users(value).any(|site| later_in_block(site, block, index))
            || liveness.live_at_end(block).contains(&value);
        let cloned = if read_after { &operands[..] } else { &operands[..operands.len() - 1] };
        for position in cloned {
            let clone = editor.insert(Position::Before(inst), Op::StringClone { src: value }, ty, span);
            index += 1;
            editor
                .replace_operand(Site::Inst(block, index), *position, value, clone)
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
        let mut facts = EditFacts::new(am, module, body);
        let mut editor = Editor::<Discharge>::new(module, body, &mut facts);
        let mut cursor = editor.cursor();
        while let Some(inst) = cursor.next() {
            let editor = cursor.editor();
            let Op::Commit { value, .. } = editor.body().block(inst.block()).insts()[inst.index()].op() else {
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
    let mut facts = EditFacts::new(am, module, body);
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
        if matches!(editor.body().block(inst.block()).insts()[inst.index()].op(), Op::Fetch { .. }) {
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
    let Op::Take { target: RefTarget::Var(slot), path, .. } = body.block(take.block()).insts()[take.index()].op() else {
        return false;
    };
    path.is_empty() && body.context_of_slot(*slot).is_some() && body.use_count(value) == 0
}

/// Whether `inst` stores into a context's slot, whole, the value a `Fetch`
/// of that context produced: the second half of the load (context §3).
fn stores_a_load(editor: &Editor<'_, Discharge>, inst: InstRef) -> bool {
    let body = editor.body();
    let Op::Assign { target: RefTarget::Var(slot), path, value, .. } = body.block(inst.block()).insts()[inst.index()].op()
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
        body.block(loaded.block()).insts()[loaded.index()].op(),
        Op::Fetch { context } if Some(*context) == body.context_of_slot(slot)
    )
}

// -- insert_drops ------------------------------------------------------------

/// What a drop ends: a value, or the content of a place.
#[derive(Debug, Clone, Copy)]
enum Ended {
    Value(ValueId),
    Content(ValueId),
}

/// A drop the plan places, with the span it stands at.
#[derive(Debug, Clone, Copy)]
struct PlannedDrop {
    ended: Ended,
    span: Span,
}

impl PlannedDrop {
    /// How many instructions the drop puts in.
    fn width(self) -> usize {
        match self.ended {
            Ended::Value(_) => 1,
            Ended::Content(_) => 2,
        }
    }
}

/// The drops on one edge, which is split so they stand at the start of the
/// new block.
struct EdgeDrops {
    from: BlockId,
    to: BlockId,
    drops: Vec<PlannedDrop>,
}

/// Where every drop of a body stands, by the positions of the body before
/// the step: just before or just after an instruction, or on an edge.
#[derive(Default)]
struct DropPlan {
    before: FxHashMap<Site, Vec<PlannedDrop>>,
    after: FxHashMap<Site, Vec<PlannedDrop>>,
    on_edges: Vec<EdgeDrops>,
}

/// The block a cursor walks, and how many instructions drops put in it
/// before the instruction it stands at, so that instruction's index before
/// the step is known.
#[derive(Debug, Clone, Copy)]
struct Walked {
    block: BlockId,
    inserted: usize,
}

fn insert_drops_in(module: &mut Module, body: BodyId, am: &mut AnalysisManager<'_>) {
    let liveness = am.get::<Liveness>(module, body);
    let content = am.get::<ContentLiveness>(module, body);
    let loans = am.get::<Loans>(module, body);
    let facts = PlanFacts { body: module.body(body), types: module.types(), liveness: &liveness, content: &content, loans: &loans };
    let DropPlan { mut before, mut after, on_edges } = facts.plan();

    let mut no_facts = NoFacts;
    let mut editor = Editor::<Cfg>::new(module, body, &mut no_facts);
    let split: Vec<EdgeDrops> = on_edges
        .into_iter()
        .map(|EdgeDrops { from, to, drops }| EdgeDrops { from, to: editor.split_edge(from, to), drops })
        .collect();
    let changes = editor.finish();
    am.invalidate(body, &changes);

    let mut facts = EditFacts::new(am, module, body);
    let mut editor = Editor::<Drops>::new(module, body, &mut facts);
    let mut owed: Vec<PlannedDrop> = Vec::new();
    let mut walking: Option<Walked> = None;
    let mut cursor = editor.cursor();
    while let Some(inst) = cursor.next() {
        let editor = cursor.editor();
        let block = inst.block();
        let inserted = match walking {
            Some(walked) if walked.block == block => walked.inserted,
            _ => {
                if let Some(walked) = walking {
                    drop_all(editor, owed.drain(..), Position::End(walked.block));
                }
                0
            }
        };
        let original = Site::Inst(block, inst.index() - inserted);
        owed.extend(before.remove(&original).into_iter().flatten());
        let placed: usize = owed.iter().map(|drop| drop.width()).sum();
        drop_all(editor, owed.drain(..), Position::Before(inst));
        owed.extend(after.remove(&original).into_iter().flatten());
        walking = Some(Walked { block, inserted: inserted + placed });
    }
    if let Some(walked) = walking {
        drop_all(&mut editor, owed.drain(..), Position::End(walked.block));
    }
    for EdgeDrops { to: middle, drops, .. } in split {
        drop_all(&mut editor, drops.into_iter(), Position::End(middle));
    }
    assert!(
        before.is_empty() && after.is_empty(),
        "every instruction a drop stands by is walked, and {before:?} and {after:?} are left"
    );
    let changes = editor.finish();
    am.invalidate(body, &changes);
}

fn drop_all(editor: &mut Editor<'_, Drops>, drops: impl Iterator<Item = PlannedDrop>, at: Position) {
    for PlannedDrop { ended, span } in drops {
        let placed = match ended {
            Ended::Value(value) => editor.insert_drop(value, at, span),
            Ended::Content(place) => editor.drop_content(place, at, span),
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
    /// places) · edges) for the edges, and the context slots' forward pass.
    fn plan(&self) -> DropPlan {
        let body = self.body;
        let storages = self.storage_params();
        let mut plan = DropPlan::default();
        for (block, record) in body.blocks() {
            for (index, inst) in record.insts().iter().enumerate() {
                for operand in inst_operands(body, self.types, inst) {
                    if operand.taken == Taken::PartMoved && !copies(self.types, body.ty_of(operand.value)) {
                        parked_part_moved(Site::Inst(block, index), operand.value);
                    }
                }
            }
        }
        for (block, _) in body.blocks() {
            self.values_in_block(&storages, block, &mut plan);
            self.places_in_block(block, &mut plan);
            self.edges_of(&storages, block, &mut plan);
        }
        self.context_overwrites(&mut plan);
        plan
    }

    /// The parameters the body names as a storage, which are places.
    ///
    /// # Panics
    /// One of them is also passed as a value.
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
                Site::Inst(block, index) => {
                    let inst = &body.block(block).insts()[index];
                    inst_operands(body, self.types, inst).iter().any(|operand| operand.value == *param && !operand.names_place)
                }
                Site::Term(_) => true,
            });
            assert!(!as_value, "the parameter {param:?} is named both as a storage and as a value");
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
    fn values_in_block(&self, storages: &FxHashSet<ValueId>, block: BlockId, plan: &mut DropPlan) {
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
                || self.liveness.live_at_end(block).contains(&value)
                || named_by_terminator.contains(&value)
            {
                continue;
            }
            let last_use = body
                .users(value)
                .filter_map(|site| match site {
                    Site::Inst(at, index) if at == block => Some(index),
                    Site::Inst(..) | Site::Term(_) => None,
                })
                .max();
            let (site, span) = match last_use {
                Some(index) => {
                    let inst = &record.insts()[index];
                    let moved = inst_operands(body, self.types, inst)
                        .iter()
                        .any(|operand| operand.value == value && operand.taken == Taken::Moved);
                    if moved {
                        continue;
                    }
                    (Site::Inst(block, index), inst.span())
                }
                None => match body.definition(value) {
                    Definition::Inst(at, index) => (Site::Inst(at, index), body.block(at).insts()[index].span()),
                    definition => panic!(
                        "{value:?}, defined at {definition:?}, is dead where {block:?} starts; a drop there has no \
                         edge of its own to stand on, since each incoming edge passes the value it would drop"
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
            Definition::Param(_) => true,
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
    fn places_in_block(&self, block: BlockId, plan: &mut DropPlan) {
        let (body, record) = (self.body, self.body.block(block));
        for (index, inst) in record.insts().iter().enumerate() {
            let site = Site::Inst(block, index);
            let ended: BTreeSet<ValueId> = self
                .content_events(site, inst)
                .into_iter()
                .filter(|place| self.liveness_ends_content(*place) && !self.content.live_after(site).contains(place))
                .collect();
            for place in ended {
                let takes_all = inst_operands(body, self.types, inst)
                    .iter()
                    .any(|operand| operand.value == place && operand.names_place && operand.taken == Taken::Moved);
                if !takes_all {
                    let drop = PlannedDrop { ended: Ended::Content(place), span: inst.span() };
                    plan.after.entry(site).or_default().push(drop);
                }
            }
        }
        if block == body.entry() {
            for (_, param) in body.params() {
                let dead = !self.content.live_at_start(block).contains(param);
                assert!(
                    !(self.liveness_ends_content(*param) && dead),
                    "the storage of the parameter {param:?} is dead where the body starts; a drop there has no edge \
                     and no span of its own"
                );
            }
        }
    }

    /// The values and place contents live out of `block` and not into one
    /// of its successors, which end on that edge, where control leaves the
    /// block: they take the span of its terminator.
    fn edges_of(&self, storages: &FxHashSet<ValueId>, block: BlockId, plan: &mut DropPlan) {
        let (body, record) = (self.body, self.body.block(block));
        let terminator = record.terminator();
        let span = record.terminator_span();
        let named = terminator_operands(body, self.types, terminator);
        let mut values: BTreeSet<ValueId> = self.liveness.live_at_end(block).iter().copied().collect();
        values.extend(named.iter().filter(|named| named.taken == Taken::Read).map(|named| named.value));
        values.retain(|value| self.holds(storages, *value));
        for named in named.iter().filter(|named| named.taken == Taken::Moved) {
            values.remove(&named.value);
        }
        let mut places: BTreeSet<ValueId> = self.content.live_at_end(block).iter().copied().collect();
        places.extend(self.loans.reads(Site::Term(block)).iter().filter_map(|storage| storage_place(*storage)));
        places.retain(|place| self.liveness_ends_content(*place));
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

    /// Before each store that overwrites a context's slot whole, other than
    /// the store of its load, the content it replaces is the body's where
    /// the slot holds one there, and is dropped (context §3, D71).
    ///
    /// # Panics
    /// The slot holds its content on some paths to such a store only.
    fn context_overwrites(&self, plan: &mut DropPlan) {
        let body = self.body;
        let contexts: Vec<ValueId> = body
            .slots()
            .iter()
            .copied()
            .filter(|slot| body.context_of_slot(*slot).is_some() && !copies(self.types, body.ty_of(*slot)))
            .collect();
        if contexts.is_empty() {
            return;
        }
        let presence = ContextPresence { body, types: self.types, contexts: &contexts };
        let facts = solve_dense(&presence, body);
        for (block, record) in body.blocks() {
            let mut state = facts.at(block).clone();
            for (index, inst) in record.insts().iter().enumerate() {
                if let Some(slot) = presence.overwritten(inst) {
                    match state.of(slot) {
                        Held::Present => {
                            let drop = PlannedDrop { ended: Ended::Content(slot), span: inst.span() };
                            plan.before.entry(Site::Inst(block, index)).or_default().push(drop);
                        }
                        Held::Absent | Held::Unreached => {}
                        Held::Maybe => panic!(
                            "the context slot {slot:?} holds its content on some paths only where {block:?}[{index}] \
                             overwrites it; which of them drops the old content is not settled"
                        ),
                    }
                }
                presence.transfer(inst, &mut state);
            }
        }
    }
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

// -- Whether a context's slot holds content ----------------------------------

/// Whether a context's slot holds a content at a point, over the paths
/// that reach it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Held {
    Unreached,
    Absent,
    Present,
    Maybe,
}

impl Held {
    fn join(self, other: Held) -> Held {
        match (self, other) {
            (Held::Unreached, other) | (other, Held::Unreached) => other,
            (a, b) if a == b => a,
            _ => Held::Maybe,
        }
    }
}

/// Each context slot's `Held`, by slot; a slot not in the map is unreached.
#[derive(Debug, Clone, PartialEq, Default)]
struct HeldBySlot(FxHashMap<ValueId, Held>);

impl HeldBySlot {
    fn of(&self, slot: ValueId) -> Held {
        self.0.get(&slot).copied().unwrap_or(Held::Unreached)
    }
}

impl JoinSemiLattice for HeldBySlot {
    fn bottom() -> Self {
        HeldBySlot::default()
    }

    fn join(&mut self, other: &Self) -> bool {
        let mut rose = false;
        for (slot, held) in &other.0 {
            let joined = self.of(*slot).join(*held);
            if joined != self.of(*slot) {
                self.0.insert(*slot, joined);
                rose = true;
            }
        }
        rose
    }

    fn le(&self, other: &Self) -> bool {
        self.0.iter().all(|(slot, held)| other.of(*slot).join(*held) == other.of(*slot))
    }
}

/// Finite: four states per slot.
impl Widen for HeldBySlot {
    fn widen(&mut self, next: &Self) -> bool {
        self.join(next)
    }
}

/// Forward over the context slots: empty where the body starts; a whole
/// store fills one; a `Take` of all of it that does not copy empties it.
struct ContextPresence<'f> {
    body: &'f Body,
    types: &'f Types<Concrete>,
    contexts: &'f [ValueId],
}

impl ContextPresence<'_> {
    /// The context slot `inst` overwrites whole, other than by storing its
    /// own load.
    fn overwritten(&self, inst: &Inst) -> Option<ValueId> {
        let Op::Assign { target: RefTarget::Var(slot), path, value, .. } = inst.op() else {
            return None;
        };
        let stores_its_load = match self.body.definition(*value) {
            Definition::Inst(block, index) => matches!(
                self.body.block(block).insts()[index].op(),
                Op::Fetch { context } if Some(*context) == self.body.context_of_slot(*slot)
            ),
            _ => false,
        };
        (path.is_empty() && self.contexts.contains(slot) && !stores_its_load).then_some(*slot)
    }
}

impl DenseAnalysis for ContextPresence<'_> {
    type State = HeldBySlot;

    const DIRECTION: Direction = Direction::Forward;

    fn entry(&self, _: &Body) -> HeldBySlot {
        HeldBySlot(self.contexts.iter().map(|slot| (*slot, Held::Absent)).collect())
    }

    fn transfer(&self, inst: &Inst, state: &mut HeldBySlot) {
        match inst.op() {
            Op::Assign { target: RefTarget::Var(slot), path, .. } if path.is_empty() && self.contexts.contains(slot) => {
                state.0.insert(*slot, Held::Present);
            }
            Op::Take { target: RefTarget::Var(slot), path, .. }
                if path.is_empty() && self.contexts.contains(slot) && !copies(self.types, self.body.ty_of(*slot)) =>
            {
                state.0.insert(*slot, Held::Absent);
            }
            _ => {}
        }
    }

    fn transfer_terminator(&self, _: &Terminator, _: &mut HeldBySlot) {}

    fn transfer_edge(&self, _: &Edge<'_>, _: &[ValueId], _: &mut HeldBySlot) {}
}
