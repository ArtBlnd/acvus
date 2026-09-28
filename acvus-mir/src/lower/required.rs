//! The required steps after lowering (D40, D55): meaning, not optimization.
//! They run on every program, at every level, in the order `clone_strings`,
//! `remove_dead_context_ops`, the checks, `insert_drops`. Each edits every
//! body of the module through its own scope's editor, with `EditFacts` as
//! its needs, and invalidates the manager for what it changed.
//! - `clone_strings`, scope `Value`: a `String` use that is not its last is
//!   a clone, and its last use moves it (ownership §3), through `insert`
//!   and `replace_use` (G3).
//! - `remove_dead_context_ops`, scope `Discharge`: a load whose value is
//!   never read and a commit of a context the body never writes do not
//!   exist (context §3), through `discharge` (G1).
//! - `insert_drops`, scope `Cfg` for an edge split and then `Drops`: a
//!   value the program holds that does not copy is dropped where its
//!   liveness ends, and a value moved to another holder is not (ownership
//!   §6), through `split_edge` (G2) and `insert_drop` (S3). It runs after
//!   the checks, which admit the moves.

use std::collections::BTreeSet;

use acvus_ast::Span;
use rustc_hash::FxHashMap;
use smallvec::SmallVec;

use crate::ir::cfg::edit::{CannotDischarge, Cfg, Discharge, Drops, Editor, InstRef, NoFacts, Position, Value};
use crate::ir::cfg::{BlockId, Body, BodyId, Changes, Definition, Inst, Module, Op, Site, Terminator};
use crate::ir::{Callee, ForSource, RefTarget, ValueId};
use crate::passes::analysis::{AnalysisManager, ContentLiveness, Liveness};
use crate::passes::edit_facts::EditFacts;
use crate::ty::{Calls, Concrete, Mutability, Ty, TyKind, Types, copies};

#[cfg(test)]
mod tests;

/// Every body: a `String` value a use moves while the value is read again
/// on some path after it is cloned just before that use, with the use's
/// span, and the use takes the clone (ownership §3). A use is last on a
/// path when no later read of the same value follows it on that path.
///
/// Cost per body: `Liveness`', `ContentLiveness`', and for each moving
/// `String` use one insertion and one `replace_use`, each an edit and a
/// refresh of `EditFacts`.
///
/// # Panics
/// Where a body needs what no editor operation makes yet: a `String`
/// slot's content read by a `Take` and read again later, one site moving
/// the same `String` twice, or a clone before a terminator, which carries
/// no span. Each message states the missing operation.
pub fn clone_strings(module: &mut Module, am: &mut AnalysisManager<'_>) {
    for body in module.body_ids() {
        let changes = clone_strings_in(module, body, am);
        am.invalidate(body, &changes);
    }
}

/// Every body: each load whose stored value no read reaches, and each
/// commit of a context the body never writes, is offered to `discharge`,
/// which removes it where `EditFacts` proves it `NeverRead` or
/// `NeverWritten`; a volatile context's are never proven (D61). A load of
/// a context held in a slot is its `Fetch` and the `Assign` of that
/// `Fetch`'s value into the context's slot, so that store is offered first,
/// and then every `Fetch` and `Commit`.
///
/// Cost per body: O(instructions) discharges, each a refresh of
/// `EditFacts`.
pub fn remove_dead_context_ops(module: &mut Module, am: &mut AnalysisManager<'_>) {
    for body in module.body_ids() {
        let changes = remove_dead_context_ops_in(module, body, am);
        am.invalidate(body, &changes);
    }
}

/// Every body: each value the program holds that does not copy (ownership
/// §1) is dropped where its liveness ends (ownership §6). Inside a block
/// that is just after its last use, unless that use moves it to another
/// holder; on an edge out of a block where it is live into the block's
/// other successors, the edge is split and the drop stands at the new
/// block's start. A value moved on some paths is dropped on the others
/// only. The body's own parameters and its blocks' parameters are held; a
/// capture is the closure's, and a slot is storage.
///
/// Cost per body: `Liveness`', O(values · blocks) for the plan, one
/// `split_edge` per edge that ends a value, and one `insert_drop` per drop,
/// each a refresh of `EditFacts`.
///
/// # Panics
/// Where a body needs what no editor operation or ledger rule gives yet: a
/// slot whose type does not copy, whose content no operation drops; a
/// part taken out of a value that does not copy; or a drop of a parameter
/// where no instruction gives it a span. Each message states the question.
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
}

/// A taking no ledger rule settles yet; the text is the question.
#[derive(Debug, Clone, Copy)]
struct Unsettled(&'static str);

const PARTIAL_TAKE: Unsettled = Unsettled(
    "a part that does not copy is taken out of a value that does not copy; the rest of that value stays held, and \
     which operation ends the rest is not settled",
);
const FIELD_REPLACED: Unsettled = Unsettled(
    "a FieldSet makes a new value from one that does not copy; where the field it replaces ends is not settled",
);
const STORAGE_OVERWRITTEN: Unsettled = Unsettled(
    "an Assign replaces the content of a by-value parameter's storage, which does not copy; an Assign drops \
     nothing, and no editor operation drops a storage's content",
);

/// One operand of an instruction and how it is taken. `names_place` marks
/// an operand that names a storage, as a `Take`'s or an `Assign`'s target,
/// rather than passing a value; `replace_use` cannot redirect such an
/// operand to a value.
#[derive(Debug, Clone, Copy)]
struct Operand {
    value: ValueId,
    taken: Result<Taken, Unsettled>,
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

/// Every operand of `inst` but the order token, with how it is taken.
/// Cost: O(its operands).
fn inst_operands(body: &Body, types: &Types<Concrete>, inst: &Inst) -> SmallVec<[Operand; 4]> {
    let ty = |value: ValueId| body.ty_of(value);
    let taken = |value: ValueId, taken: Taken| Operand { value, taken: Ok(taken), names_place: false };
    let passed = |value: ValueId| taken(value, by_value(types, ty(value)));
    let read = |value: ValueId| taken(value, Taken::Read);
    let result_copies = || inst.result().is_some_and(|result| copies(types, ty(result)));
    let extracted = |src: ValueId| Operand {
        value: src,
        taken: if copies(types, ty(src)) || result_copies() { Ok(Taken::Read) } else { Err(PARTIAL_TAKE) },
        names_place: false,
    };
    let place = |target: &RefTarget, on_param: Result<Taken, Unsettled>| match target {
        RefTarget::Var(slot) => Operand { value: *slot, taken: Ok(Taken::Read), names_place: true },
        RefTarget::Param(param) => Operand { value: *param, taken: on_param, names_place: true },
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
        Op::Ref { target, .. } => operands.push(place(target, Ok(Taken::Read))),
        Op::Take { target, path, .. } => {
            let on_param = match target {
                RefTarget::Param(param) if copies(types, ty(*param)) => Ok(Taken::Read),
                _ if path.is_empty() => Ok(Taken::Moved),
                _ if result_copies() => Ok(Taken::Read),
                _ => Err(PARTIAL_TAKE),
            };
            operands.push(place(target, on_param));
        }
        Op::Assign { target, value, .. } => {
            let on_param = match target {
                RefTarget::Param(param) if !copies(types, ty(*param)) => Err(STORAGE_OVERWRITTEN),
                _ => Ok(Taken::Read),
            };
            operands.extend([place(target, on_param), passed(*value)]);
        }
        Op::AsSlice { container, .. } => operands.push(read(*container)),
        Op::Index { slice, index, .. } => operands.extend([read(*slice), read(*index)]),
        Op::IndexSet { slice, index, value, .. } => operands.extend([read(*slice), read(*index), passed(*value)]),
        Op::Commit { value, .. } => operands.push(passed(*value)),
        Op::FieldGet { object: src, .. }
        | Op::ObjectGet { object: src, .. }
        | Op::TupleIndex { tuple: src, .. }
        | Op::ArrayIndex { array: src, .. }
        | Op::UnwrapVariant { src } => operands.push(extracted(*src)),
        Op::FieldSet { object, value, .. } => {
            let object = match copies(types, ty(*object)) {
                true => read(*object),
                false => Operand { value: *object, taken: Err(FIELD_REPLACED), names_place: false },
            };
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

// -- clone_strings -----------------------------------------------------------

fn clone_strings_in(module: &mut Module, body: BodyId, am: &mut AnalysisManager<'_>) -> Changes {
    refuse_slot_reads_needing_clones(module, body, am);
    let liveness = am.get::<Liveness>(module, body);
    let mut facts = EditFacts::new(am, module, body);
    let mut editor = Editor::<Value>::new(module, body, &mut facts);
    let mut cursor = editor.cursor();
    while let Some(inst) = cursor.next() {
        clone_before_inst(cursor.editor(), &liveness, inst);
    }
    let blocks: Vec<BlockId> = editor.body().blocks().map(|(block, _)| block).collect();
    for block in blocks {
        refuse_terminator_needing_clones(editor.body(), editor.types(), &liveness, block);
    }
    editor.finish()
}

/// A `String` slot's content read by a `Take` while a later read of the
/// same content follows is a use that is not its last. Cloning it needs a
/// shared `Ref` of the place, of type `&String`, which the editor cannot
/// add to the module's type table, and the `Take` gone, which scope `Value`
/// cannot remove.
///
/// # Panics
/// Such a read exists.
fn refuse_slot_reads_needing_clones(module: &Module, body: BodyId, am: &mut AnalysisManager<'_>) {
    let content = am.get::<ContentLiveness>(module, body);
    let (of, types) = (module.body(body), module.types());
    for (block, record) in of.blocks() {
        for (index, inst) in record.insts().iter().enumerate() {
            let Op::Take { target: RefTarget::Var(slot), .. } = inst.op() else {
                continue;
            };
            let result = inst.result().expect("a Take yields a value");
            let site = Site::Inst(block, index);
            assert!(
                !is_string(types, of.ty_of(result)) || !content.live_after(site).contains(slot),
                "the String slot {slot:?} is read at {site:?} and read again later, so that read is a clone \
                 (ownership §3); no editor operation clones a slot's content in place of its Take"
            );
        }
    }
}

/// A `String` an instruction moves, and through how many of its operands.
struct MovedString {
    value: ValueId,
    ty: Ty,
    operands: usize,
    names_place: bool,
}

/// Clones each `String` the instruction at `inst` moves while it is read
/// again on some path after it, and gives the instruction the clone.
///
/// # Panics
/// The instruction moves one `String` through two operands, which
/// `replace_use` cannot give two different values, or moves a `String`
/// parameter's storage that is read again.
fn clone_before_inst(editor: &mut Editor<'_, Value>, liveness: &Liveness, inst: InstRef) {
    let (block, mut index) = (inst.block(), inst.index());
    let record = &editor.body().block(block).insts()[index];
    let span = record.span();
    let mut moved: Vec<MovedString> = Vec::new();
    for operand in inst_operands(editor.body(), editor.types(), record) {
        let ty = editor.body().ty_of(operand.value);
        if !is_string(editor.types(), ty) {
            continue;
        }
        let Taken::Moved = operand.taken.unwrap_or_else(|Unsettled(question)| panic!("{question}")) else {
            continue;
        };
        match moved.iter_mut().find(|string| string.value == operand.value) {
            Some(string) => string.operands += 1,
            None => moved.push(MovedString { value: operand.value, ty, operands: 1, names_place: operand.names_place }),
        }
    }
    for MovedString { value, ty, operands: count, names_place } in moved {
        let body = editor.body();
        let read_again = body.users(value).any(|site| later_in_block(site, block, index))
            || liveness.live_at_end(block).contains(&value);
        if !read_again && count == 1 {
            continue;
        }
        assert!(
            count == 1,
            "{block:?}[{index}] moves the String {value:?} through {count} operands; replace_use gives every operand \
             of one site the same value, so no editor operation clones all but one of them"
        );
        assert!(
            !names_place,
            "{block:?}[{index}] takes the String parameter {value:?} out of its storage, and it is read again \
             later; no editor operation clones a storage's content in place of its Take"
        );
        let clone = editor.insert(Position::Before(inst), Op::StringClone { src: value }, ty, span);
        index += 1;
        editor
            .replace_use(Site::Inst(block, index), value, clone)
            .expect("a clone inserted just before its use dominates it");
    }
}

/// A terminator never needs a clone in a body this step accepts: an edge
/// that passes a `String` its target also reads by name, or passes one
/// twice, would need a clone standing before the terminator, and only a
/// `Return`, which has no edge, carries a span for it.
///
/// # Panics
/// Such an edge exists.
fn refuse_terminator_needing_clones(body: &Body, types: &Types<Concrete>, liveness: &Liveness, block: BlockId) {
    for edge in body.block(block).terminator().edges() {
        for (position, arg) in edge.args.iter().enumerate() {
            if !is_string(types, body.ty_of(*arg)) {
                continue;
            }
            let passed_again = edge.args[position + 1..].contains(arg);
            assert!(
                !passed_again && !liveness.live_at_start(edge.to).contains(arg),
                "the edge {block:?} -> {:?} passes the String {arg:?} while it is read again after, so that use is a \
                 clone (ownership §3); the clone would stand before a terminator that carries no span",
                edge.to
            );
        }
    }
}

// -- remove_dead_context_ops -------------------------------------------------

fn remove_dead_context_ops_in(module: &mut Module, body: BodyId, am: &mut AnalysisManager<'_>) -> Changes {
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
        let op = editor.body().block(inst.block()).insts()[inst.index()].op();
        if op.touches_context() {
            offer(editor, inst);
        }
    }
    editor.finish()
}

fn offer(editor: &mut Editor<'_, Discharge>, inst: InstRef) {
    match editor.discharge(inst) {
        Ok(()) | Err(CannotDischarge::NotProven | CannotDischarge::WrongFact) => {}
    }
}

/// Whether `inst` stores into a context's slot, whole, the value a `Fetch`
/// of that context produced: the second half of the load (context §3).
fn stores_a_load(editor: &Editor<'_, Discharge>, inst: InstRef) -> bool {
    let body = editor.body();
    let Op::Assign { target: RefTarget::Var(slot), path, value, .. } = body.block(inst.block()).insts()[inst.index()].op()
    else {
        return false;
    };
    let Some(context) = body.context_of_slot(*slot) else {
        return false;
    };
    let Some(loaded) = editor.defining_inst(*value) else {
        return false;
    };
    let loads = matches!(
        body.block(loaded.block()).insts()[loaded.index()].op(),
        Op::Fetch { context: fetched } if *fetched == context
    );
    path.is_empty() && loads
}

// -- insert_drops ------------------------------------------------------------

/// A drop the plan places, with the span it stands at.
#[derive(Debug, Clone, Copy)]
struct PlannedDrop {
    value: ValueId,
    span: Span,
}

/// The drops on one edge, which is split so they stand at the start of the
/// new block.
struct EdgeDrops {
    from: BlockId,
    to: BlockId,
    drops: Vec<PlannedDrop>,
}

/// Where every drop of a body stands, by the positions of the body as
/// lowered: at a block's start, just after an instruction, or on an edge.
struct DropPlan {
    at_start: FxHashMap<BlockId, Vec<PlannedDrop>>,
    /// By the site of the instruction they follow.
    after: FxHashMap<Site, Vec<PlannedDrop>>,
    on_edges: Vec<EdgeDrops>,
}

/// The block a cursor walks, and how many drops were put in it before the
/// instruction it stands at, so that instruction's index as lowered is
/// known.
#[derive(Debug, Clone, Copy)]
struct Walked {
    block: BlockId,
    inserted: usize,
}

fn insert_drops_in(module: &mut Module, body: BodyId, am: &mut AnalysisManager<'_>) {
    let liveness = am.get::<Liveness>(module, body);
    let DropPlan { mut at_start, mut after, on_edges } = DropPlan::of(module.body(body), module.types(), &liveness);

    let mut no_facts = NoFacts;
    let mut editor = Editor::<Cfg>::new(module, body, &mut no_facts);
    let split: FxHashMap<BlockId, Vec<PlannedDrop>> =
        on_edges.into_iter().map(|EdgeDrops { from, to, drops }| (editor.split_edge(from, to), drops)).collect();
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
                owed.extend(at_start.remove(&block).into_iter().flatten());
                0
            }
        };
        let original = inst.index() - inserted;
        let placed = owed.len();
        drop_all(editor, owed.drain(..), Position::Before(inst));
        owed.extend(after.remove(&Site::Inst(block, original)).into_iter().flatten());
        walking = Some(Walked { block, inserted: inserted + placed });
    }
    if let Some(walked) = walking {
        drop_all(&mut editor, owed.drain(..), Position::End(walked.block));
    }
    let empty_blocks = at_start.into_iter().chain(split);
    for (block, drops) in empty_blocks {
        drop_all(&mut editor, drops.into_iter(), Position::End(block));
    }
    assert!(after.is_empty(), "every instruction a drop follows is walked, and {after:?} are left");
    let changes = editor.finish();
    am.invalidate(body, &changes);
}

fn drop_all(editor: &mut Editor<'_, Drops>, drops: impl Iterator<Item = PlannedDrop>, at: Position) {
    for PlannedDrop { value, span } in drops {
        editor
            .insert_drop(value, at, span)
            .unwrap_or_else(|refused| panic!("the plan drops {value:?} at {at:?}, where it is still used: {refused:?}"));
    }
}

impl DropPlan {
    /// Cost: O(the body's operands · the uses of each), and O(values ·
    /// edges) for the edges.
    ///
    /// # Panics
    /// As `insert_drops`.
    fn of(body: &Body, types: &Types<Concrete>, liveness: &Liveness) -> DropPlan {
        refuse_slots_that_do_not_copy(body, types);
        refuse_unsettled_takings(body, types);
        let mut plan = DropPlan { at_start: FxHashMap::default(), after: FxHashMap::default(), on_edges: Vec::new() };
        for (block, record) in body.blocks() {
            plan.in_block(body, types, liveness, block);
            let terminator = record.terminator();
            let named = terminator_operands(body, types, terminator);
            let mut held_after: BTreeSet<ValueId> = liveness.live_at_end(block).iter().copied().collect();
            held_after.extend(named.iter().filter(|named| named.taken == Taken::Read).map(|named| named.value));
            held_after.retain(|value| is_held(body, types, *value));
            for named in named.iter().filter(|named| named.taken == Taken::Moved) {
                held_after.remove(&named.value);
            }
            for edge in terminator.edges() {
                let moved_on_edge = |value: &ValueId| {
                    edge.args.iter().any(|arg| arg == value && by_value(types, body.ty_of(*arg)) == Taken::Moved)
                };
                let drops: Vec<PlannedDrop> = held_after
                    .iter()
                    .filter(|value| !moved_on_edge(value) && !liveness.live_at_start(edge.to).contains(*value))
                    .map(|value| PlannedDrop { value: *value, span: defining_span(body, *value) })
                    .collect();
                if !drops.is_empty() {
                    plan.on_edges.push(EdgeDrops { from: block, to: edge.to, drops });
                }
            }
        }
        plan
    }

    /// The drops of values whose liveness ends inside `block`: those it
    /// defines or reads that are neither live after it nor named by its
    /// terminator.
    fn in_block(&mut self, body: &Body, types: &Types<Concrete>, liveness: &Liveness, block: BlockId) {
        let record = body.block(block);
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
            if !is_held(body, types, value)
                || liveness.live_at_end(block).contains(&value)
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
            match last_use {
                Some(index) => {
                    let inst = &record.insts()[index];
                    let moved = inst_operands(body, types, inst)
                        .iter()
                        .any(|operand| operand.value == value && operand.taken.is_ok_and(|taken| taken == Taken::Moved));
                    if !moved {
                        self.after.entry(Site::Inst(block, index)).or_default().push(PlannedDrop { value, span: inst.span() });
                    }
                }
                None => match body.definition(value) {
                    Definition::Inst(at, index) => {
                        let span = body.block(at).insts()[index].span();
                        self.after.entry(Site::Inst(at, index)).or_default().push(PlannedDrop { value, span });
                    }
                    Definition::Param(_) | Definition::BlockParam(..) => {
                        let span = defining_span(body, value);
                        self.at_start.entry(block).or_default().push(PlannedDrop { value, span });
                    }
                    Definition::Capture(_) | Definition::OrderParam | Definition::Slot(_) => {
                        panic!("{value:?} is held by the program, and a capture, the order parameter or a slot is not")
                    }
                },
            }
        }
    }
}

/// Whether the program holds `value` and must end it: it does not copy,
/// and it is neither an order token, nor a slot, which is storage, nor a
/// capture, which the closure holds and a call of it only reads.
fn is_held(body: &Body, types: &Types<Concrete>, value: ValueId) -> bool {
    let holdable = match body.definition(value) {
        Definition::Param(_) | Definition::BlockParam(..) | Definition::Inst(..) => true,
        Definition::Capture(_) | Definition::OrderParam | Definition::Slot(_) => false,
    };
    holdable && !body.is_token(value) && !copies(types, body.ty_of(value))
}

/// The span of the instruction that defines `value`, for a drop that no
/// instruction of its own block ends.
///
/// # Panics
/// `value` is a parameter of the body or of a block: no instruction gives
/// it a span, and a terminator other than `Return` carries none.
fn defining_span(body: &Body, value: ValueId) -> Span {
    match body.definition(value) {
        Definition::Inst(block, index) => body.block(block).insts()[index].span(),
        definition => panic!(
            "{value:?} is dropped on an edge or at a block's start and is defined at {definition:?}, where no \
             instruction gives the drop a span"
        ),
    }
}

/// # Panics
/// A slot's type does not copy: where its content ends, it must be dropped,
/// and `insert_drop` drops a value, not a slot's content; a drop of the slot
/// just before the store that replaces its content is refused, since the
/// store names the slot.
fn refuse_slots_that_do_not_copy(body: &Body, types: &Types<Concrete>) {
    for slot in body.slots() {
        assert!(
            copies(types, body.ty_of(*slot)),
            "the slot {slot:?} holds a type that does not copy; no editor operation drops a slot's content"
        );
    }
}

/// # Panics
/// An operand the program holds is taken in a way no rule settles.
fn refuse_unsettled_takings(body: &Body, types: &Types<Concrete>) {
    for (block, record) in body.blocks() {
        for (index, inst) in record.insts().iter().enumerate() {
            for operand in inst_operands(body, types, inst) {
                if let Err(Unsettled(question)) = operand.taken
                    && is_held(body, types, operand.value)
                {
                    panic!("{block:?}[{index}] takes {:?}: {question}", operand.value);
                }
            }
        }
    }
}
