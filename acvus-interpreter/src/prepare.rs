//! The preparation: a `MirBody` becomes a `Code` once, at module load
//! (RFC-0044).
//!
//! One exhaustive match over `InstKind` chooses each operation from the
//! instruction kind and the operand types in `val_types`, so an instruction
//! kind added to the IR fails to compile until it is prepared. What the cut
//! interpreter decided per execution — the type an arithmetic runs at, the
//! literal's width, whether a read clones a `String`, which instance an
//! extern call reaches, the page key a context is stored under, where a
//! label sits — is decided here.

use std::marker::PhantomData;
use std::mem;
use std::ops::Range;
use std::sync::Arc;

use acvus_ast::{BinOp, Literal, UnaryOp};
use acvus_extern::AsyncCall;
use acvus_mir::analysis::inst_info;
use acvus_mir::graph::QualifiedRef;
use acvus_mir::ir::{
    Callee, ExternInstance, Inst, InstKind, Label, MirBody, MirModule, PathSeg, RefTarget, ValueId,
};
use acvus_mir::ty::{IntTy, Task, Ty};
use acvus_utils::{Astr, Interner, LocalIdOps};
use rustc_hash::FxHashMap;
use smallvec::SmallVec;

use crate::code::{
    Arith, Block, BlockId, Body, ChainBounds, Code, Compare, ConcatPart, Deref, EntryKonst, Expr,
    ExprBody, ExprChain, FieldSlot, Konst, Off, Op, Prepared, Root, Shape, Slot, SlotKind, Step,
    Terminator,
};
use crate::interpreter::Executable;
use crate::ops::arith::{self, Binary, Unary, for_int_ty};
use crate::ops::chain::{self, ChainTy, Plan};
use crate::ops::{call, composite, constant, control, index, pattern, storage, string, variant};
use crate::runtime::{ExternHandler, StateAbi, SyncAbi, SyncCall};
use crate::value::{Kind, Value};

/// The slot table's empty entry: a value that is neither defined nor live.
const NO_SLOT: u32 = u32::MAX;

const _: () = assert!(
    NO_SLOT > crate::regs::MAX_FRAME_SLOTS as u32,
    "the empty slot entry is not a register index"
);

pub struct PrepareCtx<'a> {
    pub interner: &'a Interner,
    pub externs: &'a FxHashMap<QualifiedRef, Executable>,
    pub context_names: &'a FxHashMap<QualifiedRef, Astr>,
}

impl PrepareCtx<'_> {
    fn page_key(&self, context: &QualifiedRef) -> Box<str> {
        let name = self
            .context_names
            .get(context)
            .unwrap_or_else(|| panic!("context: no name for {context:?}"));
        self.interner.resolve(*name).into()
    }

    fn handler(&self, id: &QualifiedRef, instance: usize) -> ExternHandler {
        let Some(Executable::Extern(handlers)) = self.externs.get(id) else {
            panic!("{id:?} is called as an ExternFn but is not one of the module's externs")
        };
        handlers
            .get(instance)
            .unwrap_or_else(|| {
                panic!(
                    "{id:?} has {} instances; the checker settled on instance {instance}",
                    handlers.len()
                )
            })
            .clone()
    }

    fn extern_is_sync(&self, id: &QualifiedRef, instance: usize) -> bool {
        self.handler(id, instance).is_sync()
    }
}

/// Every body of a module, prepared: the closures first, so a `MakeClosure`
/// resolves its body to an `Arc<Code>` at preparation.
pub fn prepare_module(module: &MirModule, ctx: &PrepareCtx<'_>) -> Prepared {
    let mut closures: FxHashMap<Label, Arc<Code>> = FxHashMap::default();
    let mut remaining: Vec<(&Label, &MirBody)> = module.closures.iter().collect();
    remaining.sort_by_key(|(label, _)| **label);

    // A closure body that makes a closure needs that body prepared first;
    // the nesting is finite, so a pass that prepares every body whose
    // closures are ready reaches all of them.
    while !remaining.is_empty() {
        let ready: Vec<(&Label, &MirBody)> = remaining
            .iter()
            .copied()
            .filter(|(_, body)| made_closures(body).all(|label| closures.contains_key(&label)))
            .collect();
        assert!(
            !ready.is_empty(),
            "closure bodies make each other in a cycle: {:?}",
            remaining
                .iter()
                .map(|(label, _)| **label)
                .collect::<Vec<_>>()
        );
        for (label, body) in ready {
            let code = prepare_body(body, ctx, &closures, BodyRole::Closure);
            closures.insert(*label, Arc::new(code));
        }
        remaining.retain(|(label, _)| !closures.contains_key(label));
    }

    let main = Arc::new(prepare_body(&module.main, ctx, &closures, BodyRole::Entry));
    Prepared { main, closures }
}

fn made_closures(body: &MirBody) -> impl Iterator<Item = Label> + '_ {
    body.insts.iter().filter_map(|inst| match &inst.kind {
        InstKind::MakeClosure { body, .. } => Some(*body),
        _ => None,
    })
}

/// Which body of a module is being prepared.
///
/// Only a closure body can become a `Code::Expr`. A module's entry body is
/// entered with a frame — `call_module` fills its parameter registers and
/// `Machine::run` walks its operations — and a chain has no frame to be
/// entered with.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum BodyRole {
    Entry,
    Closure,
}

pub fn prepare_body(
    body: &MirBody,
    ctx: &PrepareCtx<'_>,
    closures: &FxHashMap<Label, Arc<Code>>,
    role: BodyRole,
) -> Code {
    let mut prep = Prepare::new(body, ctx, closures, label_map(body));

    prep.hoist_konsts();
    let regions = prep.regions();

    if let BodyRole::Closure = role
        && let Some(expr) = prep.expression_body()
    {
        return Code::Expr(Arc::new(expr));
    }

    let blocks = prep.blocks(0..body.insts.len(), &regions);
    // RFC-0046 asked for equality here. It does not hold, and
    // `io_in_iteration` (acvus-interpreter-test/tests/extern_fn.rs)
    // measures the gap: a closure demoted to the parameter's effect is
    // typed `Async` while its operations only add, and that body holds no
    // call for `suspends_at` to read.
    let may_suspend = body.task > Task::Sync;
    debug_assert!(
        !prep.may_suspend || may_suspend,
        "body {role:?}: the prepared operations await and the checker's task does not say so"
    );

    let frame_len = prep.frame_len();
    let param_ids: Vec<ValueId> = body.params.iter().map(|(_, v)| *v).collect();
    let param_marks = prep.take_mask(&param_ids);
    let params = param_ids.iter().map(|id| prep.off(*id)).collect();
    let captures = body.captures.iter().map(|(_, v)| prep.off(*v)).collect();
    let order_param = body.order_param.map(|id| prep.off(id));
    let entry_konsts = prep.entry_konsts();
    let slot_kinds = prep.slot_kinds(frame_len);

    Code::Body(Arc::new(Body {
        blocks,
        entry: 0,
        frame_len,
        entry_konsts,
        slot_kinds,
        may_suspend,
        params,
        param_marks,
        captures,
        order_param,
        span: body
            .insts
            .first()
            .unwrap_or_else(|| panic!("body {role:?} holds no instruction, so it cannot return"))
            .span,
    }))
}

fn label_map(body: &MirBody) -> FxHashMap<Label, u32> {
    body.insts
        .iter()
        .enumerate()
        .filter_map(|(at, inst)| match &inst.kind {
            InstKind::BlockLabel { label, .. } => Some((*label, at as u32)),
            _ => None,
        })
        .collect()
}

struct Prepare<'a> {
    body: &'a MirBody,
    ctx: &'a PrepareCtx<'a>,
    closures: &'a FxHashMap<Label, Arc<Code>>,
    labels: FxHashMap<Label, u32>,
    slots: Slots,
    scratch: u32,
    scratch_used: bool,
    may_suspend: bool,
    /// The block array being emitted. A region owns its blocks, so the
    /// array a jump's target names is the innermost one being built, and
    /// `blocks` saves and restores this around every nested region.
    level: Level,
    def_inst: Vec<Option<usize>>,
    use_counts: Vec<u32>,
    konsts: Konsts,
}

/// Where each instruction of the block array under construction begins.
/// Only a `BlockLabel` a jump can name is in it; every other instruction
/// sits inside a block rather than starting one.
#[derive(Default)]
struct Level {
    block_of_inst: FxHashMap<usize, BlockId>,
    /// The first id past the array's own blocks, which is where the edge
    /// blocks below are numbered from.
    blocks_len: BlockId,
    /// One block per conditional edge that carries a parallel move: the
    /// `Mov`s and a `Goto`. An edge with no move names its target directly.
    edges: Vec<Block>,
}

#[derive(Default)]
struct Konsts {
    slot_of: FxHashMap<ValueId, u32>,
    value_at: FxHashMap<u32, Value>,
    insts: Vec<usize>,
}

impl Konsts {
    fn holds_inst(&self, at: usize) -> bool {
        self.insts.contains(&at)
    }
}

/// One `while` the recognizer matched, as indexes into `MirBody::insts`.
struct LoopRegion {
    enter_jump: Option<usize>,
    head: usize,
    head_block: Range<usize>,
    head_regions: Vec<Region>,
    jump_if: usize,
    body_block: Range<usize>,
    body_regions: Vec<Region>,
    back: usize,
}

/// One `if/else` the recognizer matched, as indexes into `MirBody::insts`.
struct DiamondRegion {
    jump_if: usize,
    on_true: ArmRegion,
    on_false: ArmRegion,
    join: usize,
}

/// One side of a diamond. `Direct` is the side the lowering gave no block
/// of its own: the `JumpIf`'s own edge reaches the join.
enum ArmRegion {
    Block {
        block: Range<usize>,
        regions: Vec<Region>,
        jump: usize,
    },
    Direct,
}

/// The arm the lowering did not put directly after the test, and where
/// the join it reaches sits.
struct FarSide {
    arm: ArmRegion,
    join_at: usize,
    join_edges: Vec<usize>,
}

/// A run of instructions the recognizer replaces with one operation.
enum Region {
    Loop(LoopRegion),
    Diamond(DiamondRegion),
}

struct StraightRun {
    stops_at: usize,
    regions: Vec<Region>,
}

impl LoopRegion {
    fn start(&self) -> usize {
        self.enter_jump.unwrap_or(self.head)
    }

    fn end(&self) -> usize {
        self.back + 1
    }
}

impl Region {
    fn start(&self) -> usize {
        match self {
            Region::Loop(region) => region.start(),
            Region::Diamond(region) => region.jump_if,
        }
    }

    fn end(&self) -> usize {
        match self {
            Region::Loop(region) => region.end(),
            Region::Diamond(region) => region.join + 1,
        }
    }
}

/// One operation of a body before it is prepared: the instruction it is,
/// or the region or the chain it collapses.
///
/// Both collapses remove instructions, so both move every operation after
/// them. One layout decides the order once, and `op_indexes` and
/// `Prepare::emit` read the same list — a jump's target cannot disagree
/// with where the operation it names was put.
enum Unit<'r> {
    Inst(usize),
    Region(&'r Region),
    Fused(&'r FusedRegion),
    Chain(&'r ChainRun),
}

impl Unit<'_> {
    /// The instruction a jump naming this operation would name.
    fn head(&self) -> usize {
        match self {
            Unit::Inst(at) => *at,
            Unit::Region(region) => region.start(),
            Unit::Fused(region) => region.insts.start,
            Unit::Chain(run) => run.insts.end - 1,
        }
    }
}

impl Prepare<'_> {
    /// The operations of `range`, in order, with each region and each
    /// chain one operation and each entry constant none.
    fn layout<'r>(
        &self,
        range: Range<usize>,
        regions: &'r [Region],
        runs: &'r [FusedRegion],
        chains: &'r [ChainRun],
    ) -> Vec<Unit<'r>> {
        let mut units = Vec::with_capacity(range.len());
        let mut at = range.start;
        while at < range.end {
            if let Some(region) = regions.iter().find(|region| region.start() == at) {
                units.push(Unit::Region(region));
                at = region.end();
                continue;
            }
            if let Some(region) = runs.iter().find(|region| region.insts.start == at) {
                units.push(Unit::Fused(region));
                at = region.insts.end;
                continue;
            }
            if let Some(run) = chains.iter().find(|run| run.insts.start == at) {
                units.push(Unit::Chain(run));
                at = run.insts.end;
                continue;
            }
            if !self.konsts.holds_inst(at) {
                units.push(Unit::Inst(at));
            }
            at += 1;
        }
        units
    }
}

fn block_heads(units: &[Unit<'_>], split: &Split) -> FxHashMap<usize, BlockId> {
    units
        .iter()
        .enumerate()
        .map(|(at, unit)| (unit.head(), split.block_of[at]))
        .collect()
}

/// The block the unit being emitted continues into.
#[derive(Clone, Copy)]
struct Next(Option<BlockId>);

impl Next {
    fn block(self) -> BlockId {
        self.0.expect(
            "an operation continues past the last block of a body, which has to return instead",
        )
    }
}

fn targets(inst: &Inst, label: Label) -> bool {
    match &inst.kind {
        InstKind::Jump { label: named, .. } => *named == label,
        InstKind::JumpIf {
            then_label,
            else_label,
            ..
        } => *then_label == label || *else_label == label,
        _ => false,
    }
}

fn block_label(inst: &Inst) -> Option<Label> {
    match &inst.kind {
        InstKind::BlockLabel { label, .. } => Some(*label),
        _ => None,
    }
}

impl<'a> Prepare<'a> {
    fn new(
        body: &'a MirBody,
        ctx: &'a PrepareCtx<'a>,
        closures: &'a FxHashMap<Label, Arc<Code>>,
        labels: FxHashMap<Label, u32>,
    ) -> Self {
        let slots = assign_slots(body, ctx, &labels);
        let values = body.val_factory.len();
        let mut def_inst: Vec<Option<usize>> = vec![None; values];
        let mut use_counts: Vec<u32> = vec![0; values];
        for (at, inst) in body.insts.iter().enumerate() {
            for def in inst_info::defs(&inst.kind) {
                def_inst[def.to_raw()] = Some(at);
            }
            for used in inst_info::uses(&inst.kind) {
                use_counts[used.to_raw()] += 1;
            }
        }
        Self {
            body,
            ctx,
            closures,
            labels,
            scratch: slots.frame,
            slots,
            scratch_used: false,
            may_suspend: false,
            level: Level::default(),
            def_inst,
            use_counts,
            konsts: Konsts::default(),
        }
    }

    fn slot(&self, id: ValueId) -> Slot {
        let raw = match self.konsts.slot_of.get(&id) {
            Some(slot) => *slot,
            None => self.slots.of(id),
        };
        Slot::try_from(raw)
            .unwrap_or_else(|_| panic!("value {id:?} is in register {raw}, past a frame's reach"))
    }

    /// The byte displacement of `id`'s register: what every operation holds
    /// (RFC-0052 §5). `slot` is the index, and it stays inside `prepare`.
    fn off(&self, id: ValueId) -> Off {
        Off::of(self.slot(id))
    }

    /// The register `order_moves` breaks a cycle through.
    fn scratch_slot(&self) -> Slot {
        Slot::try_from(self.scratch).unwrap_or_else(|_| {
            panic!(
                "a body of {} registers has no scratch register within a frame",
                self.scratch
            )
        })
    }

    fn frame_len(&self) -> u16 {
        let len = self.scratch + u32::from(self.scratch_used);
        u16::try_from(len)
            .unwrap_or_else(|_| panic!("a body of {len} registers is past a frame's reach"))
    }

    /// Whether a value of this type holds a `Large` its register owns
    /// (RFC-0048 §4), which is the `LARGE` parameter of every operation
    /// that writes or empties a register.
    fn owns(&self, id: ValueId) -> bool {
        owns_large(self.ty(id))
    }

    fn ty(&self, id: ValueId) -> &Ty {
        self.body
            .val_types
            .get(&id)
            .unwrap_or_else(|| panic!("no type for value {id:?}"))
    }

    fn is_ref(&self, id: ValueId) -> bool {
        matches!(self.ty(id), Ty::Ref(..))
    }

    /// A call into a body — another module's or a closure's — suspends
    /// where the callee's own type puts its task above `Sync`. Read per
    /// call site; `prepare_body` asserts the join of these against
    /// `MirBody::task`, which is the checker's claim for the whole body.
    fn suspends_at(&mut self, callee_ty: &Ty) -> bool {
        let suspends = call_task(callee_ty) > Task::Sync;
        if suspends {
            self.may_suspend = true;
        }
        suspends
    }

    /// A test reads through a reference, so the storage's type decides.
    fn scrutinee_ty(&self, id: ValueId) -> &Ty {
        match self.ty(id) {
            Ty::Ref(_, inner) => &inner.ty,
            ty => ty,
        }
    }

    fn is_string(&self, id: ValueId) -> bool {
        matches!(self.ty(id), Ty::String)
    }

    fn walked_under(&self, target: &RefTarget, path: &[PathSeg]) -> Under {
        let (id, root) = match target {
            RefTarget::Var(s) | RefTarget::Param(s) => (*s, self.ty(*s)),
            RefTarget::Through(r) => (*r, self.scrutinee_ty(*r)),
        };
        Under {
            base: self.off(id),
            path: self.walked(root, path),
        }
    }

    /// The path under `root` with each step resolved against the type it
    /// stands on, and the steps that read nothing dropped (RFC-0022).
    fn walked(&self, root: &Ty, path: &[PathSeg]) -> Vec<Walked> {
        let mut at = vec![root.clone()];
        let mut kept = Vec::with_capacity(path.len());
        for seg in path {
            if let Some(resolved) = resolve_step(&at, seg) {
                kept.push(Walked {
                    step: resolved,
                    array: matches!(resolved, Step::Index(_)) && on_array(&at, seg),
                });
            }
            at = step(&at, seg);
        }
        kept
    }

    /// The kind every word-typed register is opened with when the frame is
    /// made, so that each `set_word` after it writes the word alone
    /// (RFC-0052 §5).
    ///
    /// The table is complete by construction: `assign_slots` gives a
    /// register only to values of one kind class, and a value whose type
    /// `val_types` does not hold is a refusal in `ty` naming it.
    fn slot_kinds(&self, frame_len: u16) -> Box<[SlotKind]> {
        let mut opened: Vec<Option<Kind>> = vec![None; usize::from(frame_len)];
        let mut open = |slot: Slot, kind: Kind| {
            let held = &mut opened[usize::from(slot)];
            match *held {
                None => *held = Some(kind),
                Some(other) => assert_eq!(
                    other, kind,
                    "register {slot} is opened as {other:?} and written as {kind:?}"
                ),
            }
        };
        for (id, ty) in &self.body.val_types {
            let Some(kind) = word_kind(ty) else {
                continue;
            };
            let raw = self.slots.raw(*id);
            if raw == NO_SLOT {
                continue;
            }
            let slot = Slot::try_from(raw)
                .unwrap_or_else(|_| panic!("value {id:?} is in register {raw}, past a frame"));
            open(slot, kind);
        }
        for (raw, value) in &self.konsts.value_at {
            let slot = Slot::try_from(*raw).unwrap_or_else(|_| {
                panic!("an entry constant sits in register {raw}, past a frame")
            });
            open(slot, value.kind());
        }
        opened
            .into_iter()
            .enumerate()
            .filter_map(|(slot, kind)| {
                let slot = Slot::try_from(slot).expect("a frame's registers fit a Slot");
                kind.map(|kind| SlotKind {
                    slot: Off::of(slot),
                    kind,
                })
            })
            .collect()
    }

    /// The registers of a list of operands, and the mask of the ones this
    /// operation takes the frame's claim on (RFC-0048 §5).
    fn taken(&self, ids: &[ValueId]) -> Operands {
        Operands {
            slots: ids.iter().map(|id| self.off(*id)).collect(),
            takes: self.take_mask(ids),
        }
    }

    /// The bit of every operand whose type owns a `Large`, in the frame's
    /// mark word. `Off::of` is where the width one mark word reaches is
    /// asserted, so `Off::mark` is total here.
    fn take_mask(&self, ids: &[ValueId]) -> u64 {
        let mut mask = 0u64;
        for id in ids {
            if !self.owns(*id) {
                continue;
            }
            mask |= self.off(*id).mark();
        }
        mask
    }

    fn label(&self, label: &Label) -> u32 {
        *self
            .labels
            .get(label)
            .unwrap_or_else(|| panic!("unknown label {label:?}"))
    }

    /// The block a jump naming `label` goes to, in the array being emitted.
    ///
    /// Every jump's target is decided here and nowhere else, so a block
    /// array and the terminators that name it cannot disagree.
    fn target(&self, label: &Label) -> BlockId {
        let at = self.label(label) as usize;
        *self.level.block_of_inst.get(&at).unwrap_or_else(|| {
            panic!("a jump names {label:?}, which is not the head of a block of this array")
        })
    }

    fn tag_is(&self, tag: Astr, name: &str) -> bool {
        self.ctx.interner.resolve(tag) == name
    }

    /// The moves a jump makes, as the `Mov` operations of the block they
    /// belong to, ordered so every source is read before it is overwritten;
    /// a cycle is broken through the scratch register.
    ///
    /// Each move carries its own `LARGE`, so the order is the whole of the
    /// correctness argument: there is no second run for it to cross into.
    fn move_ops(&mut self, label: &Label, args: &[ValueId]) -> Vec<Box<dyn Op>> {
        let target = self.label(label) as usize;
        let InstKind::BlockLabel { params, .. } = &self.body.insts[target].kind else {
            panic!("a jump names {label:?}, whose instruction is not a block label")
        };
        let pairs: Vec<Carried> = params
            .iter()
            .zip(args)
            .map(|(param, arg)| Carried {
                at: Pair {
                    from: self.slot(*arg),
                    to: self.slot(*param),
                },
                moved: self.moved(*arg),
            })
            .collect();
        let ordered = order_moves(pairs, self.scratch_slot());
        self.scratch_used |= ordered.scratch_used;
        ordered.moves.iter().map(mov_op).collect()
    }

    /// The block a conditional edge goes to: its own, holding the edge's
    /// `Mov`s, where it carries any; the target itself where it carries none.
    fn edge(&mut self, moves: Vec<Box<dyn Op>>, target: BlockId) -> BlockId {
        if moves.is_empty() {
            return target;
        }
        let made = BlockId::try_from(self.level.edges.len())
            .expect("a block array with more edge blocks than a BlockId counts");
        let at = self.level.blocks_len + made;
        self.level
            .edges
            .push(Block::new(moves, Box::new(control::Goto { target })));
        at
    }

    /// A suspending operation is excluded along with the terminators: it
    /// leaves the block for the driver, which the machine's dispatch loop
    /// alone can reach.
    fn is_straight_line(&self, inst: &Inst) -> bool {
        match &inst.kind {
            InstKind::Jump { .. }
            | InstKind::JumpIf { .. }
            // RFC-0051: a `Switch` is a terminator. The machine has no
            // `switch` operation yet, and `optimize::switch_expand` has
            // already replaced every one with its chain before `prepare`
            // runs, so `op` below never sees one.
            | InstKind::Switch { .. }
            | InstKind::Return { .. }
            | InstKind::Diverge
            | InstKind::Eval { .. }
            | InstKind::LoadFunction { .. }
            | InstKind::Poison { .. } => false,

            InstKind::FunctionCall {
                callee, callee_ty, ..
            } => match callee {
                Callee::Extern { id, instance } => self.ctx.extern_is_sync(id, *instance),
                Callee::Direct(_) | Callee::Indirect(_) => call_task(callee_ty) <= Task::Sync,
            },

            InstKind::Const { .. }
            | InstKind::StringConcat { .. }
            | InstKind::StringEq { .. }
            | InstKind::StringClone { .. }
            | InstKind::Ref { .. }
            | InstKind::Take { .. }
            | InstKind::Assign { .. }
            | InstKind::Fetch { .. }
            | InstKind::Commit { .. }
            | InstKind::FieldGet { .. }
            | InstKind::FieldSet { .. }
            | InstKind::BinOp { .. }
            | InstKind::UnaryOp { .. }
            | InstKind::Spawn { .. }
            | InstKind::Merge { .. }
            | InstKind::MakeArray { .. }
            | InstKind::MakeObject { .. }
            | InstKind::MakeTuple { .. }
            | InstKind::TupleIndex { .. }
            | InstKind::TestLiteral { .. }
            | InstKind::TestObjectKey { .. }
            | InstKind::ArrayIndex { .. }
            | InstKind::AsSlice { .. }
            | InstKind::Index { .. }
            | InstKind::IndexSet { .. }
            | InstKind::ObjectGet { .. }
            | InstKind::MakeClosure { .. }
            | InstKind::MakeVariant { .. }
            | InstKind::TestVariant { .. }
            | InstKind::UnwrapVariant { .. }
            | InstKind::BlockLabel { .. }
            | InstKind::Undef { .. }
            | InstKind::Nop
            | InstKind::Drop { .. } => true,
        }
    }

    fn straight_run(&self, from: usize, limit: usize) -> StraightRun {
        let insts = self.body.insts.as_slice();
        let mut regions = Vec::new();
        let mut at = from;
        while at < limit {
            if let Some(region) = self.recognize_region(at)
                && region.end() <= limit
            {
                at = region.end();
                regions.push(region);
            } else if self.is_straight_line(&insts[at]) {
                at += 1;
            } else {
                break;
            }
        }
        StraightRun {
            stops_at: at,
            regions,
        }
    }

    fn recognize_region(&self, at: usize) -> Option<Region> {
        match self.recognize_loop(at) {
            Some(region) => Some(Region::Loop(region)),
            None => self.recognize_diamond(at).map(Region::Diamond),
        }
    }

    fn references(&self, label: Label) -> Vec<usize> {
        self.body
            .insts
            .iter()
            .enumerate()
            .filter(|(_, inst)| targets(inst, label))
            .map(|(at, _)| at)
            .collect()
    }

    /// The shape `acvus_mir::lower` gives a `while`. A lowering that emits
    /// another shape does not fail here; it stops matching, and the loop
    /// prepares as the separate operations it was before.
    fn recognize_loop(&self, at: usize) -> Option<LoopRegion> {
        let insts = self.body.insts.as_slice();
        let (enter_jump, head) = match &insts.get(at)?.kind {
            InstKind::Jump { label, .. } if block_label(insts.get(at + 1)?) == Some(*label) => {
                (Some(at), at + 1)
            }
            InstKind::BlockLabel { .. } => (None, at),
            _ => return None,
        };
        let header = block_label(&insts[head])?;

        let back = match (enter_jump, self.references(header).as_slice()) {
            (None, [back]) if *back > head => *back,
            (Some(entry), [first, back]) if *first == entry && *back > head => *back,
            _ => return None,
        };
        if !matches!(insts[back].kind, InstKind::Jump { .. }) {
            return None;
        }
        let exit_label = block_label(insts.get(back + 1)?)?;

        let StraightRun {
            stops_at: jump_if,
            regions: head_regions,
        } = self.straight_run(head + 1, back);
        let InstKind::JumpIf {
            then_label,
            else_label,
            ..
        } = &insts.get(jump_if)?.kind
        else {
            return None;
        };
        if *else_label != exit_label || block_label(insts.get(jump_if + 1)?) != Some(*then_label) {
            return None;
        }
        let body_label = jump_if + 1;
        if self.references(*then_label).as_slice() != [jump_if] {
            return None;
        }

        let StraightRun {
            stops_at: body_end,
            regions: body_regions,
        } = self.straight_run(body_label + 1, back);
        if body_end != back {
            return None;
        }

        let region = LoopRegion {
            enter_jump,
            head,
            head_block: head + 1..jump_if,
            head_regions,
            jump_if,
            body_block: body_label + 1..back,
            body_regions,
            back,
        };
        self.is_closed(region.start()..region.end())
            .then_some(region)
    }

    /// The shape `acvus_mir::lower` gives an `if`. One arm has a block of
    /// its own directly after the test — `if` and `&&` put the `then` side
    /// there, `||` the `else` side — and the other either has one after
    /// that or is the test's own edge into the join.
    ///
    /// A lowering that emits another shape does not fail here; it stops
    /// matching, and the branch prepares as the separate operations it was
    /// before.
    fn recognize_diamond(&self, at: usize) -> Option<DiamondRegion> {
        let insts = self.body.insts.as_slice();
        let InstKind::JumpIf {
            then_label,
            else_label,
            ..
        } = &insts.get(at)?.kind
        else {
            return None;
        };

        let near_label = block_label(insts.get(at + 1)?)?;
        let near_is_then = near_label == *then_label;
        if !near_is_then && near_label != *else_label {
            return None;
        }
        let far_label = if near_is_then {
            *else_label
        } else {
            *then_label
        };
        if self.references(near_label).as_slice() != [at] {
            return None;
        }

        let StraightRun {
            stops_at: near_jump,
            regions: near_regions,
        } = self.straight_run(at + 2, insts.len());
        let InstKind::Jump { label: join, .. } = &insts.get(near_jump)?.kind else {
            return None;
        };
        let near = ArmRegion::Block {
            block: at + 2..near_jump,
            regions: near_regions,
            jump: near_jump,
        };

        let FarSide {
            arm: far,
            join_at,
            join_edges,
        } = if far_label == *join {
            FarSide {
                arm: ArmRegion::Direct,
                join_at: near_jump + 1,
                join_edges: vec![at, near_jump],
            }
        } else {
            if block_label(insts.get(near_jump + 1)?) != Some(far_label)
                || self.references(far_label).as_slice() != [at]
            {
                return None;
            }
            let StraightRun {
                stops_at: far_jump,
                regions: far_regions,
            } = self.straight_run(near_jump + 2, insts.len());
            let InstKind::Jump { label: other, .. } = &insts.get(far_jump)?.kind else {
                return None;
            };
            if other != join {
                return None;
            }
            FarSide {
                arm: ArmRegion::Block {
                    block: near_jump + 2..far_jump,
                    regions: far_regions,
                    jump: far_jump,
                },
                join_at: far_jump + 1,
                join_edges: vec![near_jump, far_jump],
            }
        };

        let (on_true, on_false) = if near_is_then {
            (near, far)
        } else {
            (far, near)
        };

        if block_label(insts.get(join_at)?) != Some(*join) || self.references(*join) != join_edges {
            return None;
        }

        let region = DiamondRegion {
            jump_if: at,
            on_true,
            on_false,
            join: join_at,
        };
        self.is_closed(at..join_at + 1).then_some(region)
    }

    /// No jump from outside `range` names a block inside it, so collapsing
    /// the range into one operation leaves no target behind.
    fn is_closed(&self, range: Range<usize>) -> bool {
        let insts = self.body.insts.as_slice();
        insts[range.clone()]
            .iter()
            .filter_map(block_label)
            .flat_map(|label| self.references(label))
            .all(|at| range.contains(&at))
    }

    fn regions(&self) -> Vec<Region> {
        let mut found = Vec::new();
        let mut at = 0;
        while at < self.body.insts.len() {
            match self.recognize_region(at) {
                Some(region) => {
                    at = region.end();
                    found.push(region);
                }
                None => at += 1,
            }
        }
        found
    }

    /// The body's blocks, with every jump inside it resolved against them.
    fn blocks(&mut self, range: Range<usize>, nested: &[Region]) -> Box<[Block]> {
        let runs = self.fused_in(range.clone(), nested);
        let chains = self.chains_in(range.clone(), nested);
        let units = self.layout(range, nested, &runs, &chains);
        let split = self.split(&units);
        self.level = Level {
            block_of_inst: block_heads(&units, &split),
            blocks_len: split.blocks,
            edges: Vec::new(),
        };

        let mut blocks: Vec<Block> = Vec::with_capacity(split.blocks as usize);
        let mut ops: Vec<Box<dyn Op>> = Vec::new();
        for (at, unit) in units.iter().enumerate() {
            let next = Next(match at + 1 < units.len() {
                true => Some(split.block_of[at + 1]),
                false => None,
            });
            let end = self.unit(unit, next, &mut ops);
            if end.is_some() {
                assert!(
                    split.last_of_block(at),
                    "the terminator at unit {at} is not the last unit of its block"
                );
            }
            let end = match (end, split.last_of_block(at)) {
                (Some(end), _) => end,
                (None, true) => Box::new(control::Goto {
                    target: next.block(),
                }),
                (None, false) => continue,
            };
            blocks.push(Block::new(mem::take(&mut ops), end));
        }
        assert!(
            ops.is_empty(),
            "the last block of a body was left open with {} operations",
            ops.len()
        );
        assert_eq!(
            blocks.len() as BlockId,
            split.blocks,
            "a body's blocks were emitted with a length its split did not plan"
        );
        blocks.append(&mut self.level.edges);
        assert!(!blocks.is_empty(), "a body prepared to no block at all");
        blocks.into_boxed_slice()
    }

    /// Obligation across artifacts: the assert below is unreachable because
    /// `straight_run` admits a region's part only where every instruction is
    /// straight-line, and `Next(None)` is what would panic if one of them
    /// asked for a continuation.
    fn straight(
        &mut self,
        range: Range<usize>,
        nested: &[Region],
        leaving: Vec<Box<dyn Op>>,
    ) -> Box<[Box<dyn Op>]> {
        let runs = self.fused_in(range.clone(), nested);
        let chains = self.chains_in(range.clone(), nested);
        let units = self.layout(range, nested, &runs, &chains);
        let mut ops: Vec<Box<dyn Op>> = Vec::new();
        for unit in &units {
            let end = self.unit(unit, Next(None), &mut ops);
            assert!(
                end.is_none(),
                "a region's part holds a terminator, which `straight_run` does not admit"
            );
        }
        ops.extend(leaving);
        ops.into_boxed_slice()
    }

    /// The operations this unit appends, and the terminator it is where it
    /// is one. A call that yields an RFC-0007 `Order` appends the `Merge`
    /// that writes it and then ends the block, which is why a unit hands
    /// its operations to the block rather than returning one.
    fn unit(
        &mut self,
        unit: &Unit<'_>,
        next: Next,
        ops: &mut Vec<Box<dyn Op>>,
    ) -> Option<Box<dyn Terminator>> {
        match unit {
            Unit::Inst(at) => self.op(*at, next, ops),
            Unit::Region(Region::Loop(region)) => {
                self.loop_op(region, ops);
                None
            }
            Unit::Region(Region::Diamond(region)) => {
                let op = self.diamond_op(region);
                ops.push(op);
                None
            }
            Unit::Fused(region) => {
                let op = self.fused_op(region);
                ops.push(op);
                None
            }
            Unit::Chain(run) => {
                let op = self.chain_op(run);
                ops.push(op);
                None
            }
        }
    }

    fn loop_op(&mut self, region: &LoopRegion, ops: &mut Vec<Box<dyn Op>>) {
        let body = self.body;
        let InstKind::JumpIf {
            cond,
            then_label,
            then_args,
            else_label,
            else_args,
        } = &body.insts[region.jump_if].kind
        else {
            panic!("a recognized loop's test is not a conditional jump")
        };
        let InstKind::Jump {
            label: header,
            args: back_args,
        } = &body.insts[region.back].kind
        else {
            panic!("a recognized loop's back edge is not a jump")
        };

        if let Some(entry) = region.enter_jump {
            let InstKind::Jump { label, args } = &body.insts[entry].kind else {
                panic!("a recognized loop's entry is not a jump")
            };
            let (label, args) = (*label, args.clone());
            let entering = self.move_ops(&label, &args);
            ops.extend(entering);
        }
        let into_body = self.move_ops(then_label, then_args);
        let exit = self.move_ops(else_label, else_args);
        let back = self.move_ops(header, back_args);
        let cond = self.off(*cond);

        let head = self.straight(region.head_block.clone(), &region.head_regions, Vec::new());
        let ran = self.straight(region.body_block.clone(), &region.body_regions, back);
        debug_assert_eq!(
            self.references(*then_label),
            vec![region.jump_if],
            "the move into a loop's body is placed at the head of the body, \
             which `recognize_loop` admits only where the test above is its one entry"
        );
        let mut held: Vec<Box<dyn Op>> = into_body;
        held.extend(ran);

        ops.push(Box::new(control::Loop {
            head,
            cond,
            body: held.into_boxed_slice(),
        }));
        ops.extend(exit);
    }

    fn diamond_op(&mut self, region: &DiamondRegion) -> Box<dyn Op> {
        let InstKind::JumpIf {
            cond,
            then_label,
            then_args,
            else_label,
            else_args,
        } = &self.body.insts[region.jump_if].kind
        else {
            panic!("a recognized diamond's test is not a conditional jump")
        };

        let cond = self.off(*cond);
        let on_true = self.arm(&region.on_true, then_label, then_args);
        let on_false = self.arm(&region.on_false, else_label, else_args);

        Box::new(control::Diamond {
            cond,
            on_true,
            on_false,
        })
    }

    /// `label` and `args` are the diamond's own edge into this arm, which
    /// a `Direct` arm takes all the way to the join.
    fn arm(&mut self, region: &ArmRegion, label: &Label, args: &[ValueId]) -> Box<[Box<dyn Op>]> {
        let ArmRegion::Block {
            block,
            regions,
            jump,
        } = region
        else {
            let join = self.move_ops(label, args);
            return join.into_boxed_slice();
        };
        let InstKind::Jump { label, args } = &self.body.insts[*jump].kind else {
            panic!("a recognized diamond's arm does not end in a jump")
        };
        let (label, args) = (*label, args.clone());
        let join = self.move_ops(&label, &args);
        self.straight(block.clone(), regions, join)
    }

    fn op(
        &mut self,
        at: usize,
        next: Next,
        ops: &mut Vec<Box<dyn Op>>,
    ) -> Option<Box<dyn Terminator>> {
        let body = self.body;
        let inst = &body.insts[at];
        let op: Box<dyn Op> = match &inst.kind {
            InstKind::Switch { .. } => todo!(
                "the machine has no `switch` operation yet (RFC-0051, second half); \
                 `optimize::switch_expand` replaces every Switch before prepare runs"
            ),

            // -- The terminators ----------------------------------------
            InstKind::Jump { label, args } => {
                let target = self.target(label);
                let (label, args) = (*label, args.clone());
                let moves = self.move_ops(&label, &args);
                ops.extend(moves);
                return Some(Box::new(control::Goto { target }));
            }
            InstKind::JumpIf {
                cond,
                then_label,
                then_args,
                else_label,
                else_args,
            } => {
                let cond = self.off(*cond);
                let on_true = self.target(then_label);
                let on_false = self.target(else_label);
                let then_moves = self.move_ops(then_label, then_args);
                let else_moves = self.move_ops(else_label, else_args);
                let on_true = self.edge(then_moves, on_true);
                let on_false = self.edge(else_moves, on_false);
                return Some(Box::new(control::JumpIf {
                    cond,
                    on_true,
                    on_false,
                }));
            }
            InstKind::Return { value, .. } => {
                let slot = self.off(*value);
                return Some(match word_kind(self.ty(*value)).is_some() {
                    true => Box::new(control::Return::<true> { slot }) as Box<dyn Terminator>,
                    false => Box::new(control::Return::<false> { slot }),
                });
            }
            InstKind::Diverge => return Some(Box::new(control::Diverge)),
            InstKind::Poison { .. } => return Some(Box::new(control::Poison)),

            InstKind::LoadFunction { .. } => panic!(
                "a function named as a value has no operation: the machine reaches a body \
                 only through a closure or a qualified call, and RFC-0044 leaves \
                 LoadFunction without a lowering"
            ),

            InstKind::Eval { dst, src, order } => {
                self.may_suspend = true;
                self.merge(*order, ops);
                let slot = self.off(*dst);
                let handle = self.off(*src);
                let resume = next.block();
                return Some(match self.owns(*dst) {
                    true => Box::new(call::Eval::<true> {
                        dst: slot,
                        handle,
                        next: resume,
                    }),
                    false => Box::new(call::Eval::<false> {
                        dst: slot,
                        handle,
                        next: resume,
                    }),
                });
            }
            InstKind::FunctionCall {
                dst,
                callee,
                callee_ty,
                args,
                order,
            } => {
                let site = CallSite {
                    at,
                    dst: *dst,
                    callee,
                    callee_ty,
                    args,
                    order: order.map(|edge| edge.after),
                };
                return self.call_op(&site, next, ops);
            }

            // -- The operations -----------------------------------------
            InstKind::BlockLabel { .. } | InstKind::Nop => return None,

            InstKind::Const { dst, value } => self.constant(*dst, value),

            InstKind::StringConcat { dst, parts } => {
                let owns_large = self.take_mask(parts);
                let held: Box<[ConcatPart]> = parts
                    .iter()
                    .map(|part| ConcatPart {
                        slot: self.off(*part),
                        through_reference: self.is_ref(*part),
                    })
                    .collect();
                Box::new(string::Concat {
                    dst: self.off(*dst),
                    parts: held,
                    owns_large,
                })
            }
            InstKind::StringEq { dst, a, b } => Box::new(string::StringEq {
                slots: Binary {
                    dst: self.off(*dst),
                    l: self.off(*a),
                    r: self.off(*b),
                },
            }),
            InstKind::StringClone { dst, src } => {
                let slots = Unary {
                    dst: self.off(*dst),
                    src: self.off(*src),
                };
                match self.is_ref(*src) {
                    true => Box::new(string::CloneString::<true> { slots }),
                    false => Box::new(string::CloneString::<false> { slots }),
                }
            }

            InstKind::Ref {
                dst, target, path, ..
            } => {
                let under = self.walked_under(target, path);
                let slots = Unary {
                    dst: self.off(*dst),
                    src: under.base,
                };
                make_ref(slots, through_target(target), &under.path)
            }
            InstKind::Take { dst, target, path } => {
                let clone = self.is_string(*dst);
                let through = through_target(target);
                let how = Reading::of(self.ty(*dst), through);
                let under = self.walked_under(target, path);
                let slots = Unary {
                    dst: self.off(*dst),
                    src: under.base,
                };
                match (under.path.is_empty(), through, clone) {
                    (true, false, true) => Box::new(string::CloneString::<false> { slots }),
                    (true, true, true) => Box::new(string::CloneString::<true> { slots }),
                    (true, true, false) => Box::new(storage::TakeThrough { slots }),
                    (true, false, false) => match self.owns(*dst) {
                        true => Box::new(storage::TakeVar::<true> { slots }),
                        false => Box::new(storage::TakeVar::<false> { slots }),
                    },
                    (false, _, _) => read_place(slots, how, &under.path),
                }
            }
            InstKind::Assign {
                target,
                path,
                value,
            } => {
                let through = through_target(target);
                let large = self.owns(*value);
                let under = self.walked_under(target, path);
                let slots = storage::Write {
                    target: under.base,
                    value: self.off(*value),
                };
                assign_place(slots, Writing { through, large }, &under.path)
            }
            InstKind::Fetch { dst, context } => {
                let key = self.ctx.page_key(context);
                let slot = self.off(*dst);
                match self.owns(*dst) {
                    true => Box::new(storage::Fetch::<true> { dst: slot, key }),
                    false => Box::new(storage::Fetch::<false> { dst: slot, key }),
                }
            }
            InstKind::Commit { context, value } => {
                let key = self.ctx.page_key(context);
                let src = self.off(*value);
                match self.owns(*value) {
                    true => Box::new(storage::Commit::<true> { src, key }),
                    false => Box::new(storage::Commit::<false> { src, key }),
                }
            }

            InstKind::FieldGet {
                dst,
                object,
                field,
                rest,
            } => {
                let how = Reading::of(self.ty(*dst), self.is_ref(*object));
                let slots = Unary {
                    dst: self.off(*dst),
                    src: self.off(*object),
                };
                let path: Vec<Walked> = std::iter::once(*field)
                    .chain(rest.iter().copied())
                    .map(field_step)
                    .collect();
                read_place(slots, how, &path)
            }
            InstKind::ObjectGet { dst, object, key } => {
                let how = Reading::of(self.ty(*dst), self.is_ref(*object));
                let slots = Unary {
                    dst: self.off(*dst),
                    src: self.off(*object),
                };
                read_place(slots, how, &[field_step(*key)])
            }
            InstKind::FieldSet {
                dst,
                object,
                field,
                rest,
                value,
            } => {
                let large = self.owns(*value);
                let slots = storage::Update {
                    dst: self.off(*dst),
                    object: self.off(*object),
                    value: self.off(*value),
                };
                let path: Vec<Walked> = std::iter::once(*field)
                    .chain(rest.iter().copied())
                    .map(field_step)
                    .collect();
                set_place(slots, large, &path)
            }

            InstKind::BinOp {
                dst,
                op,
                left,
                right,
            } => {
                let slots = Binary {
                    dst: self.off(*dst),
                    l: self.off(*left),
                    r: self.off(*right),
                };
                match self.ty(*left) {
                    Ty::Int(k) => arith::int_binop(*op, *k, slots),
                    Ty::Float => arith::float_binop(*op, slots),
                    Ty::Bool => arith::bool_binop(*op, slots),
                    other => panic!("binop {op:?} on {other:?}"),
                }
            }
            InstKind::UnaryOp { dst, op, operand } => {
                let slots = Unary {
                    dst: self.off(*dst),
                    src: self.off(*operand),
                };
                match self.ty(*operand) {
                    Ty::Int(k) => arith::int_unaryop(*op, *k, slots),
                    Ty::Float => arith::float_unaryop(*op, slots),
                    Ty::Bool => arith::bool_unaryop(*op, slots),
                    other => panic!("unary {op:?} on {other:?}"),
                }
            }

            InstKind::Spawn {
                dst, callee, args, ..
            } => {
                let dst = self.off(*dst);
                match callee {
                    Callee::Direct(id) => {
                        let Operands { slots, takes } = self.taken(args);
                        Box::new(call::SpawnModule {
                            dst,
                            callee: *id,
                            args: slots,
                            takes,
                        })
                    }
                    Callee::Extern { id, instance } => {
                        let handler = self.ctx.handler(id, *instance);
                        let window = self.window(at, args, ops);
                        match handler {
                            ExternHandler::Sync(f) | ExternHandler::Heavy(f) => {
                                Box::new(call::SpawnExternSync { dst, window, f })
                            }
                            ExternHandler::Async(f) => {
                                Box::new(call::SpawnExternAsync { dst, window, f })
                            }
                        }
                    }
                    Callee::Indirect(_) => panic!("spawn: indirect callee not supported"),
                }
            }
            InstKind::Merge { dst, .. } => Box::new(control::Merge {
                dst: self.off(*dst),
            }),

            InstKind::MakeArray { dst, elements } => {
                let Operands {
                    slots,
                    takes: owns_large,
                } = self.taken(elements);
                Box::new(composite::MakeArray {
                    dst: self.off(*dst),
                    elements: composite::Elements { slots, owns_large },
                })
            }
            InstKind::MakeTuple { dst, elements } => {
                let Operands {
                    slots,
                    takes: owns_large,
                } = self.taken(elements);
                Box::new(composite::MakeTuple {
                    dst: self.off(*dst),
                    elements: composite::Elements { slots, owns_large },
                })
            }
            InstKind::MakeObject { dst, fields } => {
                let values: Vec<ValueId> = fields.iter().map(|(_, value)| *value).collect();
                let owns_large = self.take_mask(&values);
                let held: Box<[FieldSlot]> = fields
                    .iter()
                    .map(|(key, value)| FieldSlot {
                        key: *key,
                        slot: self.off(*value),
                    })
                    .collect();
                Box::new(composite::MakeObject {
                    dst: self.off(*dst),
                    fields: held,
                    owns_large,
                })
            }
            InstKind::TupleIndex { dst, tuple, index } => {
                let how = Reading::of(self.ty(*dst), self.is_ref(*tuple));
                let slots = Unary {
                    dst: self.off(*dst),
                    src: self.off(*tuple),
                };
                read_place(slots, how, &[index_step(*index, false)])
            }
            InstKind::ArrayIndex { dst, array, index } => {
                let how = Reading::of(self.ty(*dst), self.is_ref(*array));
                let slots = Unary {
                    dst: self.off(*dst),
                    src: self.off(*array),
                };
                read_place(slots, how, &[index_step(*index, true)])
            }

            InstKind::TestLiteral { dst, src, value } => self.test_literal(
                Tested {
                    dst: *dst,
                    src: *src,
                },
                value,
            ),
            InstKind::TestObjectKey { dst, src, key } => {
                let slots = Unary {
                    dst: self.off(*dst),
                    src: self.off(*src),
                };
                match self.is_ref(*src) {
                    true => Box::new(pattern::TestObjectKey::<true> { slots, key: *key }),
                    false => Box::new(pattern::TestObjectKey::<false> { slots, key: *key }),
                }
            }

            // An `AsSlice` is the one call of the instance the checker
            // settled on, prepared exactly as the extern call of that
            // handler would be, so a fused run sees it as a call
            // (RFC-0044, RFC-0047 §3).
            InstKind::AsSlice {
                dst,
                container,
                instance,
                ..
            } => {
                let handler = self.ctx.handler(&instance.id, instance.instance);
                let into = self.dest(*dst);
                let args = std::slice::from_ref(container);
                match handler {
                    ExternHandler::Sync(SyncCall::Plain(abi)) => {
                        self.plain_call(at, into, args, abi, ops)
                    }
                    ExternHandler::Sync(SyncCall::Stateful { state, abi }) => {
                        self.state_call(at, into, args, state, abi, ops)
                    }
                    ExternHandler::Heavy(_) | ExternHandler::Async(_) => panic!(
                        "an AsSlice names a handler that is not synchronous, so lending a \
                         container's run would outlive the borrow it stands on (RFC-0047 §3)"
                    ),
                }
            }
            InstKind::Index {
                dst,
                slice,
                index,
                mode,
            } => index::checked(
                *mode,
                index::Read {
                    dst: self.off(*dst),
                    slice: self.off(*slice),
                    index: self.off(*index),
                },
            ),
            InstKind::IndexSet {
                slice,
                index,
                value,
            } => {
                let large = self.owns(*value);
                let (slice, index, held) = (self.off(*slice), self.off(*index), self.off(*value));
                match large {
                    true => Box::new(index::IndexSet::<true, true> {
                        slice,
                        index,
                        value: held,
                    }),
                    false => Box::new(index::IndexSet::<true, false> {
                        slice,
                        index,
                        value: held,
                    }),
                }
            }

            InstKind::MakeClosure {
                dst,
                body,
                captures,
            } => {
                let entry = self
                    .closures
                    .get(body)
                    .unwrap_or_else(|| panic!("closure body not found: {body:?}"))
                    .callable();
                let Operands { slots, takes } = self.taken(captures);
                Box::new(call::MakeClosure {
                    dst: self.off(*dst),
                    entry,
                    captures: slots,
                    takes,
                })
            }

            InstKind::MakeVariant { dst, tag, payload } => self.make_variant(*dst, *tag, *payload),
            InstKind::TestVariant { dst, src, tag } => {
                let slots = Unary {
                    dst: self.off(*dst),
                    src: self.off(*src),
                };
                let through = self.is_ref(*src);
                let form = variant_form(self.scrutinee_ty(*src));
                match (form, through) {
                    (VariantForm::Option, true) => {
                        test_option::<true>(slots, self.tag_is(*tag, "Some"))
                    }
                    (VariantForm::Option, false) => {
                        test_option::<false>(slots, self.tag_is(*tag, "Some"))
                    }
                    (VariantForm::Result, true) => {
                        test_result::<true>(slots, self.tag_is(*tag, "Ok"))
                    }
                    (VariantForm::Result, false) => {
                        test_result::<false>(slots, self.tag_is(*tag, "Ok"))
                    }
                    (VariantForm::Enum, true) => {
                        Box::new(variant::TestVariant::<true> { slots, tag: *tag })
                    }
                    (VariantForm::Enum, false) => {
                        Box::new(variant::TestVariant::<false> { slots, tag: *tag })
                    }
                }
            }
            InstKind::UnwrapVariant { dst, src } => {
                let slots = Unary {
                    dst: self.off(*dst),
                    src: self.off(*src),
                };
                match (variant_form(self.ty(*src)), self.owns(*dst)) {
                    (VariantForm::Option, true) => {
                        Box::new(variant::UnwrapOption::<true> { slots })
                    }
                    (VariantForm::Option, false) => {
                        Box::new(variant::UnwrapOption::<false> { slots })
                    }
                    (VariantForm::Result, true) => {
                        Box::new(variant::UnwrapResult::<true> { slots })
                    }
                    (VariantForm::Result, false) => {
                        Box::new(variant::UnwrapResult::<false> { slots })
                    }
                    (VariantForm::Enum, true) => Box::new(variant::UnwrapVariant::<true> { slots }),
                    (VariantForm::Enum, false) => {
                        Box::new(variant::UnwrapVariant::<false> { slots })
                    }
                }
            }

            InstKind::Undef { dst } => {
                let slot = self.off(*dst);
                match word_kind(self.ty(*dst)).is_some() {
                    true => Box::new(control::Undef::<true> { dst: slot }),
                    false => Box::new(control::Undef::<false> { dst: slot }),
                }
            }
            InstKind::Drop { src } => Box::new(control::DropValue {
                slot: self.off(*src),
            }),
        };
        ops.push(op);
        None
    }

    /// The `Order` an effectful call yields, written by its own operation
    /// ahead of the call (RFC-0007). No call instance carries the register,
    /// so no call tests for one.
    fn merge(&mut self, order: Option<ValueId>, ops: &mut Vec<Box<dyn Op>>) {
        let Some(after) = order else {
            return;
        };
        let dst = self.off(after);
        ops.push(Box::new(control::Merge { dst }));
    }

    fn call_op(
        &mut self,
        site: &CallSite<'_>,
        next: Next,
        ops: &mut Vec<Box<dyn Op>>,
    ) -> Option<Box<dyn Terminator>> {
        let CallSite {
            at,
            callee,
            callee_ty,
            args,
            ..
        } = *site;
        self.merge(site.order, ops);
        let into = self.dest(site.dst);
        let Dest {
            slot,
            large,
            word: _,
        } = into;
        match callee {
            Callee::Direct(id) => {
                let id = *id;
                let operands = self.taken(args);
                if !self.suspends_at(callee_ty) {
                    ops.push(direct_call(into, id, operands));
                    return None;
                }
                let resume = next.block();
                let Operands { slots, takes } = operands;
                Some(match large {
                    true => Box::new(call::CallDirectAsync::<true> {
                        dst: slot,
                        callee: id,
                        args: slots,
                        takes,
                        next: resume,
                    }),
                    false => Box::new(call::CallDirectAsync::<false> {
                        dst: slot,
                        callee: id,
                        args: slots,
                        takes,
                        next: resume,
                    }),
                })
            }
            Callee::Indirect(handle) => {
                let through = self.is_ref(*handle);
                let handle = self.off(*handle);
                let operands = self.taken(args);
                if !self.suspends_at(callee_ty) {
                    ops.push(indirect_call(into, through, handle, operands));
                    return None;
                }
                let resume = next.block();
                Some(indirect_call_async(into, through, handle, operands, resume))
            }
            Callee::Extern { id, instance } => {
                let handler = self.ctx.handler(id, *instance);
                if !handler.is_sync() {
                    self.may_suspend = true;
                }
                match handler {
                    ExternHandler::Sync(SyncCall::Plain(abi)) => {
                        let op = self.plain_call(at, into, args, abi, ops);
                        ops.push(op);
                        None
                    }
                    ExternHandler::Sync(SyncCall::Stateful { state, abi }) => {
                        let op = self.state_call(at, into, args, state, abi, ops);
                        ops.push(op);
                        None
                    }
                    ExternHandler::Heavy(f) => {
                        let window = self.window(at, args, ops);
                        let resume = next.block();
                        Some(match large {
                            true => Box::new(call::CallHeavy::<true> {
                                dst: slot,
                                window,
                                f,
                                next: resume,
                            }),
                            false => Box::new(call::CallHeavy::<false> {
                                dst: slot,
                                window,
                                f,
                                next: resume,
                            }),
                        })
                    }
                    ExternHandler::Async(AsyncCall::Plain(f)) => {
                        let window = self.window(at, args, ops);
                        let resume = next.block();
                        Some(match large {
                            true => Box::new(call::CallExternAsync::<true> {
                                dst: slot,
                                window,
                                f,
                                next: resume,
                            }),
                            false => Box::new(call::CallExternAsync::<false> {
                                dst: slot,
                                window,
                                f,
                                next: resume,
                            }),
                        })
                    }
                    ExternHandler::Async(AsyncCall::Stateful { state, f }) => {
                        let window = self.window(at, args, ops);
                        let resume = next.block();
                        Some(match large {
                            true => Box::new(call::CallStateAsync::<true> {
                                dst: slot,
                                window,
                                state,
                                f,
                                next: resume,
                            }),
                            false => Box::new(call::CallStateAsync::<false> {
                                dst: slot,
                                window,
                                state,
                                f,
                                next: resume,
                            }),
                        })
                    }
                }
            }
        }
    }

    fn plain_call(
        &mut self,
        at: usize,
        into: Dest,
        args: &[ValueId],
        abi: SyncAbi,
        ops: &mut Vec<Box<dyn Op>>,
    ) -> Box<dyn Op> {
        let Dest {
            slot: dst,
            large,
            word,
        } = into;
        let takes = self.take_mask(args);
        let slots: Vec<Off> = args.iter().map(|id| self.off(*id)).collect();
        match abi {
            SyncAbi::Arity0(f) => match large {
                true => Box::new(call::CallExtern0::<true> { dst, f }),
                false => Box::new(call::CallExtern0::<false> { dst, f }),
            },
            SyncAbi::Arity1(f) => {
                let a = nth(&slots, 0);
                match (large, word) {
                    (true, _) => Box::new(call::CallExtern1::<true, false> { dst, a, takes, f }),
                    (false, true) => {
                        Box::new(call::CallExtern1::<false, true> { dst, a, takes, f })
                    }
                    (false, false) => {
                        Box::new(call::CallExtern1::<false, false> { dst, a, takes, f })
                    }
                }
            }
            SyncAbi::Arity2(f) => {
                let (a, b) = (nth(&slots, 0), nth(&slots, 1));
                match large {
                    true => Box::new(call::CallExtern2::<true> {
                        dst,
                        a,
                        b,
                        takes,
                        f,
                    }),
                    false => Box::new(call::CallExtern2::<false> {
                        dst,
                        a,
                        b,
                        takes,
                        f,
                    }),
                }
            }
            SyncAbi::Arity3(f) => {
                let (a, b, c) = (nth(&slots, 0), nth(&slots, 1), nth(&slots, 2));
                match large {
                    true => Box::new(call::CallExtern3::<true> {
                        dst,
                        a,
                        b,
                        c,
                        takes,
                        f,
                    }),
                    false => Box::new(call::CallExtern3::<false> {
                        dst,
                        a,
                        b,
                        c,
                        takes,
                        f,
                    }),
                }
            }
            SyncAbi::Slice(f) => Box::new(call::CallSlice {
                dst,
                a: nth(&slots, 0),
                takes,
                f,
            }),
            SyncAbi::Window(f) => {
                let window = self.window(at, args, ops);
                match large {
                    true => Box::new(call::CallWindow::<true> { dst, window, f }),
                    false => Box::new(call::CallWindow::<false> { dst, window, f }),
                }
            }
        }
    }

    fn state_call(
        &mut self,
        at: usize,
        into: Dest,
        args: &[ValueId],
        state: acvus_extern::State,
        abi: StateAbi,
        ops: &mut Vec<Box<dyn Op>>,
    ) -> Box<dyn Op> {
        let Dest {
            slot: dst,
            large,
            word: _,
        } = into;
        let takes = self.take_mask(args);
        let slots: Vec<Off> = args.iter().map(|id| self.off(*id)).collect();
        match abi {
            StateAbi::Arity0(f) => match large {
                true => Box::new(call::CallState0::<true> { dst, state, f }),
                false => Box::new(call::CallState0::<false> { dst, state, f }),
            },
            StateAbi::Arity1(f) => {
                let a = nth(&slots, 0);
                match large {
                    true => Box::new(call::CallState1::<true> {
                        dst,
                        a,
                        takes,
                        state,
                        f,
                    }),
                    false => Box::new(call::CallState1::<false> {
                        dst,
                        a,
                        takes,
                        state,
                        f,
                    }),
                }
            }
            StateAbi::Arity2(f) => {
                let (a, b) = (nth(&slots, 0), nth(&slots, 1));
                match large {
                    true => Box::new(call::CallState2::<true> {
                        dst,
                        a,
                        b,
                        takes,
                        state,
                        f,
                    }),
                    false => Box::new(call::CallState2::<false> {
                        dst,
                        a,
                        b,
                        takes,
                        state,
                        f,
                    }),
                }
            }
            StateAbi::Arity3(f) => {
                let (a, b, c) = (nth(&slots, 0), nth(&slots, 1), nth(&slots, 2));
                match large {
                    true => Box::new(call::CallState3::<true> {
                        dst,
                        a,
                        b,
                        c,
                        takes,
                        state,
                        f,
                    }),
                    false => Box::new(call::CallState3::<false> {
                        dst,
                        a,
                        b,
                        c,
                        takes,
                        state,
                        f,
                    }),
                }
            }
            StateAbi::Slice(f) => Box::new(call::CallStateSlice {
                dst,
                a: nth(&slots, 0),
                takes,
                state,
                f,
            }),
            StateAbi::Window(f) => {
                let window = self.window(at, args, ops);
                match large {
                    true => Box::new(call::CallStateWindow::<true> {
                        dst,
                        window,
                        state,
                        f,
                    }),
                    false => Box::new(call::CallStateWindow::<false> {
                        dst,
                        window,
                        state,
                        f,
                    }),
                }
            }
        }
    }

    /// The argument run `assign_slots` placed for the call at `at`, and the
    /// `Mov` operations that put the arguments in it — which the caller
    /// pushes before the call's own operation (RFC-0052 rule 1).
    fn window(
        &mut self,
        at: usize,
        args: &[ValueId],
        ops: &mut Vec<Box<dyn Op>>,
    ) -> call::ArgWindow {
        let plan = self.slots.window(at);
        let (base, arity) = (plan.base, plan.arity);
        let moved: Vec<PendingMove> = plan.moved.clone();
        let base = Slot::try_from(base)
            .unwrap_or_else(|_| panic!("an argument run at register {base} is past a frame"));
        let arity = u16::try_from(arity)
            .unwrap_or_else(|_| panic!("a call of {arity} arguments is past a frame"));

        let pairs: Vec<Carried> = moved
            .iter()
            .copied()
            .map(|PendingMove { arg, to }| {
                let into = Slot::try_from(to).unwrap_or_else(|_| {
                    panic!("an argument moves into register {to}, which is past a frame")
                });
                Carried {
                    at: Pair {
                        from: self.slot(arg),
                        to: into,
                    },
                    moved: match self.owns(arg) {
                        true => Moved::Large,
                        false => Moved::Whole,
                    },
                }
            })
            .collect();
        let ordered = order_moves(pairs, self.scratch_slot());
        self.scratch_used |= ordered.scratch_used;

        ops.extend(ordered.moves.iter().map(mov_op));
        call::ArgWindow {
            at: Off::of(base),
            arity,
            takes: self.window_take_mask(base, args),
        }
    }

    /// The claim the frame drops when a handler is lent its run: the run's
    /// own registers, because the moves above put every argument there.
    fn window_take_mask(&self, base: Slot, args: &[ValueId]) -> u64 {
        let mut mask = 0u64;
        for (k, id) in args.iter().enumerate() {
            if !self.owns(*id) {
                continue;
            }
            let at = u16::try_from(k).expect("a call's arity fits a register index");
            let slot = base
                .checked_add(at)
                .unwrap_or_else(|| panic!("an argument run at register {base} leaves a frame"));
            assert!(
                slot < crate::regs::MAX_FRAME_SLOTS,
                "an argument run reaches register {slot}, which one frame's marks do not reach"
            );
            mask |= 1u64 << slot;
        }
        mask
    }

    /// What a move of this value between two registers the frame opened
    /// carries. `word_kind` is the same predicate `slot_kinds` opened them
    /// by, so a `Moved::Word` move writes the width every other write of
    /// those registers takes.
    fn moved(&self, arg: ValueId) -> Moved {
        if word_kind(self.ty(arg)).is_some() {
            return Moved::Word;
        }
        match self.owns(arg) {
            true => Moved::Large,
            false => Moved::Whole,
        }
    }

    /// The register an operation writes, and whether the type it writes
    /// makes the frame the owner of a `Large` (RFC-0048 §4).
    fn dest(&self, dst: ValueId) -> Dest {
        Dest {
            slot: self.off(dst),
            large: self.owns(dst),
            word: word_kind(self.ty(dst)).is_some(),
        }
    }

    fn constant(&mut self, dst: ValueId, value: &Literal) -> Box<dyn Op> {
        let out = self.off(dst);
        let word = match (value, self.ty(dst)) {
            (Literal::Int(n), Ty::Int(_)) => *n as u64,
            (Literal::Int(n), other) => panic!("integer literal {n} typed as {other:?}"),
            (Literal::Float(x), _) => x.to_bits(),
            (Literal::Bool(b), _) => u64::from(*b),
            (Literal::Unit, _) => 0,
            (Literal::String(s), _) => {
                return Box::new(constant::ConstLarge {
                    dst: out,
                    konst: Konst::Str(s.clone()),
                });
            }
            (Literal::List(items), Ty::Array(elem, _)) => {
                let konst = Konst::List(items.iter().map(|item| konst_of(item, elem)).collect());
                return Box::new(constant::ConstLarge { dst: out, konst });
            }
            (Literal::List(_), other) => panic!("list literal typed as {other:?}"),
        };
        Box::new(constant::Const { dst: out, word })
    }

    fn test_literal(&mut self, at: Tested, value: &Literal) -> Box<dyn Op> {
        let Tested { dst, src } = at;
        let through = self.is_ref(src);
        let slots = Unary {
            dst: self.off(dst),
            src: self.off(src),
        };
        match value {
            Literal::Int(n) => {
                let Ty::Int(k) = self.ty(src) else {
                    panic!("TestLiteral: an integer literal against a non-integer")
                };
                let want = *n;
                for_int_ty!(*k, |T| Box::new(pattern::TestInt::<T>::new(slots, want))
                    as Box<dyn Op>)
            }
            Literal::Float(x) => {
                let want = *x;
                match through {
                    true => Box::new(pattern::TestFloat::<true> { slots, want }),
                    false => Box::new(pattern::TestFloat::<false> { slots, want }),
                }
            }
            Literal::Bool(b) => {
                let want = *b;
                match through {
                    true => Box::new(pattern::TestBool::<true> { slots, want }),
                    false => Box::new(pattern::TestBool::<false> { slots, want }),
                }
            }
            Literal::String(s) => {
                let want = s.clone();
                match through {
                    true => Box::new(pattern::TestString::<true> { slots, want }),
                    false => Box::new(pattern::TestString::<false> { slots, want }),
                }
            }
            Literal::Unit => Box::new(pattern::TestUnit { dst: slots.dst }),
            Literal::List(_) => panic!("TestLiteral on a list literal"),
        }
    }

    fn make_variant(&mut self, dst: ValueId, tag: Astr, payload: Option<ValueId>) -> Box<dyn Op> {
        let out = self.off(dst);
        let carried = payload.map(|id| Unary {
            dst: out,
            src: self.off(id),
        });
        // An option is its payload's own value (RFC-0022), and a `Result`
        // or an enum boxes whatever it carries, so `LARGE` here is always
        // the payload's own ownership.
        let large = payload.is_some_and(|id| self.owns(id));
        match (self.ty(dst), carried) {
            (Ty::Option(_), Some(slots)) => match large {
                true => Box::new(variant::MakeSome::<true> { slots }),
                false => Box::new(variant::MakeSome::<false> { slots }),
            },
            (Ty::Option(_), None) => Box::new(variant::MakeNone { dst: out }),
            (Ty::Result(..), Some(slots)) => match (self.tag_is(tag, "Ok"), large) {
                (true, true) => Box::new(variant::MakeOk::<true> { slots }),
                (true, false) => Box::new(variant::MakeOk::<false> { slots }),
                (false, true) => Box::new(variant::MakeErr::<true> { slots }),
                (false, false) => Box::new(variant::MakeErr::<false> { slots }),
            },
            (Ty::Result(..), None) => panic!("Ok and Err carry a payload"),
            (_, Some(slots)) => match large {
                true => Box::new(variant::MakeVariant::<true> { slots, tag }),
                false => Box::new(variant::MakeVariant::<false> { slots, tag }),
            },
            (_, None) => Box::new(variant::MakeUnitVariant { dst: out, tag }),
        }
    }
}

/// The two values a `TestLiteral` names, so that neither can take the
/// other's place at the call.
#[derive(Clone, Copy)]
struct Tested {
    dst: ValueId,
    src: ValueId,
}

fn konst_of(literal: &Literal, ty: &Ty) -> Konst {
    match (literal, ty) {
        (Literal::Int(n), Ty::Int(k)) => Konst::Word(Kind::int(*k), *n as u64),
        (Literal::Int(n), other) => panic!("integer literal {n} typed as {other:?}"),
        (Literal::Float(x), _) => Konst::Word(Kind::F64, x.to_bits()),
        (Literal::Bool(b), _) => Konst::Word(Kind::Bool, u64::from(*b)),
        (Literal::Unit, _) => Konst::Word(Kind::Unit, 0),
        (Literal::String(s), _) => Konst::Str(s.clone()),
        (Literal::List(items), Ty::Array(elem, _)) => {
            Konst::List(items.iter().map(|item| konst_of(item, elem)).collect())
        }
        (Literal::List(_), other) => panic!("list literal typed as {other:?}"),
    }
}

/// The moves of a jump, all carrying a word: what a test of the ordering
/// alone states, where what each move carries is `mov_op`'s subject rather
/// than the ordering's.
#[cfg(test)]
fn carried(list: &[(Slot, Slot)]) -> Vec<Carried> {
    list.iter()
        .map(|(from, to)| Carried {
            at: Pair {
                from: *from,
                to: *to,
            },
            moved: Moved::Whole,
        })
        .collect()
}

/// One move of a parallel move, in the register indexes the ordering is
/// written in.
#[derive(Clone, Copy)]
struct Pair {
    from: Slot,
    to: Slot,
}

/// What a moved value is (RFC-0052 §5).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Moved {
    /// Both registers were opened with a kind, so the move writes the word.
    Word,
    /// The move changes the owner of a `Large`, and the mark word with it.
    Large,
    /// Neither: one whole copy, no mark.
    Whole,
}

#[derive(Clone, Copy)]
struct Carried {
    at: Pair,
    moved: Moved,
}

/// The ordered move as the operation of a block (RFC-0052 rule 1): what it
/// carries is its type, not a field a `run` reads.
fn mov_op(carried: &Carried) -> Box<dyn Op> {
    let dst = Off::of(carried.at.to);
    let src = Off::of(carried.at.from);
    match carried.moved {
        Moved::Word => Box::new(control::Mov::<false, true> { dst, src }),
        Moved::Large => Box::new(control::Mov::<true, false> { dst, src }),
        Moved::Whole => Box::new(control::Mov::<false, false> { dst, src }),
    }
}

/// A jump's moves, ordered, and whether the scratch slot carried a cycle.
struct MoveOrdering {
    moves: Vec<Carried>,
    scratch_used: bool,
}

/// Order a parallel move: every source is read before it is overwritten,
/// and a cycle is broken by moving one source into `scratch` first.
fn order_moves(pairs: Vec<Carried>, scratch: Slot) -> MoveOrdering {
    let mut pending: Vec<Carried> = pairs.into_iter().filter(|m| m.at.from != m.at.to).collect();
    let mut moves: Vec<Carried> = Vec::with_capacity(pending.len());
    let mut scratch_used = false;
    let mut scratch_holds = false;

    while !pending.is_empty() {
        let emitted = moves.len();
        let mut i = 0;
        while i < pending.len() {
            let overwrites = pending[i].at.to;
            let read_later = pending
                .iter()
                .enumerate()
                .any(|(j, m)| j != i && m.at.from == overwrites);
            if read_later {
                i += 1;
            } else {
                let m = pending.remove(i);
                scratch_holds &= m.at.from != scratch;
                moves.push(m);
            }
        }

        if moves.len() == emitted {
            assert!(
                !scratch_holds,
                "a second cycle reached the scratch slot while it still held a value"
            );
            let cycled = pending[0].at.from;
            // The scratch register is one past the registers `assign_slots`
            // handed out, so `slot_kinds` never opened its kind: both legs
            // through it are written and read whole.
            let through = match pending[0].moved {
                Moved::Large => Moved::Large,
                Moved::Word | Moved::Whole => Moved::Whole,
            };
            moves.push(Carried {
                at: Pair {
                    from: cycled,
                    to: scratch,
                },
                moved: through,
            });
            for m in pending.iter_mut().filter(|m| m.at.from == cycled) {
                m.at.from = scratch;
                m.moved = through;
            }
            scratch_used = true;
            scratch_holds = true;
        }
    }

    MoveOrdering {
        moves,
        scratch_used,
    }
}

// -- The slot assignment (RFC-0044, stage 2b) --------------------------

/// A bit per `ValueId` of one body.
#[derive(Clone, PartialEq, Eq)]
struct ValueSet(Box<[u64]>);

impl ValueSet {
    fn new(values: usize) -> Self {
        Self(vec![0; values.div_ceil(64)].into_boxed_slice())
    }

    fn contains(&self, value: usize) -> bool {
        self.0[value / 64] >> (value % 64) & 1 == 1
    }

    fn insert(&mut self, value: usize) {
        self.0[value / 64] |= 1 << (value % 64);
    }

    fn remove(&mut self, value: usize) {
        self.0[value / 64] &= !(1 << (value % 64));
    }

    fn union_with(&mut self, other: &Self) {
        for (word, bits) in self.0.iter_mut().zip(other.0.iter()) {
            *word |= bits;
        }
    }

    fn iter(&self) -> impl Iterator<Item = usize> + '_ {
        self.0.iter().enumerate().flat_map(|(word, bits)| {
            (0..64)
                .filter(move |bit| bits >> bit & 1 == 1)
                .map(move |bit| word * 64 + bit)
        })
    }
}

/// Which shape a value of `ty` carries its tag in.
enum VariantForm {
    Option,
    Result,
    Enum,
}

fn variant_form(ty: &Ty) -> VariantForm {
    match ty {
        Ty::Option(_) => VariantForm::Option,
        Ty::Result(..) => VariantForm::Result,
        Ty::Enum { .. } => VariantForm::Enum,
        other => panic!("a variant instruction on {other:?}"),
    }
}

/// The types a step can land on; none, when `ty` has no such step.
fn step_tys(ty: &Ty, seg: &PathSeg) -> Vec<Ty> {
    match (seg, ty) {
        (PathSeg::Field(f), Ty::Object(fields)) => fields.get(f).cloned().into_iter().collect(),
        (PathSeg::Index(_), Ty::Array(elem, _)) => vec![(**elem).clone()],
        (PathSeg::Index(i), Ty::Tuple(elems)) => elems.get(*i).cloned().into_iter().collect(),
        (PathSeg::Payload, Ty::Option(inner)) => vec![(**inner).clone()],
        (PathSeg::Payload, Ty::Result(ok, err)) => vec![(**ok).clone(), (**err).clone()],
        (PathSeg::Payload, Ty::Enum { variants, .. }) => {
            variants.values().flatten().map(|t| (**t).clone()).collect()
        }
        _ => Vec::new(),
    }
}

/// One step of a path over every type the walk could be at. A `Result` or
/// an `Enum` payload does not say which branch it took, so the walk carries
/// them all and the rest of the path rules out the ones it cannot belong
/// to.
fn step(at: &[Ty], seg: &PathSeg) -> Vec<Ty> {
    let next: Vec<Ty> = at.iter().flat_map(|ty| step_tys(ty, seg)).collect();
    assert!(!next.is_empty(), "a path step {seg:?} on {at:?}");
    next
}

/// The step the machine runs, or none where the type makes it a no-op: an
/// option's payload is the option's own value, unless the payload type is
/// itself an option and the depth word is what separates them (RFC-0022).
fn resolve_step(at: &[Ty], seg: &PathSeg) -> Option<Step> {
    match seg {
        PathSeg::Field(f) => Some(Step::Field(*f)),
        PathSeg::Index(i) => Some(Step::Index(*i)),
        PathSeg::Payload => payload_step(at),
    }
}

/// Whether an index step stands on an array rather than a tuple, over the
/// candidate types the walk carries. A candidate set holding both is a
/// checker gap surfaced here, not a shape to decide at run time.
fn on_array(at: &[Ty], seg: &PathSeg) -> bool {
    let live = || at.iter().filter(|ty| !step_tys(ty, seg).is_empty());
    let arrays = live().filter(|ty| matches!(ty, Ty::Array(..))).count();
    let tuples = live().filter(|ty| matches!(ty, Ty::Tuple(_))).count();
    match (arrays, tuples) {
        (0, 0) => panic!("an index step on {at:?}, which is neither an array nor a tuple"),
        (_, 0) => true,
        (0, _) => false,
        _ => panic!("an index step whose candidates hold both an array and a tuple: {at:?}"),
    }
}

fn payload_step(at: &[Ty]) -> Option<Step> {
    let shape = |ty: &Ty| match ty {
        Ty::Option(inner) if matches!(**inner, Ty::Option(_)) => Some(Step::OptionPayload),
        Ty::Option(_) => None,
        Ty::Result(..) => Some(Step::ResultPayload),
        Ty::Enum { .. } => Some(Step::VariantPayload),
        other => panic!("a payload step on {other:?}"),
    };
    let mut live = at
        .iter()
        .filter(|ty| !step_tys(ty, &PathSeg::Payload).is_empty());
    let first = live
        .next()
        .unwrap_or_else(|| panic!("a payload step on {at:?}"));
    let answer = shape(first);
    assert!(
        live.all(|ty| discriminant_of(shape(ty)) == discriminant_of(answer)),
        "a payload step reaches more than one shape: {at:?}"
    );
    answer
}

fn discriminant_of(step: Option<Step>) -> Option<mem::Discriminant<Step>> {
    step.map(|s| mem::discriminant(&s))
}

fn touch(ranges: &mut [Option<LiveRange>], value: usize, at: usize) {
    let here = LiveRange::at(at);
    ranges[value] = Some(match ranges[value] {
        Some(range) => range.joined(here),
        None => here,
    });
}

/// A closed range of instruction indexes.
#[derive(Clone, Copy)]
struct LiveRange {
    lo: usize,
    hi: usize,
}

impl LiveRange {
    fn at(index: usize) -> Self {
        Self {
            lo: index,
            hi: index,
        }
    }

    fn joined(self, other: Self) -> Self {
        Self {
            lo: self.lo.min(other.lo),
            hi: self.hi.max(other.hi),
        }
    }

    fn overlaps(self, other: Self) -> bool {
        self.lo <= other.hi && other.lo <= self.hi
    }
}

/// Where a value is written. A parameter, a capture and the order
/// parameter are written before the first instruction runs.
#[derive(Clone, Copy)]
enum DefSite {
    Entry,
    At(usize),
}

/// The jumps of a body as a control-flow graph over the linear `insts`.
struct Edges<'a> {
    insts: &'a [Inst],
    labels: &'a FxHashMap<Label, u32>,
}

impl Edges<'_> {
    fn target(&self, label: &Label) -> usize {
        *self
            .labels
            .get(label)
            .unwrap_or_else(|| panic!("unknown label {label:?}")) as usize
    }

    fn successors<F>(&self, at: usize, mut visit: F)
    where
        F: FnMut(usize),
    {
        match &self.insts[at].kind {
            InstKind::Jump { label, .. } => visit(self.target(label)),
            InstKind::JumpIf {
                then_label,
                else_label,
                ..
            } => {
                visit(self.target(then_label));
                visit(self.target(else_label));
            }
            InstKind::Return { .. } | InstKind::Diverge => {}
            _ => {
                if at + 1 < self.insts.len() {
                    visit(at + 1);
                }
            }
        }
    }

    fn block_params(&self, label: &Label) -> &[ValueId] {
        let at = self.target(label);
        let InstKind::BlockLabel { params, .. } = &self.insts[at].kind else {
            panic!("a jump names {label:?}, whose instruction is not a block label")
        };
        params
    }
}

/// Which values are live where: a backward dataflow to a fixed point.
struct Live {
    live_in: Vec<ValueSet>,
    live_out: Vec<ValueSet>,
    entry: ValueSet,
}

impl Live {
    fn of(edges: &Edges<'_>, values: usize) -> Self {
        let insts = edges.insts;
        let mut live_in = vec![ValueSet::new(values); insts.len()];
        let mut live_out = vec![ValueSet::new(values); insts.len()];

        let mut changed = true;
        while changed {
            changed = false;
            for at in (0..insts.len()).rev() {
                let mut out = ValueSet::new(values);
                edges.successors(at, |next| out.union_with(&live_in[next]));
                let mut into = out.clone();
                for def in inst_info::defs(&insts[at].kind) {
                    into.remove(def.to_raw());
                }
                for used in inst_info::uses(&insts[at].kind) {
                    into.insert(used.to_raw());
                }
                if live_out[at] != out {
                    live_out[at] = out;
                    changed = true;
                }
                if live_in[at] != into {
                    live_in[at] = into;
                    changed = true;
                }
            }
        }

        let entry = match live_in.first() {
            Some(first) => first.clone(),
            None => ValueSet::new(values),
        };
        Self {
            live_in,
            live_out,
            entry,
        }
    }

    fn after(&self, site: DefSite) -> &ValueSet {
        match site {
            DefSite::Entry => &self.entry,
            DefSite::At(at) => &self.live_out[at],
        }
    }
}

/// Two values interfere when one is live where the other is written; that
/// is the whole condition for giving them one slot, the copy between them
/// included, because a copy writes its destination where its source is
/// about to die.
struct Interference<'a> {
    def_sites: &'a [Vec<DefSite>],
    live: &'a Live,
}

impl Interference<'_> {
    fn between(&self, x: usize, y: usize) -> bool {
        let live_at_def = |defined: usize, other: usize| {
            self.def_sites[defined]
                .iter()
                .any(|site| self.live.after(*site).contains(other))
        };
        live_at_def(x, y) || live_at_def(y, x)
    }

    fn between_classes(&self, a: &[usize], b: &[usize]) -> bool {
        a.iter().any(|x| b.iter().any(|y| self.between(*x, *y)))
    }
}

/// The values a coalesced edge put in one slot, as a disjoint-set forest
/// over `ValueId`s.
struct Classes {
    parent: Vec<usize>,
    members: Vec<Vec<usize>>,
}

impl Classes {
    fn new(values: usize) -> Self {
        Self {
            parent: (0..values).collect(),
            members: (0..values).map(|value| vec![value]).collect(),
        }
    }

    fn find(&mut self, value: usize) -> usize {
        let mut root = value;
        while self.parent[root] != root {
            root = self.parent[root];
        }
        let mut at = value;
        while self.parent[at] != root {
            at = std::mem::replace(&mut self.parent[at], root);
        }
        root
    }

    fn unite(&mut self, into: usize, from: usize) {
        let members = std::mem::take(&mut self.members[from]);
        self.members[into].extend(members);
        self.parent[from] = into;
    }
}

/// One (jump argument, block parameter) pair: the move a coalesced edge
/// does not make.
struct EdgeMove {
    arg: ValueId,
    param: ValueId,
}

fn edge_moves(edges: &Edges<'_>) -> Vec<EdgeMove> {
    let mut moves = Vec::new();
    let mut edge = |label: &Label, args: &[ValueId]| {
        for (arg, param) in args.iter().zip(edges.block_params(label)) {
            moves.push(EdgeMove {
                arg: *arg,
                param: *param,
            });
        }
    };
    for inst in edges.insts {
        match &inst.kind {
            InstKind::Jump { label, args } => edge(label, args),
            InstKind::JumpIf {
                then_label,
                then_args,
                else_label,
                else_args,
                ..
            } => {
                edge(then_label, then_args);
                edge(else_label, else_args);
            }
            _ => {}
        }
    }
    moves
}

/// The arguments of the extern call or spawn at this instruction.
/// A spawn stages at every arity: its work owns its arguments past this
/// frame (RFC-0044, stage 2c).
fn window_args<'a>(inst: &'a Inst, ctx: &PrepareCtx<'_>) -> Option<&'a [ValueId]> {
    match &inst.kind {
        InstKind::FunctionCall {
            callee: Callee::Extern { id, instance },
            args,
            ..
        } => needs_window(&ctx.handler(id, *instance)).then_some(args.as_slice()),
        InstKind::Spawn {
            callee: Callee::Extern { .. },
            args,
            ..
        } => Some(args),
        _ => None,
    }
}

/// A handler that takes a slice is lent the caller's registers, so they
/// must be contiguous.
fn needs_window(handler: &ExternHandler) -> bool {
    match handler {
        ExternHandler::Sync(f) => f.arity().is_none(),
        ExternHandler::Heavy(_) | ExternHandler::Async(_) => true,
    }
}

/// What one argument position of a window holds until the call.
enum ArgPlace {
    /// The argument value itself lives here, from its definition to the
    /// call that takes it.
    Allocated { class: usize, range: LiveRange },
    /// The slot is written by a move the call carries, so it is needed
    /// only at the call.
    Moved,
}

impl ArgPlace {
    fn occupies(&self, call: usize) -> LiveRange {
        match self {
            ArgPlace::Allocated { range, .. } => *range,
            ArgPlace::Moved => LiveRange::at(call),
        }
    }
}

/// An argument that could not be allocated into its window slot, and the
/// slot the call moves it into.
#[derive(Clone, Copy)]
struct PendingMove {
    arg: ValueId,
    to: u32,
}

struct WindowPlan {
    base: u32,
    arity: u32,
    moved: Vec<PendingMove>,
}

/// The kind a register was claimed for. `Free` is a register nothing has
/// taken yet; `Held(None)` one whose values are written whole.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Claim {
    Free,
    Held(Option<Kind>),
}

/// Which ranges of instructions each frame slot is already spoken for, and
/// the kind class it was claimed for.
///
/// A register's kind byte is written once, when the frame is made
/// (RFC-0052 §5), and every `set_word` after it writes the word alone. So
/// two values may share a register only where `word_kind` gives them the
/// same answer: this is where that rule holds, and `Prepare::slot_kinds`
/// is the table it makes complete.
struct Occupancy {
    taken: Vec<Vec<LiveRange>>,
    claimed: Vec<Claim>,
}

impl Occupancy {
    fn new() -> Occupancy {
        Occupancy {
            taken: Vec::new(),
            claimed: Vec::new(),
        }
    }

    fn len(&self) -> usize {
        self.taken.len()
    }

    fn free(&self, slot: usize, range: LiveRange, want: Option<Kind>) -> bool {
        let clear = match self.taken.get(slot) {
            Some(taken) => !taken.iter().any(|held| held.overlaps(range)),
            None => true,
        };
        let fits = match self.claimed.get(slot) {
            Some(Claim::Held(held)) => *held == want,
            Some(Claim::Free) | None => true,
        };
        clear && fits
    }

    fn take(&mut self, slot: usize, range: LiveRange, want: Option<Kind>) {
        if self.taken.len() <= slot {
            self.taken.resize_with(slot + 1, Vec::new);
            self.claimed.resize(slot + 1, Claim::Free);
        }
        self.taken[slot].push(range);
        self.claimed[slot] = Claim::Held(want);
    }

    fn lowest_free(&self, range: LiveRange, want: Option<Kind>) -> usize {
        (0..self.len())
            .find(|slot| self.free(*slot, range, want))
            .unwrap_or(self.len())
    }

    /// The lowest base whose positions are each free over the range that
    /// position will hold and for the kind it will carry.
    fn lowest_free_run(&self, places: &[ArgPlace], call: usize, wants: &[Option<Kind>]) -> usize {
        let fits = |base: usize| {
            places
                .iter()
                .enumerate()
                .all(|(k, place)| self.free(base + k, place.occupies(call), wants[k]))
        };
        (0..self.len())
            .find(|base| fits(*base))
            .unwrap_or(self.len())
    }
}

struct ClassRange {
    class: usize,
    range: LiveRange,
}

/// Where each `ValueId` of a body lives, and what each extern call site's
/// argument window is (RFC-0044, stage 2b).
pub struct Slots {
    of: Box<[u32]>,
    frame: u32,
    windows: FxHashMap<usize, WindowPlan>,
}

impl Slots {
    fn of(&self, id: ValueId) -> u32 {
        let slot = self.of[id.to_raw()];
        assert!(
            slot != NO_SLOT,
            "value {id:?} is named by an operation but is neither defined nor live"
        );
        slot
    }

    /// The register table's entry as it stands, empty entry included: what
    /// `slot_kinds` walks, where `of` is what an operation asks.
    fn raw(&self, id: ValueId) -> u32 {
        self.of[id.to_raw()]
    }

    fn window(&self, call: usize) -> &WindowPlan {
        self.windows
            .get(&call)
            .unwrap_or_else(|| panic!("instruction {call} is an extern call with no window"))
    }
}

/// A value takes its coalescing partner's slot, else its argument
/// window's slot, else the lowest slot free over its live range.
///
/// A storage that a place names directly keeps one slot for the whole
/// body, and no other value joins it. Two facts about a storage are
/// outside what `ValueId` liveness can see: `storage::ref_var` builds a
/// pointer into its register, and how long that pointer is read belongs to
/// the reference's live range, not the storage's; and a write through a
/// path reads the storage it writes, which `inst_info` reports as a
/// definition alone. Until the assignment reads `analysis::loans`, the
/// conservative range is the body.
fn assign_slots(body: &MirBody, ctx: &PrepareCtx<'_>, labels: &FxHashMap<Label, u32>) -> Slots {
    let values = body.val_factory.len();
    let edges = Edges {
        insts: body.insts.as_slice(),
        labels,
    };
    let insts = edges.insts;
    let live = Live::of(&edges, values);

    let mut def_sites: Vec<Vec<DefSite>> = vec![Vec::new(); values];
    let mut pinned = vec![false; values];
    let mut entry_values: Vec<ValueId> = body
        .params
        .iter()
        .chain(&body.captures)
        .map(|(_, id)| *id)
        .collect();
    entry_values.extend(body.order_param);
    for id in &entry_values {
        def_sites[id.to_raw()].push(DefSite::Entry);
    }
    for (at, inst) in insts.iter().enumerate() {
        for def in inst_info::defs(&inst.kind) {
            def_sites[def.to_raw()].push(DefSite::At(at));
        }
        let place = match &inst.kind {
            InstKind::Ref { target, .. }
            | InstKind::Take { target, .. }
            | InstKind::Assign { target, .. } => inst_info::storage(target),
            _ => None,
        };
        if let Some(storage) = place {
            pinned[storage.to_raw()] = true;
        }
    }

    let mut ranges: Vec<Option<LiveRange>> = vec![None; values];
    for id in &entry_values {
        touch(&mut ranges, id.to_raw(), 0);
    }
    for at in 0..insts.len() {
        for value in live.live_in[at].iter().chain(live.live_out[at].iter()) {
            touch(&mut ranges, value, at);
        }
        for def in inst_info::defs(&insts[at].kind) {
            touch(&mut ranges, def.to_raw(), at);
        }
    }
    let whole_body = LiveRange {
        lo: 0,
        hi: insts.len().saturating_sub(1),
    };
    for (value, held) in pinned.iter().enumerate() {
        if *held && ranges[value].is_some() {
            ranges[value] = Some(whole_body);
        }
    }

    let kinds: Vec<Option<Kind>> = (0..values)
        .map(|value| {
            body.val_types
                .get(&ValueId::from_raw(value))
                .and_then(word_kind)
        })
        .collect();

    let mut classes = Classes::new(values);
    let interference = Interference {
        def_sites: &def_sites,
        live: &live,
    };
    for EdgeMove { arg, param } in edge_moves(&edges) {
        let (arg, param) = (arg.to_raw(), param.to_raw());
        if pinned[arg] || pinned[param] || ranges[arg].is_none() || ranges[param].is_none() {
            continue;
        }
        if kinds[arg] != kinds[param] {
            continue;
        }
        let (a, b) = (classes.find(arg), classes.find(param));
        if a != b && !interference.between_classes(&classes.members[a], &classes.members[b]) {
            classes.unite(a, b);
        }
    }

    let mut class_ranges: Vec<Option<LiveRange>> = vec![None; values];
    for value in 0..values {
        let Some(range) = ranges[value] else {
            continue;
        };
        let root = classes.find(value);
        class_ranges[root] = Some(match class_ranges[root] {
            Some(held) => held.joined(range),
            None => range,
        });
    }

    let mut class_kind: Vec<Option<Kind>> = vec![None; values];
    for value in 0..values {
        if ranges[value].is_none() {
            continue;
        }
        let root = classes.find(value);
        match class_kind[root] {
            None => class_kind[root] = kinds[value],
            Some(held) => assert_eq!(
                Some(held),
                kinds[value],
                "value {value} joined a register class opened as {held:?}"
            ),
        }
    }

    // The windows first: a contiguous run is the constrained resource.
    let mut slot_of: Vec<Option<u32>> = vec![None; values];
    let mut occupancy = Occupancy::new();
    let mut plans: Vec<(usize, WindowPlan)> = Vec::new();
    for (at, inst) in insts.iter().enumerate() {
        let Some(args) = window_args(inst, ctx) else {
            continue;
        };
        let mut allocated: Vec<usize> = Vec::new();
        let places: Vec<ArgPlace> = args
            .iter()
            .map(|arg| {
                let class = classes.find(arg.to_raw());
                let dies_here = class_ranges[class].is_some_and(|range| range.hi == at);
                let free = slot_of[class].is_none() && !allocated.contains(&class);
                match class_ranges[class] {
                    Some(range) if dies_here && free => {
                        allocated.push(class);
                        ArgPlace::Allocated { class, range }
                    }
                    _ => ArgPlace::Moved,
                }
            })
            .collect();
        let wants: Vec<Option<Kind>> = args.iter().map(|arg| kinds[arg.to_raw()]).collect();
        let base = occupancy.lowest_free_run(&places, at, &wants);

        let mut moved = Vec::new();
        for (k, (place, arg)) in places.iter().zip(args).enumerate() {
            occupancy.take(base + k, place.occupies(at), wants[k]);
            match place {
                ArgPlace::Allocated { class, .. } => slot_of[*class] = Some((base + k) as u32),
                ArgPlace::Moved => moved.push(PendingMove {
                    arg: *arg,
                    to: (base + k) as u32,
                }),
            }
        }
        plans.push((
            at,
            WindowPlan {
                base: base as u32,
                arity: args.len() as u32,
                moved,
            },
        ));
    }

    let mut rest: Vec<ClassRange> = (0..values)
        .filter(|value| classes.find(*value) == *value && slot_of[*value].is_none())
        .filter_map(|class| class_ranges[class].map(|range| ClassRange { class, range }))
        .collect();
    rest.sort_by_key(|entry| (entry.range.lo, entry.range.hi, entry.class));
    for ClassRange { class, range } in rest {
        let want = class_kind[class];
        let slot = occupancy.lowest_free(range, want);
        occupancy.take(slot, range, want);
        slot_of[class] = Some(slot as u32);
    }

    let mut of = vec![NO_SLOT; values];
    for value in 0..values {
        let root = classes.find(value);
        if let Some(slot) = slot_of[root]
            && ranges[value].is_some()
        {
            of[value] = slot;
        }
    }
    let frame = of
        .iter()
        .filter(|slot| **slot != NO_SLOT)
        .map(|slot| slot + 1)
        .max()
        .unwrap_or(0);

    let windows = plans.into_iter().collect();

    let slots = Slots {
        of: of.into_boxed_slice(),
        frame,
        windows,
    };
    #[cfg(debug_assertions)]
    check_assignment(&edges, ctx, &live, &slots);
    slots
}

/// No two values live at one program point share a slot, and no value
/// occupies a slot the call at that point writes or empties.
#[cfg(debug_assertions)]
fn check_assignment(edges: &Edges<'_>, ctx: &PrepareCtx<'_>, live: &Live, slots: &Slots) {
    let mut seen: FxHashMap<u32, usize> = FxHashMap::default();
    let mut distinct = |set: &ValueSet, at: usize, which: &str| {
        seen.clear();
        for value in set.iter() {
            let slot = slots.of[value];
            if slot == NO_SLOT {
                continue;
            }
            if let Some(other) = seen.insert(slot, value) {
                panic!(
                    "slot {slot} holds values {value} and {other}, both live {which} \
                     instruction {at}"
                );
            }
        }
    };
    for at in 0..edges.insts.len() {
        distinct(&live.live_in[at], at, "into");
        distinct(&live.live_out[at], at, "out of");
    }

    for (at, inst) in edges.insts.iter().enumerate() {
        let (Some(args), Some(window)) = (window_args(inst, ctx), slots.windows.get(&at)) else {
            continue;
        };
        let run = window.base..window.base + window.arity;
        let args: Vec<usize> = args.iter().map(|arg| arg.to_raw()).collect();
        for value in live.live_in[at].iter() {
            assert!(
                !run.contains(&slots.of[value]) || args.contains(&value),
                "value {value} is live into instruction {at} in a slot that call's \
                 argument window overwrites"
            );
        }
        for value in live.live_out[at].iter() {
            assert!(
                !run.contains(&slots.of[value]),
                "value {value} is live out of instruction {at} in a slot that call's \
                 handler emptied"
            );
        }
    }
}

#[cfg(test)]
mod recognizer_tests {
    use acvus_ast::{BinOp as AstBinOp, Span};

    use super::*;

    fn inst(kind: InstKind) -> Inst {
        Inst {
            span: Span::ZERO,
            kind,
        }
    }

    fn val(n: usize) -> ValueId {
        ValueId::from_raw(n)
    }

    fn block(label: u32) -> Inst {
        inst(InstKind::BlockLabel {
            label: Label(label),
            params: Vec::new(),
            merge_of: None,
        })
    }

    fn jump(label: u32) -> Inst {
        inst(InstKind::Jump {
            label: Label(label),
            args: Vec::new(),
        })
    }

    fn jump_if(cond: usize, then_label: u32, else_label: u32) -> Inst {
        inst(InstKind::JumpIf {
            cond: val(cond),
            then_label: Label(then_label),
            then_args: Vec::new(),
            else_label: Label(else_label),
            else_args: Vec::new(),
        })
    }

    fn add(dst: usize, left: usize, right: usize) -> Inst {
        inst(InstKind::BinOp {
            dst: val(dst),
            op: AstBinOp::Add,
            left: val(left),
            right: val(right),
        })
    }

    fn call(dst: usize, callee: QualifiedRef) -> Inst {
        inst(InstKind::FunctionCall {
            dst: val(dst),
            callee: Callee::Extern {
                id: callee,
                instance: 0,
            },
            callee_ty: Ty::Unit,
            args: Vec::new(),
            order: None,
        })
    }

    /// Every lowered body ends in one, and the preparation reads that: a
    /// body's last block is its terminator.
    fn ret(value: usize) -> Inst {
        inst(InstKind::Return {
            value: val(value),
            order: None,
        })
    }

    /// A synthetic body whose value factory covers every id its
    /// instructions mention, so the assignment sees the same range of
    /// values a lowered body would.
    fn body_of(insts: Vec<Inst>) -> MirBody {
        let highest = insts
            .iter()
            .flat_map(|inst| {
                let kind = &inst.kind;
                inst_info::defs(kind)
                    .into_iter()
                    .chain(inst_info::uses(kind))
            })
            .map(|id| id.to_raw())
            .max();
        let mut body = MirBody::new();
        for _ in 0..highest.map_or(0, |top| top + 1) {
            body.val_factory.next();
        }
        body.insts = insts;
        body
    }

    #[derive(Debug, PartialEq)]
    enum Matched {
        Loop {
            covers: Range<usize>,
            nested: Vec<Matched>,
        },
        Diamond {
            covers: Range<usize>,
            on_true: Vec<Matched>,
            on_false: Vec<Matched>,
        },
    }

    fn matched(regions: &[Region]) -> Vec<Matched> {
        regions.iter().map(one_matched).collect()
    }

    fn one_matched(region: &Region) -> Matched {
        let covers = region.start()..region.end();
        match region {
            Region::Loop(region) => Matched::Loop {
                covers,
                nested: matched(&region.head_regions)
                    .into_iter()
                    .chain(matched(&region.body_regions))
                    .collect(),
            },
            Region::Diamond(region) => Matched::Diamond {
                covers,
                on_true: matched(arm_regions(&region.on_true)),
                on_false: matched(arm_regions(&region.on_false)),
            },
        }
    }

    fn arm_regions(arm: &ArmRegion) -> &[Region] {
        match arm {
            ArmRegion::Block { regions, .. } => regions,
            ArmRegion::Direct => &[],
        }
    }

    struct Fixture {
        interner: Interner,
        externs: FxHashMap<QualifiedRef, Executable>,
    }

    impl Fixture {
        fn new() -> Self {
            Self {
                interner: Interner::new(),
                externs: FxHashMap::default(),
            }
        }

        fn async_extern(&mut self, name: &str) -> QualifiedRef {
            let id = QualifiedRef {
                namespace: None,
                name: self.interner.intern(name),
            };
            let handler = ExternHandler::Async(AsyncCall::Plain(|_, _| {
                Box::pin(async { panic!("the recognizer must not run a handler") })
            }));
            self.externs.insert(id, Executable::Extern(vec![handler]));
            id
        }

        fn sync_extern(&mut self, name: &str) -> QualifiedRef {
            let id = QualifiedRef {
                namespace: None,
                name: self.interner.intern(name),
            };
            let handler = ExternHandler::Sync(SyncCall::Plain(SyncAbi::Window(|_, _| {
                panic!("the preparation must not run a handler")
            })));
            self.externs.insert(id, Executable::Extern(vec![handler]));
            id
        }

        fn prepared(&self, insts: Vec<Inst>) -> Code {
            self.prepared_at(insts, Task::Sync)
        }

        fn prepared_at(&self, insts: Vec<Inst>, task: Task) -> Code {
            let context_names = FxHashMap::default();
            let ctx = PrepareCtx {
                interner: &self.interner,
                externs: &self.externs,
                context_names: &context_names,
            };
            let mut body = body_of(insts);
            body.task = task;
            let mentioned: Vec<ValueId> = body
                .insts
                .iter()
                .flat_map(|inst| {
                    let kind = &inst.kind;
                    inst_info::defs(kind)
                        .into_iter()
                        .chain(inst_info::uses(kind))
                })
                .collect();
            for id in mentioned {
                body.val_types.insert(id, Ty::Int(IntTy::I64));
            }
            prepare_body(&body, &ctx, &FxHashMap::default(), BodyRole::Entry)
        }

        fn recognize(&self, insts: Vec<Inst>) -> Vec<Matched> {
            let context_names = FxHashMap::default();
            let ctx = PrepareCtx {
                interner: &self.interner,
                externs: &self.externs,
                context_names: &context_names,
            };
            let closures = FxHashMap::default();
            let body = body_of(insts);
            let prep = Prepare::new(&body, &ctx, &closures, label_map(&body));
            matched(&prep.regions())
        }
    }

    fn plain_while() -> Vec<Inst> {
        vec![
            jump(0),
            block(0),
            add(1, 2, 3),
            jump_if(1, 1, 2),
            block(1),
            add(4, 5, 6),
            jump(0),
            block(2),
        ]
    }

    #[test]
    fn a_plain_while_is_one_operation() {
        let fixture = Fixture::new();
        assert_eq!(
            fixture.recognize(plain_while()),
            vec![Matched::Loop {
                covers: 0..7,
                nested: vec![]
            }]
        );
    }

    #[test]
    fn a_nested_while_is_one_operation_inside_another() {
        let fixture = Fixture::new();
        let insts = vec![
            jump(0),
            block(0),
            add(1, 2, 3),
            jump_if(1, 1, 2),
            block(1),
            jump(3),
            block(3),
            add(4, 5, 6),
            jump_if(4, 4, 5),
            block(4),
            add(7, 8, 9),
            jump(3),
            block(5),
            jump(0),
            block(2),
        ];
        assert_eq!(
            fixture.recognize(insts),
            vec![Matched::Loop {
                covers: 0..14,
                nested: vec![Matched::Loop {
                    covers: 5..12,
                    nested: vec![]
                }]
            }]
        );
    }

    #[test]
    fn an_async_extern_in_the_body_leaves_the_while_alone() {
        let mut fixture = Fixture::new();
        let suspends = fixture.async_extern("suspends");
        let mut insts = plain_while();
        insts[5] = call(4, suspends);
        assert_eq!(fixture.recognize(insts), vec![]);
    }

    #[test]
    fn a_branch_in_the_body_is_a_diamond_inside_the_while() {
        let fixture = Fixture::new();
        let insts = vec![
            jump(0),
            block(0),
            add(1, 2, 3),
            jump_if(1, 1, 2),
            block(1),
            jump_if(4, 3, 4),
            block(3),
            jump(5),
            block(4),
            jump(5),
            block(5),
            jump(0),
            block(2),
        ];
        assert_eq!(
            fixture.recognize(insts),
            vec![Matched::Loop {
                covers: 0..12,
                nested: vec![Matched::Diamond {
                    covers: 5..11,
                    on_true: vec![],
                    on_false: vec![]
                }]
            }]
        );
    }

    #[test]
    fn an_if_without_an_else_takes_the_tests_own_edge_into_the_join() {
        let fixture = Fixture::new();
        let insts = vec![jump_if(1, 3, 5), block(3), add(1, 2, 3), jump(5), block(5)];
        assert_eq!(
            fixture.recognize(insts),
            vec![Matched::Diamond {
                covers: 0..5,
                on_true: vec![],
                on_false: vec![]
            }]
        );
    }

    #[test]
    fn a_while_in_an_arm_is_one_operation_inside_the_diamond() {
        let mut fixture = Fixture::new();
        let stays = fixture.sync_extern("stays");
        let insts = vec![
            jump_if(1, 3, 4),
            block(3),
            jump(6),
            block(6),
            add(1, 2, 3),
            jump_if(1, 7, 8),
            block(7),
            call(9, stays),
            jump(6),
            block(8),
            jump(5),
            block(4),
            jump(5),
            block(5),
        ];
        assert_eq!(
            fixture.recognize(insts),
            vec![Matched::Diamond {
                covers: 0..14,
                on_true: vec![Matched::Loop {
                    covers: 2..9,
                    nested: vec![]
                }],
                on_false: vec![]
            }]
        );
    }

    #[test]
    fn an_async_extern_in_an_arm_leaves_the_branch_alone() {
        let mut fixture = Fixture::new();
        let suspends = fixture.async_extern("suspends");
        let insts = vec![
            jump_if(1, 3, 4),
            block(3),
            call(9, suspends),
            jump(5),
            block(4),
            jump(5),
            block(5),
        ];
        assert_eq!(fixture.recognize(insts), vec![]);
    }

    #[test]
    fn a_jump_into_an_arm_from_outside_leaves_the_branch_alone() {
        let fixture = Fixture::new();
        let insts = vec![
            jump_if(1, 3, 4),
            block(3),
            add(1, 2, 3),
            jump(5),
            block(4),
            jump(5),
            block(5),
            jump(3),
        ];
        assert_eq!(fixture.recognize(insts), vec![]);
    }

    #[test]
    fn a_body_of_arithmetic_and_a_synchronous_extern_cannot_suspend() {
        let mut fixture = Fixture::new();
        let stays = fixture.sync_extern("stays");
        let insts = vec![add(1, 2, 3), call(4, stays), ret(4)];
        assert!(!fixture.prepared(insts).may_suspend());
    }

    #[test]
    fn a_body_that_calls_an_asynchronous_extern_can_suspend() {
        let mut fixture = Fixture::new();
        let suspends = fixture.async_extern("suspends");
        let insts = vec![add(1, 2, 3), call(4, suspends), ret(4)];
        assert!(fixture.prepared_at(insts, Task::Async).may_suspend());
    }

    /// The type decides, not the call shape: an indirect call through a
    /// closure whose effect is Sync does not suspend, and the same call
    /// through one whose effect is Async does (RFC-0046).
    #[test]
    fn a_closure_call_suspends_where_the_closure_type_says_so() {
        let indirect = |task| {
            vec![
                inst(InstKind::FunctionCall {
                    dst: val(1),
                    callee: Callee::Indirect(val(2)),
                    callee_ty: Ty::Fn {
                        params: Vec::new(),
                        ret: Box::new(Ty::Unit),
                        captures: Vec::new(),
                        effect: acvus_mir::ty::EffectTerm::Known(
                            acvus_mir::ty::Effect::OPAQUE.at_task(task),
                        ),
                    },
                    args: Vec::new(),
                    order: None,
                }),
                ret(1),
            ]
        };
        let fixture = Fixture::new();
        assert!(
            !fixture
                .prepared_at(indirect(Task::Sync), Task::Sync)
                .may_suspend()
        );
        assert!(
            fixture
                .prepared_at(indirect(Task::Async), Task::Async)
                .may_suspend()
        );
    }

    #[test]
    fn a_jump_into_the_loop_from_outside_leaves_it_alone() {
        let fixture = Fixture::new();
        let mut insts = plain_while();
        insts.push(jump(1));
        assert_eq!(fixture.recognize(insts), vec![]);
    }
}

#[cfg(test)]
mod assignment_tests {
    use acvus_ast::{BinOp as AstBinOp, Span};

    use super::*;

    fn val(n: usize) -> ValueId {
        ValueId::from_raw(n)
    }

    fn inst(kind: InstKind) -> Inst {
        Inst {
            span: Span::ZERO,
            kind,
        }
    }

    fn block(label: u32, params: &[usize]) -> Inst {
        inst(InstKind::BlockLabel {
            label: Label(label),
            params: params.iter().copied().map(val).collect(),
            merge_of: None,
        })
    }

    fn jump(label: u32, args: &[usize]) -> Inst {
        inst(InstKind::Jump {
            label: Label(label),
            args: args.iter().copied().map(val).collect(),
        })
    }

    fn add(dst: usize, left: usize, right: usize) -> Inst {
        inst(InstKind::BinOp {
            dst: val(dst),
            op: AstBinOp::Add,
            left: val(left),
            right: val(right),
        })
    }

    fn konst(dst: usize) -> Inst {
        inst(InstKind::Const {
            dst: val(dst),
            value: Literal::Int(1),
        })
    }

    fn ret(value: usize) -> Inst {
        inst(InstKind::Return {
            value: val(value),
            order: None,
        })
    }

    static SYMBOLS: std::sync::LazyLock<Interner> = std::sync::LazyLock::new(Interner::new);

    const BY_VALUE: &str = "by_value";
    const WINDOW: &str = "window";

    fn extern_ref(name: &str) -> QualifiedRef {
        QualifiedRef {
            namespace: None,
            name: SYMBOLS.intern(name),
        }
    }

    fn call(dst: usize, name: &str, args: &[usize]) -> Inst {
        inst(InstKind::FunctionCall {
            dst: val(dst),
            callee: Callee::Extern {
                id: extern_ref(name),
                instance: 0,
            },
            callee_ty: Ty::Unit,
            args: args.iter().copied().map(val).collect(),
            order: None,
        })
    }

    /// One extern of each ABI: `by_value` takes its one argument in a
    /// register, `window` is lent a slice.
    fn externs() -> FxHashMap<QualifiedRef, Executable> {
        let by_value = ExternHandler::Sync(SyncCall::Plain(SyncAbi::Arity1(|_, v| v)));
        let window = ExternHandler::Sync(SyncCall::Plain(SyncAbi::Window(|_, args| args[0])));
        [
            (extern_ref(BY_VALUE), Executable::Extern(vec![by_value])),
            (extern_ref(WINDOW), Executable::Extern(vec![window])),
        ]
        .into_iter()
        .collect()
    }

    fn assign(insts: Vec<Inst>) -> Slots {
        let mut body = MirBody::new();
        let highest = insts
            .iter()
            .flat_map(|inst| {
                let kind = &inst.kind;
                inst_info::defs(kind)
                    .into_iter()
                    .chain(inst_info::uses(kind))
            })
            .map(|id| id.to_raw())
            .max();
        for _ in 0..highest.map_or(0, |top| top + 1) {
            body.val_factory.next();
        }
        body.insts = insts;
        let labels = label_map(&body);
        let externs = externs();
        let context_names = FxHashMap::default();
        let ctx = PrepareCtx {
            interner: &SYMBOLS,
            externs: &externs,
            context_names: &context_names,
        };
        assign_slots(&body, &ctx, &labels)
    }

    #[test]
    fn a_jump_edge_coalesces_to_no_move() {
        let slots = assign(vec![konst(0), jump(0, &[0]), block(0, &[1]), ret(1)]);
        assert_eq!(slots.of(val(0)), slots.of(val(1)));
    }

    #[test]
    fn a_swap_edge_keeps_its_scratch_cycle() {
        let slots = assign(vec![
            konst(0),
            konst(1),
            jump(0, &[0, 1]),
            block(0, &[2, 3]),
            add(4, 2, 3),
            jump(0, &[3, 2]),
        ]);
        let fits = |raw: u32| Slot::try_from(raw).expect("the fixture's frame fits a Slot");
        let (a, b) = (fits(slots.of(val(2))), fits(slots.of(val(3))));
        assert_ne!(a, b, "a swap's two parameters cannot share one slot");
        let pairs = carried(&[(b, a), (a, b)]);
        assert!(order_moves(pairs, fits(slots.frame)).scratch_used);
    }

    #[test]
    fn an_argument_that_dies_at_the_call_is_allocated_into_the_window() {
        let slots = assign(vec![konst(0), call(1, WINDOW, &[0]), ret(1)]);
        let window = slots.window(1);
        assert!(window.moved.is_empty());
        assert_eq!(slots.of(val(0)), window.base);
    }

    #[test]
    fn an_argument_used_after_the_call_is_moved_into_the_window() {
        let slots = assign(vec![konst(0), call(1, WINDOW, &[0]), add(2, 0, 1), ret(2)]);
        let window = slots.window(1);
        let arg = slots.of(val(0));
        assert_ne!(arg, window.base);
        let moved: Vec<(u32, u32)> = window
            .moved
            .iter()
            .map(|m| (slots.of(m.arg), m.to))
            .collect();
        assert_eq!(moved, vec![(arg, window.base)]);
    }

    #[test]
    fn a_by_value_call_constrains_no_slot() {
        let slots = assign(vec![
            konst(0),
            call(1, BY_VALUE, &[0]),
            add(2, 0, 1),
            ret(2),
        ]);
        assert!(
            slots.windows.is_empty(),
            "a handler that takes its argument by value needs no window"
        );
    }
}

#[cfg(test)]
mod move_ordering_tests {
    use super::*;

    fn emitted(ordering: &MoveOrdering) -> Vec<(Slot, Slot)> {
        ordering
            .moves
            .iter()
            .map(|m| (m.at.from, m.at.to))
            .collect()
    }

    #[test]
    fn a_source_is_read_before_it_is_overwritten() {
        let ordering = order_moves(carried(&[(0, 1), (1, 2)]), 9);
        assert_eq!(emitted(&ordering), vec![(1, 2), (0, 1)]);
        assert!(!ordering.scratch_used);
    }

    #[test]
    fn a_cycle_goes_through_the_scratch_slot() {
        let ordering = order_moves(carried(&[(0, 1), (1, 0)]), 9);
        assert_eq!(emitted(&ordering), vec![(0, 9), (1, 0), (9, 1)]);
        assert!(ordering.scratch_used);
    }

    #[test]
    fn a_self_move_is_no_move() {
        let ordering = order_moves(carried(&[(3, 3)]), 9);
        assert!(ordering.moves.is_empty());
    }

    #[test]
    fn two_cycles_reuse_the_one_scratch_slot() {
        let ordering = order_moves(carried(&[(0, 1), (1, 0), (2, 3), (3, 2)]), 9);
        assert!(ordering.scratch_used);
        assert_eq!(ordering.moves.len(), 6);
    }
}

// -- The arithmetic-chain recognizer (RFC-0044, stage 4) ----------------

enum Tree {
    Leaf(Off),
    /// The right operand of a `Neg`, which the node's operator never
    /// reads. It still becomes a leaf offset, because the generic tier
    /// reads both operands before it knows the operator, so the offset
    /// has to be one the operand space holds.
    Unread,
    Node {
        op: Arith,
        left: Box<Tree>,
        right: Box<Tree>,
    },
}

impl Tree {
    fn word(&self, out: &mut String) {
        match self {
            Tree::Leaf(_) | Tree::Unread => out.push('L'),
            Tree::Node { left, right, .. } => {
                out.push('N');
                left.word(out);
                right.word(out);
            }
        }
    }

    fn flatten_into(&self, out: &mut Flattened) {
        match self {
            Tree::Leaf(slot) => out.leaves.push(Some(*slot)),
            Tree::Unread => out.leaves.push(None),
            Tree::Node { op, left, right } => {
                left.flatten_into(out);
                right.flatten_into(out);
                out.ops.push(*op);
            }
        }
    }
}

#[derive(Default)]
struct Flattened {
    ops: Vec<Arith>,
    leaves: Vec<Option<Off>>,
}

/// A chain's root: the operator it applies and the two subtrees below it.
struct RootNode {
    op: Root,
    left: Tree,
    right: Tree,
}

impl RootNode {
    fn word(&self) -> String {
        let mut word = String::from("N");
        self.left.word(&mut word);
        self.right.word(&mut word);
        word
    }

    fn flatten(&self) -> Flattened {
        let mut out = Flattened::default();
        self.left.flatten_into(&mut out);
        self.right.flatten_into(&mut out);
        out
    }
}

/// A chain the recognizer matched: the instructions it replaces, and the
/// tree that replaces them.
struct ChainRun {
    /// The instructions of the body this chain stands for, contiguous and
    /// ending at its root.
    insts: Range<usize>,
    dst: ValueId,
    ty: ChainTy,
    node: RootNode,
}

impl ChainRun {
    /// How many instructions the root absorbed. A chain that absorbed none
    /// is one operation written a longer way, so the recognizer does not
    /// make it.
    fn absorbed(&self) -> usize {
        self.insts.len() - 1
    }

    /// The kind the chain's root writes: the operand type's for an
    /// arithmetic root, `Bool` for a comparison.
    fn kind(&self) -> Kind {
        match self.node.op {
            Root::Cmp(_) => Kind::Bool,
            Root::Num(_) => match self.ty {
                ChainTy::Int(k) => Kind::int(k),
                ChainTy::Float => Kind::F64,
            },
        }
    }

    fn plan(&self, remap: &dyn Fn(Off) -> Off) -> Plan {
        let word = self.node.word();
        let shape = Shape::of_word(&word).unwrap_or_else(|| {
            panic!(
                "a chain of preorder shape {word} has more than {} nodes",
                ChainBounds::MAX_NODES
            )
        });

        let Flattened { ops: found, leaves } = self.node.flatten();
        assert_eq!(
            found.len(),
            shape.interior(),
            "shape {shape:?} and its operators disagree"
        );
        assert_eq!(
            leaves.len(),
            shape.leaves(),
            "shape {shape:?} and its leaves disagree"
        );

        let read = leaves
            .iter()
            .flatten()
            .next()
            .copied()
            .unwrap_or_else(|| panic!("a chain of shape {shape:?} reads no register"));
        let in_space = |at: Off| ChainBounds::byte_offset_of_word(remap(at));

        let mut ops = [Arith::Add; ChainBounds::MAX_INTERIOR];
        ops[..found.len()].copy_from_slice(&found);
        let mut offsets = [in_space(read); ChainBounds::MAX_LEAVES];
        for (offset, leaf) in offsets.iter_mut().zip(&leaves) {
            *offset = in_space(leaf.unwrap_or(read));
        }

        Plan {
            shape,
            root: self.node.op,
            ops,
            leaves: offsets,
        }
    }
}

/// The operator a binary operator becomes inside a chain.
///
/// Exhaustive over `BinOp`: the operators the chain claims are the five
/// arithmetic ones. A comparison is a chain's root, not an interior node,
/// and the bitwise and shift operators and the boolean connectives are not
/// claimed at all — an instruction carrying one stops the recognizer.
fn arith_of(op: BinOp) -> Option<Arith> {
    match op {
        BinOp::Add => Some(Arith::Add),
        BinOp::Sub => Some(Arith::Sub),
        BinOp::Mul => Some(Arith::Mul),
        BinOp::Div => Some(Arith::Div),
        BinOp::Mod => Some(Arith::Rem),
        BinOp::Eq
        | BinOp::Neq
        | BinOp::Lt
        | BinOp::Gt
        | BinOp::Lte
        | BinOp::Gte
        | BinOp::BitAnd
        | BinOp::BitOr
        | BinOp::Xor
        | BinOp::Shl
        | BinOp::Shr
        | BinOp::And
        | BinOp::Or => None,
    }
}

/// The comparison a chain's root applies, for the operators that are one.
///
/// Exhaustive over `BinOp` for the same reason `arith_of` is.
fn compare_of(op: BinOp) -> Option<Compare> {
    match op {
        BinOp::Lt => Some(Compare::Lt),
        BinOp::Lte => Some(Compare::Le),
        BinOp::Gt => Some(Compare::Gt),
        BinOp::Gte => Some(Compare::Ge),
        BinOp::Eq => Some(Compare::Eq),
        BinOp::Neq => Some(Compare::Ne),
        BinOp::Add
        | BinOp::Sub
        | BinOp::Mul
        | BinOp::Div
        | BinOp::Mod
        | BinOp::BitAnd
        | BinOp::BitOr
        | BinOp::Xor
        | BinOp::Shl
        | BinOp::Shr
        | BinOp::And
        | BinOp::Or => None,
    }
}

/// The type a chain runs at, for the types a chain runs at.
fn chain_ty(ty: &Ty) -> Option<ChainTy> {
    match ty {
        Ty::Int(k) => Some(ChainTy::Int(*k)),
        Ty::Float => Some(ChainTy::Float),
        _ => None,
    }
}

/// The state of one descent from a chain's root.
struct Growing<'a> {
    prep: &'a Prepare<'a>,
    ty: ChainTy,
    /// The lowest instruction this descent may absorb. A retry raises it
    /// past whatever broke the run's contiguity.
    floor: usize,
    root: usize,
    absorbed: Vec<usize>,
    /// The operator nodes committed so far, the root included.
    nodes: usize,
}

impl Growing<'_> {
    /// Whether the instruction at `at` may become part of this chain: it is
    /// inside the window, its value is read once and only by this chain,
    /// and it is an arithmetic operation or a constant at the chain's type.
    fn absorbable(&self, value: ValueId) -> Option<usize> {
        let at = self.prep.def_at(value)?;
        if at < self.floor || at >= self.root || self.prep.use_count(value) != 1 {
            return None;
        }
        let kind = &self.prep.body.insts[at].kind;
        match kind {
            InstKind::BinOp { op, left, .. } => {
                let same = chain_ty(self.prep.ty(*left)) == Some(self.ty);
                (same && arith_of(*op).is_some()).then_some(at)
            }
            InstKind::UnaryOp { op, operand, .. } => {
                let same = chain_ty(self.prep.ty(*operand)) == Some(self.ty);
                (same && matches!(op, UnaryOp::Neg)).then_some(at)
            }
            _ => None,
        }
    }

    /// How many operator nodes absorbing `value` whole would add.
    fn nodes_of(&self, value: ValueId) -> usize {
        let Some(at) = self.absorbable(value) else {
            return 0;
        };
        match &self.prep.body.insts[at].kind {
            InstKind::BinOp { left, right, .. } => 1 + self.nodes_of(*left) + self.nodes_of(*right),
            InstKind::UnaryOp { operand, .. } => 1 + self.nodes_of(*operand),
            other => panic!("absorbable admitted {other:?}, which is not a chain step"),
        }
    }

    /// The subtree for `value`. An operand the chain does not absorb — one
    /// read twice, one outside the window, one of another type, or one
    /// whose subtree would carry the chain past `Chain::MAX_NODES` — is a
    /// leaf reading the register the operation that computes it writes.
    fn emit(&mut self, value: ValueId) -> Tree {
        let admitted = self
            .absorbable(value)
            .filter(|_| self.nodes + self.nodes_of(value) <= ChainBounds::MAX_NODES);
        let Some(at) = admitted else {
            return Tree::Leaf(self.prep.leaf_slot(value));
        };
        self.absorbed.push(at);
        match &self.prep.body.insts[at].kind {
            InstKind::BinOp {
                op, left, right, ..
            } => {
                let (op, left, right) = (*op, *left, *right);
                let op = arith_of(op)
                    .unwrap_or_else(|| panic!("absorbable admitted {op:?}, which is not a node"));
                self.nodes += 1;
                let left = self.emit(left);
                let right = self.emit(right);
                Tree::Node {
                    op,
                    left: Box::new(left),
                    right: Box::new(right),
                }
            }
            InstKind::UnaryOp { operand, .. } => {
                let operand = *operand;
                self.nodes += 1;
                let left = self.emit(operand);
                let right = Tree::Unread;
                Tree::Node {
                    op: Arith::Neg,
                    left: Box::new(left),
                    right: Box::new(right),
                }
            }
            other => panic!("absorbable admitted {other:?}, which is not a chain step"),
        }
    }
}

/// `None` is a type whose register is written whole every time, so the
/// frame writes no kind ahead of it and no operation on it is a `set_word`
/// (RFC-0052 §5).
fn word_kind(ty: &Ty) -> Option<Kind> {
    match ty {
        Ty::Int(k) => Some(Kind::int(*k)),
        Ty::Float => Some(Kind::F64),
        Ty::Bool => Some(Kind::Bool),
        Ty::Unit => Some(Kind::Unit),
        _ => None,
    }
}

/// Where a body's blocks begin and end (RFC-0052 §1). A block begins at the
/// entry, at every label a jump names, and after every terminator; it ends
/// at its terminator, or falls through to the next block with a `Goto`.
struct Split {
    /// The block the unit at this index belongs to.
    block_of: Vec<BlockId>,
    blocks: u32,
}

impl Split {
    /// Whether the unit at this index is the last one of its block.
    fn last_of_block(&self, at: usize) -> bool {
        at + 1 == self.block_of.len() || self.block_of[at + 1] != self.block_of[at]
    }
}

impl Prepare<'_> {
    fn unit_terminates(&self, unit: &Unit<'_>) -> bool {
        match unit {
            Unit::Inst(at) => !self.is_straight_line(&self.body.insts[*at]),
            Unit::Region(_) | Unit::Fused(_) | Unit::Chain(_) => false,
        }
    }

    fn split(&self, units: &[Unit<'_>]) -> Split {
        let mut starts = vec![false; units.len()];
        if !units.is_empty() {
            starts[0] = true;
        }
        for (at, unit) in units.iter().enumerate() {
            if let Unit::Inst(inst) = unit
                && let Some(label) = block_label(&self.body.insts[*inst])
                && !self.references(label).is_empty()
            {
                starts[at] = true;
            }
            if self.unit_terminates(unit) && at + 1 < units.len() {
                starts[at + 1] = true;
            }
        }
        let mut block_of = Vec::with_capacity(units.len());
        let mut blocks = 0;
        for start in &starts {
            if *start {
                blocks += 1;
            }
            block_of.push(blocks - 1);
        }
        Split { block_of, blocks }
    }
}

/// The word a numeric literal of an inline numeric type becomes, for the
/// literals and types a register can hold as one word.
fn konst_value(literal: &Literal, ty: &Ty) -> Option<Value> {
    match (literal, ty) {
        (Literal::Int(n), Ty::Int(k)) => Some(Value::inline(Kind::int(*k), *n as u64)),
        (Literal::Float(x), Ty::Float) => Some(Value::inline(Kind::F64, x.to_bits())),
        (Literal::Int(_), _)
        | (Literal::Float(_), _)
        | (Literal::Bool(_), _)
        | (Literal::Unit, _)
        | (Literal::String(_), _)
        | (Literal::List(_), _) => None,
    }
}

impl Prepare<'_> {
    /// Whether every instruction that reads `value` reads it as a word out
    /// of a fixed register: an arithmetic operand, or an argument of a call
    /// the fusion rule admits. A window call is neither, and its argument
    /// registers are the ones `assign_slots` placed.
    fn every_reader_takes_a_word(&self, value: ValueId) -> bool {
        self.body.insts.iter().enumerate().all(|(at, inst)| {
            let reads = inst_info::uses(&inst.kind)
                .iter()
                .any(|used| *used == value);
            !reads
                || matches!(inst.kind, InstKind::BinOp { .. } | InstKind::UnaryOp { .. })
                || self.fusable_call(at).is_some()
        })
    }

    fn hoist_konsts(&mut self) {
        for at in 0..self.body.insts.len() {
            let InstKind::Const { dst, value } = &self.body.insts[at].kind else {
                continue;
            };
            let (dst, value) = (*dst, value.clone());
            let Some(word) = konst_value(&value, self.ty(dst)) else {
                continue;
            };
            if !self.every_reader_takes_a_word(dst) {
                continue;
            }
            let slot = self.scratch;
            self.scratch += 1;
            self.konsts.slot_of.insert(dst, slot);
            self.konsts.value_at.insert(slot, word);
            self.konsts.insts.push(at);
        }
    }

    fn entry_konsts(&self) -> Box<[EntryKonst]> {
        let mut found: Vec<EntryKonst> = self
            .konsts
            .value_at
            .iter()
            .map(|(slot, value)| EntryKonst {
                slot: Off::of(Slot::try_from(*slot).unwrap_or_else(|_| {
                    panic!("an entry constant sits in register {slot}, past a frame")
                })),
                value: *value,
            })
            .collect();
        found.sort_by_key(|konst| konst.slot);
        found.into_boxed_slice()
    }
}

/// One extern call the fusion rule admits: synchronous, its arguments in
/// the call's own words rather than a window, and carrying no RFC-0007
/// order edge for the run to reorder.
struct FusableCall<'a> {
    dst: ValueId,
    args: &'a [ValueId],
    abi: SyncAbi,
}

/// A run of extern calls the recognizer matched, as indexes into
/// `MirBody::insts`. A hoisted literal may sit inside `insts` between two
/// of them; it is no operation, so it does not break the run.
struct FusedRegion {
    insts: Range<usize>,
    calls: Vec<usize>,
    deref: Option<usize>,
}

impl FusedRegion {
    /// A lone call with no deref is the operation `call_extern_k` already
    /// is, written through one more indirection, so it is not a run.
    fn is_run(&self) -> bool {
        self.calls.len() > 1 || (self.calls.len() == 1 && self.deref.is_some())
    }
}

impl<'a> Prepare<'a> {
    fn fusable_call(&self, at: usize) -> Option<FusableCall<'a>> {
        let (dst, id, instance, args) = match &self.body.insts.get(at)?.kind {
            InstKind::FunctionCall {
                dst,
                callee: Callee::Extern { id, instance },
                args,
                order: None,
                ..
            } => (*dst, id, *instance, args.as_slice()),
            InstKind::AsSlice {
                dst,
                container,
                instance: ExternInstance { id, instance },
                ..
            } => (*dst, id, *instance, std::slice::from_ref(container)),
            _ => return None,
        };
        // A stateful instance has no `call::Call` variant: a fused run
        // holds bare `fn` pointers, and the state would be a field the run
        // has nowhere to put (RFC-0044, stage 6).
        let ExternHandler::Sync(SyncCall::Plain(abi)) = self.ctx.handler(id, instance) else {
            return None;
        };
        let arity = match abi {
            SyncAbi::Arity0(_) => 0,
            SyncAbi::Arity1(_) | SyncAbi::Slice(_) => 1,
            SyncAbi::Arity2(_) => 2,
            SyncAbi::Arity3(_) | SyncAbi::Window(_) => return None,
        };
        if arity != args.len() {
            return None;
        }
        Some(FusableCall { dst, args, abi })
    }

    /// The `*r` closing a run whose last call left `last`: the read of a
    /// word through that reference, which the run holds instead of the
    /// register `take_through` would have read it from.
    fn fusable_deref(&self, at: usize, last: ValueId) -> Option<Deref> {
        let InstKind::Take { dst, target, path } = &self.body.insts.get(at)?.kind else {
            return None;
        };
        let RefTarget::Through(source) = target else {
            return None;
        };
        if *source != last || !self.walked(self.scrutinee_ty(*source), path).is_empty() {
            return None;
        }
        let read: Deref = if self.is_string(*dst) {
            storage::deref_word::<true>
        } else {
            storage::deref_word::<false>
        };
        Some(read)
    }

    /// The run starting at `at`: calls while each one's result is read
    /// exactly once and by the next call, then the deref that may close it.
    fn fused_at(&self, at: usize, limit: usize) -> Option<FusedRegion> {
        let mut calls = Vec::new();
        let mut deref = None;
        let mut previous: Option<ValueId> = None;
        let mut index = at;
        let mut end = at;
        while index < limit {
            if self.konsts.holds_inst(index) {
                index += 1;
                continue;
            }
            if let Some(last) = previous
                && self.use_count(last) == 1
                && self.fusable_deref(index, last).is_some()
            {
                deref = Some(index);
                end = index + 1;
                break;
            }
            if calls.len() == call::MAX_CALLS {
                break;
            }
            let Some(call) = self.fusable_call(index) else {
                break;
            };
            if let Some(held) = previous
                && (self.use_count(held) != 1 || !call.args.contains(&held))
            {
                break;
            }
            previous = Some(call.dst);
            calls.push(index);
            index += 1;
            end = index;
        }

        let region = FusedRegion {
            insts: at..end,
            calls,
            deref,
        };
        region.is_run().then_some(region)
    }

    fn fused_runs(&self, window: Range<usize>) -> Vec<FusedRegion> {
        let mut found = Vec::new();
        let mut at = window.start;
        while at < window.end {
            match self.fused_at(at, window.end) {
                Some(region) => {
                    at = region.insts.end;
                    found.push(region);
                }
                None => at += 1,
            }
        }
        found
    }

    /// Every fused run of `range`, in the windows the regions of `nested`
    /// leave between them.
    fn fused_in(&self, range: Range<usize>, nested: &[Region]) -> Vec<FusedRegion> {
        let mut found = Vec::new();
        let mut at = range.start;
        while at < range.end {
            if let Some(region) = nested.iter().find(|region| region.start() == at) {
                at = region.end();
                continue;
            }
            let stop = nested
                .iter()
                .map(Region::start)
                .find(|start| *start > at)
                .unwrap_or(range.end);
            found.extend(self.fused_runs(at..stop));
            at = stop;
        }
        found
    }

    fn fused_op(&mut self, region: &FusedRegion) -> Box<dyn Op> {
        let mut calls: SmallVec<[call::Call; 2]> = SmallVec::new();
        let mut read: Vec<ValueId> = Vec::new();
        let mut previous: Option<ValueId> = None;
        for at in &region.calls {
            let found = self
                .fusable_call(*at)
                .expect("a recognized run holds a fusable call at every index it named");
            let mut args = [call::PREVIOUS; 2];
            for (word, id) in args.iter_mut().zip(found.args) {
                *word = match previous {
                    Some(held) if *id == held => call::PREVIOUS,
                    Some(_) | None => {
                        read.push(*id);
                        self.off(*id)
                    }
                };
            }
            previous = Some(found.dst);
            calls.push(match found.abi {
                SyncAbi::Arity0(f) => call::Call::Nullary { f },
                SyncAbi::Arity1(f) => call::Call::Unary { f, a: args[0] },
                SyncAbi::Arity2(f) => call::Call::Binary {
                    f,
                    a: args[0],
                    b: args[1],
                },
                SyncAbi::Slice(f) => call::Call::Slice { f, a: args[0] },
                SyncAbi::Arity3(_) | SyncAbi::Window(_) => {
                    panic!("fusable_call admitted an arity a fused run holds no shape for")
                }
            });
        }
        let last = previous.expect("a recognized run holds at least one call");

        let (dst, tail) = match region.deref {
            Some(at) => {
                let InstKind::Take { dst, .. } = &self.body.insts[at].kind else {
                    panic!("a recognized run's deref is not a take")
                };
                let dst = *dst;
                let read = self
                    .fusable_deref(at, last)
                    .expect("a recognized run's deref is the one the recognizer matched");
                (dst, Some(read))
            }
            None => (last, None),
        };

        let takes = self.take_mask(&read);
        call::fused(self.owns(dst), self.off(dst), calls, tail, takes)
    }

    fn def_at(&self, value: ValueId) -> Option<usize> {
        self.def_inst[value.to_raw()]
    }

    fn use_count(&self, value: ValueId) -> u32 {
        self.use_counts[value.to_raw()]
    }

    fn leaf_slot(&self, value: ValueId) -> Off {
        self.off(value)
    }

    /// The chain rooted at `root`, if there is one: the descent absorbs
    /// what it can, and a retry raises the floor past anything that broke
    /// the run's contiguity, until the absorbed instructions are exactly
    /// the window below the root.
    fn chain_at(&self, window: Range<usize>, root: usize) -> Option<ChainRun> {
        let kind = &self.body.insts[root].kind;
        let (ty, dst) = match kind {
            InstKind::BinOp { dst, op, left, .. } => {
                let ty = chain_ty(self.ty(*left))?;
                match (arith_of(*op), compare_of(*op)) {
                    (Some(_), None) | (None, Some(_)) => {}
                    (None, None) => return None,
                    (Some(_), Some(_)) => panic!("{op:?} is both a chain node and a chain's root"),
                }
                (ty, *dst)
            }
            InstKind::UnaryOp {
                dst, op, operand, ..
            } if matches!(op, UnaryOp::Neg) => (chain_ty(self.ty(*operand))?, *dst),
            _ => return None,
        };

        let mut floor = window.start;
        loop {
            let mut growing = Growing {
                prep: self,
                ty,
                floor,
                root,
                absorbed: Vec::new(),
                nodes: 1,
            };
            let node = match &self.body.insts[root].kind {
                InstKind::BinOp {
                    op, left, right, ..
                } => {
                    let root_op = match (arith_of(*op), compare_of(*op)) {
                        (Some(op), None) => Root::Num(op),
                        (None, Some(how)) => Root::Cmp(how),
                        _ => panic!("the root changed kind"),
                    };
                    let left = growing.emit(*left);
                    let right = growing.emit(*right);
                    RootNode {
                        op: root_op,
                        left,
                        right,
                    }
                }
                InstKind::UnaryOp { operand, .. } => {
                    let left = growing.emit(*operand);
                    let right = Tree::Unread;
                    RootNode {
                        op: Root::Num(Arith::Neg),
                        left,
                        right,
                    }
                }
                other => panic!("a chain root is not an arithmetic instruction: {other:?}"),
            };

            let lowest = growing.absorbed.iter().copied().min().unwrap_or(root);
            let gap = (lowest..root)
                .rev()
                .find(|at| !growing.absorbed.contains(at) && !self.konsts.holds_inst(*at));
            if let Some(at) = gap {
                floor = at + 1;
                continue;
            }

            assert!(
                growing.nodes <= ChainBounds::MAX_NODES,
                "a chain of {} nodes was prepared, past the tree it runs on",
                growing.nodes
            );
            return Some(ChainRun {
                insts: lowest..root + 1,
                dst,
                ty,
                node,
            });
        }
    }

    /// Every chain of a straight-line window, latest root first so each one
    /// is maximal, returned lowest first.
    fn chain_runs(&self, window: Range<usize>) -> Vec<ChainRun> {
        let mut found: Vec<ChainRun> = Vec::new();
        let mut root = window.end;
        while root > window.start {
            root -= 1;
            let Some(run) = self.chain_at(window.start..root + 1, root) else {
                continue;
            };
            if run.absorbed() == 0 {
                continue;
            }
            root = run.insts.start;
            found.push(run);
        }
        found.reverse();
        found
    }

    fn chain_op(&mut self, run: &ChainRun) -> Box<dyn Op> {
        self.check_chain(run);
        chain::chain_op(run.ty, self.off(run.dst), run.plan(&|slot| slot))
    }

    /// Every chain of `range`, in the straight-line windows the regions of
    /// `nested` leave between them.
    fn chains_in(&self, range: Range<usize>, nested: &[Region]) -> Vec<ChainRun> {
        let mut found = Vec::new();
        let mut at = range.start;
        while at < range.end {
            if let Some(region) = nested.iter().find(|region| region.start() == at) {
                at = region.end();
                continue;
            }
            let stop = nested
                .iter()
                .map(Region::start)
                .find(|start| *start > at)
                .unwrap_or(range.end);
            found.extend(self.chain_runs(at..stop));
            at = stop;
        }
        found
    }

    /// Every register a chain reads is live where the chain runs and
    /// carries the chain's own type — the chain-shaped half of what
    /// `check_assignment` states for slots.
    fn check_chain(&self, run: &ChainRun) {
        for leaf in run.node.flatten().leaves {
            let Some(slot) = leaf else {
                continue;
            };
            assert!(
                (slot.index() as u32) < self.scratch + u32::from(self.scratch_used),
                "a chain reads register {}, which its body's frame does not have",
                slot.index()
            );
            let read = (self.body.insts[run.insts.start..run.insts.end])
                .iter()
                .flat_map(|inst| inst_info::uses(&inst.kind))
                .find(|value| self.leaf_slot(*value) == slot)
                .unwrap_or_else(|| {
                    panic!(
                        "a chain reads register {}, which no instruction it replaces reads",
                        slot.index()
                    )
                });
            assert_eq!(
                chain_ty(self.ty(read)),
                Some(run.ty),
                "a chain reads register {} at another type than its own",
                slot.index()
            );
        }
    }
}

/// Where `konst` sits in the operand space's constant tail, appending it
/// if this is the first leaf to read it.
fn intern(konsts: &mut Vec<Value>, konst: &Value) -> usize {
    let same = |held: &Value| held.kind() == konst.kind() && held.bits() == konst.bits();
    match konsts.iter().position(same) {
        Some(at) => at,
        None => {
            konsts.push(*konst);
            konsts.len() - 1
        }
    }
}

impl Prepare<'_> {
    /// The body as a `Code::Expr`, when it is exactly `params -> one chain
    /// -> return` or `params -> return`.
    ///
    /// A body with captures is not one: a capture reaches the body as
    /// `Value::reference`, which a chain leaf cannot read as an inline
    /// word. A body with an order parameter is not one either: it has an
    /// effect to sequence, and a chain has none.
    fn expression_body(&mut self) -> Option<Expr> {
        let body = self.body;
        if !body.captures.is_empty() || body.order_param.is_some() {
            return None;
        }
        let (last, rest) = body.insts.split_last()?;
        let InstKind::Return { value, order: None } = &last.kind else {
            return None;
        };
        let value = *value;
        let arity = u32::try_from(body.params.len()).expect("a body has at most u32::MAX params");
        let at_of = |target: ValueId| -> Option<u16> {
            let at = body.params.iter().position(|(_, id)| *id == target)?;
            Some(u16::try_from(at).expect("a body has at most u16::MAX params"))
        };

        if rest.is_empty() {
            return Some(Expr {
                arity,
                body: ExprBody::Argument(at_of(value)?),
                span: last.span,
            });
        }

        let run = self.chain_at(0..rest.len(), rest.len() - 1)?;
        if run.insts.end != rest.len() || run.dst != value {
            return None;
        }
        if !(0..run.insts.start).all(|at| self.konsts.holds_inst(at)) {
            return None;
        }

        let mut konsts: Vec<Value> = Vec::new();
        let mut in_space: FxHashMap<Off, Off> = FxHashMap::default();
        for leaf in run.node.flatten().leaves.iter().flatten() {
            let at = match body
                .params
                .iter()
                .position(|(_, id)| *leaf == self.off(*id))
            {
                Some(at) => at,
                None => {
                    let raw = u32::try_from(leaf.index())
                        .expect("a frame's register index fits the konst table's key");
                    let konst = self.konsts.value_at.get(&raw)?;
                    body.params.len() + intern(&mut konsts, konst)
                }
            };
            let at = u16::try_from(at).expect("a body reads at most u16::MAX operands");
            in_space.insert(*leaf, Off::of(at));
        }
        if body.params.len() + konsts.len() > ExprChain::MAX_OPERANDS {
            return None;
        }

        self.check_chain(&run);
        let plan = run.plan(&|at| {
            *in_space.get(&at).unwrap_or_else(|| {
                panic!(
                    "register {} is neither a parameter nor a constant",
                    at.index()
                )
            })
        });
        let eval = chain::chain_eval(run.ty, &plan);
        Some(Expr {
            arity,
            body: ExprBody::Chain(ExprChain {
                kind: run.kind(),
                plan,
                eval,
                konsts: konsts.into_boxed_slice(),
            }),
            span: last.span,
        })
    }
}

// -- What an instruction's operands become -----------------------------

/// The registers an operation's operands sit in, and the frame's claim on
/// the ones it consumes (RFC-0048 §5).
struct Operands {
    slots: Box<[Off]>,
    takes: u64,
}

/// The register an operation writes, and whether the frame owns a `Large`
/// once it has.
#[derive(Clone, Copy)]
struct Dest {
    slot: Off,
    large: bool,
    /// `word_kind`, the same predicate `slot_kinds` opened the register by.
    word: bool,
}

/// A place under a register: the register the walk starts at, and the
/// resolved path to the place.
struct Under {
    base: Off,
    path: Vec<Walked>,
}

/// The three facts a read of a place is chosen by, each off a type:
/// RFC-0026 for `clone`, RFC-0018 for `owned`.
#[derive(Clone, Copy)]
struct Reading {
    through: bool,
    clone: bool,
    owned: bool,
}

impl Reading {
    fn of(ty: &Ty, through: bool) -> Self {
        Self {
            through,
            clone: matches!(ty, Ty::String),
            owned: owns_large(ty),
        }
    }
}

/// The two facts a write to a place is chosen by: whether the base register
/// holds a reference, and whether the written value owns a `Large`.
#[derive(Clone, Copy)]
struct Writing {
    through: bool,
    large: bool,
}

/// One `InstKind::FunctionCall`, as the emitters read it. The IR gives the
/// call as an enum variant, which has no type of its own to pass.
struct CallSite<'i> {
    at: usize,
    dst: ValueId,
    callee: &'i Callee,
    callee_ty: &'i Ty,
    args: &'i [ValueId],
    order: Option<ValueId>,
}

/// One resolved path step, and — for an index — whether the type it stands
/// on is an array, which picks `ops::storage::Index<ARRAY>`.
///
/// `code::Step` does not carry `array`: a boxed walk asks the value, which
/// is the decision `ops::storage::walk` records rather than instantiate the
/// whole storage family once per path shape a program writes.
#[derive(Clone, Copy)]
struct Walked {
    step: Step,
    array: bool,
}

fn field_step(field: Astr) -> Walked {
    Walked {
        step: Step::Field(field),
        array: false,
    }
}

fn index_step(index: usize, array: bool) -> Walked {
    Walked {
        step: Step::Index(index),
        array,
    }
}

fn through_target(target: &RefTarget) -> bool {
    matches!(target, RefTarget::Through(_))
}

fn steps_of(path: &[Walked]) -> Box<[Step]> {
    path.iter().map(|walked| walked.step).collect()
}

fn nth(slots: &[Off], k: usize) -> Off {
    *slots.get(k).unwrap_or_else(|| {
        panic!(
            "an extern of arity {} is called with {} arguments",
            k + 1,
            slots.len()
        )
    })
}

/// Whether a register holding a value of this type owns a `Large` the frame
/// has to release (RFC-0048 §4).
fn owns_large(ty: &Ty) -> bool {
    match ty {
        Ty::Int(_) | Ty::Float | Ty::Bool | Ty::Unit | Ty::Never | Ty::Order => false,
        // A reference is a word. The run an `AsSlice` boxed is not: it
        // reaches its register as the erased `Elements` (RFC-0047 §6).
        Ty::Ref(_, target) => matches!(target.ty, Ty::Slice(_)),
        // RFC-0022: an option is its payload's own value, so it owns what
        // the payload owns and nothing else.
        Ty::Option(inner) => owns_large(inner),
        Ty::String
        | Ty::Array(..)
        | Ty::Object(_)
        | Ty::Tuple(_)
        | Ty::Result(..)
        | Ty::Enum { .. }
        | Ty::Fn { .. }
        | Ty::Handle(_)
        | Ty::UserDefined { .. } => true,
        Ty::Slice(_) => panic!(
            "a slice has no storage of its own: it reaches a register under a reference \
             (RFC-0047)"
        ),
        Ty::Error(_) | Ty::Var(_) => panic!("prepare reached the unresolved type {ty:?}"),
    }
}

// -- One step as the `Segment` type that reaches it --------------------

/// Runs `$body` with `$seg` bound to the `Segment` value of one resolved
/// step, so that the operation built from it holds no `match` of its own.
macro_rules! at_step {
    ($walked:expr, |$seg:ident| $body:expr) => {{
        let walked: Walked = $walked;
        match (walked.step, walked.array) {
            (Step::Field(f), _) => {
                let $seg = storage::Field(f);
                $body
            }
            (Step::Index(i), true) => {
                let $seg = storage::Index::<true>(i);
                $body
            }
            (Step::Index(i), false) => {
                let $seg = storage::Index::<false>(i);
                $body
            }
            (Step::OptionPayload, _) => {
                let $seg = storage::OptionPayload;
                $body
            }
            (Step::ResultPayload, _) => {
                let $seg = storage::ResultPayload;
                $body
            }
            (Step::VariantPayload, _) => {
                let $seg = storage::VariantPayload;
                $body
            }
        }
    }};
}

fn make_ref_step<const THROUGH: bool>(slots: Unary, one: Walked) -> Box<dyn Op> {
    at_step!(
        one,
        |step| Box::new(storage::MakeRefStep::<_, THROUGH> { slots, step }) as Box<dyn Op>
    )
}

fn read_step<M>(slots: Unary, one: Walked) -> Box<dyn Op>
where
    M: storage::Reads,
{
    at_step!(one, |step| Box::new(storage::ReadStep::<_, M> {
        slots,
        step,
        mode: PhantomData,
    }) as Box<dyn Op>)
}

fn assign_step<const THROUGH: bool, const LARGE: bool>(
    slots: storage::Write,
    one: Walked,
) -> Box<dyn Op> {
    at_step!(
        one,
        |step| Box::new(storage::AssignStep::<_, THROUGH, LARGE> { slots, step }) as Box<dyn Op>
    )
}

fn set_step<const LARGE: bool>(slots: storage::Update, one: Walked) -> Box<dyn Op> {
    at_step!(
        one,
        |step| Box::new(storage::SetStep::<_, LARGE> { slots, step }) as Box<dyn Op>
    )
}

// -- The place operations ----------------------------------------------

fn make_ref(slots: Unary, through: bool, path: &[Walked]) -> Box<dyn Op> {
    match (through, path) {
        (false, []) => Box::new(storage::MakeRef::<false> { slots }),
        (true, []) => Box::new(storage::MakeRef::<true> { slots }),
        (false, [one]) => make_ref_step::<false>(slots, *one),
        (true, [one]) => make_ref_step::<true>(slots, *one),
        (false, many) => Box::new(storage::MakeRefPath::<false> {
            slots,
            steps: steps_of(many),
        }),
        (true, many) => Box::new(storage::MakeRefPath::<true> {
            slots,
            steps: steps_of(many),
        }),
    }
}

fn read_place(slots: Unary, how: Reading, path: &[Walked]) -> Box<dyn Op> {
    assert!(
        !path.is_empty(),
        "a read of a place with no step is a read of the register itself"
    );
    assert!(
        !(how.owned && !how.clone && how.through),
        "a part is moved out of a storage, never through a borrow of one"
    );
    match how {
        Reading {
            clone: false,
            owned: true,
            ..
        } => read_at::<storage::Moved>(slots, path),
        Reading {
            clone: true,
            through: false,
            ..
        } => read_at::<storage::Cloned<false>>(slots, path),
        Reading {
            clone: true,
            through: true,
            ..
        } => read_at::<storage::Cloned<true>>(slots, path),
        Reading { through: false, .. } => read_at::<storage::Copied<false>>(slots, path),
        Reading { through: true, .. } => read_at::<storage::Copied<true>>(slots, path),
    }
}

fn read_at<M>(slots: Unary, path: &[Walked]) -> Box<dyn Op>
where
    M: storage::Reads,
{
    match path {
        [one] => read_step::<M>(slots, *one),
        many => Box::new(storage::ReadPath::<M> {
            slots,
            steps: steps_of(many),
            mode: PhantomData,
        }),
    }
}

fn assign_place(slots: storage::Write, how: Writing, path: &[Walked]) -> Box<dyn Op> {
    match (how.through, how.large, path) {
        (false, false, []) => Box::new(storage::AssignVar::<false> { slots }),
        (false, true, []) => Box::new(storage::AssignVar::<true> { slots }),
        (true, false, []) => Box::new(storage::AssignThrough::<false> { slots }),
        (true, true, []) => Box::new(storage::AssignThrough::<true> { slots }),
        (false, false, [one]) => assign_step::<false, false>(slots, *one),
        (false, true, [one]) => assign_step::<false, true>(slots, *one),
        (true, false, [one]) => assign_step::<true, false>(slots, *one),
        (true, true, [one]) => assign_step::<true, true>(slots, *one),
        (false, false, many) => Box::new(storage::AssignPath::<false, false> {
            slots,
            steps: steps_of(many),
        }),
        (false, true, many) => Box::new(storage::AssignPath::<false, true> {
            slots,
            steps: steps_of(many),
        }),
        (true, false, many) => Box::new(storage::AssignPath::<true, false> {
            slots,
            steps: steps_of(many),
        }),
        (true, true, many) => Box::new(storage::AssignPath::<true, true> {
            slots,
            steps: steps_of(many),
        }),
    }
}

fn set_place(slots: storage::Update, large: bool, path: &[Walked]) -> Box<dyn Op> {
    assert!(
        !path.is_empty(),
        "a field set with no step names no field to write"
    );
    match (large, path) {
        (false, [one]) => set_step::<false>(slots, *one),
        (true, [one]) => set_step::<true>(slots, *one),
        (false, many) => Box::new(storage::SetPath::<false> {
            slots,
            steps: steps_of(many),
        }),
        (true, many) => Box::new(storage::SetPath::<true> {
            slots,
            steps: steps_of(many),
        }),
    }
}

// -- The variant tests -------------------------------------------------

fn test_option<const THROUGH: bool>(slots: Unary, some: bool) -> Box<dyn Op> {
    match some {
        true => Box::new(variant::TestOption::<THROUGH, true> { slots }),
        false => Box::new(variant::TestOption::<THROUGH, false> { slots }),
    }
}

fn test_result<const THROUGH: bool>(slots: Unary, ok: bool) -> Box<dyn Op> {
    match ok {
        true => Box::new(variant::TestResult::<THROUGH, true> { slots }),
        false => Box::new(variant::TestResult::<THROUGH, false> { slots }),
    }
}

/// The task the checker settled for a called body, read off the callee
/// register's type (RFC-0046). A type with no effect is `Sync`.
fn call_task(callee_ty: &Ty) -> Task {
    callee_ty.effect().map_or(Task::Sync, |effect| effect.task)
}

/// The `(large, word)` pair every call's result store is picked by, the same
/// three forms `CallExtern1` has: a `Large` the frame takes ownership of, a
/// word whose kind the frame opened, or neither.
fn direct_call(into: Dest, callee: QualifiedRef, operands: Operands) -> Box<dyn Op> {
    let Dest {
        slot: dst,
        large,
        word,
    } = into;
    let Operands { slots: args, takes } = operands;
    match (large, word) {
        (true, _) => Box::new(call::CallDirect::<true, false> {
            dst,
            callee,
            args,
            takes,
        }),
        (false, true) => Box::new(call::CallDirect::<false, true> {
            dst,
            callee,
            args,
            takes,
        }),
        (false, false) => Box::new(call::CallDirect::<false, false> {
            dst,
            callee,
            args,
            takes,
        }),
    }
}

fn indirect_call(into: Dest, through: bool, callee: Off, operands: Operands) -> Box<dyn Op> {
    let Dest {
        slot: dst,
        large,
        word,
    } = into;
    let Operands { slots: args, takes } = operands;
    match (large, word, through) {
        (true, _, false) => Box::new(call::CallIndirect::<true, false, false> {
            dst,
            callee,
            args,
            takes,
        }),
        (true, _, true) => Box::new(call::CallIndirect::<true, false, true> {
            dst,
            callee,
            args,
            takes,
        }),
        (false, true, false) => Box::new(call::CallIndirect::<false, true, false> {
            dst,
            callee,
            args,
            takes,
        }),
        (false, true, true) => Box::new(call::CallIndirect::<false, true, true> {
            dst,
            callee,
            args,
            takes,
        }),
        (false, false, false) => Box::new(call::CallIndirect::<false, false, false> {
            dst,
            callee,
            args,
            takes,
        }),
        (false, false, true) => Box::new(call::CallIndirect::<false, false, true> {
            dst,
            callee,
            args,
            takes,
        }),
    }
}

fn indirect_call_async(
    into: Dest,
    through: bool,
    callee: Off,
    operands: Operands,
    next: BlockId,
) -> Box<dyn Terminator> {
    let Dest {
        slot: dst,
        large,
        word: _,
    } = into;
    let Operands { slots: args, takes } = operands;
    match (large, through) {
        (false, false) => Box::new(call::CallIndirectAsync::<false, false> {
            dst,
            callee,
            args,
            takes,
            next,
        }),
        (false, true) => Box::new(call::CallIndirectAsync::<false, true> {
            dst,
            callee,
            args,
            takes,
            next,
        }),
        (true, false) => Box::new(call::CallIndirectAsync::<true, false> {
            dst,
            callee,
            args,
            takes,
            next,
        }),
        (true, true) => Box::new(call::CallIndirectAsync::<true, true> {
            dst,
            callee,
            args,
            takes,
            next,
        }),
    }
}
