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

use std::mem;
use std::ops::Range;
use std::sync::Arc;

use acvus_ast::{BinOp, Literal, Span, UnaryOp};
use acvus_mir::analysis::inst_info;
use acvus_mir::graph::QualifiedRef;
use acvus_mir::ir::{
    Callee, ExternInstance, IndexMode, Inst, InstKind, Label, MirBody, MirModule, PathSeg,
    RefTarget, ValueId,
};
use acvus_mir::ty::{IntTy, Task, Ty};
use acvus_utils::{Astr, Interner, LocalIdOps};
use rustc_hash::FxHashMap;

use crate::code::{
    ArgWindow, Arith, BasicBlock, Body, Chain, Code, Compare, ConcatPart, Deref, Diamond,
    DiamondArm, EntryKonst, Expr, ExprBody, ExprChain, ExternArgs, ExternCall, FieldSlot,
    FusedCall, FusedRun, Konst, LoopBody, NO_SLOT, Op, OpFn, PREVIOUS, Payload, Prepared, Root,
    Shape, SlotMove, Step,
};
use crate::interpreter::Executable;
use crate::ops::arith::{self, for_int_ty};
use crate::ops::chain::{self, ChainTy};
use crate::ops::{call, composite, constant, control, index, pattern, storage, string, variant};
use crate::runtime::{ExternHandler, SyncHandler};
use crate::value::{Kind, Value};

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
    let runs = prep.fused_in(0..body.insts.len(), &regions);
    let chains = prep.chains_in(0..body.insts.len(), &regions);
    let units = prep.layout(0..body.insts.len(), &regions, &runs, &chains);
    prep.op_index = op_indexes(body.insts.len(), &units);

    if let BodyRole::Closure = role
        && let Some(expr) = prep.expression_body()
    {
        return Code::Expr(expr);
    }

    let emitted = prep.emit(&units);
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
    let mut ops: Vec<Op> = Vec::with_capacity(emitted.len());
    let mut spans: Vec<Span> = Vec::with_capacity(emitted.len());
    for Emitted { op, span } in emitted {
        ops.push(op);
        spans.push(span);
    }

    let frame_len = prep.scratch + u32::from(prep.scratch_used);
    let params = body.params.iter().map(|(_, v)| prep.slot(*v)).collect();
    let captures = body.captures.iter().map(|(_, v)| prep.slot(*v)).collect();
    let order_param = body.order_param.map(|id| prep.slot(id));
    let entry_konsts = prep.entry_konsts();

    Code::Body(Body {
        ops: ops.into_boxed_slice(),
        spans: spans.into_boxed_slice(),
        payloads: prep.payloads.into_boxed_slice(),
        frame_len,
        entry_konsts,
        may_suspend,
        params,
        captures,
        order_param,
    })
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
    payloads: Vec<Payload>,
    scratch: u32,
    scratch_used: bool,
    may_suspend: bool,
    op_index: Vec<Option<u32>>,
    def_inst: Vec<Option<usize>>,
    use_counts: Vec<u32>,
    konsts: Konsts,
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

fn op_indexes(len: usize, units: &[Unit<'_>]) -> Vec<Option<u32>> {
    let mut index = vec![None; len];
    for (op, unit) in units.iter().enumerate() {
        index[unit.head()] = Some(op as u32);
    }
    index
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
            payloads: Vec::new(),
            scratch_used: false,
            may_suspend: false,
            op_index: Vec::new(),
            def_inst,
            use_counts,
            konsts: Konsts::default(),
        }
    }

    fn slot(&self, id: ValueId) -> u32 {
        match self.konsts.slot_of.get(&id) {
            Some(slot) => *slot,
            None => self.slots.of(id),
        }
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
    fn suspends_at(&mut self, callee_ty: &Ty) {
        let task = callee_ty.effect().map_or(Task::Sync, |effect| effect.task);
        if task > Task::Sync {
            self.may_suspend = true;
        }
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

    fn put(&mut self, payload: Payload) -> usize {
        self.payloads.push(payload);
        self.payloads.len() - 1
    }

    fn path(&mut self, path: &[Step]) -> usize {
        self.put(Payload::Path(path.to_vec().into_boxed_slice()))
    }

    fn walked_under(&self, target: &RefTarget, path: &[PathSeg]) -> (u32, Vec<Step>) {
        let (id, root) = match target {
            RefTarget::Var(s) | RefTarget::Param(s) => (*s, self.ty(*s)),
            RefTarget::Through(r) => (*r, self.scrutinee_ty(*r)),
        };
        (self.slot(id), self.walked(root, path))
    }

    /// The path under `root` with each step resolved against the type it
    /// stands on, and the steps that read nothing dropped (RFC-0022).
    fn walked(&self, root: &Ty, path: &[PathSeg]) -> Vec<Step> {
        let mut at = vec![root.clone()];
        let mut kept = Vec::with_capacity(path.len());
        for seg in path {
            let resolved = resolve_step(&at, seg);
            at = step(&at, seg);
            kept.extend(resolved);
        }
        kept
    }

    fn slots(&mut self, ids: &[ValueId]) -> usize {
        self.put(Payload::Slots(
            ids.iter().copied().map(|id| self.slot(id)).collect(),
        ))
    }

    fn label(&self, label: &Label) -> u32 {
        *self
            .labels
            .get(label)
            .unwrap_or_else(|| panic!("unknown label {label:?}"))
    }

    fn target(&self, label: &Label) -> u32 {
        let at = self.label(label) as usize;
        self.op_index[at]
            .unwrap_or_else(|| panic!("a jump names {label:?}, which a loop operation absorbed"))
    }

    fn tag_is(&self, tag: Astr, name: &str) -> bool {
        self.ctx.interner.resolve(tag) == name
    }

    /// The moves a jump makes, ordered so every source is read before it is
    /// overwritten; a cycle is broken through the scratch slot.
    fn move_list(&mut self, label: &Label, args: &[ValueId]) -> Box<[SlotMove]> {
        let target = self.label(label) as usize;
        let InstKind::BlockLabel { params, .. } = &self.body.insts[target].kind else {
            panic!("a jump names {label:?}, whose instruction is not a block label")
        };
        let pairs: Vec<SlotMove> = params
            .iter()
            .zip(args)
            .map(|(param, arg)| SlotMove {
                from: self.slot(*arg),
                to: self.slot(*param),
            })
            .collect();
        let ordered = order_moves(pairs, self.scratch);
        self.scratch_used |= ordered.scratch_used;
        ordered.moves.into_boxed_slice()
    }

    fn moves(&mut self, label: &Label, args: &[ValueId]) -> usize {
        let list = self.move_list(label, args);
        self.put(Payload::Moves(list))
    }

    /// A suspending operation is excluded along with the terminators: it
    /// leaves the block for the driver, which the machine's dispatch loop
    /// alone can reach.
    fn is_straight_line(&self, inst: &Inst) -> bool {
        match &inst.kind {
            InstKind::Jump { .. }
            | InstKind::JumpIf { .. }
            | InstKind::Return { .. }
            | InstKind::Diverge
            | InstKind::Eval { .. }
            | InstKind::LoadFunction { .. }
            | InstKind::Poison { .. } => false,

            InstKind::FunctionCall { callee, .. } => match callee {
                Callee::Extern { id, instance } => self.ctx.extern_is_sync(id, *instance),
                Callee::Direct(_) | Callee::Indirect(_) => false,
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

    fn block(&mut self, range: Range<usize>, nested: &[Region]) -> BasicBlock {
        let runs = self.fused_in(range.clone(), nested);
        let chains = self.chains_in(range.clone(), nested);
        let units = self.layout(range, nested, &runs, &chains);
        BasicBlock::new(self.emit(&units).into_iter().map(|e| e.op))
    }

    fn loop_op(&mut self, region: &LoopRegion) -> Op {
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

        let enter = match region.enter_jump {
            Some(entry) => {
                let InstKind::Jump { label, args } = &body.insts[entry].kind else {
                    panic!("a recognized loop's entry is not a jump")
                };
                self.move_list(label, args)
            }
            None => Box::default(),
        };
        let into_body = self.move_list(then_label, then_args);
        let exit = self.move_list(else_label, else_args);
        let back = self.move_list(header, back_args);

        let head = self.block(region.head_block.clone(), &region.head_regions);
        let block = self.block(region.body_block.clone(), &region.body_regions);

        let at = self.put(Payload::Loop(LoopBody {
            enter,
            head,
            cond_slot: self.slot(*cond),
            into_body,
            body: block,
            back,
            exit,
        }));
        Op::new(control::while_loop).p(at)
    }

    fn diamond_op(&mut self, region: &DiamondRegion) -> Op {
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

        let cond_slot = self.slot(*cond);
        let on_true = self.arm(&region.on_true, then_label, then_args);
        let on_false = self.arm(&region.on_false, else_label, else_args);

        let at = self.put(Payload::Diamond(Diamond { on_true, on_false }));
        Op::new(control::diamond).a(cond_slot).p(at)
    }

    /// `label` and `args` are the diamond's own edge into this arm, which
    /// a `Direct` arm takes all the way to the join.
    fn arm(&mut self, region: &ArmRegion, label: &Label, args: &[ValueId]) -> DiamondArm {
        let ArmRegion::Block {
            block,
            regions,
            jump,
        } = region
        else {
            return DiamondArm {
                block: BasicBlock::new([]),
                join: self.move_list(label, args),
            };
        };
        let InstKind::Jump { label, args } = &self.body.insts[*jump].kind else {
            panic!("a recognized diamond's arm does not end in a jump")
        };
        DiamondArm {
            join: self.move_list(label, args),
            block: self.block(block.clone(), regions),
        }
    }

    fn op(&mut self, at: usize) -> Op {
        let body = self.body;
        let inst = &body.insts[at];
        match &inst.kind {
            InstKind::Const { dst, value } => self.constant(*dst, value),

            InstKind::StringConcat { dst, parts } => {
                let parts: Box<[ConcatPart]> = parts
                    .iter()
                    .map(|part| ConcatPart {
                        slot: self.slot(*part),
                        through_reference: self.is_ref(*part),
                    })
                    .collect();
                let at = self.put(Payload::Parts(parts));
                Op::new(string::concat).a(self.slot(*dst)).p(at)
            }
            InstKind::StringEq { dst, a, b } => Op::new(string::string_eq)
                .a(self.slot(*dst))
                .b(self.slot(*a))
                .c(self.slot(*b)),
            InstKind::StringClone { dst, src } => {
                let f: OpFn = if self.is_ref(*src) {
                    string::clone_string::<true>
                } else {
                    string::clone_string::<false>
                };
                Op::new(f).a(self.slot(*dst)).b(self.slot(*src))
            }

            InstKind::Ref {
                dst, target, path, ..
            } => {
                let (src, path) = self.walked_under(target, path);
                match (target, path.is_empty()) {
                    (RefTarget::Var(_) | RefTarget::Param(_), true) => {
                        Op::new(storage::ref_var).a(self.slot(*dst)).b(src)
                    }
                    (RefTarget::Var(_) | RefTarget::Param(_), false) => {
                        let at = self.path(&path);
                        Op::new(storage::ref_var_path)
                            .a(self.slot(*dst))
                            .b(src)
                            .p(at)
                    }
                    (RefTarget::Through(_), true) => {
                        Op::new(storage::ref_through).a(self.slot(*dst)).b(src)
                    }
                    (RefTarget::Through(_), false) => {
                        let at = self.path(&path);
                        Op::new(storage::ref_through_path)
                            .a(self.slot(*dst))
                            .b(src)
                            .p(at)
                    }
                }
            }
            InstKind::Take { dst, target, path } => {
                let clone = self.is_string(*dst);
                let (src, path) = self.walked_under(target, path);
                let f: OpFn = match (target, path.is_empty(), clone) {
                    (RefTarget::Var(_) | RefTarget::Param(_), true, true) => {
                        storage::take_var::<true>
                    }
                    (RefTarget::Var(_) | RefTarget::Param(_), true, false) => {
                        storage::take_var::<false>
                    }
                    (RefTarget::Var(_) | RefTarget::Param(_), false, true) => {
                        storage::read_path::<true>
                    }
                    (RefTarget::Var(_) | RefTarget::Param(_), false, false) => {
                        storage::read_path::<false>
                    }
                    (RefTarget::Through(_), true, true) => storage::take_through::<true>,
                    (RefTarget::Through(_), true, false) => storage::take_through::<false>,
                    (RefTarget::Through(_), false, true) => storage::take_through_path::<true>,
                    (RefTarget::Through(_), false, false) => storage::take_through_path::<false>,
                };
                let op = Op::new(f).a(self.slot(*dst)).b(src);
                if path.is_empty() {
                    op
                } else {
                    let at = self.path(&path);
                    op.p(at)
                }
            }
            InstKind::Assign {
                target,
                path,
                value,
            } => {
                let (dst, path) = self.walked_under(target, path);
                let f: OpFn = match (target, path.is_empty()) {
                    (RefTarget::Var(_) | RefTarget::Param(_), true) => storage::assign_var,
                    (RefTarget::Var(_) | RefTarget::Param(_), false) => storage::assign_var_path,
                    (RefTarget::Through(_), true) => storage::assign_through,
                    (RefTarget::Through(_), false) => storage::assign_through_path,
                };
                let op = Op::new(f).a(dst).b(self.slot(*value));
                if path.is_empty() {
                    op
                } else {
                    let at = self.path(&path);
                    op.p(at)
                }
            }
            InstKind::Fetch { dst, context } => {
                let at = self.put(Payload::PageKey(self.ctx.page_key(context)));
                Op::new(storage::fetch).a(self.slot(*dst)).p(at)
            }
            InstKind::Commit { context, value } => {
                let at = self.put(Payload::PageKey(self.ctx.page_key(context)));
                Op::new(storage::commit).a(self.slot(*value)).p(at)
            }

            InstKind::FieldGet {
                dst,
                object,
                field,
                rest,
            } => {
                let clone = self.is_string(*dst);
                if rest.is_empty() {
                    let f: OpFn = if clone {
                        storage::read_field::<true>
                    } else {
                        storage::read_field::<false>
                    };
                    let at = self.put(Payload::Name(*field));
                    Op::new(f).a(self.slot(*dst)).b(self.slot(*object)).p(at)
                } else {
                    let f: OpFn = if clone {
                        storage::read_path::<true>
                    } else {
                        storage::read_path::<false>
                    };
                    let path: Vec<Step> = std::iter::once(*field)
                        .chain(rest.iter().copied())
                        .map(Step::Field)
                        .collect();
                    let at = self.path(&path);
                    Op::new(f).a(self.slot(*dst)).b(self.slot(*object)).p(at)
                }
            }
            InstKind::FieldSet {
                dst,
                object,
                field,
                rest,
                value,
            } => {
                let path: Vec<Step> = std::iter::once(*field)
                    .chain(rest.iter().copied())
                    .map(Step::Field)
                    .collect();
                let at = self.path(&path);
                Op::new(storage::field_set)
                    .a(self.slot(*dst))
                    .b(self.slot(*object))
                    .c(self.slot(*value))
                    .p(at)
            }

            InstKind::BinOp {
                dst,
                op,
                left,
                right,
            } => {
                let f = match self.ty(*left) {
                    Ty::Int(k) => arith::int_binop(*op, *k),
                    Ty::Float => arith::float_binop(*op),
                    Ty::Bool => arith::bool_binop(*op),
                    other => panic!("binop {op:?} on {other:?}"),
                };
                Op::new(f)
                    .a(self.slot(*dst))
                    .b(self.slot(*left))
                    .c(self.slot(*right))
            }
            InstKind::UnaryOp { dst, op, operand } => {
                let f = match self.ty(*operand) {
                    Ty::Int(k) => arith::int_unaryop(*op, *k),
                    Ty::Float => arith::float_unaryop(*op),
                    Ty::Bool => arith::bool_unaryop(*op),
                    other => panic!("unary {op:?} on {other:?}"),
                };
                Op::new(f).a(self.slot(*dst)).b(self.slot(*operand))
            }

            InstKind::LoadFunction { .. } => Op::new(control::load_function),
            InstKind::FunctionCall {
                dst,
                callee,
                callee_ty,
                args,
                order,
            } => {
                let after = order.map_or(NO_SLOT, |edge| self.slot(edge.after));
                match callee {
                    Callee::Direct(id) => {
                        self.suspends_at(callee_ty);
                        let at = self.put(Payload::Direct {
                            callee: *id,
                            args: args.iter().copied().map(|id| self.slot(id)).collect(),
                        });
                        Op::new(call::call_direct).a(self.slot(*dst)).d(after).p(at)
                    }
                    Callee::Extern { id, instance } => {
                        let handler = self.ctx.handler(id, *instance);
                        if !handler.is_sync() {
                            self.may_suspend = true;
                        }
                        let op = Op::new(extern_call_op(&handler)).a(self.slot(*dst));
                        let call_args = if needs_window(&handler) {
                            ExternArgs::Window(self.slots.window(at).clone())
                        } else {
                            ExternArgs::ByValue
                        };
                        let op = match &call_args {
                            ExternArgs::ByValue => {
                                let slot_at =
                                    |k: usize| args.get(k).map_or(NO_SLOT, |id| self.slot(*id));
                                op.b(slot_at(0)).c(slot_at(1)).d(slot_at(2))
                            }
                            ExternArgs::Window(_) => op,
                        };
                        let payload = self.put(Payload::Extern(ExternCall {
                            handler,
                            order: after,
                            args: call_args,
                        }));
                        op.p(payload)
                    }
                    Callee::Indirect(callee_slot) => {
                        self.suspends_at(callee_ty);
                        let f: OpFn = if self.is_ref(*callee_slot) {
                            call::call_indirect::<true>
                        } else {
                            call::call_indirect::<false>
                        };
                        let at = self.slots(args);
                        Op::new(f)
                            .a(self.slot(*dst))
                            .b(self.slot(*callee_slot))
                            .d(after)
                            .p(at)
                    }
                }
            }
            InstKind::Spawn {
                dst, callee, args, ..
            } => match callee {
                Callee::Direct(id) => {
                    let at = self.put(Payload::Direct {
                        callee: *id,
                        args: args.iter().copied().map(|id| self.slot(id)).collect(),
                    });
                    Op::new(call::spawn_module).a(self.slot(*dst)).p(at)
                }
                Callee::Extern { id, instance } => {
                    let handler = self.ctx.handler(id, *instance);
                    let f: OpFn = match handler {
                        ExternHandler::Sync(_) | ExternHandler::Heavy(_) => call::spawn_extern_sync,
                        ExternHandler::Async(_) => call::spawn_extern_async,
                    };
                    let window = self.slots.window(at).clone();
                    let payload = self.put(Payload::Extern(ExternCall {
                        handler,
                        order: NO_SLOT,
                        args: ExternArgs::Window(window),
                    }));
                    Op::new(f).a(self.slot(*dst)).p(payload)
                }
                Callee::Indirect(_) => panic!("spawn: indirect callee not supported"),
            },
            InstKind::Eval { dst, src, order } => {
                self.may_suspend = true;
                Op::new(call::eval)
                    .a(self.slot(*dst))
                    .b(self.slot(*src))
                    .d(order.map_or(NO_SLOT, |id| self.slot(id)))
            }
            InstKind::Merge { dst, .. } => Op::new(control::merge).a(self.slot(*dst)),

            InstKind::MakeArray { dst, elements } => {
                let at = self.slots(elements);
                Op::new(composite::make_array).a(self.slot(*dst)).p(at)
            }
            InstKind::MakeObject { dst, fields } => {
                let fields: Box<[FieldSlot]> = fields
                    .iter()
                    .map(|(key, value)| FieldSlot {
                        key: *key,
                        slot: self.slot(*value),
                    })
                    .collect();
                let at = self.put(Payload::Fields(fields));
                Op::new(composite::make_object).a(self.slot(*dst)).p(at)
            }
            InstKind::MakeTuple { dst, elements } => {
                let at = self.slots(elements);
                Op::new(composite::make_tuple).a(self.slot(*dst)).p(at)
            }
            InstKind::TupleIndex { dst, tuple, index } => {
                let f: OpFn = if self.is_string(*dst) {
                    storage::read_index::<true>
                } else {
                    storage::read_index::<false>
                };
                Op::new(f).a(self.slot(*dst)).b(self.slot(*tuple)).p(*index)
            }

            InstKind::TestLiteral { dst, src, value } => self.test_literal(*dst, *src, value),
            InstKind::TestObjectKey { dst, src, key } => {
                let f: OpFn = if self.is_ref(*src) {
                    pattern::test_object_key::<true>
                } else {
                    pattern::test_object_key::<false>
                };
                let at = self.put(Payload::Name(*key));
                Op::new(f).a(self.slot(*dst)).b(self.slot(*src)).p(at)
            }
            InstKind::ArrayIndex { dst, array, index } => {
                let f: OpFn = if self.is_string(*dst) {
                    storage::read_index::<true>
                } else {
                    storage::read_index::<false>
                };
                Op::new(f).a(self.slot(*dst)).b(self.slot(*array)).p(*index)
            }

            // An `AsSlice` is the one call of the instance the checker
            // settled on, prepared exactly as `call_extern_1` of that
            // handler would be, so a fused run sees it as a call
            // (RFC-0044, RFC-0047 §3).
            InstKind::AsSlice {
                dst,
                container,
                instance,
                ..
            } => {
                let handler = self.ctx.handler(&instance.id, instance.instance);
                let op = Op::new(extern_call_op(&handler))
                    .a(self.slot(*dst))
                    .b(self.slot(*container));
                let payload = self.put(Payload::Extern(ExternCall {
                    handler,
                    order: NO_SLOT,
                    args: ExternArgs::ByValue,
                }));
                op.p(payload)
            }
            InstKind::Index {
                dst,
                slice,
                index,
                mode,
            } => Op::new(index::checked(*mode))
                .a(self.slot(*dst))
                .b(self.slot(*slice))
                .c(self.slot(*index)),
            InstKind::IndexSet {
                slice,
                index,
                value,
            } => Op::new(index::index_set::<true>)
                .a(self.slot(*slice))
                .b(self.slot(*index))
                .c(self.slot(*value)),
            InstKind::ObjectGet { dst, object, key } => {
                let f: OpFn = if self.is_string(*dst) {
                    storage::read_field::<true>
                } else {
                    storage::read_field::<false>
                };
                let at = self.put(Payload::Name(*key));
                Op::new(f).a(self.slot(*dst)).b(self.slot(*object)).p(at)
            }

            InstKind::MakeClosure {
                dst,
                body,
                captures,
            } => {
                let code = Arc::clone(
                    self.closures
                        .get(body)
                        .unwrap_or_else(|| panic!("closure body not found: {body:?}")),
                );
                let at = self.put(Payload::Closure {
                    code,
                    captures: captures.iter().copied().map(|id| self.slot(id)).collect(),
                });
                Op::new(call::make_closure).a(self.slot(*dst)).p(at)
            }

            InstKind::MakeVariant { dst, tag, payload } => self.make_variant(*dst, *tag, *payload),
            InstKind::TestVariant { dst, src, tag } => {
                let through = self.is_ref(*src);
                let f: OpFn = match (variant_form(self.scrutinee_ty(*src)), through) {
                    (VariantForm::Option, false) => variant::test_option::<false>,
                    (VariantForm::Option, true) => variant::test_option::<true>,
                    (VariantForm::Result, false) => variant::test_result::<false>,
                    (VariantForm::Result, true) => variant::test_result::<true>,
                    (VariantForm::Enum, false) => variant::test_variant::<false>,
                    (VariantForm::Enum, true) => variant::test_variant::<true>,
                };
                let at = self.put(Payload::Name(*tag));
                Op::new(f)
                    .a(self.slot(*dst))
                    .b(self.slot(*src))
                    .c(u32::from(self.tag_is(*tag, "Some")))
                    .d(u32::from(self.tag_is(*tag, "Ok")))
                    .p(at)
            }
            InstKind::UnwrapVariant { dst, src } => {
                let f: OpFn = match variant_form(self.ty(*src)) {
                    VariantForm::Option => variant::unwrap_option,
                    VariantForm::Result => variant::unwrap_result,
                    VariantForm::Enum => variant::unwrap_variant,
                };
                Op::new(f).a(self.slot(*dst)).b(self.slot(*src))
            }

            InstKind::BlockLabel { .. } => Op::new(control::nop),
            InstKind::Jump { label, args } => {
                let target = self.target(label);
                let at = self.moves(label, args);
                Op::new(control::jump).b(target).p(at)
            }
            InstKind::JumpIf {
                cond,
                then_label,
                then_args,
                else_label,
                else_args,
            } => {
                let then_target = self.target(then_label);
                let else_target = self.target(else_label);
                let then_at = self.moves(then_label, then_args);
                let else_at = self.moves(else_label, else_args);
                Op::new(control::jump_if)
                    .a(self.slot(*cond))
                    .b(then_target)
                    .c(else_target)
                    .d(else_at as u32)
                    .p(then_at)
            }
            InstKind::Return { value, .. } => Op::new(control::ret).a(self.slot(*value)),
            InstKind::Diverge => Op::new(control::diverge),
            InstKind::Undef { dst } => Op::new(control::undef).a(self.slot(*dst)),
            InstKind::Nop => Op::new(control::nop),
            InstKind::Drop { src } => Op::new(control::drop_value).a(self.slot(*src)),
            InstKind::Poison { .. } => Op::new(control::poison),
        }
    }

    fn constant(&mut self, dst: ValueId, value: &Literal) -> Op {
        let out = self.slot(dst);
        match (value, self.ty(dst)) {
            (Literal::Int(n), Ty::Int(k)) => {
                let f = for_int_ty!(*k, |T| constant::int::<T> as OpFn);
                Op::new(f).a(out).p(*n as u64 as usize)
            }
            (Literal::Int(n), other) => panic!("integer literal {n} typed as {other:?}"),
            (Literal::Float(x), _) => Op::new(constant::float).a(out).p(x.to_bits() as usize),
            (Literal::Bool(b), _) => Op::new(constant::boolean).a(out).p(usize::from(*b)),
            (Literal::Unit, _) => Op::new(constant::unit).a(out),
            (Literal::String(s), _) => {
                let at = self.put(Payload::Konst(Konst::Str(s.clone())));
                Op::new(constant::konst).a(out).p(at)
            }
            (Literal::List(items), Ty::Array(elem, _)) => {
                let konst = Konst::List(items.iter().map(|item| konst_of(item, elem)).collect());
                let at = self.put(Payload::Konst(konst));
                Op::new(constant::konst).a(out).p(at)
            }
            (Literal::List(_), other) => panic!("list literal typed as {other:?}"),
        }
    }

    fn test_literal(&mut self, dst: ValueId, src: ValueId, value: &Literal) -> Op {
        let through = self.is_ref(src);
        let (f, word): (OpFn, usize) = match value {
            Literal::Int(n) => {
                let Ty::Int(k) = self.ty(src) else {
                    panic!("TestLiteral: an integer literal against a non-integer")
                };
                (
                    for_int_ty!(*k, |T| pattern::test_int::<T> as OpFn),
                    self.put(Payload::Wide(*n)),
                )
            }
            Literal::Float(x) => (
                if through {
                    pattern::test_float::<true>
                } else {
                    pattern::test_float::<false>
                },
                x.to_bits() as usize,
            ),
            Literal::Bool(b) => (
                if through {
                    pattern::test_bool::<true>
                } else {
                    pattern::test_bool::<false>
                },
                usize::from(*b),
            ),
            Literal::String(s) => (
                if through {
                    pattern::test_string::<true>
                } else {
                    pattern::test_string::<false>
                },
                self.put(Payload::Text(s.clone())),
            ),
            Literal::Unit => (pattern::test_unit, 0),
            Literal::List(_) => panic!("TestLiteral on a list literal"),
        };
        Op::new(f).a(self.slot(dst)).b(self.slot(src)).p(word)
    }

    fn make_variant(&mut self, dst: ValueId, tag: Astr, payload: Option<ValueId>) -> Op {
        let carries = payload.is_some();
        let (f, word): (OpFn, usize) = match self.ty(dst) {
            Ty::Option(_) => (
                if carries {
                    variant::make_option::<true>
                } else {
                    variant::make_option::<false>
                },
                0,
            ),
            Ty::Result(..) => {
                assert!(carries, "Ok and Err carry a payload");
                (
                    if self.tag_is(tag, "Ok") {
                        variant::make_result::<true>
                    } else {
                        variant::make_result::<false>
                    },
                    0,
                )
            }
            _ => (
                if carries {
                    variant::make_variant::<true>
                } else {
                    variant::make_variant::<false>
                },
                self.put(Payload::Name(tag)),
            ),
        };
        Op::new(f)
            .a(self.slot(dst))
            .b(payload.map_or(NO_SLOT, |id| self.slot(id)))
            .p(word)
    }
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

/// A jump's moves, ordered, and whether the scratch slot carried a cycle.
struct MoveOrdering {
    moves: Vec<SlotMove>,
    scratch_used: bool,
}

/// Order a parallel move: every source is read before it is overwritten,
/// and a cycle is broken by moving one source into `scratch` first.
fn order_moves(pairs: Vec<SlotMove>, scratch: u32) -> MoveOrdering {
    let mut pending: Vec<SlotMove> = pairs.into_iter().filter(|m| m.from != m.to).collect();
    let mut moves: Vec<SlotMove> = Vec::with_capacity(pending.len());
    let mut scratch_used = false;
    let mut scratch_holds = false;

    while !pending.is_empty() {
        let emitted = moves.len();
        let mut i = 0;
        while i < pending.len() {
            let overwrites = pending[i].to;
            let read_later = pending
                .iter()
                .enumerate()
                .any(|(j, m)| j != i && m.from == overwrites);
            if read_later {
                i += 1;
            } else {
                let m = pending.remove(i);
                scratch_holds &= m.from != scratch;
                moves.push(m);
            }
        }

        if moves.len() == emitted {
            assert!(
                !scratch_holds,
                "a second cycle reached the scratch slot while it still held a value"
            );
            let cycled = pending[0].from;
            moves.push(SlotMove {
                from: cycled,
                to: scratch,
            });
            for m in pending.iter_mut().filter(|m| m.from == cycled) {
                m.from = scratch;
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

fn extern_call_op(handler: &ExternHandler) -> OpFn {
    match handler {
        ExternHandler::Async(_) => call::call_extern_async,
        ExternHandler::Heavy(_) => call::call_extern_heavy,
        ExternHandler::Sync(SyncHandler::Arity0(_)) => call::call_extern_0,
        ExternHandler::Sync(SyncHandler::Arity1(_)) => call::call_extern_1,
        ExternHandler::Sync(SyncHandler::Arity2(_)) => call::call_extern_2,
        ExternHandler::Sync(SyncHandler::Arity3(_)) => call::call_extern_3,
        ExternHandler::Sync(SyncHandler::ArityN(_)) => call::call_extern_n,
        ExternHandler::Sync(SyncHandler::Slice(_)) => call::call_extern_slice,
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
struct PendingMove {
    arg: ValueId,
    to: u32,
}

struct WindowPlan {
    base: u32,
    arity: u32,
    moved: Vec<PendingMove>,
}

/// Which ranges of instructions each frame slot is already spoken for.
struct Occupancy(Vec<Vec<LiveRange>>);

impl Occupancy {
    fn free(&self, slot: usize, range: LiveRange) -> bool {
        match self.0.get(slot) {
            Some(taken) => !taken.iter().any(|held| held.overlaps(range)),
            None => true,
        }
    }

    fn take(&mut self, slot: usize, range: LiveRange) {
        if self.0.len() <= slot {
            self.0.resize_with(slot + 1, Vec::new);
        }
        self.0[slot].push(range);
    }

    fn lowest_free(&self, range: LiveRange) -> usize {
        (0..self.0.len())
            .find(|slot| self.free(*slot, range))
            .unwrap_or(self.0.len())
    }

    /// The lowest base whose `arity` slots are each free over the range
    /// that position will hold.
    fn lowest_free_run(&self, places: &[ArgPlace], call: usize) -> usize {
        let fits = |base: usize| {
            places
                .iter()
                .enumerate()
                .all(|(k, place)| self.free(base + k, place.occupies(call)))
        };
        (0..self.0.len())
            .find(|base| fits(*base))
            .unwrap_or(self.0.len())
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
    windows: FxHashMap<usize, ArgWindow>,
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

    fn window(&self, call: usize) -> &ArgWindow {
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

    // The windows first: a contiguous run is the constrained resource.
    let mut slot_of: Vec<Option<u32>> = vec![None; values];
    let mut occupancy = Occupancy(Vec::new());
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
        let base = occupancy.lowest_free_run(&places, at);

        let mut moved = Vec::new();
        for (k, (place, arg)) in places.iter().zip(args).enumerate() {
            occupancy.take(base + k, place.occupies(at));
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
        let slot = occupancy.lowest_free(range);
        occupancy.take(slot, range);
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

    let windows = plans
        .into_iter()
        .map(|(at, plan)| {
            let moves = plan
                .moved
                .into_iter()
                .map(|PendingMove { arg, to }| SlotMove {
                    from: of[arg.to_raw()],
                    to,
                })
                .collect();
            (
                at,
                ArgWindow {
                    at: plan.base,
                    arity: plan.arity,
                    moves,
                },
            )
        })
        .collect();

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
        let run = window.at..window.at + window.arity;
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
    use acvus_ast::BinOp as AstBinOp;

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
            let handler = ExternHandler::Async(Arc::new(|_, _| {
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
            let handler = ExternHandler::Sync(SyncHandler::ArityN(Arc::new(|_, _| {
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
        let insts = vec![add(1, 2, 3), call(4, stays)];
        assert!(!fixture.prepared(insts).may_suspend());
    }

    #[test]
    fn a_body_that_calls_an_asynchronous_extern_can_suspend() {
        let mut fixture = Fixture::new();
        let suspends = fixture.async_extern("suspends");
        let insts = vec![add(1, 2, 3), call(4, suspends)];
        assert!(fixture.prepared_at(insts, Task::Async).may_suspend());
    }

    /// The type decides, not the call shape: an indirect call through a
    /// closure whose effect is Sync does not suspend, and the same call
    /// through one whose effect is Async does (RFC-0046).
    #[test]
    fn a_closure_call_suspends_where_the_closure_type_says_so() {
        let indirect = |task| {
            vec![inst(InstKind::FunctionCall {
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
            })]
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
    use acvus_ast::BinOp as AstBinOp;

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
        let by_value = ExternHandler::Sync(SyncHandler::Arity1(Arc::new(|_, v| v)));
        let window = ExternHandler::Sync(SyncHandler::ArityN(Arc::new(|_, args| {
            std::mem::take(&mut args[0])
        })));
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
        let (a, b) = (slots.of(val(2)), slots.of(val(3)));
        assert_ne!(a, b, "a swap's two parameters cannot share one slot");
        let pairs = vec![SlotMove { from: b, to: a }, SlotMove { from: a, to: b }];
        assert!(order_moves(pairs, slots.frame).scratch_used);
    }

    #[test]
    fn an_argument_that_dies_at_the_call_is_allocated_into_the_window() {
        let slots = assign(vec![konst(0), call(1, WINDOW, &[0]), ret(1)]);
        let window = slots.window(1);
        assert!(window.moves.is_empty());
        assert_eq!(slots.of(val(0)), window.at);
    }

    #[test]
    fn an_argument_used_after_the_call_is_moved_into_the_window() {
        let slots = assign(vec![konst(0), call(1, WINDOW, &[0]), add(2, 0, 1), ret(2)]);
        let window = slots.window(1);
        let arg = slots.of(val(0));
        assert_ne!(arg, window.at);
        let moved: Vec<(u32, u32)> = window.moves.iter().map(|m| (m.from, m.to)).collect();
        assert_eq!(moved, vec![(arg, window.at)]);
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

    fn pairs(list: &[(u32, u32)]) -> Vec<SlotMove> {
        list.iter()
            .map(|(from, to)| SlotMove {
                from: *from,
                to: *to,
            })
            .collect()
    }

    fn emitted(ordering: &MoveOrdering) -> Vec<(u32, u32)> {
        ordering.moves.iter().map(|m| (m.from, m.to)).collect()
    }

    #[test]
    fn a_source_is_read_before_it_is_overwritten() {
        let ordering = order_moves(pairs(&[(0, 1), (1, 2)]), 9);
        assert_eq!(emitted(&ordering), vec![(1, 2), (0, 1)]);
        assert!(!ordering.scratch_used);
    }

    #[test]
    fn a_cycle_goes_through_the_scratch_slot() {
        let ordering = order_moves(pairs(&[(0, 1), (1, 0)]), 9);
        assert_eq!(emitted(&ordering), vec![(0, 9), (1, 0), (9, 1)]);
        assert!(ordering.scratch_used);
    }

    #[test]
    fn a_self_move_is_no_move() {
        let ordering = order_moves(pairs(&[(3, 3)]), 9);
        assert!(ordering.moves.is_empty());
    }

    #[test]
    fn two_cycles_reuse_the_one_scratch_slot() {
        let ordering = order_moves(pairs(&[(0, 1), (1, 0), (2, 3), (3, 2)]), 9);
        assert!(ordering.scratch_used);
        assert_eq!(ordering.moves.len(), 6);
    }
}

// -- The arithmetic-chain recognizer (RFC-0044, stage 4) ----------------

/// One operation and the span of the instruction it came from.
struct Emitted {
    op: Op,
    span: Span,
}

enum Tree {
    Leaf(u16),
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
    leaves: Vec<Option<u16>>,
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

    fn chain(&self, remap: &dyn Fn(u16) -> u16) -> Chain {
        let word = self.node.word();
        let shape = Shape::of_word(&word).unwrap_or_else(|| {
            panic!(
                "a chain of preorder shape {word} has more than {} nodes",
                Chain::MAX_NODES
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
        let in_space = |slot: u16| Chain::offset(u32::from(remap(slot)));

        let mut post_order_ops = [Arith::Add; Chain::MAX_INTERIOR];
        post_order_ops[..found.len()].copy_from_slice(&found);
        let mut leaf_offsets = [in_space(read); Chain::MAX_LEAVES];
        for (offset, leaf) in leaf_offsets.iter_mut().zip(&leaves) {
            *offset = in_space(leaf.unwrap_or(read));
        }

        Chain {
            shape,
            post_order_ops,
            root: self.node.op,
            leaf_offsets,
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
            .filter(|_| self.nodes + self.nodes_of(value) <= Chain::MAX_NODES);
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
                slot: *slot,
                value: value.copy_word(),
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
    handler: SyncHandler,
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
        let ExternHandler::Sync(handler) = self.ctx.handler(id, instance) else {
            return None;
        };
        if handler.arity() != Some(args.len()) || args.len() > 2 {
            return None;
        }
        Some(FusableCall { dst, args, handler })
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

    fn fused_op(&mut self, region: &FusedRegion) -> Op {
        let mut calls: Vec<FusedCall> = Vec::with_capacity(region.calls.len());
        let mut previous: Option<ValueId> = None;
        for at in &region.calls {
            let call = self
                .fusable_call(*at)
                .expect("a recognized run holds a fusable call at every index it named");
            let mut args = [NO_SLOT; 3];
            for (word, id) in args.iter_mut().zip(call.args) {
                *word = match previous {
                    Some(held) if *id == held => PREVIOUS,
                    Some(_) | None => self.slot(*id),
                };
            }
            previous = Some(call.dst);
            calls.push(FusedCall {
                handler: call.handler,
                args,
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

        let slot = self.slot(dst);
        let tail_present = tail.is_some();
        let payload = self.put(Payload::Fused(FusedRun {
            calls: calls.into_boxed_slice(),
            tail,
        }));
        Op::new(call::fused_instance(region.calls.len(), tail_present))
            .a(slot)
            .p(payload)
    }

    fn def_at(&self, value: ValueId) -> Option<usize> {
        self.def_inst[value.to_raw()]
    }

    fn use_count(&self, value: ValueId) -> u32 {
        self.use_counts[value.to_raw()]
    }

    fn leaf_slot(&self, value: ValueId) -> u16 {
        u16::try_from(self.slot(value)).expect("a chain reads a register below u16::MAX")
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
                growing.nodes <= Chain::MAX_NODES,
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

    fn chain_op(&mut self, run: &ChainRun) -> Op {
        self.check_chain(run);
        let chain = Box::new(run.chain(&|slot| slot));
        let f = chain::instance(run.ty, &chain).op;
        let held = std::ptr::from_ref::<Chain>(&chain) as usize;
        let at = self.put(Payload::Chain(chain));
        let at = u32::try_from(at).expect("a body holds at most u32::MAX payloads");
        Op::new(f).a(self.slot(run.dst)).b(at).p(held)
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

    /// The operations of a laid-out range: the regions and the chains one
    /// operation each, and everything else one for one.
    fn emit(&mut self, units: &[Unit<'_>]) -> Vec<Emitted> {
        let mut out: Vec<Emitted> = Vec::with_capacity(units.len());
        for unit in units {
            let span = self.body.insts[unit.head()].span;
            let op = match unit {
                Unit::Inst(at) => self.op(*at),
                Unit::Region(Region::Loop(region)) => self.loop_op(region),
                Unit::Region(Region::Diamond(region)) => self.diamond_op(region),
                Unit::Fused(region) => self.fused_op(region),
                Unit::Chain(run) => self.chain_op(run),
            };
            out.push(Emitted { op, span });
        }
        out
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
                u32::from(slot) < self.scratch + u32::from(self.scratch_used),
                "a chain reads register {slot}, which its body's frame does not have"
            );
            let read = (self.body.insts[run.insts.start..run.insts.end])
                .iter()
                .flat_map(|inst| inst_info::uses(&inst.kind))
                .find(|value| self.leaf_slot(*value) == slot)
                .unwrap_or_else(|| {
                    panic!("a chain reads register {slot}, which no instruction it replaces reads")
                });
            assert_eq!(
                chain_ty(self.ty(read)),
                Some(run.ty),
                "a chain reads register {slot} at another type than its own"
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
            konsts.push(konst.copy_word());
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
        let mut in_space: FxHashMap<u16, u16> = FxHashMap::default();
        for leaf in run.node.flatten().leaves.iter().flatten() {
            let at = match body
                .params
                .iter()
                .position(|(_, id)| u32::from(*leaf) == self.slot(*id))
            {
                Some(at) => at,
                None => {
                    let konst = self.konsts.value_at.get(&u32::from(*leaf))?;
                    body.params.len() + intern(&mut konsts, konst)
                }
            };
            in_space.insert(
                *leaf,
                u16::try_from(at).expect("a body reads at most u16::MAX operands"),
            );
        }
        if body.params.len() + konsts.len() > ExprChain::MAX_OPERANDS {
            return None;
        }

        self.check_chain(&run);
        let chain = run.chain(&|slot| {
            *in_space
                .get(&slot)
                .unwrap_or_else(|| panic!("register {slot} is neither a parameter nor a constant"))
        });
        let eval = chain::instance(run.ty, &chain).expr;
        Some(Expr {
            arity,
            body: ExprBody::Chain(ExprChain {
                chain,
                eval,
                konsts: konsts.into_boxed_slice(),
            }),
            span: last.span,
        })
    }
}
