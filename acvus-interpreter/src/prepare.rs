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
use acvus_extern::Width;
use acvus_mir::analysis::inst_info;
use acvus_mir::graph::QualifiedRef;
use acvus_mir::ir::{
    Callee, Inst, InstKind, Label, MirBody, MirModule, PathSeg, RefTarget, ValueId,
};
use acvus_mir::ty::{CastTy, IntTy, Task, Ty};
use acvus_utils::{Astr, Interner, LocalIdOps};
use rustc_hash::{FxHashMap, FxHashSet};
use smallvec::SmallVec;

use crate::code::{
    Arith, BlockId, Body, ChainBounds, Code, Compare, ConcatPart, Deref, EntryKonst, Expr,
    ExprBody, ExprChain, FieldSlot, Konst, Node, Off, Op, Prepared, Root, Shape, SlicePair, Slot,
    SlotKind, Step, Where, chain, made, node,
};
use crate::interpreter::Executable;
use crate::ops::arith::{self, Binary, Unary, for_int_ty};
use crate::ops::chain::{self, ChainTy, LeafRead, Plan, Reads};
use crate::ops::place;
use crate::ops::{
    call, cast, composite, constant, control, index, pattern, select, storage, string, switch,
    variant,
};
use crate::runtime::ExternHandler;
use crate::value::{Kind, Value};

pub mod runs;

/// The registers `order_moves` may break a cycle through, which a run's base is
/// placed above. It is two rather than one because a cycle carrying a slice moves
/// the pair through the scratch (`Moved::Pair`), and `scratch_used` grows while
/// the body is being emitted — after every run's `Off` is already written.
const MAX_SCRATCH_SLOTS: u16 = 2;

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

    let frame_len = prep.scalar_len();
    let param_ids: Vec<ValueId> = body.params.iter().map(|(_, v)| *v).collect();
    let param_marks = prep.take_mask(&param_ids);
    let params = param_ids.iter().map(|id| prep.off(*id)).collect();
    let captures = body.captures.iter().map(|(_, v)| prep.off(*v)).collect();
    let order_param = body.order_param.map(|id| prep.off(id));
    let entry_konsts = prep.entry_konsts();
    let slot_kinds = prep.slot_kinds(frame_len, prep.param_run());

    Code::Body(Arc::new(Body {
        heads: blocks,
        entry: 0,
        frame_len,
        frame_cells: u16::try_from(crate::regs::cells_for(frame_len))
            .expect("the cells of the widest frame fit a u16"),
        mark_words: crate::regs::mark_words(frame_len),
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
    /// Where each addressed aggregate of this body lives (RFC-0050 rule 2).
    ///
    /// No operation reads a run yet, so `scalar_len` does not add `plan.total` to
    /// the frame. That is a decision: a frame handed registers no operation reads
    /// pays for them twice, once in the cells every bind touches and once at
    /// every call into the body, because a frame past the window a caller keeps
    /// roots a `Store` of its own.
    plan: runs::RunPlan,
    scratch: u32,
    /// The registers `order_moves` broke a cycle through: none, one, or the
    /// two of a slice's pair.
    scratch_used: u32,
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
    /// One chain per conditional edge that carries a parallel move: the
    /// `Mov`s and a `Goto`. An edge with no move names its target directly.
    edges: Vec<Box<dyn Op>>,
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

/// The facts a `Select` is built from, read off a diamond whose one arm is a
/// single pure node and whose other arm passes the incoming word through
/// (RFC-0052 §"a diamond of two pure arms is a select").
struct SelectShape {
    cond: ValueId,
    dst: ValueId,
    ty: ChainTy,
    root: Root,
    left: ValueId,
    right: ValueId,
    passed: ValueId,
    computes_on_true: bool,
}

struct EdgeArgs<'r> {
    on_true: &'r [ValueId],
    on_false: &'r [ValueId],
}

struct Sides<'r> {
    arm_block: Range<usize>,
    arm_regions: &'r [Region],
    arm_jump: usize,
    passed_args: &'r [ValueId],
    computes_on_true: bool,
}

impl<'r> Sides<'r> {
    fn of(region: &'r DiamondRegion, args: EdgeArgs<'r>) -> Option<Sides<'r>> {
        match (&region.on_true, &region.on_false) {
            (
                ArmRegion::Block {
                    block,
                    regions,
                    jump,
                },
                ArmRegion::Direct,
            ) => Some(Sides {
                arm_block: block.clone(),
                arm_regions: regions,
                arm_jump: *jump,
                passed_args: args.on_false,
                computes_on_true: true,
            }),
            (
                ArmRegion::Direct,
                ArmRegion::Block {
                    block,
                    regions,
                    jump,
                },
            ) => Some(Sides {
                arm_block: block.clone(),
                arm_regions: regions,
                arm_jump: *jump,
                passed_args: args.on_true,
                computes_on_true: false,
            }),
            _ => None,
        }
    }
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

    /// The value if it is a word with exactly one use — the two facts that do
    /// not depend on where it is produced or read.
    fn rides_word(&self, dst: ValueId) -> Option<ValueId> {
        let ty = self.ty(dst);
        let word = word_kind(ty).is_some() && !owns_large(ty);
        (word && self.use_count(dst) == 1).then_some(dst)
    }

    /// The word this unit leaves that could ride.
    ///
    /// Obligation across artifacts: each producer below prepares to exactly
    /// one node, with no move before or after it, so that "the next unit"
    /// here and "the next operation" in the chain name the same thing.
    fn rides_from(&self, unit: &Unit<'_>) -> Option<ValueId> {
        match unit {
            Unit::Inst(at) => match &self.body.insts[*at].kind {
                InstKind::BinOp { dst, .. }
                | InstKind::UnaryOp { dst, .. }
                | InstKind::Cast { dst, .. } => self.rides_word(*dst),
                _ => None,
            },
            Unit::Chain(run) => self.rides_word(run.dst),
            // A `Select` writes the join's one register and nothing else, so
            // the word it leaves can ride where a `Diamond`'s cannot: the
            // arms of a diamond each write that register themselves.
            Unit::Region(Region::Diamond(region)) => {
                self.rides_word(self.select_shape(region)?.dst)
            }
            _ => None,
        }
    }

    /// Whether the operation `at` prepares to reads `value` through a place
    /// it is specialized on.
    fn reads_place(&self, at: usize, value: ValueId) -> bool {
        match &self.body.insts[at].kind {
            InstKind::BinOp { left, right, .. } => *left == value || *right == value,
            InstKind::UnaryOp { operand, .. } => *operand == value,
            InstKind::Cast { src, .. } => *src == value,
            InstKind::JumpIf { cond, .. } => *cond == value,
            _ => false,
        }
    }

    /// A `Loop` reads its condition after its head chain has returned, so the
    /// word cannot ride to it: the joint is between them. A `Diamond` is an
    /// operation of this chain, so it can.
    fn consumes_place(&self, unit: &Unit<'_>, value: ValueId) -> bool {
        match unit {
            Unit::Inst(at) => self.reads_place(*at, value),
            Unit::Region(Region::Diamond(region)) => self.reads_place(region.jump_if, value),
            _ => false,
        }
    }

    /// `split` is how a block array's units are cut into blocks; a region's
    /// part is one chain and passes `None`.
    fn rides_in(&self, units: &[Unit<'_>], split: Option<&Split>) -> Rides {
        let mut found = FxHashSet::default();
        for (at, unit) in units.iter().enumerate() {
            let Some(next) = units.get(at + 1) else {
                continue;
            };
            let Some(value) = self.rides_from(unit) else {
                continue;
            };
            let joined = split.is_none_or(|split| split.block_of[at] == split.block_of[at + 1]);
            if joined && self.consumes_place(next, value) {
                found.insert(value);
            }
        }
        Rides(found)
    }

    /// Where an operation reads or writes `value`.
    fn place_of(&self, rides: &Rides, value: ValueId) -> Where {
        match rides.holds(value) {
            true => Where::Register,
            false => Where::Frame(self.off(value)),
        }
    }
}

fn block_heads(units: &[Unit<'_>], split: &Split) -> FxHashMap<usize, BlockId> {
    units
        .iter()
        .enumerate()
        .map(|(at, unit)| (unit.head(), split.block_of[at]))
        .collect()
}

/// The values that ride in the argument register instead of a frame register
/// (RFC-0052 rule 5). `rides_in` is what decides membership.
///
/// Decision not to build: no ride crosses a level. This is a value `blocks`
/// and `straight` each build for their own unit list and pass down, not a
/// field of `Prepare`, so there is no path by which one level's set could
/// reach another's operations.
#[derive(Default)]
struct Rides(FxHashSet<ValueId>);

impl Rides {
    fn holds(&self, value: ValueId) -> bool {
        self.0.contains(&value)
    }

    /// The word a region's part hands its region, which `Prepare::part`
    /// added after `rides_in` had the pairs inside the part.
    fn add(&mut self, value: ValueId) {
        self.0.insert(value);
    }
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
        InstKind::Switch { arms, default, .. } => {
            arms.iter().any(|(_, named, _)| *named == label)
                || default.as_ref().is_some_and(|(named, _)| *named == label)
        }
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
        // A run begins above the registers `order_moves` may take while the body
        // is being emitted, because a run's `Off` is written before it takes
        // them (RFC-0050 rule 2).
        let run_base = Slot::try_from(slots.frame + u32::from(MAX_SCRATCH_SLOTS))
            .unwrap_or_else(|_| panic!("a body of {} registers has no run base", slots.frame));
        let plan = runs::plan(body, &labels, run_base, &slots.ranges);
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
            plan,
            scratch_used: 0,
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

    /// The two registers a slice-typed value occupies, both fixed here so no
    /// `run` computes the second (RFC-0047 amended, rule 1).
    fn pair(&self, id: ValueId) -> SlicePair {
        SlicePair::at(self.off(id))
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

    /// The scalar registers and the one `order_moves` may break a cycle
    /// through. A run begins above this, so it is final before the first `Off`
    /// of a run is written (RFC-0050 rule 2).
    fn scalar_len(&self) -> u16 {
        let len = self.scratch + self.scratch_used;
        let len = u16::try_from(len)
            .unwrap_or_else(|_| panic!("a body of {len} registers is past a frame's reach"));
        assert!(
            len <= crate::regs::MAX_SCALAR_SLOTS,
            "a body was coloured into {len} scalar registers, past the {}",
            crate::regs::MAX_SCALAR_SLOTS
        );
        assert!(
            self.scratch_used <= u32::from(MAX_SCRATCH_SLOTS),
            "a jump's cycle went through {} scratch registers, past the {MAX_SCRATCH_SLOTS} a \
             run's base is placed above",
            self.scratch_used
        );
        // Rule 3's frame equation, checked where both halves are final. The
        // frame does not yet carry the runs, so this asserts what it will be
        // rather than what it is.
        let frame = len + MAX_SCRATCH_SLOTS + self.plan.total;
        assert!(
            frame <= crate::regs::MAX_FRAME_SLOTS,
            "a body's {len} scalar registers and {} run registers reach {frame}, past the {}",
            self.plan.total,
            crate::regs::MAX_FRAME_SLOTS
        );
        len
    }

    /// # Panics
    /// A parameter sits elsewhere, which is `assign_slots`'s first phase
    /// disagreeing with the argument run `Prepare::laid` lays.
    fn param_run(&self) -> u16 {
        let mut run: u16 = 0;
        for (_, id) in &self.body.params {
            assert_eq!(
                self.off(*id).index(),
                usize::from(run),
                "parameter {id:?} is in register {}, not the {run} its argument lands in",
                self.off(*id).index()
            );
            run += u16::try_from(SlotClass::of(self.ty(*id)).width())
                .expect("a register class is two registers at most");
        }
        run
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
    ///
    /// The parameter run is not in it. A caller writes those registers whole
    /// before the frame is entered, so `machine::open_frame` opening them
    /// would overwrite the arguments it was handed; the kind the frame would
    /// have written is the kind the caller's value carries.
    fn slot_kinds(&self, frame_len: u16, param_run: u16) -> Box<[SlotKind]> {
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
            let class = SlotClass::of(ty);
            let raw = self.slots.raw(*id);
            if raw == NO_SLOT {
                continue;
            }
            let base = Slot::try_from(raw)
                .unwrap_or_else(|_| panic!("value {id:?} is in register {raw}, past a frame"));
            let SlotClaim::Word(kind) = class.claim() else {
                continue;
            };
            for k in 0..class.width() {
                let at = u16::try_from(k).expect("a register class is two wide at most");
                let slot = base.checked_add(at).unwrap_or_else(|| {
                    panic!("value {id:?} is a pair at register {base}, which leaves a frame")
                });
                open(slot, kind);
            }
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
            .skip(usize::from(param_run))
            .filter_map(|(slot, kind)| {
                let slot = Slot::try_from(slot).expect("a frame's registers fit a Slot");
                kind.map(|kind| SlotKind {
                    slot: Off::of(slot),
                    kind,
                })
            })
            .collect()
    }

    /// A synchronous call into a body lays its arguments in the window above
    /// this frame, which is the callee's frame (RFC-0052 rule 7).
    ///
    /// # Panics
    /// The run leaves the cell the window begins with, which is the widest
    /// argument list a call can lay.
    fn laid(&mut self, args: &[ValueId], ops: &mut Vec<Node>) -> Laid {
        let mut arity: Slot = 0;
        for id in args {
            let src = self.off(*id);
            let at = Off::of(arity);
            let class = SlotClass::of(self.ty(*id));
            ops.push(match class {
                SlotClass::Slice => node(move |next| call::LayPair {
                    at: SlicePair::at(at),
                    src: SlicePair::at(src),
                    next,
                }),
                SlotClass::Word(_) | SlotClass::Whole => {
                    node(move |next| call::LayArg { at, src, next })
                }
            });
            arity += u16::try_from(class.width()).expect("a register class is two wide at most");
            assert!(
                arity <= crate::regs::CELL_SLOTS,
                "a call of {arity} argument registers is past the {} one window's first cell \
                 holds",
                crate::regs::CELL_SLOTS
            );
        }
        Laid {
            arity,
            takes: self.take_mask(args),
        }
    }

    /// The registers of a list of operands, and the mask of the ones this
    /// operation takes the frame's claim on (RFC-0048 §5).
    fn taken(&self, ids: &[ValueId]) -> Operands {
        Operands {
            slots: ids.iter().map(|id| self.off(*id)).collect(),
            takes: self.take_mask(ids),
        }
    }

    /// `Regs::take_mask` reads mark word 0 alone, and the assertion here is what
    /// makes that read total: no structure in `regs.rs` can check that the
    /// registers of a mask fall in one word.
    fn take_mask(&self, ids: &[ValueId]) -> u64 {
        let mut mask = 0u64;
        for id in ids {
            if !self.owns(*id) {
                continue;
            }
            let off = self.off(*id);
            assert_eq!(
                off.mark_word(),
                0,
                "value {id:?} is in register {}, whose mark bit is outside mark word 0",
                off.index()
            );
            mask |= off.mark_bit();
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
    fn move_ops(&mut self, label: &Label, args: &[ValueId]) -> Vec<Node> {
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
        self.scratch_used = self.scratch_used.max(ordered.scratch_used);
        ordered.moves.iter().map(mov_op).collect()
    }

    /// The block a conditional edge goes to: its own, holding the edge's
    /// `Mov`s, where it carries any; the target itself where it carries none.
    fn edge(&mut self, moves: Vec<Node>, target: BlockId) -> BlockId {
        if moves.is_empty() {
            return target;
        }
        let made = BlockId::try_from(self.level.edges.len())
            .expect("a block array with more edge blocks than a BlockId counts");
        let at = self.level.blocks_len + made;
        self.level
            .edges
            .push(chain(moves, Box::new(control::Goto { target })));
        at
    }

    /// One arm of a dispatch as the block the machine enters for it.
    fn dispatch_edge(&mut self, label: Label, args: &[ValueId]) -> BlockId {
        let target = self.target(&label);
        let moves = self.move_ops(&label, args);
        self.edge(moves, target)
    }

    /// The block a two-sided dispatch enters for the tag `name`: the arm
    /// that names it, or the edge the dispatch falls out of.
    fn side(&self, placed: &[Placed], default: BlockId, name: &str) -> BlockId {
        placed
            .iter()
            .find(|arm| self.tag_is(arm.key, name))
            .map_or(default, |arm| arm.block)
    }

    /// `Terminator::Switch` as the machine's one dispatch (RFC-0051 §5).
    ///
    /// Obligation across artifacts: the operation is handed a `default` that
    /// is a real edge, so no `ops::switch` run decides that no arm holds. It
    /// is the catch-all where the `match` wrote one, and otherwise the last
    /// arm — the one tag then left untested, which is sound exactly because
    /// `validate::exhaustive` (RFC-0051 §3) decided some arm holds.
    ///
    /// `arms` is the IR's own positional shape, which RFC-0051 §5 fixes as
    /// `(tag, label, args)`; `Placed` is that arm once its block is decided.
    fn switch_op(
        &mut self,
        tag: ValueId,
        arms: &[(Astr, Label, Vec<ValueId>)],
        default: Option<&(Label, Vec<ValueId>)>,
    ) -> Box<dyn Op> {
        let src = self.off(tag);
        let through = self.is_ref(tag);
        let form = variant_form(self.scrutinee_ty(tag));
        let (tested, default) = match default {
            Some((label, args)) => (arms, self.dispatch_edge(*label, args)),
            None => {
                let ((_, label, args), tested) = arms
                    .split_last()
                    .expect("a Switch names at least one successor: an arm or a default");
                (tested, self.dispatch_edge(*label, args))
            }
        };
        let placed: Vec<Placed> = tested
            .iter()
            .map(|(key, label, args)| Placed {
                key: *key,
                block: self.dispatch_edge(*label, args),
            })
            .collect();

        match form {
            VariantForm::Option => {
                let on_some = self.side(&placed, default, "Some");
                let on_none = self.side(&placed, default, "None");
                match through {
                    true => Box::new(switch::SwitchOption::<true> {
                        src,
                        on_some,
                        on_none,
                    }) as Box<dyn Op>,
                    false => Box::new(switch::SwitchOption::<false> {
                        src,
                        on_some,
                        on_none,
                    }),
                }
            }
            VariantForm::Result => {
                let on_ok = self.side(&placed, default, "Ok");
                let on_err = self.side(&placed, default, "Err");
                match through {
                    true => {
                        Box::new(switch::SwitchResult::<true> { src, on_ok, on_err }) as Box<dyn Op>
                    }
                    false => Box::new(switch::SwitchResult::<false> { src, on_ok, on_err }),
                }
            }
            VariantForm::Enum => {
                let arms: Box<[switch::Arm]> = placed
                    .iter()
                    .map(|arm| switch::Arm {
                        key: arm.key,
                        target: arm.block,
                    })
                    .collect();
                match through {
                    true => Box::new(switch::Switch::<true> { src, arms, default }) as Box<dyn Op>,
                    false => Box::new(switch::Switch::<false> { src, arms, default }),
                }
            }
        }
    }

    /// A suspending operation is excluded along with the terminators: it
    /// leaves the block for the driver, which the machine's dispatch loop
    /// alone can reach.
    fn is_straight_line(&self, inst: &Inst) -> bool {
        match &inst.kind {
            InstKind::Jump { .. }
            | InstKind::JumpIf { .. }
            // RFC-0051: a `Switch` is a terminator, so it ends its block
            // and `straight_run` admits it into no region's part. A `For`
            // is one too (RFC-0057).
            | InstKind::Switch { .. }
            | InstKind::For { .. }
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
            | InstKind::Cast { .. }
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
    fn blocks(&mut self, range: Range<usize>, nested: &[Region]) -> Box<[Box<dyn Op>]> {
        let runs = self.fused_in(range.clone(), nested);
        let chains = self.chains_in(range.clone(), nested);
        let units = self.layout(range, nested, &runs, &chains);
        let split = self.split(&units);
        let rides = self.rides_in(&units, Some(&split));
        self.level = Level {
            block_of_inst: block_heads(&units, &split),
            blocks_len: split.blocks,
            edges: Vec::new(),
        };

        let mut blocks: Vec<Box<dyn Op>> = Vec::with_capacity(split.blocks as usize);
        let mut ops: Vec<Node> = Vec::new();
        for (at, unit) in units.iter().enumerate() {
            let next = Next(match at + 1 < units.len() {
                true => Some(split.block_of[at + 1]),
                false => None,
            });
            let end = self.unit(unit, next, &rides, &mut ops);
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
            blocks.push(chain(mem::take(&mut ops), end));
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
        leaving: Vec<Node>,
    ) -> Box<dyn Op> {
        self.part(range, nested, leaving, None).0
    }

    /// The part whose region reads the word it ends with — a `while`'s head
    /// and the condition the `Loop` tests. Where that word is what the
    /// part's last operation produced, it rides out of the part in the
    /// argument register the `Yield` hands back, and the region is built at
    /// `Where::Register`; otherwise it reaches its frame register as any
    /// other value does.
    fn handing(
        &mut self,
        range: Range<usize>,
        nested: &[Region],
        leaving: Vec<Node>,
        hands: ValueId,
    ) -> (Box<dyn Op>, Where) {
        let (head, rode) = self.part(range, nested, leaving, Some(hands));
        let at = match rode {
            true => Where::Register,
            false => Where::Frame(self.off(hands)),
        };
        (head, at)
    }

    /// One region part as a chain, and whether the word it hands its region
    /// rode out of it.
    fn part(
        &mut self,
        range: Range<usize>,
        nested: &[Region],
        leaving: Vec<Node>,
        hands: Option<ValueId>,
    ) -> (Box<dyn Op>, bool) {
        let runs = self.fused_in(range.clone(), nested);
        let chains = self.chains_in(range.clone(), nested);
        let units = self.layout(range, nested, &runs, &chains);
        let mut rides = self.rides_in(&units, None);
        // The part's last operation is the one whose word `Yield` hands
        // back, so a word rides out only where that operation produced it
        // and no move follows it.
        let rode = hands.is_some()
            && leaving.is_empty()
            && units.last().and_then(|unit| self.rides_from(unit)) == hands;
        if let (true, Some(value)) = (rode, hands) {
            rides.add(value);
        }
        let mut ops: Vec<Node> = Vec::new();
        for unit in &units {
            let end = self.unit(unit, Next(None), &rides, &mut ops);
            assert!(
                end.is_none(),
                "a region's part holds a terminator, which `straight_run` does not admit"
            );
        }
        ops.extend(leaving);
        (chain(ops, Box::new(control::Yield)), rode)
    }

    /// The operations this unit appends, and the terminator it is where it
    /// is one. A call that yields an RFC-0007 `Order` appends the `Merge`
    /// that writes it and then ends the block, which is why a unit hands
    /// its operations to the block rather than returning one.
    fn unit(
        &mut self,
        unit: &Unit<'_>,
        next: Next,
        rides: &Rides,
        ops: &mut Vec<Node>,
    ) -> Option<Box<dyn Op>> {
        match unit {
            Unit::Inst(at) => self.op(*at, next, rides, ops),
            Unit::Region(Region::Loop(region)) => {
                self.loop_op(region, ops);
                None
            }
            Unit::Region(Region::Diamond(region)) => {
                let op = match self.select_op(region, rides) {
                    Some(op) => op,
                    None => self.diamond_op(region, rides),
                };
                ops.push(op);
                None
            }
            Unit::Fused(region) => {
                let op = self.fused_op(region);
                ops.push(op);
                None
            }
            Unit::Chain(run) => {
                let op = self.chain_op(run, rides);
                ops.push(op);
                None
            }
        }
    }

    fn loop_op(&mut self, region: &LoopRegion, ops: &mut Vec<Node>) {
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

        let (head, cond) = self.handing(
            region.head_block.clone(),
            &region.head_regions,
            Vec::new(),
            *cond,
        );
        let ran = self.straight(region.body_block.clone(), &region.body_regions, back);
        debug_assert_eq!(
            self.references(*then_label),
            vec![region.jump_if],
            "the move into a loop's body is placed at the head of the body, \
             which `recognize_loop` admits only where the test above is its one entry"
        );
        let body = chain(into_body, ran);

        ops.push(made(move |next| match cond {
            Where::Frame(off) => Box::new(control::Loop::<place::Slot> {
                head,
                cond: off,
                body,
                next,
                at: PhantomData,
            }) as Box<dyn Op>,
            Where::Register => Box::new(control::Loop::<place::R0> {
                head,
                cond: (),
                body,
                next,
                at: PhantomData,
            }),
        }));
        ops.extend(exit);
    }

    /// The diamond `region` is, where it is a select: one arm a single
    /// arithmetic or comparison node, the other arm the test's own edge into
    /// a join of one word parameter.
    ///
    /// Every refusal below is a soundness statement, because the node runs on
    /// both paths once this answers `Some`. `/` and `%` are refused because
    /// RFC-0037 names them as the only integer operations that can raise, and
    /// an arm of more than one operation is refused because its intermediate
    /// value would reach a register on a path the program does not take, and
    /// `assign_slots` — which runs before this recognizer — may have given
    /// that register to a value live outside the arm.
    fn select_shape(&self, region: &DiamondRegion) -> Option<SelectShape> {
        let insts = self.body.insts.as_slice();
        let InstKind::JumpIf {
            cond,
            then_args,
            else_args,
            ..
        } = &insts[region.jump_if].kind
        else {
            return None;
        };
        let Sides {
            arm_block,
            arm_regions,
            arm_jump,
            passed_args,
            computes_on_true,
        } = Sides::of(
            region,
            EdgeArgs {
                on_true: then_args,
                on_false: else_args,
            },
        )?;
        if !arm_regions.is_empty() {
            return None;
        }

        let InstKind::BlockLabel { params, .. } = &insts[region.join].kind else {
            return None;
        };
        let [dst] = params.as_slice() else {
            return None;
        };
        let joined = self.ty(*dst);
        if word_kind(joined).is_none() || owns_large(joined) {
            return None;
        }
        let [passed] = passed_args else {
            return None;
        };
        let InstKind::Jump { args, .. } = &insts[arm_jump].kind else {
            return None;
        };
        let [handed] = args.as_slice() else {
            return None;
        };

        let mut work = arm_block.filter(|at| !self.konsts.holds_inst(*at));
        let at = work.next()?;
        if work.next().is_some() {
            return None;
        }
        let InstKind::BinOp {
            dst: computed,
            op,
            left,
            right,
        } = &insts[at].kind
        else {
            return None;
        };
        if computed != handed || self.use_count(*computed) != 1 {
            return None;
        }
        let root = match (arith_of(*op), compare_of(*op)) {
            (Some(Arith::Div | Arith::Rem), _) => return None,
            (Some(op), None) => Root::Num(op),
            (None, Some(how)) => Root::Cmp(how),
            _ => return None,
        };
        let ty = chain_ty(self.ty(*left))?;
        if chain_ty(self.ty(*right)) != Some(ty) {
            return None;
        }

        Some(SelectShape {
            cond: *cond,
            dst: *dst,
            ty,
            root,
            left: *left,
            right: *right,
            passed: *passed,
            computes_on_true,
        })
    }

    fn select_op(&self, region: &DiamondRegion, rides: &Rides) -> Option<Node> {
        let shape = self.select_shape(region)?;
        let places = select::Places {
            cond: self.place_of(rides, shape.cond),
            dst: self.place_of(rides, shape.dst),
        };
        let arms = select::Arms {
            passed: self.off(shape.passed),
            computes_on_true: shape.computes_on_true,
        };
        let mut leaves =
            [ChainBounds::byte_offset_of_word(self.off(shape.left)); ChainBounds::MAX_LEAVES];
        leaves[1] = ChainBounds::byte_offset_of_word(self.off(shape.right));
        let plan = Plan {
            shape: Shape::NLL,
            root: shape.root,
            ops: [Arith::Add; ChainBounds::MAX_INTERIOR],
            leaves,
            reads: Reads::Own,
        };
        let ty = shape.ty;
        Some(made(move |next| {
            select::select_op(ty, places, plan, arms, next)
        }))
    }

    fn diamond_op(&mut self, region: &DiamondRegion, rides: &Rides) -> Node {
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

        let cond = self.place_of(rides, *cond);
        let on_true = self.arm(&region.on_true, then_label, then_args);
        let on_false = self.arm(&region.on_false, else_label, else_args);

        made(move |next| match cond {
            Where::Frame(off) => Box::new(control::Diamond::<place::Slot> {
                cond: off,
                on_true,
                on_false,
                next,
                at: PhantomData,
            }) as Box<dyn Op>,
            Where::Register => Box::new(control::Diamond::<place::R0> {
                cond: (),
                on_true,
                on_false,
                next,
                at: PhantomData,
            }),
        })
    }

    /// `label` and `args` are the diamond's own edge into this arm, which
    /// a `Direct` arm takes all the way to the join.
    fn arm(&mut self, region: &ArmRegion, label: &Label, args: &[ValueId]) -> Box<dyn Op> {
        let ArmRegion::Block {
            block,
            regions,
            jump,
        } = region
        else {
            let join = self.move_ops(label, args);
            return chain(join, Box::new(control::Yield));
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
        rides: &Rides,
        ops: &mut Vec<Node>,
    ) -> Option<Box<dyn Op>> {
        let body = self.body;
        let inst = &body.insts[at];
        let op: Node = match &inst.kind {
            // -- The terminators ----------------------------------------
            InstKind::Switch { tag, arms, default } => {
                return Some(self.switch_op(*tag, arms, default.as_ref()));
            }

            // RFC-0057: the machine has no `For` operation yet. Expanding one
            // here into the comparison, the element read and the advance
            // needs a counter register initialized above the header, and the
            // header block is the only block this arm writes: the preheader's
            // `Mov`s and `move_ops`' parameter positions are outside it. The
            // machine run gives `For` the region op that owns the index
            // register (RFC-0057 Decision 3).
            InstKind::For { source, .. } => {
                panic!(
                    "a `for` over {source:?} cannot be prepared: the machine has no `For` \
                     operation yet (RFC-0057 Decision 3); the script is refused before it runs"
                )
            }

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
                let cond = self.place_of(rides, *cond);
                let on_true = self.target(then_label);
                let on_false = self.target(else_label);
                let then_moves = self.move_ops(then_label, then_args);
                let else_moves = self.move_ops(else_label, else_args);
                let on_true = self.edge(then_moves, on_true);
                let on_false = self.edge(else_moves, on_false);
                return Some(match cond {
                    Where::Frame(off) => Box::new(control::JumpIf::<place::Slot> {
                        cond: off,
                        on_true,
                        on_false,
                        at: PhantomData,
                    }) as Box<dyn Op>,
                    Where::Register => Box::new(control::JumpIf::<place::R0> {
                        cond: (),
                        on_true,
                        on_false,
                        at: PhantomData,
                    }),
                });
            }
            InstKind::Return { value, .. } => {
                let slot = self.off(*value);
                return Some(match word_kind(self.ty(*value)).is_some() {
                    true => Box::new(control::Return::<true> { slot }) as Box<dyn Op>,
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
                {
                    let dst = self.off(*dst);
                    node(move |next| string::Concat {
                        dst,
                        parts: held,
                        owns_large,
                        next,
                    })
                }
            }
            InstKind::StringEq { dst, a, b } => {
                let slots = Binary {
                    dst: self.off(*dst),
                    l: self.off(*a),
                    r: self.off(*b),
                };
                node(move |next| string::StringEq { slots, next })
            }
            InstKind::StringClone { dst, src } => {
                let slots = Unary {
                    dst: self.off(*dst),
                    src: self.off(*src),
                };
                match self.is_ref(*src) {
                    true => node(move |next| string::CloneString::<true> { slots, next }),
                    false => node(move |next| string::CloneString::<false> { slots, next }),
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
                    (true, false, true) => {
                        node(move |next| string::CloneString::<false> { slots, next })
                    }
                    (true, true, true) => {
                        node(move |next| string::CloneString::<true> { slots, next })
                    }
                    (true, true, false) => node(move |next| storage::TakeThrough { slots, next }),
                    (true, false, false) => match self.owns(*dst) {
                        true => node(move |next| storage::TakeVar::<true> { slots, next }),
                        false => node(move |next| storage::TakeVar::<false> { slots, next }),
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
                    true => node(move |next| storage::Fetch::<true> {
                        dst: slot,
                        key,
                        next,
                    }),
                    false => node(move |next| storage::Fetch::<false> {
                        dst: slot,
                        key,
                        next,
                    }),
                }
            }
            InstKind::Commit { context, value } => {
                let key = self.ctx.page_key(context);
                let src = self.off(*value);
                match self.owns(*value) {
                    true => node(move |next| storage::Commit::<true> { src, key, next }),
                    false => node(move |next| storage::Commit::<false> { src, key, next }),
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
                let places = place::Binary {
                    dst: self.place_of(rides, *dst),
                    l: self.place_of(rides, *left),
                    r: self.place_of(rides, *right),
                };
                let (op, k) = (*op, self.ty(*left).clone());
                made(move |next| match k {
                    Ty::Int(k) => arith::int_binop(op, k, places, next),
                    // A `char`'s word is its scalar value as a `u32` and
                    // Rust orders a `char` by it (RFC-0058), so `==` and
                    // `<` on chars are the `u32` operations.
                    Ty::Char => arith::int_binop(op, IntTy::U32, places, next),
                    Ty::Float => arith::float_binop(op, places, next),
                    Ty::Bool => arith::bool_binop(op, places, next),
                    other => panic!("binop {op:?} on {other:?}"),
                })
            }
            InstKind::UnaryOp { dst, op, operand } => {
                let places = place::Unary {
                    dst: self.place_of(rides, *dst),
                    src: self.place_of(rides, *operand),
                };
                let (op, k) = (*op, self.ty(*operand).clone());
                made(move |next| match k {
                    Ty::Int(k) => arith::int_unaryop(op, k, places, next),
                    Ty::Float => arith::float_unaryop(op, places, next),
                    Ty::Bool => arith::bool_unaryop(op, places, next),
                    other => panic!("unary {op:?} on {other:?}"),
                })
            }

            InstKind::Cast { dst, src, to } => {
                let places = place::Unary {
                    dst: self.place_of(rides, *dst),
                    src: self.place_of(rides, *src),
                };
                let from = CastTy::of_ty(self.ty(*src)).unwrap_or_else(|| {
                    panic!(
                        "a cast reads {:?}, which is neither a number nor a char",
                        self.ty(*src)
                    )
                });
                let conversion = cast::Conversion {
                    from: from.word(),
                    into: to.word(),
                };
                made(move |next| cast::cast_op(conversion, places, next))
            }

            InstKind::Spawn {
                dst, callee, args, ..
            } => {
                let dst = self.off(*dst);
                match callee {
                    Callee::Direct(id) => {
                        let Operands { slots, takes } = self.taken(args);
                        let callee = *id;
                        node(move |next| call::SpawnModule {
                            dst,
                            callee,
                            args: slots,
                            takes,
                            next,
                        })
                    }
                    Callee::Extern { id, instance } => {
                        let handler = self.ctx.handler(id, *instance);
                        let window = self.window(at, args, ops);
                        assert_eq!(
                            handler.width().ret,
                            1,
                            "a spawn names a handler that returns a run, which would \
                             outlive the frame that lent it (RFC-0047 §3)"
                        );
                        match handler {
                            ExternHandler::Sync(f) | ExternHandler::Heavy(f) => made(move |next| {
                                f.into_op(call::CallShape::Spawn { dst, window, next })
                            }),
                            ExternHandler::Async(f) => made(move |next| {
                                f.into_op(call::AsyncShape::Spawn { dst, window, next })
                            }),
                        }
                    }
                    Callee::Indirect(_) => panic!("spawn: indirect callee not supported"),
                }
            }
            InstKind::Merge { dst, .. } => {
                let dst = self.off(*dst);
                node(move |next| control::Merge { dst, next })
            }

            InstKind::MakeArray { dst, elements } => {
                let Operands {
                    slots,
                    takes: owns_large,
                } = self.taken(elements);
                {
                    let dst = self.off(*dst);
                    node(move |next| composite::MakeArray {
                        dst,
                        elements: composite::Elements { slots, owns_large },
                        next,
                    })
                }
            }
            InstKind::MakeTuple { dst, elements } => {
                let Operands {
                    slots,
                    takes: owns_large,
                } = self.taken(elements);
                {
                    let dst = self.off(*dst);
                    node(move |next| composite::MakeTuple {
                        dst,
                        elements: composite::Elements { slots, owns_large },
                        next,
                    })
                }
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
                {
                    let dst = self.off(*dst);
                    node(move |next| composite::MakeObject {
                        dst,
                        fields: held,
                        owns_large,
                        next,
                    })
                }
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
                let key = *key;
                match self.is_ref(*src) {
                    true => node(move |next| pattern::TestObjectKey::<true> { slots, key, next }),
                    false => node(move |next| pattern::TestObjectKey::<false> { slots, key, next }),
                }
            }

            // The handler hands back two words, which go to the two
            // registers the slice's pair is (RFC-0047 amended, rule 2).
            InstKind::AsSlice {
                dst,
                container,
                instance,
                ..
            } => {
                let dst = self.pair(*dst);
                let args = std::slice::from_ref(container);
                let takes = self.take_mask(args);
                let a = self.off(*container);
                let handler = self.ctx.handler(&instance.id, instance.instance);
                let ExternHandler::Sync(f) = handler else {
                    panic!(
                        "an AsSlice names a handler of task {:?}; a run of a container's \
                         elements is lent for the caller's frame and no other task can hold \
                         it (RFC-0047 §3)",
                        handler.task()
                    )
                };
                assert_eq!(
                    f.width(),
                    Width { args: 1, ret: 2 },
                    "an AsSlice names a handler of another shape than the one container in \
                     and the two words of a run out, which is `Handler::call_slice`'s \
                     contract (RFC-0047 amended, rule 2)"
                );
                made(move |next| {
                    f.into_op(call::CallShape::Slice {
                        dst,
                        a,
                        takes,
                        next,
                    })
                })
            }
            InstKind::Index {
                dst,
                slice,
                index,
                mode,
            } => {
                let mode = *mode;
                let read = index::Read {
                    dst: self.off(*dst),
                    slice: self.pair(*slice),
                    index: self.off(*index),
                };
                made(move |next| index::checked(mode, read, next))
            }
            InstKind::IndexSet {
                slice,
                index,
                value,
            } => {
                let large = self.owns(*value);
                let (slice, index, held) = (self.pair(*slice), self.off(*index), self.off(*value));
                match large {
                    true => node(move |next| index::IndexSet::<true, true> {
                        slice,
                        index,
                        value: held,
                        next,
                    }),
                    false => node(move |next| index::IndexSet::<true, false> {
                        slice,
                        index,
                        value: held,
                        next,
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
                {
                    let dst = self.off(*dst);
                    node(move |next| call::MakeClosure {
                        dst,
                        entry,
                        captures: slots,
                        takes,
                        next,
                    })
                }
            }

            InstKind::MakeVariant { dst, tag, payload } => self.make_variant(*dst, *tag, *payload),
            InstKind::TestVariant { dst, src, tag } => {
                let slots = Unary {
                    dst: self.off(*dst),
                    src: self.off(*src),
                };
                let through = self.is_ref(*src);
                let form = variant_form(self.scrutinee_ty(*src));
                let tag = *tag;
                match (form, through) {
                    (VariantForm::Option, true) => {
                        test_option::<true>(slots, self.tag_is(tag, "Some"))
                    }
                    (VariantForm::Option, false) => {
                        test_option::<false>(slots, self.tag_is(tag, "Some"))
                    }
                    (VariantForm::Result, true) => {
                        test_result::<true>(slots, self.tag_is(tag, "Ok"))
                    }
                    (VariantForm::Result, false) => {
                        test_result::<false>(slots, self.tag_is(tag, "Ok"))
                    }
                    (VariantForm::Enum, true) => {
                        node(move |next| variant::TestVariant::<true> { slots, tag, next })
                    }
                    (VariantForm::Enum, false) => {
                        node(move |next| variant::TestVariant::<false> { slots, tag, next })
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
                        node(move |next| variant::UnwrapOption::<true> { slots, next })
                    }
                    (VariantForm::Option, false) => {
                        node(move |next| variant::UnwrapOption::<false> { slots, next })
                    }
                    (VariantForm::Result, true) => {
                        node(move |next| variant::UnwrapResult::<true> { slots, next })
                    }
                    (VariantForm::Result, false) => {
                        node(move |next| variant::UnwrapResult::<false> { slots, next })
                    }
                    (VariantForm::Enum, true) => {
                        node(move |next| variant::UnwrapVariant::<true> { slots, next })
                    }
                    (VariantForm::Enum, false) => {
                        node(move |next| variant::UnwrapVariant::<false> { slots, next })
                    }
                }
            }

            InstKind::Undef { dst } => {
                let slot = self.off(*dst);
                match SlotClass::of(self.ty(*dst)) {
                    SlotClass::Slice => {
                        let dst = SlicePair::at(slot);
                        node(move |next| control::UndefWide { dst, next })
                    }
                    SlotClass::Word(_) => {
                        node(move |next| control::Undef::<true> { dst: slot, next })
                    }
                    SlotClass::Whole => {
                        node(move |next| control::Undef::<false> { dst: slot, next })
                    }
                }
            }
            InstKind::Drop { src } => {
                let slot = self.off(*src);
                node(move |next| control::DropValue { slot, next })
            }
        };
        ops.push(op);
        None
    }

    /// The `Order` an effectful call yields, written by its own operation
    /// ahead of the call (RFC-0007). No call instance carries the register,
    /// so no call tests for one.
    fn merge(&mut self, order: Option<ValueId>, ops: &mut Vec<Node>) {
        let Some(after) = order else {
            return;
        };
        let dst = self.off(after);
        ops.push(node(move |next| control::Merge { dst, next }));
    }

    fn call_op(
        &mut self,
        site: &CallSite<'_>,
        next: Next,
        ops: &mut Vec<Node>,
    ) -> Option<Box<dyn Op>> {
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
                if !self.suspends_at(callee_ty) {
                    let laid = self.laid(args, ops);
                    ops.push(direct_call(into, id, laid));
                    return None;
                }
                let resume = next.block();
                let Operands { slots, takes } = self.taken(args);
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
                if !self.suspends_at(callee_ty) {
                    let laid = self.laid(args, ops);
                    ops.push(indirect_call(into, through, handle, laid));
                    return None;
                }
                let resume = next.block();
                let operands = self.taken(args);
                Some(indirect_call_async(into, through, handle, operands, resume))
            }
            Callee::Extern { id, instance } => {
                let handler = self.ctx.handler(id, *instance);
                if !handler.is_sync() {
                    self.may_suspend = true;
                }
                match handler {
                    ExternHandler::Sync(f) => {
                        let op = self.extern_call(at, into, args, f, ops);
                        ops.push(op);
                        None
                    }
                    ExternHandler::Heavy(f) => {
                        let window = self.window(at, args, ops);
                        let resume = next.block();
                        Some(f.into_op(call::CallShape::Heavy {
                            dst: slot,
                            window,
                            large,
                            resume,
                        }))
                    }
                    ExternHandler::Async(f) => {
                        let window = self.window(at, args, ops);
                        let resume = next.block();
                        Some(f.into_op(call::AsyncShape::Await {
                            dst: slot,
                            window,
                            large,
                            resume,
                        }))
                    }
                }
            }
        }
    }

    /// The operation one synchronous extern call runs as. The handler's
    /// `Width` names the form — how many of the runtime's values the
    /// arguments are, and how many the result is — and nothing here counts
    /// anything of its own (RFC-0059 rule 7).
    fn extern_call(
        &mut self,
        at: usize,
        into: Dest,
        args: &[ValueId],
        f: call::Handler,
        ops: &mut Vec<Node>,
    ) -> Node {
        let Dest {
            slot: dst,
            large,
            word,
        } = into;
        let width = f.width();
        assert_eq!(
            width.ret, 1,
            "a function call names a handler whose result is wider than one value; a run of \
             a container's elements reaches the machine as an AsSlice and nothing else \
             (RFC-0047 amended, rule 2)"
        );
        let takes = self.take_mask(args);
        let slots: Vec<Off> = args.iter().map(|id| self.off(*id)).collect();
        match CallForm::of(&width, args.len()) {
            CallForm::Registers(0) => {
                made(move |next| f.into_op(call::CallShape::Registers0 { dst, large, next }))
            }
            CallForm::Registers(1) => {
                let a = nth(&slots, 0);
                made(move |next| {
                    f.into_op(call::CallShape::Registers1 {
                        dst,
                        a,
                        takes,
                        large,
                        word,
                        next,
                    })
                })
            }
            CallForm::Registers(2) => {
                let (a, b) = (nth(&slots, 0), nth(&slots, 1));
                made(move |next| {
                    f.into_op(call::CallShape::Registers2 {
                        dst,
                        a,
                        b,
                        takes,
                        large,
                        next,
                    })
                })
            }
            CallForm::Registers(3) => {
                let (a, b, c) = (nth(&slots, 0), nth(&slots, 1), nth(&slots, 2));
                made(move |next| {
                    f.into_op(call::CallShape::Registers3 {
                        dst,
                        a,
                        b,
                        c,
                        takes,
                        large,
                        next,
                    })
                })
            }
            CallForm::Registers(_) | CallForm::Window => {
                let window = self.window(at, args, ops);
                made(move |next| {
                    f.into_op(call::CallShape::Window {
                        dst,
                        window,
                        large,
                        next,
                    })
                })
            }
        }
    }

    /// The argument run `assign_slots` placed for the call at `at`, and the
    /// `Mov` operations that put the arguments in it — which the caller
    /// pushes before the call's own operation (RFC-0052 rule 1).
    fn window(&mut self, at: usize, args: &[ValueId], ops: &mut Vec<Node>) -> call::ArgWindow {
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
                    moved: match SlotClass::of(self.ty(arg)) {
                        SlotClass::Slice => Moved::Pair,
                        SlotClass::Whole | SlotClass::Word(_) => match self.owns(arg) {
                            true => Moved::Large,
                            false => Moved::Whole,
                        },
                    },
                }
            })
            .collect();
        let ordered = order_moves(pairs, self.scratch_slot());
        self.scratch_used = self.scratch_used.max(ordered.scratch_used);

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
        let mut slot = base;
        for id in args {
            let width = u16::try_from(SlotClass::of(self.ty(*id)).width())
                .expect("a register class is two wide at most");
            if self.owns(*id) {
                assert!(
                    slot < crate::regs::MARK_WORD_SLOTS,
                    "an argument run reaches register {slot}, which is outside mark word 0"
                );
                mask |= 1u64 << slot;
            }
            slot = slot
                .checked_add(width)
                .unwrap_or_else(|| panic!("an argument run at register {base} leaves a frame"));
        }
        mask
    }

    /// What a move of this value between two registers the frame opened
    /// carries. `word_kind` is the same predicate `slot_kinds` opened them
    /// by, so a `Moved::Word` move writes the width every other write of
    /// those registers takes.
    fn moved(&self, arg: ValueId) -> Moved {
        match SlotClass::of(self.ty(arg)) {
            SlotClass::Slice => Moved::Pair,
            SlotClass::Word(_) => Moved::Word,
            SlotClass::Whole => match self.owns(arg) {
                true => Moved::Large,
                false => Moved::Whole,
            },
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

    fn constant(&mut self, dst: ValueId, value: &Literal) -> Node {
        let out = self.off(dst);
        let word = match (value, self.ty(dst)) {
            (Literal::Int(n), Ty::Int(_)) => *n as u64,
            (Literal::Int(n), other) => panic!("integer literal {n} typed as {other:?}"),
            (Literal::Float(x), _) => x.to_bits(),
            (Literal::Bool(b), _) => u64::from(*b),
            (Literal::Unit, _) => 0,
            (Literal::String(s), _) => {
                let konst = Konst::Str(s.clone());
                return node(move |next| constant::ConstLarge {
                    dst: out,
                    konst,
                    next,
                });
            }
            (Literal::List(items), Ty::Array(elem, _)) => {
                let konst = Konst::List(items.iter().map(|item| konst_of(item, elem)).collect());
                return node(move |next| constant::ConstLarge {
                    dst: out,
                    konst,
                    next,
                });
            }
            (Literal::List(_), other) => panic!("list literal typed as {other:?}"),
            (Literal::Char(c), _) => u64::from(u32::from(*c)),
            (sugar @ (Literal::IntOf(_) | Literal::Bytes(_)), _) => {
                return self.constant(dst, &sugar.desugared());
            }
        };
        node(move |next| constant::Const {
            dst: out,
            word,
            next,
        })
    }

    fn test_literal(&mut self, at: Tested, value: &Literal) -> Node {
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
                for_int_ty!(*k, |T| node(move |next| pattern::TestInt::<T>::new(
                    slots, want, next
                )))
            }
            Literal::Float(x) => {
                let want = *x;
                match through {
                    true => node(move |next| pattern::TestFloat::<true> { slots, want, next }),
                    false => node(move |next| pattern::TestFloat::<false> { slots, want, next }),
                }
            }
            Literal::Bool(b) => {
                let want = *b;
                match through {
                    true => node(move |next| pattern::TestBool::<true> { slots, want, next }),
                    false => node(move |next| pattern::TestBool::<false> { slots, want, next }),
                }
            }
            Literal::String(s) => {
                let want = s.clone();
                match through {
                    true => node(move |next| pattern::TestString::<true> { slots, want, next }),
                    false => node(move |next| pattern::TestString::<false> { slots, want, next }),
                }
            }
            Literal::Unit => node(move |next| pattern::TestUnit {
                dst: slots.dst,
                next,
            }),
            Literal::List(_) => panic!("TestLiteral on a list literal"),
            Literal::Char(c) => {
                let want = i128::from(u32::from(*c));
                node(move |next| pattern::TestInt::<u32>::new(slots, want, next))
            }
            sugar @ (Literal::IntOf(_) | Literal::Bytes(_)) => {
                self.test_literal(at, &sugar.desugared())
            }
        }
    }

    fn make_variant(&mut self, dst: ValueId, tag: Astr, payload: Option<ValueId>) -> Node {
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
                true => node(move |next| variant::MakeSome::<true> { slots, next }),
                false => node(move |next| variant::MakeSome::<false> { slots, next }),
            },
            (Ty::Option(_), None) => node(move |next| variant::MakeNone { dst: out, next }),
            (Ty::Result(..), Some(slots)) => match (self.tag_is(tag, "Ok"), large) {
                (true, true) => node(move |next| variant::MakeOk::<true> { slots, next }),
                (true, false) => node(move |next| variant::MakeOk::<false> { slots, next }),
                (false, true) => node(move |next| variant::MakeErr::<true> { slots, next }),
                (false, false) => node(move |next| variant::MakeErr::<false> { slots, next }),
            },
            (Ty::Result(..), None) => panic!("Ok and Err carry a payload"),
            (_, Some(slots)) => match large {
                true => node(move |next| variant::MakeVariant::<true> { slots, tag, next }),
                false => node(move |next| variant::MakeVariant::<false> { slots, tag, next }),
            },
            (_, None) => node(move |next| variant::MakeUnitVariant {
                dst: out,
                tag,
                next,
            }),
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
        (Literal::Char(c), _) => Konst::Word(Kind::Char, u64::from(u32::from(*c))),
        (sugar @ (Literal::IntOf(_) | Literal::Bytes(_)), _) => konst_of(&sugar.desugared(), ty),
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
    /// Two adjacent word registers: a slice's `ptr` and `len` (RFC-0047
    /// amended, rule 4).
    Pair,
    /// The move changes the owner of a `Large`, and the mark word with it.
    Large,
    /// Neither: one whole copy, no mark.
    Whole,
}

impl Moved {
    /// The registers this move touches at each end.
    const fn width(self) -> u32 {
        match self {
            Moved::Pair => 2,
            Moved::Word | Moved::Large | Moved::Whole => 1,
        }
    }
}

#[derive(Clone, Copy)]
struct Carried {
    at: Pair,
    moved: Moved,
}

/// The ordered move as the operation of a block (RFC-0052 rule 1): what it
/// carries is its type, not a field a `run` reads.
fn mov_op(carried: &Carried) -> Node {
    let dst = Off::of(carried.at.to);
    let src = Off::of(carried.at.from);
    match carried.moved {
        Moved::Word => node(move |next| control::Mov::<false, true> { dst, src, next }),
        Moved::Pair => {
            let (dst, src) = (SlicePair::at(dst), SlicePair::at(src));
            node(move |next| control::MovWide { dst, src, next })
        }
        Moved::Large => node(move |next| control::Mov::<true, false> { dst, src, next }),
        Moved::Whole => node(move |next| control::Mov::<false, false> { dst, src, next }),
    }
}

/// A jump's moves, ordered, and how many scratch registers a cycle through
/// them took.
struct MoveOrdering {
    moves: Vec<Carried>,
    scratch_used: u32,
}

/// Order a parallel move: every source is read before it is overwritten,
/// and a cycle is broken by moving one source into `scratch` first.
fn order_moves(pairs: Vec<Carried>, scratch: Slot) -> MoveOrdering {
    let mut pending: Vec<Carried> = pairs.into_iter().filter(|m| m.at.from != m.at.to).collect();
    let mut moves: Vec<Carried> = Vec::with_capacity(pending.len());
    let mut scratch_used = 0u32;
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
                Moved::Pair => Moved::Pair,
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
            scratch_used = scratch_used.max(through.width());
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

/// One arm of a dispatch once its block is decided: the tag the arm names,
/// and the block the machine enters for it.
struct Placed {
    key: Astr,
    block: BlockId,
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
pub(crate) struct LiveRange {
    pub(crate) lo: usize,
    pub(crate) hi: usize,
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

    pub(crate) fn overlaps(self, other: Self) -> bool {
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
            InstKind::Switch { arms, default, .. } => {
                for (_, label, _) in arms {
                    visit(self.target(label));
                }
                if let Some((label, _)) = default {
                    visit(self.target(label));
                }
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
            InstKind::Switch { arms, default, .. } => {
                for (_, label, args) in arms {
                    edge(label, args);
                }
                if let Some((label, args)) = default {
                    edge(label, args);
                }
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
        } => needs_window(&ctx.handler(id, *instance), args.len()).then_some(args.as_slice()),
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
fn needs_window(handler: &ExternHandler, args: usize) -> bool {
    match handler {
        ExternHandler::Sync(f) => matches!(CallForm::of(&f.width(), args), CallForm::Window),
        ExternHandler::Heavy(_) | ExternHandler::Async(_) => true,
    }
}

/// How a synchronous extern call hands over its arguments.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CallForm {
    /// One register operand per argument: the run fits the register forms
    /// and every argument is one value.
    Registers(usize),
    /// The argument run laid in the caller's window.
    Window,
}

impl CallForm {
    fn of(width: &Width, args: usize) -> CallForm {
        if width.in_registers() && width.args == args {
            CallForm::Registers(width.args)
        } else {
            CallForm::Window
        }
    }
}

/// A handler the preparation picks and no test runs: the fixtures below read
/// the operation's shape, never its result.
#[cfg(test)]
fn refuses_to_run(
    _: &crate::runtime::AcvusRuntime,
    _: &mut crate::regs::FrameState,
    a: Value,
    _: Value,
    _: Value,
    _: Value,
) -> Value {
    let _ = a;
    panic!("the preparation must not run a handler")
}

/// An extern lent its window: four arguments is past the register forms.
#[cfg(test)]
fn window_handler() -> ExternHandler {
    ExternHandler::sync(acvus_extern::glue4::<
        crate::runtime::AcvusRuntime,
        _,
        acvus_extern::ByValue<Value>,
        acvus_extern::ByValue<Value>,
        acvus_extern::ByValue<Value>,
        acvus_extern::ByValue<Value>,
        acvus_extern::Val<Value>,
    >(refuses_to_run))
}

/// An extern of one parameter written `&str` in Rust, which is two of the
/// runtime's values at one parameter.
#[cfg(test)]
fn str_handler() -> ExternHandler {
    ExternHandler::sync(acvus_extern::glue1::<
        crate::runtime::AcvusRuntime,
        _,
        acvus_extern::ByStr,
        acvus_extern::Val<Value>,
    >(refuses_a_str))
}

#[cfg(test)]
fn refuses_a_str(
    _: &crate::runtime::AcvusRuntime,
    _: &mut crate::regs::FrameState,
    s: &str,
) -> Value {
    let _ = s;
    panic!("the preparation must not run a handler")
}

#[cfg(test)]
mod call_form_tests {
    use super::*;

    /// The form is the handler's width and never its count of parameters: a
    /// parameter written `&str` is a pair, so one of them is already past the
    /// register forms and the call is lent its window.
    #[test]
    fn one_str_parameter_is_lent_its_window() {
        let ExternHandler::Sync(factory) = str_handler() else {
            panic!("an extern declared with a plain `fn` is a Sync handler")
        };
        let width = factory.width();
        assert_eq!(width, Width { args: 2, ret: 1 });
        assert_eq!(CallForm::of(&width, 1), CallForm::Window);

        let op = factory.into_op(call::CallShape::Window {
            dst: Off::of(2),
            window: call::ArgWindow {
                at: Off::of(0),
                arity: 2,
                takes: 0,
            },
            large: false,
            next: Box::new(crate::ops::control::Return::<false> { slot: Off::of(2) }),
        });
        let built = crate::listing::last_path_segment(&*op);
        assert!(
            built.starts_with("CallWindow"),
            "a `&str` parameter is lent its window, and the operation built for it is \
             {built}"
        );
    }
}

#[cfg(test)]
fn never_runs_awaited<'a>(
    _: &'a crate::runtime::AcvusRuntime,
    _: &'a mut &mut crate::regs::FrameState,
) -> acvus_extern::BoxFuture<'a, Value> {
    Box::pin(async { panic!("the recognizer must not run a handler") })
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
    Held(SlotClaim),
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

    fn free(&self, slot: usize, range: LiveRange, want: SlotClaim) -> bool {
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

    fn take(&mut self, slot: usize, range: LiveRange, want: SlotClaim) {
        if self.taken.len() <= slot {
            self.taken.resize_with(slot + 1, Vec::new);
            self.claimed.resize(slot + 1, Claim::Free);
        }
        self.taken[slot].push(range);
        self.claimed[slot] = Claim::Held(want);
    }

    /// Whether every register of a `class` placed at `base` is free over
    /// `range` and claimed for what that register will hold.
    fn fits(&self, base: usize, range: LiveRange, class: SlotClass) -> bool {
        (0..class.width()).all(|k| self.free(base + k, range, class.claim()))
    }

    fn hold(&mut self, base: usize, range: LiveRange, class: SlotClass) {
        for k in 0..class.width() {
            self.take(base + k, range, class.claim());
        }
    }

    fn lowest_free(&self, range: LiveRange, class: SlotClass) -> usize {
        (0..self.len())
            .find(|base| self.fits(*base, range, class))
            .unwrap_or(self.len())
    }

    /// The lowest base whose registers each hold what the argument at that
    /// position will hold, for as long as it holds it. A slice argument
    /// takes two of them, so a position is not an index.
    fn lowest_free_run(&self, window: &[WindowSlot], call: usize) -> usize {
        let fits = |base: usize| {
            let mut at = base;
            window.iter().all(|slot| {
                let held = self.fits(at, slot.place.occupies(call), slot.class);
                at += slot.class.width();
                held
            })
        };
        (0..self.len())
            .find(|base| fits(*base))
            .unwrap_or(self.len())
    }
}

/// One argument position of a window: where the value sits until the call,
/// and how many registers it takes there.
struct WindowSlot {
    place: ArgPlace,
    class: SlotClass,
}

struct ClassRange {
    class: usize,
    range: LiveRange,
}

/// Where each `ValueId` of a body lives, and what each extern call site's
/// argument window is (RFC-0044, stage 2b).
pub struct Slots {
    of: Box<[u32]>,
    class_of: Box<[SlotClass]>,
    frame: u32,
    windows: FxHashMap<usize, WindowPlan>,
    /// Each value's live range, indexed by `ValueId::to_raw`. `runs::plan` reads
    /// it so that the run placement scans the same intervals the scalar
    /// colouring did rather than recomputing them from the instruction list.
    ranges: Box<[Option<LiveRange>]>,
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

    /// The registers value `id` occupies: one, or the two of a slice pair.
    fn run_of(&self, id: usize) -> std::ops::Range<u32> {
        let base = self.of[id];
        base..base + self.class_of[id].width() as u32
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

    let classes_of: Vec<SlotClass> = (0..values)
        .map(
            |value| match body.val_types.get(&ValueId::from_raw(value)) {
                Some(ty) => SlotClass::of(ty),
                None => SlotClass::Whole,
            },
        )
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
        if classes_of[arg] != classes_of[param] {
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

    let mut class_class: Vec<Option<SlotClass>> = vec![None; values];
    for value in 0..values {
        if ranges[value].is_none() {
            continue;
        }
        let root = classes.find(value);
        match class_class[root] {
            None => class_class[root] = Some(classes_of[value]),
            Some(held) => assert_eq!(
                held, classes_of[value],
                "value {value} joined a register class opened as {held:?}"
            ),
        }
    }

    let mut slot_of: Vec<Option<u32>> = vec![None; values];
    let mut occupancy = Occupancy::new();

    // The parameters first: a caller lays the argument run it hands the call
    // in the frame's first registers (RFC-0052 rule 7, `Prepare::laid`).
    let mut run: u32 = 0;
    for (_, id) in &body.params {
        let value = id.to_raw();
        let root = classes.find(value);
        assert!(
            slot_of[root].is_none(),
            "parameter {id:?} shares a register class with a parameter before it"
        );
        let range = class_ranges[root]
            .expect("a parameter is live from the entry, which gives its class a range");
        let want =
            class_class[root].expect("a parameter's class holds the kind the parameter is of");
        let width = u32::try_from(classes_of[value].width())
            .expect("a register class is two registers at most");
        let over = run + width;
        assert!(
            over <= u32::from(crate::regs::MAX_SCALAR_SLOTS),
            "the parameters of one body reach register {over}, past the {} a scalar body colours",
            crate::regs::MAX_SCALAR_SLOTS
        );
        occupancy.hold(
            usize::try_from(run).expect("a frame's registers fit a usize"),
            range,
            want,
        );
        slot_of[root] = Some(run);
        run = over;
    }

    // The windows next: a contiguous run is the constrained resource.
    let mut plans: Vec<(usize, WindowPlan)> = Vec::new();
    for (at, inst) in insts.iter().enumerate() {
        let Some(args) = window_args(inst, ctx) else {
            continue;
        };
        let mut allocated: Vec<usize> = Vec::new();
        let window: Vec<WindowSlot> = args
            .iter()
            .map(|arg| {
                let class = classes.find(arg.to_raw());
                let dies_here = class_ranges[class].is_some_and(|range| range.hi == at);
                let free = slot_of[class].is_none() && !allocated.contains(&class);
                let place = match class_ranges[class] {
                    Some(range) if dies_here && free => {
                        allocated.push(class);
                        ArgPlace::Allocated { class, range }
                    }
                    _ => ArgPlace::Moved,
                };
                WindowSlot {
                    place,
                    class: classes_of[arg.to_raw()],
                }
            })
            .collect();
        let base = occupancy.lowest_free_run(&window, at);

        let mut moved = Vec::new();
        let mut slot = base;
        for (held, arg) in window.iter().zip(args) {
            occupancy.hold(slot, held.place.occupies(at), held.class);
            match held.place {
                ArgPlace::Allocated { class, .. } => slot_of[class] = Some(slot as u32),
                ArgPlace::Moved => moved.push(PendingMove {
                    arg: *arg,
                    to: slot as u32,
                }),
            }
            slot += held.class.width();
        }
        plans.push((
            at,
            WindowPlan {
                base: base as u32,
                arity: (slot - base) as u32,
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
        let want = class_class[class].unwrap_or(SlotClass::Whole);
        let slot = occupancy.lowest_free(range, want);
        occupancy.hold(slot, range, want);
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
    let frame = (0..values)
        .filter(|value| of[*value] != NO_SLOT)
        .map(|value| of[value] + classes_of[value].width() as u32)
        .max()
        .unwrap_or(0);

    let windows = plans.into_iter().collect();

    let slots = Slots {
        of: of.into_boxed_slice(),
        class_of: classes_of.into_boxed_slice(),
        frame,
        windows,
        ranges: ranges.into_boxed_slice(),
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
            if slots.of[value] == NO_SLOT {
                continue;
            }
            for slot in slots.run_of(value) {
                if let Some(other) = seen.insert(slot, value) {
                    panic!(
                        "slot {slot} holds values {value} and {other}, both live {which} \
                         instruction {at}"
                    );
                }
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
        let overlaps = |value: usize| {
            slots.of[value] != NO_SLOT && slots.run_of(value).any(|slot| run.contains(&slot))
        };
        for value in live.live_in[at].iter() {
            assert!(
                !overlaps(value) || args.contains(&value),
                "value {value} is live into instruction {at} in a slot that call's \
                 argument window overwrites"
            );
        }
        for value in live.live_out[at].iter() {
            assert!(
                !overlaps(value),
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
            let handler = ExternHandler::awaited(acvus_extern::async_glue0::<
                crate::runtime::AcvusRuntime,
                _,
            >(never_runs_awaited));
            self.externs.insert(id, Executable::Extern(vec![handler]));
            id
        }

        fn sync_extern(&mut self, name: &str) -> QualifiedRef {
            let id = QualifiedRef {
                namespace: None,
                name: self.interner.intern(name),
            };
            let handler = window_handler();
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
        let by_value = ExternHandler::sync(acvus_extern::glue1::<
            crate::runtime::AcvusRuntime,
            _,
            acvus_extern::ByValue<Value>,
            acvus_extern::Val<Value>,
        >(|_, _, v| v));
        let window = window_handler();
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
        assert_eq!(order_moves(pairs, fits(slots.frame)).scratch_used, 1);
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
        assert_eq!(ordering.scratch_used, 0);
    }

    #[test]
    fn a_cycle_goes_through_the_scratch_slot() {
        let ordering = order_moves(carried(&[(0, 1), (1, 0)]), 9);
        assert_eq!(emitted(&ordering), vec![(0, 9), (1, 0), (9, 1)]);
        assert_eq!(ordering.scratch_used, 1);
    }

    #[test]
    fn a_self_move_is_no_move() {
        let ordering = order_moves(carried(&[(3, 3)]), 9);
        assert!(ordering.moves.is_empty());
    }

    #[test]
    fn two_cycles_reuse_the_one_scratch_slot() {
        let ordering = order_moves(carried(&[(0, 1), (1, 0), (2, 3), (3, 2)]), 9);
        assert_eq!(ordering.scratch_used, 1);
        assert_eq!(ordering.moves.len(), 6);
    }
}

// -- The arithmetic-chain recognizer (RFC-0044, stage 4) ----------------

enum Tree {
    Leaf(Off, LeafRead),
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
            Tree::Leaf(..) | Tree::Unread => out.push('L'),
            Tree::Node { left, right, .. } => {
                out.push('N');
                left.word(out);
                right.word(out);
            }
        }
    }

    fn flatten_into(&self, out: &mut Flattened) {
        match self {
            Tree::Leaf(slot, read) => out.leaves.push(Some(Leaf {
                slot: *slot,
                read: *read,
            })),
            Tree::Unread => out.leaves.push(None),
            Tree::Node { op, left, right } => {
                left.flatten_into(out);
                right.flatten_into(out);
                out.ops.push(*op);
            }
        }
    }
}

/// One leaf of a chain: the register it reads, and the type it reads it
/// at, which is the chain's own unless a cast the chain absorbed says
/// otherwise (RFC-0049).
#[derive(Clone, Copy)]
struct Leaf {
    slot: Off,
    read: LeafRead,
}

#[derive(Default)]
struct Flattened {
    ops: Vec<Arith>,
    leaves: Vec<Option<Leaf>>,
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

        let any = leaves
            .iter()
            .flatten()
            .next()
            .copied()
            .unwrap_or_else(|| panic!("a chain of shape {shape:?} reads no register"));
        let in_space = |at: Off| ChainBounds::byte_offset_of_word(remap(at));

        let mut ops = [Arith::Add; ChainBounds::MAX_INTERIOR];
        ops[..found.len()].copy_from_slice(&found);
        let mut offsets = [in_space(any.slot); ChainBounds::MAX_LEAVES];
        let mut reads = [any.read; ChainBounds::MAX_LEAVES];
        for ((offset, read), leaf) in offsets.iter_mut().zip(reads.iter_mut()).zip(&leaves) {
            let leaf = leaf.unwrap_or(any);
            *offset = in_space(leaf.slot);
            *read = leaf.read;
        }

        Plan {
            shape,
            root: self.node.op,
            ops,
            leaves: offsets,
            reads: Reads::of(reads),
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

    /// The leaf `value` becomes. A cast into the chain's own type is the
    /// leaf's read and adds no node (RFC-0049 rule 3), so a chain that
    /// absorbs one stays at one `T`; anything else reads the register the
    /// operation that computes it writes.
    fn leaf(&mut self, value: ValueId) -> Tree {
        let Some((at, src, from)) = self.cast_into(value) else {
            return Tree::Leaf(self.prep.leaf_slot(value), LeafRead::Own);
        };
        self.absorbed.push(at);
        Tree::Leaf(self.prep.leaf_slot(src), LeafRead::Cast(from))
    }

    /// The cast `value` is, where it is one this chain may read through:
    /// its instruction, the register it reads, and the type it reads at.
    fn cast_into(&self, value: ValueId) -> Option<(usize, ValueId, ChainTy)> {
        let at = self.prep.def_at(value)?;
        if at < self.floor || at >= self.root || self.prep.use_count(value) != 1 {
            return None;
        }
        let InstKind::Cast { src, to, .. } = &self.prep.body.insts[at].kind else {
            return None;
        };
        let from = chain_ty(self.prep.ty(*src))?;
        (chain_ty(&Ty::from(*to)) == Some(self.ty)).then_some((at, *src, from))
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
            return self.leaf(value);
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

/// Whether a value of this type is a slice: a borrow of a run, which the
/// machine keeps in two adjacent word registers (RFC-0047 amended).
/// Whether a value of this type is the register pair the machine keeps a
/// run in: a container's elements (RFC-0047) or a `String`'s bytes
/// (RFC-0062 Decision 1).
fn is_slice(ty: &Ty) -> bool {
    matches!(ty, Ty::Ref(_, target) if matches!(target.ty, Ty::Slice(_) | Ty::Str))
}

/// How many registers a value takes, and what the frame opens them with
/// (RFC-0052 rule 5, RFC-0047 amended rule 1). Two values share a register
/// only where this is the same, which is what keeps a kind byte written
/// once true.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum SlotClass {
    /// Written whole every time, so the frame opens no kind ahead of it.
    Whole,
    /// One word register the frame opens with this kind.
    Word(Kind),
    /// A slice: `ptr` then `len`, adjacent, both word class, no mark bit.
    Slice,
}

/// What one register was claimed for, which is `slot_kinds`'s entry for it.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum SlotClaim {
    Whole,
    Word(Kind),
}

impl SlotClass {
    fn of(ty: &Ty) -> SlotClass {
        if is_slice(ty) {
            return SlotClass::Slice;
        }
        match word_kind(ty) {
            Some(kind) => SlotClass::Word(kind),
            None => SlotClass::Whole,
        }
    }

    const fn width(self) -> usize {
        match self {
            SlotClass::Slice => 2,
            SlotClass::Whole | SlotClass::Word(_) => 1,
        }
    }

    /// Decided against a `Kind::Ref` for a slice's pointer register: a frame
    /// opens a kind through `Value::inline`, which carries no `Kind::Ref`.
    /// That no operation reads a pair register's kind byte is pinned by
    /// `acvus-interpreter-test/tests/slice_pair.rs`.
    fn claim(self) -> SlotClaim {
        match self {
            SlotClass::Whole => SlotClaim::Whole,
            SlotClass::Word(kind) => SlotClaim::Word(kind),
            SlotClass::Slice => SlotClaim::Word(Kind::U64),
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
        Ty::Char => Some(Kind::Char),
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
        (Literal::Char(c), Ty::Char) => Some(Value::char_(*c)),
        (sugar @ (Literal::IntOf(_) | Literal::Bytes(_)), _) => konst_value(&sugar.desugared(), ty),
        (Literal::Int(_), _)
        | (Literal::Float(_), _)
        | (Literal::Char(_), _)
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
    handler: call::Handler,
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
            _ => return None,
        };
        let ExternHandler::Sync(handler) = self.ctx.handler(id, instance) else {
            return None;
        };
        let width = handler.width();
        if width.ret != 1 || width.args > 2 || width.args != args.len() {
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

    fn fused_op(&mut self, region: &FusedRegion) -> Node {
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
            let f = found.handler;
            let shape = match f.width().args {
                0 => call::FusedShape::Nullary,
                1 => call::FusedShape::Unary { a: args[0] },
                2 => call::FusedShape::Binary {
                    a: args[0],
                    b: args[1],
                },
                other => panic!(
                    "fusable_call admitted {other} arguments, which a fused run holds no \
                     shape for"
                ),
            };
            calls.push(f.into_fused(shape));
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
        let (large, at) = (self.owns(dst), self.off(dst));
        made(move |next| call::fused(large, at, calls, tail, takes, next))
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

    fn chain_op(&mut self, run: &ChainRun, rides: &Rides) -> Node {
        self.check_chain(run);
        let (ty, plan) = (run.ty, run.plan(&|slot| slot));
        let dst = self.place_of(rides, run.dst);
        made(move |next| chain::chain_op(ty, dst, plan, next))
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
    /// carries the type its leaf reads it at — the chain's own, or the
    /// source type of a cast the chain absorbed (RFC-0049). This is the
    /// chain-shaped half of what `check_assignment` states for slots.
    fn check_chain(&self, run: &ChainRun) {
        for leaf in run.node.flatten().leaves {
            let Some(Leaf { slot, read }) = leaf else {
                continue;
            };
            assert!(
                (slot.index() as u32) < self.scratch + u32::from(self.scratch_used),
                "a chain reads register {}, which its body's frame does not have",
                slot.index()
            );
            let value = (self.body.insts[run.insts.start..run.insts.end])
                .iter()
                .flat_map(|inst| inst_info::uses(&inst.kind))
                .find(|value| self.leaf_slot(*value) == slot)
                .unwrap_or_else(|| {
                    panic!(
                        "a chain reads register {}, which no instruction it replaces reads",
                        slot.index()
                    )
                });
            let expected = match read {
                LeafRead::Own => run.ty,
                LeafRead::Cast(from) => from,
            };
            assert_eq!(
                chain_ty(self.ty(value)),
                Some(expected),
                "a chain reads register {} at another type than its leaf names",
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
        for Leaf { slot, .. } in run.node.flatten().leaves.iter().flatten() {
            let at = match body
                .params
                .iter()
                .position(|(_, id)| *slot == self.off(*id))
            {
                Some(at) => at,
                None => {
                    let raw = u32::try_from(slot.index())
                        .expect("a frame's register index fits the konst table's key");
                    let konst = self.konsts.value_at.get(&raw)?;
                    body.params.len() + intern(&mut konsts, konst)
                }
            };
            let at = u16::try_from(at).expect("a body reads at most u16::MAX operands");
            in_space.insert(*slot, Off::of(at));
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

/// The registers an argument run occupies in the callee's frame, and the mask
/// of the caller's registers the call takes the frame's claim on.
struct Laid {
    arity: u16,
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
        Ty::Int(_) | Ty::Float | Ty::Char | Ty::Bool | Ty::Unit | Ty::Never | Ty::Order => false,
        Ty::Ref(..) => false,
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
        Ty::Slice(_) | Ty::Str => panic!(
            "a slice has no storage of its own: it reaches a register under a reference \
             (RFC-0047, RFC-0062)"
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

fn make_ref_step<const THROUGH: bool>(slots: Unary, one: Walked) -> Node {
    at_step!(one, |step| node(move |next| storage::MakeRefStep::<
        _,
        THROUGH,
    > {
        slots,
        step,
        next
    }))
}

fn read_step<M>(slots: Unary, one: Walked) -> Node
where
    M: storage::Reads,
{
    at_step!(one, |step| node(move |next| storage::ReadStep::<_, M> {
        slots,
        step,
        mode: PhantomData,
        next
    }))
}

fn assign_step<const THROUGH: bool, const LARGE: bool>(slots: storage::Write, one: Walked) -> Node {
    at_step!(one, |step| node(move |next| storage::AssignStep::<
        _,
        THROUGH,
        LARGE,
    > {
        slots,
        step,
        next
    }))
}

fn set_step<const LARGE: bool>(slots: storage::Update, one: Walked) -> Node {
    at_step!(one, |step| node(move |next| storage::SetStep::<_, LARGE> {
        slots,
        step,
        next
    }))
}

// -- The place operations ----------------------------------------------

fn make_ref(slots: Unary, through: bool, path: &[Walked]) -> Node {
    match (through, path) {
        (false, []) => node(move |next| storage::MakeRef::<false> { slots, next }),
        (true, []) => node(move |next| storage::MakeRef::<true> { slots, next }),
        (false, [one]) => make_ref_step::<false>(slots, *one),
        (true, [one]) => make_ref_step::<true>(slots, *one),
        (false, many) => {
            let steps = steps_of(many);
            node(move |next| storage::MakeRefPath::<false> { slots, steps, next })
        }
        (true, many) => {
            let steps = steps_of(many);
            node(move |next| storage::MakeRefPath::<true> { slots, steps, next })
        }
    }
}

fn read_place(slots: Unary, how: Reading, path: &[Walked]) -> Node {
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

fn read_at<M>(slots: Unary, path: &[Walked]) -> Node
where
    M: storage::Reads,
{
    match path {
        [one] => read_step::<M>(slots, *one),
        many => {
            let steps = steps_of(many);
            node(move |next| storage::ReadPath::<M> {
                slots,
                steps,
                mode: PhantomData,
                next,
            })
        }
    }
}

fn assign_place(slots: storage::Write, how: Writing, path: &[Walked]) -> Node {
    match (how.through, how.large, path) {
        (false, false, []) => node(move |next| storage::AssignVar::<false> { slots, next }),
        (false, true, []) => node(move |next| storage::AssignVar::<true> { slots, next }),
        (true, false, []) => node(move |next| storage::AssignThrough::<false> { slots, next }),
        (true, true, []) => node(move |next| storage::AssignThrough::<true> { slots, next }),
        (false, false, [one]) => assign_step::<false, false>(slots, *one),
        (false, true, [one]) => assign_step::<false, true>(slots, *one),
        (true, false, [one]) => assign_step::<true, false>(slots, *one),
        (true, true, [one]) => assign_step::<true, true>(slots, *one),
        (false, false, many) => {
            let steps = steps_of(many);
            node(move |next| storage::AssignPath::<false, false> { slots, steps, next })
        }
        (false, true, many) => {
            let steps = steps_of(many);
            node(move |next| storage::AssignPath::<false, true> { slots, steps, next })
        }
        (true, false, many) => {
            let steps = steps_of(many);
            node(move |next| storage::AssignPath::<true, false> { slots, steps, next })
        }
        (true, true, many) => {
            let steps = steps_of(many);
            node(move |next| storage::AssignPath::<true, true> { slots, steps, next })
        }
    }
}

fn set_place(slots: storage::Update, large: bool, path: &[Walked]) -> Node {
    assert!(
        !path.is_empty(),
        "a field set with no step names no field to write"
    );
    match (large, path) {
        (false, [one]) => set_step::<false>(slots, *one),
        (true, [one]) => set_step::<true>(slots, *one),
        (false, many) => {
            let steps = steps_of(many);
            node(move |next| storage::SetPath::<false> { slots, steps, next })
        }
        (true, many) => {
            let steps = steps_of(many);
            node(move |next| storage::SetPath::<true> { slots, steps, next })
        }
    }
}

// -- The variant tests -------------------------------------------------

fn test_option<const THROUGH: bool>(slots: Unary, some: bool) -> Node {
    match some {
        true => node(move |next| variant::TestOption::<THROUGH, true> { slots, next }),
        false => node(move |next| variant::TestOption::<THROUGH, false> { slots, next }),
    }
}

fn test_result<const THROUGH: bool>(slots: Unary, ok: bool) -> Node {
    match ok {
        true => node(move |next| variant::TestResult::<THROUGH, true> { slots, next }),
        false => node(move |next| variant::TestResult::<THROUGH, false> { slots, next }),
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
fn direct_call(into: Dest, callee: QualifiedRef, laid: Laid) -> Node {
    let Dest {
        slot: dst,
        large,
        word,
    } = into;
    let Laid { arity, takes } = laid;
    match (large, word) {
        (true, _) => node(move |next| call::CallDirect::<true, false> {
            dst,
            callee,
            arity,
            takes,
            next,
        }),
        (false, true) => node(move |next| call::CallDirect::<false, true> {
            dst,
            callee,
            arity,
            takes,
            next,
        }),
        (false, false) => node(move |next| call::CallDirect::<false, false> {
            dst,
            callee,
            arity,
            takes,
            next,
        }),
    }
}

fn indirect_call(into: Dest, through: bool, callee: Off, laid: Laid) -> Node {
    let Dest {
        slot: dst,
        large,
        word,
    } = into;
    let Laid { arity, takes } = laid;
    match (large, word, through) {
        (true, _, false) => node(move |next| call::CallIndirect::<true, false, false> {
            dst,
            callee,
            arity,
            takes,
            next,
        }),
        (true, _, true) => node(move |next| call::CallIndirect::<true, false, true> {
            dst,
            callee,
            arity,
            takes,
            next,
        }),
        (false, true, false) => node(move |next| call::CallIndirect::<false, true, false> {
            dst,
            callee,
            arity,
            takes,
            next,
        }),
        (false, true, true) => node(move |next| call::CallIndirect::<false, true, true> {
            dst,
            callee,
            arity,
            takes,
            next,
        }),
        (false, false, false) => node(move |next| call::CallIndirect::<false, false, false> {
            dst,
            callee,
            arity,
            takes,
            next,
        }),
        (false, false, true) => node(move |next| call::CallIndirect::<false, false, true> {
            dst,
            callee,
            arity,
            takes,
            next,
        }),
    }
}

fn indirect_call_async(
    into: Dest,
    through: bool,
    callee: Off,
    operands: Operands,
    next: BlockId,
) -> Box<dyn Op> {
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
