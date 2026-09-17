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

use std::ops::Range;
use std::sync::Arc;

use acvus_ast::{Literal, Span};
use acvus_mir::analysis::inst_info;
use acvus_mir::graph::QualifiedRef;
use acvus_mir::ir::{
    Callee, Inst, InstKind, Label, MirBody, MirModule, PathSeg, RefTarget, ValueId,
};
use acvus_mir::ty::{IntTy, Ty};
use acvus_utils::{Astr, Interner, LocalIdOps};
use rustc_hash::FxHashMap;

use crate::code::{
    ArgWindow, BasicBlock, Code, ConcatPart, FieldSlot, Konst, LoopBody, NO_SLOT, Op, OpFn,
    Payload, Prepared, SlotMove,
};
use crate::interpreter::Executable;
use crate::ops::arith::{self, for_int_ty};
use crate::ops::{call, composite, constant, control, pattern, storage, string, variant};
use crate::runtime::ExternHandler;
use crate::value::Tag;

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
            let code = prepare_body(body, ctx, &closures);
            closures.insert(*label, Arc::new(code));
        }
        remaining.retain(|(label, _)| !closures.contains_key(label));
    }

    let main = Arc::new(prepare_body(&module.main, ctx, &closures));
    Prepared { main, closures }
}

fn made_closures(body: &MirBody) -> impl Iterator<Item = Label> + '_ {
    body.insts.iter().filter_map(|inst| match &inst.kind {
        InstKind::MakeClosure { body, .. } => Some(*body),
        _ => None,
    })
}

pub fn prepare_body(
    body: &MirBody,
    ctx: &PrepareCtx<'_>,
    closures: &FxHashMap<Label, Arc<Code>>,
) -> Code {
    let mut prep = Prepare::new(body, ctx, closures, label_map(body));

    let loops = prep.loops();
    prep.op_index = op_indexes(body.insts.len(), &loops);

    let mut ops: Vec<Op> = Vec::with_capacity(body.insts.len());
    let mut spans: Vec<Span> = Vec::with_capacity(body.insts.len());
    let mut at = 0;
    for region in &loops {
        for index in at..region.start() {
            ops.push(prep.op(index));
            spans.push(body.insts[index].span);
        }
        ops.push(prep.loop_op(region));
        spans.push(body.insts[region.head].span);
        at = region.end();
    }
    for index in at..body.insts.len() {
        ops.push(prep.op(index));
        spans.push(body.insts[index].span);
    }

    let frame_len = prep.scratch + u32::from(prep.scratch_used);
    let params = body.params.iter().map(|(_, v)| prep.slot(*v)).collect();
    let captures = body.captures.iter().map(|(_, v)| prep.slot(*v)).collect();
    let order_param = body.order_param.map(|id| prep.slot(id));

    Code {
        ops: ops.into_boxed_slice(),
        spans: spans.into_boxed_slice(),
        payloads: prep.payloads.into_boxed_slice(),
        frame_len,
        may_suspend: prep.may_suspend,
        params,
        captures,
        order_param,
    }
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
}

/// One `while` the recognizer matched, as indexes into `MirBody::insts`.
struct LoopRegion {
    enter_jump: Option<usize>,
    head: usize,
    head_block: Range<usize>,
    head_loops: Vec<LoopRegion>,
    jump_if: usize,
    body_block: Range<usize>,
    body_loops: Vec<LoopRegion>,
    back: usize,
}

struct StraightRun {
    stops_at: usize,
    loops: Vec<LoopRegion>,
}

impl LoopRegion {
    fn start(&self) -> usize {
        self.enter_jump.unwrap_or(self.head)
    }

    fn end(&self) -> usize {
        self.back + 1
    }

    fn covers(&self, at: usize) -> bool {
        (self.start()..self.end()).contains(&at)
    }
}

fn op_indexes(len: usize, loops: &[LoopRegion]) -> Vec<Option<u32>> {
    let mut index = vec![None; len];
    let mut next = 0;
    let mut at = 0;
    for region in loops {
        for slot in index.iter_mut().take(region.start()).skip(at) {
            *slot = Some(next);
            next += 1;
        }
        index[region.start()] = Some(next);
        next += 1;
        at = region.end();
    }
    for slot in index.iter_mut().skip(at) {
        *slot = Some(next);
        next += 1;
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
        let slots = assign_slots(body, &labels);
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
        }
    }

    fn slot(&self, id: ValueId) -> u32 {
        self.slots.of(id)
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

    fn is_string(&self, id: ValueId) -> bool {
        matches!(self.ty(id), Ty::String)
    }

    fn put(&mut self, payload: Payload) -> usize {
        self.payloads.push(payload);
        self.payloads.len() - 1
    }

    fn path(&mut self, path: &[PathSeg]) -> usize {
        self.put(Payload::Path(path.to_vec().into_boxed_slice()))
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
            | InstKind::ArrayGet { .. }
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
        let mut loops = Vec::new();
        let mut at = from;
        while at < limit {
            if let Some(region) = self.recognize_loop(at)
                && region.end() <= limit
            {
                at = region.end();
                loops.push(region);
            } else if self.is_straight_line(&insts[at]) {
                at += 1;
            } else {
                break;
            }
        }
        StraightRun {
            stops_at: at,
            loops,
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
            loops: head_loops,
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
            loops: body_loops,
        } = self.straight_run(body_label + 1, back);
        if body_end != back {
            return None;
        }

        let region = LoopRegion {
            enter_jump,
            head,
            head_block: head + 1..jump_if,
            head_loops,
            jump_if,
            body_block: body_label + 1..back,
            body_loops,
            back,
        };
        self.is_closed(&region).then_some(region)
    }

    fn is_closed(&self, region: &LoopRegion) -> bool {
        let insts = self.body.insts.as_slice();
        insts[region.start()..region.end()]
            .iter()
            .filter_map(block_label)
            .flat_map(|label| self.references(label))
            .all(|at| region.covers(at))
    }

    fn loops(&self) -> Vec<LoopRegion> {
        let mut found = Vec::new();
        let mut at = 0;
        while at < self.body.insts.len() {
            match self.recognize_loop(at) {
                Some(region) => {
                    at = region.end();
                    found.push(region);
                }
                None => at += 1,
            }
        }
        found
    }

    fn block(&mut self, range: Range<usize>, nested: &[LoopRegion]) -> BasicBlock {
        let mut operations: Vec<Op> = Vec::with_capacity(range.len());
        let mut at = range.start;
        while at < range.end {
            match nested.iter().find(|region| region.start() == at) {
                Some(region) => {
                    operations.push(self.loop_op(region));
                    at = region.end();
                }
                None => {
                    operations.push(self.op(at));
                    at += 1;
                }
            }
        }
        BasicBlock::new(operations)
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

        let head = self.block(region.head_block.clone(), &region.head_loops);
        let block = self.block(region.body_block.clone(), &region.body_loops);

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
            } => match target {
                RefTarget::Var(s) | RefTarget::Param(s) if path.is_empty() => {
                    Op::new(storage::ref_var)
                        .a(self.slot(*dst))
                        .b(self.slot(*s))
                }
                RefTarget::Var(s) | RefTarget::Param(s) => {
                    let at = self.path(path);
                    Op::new(storage::ref_var_path)
                        .a(self.slot(*dst))
                        .b(self.slot(*s))
                        .p(at)
                }
                RefTarget::Through(r) if path.is_empty() => Op::new(storage::ref_through)
                    .a(self.slot(*dst))
                    .b(self.slot(*r)),
                RefTarget::Through(r) => {
                    let at = self.path(path);
                    Op::new(storage::ref_through_path)
                        .a(self.slot(*dst))
                        .b(self.slot(*r))
                        .p(at)
                }
            },
            InstKind::Take { dst, target, path } => {
                let clone = self.is_string(*dst);
                match target {
                    RefTarget::Var(s) | RefTarget::Param(s) if path.is_empty() => {
                        let f: OpFn = if clone {
                            storage::take_var::<true>
                        } else {
                            storage::take_var::<false>
                        };
                        Op::new(f).a(self.slot(*dst)).b(self.slot(*s))
                    }
                    RefTarget::Var(s) | RefTarget::Param(s) => {
                        let f: OpFn = if clone {
                            storage::read_path::<true>
                        } else {
                            storage::read_path::<false>
                        };
                        let at = self.path(path);
                        Op::new(f).a(self.slot(*dst)).b(self.slot(*s)).p(at)
                    }
                    RefTarget::Through(r) if path.is_empty() => {
                        let f: OpFn = if clone {
                            storage::take_through::<true>
                        } else {
                            storage::take_through::<false>
                        };
                        Op::new(f).a(self.slot(*dst)).b(self.slot(*r))
                    }
                    RefTarget::Through(r) => {
                        let f: OpFn = if clone {
                            storage::take_through_path::<true>
                        } else {
                            storage::take_through_path::<false>
                        };
                        let at = self.path(path);
                        Op::new(f).a(self.slot(*dst)).b(self.slot(*r)).p(at)
                    }
                }
            }
            InstKind::Assign {
                target,
                path,
                value,
            } => match target {
                RefTarget::Var(s) | RefTarget::Param(s) if path.is_empty() => {
                    Op::new(storage::assign_var)
                        .a(self.slot(*s))
                        .b(self.slot(*value))
                }
                RefTarget::Var(s) | RefTarget::Param(s) => {
                    let at = self.path(path);
                    Op::new(storage::assign_var_path)
                        .a(self.slot(*s))
                        .b(self.slot(*value))
                        .p(at)
                }
                RefTarget::Through(r) if path.is_empty() => Op::new(storage::assign_through)
                    .a(self.slot(*r))
                    .b(self.slot(*value)),
                RefTarget::Through(r) => {
                    let at = self.path(path);
                    Op::new(storage::assign_through_path)
                        .a(self.slot(*r))
                        .b(self.slot(*value))
                        .p(at)
                }
            },
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
                    let path: Vec<PathSeg> = std::iter::once(*field)
                        .chain(rest.iter().copied())
                        .map(PathSeg::Field)
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
                let path: Vec<PathSeg> = std::iter::once(*field)
                    .chain(rest.iter().copied())
                    .map(PathSeg::Field)
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
                args,
                order,
                ..
            } => {
                let after = order.map_or(NO_SLOT, |edge| self.slot(edge.after));
                match callee {
                    Callee::Direct(id) => {
                        self.may_suspend = true;
                        let at = self.put(Payload::Direct {
                            callee: *id,
                            args: args.iter().copied().map(|id| self.slot(id)).collect(),
                        });
                        Op::new(call::call_direct).a(self.slot(*dst)).d(after).p(at)
                    }
                    Callee::Extern { id, instance } => {
                        let handler = self.ctx.handler(id, *instance);
                        let f: OpFn = match handler {
                            ExternHandler::Sync(_) => call::call_extern_sync,
                            ExternHandler::Async(_) => {
                                self.may_suspend = true;
                                call::call_extern_async
                            }
                        };
                        let window = self.slots.window(at).clone();
                        let payload = self.put(Payload::Extern { handler, window });
                        Op::new(f).a(self.slot(*dst)).d(after).p(payload)
                    }
                    Callee::Indirect(callee_slot) => {
                        self.may_suspend = true;
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
                        ExternHandler::Sync(_) => call::spawn_extern_sync,
                        ExternHandler::Async(_) => call::spawn_extern_async,
                    };
                    let window = self.slots.window(at).clone();
                    let payload = self.put(Payload::Extern { handler, window });
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
            InstKind::ArrayGet { dst, array, index } => {
                let f: OpFn = if self.is_string(*dst) {
                    pattern::array_get::<true>
                } else {
                    pattern::array_get::<false>
                };
                Op::new(f)
                    .a(self.slot(*dst))
                    .b(self.slot(*array))
                    .c(self.slot(*index))
            }
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
                let f: OpFn = if self.is_ref(*src) {
                    variant::test_variant::<true>
                } else {
                    variant::test_variant::<false>
                };
                let at = self.put(Payload::Name(*tag));
                Op::new(f)
                    .a(self.slot(*dst))
                    .b(self.slot(*src))
                    .c(u32::from(self.tag_is(*tag, "Some")))
                    .d(u32::from(self.tag_is(*tag, "Ok")))
                    .p(at)
            }
            InstKind::UnwrapVariant { dst, src } => Op::new(variant::unwrap_variant)
                .a(self.slot(*dst))
                .b(self.slot(*src)),

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
        (Literal::Int(n), Ty::Int(k)) => Konst::Word(Tag::int(*k), *n as u64),
        (Literal::Int(n), other) => panic!("integer literal {n} typed as {other:?}"),
        (Literal::Float(x), _) => Konst::Word(Tag::F64, x.to_bits()),
        (Literal::Bool(b), _) => Konst::Word(Tag::Bool, u64::from(*b)),
        (Literal::Unit, _) => Konst::Word(Tag::Unit, 0),
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
fn extern_args(inst: &Inst) -> Option<&[ValueId]> {
    match &inst.kind {
        InstKind::FunctionCall {
            callee: Callee::Extern { .. },
            args,
            ..
        }
        | InstKind::Spawn {
            callee: Callee::Extern { .. },
            args,
            ..
        } => Some(args),
        _ => None,
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
fn assign_slots(body: &MirBody, labels: &FxHashMap<Label, u32>) -> Slots {
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
        let Some(args) = extern_args(inst) else {
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
    check_assignment(&edges, &live, &slots);
    slots
}

/// No two values live at one program point share a slot, and no value
/// occupies a slot the call at that point writes or empties.
#[cfg(debug_assertions)]
fn check_assignment(edges: &Edges<'_>, live: &Live, slots: &Slots) {
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
        let (Some(args), Some(window)) = (extern_args(inst), slots.windows.get(&at)) else {
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
    struct Matched {
        covers: Range<usize>,
        nested: usize,
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
            let handler = ExternHandler::Sync(Arc::new(|_, _| {
                panic!("the preparation must not run a handler")
            }));
            self.externs.insert(id, Executable::Extern(vec![handler]));
            id
        }

        fn prepared(&self, insts: Vec<Inst>) -> Code {
            let context_names = FxHashMap::default();
            let ctx = PrepareCtx {
                interner: &self.interner,
                externs: &self.externs,
                context_names: &context_names,
            };
            let mut body = body_of(insts);
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
            prepare_body(&body, &ctx, &FxHashMap::default())
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
            prep.loops()
                .iter()
                .map(|region| Matched {
                    covers: region.start()..region.end(),
                    nested: region.head_loops.len() + region.body_loops.len(),
                })
                .collect()
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
            vec![Matched {
                covers: 0..7,
                nested: 0
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
            vec![Matched {
                covers: 0..14,
                nested: 1
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
    fn a_branch_in_the_body_leaves_the_while_alone() {
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
        assert_eq!(fixture.recognize(insts), vec![]);
    }

    #[test]
    fn a_body_of_arithmetic_and_a_synchronous_extern_cannot_suspend() {
        let mut fixture = Fixture::new();
        let stays = fixture.sync_extern("stays");
        let insts = vec![add(1, 2, 3), call(4, stays)];
        assert!(!fixture.prepared(insts).may_suspend);
    }

    #[test]
    fn a_body_that_calls_an_asynchronous_extern_can_suspend() {
        let mut fixture = Fixture::new();
        let suspends = fixture.async_extern("suspends");
        let insts = vec![add(1, 2, 3), call(4, suspends)];
        assert!(fixture.prepared(insts).may_suspend);
    }

    #[test]
    fn a_body_that_calls_a_closure_can_suspend_until_the_closure_is_in_hand() {
        let fixture = Fixture::new();
        let insts = vec![inst(InstKind::FunctionCall {
            dst: val(1),
            callee: Callee::Indirect(val(2)),
            callee_ty: Ty::Unit,
            args: Vec::new(),
            order: None,
        })];
        assert!(fixture.prepared(insts).may_suspend);
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

    fn call(dst: usize, args: &[usize]) -> Inst {
        inst(InstKind::FunctionCall {
            dst: val(dst),
            callee: Callee::Extern {
                id: QualifiedRef {
                    namespace: None,
                    name: Interner::new().intern("f"),
                },
                instance: 0,
            },
            callee_ty: Ty::Unit,
            args: args.iter().copied().map(val).collect(),
            order: None,
        })
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
        assign_slots(&body, &labels)
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
        let slots = assign(vec![konst(0), call(1, &[0]), ret(1)]);
        let window = slots.window(1);
        assert!(window.moves.is_empty());
        assert_eq!(slots.of(val(0)), window.at);
    }

    #[test]
    fn an_argument_used_after_the_call_is_moved_into_the_window() {
        let slots = assign(vec![konst(0), call(1, &[0]), add(2, 0, 1), ret(2)]);
        let window = slots.window(1);
        let arg = slots.of(val(0));
        assert_ne!(arg, window.at);
        let moved: Vec<(u32, u32)> = window.moves.iter().map(|m| (m.from, m.to)).collect();
        assert_eq!(moved, vec![(arg, window.at)]);
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
