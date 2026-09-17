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
use acvus_mir::graph::QualifiedRef;
use acvus_mir::ir::{
    Callee, Inst, InstKind, Label, MirBody, MirModule, PathSeg, RefTarget, ValueId,
};
use acvus_mir::ty::{IntTy, Ty};
use acvus_utils::{Astr, Interner, LocalIdOps};
use rustc_hash::FxHashMap;

use crate::code::{
    BasicBlock, Code, ConcatPart, FieldSlot, Konst, LoopBody, NO_SLOT, Op, OpFn, Payload, Prepared,
    SlotMove,
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
    let labels: FxHashMap<Label, u32> = body
        .insts
        .iter()
        .enumerate()
        .filter_map(|(i, inst)| match &inst.kind {
            InstKind::BlockLabel { label, .. } => Some((*label, i as u32)),
            _ => None,
        })
        .collect();

    let mut prep = Prepare {
        body,
        ctx,
        closures,
        labels,
        payloads: Vec::new(),
        scratch: body.val_factory.len() as u32,
        scratch_used: false,
        op_index: Vec::new(),
    };

    let loops = prep.loops();
    prep.op_index = op_indexes(body.insts.len(), &loops);

    let mut ops: Vec<Op> = Vec::with_capacity(body.insts.len());
    let mut spans: Vec<Span> = Vec::with_capacity(body.insts.len());
    let mut at = 0;
    for region in &loops {
        for inst in &body.insts[at..region.start()] {
            ops.push(prep.op(inst));
            spans.push(inst.span);
        }
        ops.push(prep.loop_op(region));
        spans.push(body.insts[region.head].span);
        at = region.end();
    }
    for inst in &body.insts[at..] {
        ops.push(prep.op(inst));
        spans.push(inst.span);
    }

    let frame_len = prep.scratch + u32::from(prep.scratch_used);

    Code {
        ops: ops.into_boxed_slice(),
        spans: spans.into_boxed_slice(),
        payloads: prep.payloads.into_boxed_slice(),
        frame_len,
        params: body.params.iter().map(|(_, v)| slot(*v)).collect(),
        captures: body.captures.iter().map(|(_, v)| slot(*v)).collect(),
        order_param: body.order_param.map(slot),
    }
}

fn slot(id: ValueId) -> u32 {
    id.to_raw() as u32
}

struct Prepare<'a> {
    body: &'a MirBody,
    ctx: &'a PrepareCtx<'a>,
    closures: &'a FxHashMap<Label, Arc<Code>>,
    labels: FxHashMap<Label, u32>,
    payloads: Vec<Payload>,
    scratch: u32,
    scratch_used: bool,
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

impl Prepare<'_> {
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
        self.put(Payload::Slots(ids.iter().copied().map(slot).collect()))
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
                from: slot(*arg),
                to: slot(*param),
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
        let body = self.body;
        let mut operations: Vec<(Op, Span)> = Vec::with_capacity(range.len());
        let mut at = range.start;
        while at < range.end {
            match nested.iter().find(|region| region.start() == at) {
                Some(region) => {
                    operations.push((self.loop_op(region), body.insts[region.head].span));
                    at = region.end();
                }
                None => {
                    operations.push((self.op(&body.insts[at]), body.insts[at].span));
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
            cond_slot: slot(*cond),
            into_body,
            body: block,
            back,
            exit,
        }));
        Op::new(control::while_loop).p(at)
    }

    fn op(&mut self, inst: &Inst) -> Op {
        match &inst.kind {
            InstKind::Const { dst, value } => self.constant(*dst, value),

            InstKind::StringConcat { dst, parts } => {
                let parts: Box<[ConcatPart]> = parts
                    .iter()
                    .map(|part| ConcatPart {
                        slot: slot(*part),
                        through_reference: self.is_ref(*part),
                    })
                    .collect();
                let at = self.put(Payload::Parts(parts));
                Op::new(string::concat).a(slot(*dst)).p(at)
            }
            InstKind::StringEq { dst, a, b } => Op::new(string::string_eq)
                .a(slot(*dst))
                .b(slot(*a))
                .c(slot(*b)),
            InstKind::StringClone { dst, src } => {
                let f: OpFn = if self.is_ref(*src) {
                    string::clone_string::<true>
                } else {
                    string::clone_string::<false>
                };
                Op::new(f).a(slot(*dst)).b(slot(*src))
            }

            InstKind::Ref {
                dst, target, path, ..
            } => match target {
                RefTarget::Var(s) | RefTarget::Param(s) if path.is_empty() => {
                    Op::new(storage::ref_var).a(slot(*dst)).b(slot(*s))
                }
                RefTarget::Var(s) | RefTarget::Param(s) => {
                    let at = self.path(path);
                    Op::new(storage::ref_var_path)
                        .a(slot(*dst))
                        .b(slot(*s))
                        .p(at)
                }
                RefTarget::Through(r) if path.is_empty() => {
                    Op::new(storage::ref_through).a(slot(*dst)).b(slot(*r))
                }
                RefTarget::Through(r) => {
                    let at = self.path(path);
                    Op::new(storage::ref_through_path)
                        .a(slot(*dst))
                        .b(slot(*r))
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
                        Op::new(f).a(slot(*dst)).b(slot(*s))
                    }
                    RefTarget::Var(s) | RefTarget::Param(s) => {
                        let f: OpFn = if clone {
                            storage::read_path::<true>
                        } else {
                            storage::read_path::<false>
                        };
                        let at = self.path(path);
                        Op::new(f).a(slot(*dst)).b(slot(*s)).p(at)
                    }
                    RefTarget::Through(r) if path.is_empty() => {
                        let f: OpFn = if clone {
                            storage::take_through::<true>
                        } else {
                            storage::take_through::<false>
                        };
                        Op::new(f).a(slot(*dst)).b(slot(*r))
                    }
                    RefTarget::Through(r) => {
                        let f: OpFn = if clone {
                            storage::take_through_path::<true>
                        } else {
                            storage::take_through_path::<false>
                        };
                        let at = self.path(path);
                        Op::new(f).a(slot(*dst)).b(slot(*r)).p(at)
                    }
                }
            }
            InstKind::Assign {
                target,
                path,
                value,
            } => match target {
                RefTarget::Var(s) | RefTarget::Param(s) if path.is_empty() => {
                    Op::new(storage::assign_var).a(slot(*s)).b(slot(*value))
                }
                RefTarget::Var(s) | RefTarget::Param(s) => {
                    let at = self.path(path);
                    Op::new(storage::assign_var_path)
                        .a(slot(*s))
                        .b(slot(*value))
                        .p(at)
                }
                RefTarget::Through(r) if path.is_empty() => {
                    Op::new(storage::assign_through).a(slot(*r)).b(slot(*value))
                }
                RefTarget::Through(r) => {
                    let at = self.path(path);
                    Op::new(storage::assign_through_path)
                        .a(slot(*r))
                        .b(slot(*value))
                        .p(at)
                }
            },
            InstKind::Fetch { dst, context } => {
                let at = self.put(Payload::PageKey(self.ctx.page_key(context)));
                Op::new(storage::fetch).a(slot(*dst)).p(at)
            }
            InstKind::Commit { context, value } => {
                let at = self.put(Payload::PageKey(self.ctx.page_key(context)));
                Op::new(storage::commit).a(slot(*value)).p(at)
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
                    Op::new(f).a(slot(*dst)).b(slot(*object)).p(at)
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
                    Op::new(f).a(slot(*dst)).b(slot(*object)).p(at)
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
                    .a(slot(*dst))
                    .b(slot(*object))
                    .c(slot(*value))
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
                Op::new(f).a(slot(*dst)).b(slot(*left)).c(slot(*right))
            }
            InstKind::UnaryOp { dst, op, operand } => {
                let f = match self.ty(*operand) {
                    Ty::Int(k) => arith::int_unaryop(*op, *k),
                    Ty::Float => arith::float_unaryop(*op),
                    Ty::Bool => arith::bool_unaryop(*op),
                    other => panic!("unary {op:?} on {other:?}"),
                };
                Op::new(f).a(slot(*dst)).b(slot(*operand))
            }

            InstKind::LoadFunction { .. } => Op::new(control::load_function),
            InstKind::FunctionCall {
                dst,
                callee,
                args,
                order,
                ..
            } => {
                let after = order.map_or(NO_SLOT, |edge| slot(edge.after));
                match callee {
                    Callee::Direct(id) => {
                        let at = self.put(Payload::Direct {
                            callee: *id,
                            args: args.iter().copied().map(slot).collect(),
                        });
                        Op::new(call::call_direct).a(slot(*dst)).d(after).p(at)
                    }
                    Callee::Extern { id, instance } => {
                        let handler = self.ctx.handler(id, *instance);
                        let f: OpFn = match handler {
                            ExternHandler::Sync(_) => call::call_extern_sync,
                            ExternHandler::Async(_) => call::call_extern_async,
                        };
                        let at = self.put(Payload::Extern {
                            handler,
                            args: args.iter().copied().map(slot).collect(),
                        });
                        Op::new(f).a(slot(*dst)).d(after).p(at)
                    }
                    Callee::Indirect(callee_slot) => {
                        let f: OpFn = if self.is_ref(*callee_slot) {
                            call::call_indirect::<true>
                        } else {
                            call::call_indirect::<false>
                        };
                        let at = self.slots(args);
                        Op::new(f)
                            .a(slot(*dst))
                            .b(slot(*callee_slot))
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
                        args: args.iter().copied().map(slot).collect(),
                    });
                    Op::new(call::spawn_module).a(slot(*dst)).p(at)
                }
                Callee::Extern { id, instance } => {
                    let handler = self.ctx.handler(id, *instance);
                    let f: OpFn = match handler {
                        ExternHandler::Sync(_) => call::spawn_extern_sync,
                        ExternHandler::Async(_) => call::spawn_extern_async,
                    };
                    let at = self.put(Payload::Extern {
                        handler,
                        args: args.iter().copied().map(slot).collect(),
                    });
                    Op::new(f).a(slot(*dst)).p(at)
                }
                Callee::Indirect(_) => panic!("spawn: indirect callee not supported"),
            },
            InstKind::Eval { dst, src, order } => Op::new(call::eval)
                .a(slot(*dst))
                .b(slot(*src))
                .d(order.map_or(NO_SLOT, slot)),
            InstKind::Merge { dst, .. } => Op::new(control::merge).a(slot(*dst)),

            InstKind::MakeArray { dst, elements } => {
                let at = self.slots(elements);
                Op::new(composite::make_array).a(slot(*dst)).p(at)
            }
            InstKind::MakeObject { dst, fields } => {
                let fields: Box<[FieldSlot]> = fields
                    .iter()
                    .map(|(key, value)| FieldSlot {
                        key: *key,
                        slot: slot(*value),
                    })
                    .collect();
                let at = self.put(Payload::Fields(fields));
                Op::new(composite::make_object).a(slot(*dst)).p(at)
            }
            InstKind::MakeTuple { dst, elements } => {
                let at = self.slots(elements);
                Op::new(composite::make_tuple).a(slot(*dst)).p(at)
            }
            InstKind::TupleIndex { dst, tuple, index } => {
                let f: OpFn = if self.is_string(*dst) {
                    storage::read_index::<true>
                } else {
                    storage::read_index::<false>
                };
                Op::new(f).a(slot(*dst)).b(slot(*tuple)).p(*index)
            }

            InstKind::TestLiteral { dst, src, value } => self.test_literal(*dst, *src, value),
            InstKind::TestObjectKey { dst, src, key } => {
                let f: OpFn = if self.is_ref(*src) {
                    pattern::test_object_key::<true>
                } else {
                    pattern::test_object_key::<false>
                };
                let at = self.put(Payload::Name(*key));
                Op::new(f).a(slot(*dst)).b(slot(*src)).p(at)
            }
            InstKind::ArrayIndex { dst, array, index } => {
                let f: OpFn = if self.is_string(*dst) {
                    storage::read_index::<true>
                } else {
                    storage::read_index::<false>
                };
                Op::new(f).a(slot(*dst)).b(slot(*array)).p(*index)
            }
            InstKind::ArrayGet { dst, array, index } => {
                let f: OpFn = if self.is_string(*dst) {
                    pattern::array_get::<true>
                } else {
                    pattern::array_get::<false>
                };
                Op::new(f).a(slot(*dst)).b(slot(*array)).c(slot(*index))
            }
            InstKind::ObjectGet { dst, object, key } => {
                let f: OpFn = if self.is_string(*dst) {
                    storage::read_field::<true>
                } else {
                    storage::read_field::<false>
                };
                let at = self.put(Payload::Name(*key));
                Op::new(f).a(slot(*dst)).b(slot(*object)).p(at)
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
                    captures: captures.iter().copied().map(slot).collect(),
                });
                Op::new(call::make_closure).a(slot(*dst)).p(at)
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
                    .a(slot(*dst))
                    .b(slot(*src))
                    .c(u32::from(self.tag_is(*tag, "Some")))
                    .d(u32::from(self.tag_is(*tag, "Ok")))
                    .p(at)
            }
            InstKind::UnwrapVariant { dst, src } => {
                Op::new(variant::unwrap_variant).a(slot(*dst)).b(slot(*src))
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
                    .a(slot(*cond))
                    .b(then_target)
                    .c(else_target)
                    .d(else_at as u32)
                    .p(then_at)
            }
            InstKind::Return { value, .. } => Op::new(control::ret).a(slot(*value)),
            InstKind::Diverge => Op::new(control::diverge),
            InstKind::Undef { dst } => Op::new(control::undef).a(slot(*dst)),
            InstKind::Nop => Op::new(control::nop),
            InstKind::Drop { src } => Op::new(control::drop_value).a(slot(*src)),
            InstKind::Poison { .. } => Op::new(control::poison),
        }
    }

    fn constant(&mut self, dst: ValueId, value: &Literal) -> Op {
        let out = slot(dst);
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
        Op::new(f).a(slot(dst)).b(slot(src)).p(word)
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
            .a(slot(dst))
            .b(payload.map_or(NO_SLOT, slot))
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

        fn recognize(&self, insts: Vec<Inst>) -> Vec<Matched> {
            let context_names = FxHashMap::default();
            let ctx = PrepareCtx {
                interner: &self.interner,
                externs: &self.externs,
                context_names: &context_names,
            };
            let closures = FxHashMap::default();
            let mut body = MirBody::new();
            body.insts = insts;
            let prep = Prepare {
                body: &body,
                ctx: &ctx,
                closures: &closures,
                labels: FxHashMap::default(),
                payloads: Vec::new(),
                scratch: 0,
                scratch_used: false,
                op_index: Vec::new(),
            };
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
    fn a_jump_into_the_loop_from_outside_leaves_it_alone() {
        let fixture = Fixture::new();
        let mut insts = plain_while();
        insts.push(jump(1));
        assert_eq!(fixture.recognize(insts), vec![]);
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
