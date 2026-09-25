//! RFC-0066 rule 1's normal form for an exit that leaves a nested loop's
//! parent: the edge becomes a `break` of the nested loop, whose exit block
//! gains one parameter, the verdict, and the parent branches on it right
//! after the nested loop. RFC-0089 rule 5 reads an exit of the loop being
//! cut, and an edge that leaves two loops at once is an exit of neither.
//!
//! The pass runs before drop insertion, which places every drop on the
//! edges of the rewritten body (RFC-0057 rules 4, 7 and 8, RFC-0048), so a
//! value alive across the moved edge is released once on each path.
//!
//! An edge whose moved code would carry a reference or an `Order`, or
//! touch a storage only the nested loop's iteration touches, is not moved,
//! and neither is one out of a loop over an `Array` or whose exit block is
//! entered from outside it or already takes a value, the trip count among
//! them. These are decisions not to build: an `Option` does not hold a
//! reference, no value represents an `Order`, a storage read past the
//! nested loop's exit would become live at its header, an exit block that
//! takes values would need, on each moved edge, values that mean nothing,
//! and an `Array`'s release on the moved edge is a drop block of its own,
//! which runs the nested loop as joints where the `return` left it one
//! region (RFC-0057 rules 6 and 7). Such a loop stays as it was.

use rustc_hash::{FxHashMap, FxHashSet};
use smallvec::SmallVec;

use acvus_ast::{Literal, Span};
use acvus_utils::{Astr, Interner};

use crate::analysis::domtree::DomTree;
use crate::analysis::inst_info;
use crate::analysis::loans::Loans;
use crate::analysis::loops::{NaturalLoop, natural_loops_innermost_first};
use crate::analysis::targets::touched_slots;
use crate::cfg::{Block, BlockIdx, CfgBody, ENTRY_LABEL, Terminator};
use crate::ir::{ExitTrip, ForSource, Inst, InstKind, Label, ValueId};
use crate::optimize::ssa_pass::{apply_subst, apply_subst_terminator};
use crate::ty::Ty;

type Preds = FxHashMap<BlockIdx, SmallVec<[BlockIdx; 2]>>;

pub fn run(interner: &Interner, cfg: &mut CfgBody) {
    let tags = Tags {
        some: interner.intern("Some"),
        none: interner.intern("None"),
    };
    while let Some(moved) = Move::find(cfg) {
        moved.apply(cfg, &tags);
    }
}

struct Tags {
    some: Astr,
    none: Astr,
}

enum Verdict {
    BoolLeaves,
    SomeCarries { payload: Ty },
}

impl Verdict {
    fn of(carried: &[Ty]) -> Verdict {
        match carried {
            [] => Verdict::BoolLeaves,
            [one] => Verdict::SomeCarries {
                payload: one.clone(),
            },
            many => Verdict::SomeCarries {
                payload: Ty::Tuple(many.to_vec()),
            },
        }
    }

    fn ty(&self) -> Ty {
        match self {
            Verdict::BoolLeaves => Ty::Bool,
            Verdict::SomeCarries { payload } => Ty::Option(Box::new(payload.clone())),
        }
    }
}

struct Move {
    header: BlockIdx,
    exit: BlockIdx,
    breaks: Vec<BlockIdx>,
    from: Vec<BlockIdx>,
    target_outside_parent: BlockIdx,
    code: Vec<BlockIdx>,
    read_of_body: Vec<ValueId>,
    dominating_entry: BlockIdx,
    span: Span,
}

#[derive(Clone, Copy)]
struct Nest<'l> {
    nested: &'l NaturalLoop,
    parent: &'l NaturalLoop,
}

struct Rewrite<'m> {
    moved: &'m Move,
    tags: &'m Tags,
    verdict: Verdict,
    carried: Vec<Ty>,
    target_params: Vec<ValueId>,
}

struct Edge {
    label: Label,
    args: Vec<ValueId>,
}

impl Move {
    fn find(cfg: &CfgBody) -> Option<Move> {
        let domtree = DomTree::build(cfg);
        let loops = natural_loops_innermost_first(cfg, &domtree);
        let preds = cfg.predecessors();
        let loans = Loans::build(cfg);
        loops.iter().find_map(|nested| {
            let parent = loops
                .iter()
                .filter(|other| other.header != nested.header && other.contains(nested.header))
                .min_by_key(|other| other.block_count())?;
            Self::out_of(cfg, &domtree, &preds, &loans, &Nest { nested, parent })
        })
    }

    fn out_of(
        cfg: &CfgBody,
        domtree: &DomTree,
        preds: &Preds,
        loans: &Loans<'_>,
        nest: &Nest<'_>,
    ) -> Option<Move> {
        let Nest { nested, parent } = *nest;
        let header = nested.header;
        let Terminator::For {
            source,
            exit,
            exit_trip: ExitTrip::Absent,
            exit_args,
            ..
        } = &cfg.blocks[header.0].terminator
        else {
            return None;
        };
        if let ForSource::Array(_) = source {
            return None;
        }
        let exit = *cfg.label_to_block.get(exit)?;
        let breaks: Vec<BlockIdx> = preds
            .get(&exit)?
            .iter()
            .copied()
            .filter(|pred| *pred != header)
            .collect();
        let entered_from_the_nested_loop_alone =
            breaks.iter().all(|block| nested.contains(*block));
        let filled_by_nothing = exit_args.is_empty() && cfg.blocks[exit.0].params.is_empty();
        if !entered_from_the_nested_loop_alone || !filled_by_nothing {
            return None;
        }
        let target_outside_parent = nested
            .blocks()
            .filter(|block| *block != header)
            .flat_map(|block| cfg.successors(block))
            .find(|succ| !parent.contains(*succ))?;
        let from: Vec<BlockIdx> = preds.get(&target_outside_parent)?.to_vec();
        if from
            .iter()
            .any(|block| *block == header || !nested.contains(*block))
        {
            return None;
        }
        let code = Self::entered_only_from(cfg, preds, target_outside_parent)?;
        if code.iter().any(|block| parent.contains(*block)) {
            return None;
        }
        let read_of_body = Self::read_of_body(cfg, nested, &code);
        let carried = cfg.blocks[target_outside_parent.0]
            .params
            .iter()
            .chain(&read_of_body);
        for value in carried {
            if !Self::an_option_holds(cfg.val_types.get(value)?) {
                return None;
            }
        }
        if Self::touches_a_storage_of_the_iteration(cfg, loans, nested, &code) {
            return None;
        }
        let dominating_entry = domtree.idom(header)?;
        if nested.contains(dominating_entry) {
            return None;
        }
        let span = Self::span_of(cfg, &code)?;
        Some(Move {
            header,
            exit,
            breaks,
            from,
            target_outside_parent,
            code,
            read_of_body,
            dominating_entry,
            span,
        })
    }

    fn entered_only_from(cfg: &CfgBody, preds: &Preds, target: BlockIdx) -> Option<Vec<BlockIdx>> {
        let mut code = vec![target];
        let mut work = vec![target];
        while let Some(block) = work.pop() {
            for succ in cfg.successors(block) {
                if !code.contains(&succ) {
                    code.push(succ);
                    work.push(succ);
                }
            }
        }
        let entered_from_elsewhere = code.iter().skip(1).any(|block| {
            preds
                .get(block)
                .into_iter()
                .flatten()
                .any(|pred| !code.contains(pred))
        });
        (!entered_from_elsewhere).then_some(code)
    }

    fn read_of_body(cfg: &CfgBody, nested: &NaturalLoop, code: &[BlockIdx]) -> Vec<ValueId> {
        let mut defined: FxHashSet<ValueId> = FxHashSet::default();
        for block in nested.blocks().filter(|block| *block != nested.header) {
            let held = &cfg.blocks[block.0];
            defined.extend(held.params.iter().copied());
            for inst in &held.insts {
                defined.extend(inst_info::defs(&inst.kind));
            }
        }
        let mut read: Vec<ValueId> = Vec::new();
        for &block in code {
            let held = &cfg.blocks[block.0];
            let uses = held
                .insts
                .iter()
                .flat_map(|inst| inst_info::uses(&inst.kind))
                .chain(inst_info::terminator_uses(&held.terminator));
            for used in uses {
                if defined.contains(&used) && !read.contains(&used) {
                    read.push(used);
                }
            }
        }
        read
    }

    fn an_option_holds(ty: &Ty) -> bool {
        !matches!(ty, Ty::Ref(..) | Ty::Order | Ty::Never) && ty.is_word() == Some(true)
    }

    fn touches_a_storage_of_the_iteration(
        cfg: &CfgBody,
        loans: &Loans<'_>,
        nested: &NaturalLoop,
        code: &[BlockIdx],
    ) -> bool {
        let touched_in = |blocks: &mut dyn Iterator<Item = BlockIdx>| -> FxHashSet<ValueId> {
            blocks
                .flat_map(|block| &cfg.blocks[block.0].insts)
                .flat_map(|inst| touched_slots(loans, &inst.kind))
                .collect()
        };
        let by_code = touched_in(&mut code.iter().copied());
        let by_nested = touched_in(&mut nested.blocks());
        let elsewhere = touched_in(
            &mut (0..cfg.blocks.len())
                .map(BlockIdx)
                .filter(|block| !nested.contains(*block) && !code.contains(block)),
        );
        by_code
            .iter()
            .any(|slot| by_nested.contains(slot) && !elsewhere.contains(slot))
    }

    fn span_of(cfg: &CfgBody, code: &[BlockIdx]) -> Option<Span> {
        code.iter().find_map(|block| {
            let held = &cfg.blocks[block.0];
            match (held.insts.first(), &held.terminator) {
                (Some(inst), _) => Some(inst.span),
                (None, Terminator::Return { span, .. }) => Some(*span),
                (None, _) => None,
            }
        })
    }

    fn apply(self, cfg: &mut CfgBody, tags: &Tags) {
        let target_params = cfg.blocks[self.target_outside_parent.0].params.clone();
        let carried: Vec<Ty> = target_params
            .iter()
            .chain(&self.read_of_body)
            .map(|value| cfg.val_types[value].clone())
            .collect();
        let rewrite = Rewrite {
            moved: &self,
            tags,
            verdict: Verdict::of(&carried),
            carried,
            target_params,
        };
        let mut labels = Labels::of(cfg);
        rewrite.pass_stay_on_the_header_exit(cfg);
        for &from in &self.from {
            rewrite.break_with_leave_from(cfg, &mut labels, from);
        }
        let leave = labels.fresh();
        let received = rewrite.branch_to_leave_or_to_what_followed(cfg, &mut labels, leave);
        rewrite.unpack_and_run_the_moved_code(cfg, leave, received);
    }
}

impl Rewrite<'_> {
    fn emit(&self, insts: &mut Vec<Inst>, kind: InstKind) {
        insts.push(Inst {
            span: self.moved.span,
            kind,
        });
    }

    fn pass_stay_on_the_header_exit(&self, cfg: &mut CfgBody) {
        let stay = fresh(cfg, self.verdict.ty());
        let kind = match self.verdict {
            Verdict::BoolLeaves => InstKind::Const {
                dst: stay,
                value: Literal::Bool(false),
            },
            Verdict::SomeCarries { .. } => InstKind::MakeVariant {
                dst: stay,
                tag: self.tags.none,
                payload: None,
            },
        };
        let entry = self.moved.dominating_entry;
        self.emit(&mut cfg.blocks[entry.0].insts, kind);
        let header = self.moved.header;
        let Terminator::For { exit_args, .. } = &mut cfg.blocks[header.0].terminator else {
            panic!("block {} heads the `for` the pass read", header.0)
        };
        exit_args.push(stay);
        let exit = cfg.blocks[self.moved.exit.0].label;
        for block in &self.moved.breaks {
            pass_on_edges_to(&mut cfg.blocks[block.0].terminator, exit, stay);
        }
    }

    fn break_with_leave_from(&self, cfg: &mut CfgBody, labels: &mut Labels, from: BlockIdx) {
        let label = labels.fresh();
        let params: Vec<ValueId> = self
            .target_params
            .iter()
            .map(|param| {
                let ty = cfg.val_types[param].clone();
                fresh(cfg, ty)
            })
            .collect();
        let leave = fresh(cfg, self.verdict.ty());
        let mut insts: Vec<Inst> = Vec::new();
        match &self.verdict {
            Verdict::BoolLeaves => self.emit(
                &mut insts,
                InstKind::Const {
                    dst: leave,
                    value: Literal::Bool(true),
                },
            ),
            Verdict::SomeCarries { payload } => {
                let elements: Vec<ValueId> = params
                    .iter()
                    .chain(&self.moved.read_of_body)
                    .copied()
                    .collect();
                let held = match elements[..] {
                    [one] => one,
                    _ => {
                        let tuple = fresh(cfg, payload.clone());
                        self.emit(
                            &mut insts,
                            InstKind::MakeTuple {
                                dst: tuple,
                                elements,
                            },
                        );
                        tuple
                    }
                };
                self.emit(
                    &mut insts,
                    InstKind::MakeVariant {
                        dst: leave,
                        tag: self.tags.some,
                        payload: Some(held),
                    },
                );
            }
        }
        push_block(
            cfg,
            Block {
                label,
                params,
                insts,
                terminator: Terminator::Jump {
                    label: cfg.blocks[self.moved.exit.0].label,
                    args: vec![leave],
                },
            },
        );
        let target = cfg.blocks[self.moved.target_outside_parent.0].label;
        retarget(&mut cfg.blocks[from.0].terminator, target, label);
    }

    /// What followed the nested loop moves to a block of its own unless it
    /// is one plain jump, which the branch takes in its place.
    fn branch_to_leave_or_to_what_followed(
        &self,
        cfg: &mut CfgBody,
        labels: &mut Labels,
        leave: Label,
    ) -> ValueId {
        let received = fresh(cfg, self.verdict.ty());
        let exit = self.moved.exit;
        let exit_block = &mut cfg.blocks[exit.0];
        exit_block.params.push(received);
        let followed = Block {
            label: labels.fresh(),
            params: Vec::new(),
            insts: std::mem::take(&mut exit_block.insts),
            terminator: std::mem::replace(&mut exit_block.terminator, Terminator::Fallthrough),
        };
        let stay = match followed {
            Block {
                insts,
                terminator: Terminator::Jump { label, args },
                ..
            } if insts.is_empty() => Edge { label, args },
            followed => {
                let label = followed.label;
                push_block(cfg, followed);
                Edge {
                    label,
                    args: Vec::new(),
                }
            }
        };
        let mut insts: Vec<Inst> = Vec::new();
        let cond = match self.verdict {
            Verdict::BoolLeaves => received,
            Verdict::SomeCarries { .. } => {
                let holds = fresh(cfg, Ty::Bool);
                self.emit(
                    &mut insts,
                    InstKind::TestVariant {
                        dst: holds,
                        src: received,
                        tag: self.tags.some,
                    },
                );
                holds
            }
        };
        let exit_block = &mut cfg.blocks[exit.0];
        exit_block.insts = insts;
        exit_block.terminator = Terminator::JumpIf {
            cond,
            then_label: leave,
            then_args: Vec::new(),
            else_label: stay.label,
            else_args: stay.args,
        };
        received
    }

    fn unpack_and_run_the_moved_code(&self, cfg: &mut CfgBody, leave: Label, received: ValueId) {
        let mut insts: Vec<Inst> = Vec::new();
        let mut unpacked: Vec<ValueId> = Vec::new();
        if let Verdict::SomeCarries { payload } = &self.verdict {
            let held = fresh(cfg, payload.clone());
            self.emit(
                &mut insts,
                InstKind::UnwrapVariant {
                    dst: held,
                    src: received,
                },
            );
            match &self.carried[..] {
                [_] => unpacked.push(held),
                many => {
                    for (index, ty) in many.iter().enumerate() {
                        let element = fresh(cfg, ty.clone());
                        self.emit(
                            &mut insts,
                            InstKind::TupleIndex {
                                dst: element,
                                tuple: held,
                                index,
                            },
                        );
                        unpacked.push(element);
                    }
                }
            }
        }
        let (args, read_of_body) = unpacked.split_at(self.target_params.len());
        let target = self.moved.target_outside_parent;
        push_block(
            cfg,
            Block {
                label: leave,
                params: Vec::new(),
                insts,
                terminator: Terminator::Jump {
                    label: cfg.blocks[target.0].label,
                    args: args.to_vec(),
                },
            },
        );
        let subst: FxHashMap<ValueId, ValueId> = self
            .moved
            .read_of_body
            .iter()
            .copied()
            .zip(read_of_body.iter().copied())
            .collect();
        for block in &self.moved.code {
            let held = &mut cfg.blocks[block.0];
            for inst in &mut held.insts {
                apply_subst(&mut inst.kind, &subst);
            }
            apply_subst_terminator(&mut held.terminator, &subst);
        }
    }
}

fn fresh(cfg: &mut CfgBody, ty: Ty) -> ValueId {
    let value = cfg.val_factory.next();
    cfg.val_types.insert(value, ty);
    value
}

fn push_block(cfg: &mut CfgBody, block: Block) {
    cfg.label_to_block
        .insert(block.label, BlockIdx(cfg.blocks.len()));
    cfg.blocks.push(block);
}

fn retarget(term: &mut Terminator, from: Label, to: Label) {
    let swap = |label: &mut Label| {
        if *label == from {
            *label = to;
        }
    };
    match term {
        Terminator::Jump { label, .. } => swap(label),
        Terminator::JumpIf {
            then_label,
            else_label,
            ..
        }
        | Terminator::Diamond {
            then_label,
            else_label,
            ..
        } => {
            swap(then_label);
            swap(else_label);
        }
        Terminator::Switch { arms, default, .. } => {
            for (_, label, _) in arms {
                swap(label);
            }
            if let Some((label, _)) = default {
                swap(label);
            }
        }
        Terminator::For { .. }
        | Terminator::While { .. }
        | Terminator::Return { .. }
        | Terminator::Diverge
        | Terminator::Fallthrough => {}
    }
}

/// Every edge of `term` to `to` passes `arg` after the arguments it passed.
fn pass_on_edges_to(term: &mut Terminator, to: Label, arg: ValueId) {
    let pass = |label: &Label, args: &mut Vec<ValueId>| {
        if *label == to {
            args.push(arg);
        }
    };
    match term {
        Terminator::Jump { label, args } => pass(label, args),
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
            pass(then_label, then_args);
            pass(else_label, else_args);
        }
        Terminator::Switch { arms, default, .. } => {
            for (_, label, args) in arms {
                pass(label, args);
            }
            if let Some((label, args)) = default {
                pass(label, args);
            }
        }
        Terminator::For { .. }
        | Terminator::While { .. }
        | Terminator::Return { .. }
        | Terminator::Diverge
        | Terminator::Fallthrough => {}
    }
}

struct Labels {
    next: u32,
}

impl Labels {
    fn of(cfg: &CfgBody) -> Self {
        let next = cfg
            .blocks
            .iter()
            .map(|block| block.label)
            .filter(|label| *label != ENTRY_LABEL)
            .map(|label| label.0 + 1)
            .max()
            .unwrap_or(0);
        Self { next }
    }

    fn fresh(&mut self) -> Label {
        let label = Label(self.next);
        self.next += 1;
        label
    }
}
