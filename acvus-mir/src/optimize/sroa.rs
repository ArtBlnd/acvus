//! Scalar replacement: an aggregate that does not escape never exists
//! (RFC-0053).
//!
//! A storage slot holding an object or an enum, which nothing outside the
//! body ever reaches ([`crate::analysis::escape`]) and whose every use one
//! of [`Action`]'s arms covers, is replaced by one register per field or by
//! a `(tag, payload)` pair. The `MakeObject`/`MakeVariant` that filled it
//! keeps no reader and `dce` sweeps it, so no hash map is built and no
//! field read is a lookup.
//!
//! The phis are the ones [`SSABuilder`] already places: the key widens from
//! [`SsaVar::Local`] to [`SsaVar::Part`], and a field written under a
//! branch gets its block parameter from the machinery a whole variable
//! uses.
//!
//! The pass runs before `ssa_pass` in pass 2 and so before
//! `drop_insertion`, which is the only writer of `InstKind::Drop`. A slot
//! this pass sees therefore has no `Drop` yet, and the drops that a
//! replaced aggregate's move-only fields need are the ones
//! `drop_insertion` later places on the part registers, at each one's own
//! last use.

use std::collections::{BTreeMap, BTreeSet};

use acvus_ast::{BinOp, Literal, Span};
use acvus_utils::{Astr, LocalFactory};
use rustc_hash::{FxHashMap, FxHashSet};
use smallvec::SmallVec;

use super::ssa::{ENTRY_BLOCK, Part, SSABuilder, SsaVar};
use super::ssa_pass::{apply_subst, apply_subst_terminator, patch_instructions};
use crate::analysis::{escape, inst_info};
use crate::cfg::{BlockIdx, CfgBody, Terminator};
use crate::ir::{Inst, InstKind, Label, PathSeg, RefTarget, ValueId};
use crate::ty::Ty;

/// The registers one storage slot is replaced by, and -- for an enum --
/// the number each variant name stands for once the tag is a register.
#[derive(Debug)]
struct Shape {
    parts: BTreeMap<Part, Ty>,
    tags: BTreeMap<Astr, i64>,
}

/// What one instruction becomes.
#[derive(Debug, Clone)]
enum Action {
    Gone,
    DefineObject {
        slot: ValueId,
        fields: Vec<(Astr, ValueId)>,
    },
    DefineVariant {
        slot: ValueId,
        tag: Astr,
        payload: ValueId,
    },
    Define {
        slot: ValueId,
        part: Part,
        value: ValueId,
    },
    Use {
        dst: ValueId,
        slot: ValueId,
        part: Part,
    },
    TestTag {
        dst: ValueId,
        slot: ValueId,
        tag: Astr,
    },
}

/// A define that belongs at the end of a block other than the one whose
/// instruction asked for it.
#[derive(Debug, Clone)]
struct Tail {
    at: BlockIdx,
    action: Action,
}

pub fn run(cfg: &mut CfgBody) {
    if cfg.blocks.is_empty() {
        return;
    }
    let Some(plan) = plan(cfg) else {
        return;
    };
    rewrite(cfg, plan);
}

// -- Step 1: what can be replaced, and into what --------------------

/// Indexed by `BlockIdx`, as the block-keyed tables in `ssa_pass` are.
struct Plan {
    shapes: FxHashMap<ValueId, Shape>,
    actions: Vec<FxHashMap<usize, Action>>,
    tails: Vec<Vec<Action>>,
}

fn plan(cfg: &CfgBody) -> Option<Plan> {
    let escaped = escaped_storages(cfg);
    let from_outside: FxHashSet<ValueId> = cfg
        .params
        .iter()
        .chain(&cfg.captures)
        .map(|(_, v)| *v)
        .collect();

    let mut shapes: FxHashMap<ValueId, Shape> = FxHashMap::default();
    for slot in cfg
        .blocks
        .iter()
        .flat_map(|b| &b.insts)
        .filter_map(|inst| slot_of(&inst.kind))
        .filter(|slot| !escaped.escapes(*slot) && !from_outside.contains(slot))
    {
        // A slot whose type this body does not record has no types to give
        // its parts, so it keeps its aggregate.
        let Some(shape) = cfg.val_types.get(&slot).and_then(shape_of) else {
            continue;
        };
        shapes.insert(slot, shape);
    }
    if shapes.is_empty() {
        return None;
    }

    let aliases: FxHashMap<ValueId, ValueId> = cfg
        .blocks
        .iter()
        .flat_map(|b| &b.insts)
        .filter_map(|inst| match &inst.kind {
            InstKind::Ref {
                dst,
                target: RefTarget::Var(slot),
                path,
                ..
            } if path.is_empty() && shapes.contains_key(slot) => Some((*dst, *slot)),
            _ => None,
        })
        .collect();

    let mut plan = Plan {
        shapes,
        actions: vec![FxHashMap::default(); cfg.blocks.len()],
        tails: vec![Vec::new(); cfg.blocks.len()],
    };
    // Refusing a slot invalidates the actions already recorded for it, so
    // the walk starts over until it refuses nothing. Each round refuses at
    // least one slot, and there are finitely many.
    loop {
        plan.actions.iter_mut().for_each(|a| a.clear());
        plan.tails.iter_mut().for_each(|t| t.clear());
        let refused = classify(cfg, &aliases, &mut plan);
        if refused.is_empty() {
            break;
        }
        for slot in refused {
            plan.shapes.remove(&slot);
        }
        if plan.shapes.is_empty() {
            return None;
        }
    }
    match plan.actions.iter().all(|a| a.is_empty()) {
        true => None,
        false => Some(plan),
    }
}

/// Every slot whose replacement this walk refuses, with the plan filled in
/// for the ones it accepts.
///
/// A terminator is not examined: the only value one lets out is the one a
/// `Return` carries, and [`escaped_storages`] has already refused that.
fn classify(cfg: &CfgBody, aliases: &FxHashMap<ValueId, ValueId>, plan: &mut Plan) -> Vec<ValueId> {
    let mut refused = Vec::new();
    for (bi, block) in cfg.blocks.iter().enumerate() {
        for (ii, inst) in block.insts.iter().enumerate() {
            let touched = touched_slots(&inst.kind, aliases, &plan.shapes);
            if touched.is_empty() {
                continue;
            }
            match action_for(cfg, &inst.kind, aliases, plan, BlockIdx(bi)) {
                Some(action) => {
                    let taken = plan.actions[bi].insert(ii, action);
                    assert!(taken.is_none(), "two actions for one instruction");
                }
                None => refused.extend(touched),
            }
        }
    }
    refused
}

/// The action that replaces `kind`, or `None` when no rule covers the way
/// it names a slot -- which refuses that slot.
fn action_for(
    cfg: &CfgBody,
    kind: &InstKind,
    aliases: &FxHashMap<ValueId, ValueId>,
    plan: &mut Plan,
    at: BlockIdx,
) -> Option<Action> {
    let named = |v: &ValueId| -> Option<ValueId> {
        aliases
            .get(v)
            .copied()
            .or_else(|| plan.shapes.contains_key(v).then_some(*v))
    };
    match kind {
        InstKind::Assign {
            target: RefTarget::Var(slot),
            path,
            value,
        } if plan.shapes.contains_key(slot) => match path.as_slice() {
            [] => whole_assign(cfg, *slot, *value, plan, at),
            [PathSeg::Field(f)] if plan.shapes[slot].parts.contains_key(&Part::Field(*f)) => {
                Some(Action::Define {
                    slot: *slot,
                    part: Part::Field(*f),
                    value: *value,
                })
            }
            _ => None,
        },
        InstKind::Take { dst, target, path } => {
            let slot = named(&escape::named_storage(target))?;
            let part = match path.as_slice() {
                [PathSeg::Field(f)] => Part::Field(*f),
                [PathSeg::Payload] => Part::Payload,
                _ => return None,
            };
            plan.shapes[&slot].parts.contains_key(&part).then_some(())?;
            Some(Action::Use {
                dst: *dst,
                slot,
                part,
            })
        }
        InstKind::Ref {
            target: RefTarget::Var(slot),
            path,
            ..
        } if plan.shapes.contains_key(slot) && path.is_empty() => Some(Action::Gone),
        InstKind::TestVariant { dst, src, tag } => {
            let slot = named(src)?;
            plan.shapes[&slot].tags.contains_key(tag).then_some(())?;
            Some(Action::TestTag {
                dst: *dst,
                slot,
                tag: *tag,
            })
        }
        InstKind::UnwrapVariant { dst, src } => {
            let slot = named(src)?;
            plan.shapes[&slot]
                .parts
                .contains_key(&Part::Payload)
                .then_some(())?;
            Some(Action::Use {
                dst: *dst,
                slot,
                part: Part::Payload,
            })
        }
        _ => None,
    }
}

/// `Assign { Var(slot), [], value }`: the parts `value` is made of, defined
/// where the constructor that made them is.
fn whole_assign(
    cfg: &CfgBody,
    slot: ValueId,
    value: ValueId,
    plan: &mut Plan,
    at: BlockIdx,
) -> Option<Action> {
    if let Some(action) = constructor_action(cfg, slot, value) {
        return Some(action);
    }
    // A value that reaches the slot through a block parameter is a join of
    // constructors: each predecessor takes its own apart, and the phi the
    // parts then need is the one the SSA builder places for them.
    let param = cfg.blocks[at.0].params.iter().position(|p| *p == value)?;
    let label = cfg.blocks[at.0].label;
    let mut tails: Vec<Tail> = Vec::new();
    for (pi, pred) in cfg.blocks.iter().enumerate() {
        for args in incoming(&pred.terminator, label) {
            let arg = args
                .get(param)
                .expect("a jump carries one argument per block parameter");
            tails.push(Tail {
                at: BlockIdx(pi),
                action: constructor_action(cfg, slot, *arg)?,
            });
        }
    }
    for tail in tails {
        plan.tails[tail.at.0].push(tail.action);
    }
    Some(Action::Gone)
}

fn constructor_action(cfg: &CfgBody, slot: ValueId, value: ValueId) -> Option<Action> {
    let defining = cfg
        .blocks
        .iter()
        .flat_map(|b| &b.insts)
        .find(|inst| inst_info::defs(&inst.kind).contains(&value))?;
    match &defining.kind {
        InstKind::MakeObject { fields, .. } => Some(Action::DefineObject {
            slot,
            fields: fields.clone(),
        }),
        InstKind::MakeVariant {
            tag,
            payload: Some(payload),
            ..
        } => Some(Action::DefineVariant {
            slot,
            tag: *tag,
            payload: *payload,
        }),
        _ => None,
    }
}

/// The candidate slots this instruction names, whether or not a rule covers
/// the way it names them.
fn touched_slots(
    kind: &InstKind,
    aliases: &FxHashMap<ValueId, ValueId>,
    shapes: &FxHashMap<ValueId, Shape>,
) -> SmallVec<[ValueId; 2]> {
    let mut out: SmallVec<[ValueId; 2]> = SmallVec::new();
    let mut push = |v: ValueId| {
        let slot = aliases.get(&v).copied().unwrap_or(v);
        if shapes.contains_key(&slot) && !out.contains(&slot) {
            out.push(slot);
        }
    };
    if let Some(slot) = slot_of(kind) {
        push(slot);
    }
    inst_info::uses(kind).into_iter().for_each(push);
    out
}

/// The local slot a place names directly. An extern parameter and a place
/// through a reference name no local of this body.
fn slot_of(kind: &InstKind) -> Option<ValueId> {
    match kind {
        InstKind::Ref {
            target: RefTarget::Var(slot),
            ..
        }
        | InstKind::Take {
            target: RefTarget::Var(slot),
            ..
        }
        | InstKind::Assign {
            target: RefTarget::Var(slot),
            ..
        } => Some(*slot),
        _ => None,
    }
}

fn escaped_storages(cfg: &CfgBody) -> escape::Escaped {
    let mut scan = escape::EscapeScan::default();
    for block in &cfg.blocks {
        for inst in &block.insts {
            scan.observe(&inst.kind);
        }
        if let Terminator::Return { value, .. } = &block.terminator {
            scan.observe_returned(*value);
        }
    }
    scan.finish()
}

/// The registers a type is made of, or `None` where its parts have none: a
/// tuple, an array or any other aggregate this pass does not name; an enum
/// with a variant that carries nothing, which would leave the payload
/// register undefined on that arm; and an enum whose variants disagree on
/// the payload type, whose merge would have no type at all. Each of those
/// keeps its aggregate.
fn shape_of(ty: &Ty) -> Option<Shape> {
    match ty {
        Ty::Object(fields) => Some(Shape {
            parts: fields
                .iter()
                .map(|(name, ty)| (Part::Field(*name), ty.clone()))
                .collect(),
            tags: BTreeMap::new(),
        }),
        Ty::Enum { variants, .. } => {
            let mut payload: Option<&Ty> = None;
            for carried in variants.values() {
                let carried = carried.as_deref()?;
                match payload {
                    None => payload = Some(carried),
                    Some(seen) if seen == carried => {}
                    Some(_) => return None,
                }
            }
            let tags = variants
                .keys()
                .copied()
                .collect::<BTreeSet<Astr>>()
                .into_iter()
                .enumerate()
                .map(|(number, tag)| {
                    let number =
                        i64::try_from(number).expect("an enum has fewer variants than i64 counts");
                    (tag, number)
                })
                .collect();
            Some(Shape {
                parts: BTreeMap::from([(Part::Tag, Ty::I64), (Part::Payload, payload?.clone())]),
                tags,
            })
        }
        _ => None,
    }
}

fn incoming(term: &Terminator, label: Label) -> Vec<&Vec<ValueId>> {
    match term {
        Terminator::Jump { label: to, args } if *to == label => vec![args],
        Terminator::JumpIf {
            then_label,
            then_args,
            else_label,
            else_args,
            ..
        } => [(then_label, then_args), (else_label, else_args)]
            .into_iter()
            .filter(|(to, _)| **to == label)
            .map(|(_, args)| args)
            .collect(),
        Terminator::Switch { arms, default, .. } => arms
            .iter()
            .map(|(_, to, args)| (to, args))
            .chain(default.iter().map(|(to, args)| (to, args)))
            .filter(|(to, _)| **to == label)
            .map(|(_, args)| args)
            .collect(),
        Terminator::Jump { .. }
        | Terminator::Return { .. }
        | Terminator::Fallthrough
        | Terminator::Diverge => Vec::new(),
    }
}

// -- Step 2: run the SSA builder over the parts ---------------------

/// The SSA construction in progress, with the two tables every new register
/// is born into and the block being rewritten.
struct Rewriter<'a> {
    plan: &'a Plan,
    ssa: SSABuilder,
    val_factory: &'a mut LocalFactory<ValueId>,
    val_types: &'a mut FxHashMap<ValueId, Ty>,
    out: Vec<Inst>,
    /// What a deleted read's result now stands for.
    subst: FxHashMap<ValueId, ValueId>,
}

fn rewrite(cfg: &mut CfgBody, plan: Plan) {
    let preds = cfg.predecessors();
    let successors: Vec<SmallVec<[BlockIdx; 2]>> = (0..cfg.blocks.len())
        .map(|i| cfg.successors(BlockIdx(i)))
        .collect();
    let labels: Vec<Label> = cfg.blocks.iter().map(|b| b.label).collect();
    let loop_headers: BTreeSet<BlockIdx> = successors
        .iter()
        .enumerate()
        .flat_map(|(bi, succs)| succs.iter().filter(move |s| s.0 <= bi))
        .copied()
        .collect();

    let mut ssa = SSABuilder::new();
    for (block, block_preds) in &preds {
        for pred in block_preds {
            ssa.add_predecessor(labels[block.0], labels[pred.0]);
        }
    }

    let CfgBody {
        blocks,
        val_factory,
        val_types,
        ..
    } = cfg;
    let mut rw = Rewriter {
        plan: &plan,
        ssa,
        val_factory,
        val_types,
        out: Vec::new(),
        subst: FxHashMap::default(),
    };

    // Every part starts from the `Undef` a promoted local starts from
    // (`ssa_pass::materialize_entry_defs`), so a read on a path that has
    // not written it still reaches a definition.
    let mut entry_undefs = Vec::new();
    for (slot, shape) in &plan.shapes {
        for part in shape.parts.keys() {
            let var = SsaVar::Part(*slot, *part);
            let undef = rw.alloc(var);
            rw.ssa.define(ENTRY_BLOCK, var, undef);
            entry_undefs.push(undef);
        }
    }

    let old: Vec<Vec<Inst>> = blocks
        .iter_mut()
        .map(|b| std::mem::take(&mut b.insts))
        .collect();

    for (bi, insts) in old.into_iter().enumerate() {
        let label = labels[bi];
        if bi > 0 && !loop_headers.contains(&BlockIdx(bi)) {
            rw.seal(label);
        }
        rw.out = Vec::with_capacity(insts.len());
        if bi == 0 {
            let head = entry_undefs.iter().map(|dst| Inst {
                span: Span::ZERO,
                kind: InstKind::Undef { dst: *dst },
            });
            rw.out.extend(head);
        }
        for (ii, inst) in insts.into_iter().enumerate() {
            match plan.actions[bi].get(&ii) {
                None => rw.out.push(inst),
                Some(action) => rw.apply(action, inst.span, label),
            }
        }
        for action in &plan.tails[bi] {
            rw.apply(action, Span::ZERO, label);
        }
        blocks[bi].insts = std::mem::take(&mut rw.out);
    }

    for &header in &loop_headers {
        rw.seal(labels[header.0]);
    }

    let Rewriter { ssa, mut subst, .. } = rw;
    let (phis, trivial) = ssa.finish();
    for value in subst.values_mut() {
        *value = resolved(&trivial, *value);
    }
    subst.extend(
        trivial
            .iter()
            .map(|(removed, stands_for)| (*removed, resolved(&trivial, *stands_for))),
    );

    if !phis.is_empty() {
        patch_instructions(cfg, &phis);
    }
    for block in &mut cfg.blocks {
        for inst in &mut block.insts {
            apply_subst(&mut inst.kind, &subst);
        }
        apply_subst_terminator(&mut block.terminator, &subst);
    }
}

impl Rewriter<'_> {
    fn alloc(&mut self, var: SsaVar) -> ValueId {
        part_val(self.plan, var, self.val_factory, self.val_types)
    }

    fn seal(&mut self, label: Label) {
        let Self {
            plan,
            ssa,
            val_factory,
            val_types,
            ..
        } = self;
        ssa.seal_block(label, &mut |var| {
            part_val(plan, var, val_factory, val_types)
        });
    }

    fn use_part(&mut self, label: Label, var: SsaVar) -> ValueId {
        let Self {
            plan,
            ssa,
            val_factory,
            val_types,
            ..
        } = self;
        ssa.use_var(label, var, &mut |var| {
            part_val(plan, var, val_factory, val_types)
        })
    }

    /// A register holding the number `tag` stands for, defined right here.
    fn tag_const(&mut self, slot: ValueId, tag: Astr, span: Span) -> ValueId {
        let number = self.plan.shapes[&slot].tags[&tag];
        let dst = self.val_factory.next();
        self.val_types.insert(dst, Ty::I64);
        self.out.push(Inst {
            span,
            kind: InstKind::Const {
                dst,
                value: Literal::Int(i128::from(number)),
            },
        });
        dst
    }

    fn apply(&mut self, action: &Action, span: Span, label: Label) {
        match action {
            Action::Gone => {}
            Action::Define { slot, part, value } => {
                self.ssa.define(label, SsaVar::Part(*slot, *part), *value);
            }
            Action::DefineObject { slot, fields } => {
                for (name, value) in fields {
                    self.ssa
                        .define(label, SsaVar::Part(*slot, Part::Field(*name)), *value);
                }
            }
            Action::DefineVariant { slot, tag, payload } => {
                let number = self.tag_const(*slot, *tag, span);
                self.ssa
                    .define(label, SsaVar::Part(*slot, Part::Tag), number);
                self.ssa
                    .define(label, SsaVar::Part(*slot, Part::Payload), *payload);
            }
            Action::Use { dst, slot, part } => {
                let value = self.use_part(label, SsaVar::Part(*slot, *part));
                let taken = self.subst.insert(*dst, value);
                assert!(taken.is_none(), "{dst:?} is read out of storage twice");
            }
            Action::TestTag { dst, slot, tag } => {
                let held = self.use_part(label, SsaVar::Part(*slot, Part::Tag));
                let number = self.tag_const(*slot, *tag, span);
                self.out.push(Inst {
                    span,
                    kind: InstKind::BinOp {
                        dst: *dst,
                        op: BinOp::Eq,
                        left: held,
                        right: number,
                    },
                });
            }
        }
    }
}

/// The value `removed` stands for once every removed phi on the way is
/// replaced. A phi stands for a value defined before it, so the walk
/// descends.
fn resolved(trivial: &FxHashMap<ValueId, ValueId>, removed: ValueId) -> ValueId {
    let mut at = removed;
    let mut seen = FxHashSet::default();
    while let Some(&next) = trivial.get(&at) {
        assert!(seen.insert(at), "a removed phi stands for itself: {at:?}");
        at = next;
    }
    at
}

fn part_val(
    plan: &Plan,
    var: SsaVar,
    val_factory: &mut LocalFactory<ValueId>,
    val_types: &mut FxHashMap<ValueId, Ty>,
) -> ValueId {
    let SsaVar::Part(slot, part) = var else {
        panic!("scalar replacement tracks only parts, got {var:?}");
    };
    let ty = plan.shapes[&slot].parts[&part].clone();
    let dst = val_factory.next();
    val_types.insert(dst, ty);
    dst
}
