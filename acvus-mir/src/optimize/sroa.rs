//! Scalar replacement: an aggregate that does not escape never exists
//! (RFC-0050).
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
//! Jump threading lives here rather than in a pass of its own because
//! after this pass a tag is a numeric register, and no IR form hands a
//! numeric tag phi to a later pass to thread.
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
use crate::analysis::domtree::DomTree;
use crate::analysis::{escape, inst_info};
use crate::cfg::{BlockIdx, CfgBody, Terminator, prune, reachable};
use crate::ir::{Inst, InstKind, Label, PathSeg, RefTarget, SwitchKey, ValueId};
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

/// One edge of a dispatch: where it goes and what it carries.
#[derive(Debug, Clone)]
struct Edge {
    label: Label,
    args: Vec<ValueId>,
}

#[derive(Debug, Clone)]
struct Arm {
    tag: Astr,
    edge: Edge,
}

/// A `Switch` (RFC-0051) over a slot this pass replaces, of any width.
#[derive(Debug, Clone)]
struct Dispatch {
    slot: ValueId,
    arms: Vec<Arm>,
    default: Option<Edge>,
}

impl Dispatch {
    /// `None` is a tag no arm names in a `Switch` without a `default`,
    /// which is a path `validate::exhaustive` has already ruled out.
    fn edge_for(&self, tag: Astr) -> Option<&Edge> {
        self.arms
            .iter()
            .find(|arm| arm.tag == tag)
            .map(|arm| &arm.edge)
            .or(self.default.as_ref())
    }
}

/// A dispatch of three or more edges has no compare form here -- the chain
/// would need blocks this pass does not make, and building them would put a
/// second block layout next to the one `lower` already emits. A slot whose
/// dispatch is neither threaded nor two-edged keeps its aggregate instead,
/// and the machine's own `ops::switch` reads its tag.
#[derive(Debug, Clone)]
struct Compare {
    slot: ValueId,
    tag: Astr,
    taken: Edge,
    fallen: Edge,
}

/// One argument of a threaded jump, read against the arguments that jump
/// already carried to the dispatch block.
#[derive(Debug, Clone, Copy)]
enum Carry {
    Arg(usize),
    Value(ValueId),
}

#[derive(Debug, Clone)]
struct Threaded {
    pred: BlockIdx,
    to: Label,
    carry: Vec<Carry>,
}

#[derive(Debug, Clone)]
enum Decision {
    /// RFC-0051 rule 4.
    Jump(Edge),
    Thread {
        edges: Vec<Threaded>,
        unthreaded: Option<Compare>,
    },
    Compare(Compare),
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
    /// The `Switch` each block ends in, where this pass replaces the slot
    /// it reads.
    dispatches: Vec<Option<Dispatch>>,
    decisions: Vec<Option<Decision>>,
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
        dispatches: vec![None; cfg.blocks.len()],
        decisions: vec![None; cfg.blocks.len()],
    };
    // Refusing a slot invalidates the actions already recorded for it, so
    // the walk starts over until it refuses nothing. Each round refuses at
    // least one slot, and there are finitely many.
    loop {
        plan.actions.iter_mut().for_each(|a| a.clear());
        plan.tails.iter_mut().for_each(|t| t.clear());
        plan.dispatches.iter_mut().for_each(|d| *d = None);
        plan.decisions.iter_mut().for_each(|d| *d = None);
        let mut refused = classify(cfg, &aliases, &mut plan);
        if refused.is_empty() {
            refused = decide(cfg, &mut plan);
        }
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
/// The only terminator that names a slot is `Terminator::Switch`, whose tag
/// is one (RFC-0051 rule 5); the only value any other lets out is the one a
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
        let Terminator::Switch { tag, arms, default } = &block.terminator else {
            continue;
        };
        // A tag that names no slot still planned — never one, or one an
        // earlier round refused — is the machine's to dispatch.
        let slot = aliases.get(tag).copied().unwrap_or(*tag);
        let Some(shape) = plan.shapes.get(&slot) else {
            continue;
        };
        match dispatch_for(shape, slot, arms, default.as_ref()) {
            Some(dispatch) => plan.dispatches[bi] = Some(dispatch),
            None => refused.push(slot),
        }
    }
    refused
}

/// The `Switch` over a replaced slot, of any width, or `None` where an arm
/// names a key this shape does not number. Only a tag is numbered: the slot
/// a replacement takes apart is an enum, and a literal dispatch reads a word
/// this pass never built.
fn dispatch_for(
    shape: &Shape,
    slot: ValueId,
    arms: &[(SwitchKey, Label, Vec<ValueId>)],
    default: Option<&(Label, Vec<ValueId>)>,
) -> Option<Dispatch> {
    let edge = |label: &Label, args: &Vec<ValueId>| Edge {
        label: *label,
        args: args.clone(),
    };
    let numbered = |key: &SwitchKey| key.tag().filter(|tag| shape.tags.contains_key(tag));
    Some(Dispatch {
        slot,
        arms: arms
            .iter()
            .map(|(key, label, args)| {
                Some(Arm {
                    tag: numbered(key)?,
                    edge: edge(label, args),
                })
            })
            .collect::<Option<Vec<Arm>>>()?,
        default: default.map(|(label, args)| edge(label, args)),
    })
}

// -- Step 1b: which dispatches a known tag threads ------------------

/// The tags that reach a point: `None` before any predecessor has been
/// read, `Many` where two edges disagree or the part is still the entry
/// `Undef`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Reaching {
    None,
    One(Astr),
    Many,
}

impl Reaching {
    fn meet(self, other: Self) -> Self {
        match (self, other) {
            (Self::None, seen) | (seen, Self::None) => seen,
            (Self::One(a), Self::One(b)) if a == b => Self::One(a),
            _ => Self::Many,
        }
    }
}

/// The tag `slot` holds at the end of each block: the last variant this
/// plan writes there, or what its predecessors agree on. The entry block
/// starts at `Many` because a part begins at `Undef`, which stands for no
/// tag at all.
fn reaching_tags(
    cfg: &CfgBody,
    plan: &Plan,
    preds: &FxHashMap<BlockIdx, SmallVec<[BlockIdx; 2]>>,
    slot: ValueId,
) -> Vec<Reaching> {
    let written: Vec<Option<Astr>> = (0..cfg.blocks.len())
        .map(|bi| written_tag(plan, bi, slot))
        .collect();
    let mut out = vec![Reaching::None; cfg.blocks.len()];
    let mut moved = true;
    while moved {
        moved = false;
        for bi in 0..cfg.blocks.len() {
            let entering = match bi {
                0 => Reaching::Many,
                _ => preds
                    .get(&BlockIdx(bi))
                    .into_iter()
                    .flatten()
                    .fold(Reaching::None, |seen, pred| seen.meet(out[pred.0])),
            };
            let leaving = written[bi].map_or(entering, Reaching::One);
            if leaving != out[bi] {
                out[bi] = leaving;
                moved = true;
            }
        }
    }
    out
}

/// The tag the last `DefineVariant` of this block writes into `slot`. A
/// tail is written after every instruction of the block, so it wins.
fn written_tag(plan: &Plan, bi: usize, slot: ValueId) -> Option<Astr> {
    let tag_of = |action: &Action| match action {
        Action::DefineVariant { slot: at, tag, .. } if *at == slot => Some(*tag),
        _ => None,
    };
    plan.tails[bi].iter().rev().find_map(tag_of).or_else(|| {
        plan.actions[bi]
            .iter()
            .filter_map(|(ii, action)| tag_of(action).map(|tag| (*ii, tag)))
            .max_by_key(|(ii, _)| *ii)
            .map(|(_, tag)| tag)
    })
}

/// The graph a threading decision reads: which blocks precede a block, and
/// whether a register reaches one.
struct Reachability<'a> {
    cfg: &'a CfgBody,
    preds: FxHashMap<BlockIdx, SmallVec<[BlockIdx; 2]>>,
    domtree: DomTree,
    /// Where each register an instruction defines is defined. A parameter, a
    /// capture and a block parameter are absent: each reaches every block a
    /// walk from the entry passes.
    defs: FxHashMap<ValueId, BlockIdx>,
}

impl<'a> Reachability<'a> {
    fn of(cfg: &'a CfgBody) -> Self {
        Self {
            cfg,
            preds: cfg.predecessors(),
            domtree: DomTree::build(cfg),
            defs: cfg
                .blocks
                .iter()
                .enumerate()
                .flat_map(|(bi, block)| {
                    block
                        .insts
                        .iter()
                        .flat_map(|inst| inst_info::defs(&inst.kind))
                        .map(move |dst| (dst, BlockIdx(bi)))
                })
                .collect(),
        }
    }

    fn preds_of(&self, at: BlockIdx) -> &[BlockIdx] {
        self.preds.get(&at).map_or(&[], |preds| preds.as_slice())
    }

    fn reaches(&self, value: ValueId, at: BlockIdx) -> bool {
        self.defs
            .get(&value)
            .is_none_or(|def| self.domtree.dominates(*def, at))
    }
}

/// What each dispatch becomes, with every slot whose dispatch has no form
/// this pass can write.
fn decide(cfg: &CfgBody, plan: &mut Plan) -> Vec<ValueId> {
    let graph = Reachability::of(cfg);
    let mut cache: FxHashMap<ValueId, Vec<Reaching>> = FxHashMap::default();
    let mut decisions: Vec<Option<Decision>> = vec![None; cfg.blocks.len()];
    let mut refused = Vec::new();
    for bi in 0..cfg.blocks.len() {
        let Some(dispatch) = plan.dispatches[bi].clone() else {
            continue;
        };
        let tags = cache
            .entry(dispatch.slot)
            .or_insert_with(|| reaching_tags(cfg, plan, &graph.preds, dispatch.slot));
        match decide_one(&graph, plan, &dispatch, tags, BlockIdx(bi)) {
            Some(decision) => decisions[bi] = Some(decision),
            None => refused.push(dispatch.slot),
        }
    }
    plan.decisions = decisions;
    refused
}

fn decide_one(
    graph: &Reachability<'_>,
    plan: &Plan,
    dispatch: &Dispatch,
    tags: &[Reaching],
    at: BlockIdx,
) -> Option<Decision> {
    if let Reaching::One(tag) = tags[at.0]
        && let Some(edge) = dispatch.edge_for(tag)
    {
        return Some(Decision::Jump(edge.clone()));
    }
    let open = threadable(graph.cfg, plan, dispatch, at);
    let mut edges = Vec::new();
    let mut left = 0usize;
    for pred in graph.preds_of(at) {
        let threaded = match (open, tags[pred.0]) {
            (true, Reaching::One(tag)) => dispatch
                .edge_for(tag)
                .and_then(|edge| thread_one(graph, edge, at, *pred)),
            _ => None,
        };
        match threaded {
            Some(threaded) => edges.push(threaded),
            None => left += 1,
        }
    }
    let unthreaded = match left {
        0 => None,
        _ => Some(compare_of(dispatch)?),
    };
    match (edges.is_empty(), unthreaded) {
        (true, Some(compare)) => Some(Decision::Compare(compare)),
        (_, unthreaded) => Some(Decision::Thread { edges, unthreaded }),
    }
}

/// A predecessor may leave for an arm only where the dispatch block runs
/// nothing of its own: every instruction it holds is one this pass deletes,
/// so the path that skips the block skips nothing.
fn threadable(cfg: &CfgBody, plan: &Plan, dispatch: &Dispatch, at: BlockIdx) -> bool {
    let block = &cfg.blocks[at.0];
    let deleted = plan.actions[at.0].len() == block.insts.len()
        && plan.actions[at.0]
            .values()
            .all(|action| matches!(action, Action::Gone))
        && plan.tails[at.0].is_empty();
    let mut targets = dispatch
        .arms
        .iter()
        .map(|arm| &arm.edge)
        .chain(dispatch.default.iter());
    deleted
        && targets
            .all(|edge| edge.label != block.label && cfg.label_to_block.contains_key(&edge.label))
}

/// The arguments the threaded jump carries, or `None` where one of them is
/// a value that does not reach `pred`.
fn thread_one(
    graph: &Reachability<'_>,
    edge: &Edge,
    at: BlockIdx,
    pred: BlockIdx,
) -> Option<Threaded> {
    let params = &graph.cfg.blocks[at.0].params;
    let carry = edge
        .args
        .iter()
        .map(|arg| match params.iter().position(|param| param == arg) {
            Some(index) => Some(Carry::Arg(index)),
            None => graph.reaches(*arg, pred).then_some(Carry::Value(*arg)),
        })
        .collect::<Option<Vec<Carry>>>()?;
    Some(Threaded {
        pred,
        to: edge.label,
        carry,
    })
}

fn compare_of(dispatch: &Dispatch) -> Option<Compare> {
    let compare = |tag: Astr, taken: &Edge, fallen: &Edge| Compare {
        slot: dispatch.slot,
        tag,
        taken: taken.clone(),
        fallen: fallen.clone(),
    };
    match (dispatch.arms.as_slice(), &dispatch.default) {
        ([one], Some(other)) => Some(compare(one.tag, &one.edge, other)),
        ([one, other], None) => Some(compare(one.tag, &one.edge, &other.edge)),
        _ => None,
    }
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
            ..
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
        InstKind::Take {
            dst, target, path, ..
        } => {
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
        for edge in incoming(&pred.terminator, label) {
            let Incoming::Carries(args) = edge else {
                return None;
            };
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

/// An edge into a block: the arguments it passes the block's parameters,
/// or the parameters it fills itself, which no constructor stands behind.
enum Incoming<'t> {
    Carries(&'t Vec<ValueId>),
    Fills,
}

fn incoming(term: &Terminator, label: Label) -> Vec<Incoming<'_>> {
    match term {
        Terminator::Jump { label: to, args } if *to == label => vec![Incoming::Carries(args)],
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
        } => [(then_label, then_args), (else_label, else_args)]
            .into_iter()
            .filter(|(to, _)| **to == label)
            .map(|(_, args)| Incoming::Carries(args))
            .collect(),
        Terminator::Switch { arms, default, .. } => arms
            .iter()
            .map(|(_, to, args)| (to, args))
            .chain(default.iter().map(|(to, args)| (to, args)))
            .filter(|(to, _)| **to == label)
            .map(|(_, args)| Incoming::Carries(args))
            .collect(),
        // A `For`'s exit edge carries its target's whole parameter list; its
        // body edge fills the body's leading parameters itself (RFC-0057).
        Terminator::For {
            exit, exit_args, ..
        } if *exit == label => vec![Incoming::Carries(exit_args)],
        Terminator::For { body, .. } if *body == label => vec![Incoming::Fills],
        Terminator::For { .. }
        | Terminator::Jump { .. }
        | Terminator::Return { .. }
        | Terminator::Fallthrough
        | Terminator::Diverge => Vec::new(),
    }
}

// -- Step 2: thread the jumps a known tag settles -------------------

/// The graph the SSA builder then reads. A predecessor whose tag is a
/// constant leaves for its arm here, so the phi the builder places in that
/// arm stands for the edges the arm actually has, and the join the dispatch
/// used to be is one no path reaches.
fn thread(cfg: &mut CfgBody, plan: &Plan) -> Vec<bool> {
    for bi in 0..cfg.blocks.len() {
        if let Some(Decision::Jump(edge)) = &plan.decisions[bi] {
            cfg.blocks[bi].terminator = Terminator::Jump {
                label: edge.label,
                args: edge.args.clone(),
            };
        }
    }
    for bi in 0..cfg.blocks.len() {
        let Some(Decision::Thread { edges, unthreaded }) = &plan.decisions[bi] else {
            continue;
        };
        let from = cfg.blocks[bi].label;
        let arrives = match unthreaded {
            Some(_) => Arrives::StillHere,
            None => Arrives::At(sole_target(edges)),
        };
        for threaded in edges {
            retarget(&mut cfg.blocks[threaded.pred.0].terminator, from, threaded);
        }
        settle_joins(cfg, from, arrives);
    }
    let alive = reachable(cfg);
    for (block, alive) in cfg.blocks.iter_mut().zip(&alive) {
        if !alive {
            block.insts.clear();
            block.terminator = Terminator::Diverge;
        }
    }
    alive
}

/// Where the paths through a threaded dispatch end up.
enum Arrives {
    /// The dispatch keeps a predecessor, so the block is still a block.
    StillHere,
    At(Option<Label>),
}

fn sole_target(edges: &[Threaded]) -> Option<Label> {
    let (first, rest) = edges.split_first()?;
    rest.iter()
        .all(|edge| edge.to == first.to)
        .then_some(first.to)
}

/// Threading leaves the dispatch a block no path reaches (see `thread`), and
/// a `Diamond` that named it as its join now names a block that is gone. The
/// pass knows what became of the paths through it: where they all end at one
/// block, that block is where the arms now meet; where they scatter, the arms
/// no longer rejoin and the branch is not a diamond.
fn settle_joins(cfg: &mut CfgBody, from: Label, arrives: Arrives) {
    let Arrives::At(to) = arrives else {
        return;
    };
    for bi in 0..cfg.blocks.len() {
        let Terminator::Diamond { join, .. } = &mut cfg.blocks[bi].terminator else {
            continue;
        };
        if *join != from {
            continue;
        }
        match to {
            Some(to) => *join = to,
            None => crate::cfg::demote_diamond(cfg, BlockIdx(bi)),
        }
    }
}

fn retarget(term: &mut Terminator, from: Label, threaded: &Threaded) {
    let leave = |label: &mut Label, args: &mut Vec<ValueId>| {
        if *label != from {
            return;
        }
        *args = threaded
            .carry
            .iter()
            .map(|carry| match carry {
                Carry::Arg(index) => *args
                    .get(*index)
                    .expect("a jump carries one argument per block parameter"),
                Carry::Value(value) => *value,
            })
            .collect();
        *label = threaded.to;
    };
    match term {
        Terminator::Jump { label, args } => leave(label, args),
        Terminator::JumpIf {
            then_label,
            then_args,
            else_label,
            else_args,
            ..
        } => {
            leave(then_label, then_args);
            leave(else_label, else_args);
        }
        Terminator::Diamond {
            then_label,
            then_args,
            else_label,
            else_args,
            join,
            ..
        } => {
            leave(then_label, then_args);
            leave(else_label, else_args);
            if *join == from {
                *join = threaded.to;
            }
        }
        Terminator::Switch { arms, default, .. } => {
            for (_, label, args) in arms {
                leave(label, args);
            }
            if let Some((label, args)) = default {
                leave(label, args);
            }
        }
        // Only a `For`'s exit edge is threaded: its body edge names the
        // block whose leading parameters the terminator fills by position,
        // and an arm the dispatch was threaded to has no such parameters
        // (RFC-0057).
        Terminator::For {
            exit, exit_args, ..
        } => leave(exit, exit_args),
        Terminator::Return { .. } | Terminator::Fallthrough | Terminator::Diverge => {}
    }
}

fn kept_compare(decision: &Option<Decision>) -> Option<&Compare> {
    match decision {
        Some(Decision::Compare(compare)) => Some(compare),
        Some(Decision::Thread { unthreaded, .. }) => unthreaded.as_ref(),
        Some(Decision::Jump(_)) | None => None,
    }
}

// -- Step 3: run the SSA builder over the parts ---------------------

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
    let alive = thread(cfg, &plan);
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
        if !alive[bi] {
            continue;
        }
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
        // The dispatch reads the tag after every define of this block, so
        // it is written where a tail action is written and for the same
        // reason.
        if let Some(compare) = kept_compare(&plan.decisions[bi]) {
            let cond = rw.compare(compare.slot, compare.tag, Span::ZERO, label);
            blocks[bi].terminator = Terminator::JumpIf {
                cond,
                then_label: compare.taken.label,
                then_args: compare.taken.args.clone(),
                else_label: compare.fallen.label,
                else_args: compare.fallen.args.clone(),
            };
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
    prune(cfg, &alive);
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

    /// The tag register against the number `tag` stands for, in a register
    /// of its own: what a `TestVariant` becomes when it has a destination
    /// already, and what a `Switch` becomes when it has none.
    fn compare(&mut self, slot: ValueId, tag: Astr, span: Span, label: Label) -> ValueId {
        let dst = self.val_factory.next();
        self.val_types.insert(dst, Ty::Bool);
        self.apply(&Action::TestTag { dst, slot, tag }, span, label);
        dst
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
