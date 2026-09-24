//! A `for`'s body as independent parts (RFC-0089 rule 5).
//!
//! The parts are the connected components of the body's instructions, where
//! two instructions are connected when one reads a value the other defines,
//! when both touch a storage or a context one of them writes, or when one
//! lies in an arm of a branch the other's terminator decides. The body is
//! then rewritten as rule 1's chain: each part's instructions in the order
//! the body held them, a block split where it held two parts' instructions,
//! and each branch kept whole, with its arms, in the part that owns it.
//!
//! A loop left from anywhere but its header, a loop whose partition is one
//! `Sequential` part, and a `while` stay as they are. So does a `for` whose
//! header holds an instruction: that instruction runs on every entry to the
//! header, the last one included, and belongs to no part of the body. And so
//! do a loop with an arm that ends in `!` and a body that holds no
//! instruction (rule 5).

use rustc_hash::{FxHashMap, FxHashSet};

use crate::analysis::affine::AffineValues;
use crate::analysis::carried::{
    Carried, CarriedState, Dependence, MergeOp, StorageMerge, carries_order,
};
use crate::analysis::domtree::DomTree;
use crate::analysis::inst_info;
use crate::analysis::loans::Loans;
use crate::analysis::loops::{Invariants, LoopNest};
use crate::cfg::{Block, BlockIdx, CfgBody, ENTRY_LABEL, Terminator};
use crate::graph::QualifiedRef;
use crate::ir::{
    Accumulator, CallLaw, ExitTrip, FoldAccumulator, ForSource, Inst, InstKind, Label, Law, LawOp,
    Part, PartKind, Traversal, ValueId,
};
use crate::laws::LawTable;
use crate::optimize::ssa_pass::{apply_subst, apply_subst_terminator};
use crate::ty::Mutability;

pub fn run(cfg: &mut CfgBody, laws: &LawTable) {
    let mut examined: FxHashSet<Label> = FxHashSet::default();
    while let Some(header) = innermost_unexamined_for(cfg, &examined) {
        examined.insert(header);
        if let Some(partitioned) = partition(cfg, header, laws) {
            *cfg = partitioned;
        }
    }
}

fn innermost_unexamined_for(cfg: &CfgBody, examined: &FxHashSet<Label>) -> Option<Label> {
    let domtree = DomTree::build(cfg);
    let invariants = Invariants::of(cfg);
    let nest = LoopNest::of(cfg, &domtree, &invariants);
    nest.iter()
        .map(|(_, loop_)| &cfg.blocks[loop_.natural.header.0])
        .filter(|header| matches!(header.terminator, Terminator::For { .. }))
        .map(|header| header.label)
        .find(|label| !examined.contains(label))
}

fn partition(cfg: &CfgBody, header_label: Label, laws: &LawTable) -> Option<CfgBody> {
    let header = cfg.label_to_block[&header_label];
    if !cfg.blocks[header.0].insts.is_empty() {
        return None;
    }
    let domtree = DomTree::build(cfg);
    let invariants = Invariants::of(cfg);
    let nest = LoopNest::of(cfg, &domtree, &invariants);
    let loop_ = nest.get(nest.by_header(header)?);
    let loans = Loans::build(cfg);
    let affine = AffineValues::of(cfg, loop_, &invariants);
    let state = CarriedState::of(&loans, loop_, &affine, laws);
    let leaves = state
        .dependences
        .iter()
        .any(|dependence| matches!(dependence, Dependence::EarlyExit { .. }));
    if leaves {
        return None;
    }

    let mut work = cfg.clone();
    let mut labels = LabelFactory::of(&work);
    let Traversal { source, body, .. } = work.blocks[header.0].terminator.traversal()?;
    let mut region = Region {
        header,
        body: work.label_to_block[&body],
        blocks: loop_
            .natural
            .blocks()
            .filter(|block| *block != header)
            .collect(),
    };
    if region.holds_no_instruction(&work) {
        return None;
    }
    region.pass_body_args(&mut work);
    region.name_fallthroughs(&mut work)?;
    if let Some(latch) = region.one_latch(&mut work, &mut labels) {
        region.drop_trivial_params(&mut work, latch);
    }
    let carried = CarriedValues {
        header: work.blocks[header.0].params.clone(),
        supplied: work.blocks[region.body.0].params.clone(),
    };

    let shape = Shape::of(&work, &region)?;
    let loans = Loans::build(&work);
    let components = Components::of(&work, &region, &shape, &loans, &carried).parts(&work, &region);
    if components.is_empty() {
        return None;
    }
    let parts: Vec<Classified> = components
        .into_iter()
        .map(|component| Classified {
            kind: part_kind(&work, &loans, &state, source, &component),
            component,
        })
        .collect();
    if let [
        Classified {
            kind: PartKind::Sequential,
            ..
        },
    ] = parts[..]
    {
        return None;
    }
    Chain::write(&mut work, &mut labels, &region, &shape, &carried, parts)?;
    Some(work)
}

struct LabelFactory {
    next: u32,
}

impl LabelFactory {
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

/// The loop's body: every block of the natural loop but the header.
struct Region {
    header: BlockIdx,
    body: BlockIdx,
    blocks: Vec<BlockIdx>,
}

struct CarriedValues {
    header: Vec<ValueId>,
    supplied: Vec<ValueId>,
}

impl Region {
    fn contains(&self, block: BlockIdx) -> bool {
        self.blocks.contains(&block)
    }

    fn holds_no_instruction(&self, cfg: &CfgBody) -> bool {
        self.blocks
            .iter()
            .all(|block| cfg.blocks[block.0].insts.is_empty())
    }

    fn substitute(&self, cfg: &mut CfgBody, subst: &FxHashMap<ValueId, ValueId>) {
        if subst.is_empty() {
            return;
        }
        for &block in &self.blocks {
            let block = &mut cfg.blocks[block.0];
            for inst in &mut block.insts {
                apply_subst(&mut inst.kind, subst);
            }
            apply_subst_terminator(&mut block.terminator, subst);
        }
    }

    /// A body parameter a `For` fills from `body_args` is that argument.
    fn pass_body_args(&self, cfg: &mut CfgBody) {
        let mut traversal = cfg.blocks[self.header.0]
            .terminator
            .traversal_mut()
            .expect("`partition` read the header as a traversal");
        let supplied = traversal.source.supplied_params();
        let args = traversal.body_args.values();
        traversal.body_args.clear();
        let params = cfg.blocks[self.body.0].params.split_off(supplied);
        let subst: FxHashMap<ValueId, ValueId> = params.into_iter().zip(args).collect();
        self.substitute(cfg, &subst);
    }

    fn name_fallthroughs(&self, cfg: &mut CfgBody) -> Option<()> {
        for &block in &self.blocks {
            if let Terminator::Fallthrough = cfg.blocks[block.0].terminator {
                let label = cfg.blocks.get(block.0 + 1)?.label;
                cfg.blocks[block.0].terminator = Terminator::Jump {
                    label,
                    args: Vec::new(),
                };
            }
        }
        Some(())
    }

    /// Every edge back to the header goes through one block that jumps
    /// there, unless a single `Jump` is already the only such edge: the
    /// block written, if one is.
    fn one_latch(&mut self, cfg: &mut CfgBody, labels: &mut LabelFactory) -> Option<BlockIdx> {
        let header_label = cfg.blocks[self.header.0].label;
        let back_edges: usize = self
            .blocks
            .iter()
            .map(|&block| edges_into(&mut cfg.blocks[block.0].terminator, header_label).len())
            .sum();
        let one_jump = back_edges == 1
            && self.blocks.iter().any(|&block| {
                matches!(&cfg.blocks[block.0].terminator,
                    Terminator::Jump { label, .. } if *label == header_label)
            });
        if one_jump {
            return None;
        }
        let params: Vec<ValueId> = cfg.blocks[self.header.0]
            .params
            .clone()
            .into_iter()
            .map(|param| fresh_like(cfg, param))
            .collect();
        let latch = labels.fresh();
        for &block in &self.blocks {
            for edge in edges_into(&mut cfg.blocks[block.0].terminator, header_label) {
                *edge.label = latch;
            }
        }
        let at = BlockIdx(cfg.blocks.len());
        cfg.label_to_block.insert(latch, at);
        self.blocks.push(at);
        cfg.blocks.push(Block {
            label: latch,
            params: params.clone(),
            insts: Vec::new(),
            terminator: Terminator::Jump {
                label: header_label,
                args: params,
            },
        });
        Some(at)
    }

    /// A parameter of the latch `one_latch` wrote that every back edge
    /// hands one value is that value. The latch takes every carried value,
    /// every part's, and one it is handed unchanged would otherwise tie the
    /// parts at a block none of them owns alone.
    fn drop_trivial_params(&self, cfg: &mut CfgBody, latch: BlockIdx) {
        let label = cfg.blocks[latch.0].label;
        loop {
            let Some(incoming) = self.plain_incoming(cfg, latch) else {
                return;
            };
            let params = &cfg.blocks[latch.0].params;
            let trivial = params.iter().enumerate().find_map(|(at, &param)| {
                let mut sent = incoming.iter().map(|args| args[at]).filter(|v| *v != param);
                let first = sent.next()?;
                sent.all(|value| value == first).then_some(TrivialParam {
                    at,
                    param,
                    sent: first,
                })
            });
            let Some(TrivialParam { at, param, sent }) = trivial else {
                return;
            };
            for &block in &self.blocks {
                for edge in edges_into(&mut cfg.blocks[block.0].terminator, label) {
                    edge.args.remove(at);
                }
            }
            cfg.blocks[latch.0].params.remove(at);
            self.substitute(cfg, &FxHashMap::from_iter([(param, sent)]));
        }
    }

    /// The argument lists of every edge into `target`, where each is an
    /// edge from inside the body that passes the target's whole parameter
    /// list; `None` where one is not.
    fn plain_incoming(&self, cfg: &mut CfgBody, target: BlockIdx) -> Option<Vec<Vec<ValueId>>> {
        let label = cfg.blocks[target.0].label;
        let preds = cfg.predecessors();
        let mut incoming = Vec::new();
        for &pred in preds.get(&target)? {
            if !self.contains(pred) {
                return None;
            }
            let term = &mut cfg.blocks[pred.0].terminator;
            if term.traversal().is_some() {
                return None;
            }
            incoming.extend(
                edges_into(term, label)
                    .into_iter()
                    .map(|edge| edge.args.clone()),
            );
        }
        Some(incoming)
    }
}

fn fresh_like(cfg: &mut CfgBody, like: ValueId) -> ValueId {
    let fresh = cfg.val_factory.next();
    let ty = cfg.val_types[&like].clone();
    cfg.val_types.insert(fresh, ty);
    if let Some(origin) = cfg.debug.get(like).cloned() {
        cfg.debug.set(fresh, origin);
    }
    fresh
}

struct TrivialParam {
    at: usize,
    param: ValueId,
    sent: ValueId,
}

struct EdgeInto<'a> {
    label: &'a mut Label,
    args: &'a mut Vec<ValueId>,
}

/// The edges of `term` to `label` that pass their target's whole parameter
/// list: a traversal's body edge is not one, nor its exit edge where that
/// edge defines the trip count (RFC-0057 rule 9).
fn edges_into(term: &mut Terminator, label: Label) -> Vec<EdgeInto<'_>> {
    fn push<'a>(
        edges: &mut Vec<EdgeInto<'a>>,
        wanted: Label,
        to: &'a mut Label,
        args: &'a mut Vec<ValueId>,
    ) {
        if *to == wanted {
            edges.push(EdgeInto { label: to, args });
        }
    }
    let mut edges = Vec::new();
    let mut push = |to, args| push(&mut edges, label, to, args);
    match term {
        Terminator::Jump { label, args } => push(label, args),
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
            push(then_label, then_args);
            push(else_label, else_args);
        }
        Terminator::Switch { arms, default, .. } => {
            for (_, label, args) in arms.iter_mut() {
                push(label, args);
            }
            if let Some((label, args)) = default {
                push(label, args);
            }
        }
        term @ (Terminator::For { .. } | Terminator::ForParts { .. }) => {
            let traversal = term.traversal_mut().expect("a `For` or a `ForParts`");
            if *traversal.exit_trip == ExitTrip::Absent {
                push(traversal.exit, traversal.exit_args);
            }
        }
        Terminator::Return { .. } | Terminator::Diverge | Terminator::Fallthrough => {}
    }
    edges
}

/// Where each block of the body stands, with the latch's edge to the header
/// as the one way out: the spine of blocks every iteration runs, the arms
/// each branch decides, and the order the body's blocks run in.
struct Shape {
    spine: Vec<BlockIdx>,
    latch_args: Vec<ValueId>,
    deciders_of: FxHashMap<BlockIdx, Vec<BlockIdx>>,
    arms_of: FxHashMap<BlockIdx, Vec<BlockIdx>>,
    reverse_post_order: FxHashMap<BlockIdx, usize>,
    loops: Vec<SpineLoop>,
}

/// A loop inside the body, by spine index: its header, and the spine
/// block whose branch leaves it. A `while` whose condition is a
/// short-circuit branches twice, and its header is the first spine block
/// the second branch's arms reach again. Every block from the header to
/// that branch, and the branch's arms, run as the loop and belong to one
/// part.
struct SpineLoop {
    header: usize,
    decided_at: usize,
}

/// The loops on `spine`: each branch whose arms reach a spine block at or
/// before it, from the first such block, with loops that overlap merged.
fn loops_on(spine: &[BlockIdx], arms_of: &FxHashMap<BlockIdx, Vec<BlockIdx>>) -> Vec<SpineLoop> {
    let mut loops: Vec<SpineLoop> = Vec::new();
    for (at, block) in spine.iter().enumerate() {
        let Some(arms) = arms_of.get(block) else {
            continue;
        };
        let Some(header) = spine[..=at].iter().position(|held| arms.contains(held)) else {
            continue;
        };
        loops.retain(|inner| inner.header < header);
        match loops.last_mut() {
            Some(outer) if outer.decided_at >= header => outer.decided_at = at,
            _ => loops.push(SpineLoop {
                header,
                decided_at: at,
            }),
        }
    }
    loops
}

impl Shape {
    fn of(cfg: &CfgBody, region: &Region) -> Option<Shape> {
        let header_label = cfg.blocks[region.header.0].label;
        let mut succs: FxHashMap<BlockIdx, Vec<BlockIdx>> = FxHashMap::default();
        let mut latch: Option<(BlockIdx, Vec<ValueId>)> = None;
        for &block in &region.blocks {
            let mut inside: Vec<BlockIdx> = Vec::new();
            for succ in cfg.successors(block) {
                if region.contains(succ) && !inside.contains(&succ) {
                    inside.push(succ);
                }
            }
            match &cfg.blocks[block.0].terminator {
                Terminator::Jump { label, args } if *label == header_label => {
                    if latch.is_some() {
                        return None;
                    }
                    latch = Some((block, args.clone()));
                }
                _ if inside.is_empty() => return None,
                _ => {}
            }
            succs.insert(block, inside);
        }
        let (latch, latch_args) = latch?;
        let ipdom = post_dominators(region, &succs, latch);
        let mut spine = vec![region.body];
        loop {
            let last = *spine.last()?;
            let Some(next) = *ipdom.get(&last)? else {
                break;
            };
            if spine.contains(&next) {
                return None;
            }
            spine.push(next);
        }
        if spine.last() != Some(&latch) {
            return None;
        }
        let mut deciders_of: FxHashMap<BlockIdx, Vec<BlockIdx>> = FxHashMap::default();
        let mut arms_of: FxHashMap<BlockIdx, Vec<BlockIdx>> = FxHashMap::default();
        for &branch in &region.blocks {
            let two_way = matches!(
                cfg.blocks[branch.0].terminator,
                Terminator::JumpIf { .. } | Terminator::Diamond { .. } | Terminator::Switch { .. }
            );
            if succs[&branch].len() < 2 && !two_way {
                continue;
            }
            let join = ipdom[&branch];
            let mut held: FxHashSet<BlockIdx> = FxHashSet::default();
            let mut work: Vec<BlockIdx> = succs[&branch].clone();
            while let Some(block) = work.pop() {
                if Some(block) == join || !held.insert(block) {
                    continue;
                }
                work.extend(succs[&block].iter().copied());
            }
            let mut held: Vec<BlockIdx> = held.into_iter().collect();
            held.sort();
            for &block in &held {
                deciders_of.entry(block).or_default().push(branch);
            }
            arms_of.insert(branch, held);
        }
        let placed = region
            .blocks
            .iter()
            .all(|block| spine.contains(block) || deciders_of.contains_key(block));
        if !placed {
            return None;
        }
        let mut post: Vec<BlockIdx> = Vec::new();
        post_order(region.body, &succs, &mut FxHashSet::default(), &mut post);
        let reverse_post_order = post
            .into_iter()
            .rev()
            .enumerate()
            .map(|(at, block)| (block, at))
            .collect();
        let loops = loops_on(&spine, &arms_of);
        Some(Shape {
            spine,
            latch_args,
            deciders_of,
            arms_of,
            reverse_post_order,
            loops,
        })
    }

    fn is_branch(&self, block: BlockIdx) -> bool {
        self.arms_of.contains_key(&block)
    }

    /// The loop whose header is spine block `at`, by spine index.
    fn loop_from(&self, at: usize) -> Option<&SpineLoop> {
        self.loops.iter().find(|spine_loop| spine_loop.header == at)
    }

    /// Whether spine block `at` lies in a loop it does not head.
    fn inside_a_loop(&self, at: usize) -> bool {
        self.loops
            .iter()
            .any(|spine_loop| spine_loop.header < at && at <= spine_loop.decided_at)
    }

    /// The branches whose arms hold `block`; none for a block of the spine.
    fn deciders(&self, block: BlockIdx) -> &[BlockIdx] {
        self.deciders_of.get(&block).map_or(&[], Vec::as_slice)
    }
}

fn post_order(
    block: BlockIdx,
    succs: &FxHashMap<BlockIdx, Vec<BlockIdx>>,
    visited: &mut FxHashSet<BlockIdx>,
    post: &mut Vec<BlockIdx>,
) {
    if !visited.insert(block) {
        return;
    }
    for &succ in &succs[&block] {
        post_order(succ, succs, visited, post);
    }
    post.push(block);
}

/// The immediate post-dominator of each block of the body, where `latch`,
/// which jumps to the header, is the only exit: `None` for the latch.
fn post_dominators(
    region: &Region,
    succs: &FxHashMap<BlockIdx, Vec<BlockIdx>>,
    latch: BlockIdx,
) -> FxHashMap<BlockIdx, Option<BlockIdx>> {
    let all: FxHashSet<BlockIdx> = region.blocks.iter().copied().collect();
    let mut sets: FxHashMap<BlockIdx, FxHashSet<BlockIdx>> = region
        .blocks
        .iter()
        .map(|&block| match block == latch {
            true => (block, FxHashSet::from_iter([block])),
            false => (block, all.clone()),
        })
        .collect();
    let mut changed = true;
    while changed {
        changed = false;
        for &block in &region.blocks {
            if block == latch {
                continue;
            }
            let mut next: FxHashSet<BlockIdx> = all.clone();
            for succ in &succs[&block] {
                next.retain(|held| sets[succ].contains(held));
            }
            next.insert(block);
            if next != sets[&block] {
                sets.insert(block, next);
                changed = true;
            }
        }
    }
    region
        .blocks
        .iter()
        .map(|&block| {
            let strict: Vec<BlockIdx> = sets[&block]
                .iter()
                .copied()
                .filter(|other| *other != block)
                .collect();
            let immediate = strict
                .iter()
                .copied()
                .find(|candidate| sets[candidate].len() == strict.len());
            (block, immediate)
        })
        .collect()
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct InstAt {
    block: BlockIdx,
    at: usize,
}

/// What a part holds: an instruction, or the terminator of a branch.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum Member {
    Inst(InstAt),
    Branch(BlockIdx),
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum Node {
    Member(Member),
    Value(ValueId),
}

impl From<Member> for Node {
    fn from(member: Member) -> Node {
        Node::Member(member)
    }
}

fn inst_node(block: BlockIdx, at: usize) -> Node {
    Node::Member(Member::Inst(InstAt { block, at }))
}

fn branch_node(block: BlockIdx) -> Node {
    Node::Member(Member::Branch(block))
}

struct UnionFind {
    parent: FxHashMap<Node, Node>,
}

impl UnionFind {
    fn find(&mut self, node: Node) -> Node {
        let parent = *self.parent.entry(node).or_insert(node);
        if parent == node {
            return node;
        }
        let root = self.find(parent);
        self.parent.insert(node, root);
        root
    }

    fn union(&mut self, a: Node, b: Node) {
        let (a, b) = (self.find(a), self.find(b));
        if a != b {
            self.parent.insert(a, b);
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum Touch {
    Storage(ValueId),
    Context(QualifiedRef),
}

#[derive(Default)]
struct Touchers {
    nodes: Vec<Node>,
    written: bool,
}

struct IncomingEdge {
    from: BlockIdx,
    args: Vec<ValueId>,
    first: usize,
}

/// RFC-0089 rule 5's relation over the body, closed.
struct Components {
    sets: UnionFind,
    carried: Vec<ValueId>,
}

impl Components {
    fn of(
        cfg: &CfgBody,
        region: &Region,
        shape: &Shape,
        loans: &Loans<'_>,
        carried: &CarriedValues,
    ) -> Self {
        let mut sets = UnionFind {
            parent: FxHashMap::default(),
        };
        let inside = values_inside(cfg, region, carried);
        let node = |value: ValueId| inside.contains(&value).then_some(Node::Value(value));
        let mut touched: FxHashMap<Touch, Touchers> = FxHashMap::default();

        for &block in &region.blocks {
            for (at, inst) in cfg.blocks[block.0].insts.iter().enumerate() {
                let me = inst_node(block, at);
                sets.find(me);
                let values = inst_info::uses(&inst.kind)
                    .into_iter()
                    .chain(inst_info::defs(&inst.kind));
                for value in values.filter_map(node) {
                    sets.union(me, value);
                }
                for &decider in shape.deciders(block) {
                    sets.union(me, branch_node(decider));
                }
                let effect = loans.storage_effect(&inst.kind);
                let mut touch = |on: Touch, writes: bool| {
                    let touchers = touched.entry(on).or_default();
                    touchers.nodes.push(me);
                    touchers.written |= writes;
                };
                for storage in &effect.reads {
                    touch(Touch::Storage(*storage), false);
                }
                for storage in &effect.writes {
                    touch(Touch::Storage(*storage), true);
                }
                match &inst.kind {
                    InstKind::Commit { context, .. } => touch(Touch::Context(*context), true),
                    InstKind::Fetch { context, .. } => touch(Touch::Context(*context), false),
                    _ => {}
                }
            }
            if shape.is_branch(block) {
                let me = branch_node(block);
                sets.find(me);
                for &decider in shape.deciders(block) {
                    sets.union(me, branch_node(decider));
                }
                let term = &cfg.blocks[block.0].terminator;
                for value in branch_reads(cfg, term).into_iter().filter_map(node) {
                    sets.union(me, value);
                }
            }
        }

        for touchers in touched.values().filter(|touchers| touchers.written) {
            for pair in touchers.nodes.windows(2) {
                sets.union(pair[0], pair[1]);
            }
        }

        for &target in &region.blocks {
            let params = &cfg.blocks[target.0].params;
            let edges = incoming(cfg, region, cfg.blocks[target.0].label);
            for (at, &param) in params.iter().enumerate() {
                let Some(param_node) = node(param) else {
                    continue;
                };
                let sent: Vec<Option<ValueId>> = edges
                    .iter()
                    .map(|edge| at.checked_sub(edge.first).map(|i| edge.args[i]))
                    .collect();
                for arg in sent.iter().flatten().copied().filter_map(node) {
                    sets.union(param_node, arg);
                }
                let Some(first) = sent.first() else {
                    continue;
                };
                let one_value = first.is_some() && sent.iter().all(|value| value == first);
                if one_value {
                    continue;
                }
                for edge in &edges {
                    for &decider in shape.deciders(edge.from) {
                        sets.union(param_node, branch_node(decider));
                    }
                    if shape.is_branch(edge.from) {
                        sets.union(param_node, branch_node(edge.from));
                    }
                }
            }
        }

        for (&param, &arg) in carried.header.iter().zip(&shape.latch_args) {
            if let Some(arg) = node(arg) {
                sets.union(Node::Value(param), arg);
            }
        }
        Self {
            sets,
            carried: carried.header.clone(),
        }
    }

    /// The components, each with its instructions and branches and the
    /// indices of its carried values, in no particular order.
    fn parts(mut self, cfg: &CfgBody, region: &Region) -> Vec<Component> {
        let mut by_root: FxHashMap<Node, Component> = FxHashMap::default();
        let mut members: Vec<Member> = Vec::new();
        for &block in &region.blocks {
            let insts = cfg.blocks[block.0].insts.len();
            members.extend((0..insts).map(|at| Member::Inst(InstAt { block, at })));
            if self.sets.parent.contains_key(&branch_node(block)) {
                members.push(Member::Branch(block));
            }
        }
        for member in members {
            let root = self.sets.find(member.into());
            by_root.entry(root).or_default().members.push(member);
        }
        for (index, &param) in self.carried.iter().enumerate() {
            let root = self.sets.find(Node::Value(param));
            by_root.entry(root).or_default().carried.push(index);
        }
        for &block in region.blocks.iter().filter(|block| **block != region.body) {
            for &param in &cfg.blocks[block.0].params {
                let root = self.sets.find(Node::Value(param));
                if let Some(component) = by_root.get_mut(&root) {
                    component.params.push(param);
                }
            }
        }
        by_root.into_values().collect()
    }
}

#[derive(Default)]
struct Component {
    members: Vec<Member>,
    /// Indices into the header's parameters.
    carried: Vec<usize>,
    /// The parameters of the body's blocks, the body block's aside, that the
    /// component's instructions or carried values reach.
    params: Vec<ValueId>,
}

struct Classified {
    component: Component,
    kind: PartKind,
}

/// The values the relation joins: every block parameter and instruction
/// result inside the body, what a traversal inside it fills, and the
/// header's parameters, which the parts read as their carried values
/// (RFC-0089 rule 1). The element and the counter are every part's and are
/// left out, and so is a storage slot, which the relation reaches through
/// what touches it.
fn values_inside(cfg: &CfgBody, region: &Region, carried: &CarriedValues) -> FxHashSet<ValueId> {
    let mut slots: FxHashSet<ValueId> = FxHashSet::default();
    for inst in cfg.blocks.iter().flat_map(|block| &block.insts) {
        if let InstKind::Ref { target, .. }
        | InstKind::Take { target, .. }
        | InstKind::Assign { target, .. } = &inst.kind
            && let Some(slot) = inst_info::storage(target)
        {
            slots.insert(slot);
        }
    }
    let mut inside: FxHashSet<ValueId> = FxHashSet::default();
    for &block in &region.blocks {
        let block = &cfg.blocks[block.0];
        inside.extend(block.params.iter().copied());
        for inst in &block.insts {
            inside.extend(inst_info::defs(&inst.kind));
        }
    }
    inside.extend(carried.header.iter().copied());
    for supplied in &carried.supplied {
        inside.remove(supplied);
    }
    inside.retain(|value| !slots.contains(value));
    inside
}

/// What a branch reads besides its edges' arguments.
fn branch_reads(cfg: &CfgBody, term: &Terminator) -> Vec<ValueId> {
    match term {
        Terminator::JumpIf { cond, .. } | Terminator::Diamond { cond, .. } => vec![*cond],
        Terminator::Switch { tag, .. } => vec![*tag],
        term @ (Terminator::For { .. } | Terminator::ForParts { .. }) => {
            let traversal = term.traversal().expect("a `For` or a `ForParts`");
            let body = &cfg.blocks[cfg.label_to_block[&traversal.body].0].params;
            let exit = &cfg.blocks[cfg.label_to_block[&traversal.exit].0].params;
            traversal
                .source
                .uses()
                .into_iter()
                .chain(body[..traversal.source.supplied_params()].iter().copied())
                .chain(traversal.exit_trip.trip_param(exit))
                .collect()
        }
        Terminator::Jump { .. }
        | Terminator::Return { .. }
        | Terminator::Diverge
        | Terminator::Fallthrough => Vec::new(),
    }
}

fn incoming(cfg: &CfgBody, region: &Region, label: Label) -> Vec<IncomingEdge> {
    let mut edges = Vec::new();
    for &from in &region.blocks {
        let mut push = |to: Label, args: &[ValueId], first: usize| {
            if to == label {
                edges.push(IncomingEdge {
                    from,
                    args: args.to_vec(),
                    first,
                });
            }
        };
        match &cfg.blocks[from.0].terminator {
            Terminator::Jump { label, args } => push(*label, args, 0),
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
                push(*then_label, then_args, 0);
                push(*else_label, else_args, 0);
            }
            Terminator::Switch { arms, default, .. } => {
                for (_, label, args) in arms {
                    push(*label, args, 0);
                }
                if let Some((label, args)) = default {
                    push(*label, args, 0);
                }
            }
            term @ (Terminator::For { .. } | Terminator::ForParts { .. }) => {
                let traversal = term.traversal().expect("a `For` or a `ForParts`");
                let body_first = traversal.source.supplied_params();
                push(traversal.body, &traversal.body_args, body_first);
                let exit_first = traversal.exit_trip.supplied_params();
                push(traversal.exit, traversal.exit_args, exit_first);
            }
            Terminator::Return { .. } | Terminator::Diverge | Terminator::Fallthrough => {}
        }
    }
    edges
}

/// RFC-0089 rule 3, with rule 7's conditions on a `Law` part: every carried
/// value is an accumulator, every order-carrying instruction is one rule 6
/// excuses, and every storage written is the `SliceMut` source's or one the
/// part folds into.
fn part_kind(
    cfg: &CfgBody,
    loans: &Loans<'_>,
    state: &CarriedState,
    source: ForSource,
    part: &Component,
) -> PartKind {
    let mut accs: Vec<Accumulator> = Vec::new();
    for &index in &part.carried {
        match state.params[index].carried {
            Carried::Merge { op, exact } => accs.push(accumulator(op, exact)),
            Carried::Iv | Carried::Recurrence => return PartKind::Sequential,
        }
    }
    let lent = lent_mutably_by_source(source, loans);
    let mut folds: Vec<StorageMerge> = Vec::new();
    for member in &part.members {
        let Member::Inst(InstAt { block, at }) = *member else {
            continue;
        };
        let kind = &cfg.blocks[block.0].insts[at].kind;
        let excused = inst_info::defs(kind)
            .iter()
            .any(|dst| state.unordered.contains(dst));
        if (carries_order(kind) && !excused) || matches!(kind, InstKind::Commit { .. }) {
            return PartKind::Sequential;
        }
        for storage in loans.storage_effect(kind).writes {
            if lent.contains(&storage) {
                continue;
            }
            let Some(fold) = state.storage_merges.iter().find(|m| m.storage == storage) else {
                return PartKind::Sequential;
            };
            if !folds.contains(fold) {
                folds.push(*fold);
            }
        }
    }
    accs.extend(folds.into_iter().map(|fold| Accumulator {
        law: Law::Fold(FoldAccumulator {
            storage: fold.storage,
            callee: fold.callee,
            instance: fold.instance,
            fold: fold.fold,
        }),
        exact: true,
        commutative: fold.fold.commutative,
    }));
    PartKind::Law(accs)
}

fn accumulator(op: MergeOp, exact: bool) -> Accumulator {
    match op {
        MergeOp::Add => Accumulator {
            law: Law::Op(LawOp::Add),
            exact,
            commutative: true,
        },
        MergeOp::Mul => Accumulator {
            law: Law::Op(LawOp::Mul),
            exact,
            commutative: true,
        },
        MergeOp::Extern(merge) => Accumulator {
            law: Law::Call(CallLaw {
                callee: merge.callee,
                instance: merge.instance,
                identity: merge.identity,
            }),
            exact,
            commutative: merge.commutative,
        },
        MergeOp::Order => Accumulator {
            law: Law::Order,
            exact,
            commutative: true,
        },
    }
}

fn lent_mutably_by_source(source: ForSource, loans: &Loans<'_>) -> Vec<ValueId> {
    let ForSource::SliceMut(slice) = source else {
        return Vec::new();
    };
    loans
        .names(slice)
        .iter()
        .filter(|loan| loan.mutability == Mutability::Mut)
        .filter_map(|loan| loan.storage.slot())
        .collect()
}

/// Rule 1's chain written over the body.
struct Chain<'a> {
    shape: &'a Shape,
    owner: FxHashMap<Member, usize>,
    param_owner: FxHashMap<ValueId, usize>,
    /// By spine index, the label each spine block goes by in the part that
    /// owns it: its own label.
    ///
    /// A spine block is owned by the part whose branch it joins, whose loop
    /// it heads, or whose values its parameters take, and otherwise by the
    /// first part that runs an instruction in it. Every other part that runs
    /// one there gets a block of its own, which is the split rule 1 asks
    /// for where one block held two parts' instructions.
    label_owner: Vec<Option<usize>>,
    /// By spine index, the label a spine block goes by as the join of the
    /// branch before it: its own, or a fresh one where it also heads a loop
    /// another part owns, since one block cannot open two parts.
    join_labels: Vec<Label>,
}

/// Where a part first stands in the body: parts with an instruction first,
/// by the first one in reverse post-order, then by their first carried
/// value.
#[derive(PartialEq, Eq, PartialOrd, Ord)]
struct PartOrder {
    no_instruction: bool,
    first: Option<Position>,
    first_carried: Option<usize>,
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct Position {
    block_order: usize,
    in_block: InBlock,
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum InBlock {
    Inst(usize),
    Terminator,
}

/// A part's blocks as written, its entry, and its last block, which the
/// caller closes with the jump onward.
struct Written {
    entry: Label,
    blocks: Vec<Block>,
    last: Open,
}

impl Chain<'_> {
    fn write(
        cfg: &mut CfgBody,
        labels: &mut LabelFactory,
        region: &Region,
        shape: &Shape,
        carried: &CarriedValues,
        mut parts: Vec<Classified>,
    ) -> Option<()> {
        let position = |member: &Member| match *member {
            Member::Inst(InstAt { block, at }) => Position {
                block_order: shape.reverse_post_order[&block],
                in_block: InBlock::Inst(at),
            },
            Member::Branch(block) => Position {
                block_order: shape.reverse_post_order[&block],
                in_block: InBlock::Terminator,
            },
        };
        for part in &mut parts {
            part.component.carried.sort();
        }
        parts.sort_by_key(|part| {
            let first = part.component.members.iter().map(position).min();
            PartOrder {
                no_instruction: first.is_none(),
                first,
                first_carried: part.component.carried.first().copied(),
            }
        });
        let owner: FxHashMap<Member, usize> = parts
            .iter()
            .enumerate()
            .flat_map(|(index, part)| {
                part.component
                    .members
                    .iter()
                    .map(move |member| (*member, index))
            })
            .collect();
        let param_owner: FxHashMap<ValueId, usize> = parts
            .iter()
            .enumerate()
            .flat_map(|(index, part)| {
                part.component
                    .params
                    .iter()
                    .map(move |param| (*param, index))
            })
            .collect();
        let mut chain = Chain {
            shape,
            owner,
            param_owner,
            label_owner: Vec::new(),
            join_labels: Vec::new(),
        };
        chain.label_owner = chain.label_owners(cfg)?;
        chain.join_labels = chain.join_labels(cfg, labels)?;

        let header_label = cfg.blocks[region.header.0].label;

        let mut written: Vec<Written> = Vec::with_capacity(parts.len());
        for index in 0..parts.len() {
            let first = (index == 0).then(|| Open {
                label: cfg.blocks[region.body.0].label,
                params: carried.supplied.clone(),
                insts: Vec::new(),
            });
            written.push(chain.part(cfg, labels, index, first)?);
        }
        let entries: Vec<Label> = written.iter().map(|part| part.entry).collect();
        let mut blocks_written: Vec<Block> = Vec::new();
        for (index, part) in written.into_iter().enumerate() {
            let onward = match entries.get(index + 1) {
                Some(&next) => Terminator::Jump {
                    label: next,
                    args: Vec::new(),
                },
                None => Terminator::Jump {
                    label: header_label,
                    args: shape.latch_args.clone(),
                },
            };
            blocks_written.extend(part.blocks);
            blocks_written.push(part.last.close(onward));
        }

        let parts: Vec<Part> = parts
            .into_iter()
            .zip(&entries)
            .map(|(part, &entry)| Part {
                entry,
                carried: part
                    .component
                    .carried
                    .iter()
                    .map(|&index| carried.header[index])
                    .collect(),
                kind: part.kind,
            })
            .collect();
        let first = region.blocks.iter().min().copied()?;
        let mut blocks: Vec<Block> = Vec::with_capacity(cfg.blocks.len() + blocks_written.len());
        for (at, block) in std::mem::take(&mut cfg.blocks).into_iter().enumerate() {
            if BlockIdx(at) == first {
                blocks.append(&mut blocks_written);
            }
            if !region.contains(BlockIdx(at)) {
                blocks.push(block);
            }
        }
        cfg.blocks = blocks;
        cfg.label_to_block = cfg
            .blocks
            .iter()
            .enumerate()
            .map(|(at, block)| (block.label, BlockIdx(at)))
            .collect();
        let header = cfg.label_to_block[&header_label];
        let Traversal {
            source,
            body,
            exit,
            exit_trip,
            exit_args,
            ..
        } = cfg.blocks[header.0].terminator.traversal()?;
        let exit_args = exit_args.to_vec();
        cfg.blocks[header.0].terminator = Terminator::ForParts {
            source,
            body,
            parts,
            exit,
            exit_trip,
            exit_args,
        };
        Some(())
    }

    fn owner_of(&self, member: Member) -> Option<usize> {
        self.owner.get(&member).copied()
    }

    /// The part a spine block's parameters belong to, where they belong to
    /// one; `None` for a block without parameters.
    fn params_owner(&self, cfg: &CfgBody, block: BlockIdx) -> Option<Option<usize>> {
        let params = &cfg.blocks[block.0].params;
        let mut owners = params
            .iter()
            .map(|param| self.param_owner.get(param).copied());
        let Some(first) = owners.next() else {
            return Some(None);
        };
        let first = first?;
        owners
            .all(|owner| owner == Some(first))
            .then_some(Some(first))
    }

    fn label_owners(&self, cfg: &CfgBody) -> Option<Vec<Option<usize>>> {
        let spine = &self.shape.spine;
        let mut owners = vec![Some(0)];
        for i in 1..spine.len() {
            let (before, at) = (spine[i - 1], spine[i]);
            let joined = self
                .shape
                .is_branch(before)
                .then(|| self.owner_of(Member::Branch(before)))
                .flatten();
            let headed = self
                .shape
                .loop_from(i)
                .and_then(|spine_loop| self.owner_of(Member::Branch(spine[spine_loop.decided_at])));
            let params = self.params_owner(cfg, at)?;
            let first_to_run = (0..cfg.blocks[at.0].insts.len())
                .filter_map(|inst| {
                    self.owner_of(Member::Inst(InstAt {
                        block: at,
                        at: inst,
                    }))
                })
                .min();
            owners.push(joined.or(headed).or(params).or(first_to_run));
        }
        Some(owners)
    }

    fn join_labels(&self, cfg: &CfgBody, labels: &mut LabelFactory) -> Option<Vec<Label>> {
        let spine = &self.shape.spine;
        let mut join_labels = vec![cfg.blocks[spine[0].0].label];
        for i in 1..spine.len() {
            let (before, at) = (spine[i - 1], spine[i]);
            let loop_owner = self
                .shape
                .loop_from(i)
                .map(|spine_loop| self.owner_of(Member::Branch(spine[spine_loop.decided_at])));
            let apart = self.shape.is_branch(before)
                && loop_owner.is_some_and(|owner| owner != self.owner_of(Member::Branch(before)));
            let label = match apart {
                true if !cfg.blocks[at.0].params.is_empty() => return None,
                true => labels.fresh(),
                false => cfg.blocks[at.0].label,
            };
            join_labels.push(label);
        }
        Some(join_labels)
    }

    /// Part `index`'s blocks, with the body's block boundaries where the
    /// part runs: a block per spine block that holds the part's
    /// instructions, a join or a loop head, or parameters the part takes,
    /// and each of its branches with their arms whole. `first` is the
    /// block it begins in, the body block for the first part; a later part
    /// begins at its first block, or at a block of its own where that one
    /// takes parameters or the part runs nothing.
    fn part(
        &self,
        cfg: &CfgBody,
        labels: &mut LabelFactory,
        index: usize,
        first: Option<Open>,
    ) -> Option<Written> {
        let spine = &self.shape.spine;
        let mine = |member: Member| self.owner_of(member) == Some(index);
        let mut writer = PartWriter {
            blocks: Vec::new(),
            entry: first.as_ref().map(|open| open.label),
            open: first,
            labels,
        };
        for (i, &at) in spine.iter().enumerate() {
            if self.shape.inside_a_loop(i) {
                continue;
            }
            let block = &cfg.blocks[at.0];
            let owns_label = self.label_owner[i] == Some(index);
            let runs_here = (0..block.insts.len()).any(|inst| {
                mine(Member::Inst(InstAt {
                    block: at,
                    at: inst,
                }))
            });
            let spine_loop = self.shape.loop_from(i);
            let heads = spine_loop
                .is_some_and(|spine_loop| mine(Member::Branch(spine[spine_loop.decided_at])));
            let mut joined_at_block = false;
            if i == 0 && !owns_label && runs_here {
                writer.begin_own();
            }
            if i > 0 {
                let before = spine[i - 1];
                let joins = self.shape.is_branch(before) && mine(Member::Branch(before));
                let join_label = self.join_labels[i];
                joined_at_block = joins && join_label == block.label;
                let entering = || match &cfg.blocks[before.0].terminator {
                    Terminator::Jump { label, args } if *label == block.label => Some(args.clone()),
                    _ if block.params.is_empty() => Some(Vec::new()),
                    _ => None,
                };
                if joins {
                    if writer.open.is_some() {
                        return None;
                    }
                    writer.open = Some(Open {
                        label: join_label,
                        params: match joined_at_block {
                            true => block.params.clone(),
                            false => Vec::new(),
                        },
                        insts: Vec::new(),
                    });
                }
                let enters_block = (heads && !joined_at_block)
                    || (spine_loop.is_none()
                        && !joins
                        && owns_label
                        && (runs_here || !block.params.is_empty()));
                if enters_block {
                    writer.begin(block.label, block.params.clone(), entering()?);
                } else if spine_loop.is_none() && !joins && !owns_label && runs_here {
                    writer.begin_own();
                }
            }
            if let Some(spine_loop) = spine_loop {
                if !heads {
                    continue;
                }
                let opened = writer.open.take()?;
                if !joined_at_block && !opened.insts.is_empty() {
                    return None;
                }
                let decided_at = spine_loop.decided_at;
                let unit: FxHashSet<BlockIdx> = spine[i..=decided_at]
                    .iter()
                    .chain(&self.shape.arms_of[&spine[decided_at]])
                    .copied()
                    .collect();
                let mut unit: Vec<BlockIdx> = unit.into_iter().collect();
                unit.sort();
                let from = cfg.blocks[spine[decided_at + 1].0].label;
                let to = self.join_labels[decided_at + 1];
                let header_first = unit.iter().position(|held| *held == at)?;
                unit[..=header_first].rotate_right(1);
                for held in unit {
                    let mut original = cfg.blocks[held.0].clone();
                    retarget_edges(&mut original.terminator, from, to);
                    writer.blocks.push(original);
                }
                continue;
            }
            for (inst_at, inst) in block.insts.iter().enumerate() {
                if mine(Member::Inst(InstAt {
                    block: at,
                    at: inst_at,
                })) {
                    writer.open.as_mut()?.insts.push(inst.clone());
                }
            }
            let decides =
                i + 1 < spine.len() && self.shape.is_branch(at) && mine(Member::Branch(at));
            if !decides {
                continue;
            }
            let from = cfg.blocks[spine[i + 1].0].label;
            let to = self.join_labels[i + 1];
            let mut term = block.terminator.clone();
            retarget_edges(&mut term, from, to);
            let deciding = writer.open.take()?;
            writer.blocks.push(deciding.close(term));
            for &arm in &self.shape.arms_of[&at] {
                let mut arm_block = cfg.blocks[arm.0].clone();
                retarget_edges(&mut arm_block.terminator, from, to);
                writer.blocks.push(arm_block);
            }
        }
        writer.finish()
    }
}

/// A part's blocks while they are written: those closed, the one open, and
/// the part's entry once it has one.
struct PartWriter<'l> {
    blocks: Vec<Block>,
    entry: Option<Label>,
    open: Option<Open>,
    labels: &'l mut LabelFactory,
}

impl PartWriter<'_> {
    /// Opens `label`, closing the open block with a jump to it. A part that
    /// has no block yet begins at `label`, or, where `label` takes
    /// parameters, at a block of its own that jumps there.
    fn begin(&mut self, label: Label, params: Vec<ValueId>, args: Vec<ValueId>) {
        let jump = Terminator::Jump { label, args };
        match (self.open.take(), self.entry) {
            (Some(before), _) => self.blocks.push(before.close(jump)),
            (None, None) if params.is_empty() => self.entry = Some(label),
            (None, None) => {
                let own = self.labels.fresh();
                self.entry = Some(own);
                self.blocks.push(Open::empty(own).close(jump));
            }
            (None, Some(_)) => {}
        }
        self.open = Some(Open {
            label,
            params,
            insts: Vec::new(),
        });
    }

    fn begin_own(&mut self) {
        let own = self.labels.fresh();
        self.begin(own, Vec::new(), Vec::new());
    }

    fn finish(mut self) -> Option<Written> {
        if self.open.is_none() && self.entry.is_none() {
            self.begin_own();
        }
        Some(Written {
            entry: self.entry?,
            blocks: self.blocks,
            last: self.open?,
        })
    }
}

struct Open {
    label: Label,
    params: Vec<ValueId>,
    insts: Vec<Inst>,
}

impl Open {
    fn empty(label: Label) -> Self {
        Open {
            label,
            params: Vec::new(),
            insts: Vec::new(),
        }
    }

    fn close(self, terminator: Terminator) -> Block {
        Block {
            label: self.label,
            params: self.params,
            insts: self.insts,
            terminator,
        }
    }
}

/// Sends every edge of `term` to `from`, and a `Diamond`'s join at `from`,
/// to `to`.
fn retarget_edges(term: &mut Terminator, from: Label, to: Label) {
    if from == to {
        return;
    }
    let retarget = |label: &mut Label| {
        if *label == from {
            *label = to;
        }
    };
    match term {
        Terminator::Jump { label, .. } => retarget(label),
        Terminator::JumpIf {
            then_label,
            else_label,
            ..
        } => {
            retarget(then_label);
            retarget(else_label);
        }
        Terminator::Diamond {
            then_label,
            else_label,
            join,
            ..
        } => {
            retarget(then_label);
            retarget(else_label);
            retarget(join);
        }
        Terminator::Switch { arms, default, .. } => {
            for (_, label, _) in arms.iter_mut() {
                retarget(label);
            }
            if let Some((label, _)) = default {
                retarget(label);
            }
        }
        term @ (Terminator::For { .. } | Terminator::ForParts { .. }) => {
            let traversal = term.traversal_mut().expect("a `For` or a `ForParts`");
            retarget(traversal.exit);
        }
        Terminator::Return { .. } | Terminator::Diverge | Terminator::Fallthrough => {}
    }
}
