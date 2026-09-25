//! Where a `for`'s body is cut (RFC-0089 rule 6). The pass reads each
//! loop's dependence cycles from `analysis::loop_deps`, keeps each cycle
//! whole in one stage, puts the work a cycle reads before it and the work
//! it does not read after it, and names no token. A branch stays whole in
//! one stage with its arms, and so does a loop inside the body. A loop the
//! body leaves is cut around its exits (RFC-0089 rule 5, `ExitSides`). A
//! loop the pass cannot cut is one stage: every loop left from a loop nested
//! in its body, and every loop whose header holds an instruction.
//!
//! Run again over a loop already cut, the pass cuts nothing new: it removes
//! each boundary between two free stages and each boundary before a stage
//! that holds no instruction, so a pass that removed a `merge` or a phi
//! leaves no boundary behind that nothing orders.

use std::collections::BTreeSet;

use rustc_hash::{FxHashMap, FxHashSet};

use crate::analysis::domtree::DomTree;
use crate::analysis::inst_info;
use crate::analysis::loans::Loans;
use crate::analysis::loop_deps::{self, Head, LoopDeps, RunAhead, Token};
use crate::analysis::loops::{Invariants, Loop, LoopNest};
use crate::analysis::targets::{TargetSlots, Written, effect, slots_lent_mutably, touched_slots};
use crate::cfg::{Block, BlockIdx, CfgBody, ENTRY_LABEL, Terminator};
use crate::ir::{ExitTrip, Inst, InstKind, Label, Stages, ValueId};
use crate::laws::LawTable;
use crate::optimize::ssa_pass::{apply_subst, apply_subst_terminator};

pub fn run(cfg: &mut CfgBody, laws: &LawTable) {
    let mut examined: FxHashSet<Label> = FxHashSet::default();
    while let Some(header) = innermost_unexamined_for(cfg, &examined) {
        examined.insert(header);
        let at = cfg.label_to_block[&header];
        let Some((_, stages)) = Head::of(&cfg.blocks[at.0].terminator) else {
            continue;
        };
        match stages.len() {
            1 => cut(cfg, laws, header),
            _ => merge_boundaries(cfg, laws, at),
        }
    }
}

fn innermost_unexamined_for(cfg: &CfgBody, examined: &FxHashSet<Label>) -> Option<Label> {
    let domtree = DomTree::build(cfg);
    let invariants = Invariants::of(cfg);
    let nest = LoopNest::of(cfg, &domtree, &invariants);
    nest.iter()
        .map(|(_, loop_)| &cfg.blocks[loop_.natural.header.0])
        .filter(|header| Head::of(&header.terminator).is_some())
        .map(|header| header.label)
        .find(|label| !examined.contains(label))
}

/// Removes each boundary before a stage that holds no instruction, or that
/// is free and follows a free stage. A chain whose shape fails is left for
/// `validate::stages` to refuse.
fn merge_boundaries(cfg: &mut CfgBody, laws: &LawTable, header: BlockIdx) {
    let Ok(deps) = LoopDeps::of(cfg, laws, header) else {
        return;
    };
    let stages = deps.membership.stages();
    let holds_nothing = |stage: usize| {
        stages[stage]
            .blocks
            .iter()
            .all(|block| cfg.blocks[block.0].insts.is_empty())
    };
    let mut named = deps.membership.body_stages();
    let Some(first) = named.next() else {
        return;
    };
    let mut kept: Vec<Label> = Vec::new();
    let mut last_kept_free = deps.is_free(first);
    for stage in named {
        let free = deps.is_free(stage);
        if holds_nothing(stage) || (free && last_kept_free) {
            continue;
        }
        kept.push(stages[stage].entry);
        last_kept_free = free;
    }
    let body = stages[first].entry;
    let (Terminator::For { stages, .. } | Terminator::While { stages, .. }) =
        &mut cfg.blocks[header.0].terminator
    else {
        panic!("block {} heads the loop `loop_deps` read", header.0)
    };
    *stages = Stages::new(body, kept);
}

struct Facts {
    header: BlockIdx,
    slots: TargetSlots,
}

fn cut(cfg: &mut CfgBody, laws: &LawTable, header_label: Label) {
    let Some(facts) = Facts::of(cfg, header_label) else {
        return;
    };
    if let Some(written) = plan(cfg, laws, &facts) {
        *cfg = written;
    }
}

impl Facts {
    fn of(cfg: &CfgBody, header_label: Label) -> Option<Facts> {
        let header = cfg.label_to_block[&header_label];
        let (head, _) = Head::of(&cfg.blocks[header.0].terminator)?;
        let domtree = DomTree::build(cfg);
        let invariants = Invariants::of(cfg);
        let nest = LoopNest::of(cfg, &domtree, &invariants);
        let loop_ = nest.get(nest.by_header(header)?);
        let loans = Loans::build(cfg);
        let loop_blocks: Vec<BlockIdx> = loop_.natural.blocks().collect();
        let slots = TargetSlots::of(&loans, head.source(), &loop_blocks);
        Some(Facts { header, slots })
    }
}

struct Planned {
    members: Vec<Member>,
    params: Vec<ValueId>,
    free: bool,
}

/// The body rewritten as the chain the cycles cut it into, or `None` where
/// the loop stays one stage.
fn plan(cfg: &CfgBody, laws: &LawTable, facts: &Facts) -> Option<CfgBody> {
    let header = facts.header;
    let (head, _) = Head::of(&cfg.blocks[header.0].terminator)?;
    // A pull loop's header holds its pull, which is the first stage and
    // no member of the body this cuts (RFC-0089 rule 1).
    if head != Head::Pull && !cfg.blocks[header.0].insts.is_empty() {
        return None;
    }
    let domtree = DomTree::build(cfg);
    let invariants = Invariants::of(cfg);
    let nest = LoopNest::of(cfg, &domtree, &invariants);
    let loop_ = nest.get(nest.by_header(header)?);
    if leaves_from_a_nested_loop(cfg, &nest, loop_) {
        return None;
    }

    let mut work = cfg.clone();
    let mut labels = LabelFactory::of(&work);
    let (_, stages) = Head::of(&work.blocks[header.0].terminator)?;
    let body = work.label_to_block[&stages.body()];
    let mut region = Region {
        header,
        body,
        blocks: loop_
            .natural
            .blocks()
            .filter(|block| *block != header)
            .collect(),
    };
    region.take_exit_arms(&work)?;
    if region.holds_no_instruction(&work) {
        return None;
    }
    region.name_fallthroughs(&mut work)?;
    if let Some(latch) = region.one_latch(&mut work, &mut labels) {
        region.drop_trivial_params(&mut work, latch);
    }
    let shape = Shape::of(&work, &region)?;
    let stages = {
        let loans = Loans::build(&work);
        let units = Units::of(&work, &region, &shape, &loans, facts)?;
        let mut graph = Graph::of(&work, &region, &shape, &units, facts)?;
        let deps = LoopDeps::of(&work, laws, region.header).ok()?;
        let joins = graph.joins(&units, &deps);
        let run_ahead = RunAhead::of(&work, laws, region.header);
        let joins = ExitSides::settle(&work, &mut graph, &units, joins, &run_ahead);
        graph.order(&units, &joins)?
    };
    if stages.len() == 1 {
        return None;
    }
    Chain::write(&mut work, &mut labels, &region, &shape, stages)?;
    Some(work)
}

type UnitId = usize;

/// What no stage boundary may cut: a branch with its arms, a loop inside
/// the body, and a join block's parameters with the branch that sends them.
/// A storage the body defines and drops within one iteration is no token
/// (RFC-0089 rule 2): it orders the members that touch it within the
/// iteration, and a boundary may fall between them.
#[derive(Default)]
struct Unit {
    members: Vec<Member>,
    params: Vec<ValueId>,
    position: Position,
    uses: Vec<ValueId>,
    touched: Vec<ValueId>,
    written: Vec<ValueId>,
}

struct Units {
    units: Vec<Unit>,
    defined_by: FxHashMap<ValueId, UnitId>,
    by_member: FxHashMap<Member, UnitId>,
    /// Per block the loop leaves from, the unit that decides the leaving.
    leaving: FxHashMap<BlockIdx, UnitId>,
    /// Per storage the body defines and drops within one iteration, the
    /// members that touch it and whether each writes it.
    local: Vec<Vec<LocalTouch>>,
}

#[derive(Clone, Copy)]
struct LocalTouch {
    member: Member,
    writes: bool,
}

struct Alias {
    param: ValueId,
    value: ValueId,
}

impl Units {
    fn of(
        cfg: &CfgBody,
        region: &Region,
        shape: &Shape,
        loans: &Loans<'_>,
        facts: &Facts,
    ) -> Option<Units> {
        let slot_values = slot_values(cfg);
        let mut sets = UnionFind {
            parent: FxHashMap::default(),
        };
        let mut local: FxHashMap<ValueId, Vec<LocalTouch>> = FxHashMap::default();
        let mut inst_defs: FxHashMap<ValueId, Node> = FxHashMap::default();
        for &block in &region.blocks {
            for (at, inst) in cfg.blocks[block.0].insts.iter().enumerate() {
                let me = inst_node(block, at);
                sets.find(me);
                for &decider in shape.deciders(block) {
                    sets.union(me, branch_node(decider));
                }
                for def in inst_info::defs(&inst.kind) {
                    if !slot_values.contains(&def) {
                        inst_defs.insert(def, me);
                    }
                }
                let writes = written_slots(loans, &inst.kind);
                for slot in touched(loans, &inst.kind) {
                    if facts.slots.target_of(slot).is_some() {
                        continue;
                    }
                    local.entry(slot).or_default().push(LocalTouch {
                        member: Member::Inst(InstAt { block, at }),
                        writes: writes.contains(&slot),
                    });
                }
            }
            if shape.is_branch(block) {
                let me = branch_node(block);
                sets.find(me);
                for &decider in shape.deciders(block) {
                    sets.union(me, branch_node(decider));
                }
                let reads = branch_reads(cfg, &cfg.blocks[block.0].terminator);
                for loan in reads.iter().flat_map(|read| loans.holds(*read)) {
                    let Some(slot) = loan.storage.slot() else {
                        continue;
                    };
                    if facts.slots.target_of(slot).is_none() {
                        local.entry(slot).or_default().push(LocalTouch {
                            member: Member::Branch(block),
                            writes: false,
                        });
                    }
                }
            }
        }

        let mut param_args: FxHashMap<ValueId, Vec<ValueId>> = FxHashMap::default();
        let mut aliases: Vec<Alias> = Vec::new();
        for &target in region.blocks.iter().filter(|block| **block != region.body) {
            let edges = incoming(cfg, region, cfg.blocks[target.0].label);
            for (at, &param) in cfg.blocks[target.0].params.iter().enumerate() {
                let sent: Vec<Option<ValueId>> = edges
                    .iter()
                    .map(|edge| at.checked_sub(edge.first).map(|i| edge.args[i]))
                    .collect();
                param_args.insert(param, sent.iter().flatten().copied().collect());
                let one_value = match sent.first() {
                    Some(Some(first)) if sent.iter().all(|value| *value == Some(*first)) => {
                        Some(*first)
                    }
                    _ => None,
                };
                if let Some(value) = one_value
                    && let Some(&definer) = inst_defs.get(&value)
                {
                    aliases.push(Alias { param, value });
                    sets.union(Node::Value(param), definer);
                    continue;
                }
                let mut linked = false;
                for edge in &edges {
                    let senders = shape
                        .deciders(edge.from)
                        .iter()
                        .copied()
                        .chain(shape.is_branch(edge.from).then_some(edge.from));
                    for sender in senders {
                        sets.union(Node::Value(param), branch_node(sender));
                        linked = true;
                    }
                }
                if !linked {
                    return None;
                }
            }
        }

        let mut ids: FxHashMap<Node, UnitId> = FxHashMap::default();
        let mut built = Units {
            units: Vec::new(),
            defined_by: FxHashMap::default(),
            by_member: FxHashMap::default(),
            leaving: FxHashMap::default(),
            local: local
                .into_values()
                .filter(|touches| touches.iter().any(|touch| touch.writes))
                .collect(),
        };
        let mut unit_of = |sets: &mut UnionFind, units: &mut Vec<Unit>, node: Node| {
            let root = sets.find(node);
            *ids.entry(root).or_insert_with(|| {
                units.push(Unit {
                    position: Position::LAST,
                    ..Unit::default()
                });
                units.len() - 1
            })
        };
        for &block in &region.blocks {
            for (at, inst) in cfg.blocks[block.0].insts.iter().enumerate() {
                let id = unit_of(&mut sets, &mut built.units, inst_node(block, at));
                let member = Member::Inst(InstAt { block, at });
                let unit = &mut built.units[id];
                unit.members.push(member);
                unit.position = unit.position.min(Position::of(shape, member));
                unit.uses.extend(
                    inst_info::uses(&inst.kind)
                        .into_iter()
                        .filter(|value| !slot_values.contains(value)),
                );
                unit.touched.extend(touched(loans, &inst.kind));
                unit.written.extend(written_slots(loans, &inst.kind));
                built.by_member.insert(member, id);
                for def in inst_info::defs(&inst.kind) {
                    if !slot_values.contains(&def) {
                        built.defined_by.insert(def, id);
                    }
                }
            }
            if shape.is_branch(block) {
                let id = unit_of(&mut sets, &mut built.units, branch_node(block));
                let member = Member::Branch(block);
                built.by_member.insert(member, id);
                let unit = &mut built.units[id];
                unit.members.push(member);
                unit.position = unit.position.min(Position::of(shape, member));
                let term = &cfg.blocks[block.0].terminator;
                let reads = branch_reads(cfg, term);
                // A branch that reads through a reference touches the storage
                // the reference lends, as an instruction reading through it
                // does: the lend lasts until the branch.
                unit.touched.extend(
                    reads
                        .iter()
                        .flat_map(|read| loans.holds(*read))
                        .filter_map(|loan| loan.storage.slot()),
                );
                unit.uses.extend(reads);
                for supplied in values_a_traversal_fills(cfg, term) {
                    built.defined_by.insert(supplied, id);
                }
            }
            if let Some(uses) = leaving_uses(cfg, region, block) {
                let decider = match shape.is_branch(block) {
                    true => block,
                    false => *shape.deciders(block).first()?,
                };
                let id = unit_of(&mut sets, &mut built.units, branch_node(decider));
                built.units[id].uses.extend(uses);
                built.leaving.insert(block, id);
            }
        }
        for &block in region.blocks.iter().filter(|block| **block != region.body) {
            for &param in &cfg.blocks[block.0].params {
                let root = sets.find(Node::Value(param));
                let &id = ids.get(&root)?;
                let unit = &mut built.units[id];
                unit.params.push(param);
                unit.uses
                    .extend(param_args.get(&param).into_iter().flatten().copied());
                built.defined_by.insert(param, id);
            }
        }
        for Alias { param, value } in aliases {
            let id = built.defined_by[&param];
            built.units[id].uses.push(value);
        }
        Some(built)
    }

    fn len(&self) -> usize {
        self.units.len()
    }

    /// The unit a member `analysis::loop_deps` names belongs to; a
    /// terminator that is no branch of the body, such as the jump between
    /// two blocks, belongs to none.
    fn of_member(&self, member: loop_deps::Member) -> Option<UnitId> {
        let member = match member {
            loop_deps::Member::Inst(at) => Member::Inst(InstAt {
                block: at.block,
                at: at.at,
            }),
            loop_deps::Member::Term(block) => match self.leaving.get(&block) {
                Some(unit) if !self.by_member.contains_key(&Member::Branch(block)) => {
                    return Some(*unit);
                }
                _ => Member::Branch(block),
            },
        };
        self.by_member.get(&member).copied()
    }

    fn with(&self, holds: impl Fn(&Unit) -> bool) -> Vec<UnitId> {
        (0..self.len())
            .filter(|id| holds(&self.units[*id]))
            .collect()
    }
}

/// What the edges by which `block` leaves the loop read, where it leaves:
/// a `return`'s value and order, and the arguments of each edge to a block
/// outside the loop.
fn leaving_uses(cfg: &CfgBody, region: &Region, block: BlockIdx) -> Option<Vec<ValueId>> {
    let term = &cfg.blocks[block.0].terminator;
    if let Terminator::Return { .. } = term {
        return Some(inst_info::terminator_uses(term).into_vec());
    }
    let header = cfg.blocks[region.header.0].label;
    let leaves = |label: &Label| {
        *label != header
            && cfg
                .label_to_block
                .get(label)
                .is_some_and(|target| !region.contains(*target))
    };
    let edges: Vec<(&Label, &Vec<ValueId>)> = match term {
        Terminator::Jump { label, args } => vec![(label, args)],
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
        } => vec![(then_label, then_args), (else_label, else_args)],
        Terminator::While {
            exit, exit_args, ..
        } => vec![(exit, exit_args)],
        Terminator::Switch { arms, default, .. } => arms
            .iter()
            .map(|(_, label, args)| (label, args))
            .chain(default.iter().map(|(label, args)| (label, args)))
            .collect(),
        Terminator::For { .. }
        | Terminator::Return { .. }
        | Terminator::Diverge
        | Terminator::Fallthrough => Vec::new(),
    };
    let leaving: Vec<&Vec<ValueId>> = edges
        .into_iter()
        .filter(|(label, _)| leaves(label))
        .map(|(_, args)| args)
        .collect();
    (!leaving.is_empty()).then(|| leaving.into_iter().flatten().copied().collect())
}

/// An exit from a loop nested in the body leaves two loops at once; RFC-0089
/// rule 5 reads an exit of the loop being cut, and such a loop stays one
/// stage.
fn leaves_from_a_nested_loop(cfg: &CfgBody, nest: &LoopNest, loop_: &Loop) -> bool {
    let header = loop_.natural.header;
    nest.iter()
        .map(|(_, inner)| &inner.natural)
        .filter(|inner| inner.header != header && loop_.natural.contains(inner.header))
        .flat_map(|inner| inner.blocks())
        .any(|block| {
            matches!(cfg.blocks[block.0].terminator, Terminator::Return { .. })
                || cfg
                    .successors(block)
                    .iter()
                    .any(|succ| !loop_.natural.contains(*succ))
        })
}

/// The slots an instruction touches: those `touched_slots` names, and each
/// one a reference it reads lends. A read through a reference is a touch of
/// what it lends, as a branch's read is, since the loan lasts until that
/// read: `string_concat [&base, ..]` reads `base`, and a `take base` the
/// source wrote after it stays after it.
fn touched(loans: &Loans<'_>, kind: &InstKind) -> Vec<ValueId> {
    let mut slots = touched_slots(loans, kind);
    for used in inst_info::uses(kind) {
        for slot in loans.holds(used).filter_map(|loan| loan.storage.slot()) {
            if !slots.contains(&slot) {
                slots.push(slot);
            }
        }
    }
    slots
}

fn written_slots(loans: &Loans<'_>, kind: &InstKind) -> Vec<ValueId> {
    effect(loans, kind)
        .writes
        .into_iter()
        .chain(slots_lent_mutably(loans, kind))
        .collect()
}

fn values_a_traversal_fills(cfg: &CfgBody, term: &Terminator) -> Vec<ValueId> {
    let Terminator::For {
        stages,
        exit,
        exit_trip,
        ..
    } = term
    else {
        return Vec::new();
    };
    let body_params = &cfg.blocks[cfg.label_to_block[&stages.body()].0].params;
    let exit_params = &cfg.blocks[cfg.label_to_block[exit].0].params;
    body_params
        .iter()
        .copied()
        .chain(exit_trip.trip_param(exit_params))
        .collect()
}

/// Every value some instruction names as a storage slot: the relation
/// reaches a slot through what touches it, not as a value.
fn slot_values(cfg: &CfgBody) -> FxHashSet<ValueId> {
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
    slots
}

/// Which units run before which within one iteration: one reads a value
/// the other defines, or both touch the element and one of them writes it.
struct Graph {
    succs: Vec<BTreeSet<UnitId>>,
    preds: Vec<BTreeSet<UnitId>>,
    readers_of_param: FxHashMap<ValueId, Vec<UnitId>>,
    element: Vec<ValueId>,
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct ElementTouch {
    position: Position,
    unit: UnitId,
    writes: bool,
}

impl Graph {
    fn of(
        cfg: &CfgBody,
        region: &Region,
        shape: &Shape,
        units: &Units,
        facts: &Facts,
    ) -> Option<Graph> {
        let n = units.len();
        let mut graph = Graph {
            succs: vec![BTreeSet::new(); n],
            preds: vec![BTreeSet::new(); n],
            readers_of_param: FxHashMap::default(),
            element: Vec::new(),
        };
        for (id, unit) in units.units.iter().enumerate() {
            for used in &unit.uses {
                if let Some(&definer) = units.defined_by.get(used)
                    && definer != id
                {
                    graph.edge(definer, id);
                }
            }
        }
        let header_params = &cfg.blocks[region.header.0].params;
        if shape.latch_args.len() != header_params.len() {
            return None;
        }
        for &param in header_params {
            graph
                .readers_of_param
                .insert(param, units.with(|unit| unit.uses.contains(&param)));
        }
        for touches in &units.local {
            let mut ordered: Vec<(Position, LocalTouch)> = touches
                .iter()
                .map(|touch| (Position::of(shape, touch.member), *touch))
                .collect();
            ordered.sort_by_key(|(position, _)| *position);
            for (i, (_, before)) in ordered.iter().enumerate() {
                for (_, after) in &ordered[i + 1..] {
                    let (from, to) = (
                        units.by_member[&before.member],
                        units.by_member[&after.member],
                    );
                    if (before.writes || after.writes) && from != to {
                        graph.edge(from, to);
                    }
                }
            }
        }
        graph.element = units
            .units
            .iter()
            .flat_map(|unit| &unit.touched)
            .copied()
            .filter(|slot| facts.slots.target_of(*slot) == Some(Written::Element))
            .collect();
        let mut on_element: Vec<ElementTouch> = units
            .units
            .iter()
            .enumerate()
            .filter(|(_, unit)| unit.touched.iter().any(|slot| graph.element.contains(slot)))
            .map(|(id, unit)| ElementTouch {
                position: unit.position,
                unit: id,
                writes: unit.written.iter().any(|slot| graph.element.contains(slot)),
            })
            .collect();
        on_element.sort();
        for (i, before) in on_element.iter().enumerate() {
            for after in &on_element[i + 1..] {
                if before.writes || after.writes {
                    graph.edge(before.unit, after.unit);
                }
            }
        }
        Some(graph)
    }

    fn edge(&mut self, from: UnitId, to: UnitId) {
        self.succs[from].insert(to);
        self.preds[to].insert(from);
    }

    fn reach(from: &[UnitId], next: &[BTreeSet<UnitId>]) -> FxHashSet<UnitId> {
        let mut seen: FxHashSet<UnitId> = FxHashSet::default();
        let mut work: Vec<UnitId> = from.to_vec();
        while let Some(unit) = work.pop() {
            if seen.insert(unit) {
                work.extend(next[unit].iter().copied());
            }
        }
        seen
    }

    /// RFC-0089 rule 3's dependence cycle: every unit on a path from one of
    /// `starts` to one of `ends`, both included.
    fn cycle(&self, starts: &[UnitId], ends: &[UnitId]) -> FxHashSet<UnitId> {
        let forward = Self::reach(starts, &self.succs);
        let backward = Self::reach(ends, &self.preds);
        forward.intersection(&backward).copied().collect()
    }

    /// Each dependence cycle `analysis::loop_deps` finds, as the units that
    /// hold its members. Joins that share a unit are one join, and a join
    /// holds every unit between two of its own, and every unit between one
    /// that reads its state and one of its own.
    fn joins(&self, units: &Units, deps: &LoopDeps) -> Vec<Join> {
        let mut joins: Vec<Join> = deps
            .unjoined
            .iter()
            .map(|cycle| Join {
                tokens: cycle.tokens.clone(),
                units: cycle
                    .members
                    .iter()
                    .filter_map(|member| units.of_member(*member))
                    .collect(),
            })
            .filter(|join| !join.units.is_empty())
            .collect();
        loop {
            let mut changed = false;
            if let Some(Overlap { keep, absorb }) = first_overlap(&joins) {
                let absorbed = joins.remove(absorb);
                joins[keep].tokens.extend(absorbed.tokens);
                joins[keep].units.extend(absorbed.units);
                changed = true;
            }
            for join in &mut joins {
                let held: Vec<UnitId> = join.units.iter().copied().collect();
                let mut starts = held.clone();
                starts.extend(self.after_state(join));
                for unit in self.cycle(&starts, &held) {
                    changed |= join.units.insert(unit);
                }
            }
            if !changed {
                return joins;
            }
        }
    }

    /// The units that read a header parameter's state outside the join of
    /// its cycle: they run after it (RFC-0089 rule 6).
    fn after_state(&self, join: &Join) -> Vec<UnitId> {
        join.tokens
            .iter()
            .filter_map(|token| token.header_param())
            .flat_map(|param| self.readers_of_param.get(&param).into_iter().flatten())
            .copied()
            .filter(|unit| !join.units.contains(unit))
            .collect()
    }

    /// The stages in the order they run: a node whose inputs are all ready
    /// runs first where the body wrote it first, and units in no cycle next to
    /// each other are one free stage.
    fn order(&self, units: &Units, joins: &[Join]) -> Option<Vec<Planned>> {
        let mut node_of: Vec<Option<usize>> = vec![None; units.len()];
        let mut nodes: Vec<OrderNode> = Vec::new();
        for (index, join) in joins.iter().enumerate() {
            for unit in &join.units {
                node_of[*unit] = Some(nodes.len());
            }
            nodes.push(OrderNode {
                join: Some(index),
                units: join.units.iter().copied().collect(),
                position: join
                    .units
                    .iter()
                    .map(|unit| units.units[*unit].position)
                    .min()
                    .unwrap_or(Position::LAST),
            });
        }
        for (id, unit) in units.units.iter().enumerate() {
            if node_of[id].is_some() {
                continue;
            }
            node_of[id] = Some(nodes.len());
            nodes.push(OrderNode {
                join: None,
                units: vec![id],
                position: unit.position,
            });
        }
        let node_of: Vec<usize> = node_of.into_iter().collect::<Option<_>>()?;
        let mut succs: Vec<FxHashSet<usize>> = vec![FxHashSet::default(); nodes.len()];
        for (unit, after) in self.succs.iter().enumerate() {
            for &next in after {
                if node_of[unit] != node_of[next] {
                    succs[node_of[unit]].insert(node_of[next]);
                }
            }
        }
        for (at, node) in nodes.iter().enumerate() {
            let Some(index) = node.join else {
                continue;
            };
            for unit in self.after_state(&joins[index]) {
                if node_of[unit] != at {
                    succs[at].insert(node_of[unit]);
                }
            }
        }
        let mut waiting: Vec<usize> = vec![0; nodes.len()];
        for next in succs.iter().flatten() {
            waiting[*next] += 1;
        }
        let mut ready: BTreeSet<Ready> = (0..nodes.len())
            .filter(|at| waiting[*at] == 0)
            .map(|at| Ready {
                position: nodes[at].position,
                node: at,
            })
            .collect();
        let mut ordered: Vec<usize> = Vec::with_capacity(nodes.len());
        while let Some(Ready { node: at, .. }) = ready.pop_first() {
            ordered.push(at);
            for &next in &succs[at] {
                waiting[next] -= 1;
                if waiting[next] == 0 {
                    ready.insert(Ready {
                        position: nodes[next].position,
                        node: next,
                    });
                }
            }
        }
        if ordered.len() != nodes.len() {
            return None;
        }

        let mut stages: Vec<Planned> = Vec::new();
        for at in ordered {
            let node = &nodes[at];
            let members = node
                .units
                .iter()
                .flat_map(|unit| units.units[*unit].members.iter().copied());
            let params = node
                .units
                .iter()
                .flat_map(|unit| units.units[*unit].params.iter().copied());
            match (node.join, stages.last_mut()) {
                (
                    None,
                    Some(Planned {
                        free: true,
                        members: free_members,
                        params: free_params,
                    }),
                ) => {
                    free_members.extend(members);
                    free_params.extend(params);
                }
                (join, _) => stages.push(Planned {
                    members: members.collect(),
                    params: params.collect(),
                    free: join.is_none(),
                }),
            }
        }
        Some(stages)
    }
}

/// Where a unit runs against the exits of a loop the body leaves (RFC-0089
/// rules 5 and 6), in the order the chain runs the three.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Side {
    /// In a free stage before the exiting one: run ahead of the control
    /// token, and discarded in an iteration past the exit.
    Ahead,
    Exiting,
    After,
}

/// The side of each unit of a body the loop leaves from.
///
/// Only a unit in no join goes ahead, and only one rule 5 lets run ahead: a
/// cycle waits for the control token. A unit the body runs before the first
/// exit goes after the exits only when the exiting iteration may skip it,
/// which holds of an instruction that neither traps, nor has an effect, nor
/// may fail to finish; a unit the body runs after the last exit goes after
/// them. Every other unit, and every unit between two exits, runs in the
/// exiting stage, which runs in order and in the body's order, so moving a
/// unit there is always sound. A join lies on one side whole, and a unit
/// that must run after another is moved to the exiting stage wherever the
/// sides would put it first.
struct ExitSides;

impl ExitSides {
    /// `joins` with every unit of the exiting side in the control token's
    /// join, and `graph` with the edges that keep each unit on its side.
    fn settle(
        cfg: &CfgBody,
        graph: &mut Graph,
        units: &Units,
        mut joins: Vec<Join>,
        run_ahead: &RunAhead<'_>,
    ) -> Vec<Join> {
        let Some(control) = joins
            .iter()
            .position(|join| join.tokens.contains(&Token::Control))
        else {
            return joins;
        };
        let position = |unit: UnitId| units.units[unit].position;
        let Some(anchor) = joins[control].units.iter().copied().min_by_key(|unit| position(*unit))
        else {
            return joins;
        };
        let first = position(anchor);
        let last = joins[control]
            .units
            .iter()
            .map(|unit| position(*unit))
            .max()
            .unwrap_or(first);
        let join_of: Vec<Option<usize>> = (0..units.len())
            .map(|unit| joins.iter().position(|join| join.units.contains(&unit)))
            .collect();
        let runs_ahead = |unit: UnitId| {
            units.units[unit]
                .members
                .iter()
                .all(|member| run_ahead.held_back(member.in_loop_deps()).is_none())
        };
        let skippable = |unit: UnitId| {
            units.units[unit]
                .members
                .iter()
                .all(|member| member.skippable(cfg))
        };
        let mut sides: Vec<Side> = (0..units.len())
            .map(|unit| match join_of[unit] {
                Some(join) if join == control => Side::Exiting,
                _ if position(unit) > last => Side::After,
                _ if position(unit) > first => Side::Exiting,
                None if runs_ahead(unit) => Side::Ahead,
                Some(_) if skippable(unit) => Side::After,
                None | Some(_) => Side::Exiting,
            })
            .collect();

        let mut precedes: Vec<(UnitId, UnitId)> = graph
            .succs
            .iter()
            .enumerate()
            .flat_map(|(from, to)| to.iter().map(move |to| (from, *to)))
            .collect();
        for join in &joins {
            for reader in graph.after_state(join) {
                precedes.extend(join.units.iter().map(|unit| (*unit, reader)));
            }
        }
        loop {
            let mut changed = false;
            for join in &joins {
                let Some(side) = join.units.iter().map(|unit| sides[*unit]).min() else {
                    continue;
                };
                for unit in &join.units {
                    changed |= std::mem::replace(&mut sides[*unit], side) != side;
                }
            }
            for &(before, after) in &precedes {
                if sides[before] <= sides[after] {
                    continue;
                }
                let moved = match sides[before] {
                    Side::After => before,
                    Side::Ahead | Side::Exiting => after,
                };
                sides[moved] = Side::Exiting;
                changed = true;
            }
            if !changed {
                break;
            }
        }

        let mut kept: Vec<Join> = Vec::with_capacity(joins.len());
        let mut exiting = joins.remove(control);
        for join in joins {
            match join.units.iter().all(|unit| sides[*unit] == Side::Exiting) {
                true => {
                    exiting.tokens.extend(join.tokens);
                    exiting.units.extend(join.units);
                }
                false => kept.push(join),
            }
        }
        for unit in 0..units.len() {
            match sides[unit] {
                Side::Ahead => graph.edge(unit, anchor),
                Side::Exiting => {
                    exiting.units.insert(unit);
                }
                Side::After => graph.edge(anchor, unit),
            }
        }
        kept.push(exiting);
        kept
    }
}

struct Overlap {
    keep: usize,
    absorb: usize,
}

fn first_overlap(joins: &[Join]) -> Option<Overlap> {
    (0..joins.len()).find_map(|keep| {
        (keep + 1..joins.len())
            .find(|absorb| !joins[keep].units.is_disjoint(&joins[*absorb].units))
            .map(|absorb| Overlap { keep, absorb })
    })
}

struct OrderNode {
    join: Option<usize>,
    units: Vec<UnitId>,
    position: Position,
}

#[derive(PartialEq, Eq, PartialOrd, Ord)]
struct Ready {
    position: Position,
    node: usize,
}

struct Join {
    tokens: Vec<Token>,
    units: FxHashSet<UnitId>,
}

impl Position {
    const LAST: Position = Position {
        block_order: usize::MAX,
        in_block: InBlock::Terminator,
    };

    fn of(shape: &Shape, member: Member) -> Position {
        let block_order = match member {
            Member::Inst(InstAt { block, .. }) | Member::Branch(block) => {
                shape.reverse_post_order[&block]
            }
        };
        let in_block = match member {
            Member::Inst(InstAt { at, .. }) => InBlock::Inst(at),
            Member::Branch(_) => InBlock::Terminator,
        };
        Position {
            block_order,
            in_block,
        }
    }
}

impl Default for Position {
    fn default() -> Self {
        Self::LAST
    }
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

impl Region {
    fn contains(&self, block: BlockIdx) -> bool {
        self.blocks.contains(&block)
    }

    /// A block outside the natural loop that only the body enters is an arm
    /// of the branch that enters it, and is written with that branch: left
    /// where it was, it would follow the chain's back edge, where the
    /// machine's region reads the exit block. Such arms are taken while each
    /// returns or jumps on out of the loop, and `None` where one does neither:
    /// what it reads of the body would order nothing.
    fn take_exit_arms(&mut self, cfg: &CfgBody) -> Option<()> {
        let preds = cfg.predecessors();
        let header = cfg.blocks[self.header.0].label;
        loop {
            let entered: Vec<BlockIdx> = (0..cfg.blocks.len())
                .map(BlockIdx)
                .filter(|block| *block != self.header && !self.contains(*block))
                .filter(|block| {
                    preds.get(block).is_some_and(|from| {
                        !from.is_empty() && from.iter().all(|at| self.contains(*at))
                    })
                })
                .collect();
            if entered.is_empty() {
                return Some(());
            }
            for &block in &entered {
                let leaves = match &cfg.blocks[block.0].terminator {
                    Terminator::Return { .. } => true,
                    Terminator::Jump { label, .. } => *label != header,
                    _ => false,
                };
                if !leaves {
                    return None;
                }
            }
            self.blocks.extend(entered);
        }
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
    /// every stage's, and one it is handed unchanged would otherwise tie the
    /// stages at a block none of them owns alone.
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
            if let Terminator::For { .. } = term {
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
        Terminator::For {
            exit,
            exit_trip,
            exit_args,
            ..
        } => {
            if *exit_trip == ExitTrip::Absent {
                push(exit, exit_args);
            }
        }
        Terminator::While {
            exit, exit_args, ..
        } => push(exit, exit_args),
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
/// stage.
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
                _ if inside.is_empty() && leaving_uses(cfg, region, block).is_none() => {
                    return None;
                }
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

/// What a stage holds: an instruction, or the terminator of a branch.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum Member {
    Inst(InstAt),
    Branch(BlockIdx),
}

impl Member {
    fn in_loop_deps(self) -> loop_deps::Member {
        match self {
            Member::Inst(InstAt { block, at }) => {
                loop_deps::Member::Inst(loop_deps::InstAt { block, at })
            }
            Member::Branch(block) => loop_deps::Member::Term(block),
        }
    }

    /// Whether an iteration that leaves before this member may skip it:
    /// it neither traps, nor has an effect, nor may fail to finish, so
    /// running it only where the iteration goes on is the same program.
    fn skippable(self, cfg: &CfgBody) -> bool {
        let Member::Inst(InstAt { block, at }) = self else {
            return false;
        };
        matches!(
            cfg.blocks[block.0].insts[at].kind,
            InstKind::Ref { .. }
                | InstKind::AsSlice { .. }
                | InstKind::Const { .. }
                | InstKind::ConstStr { .. }
        )
    }
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

/// What a branch reads besides its edges' arguments.
fn branch_reads(cfg: &CfgBody, term: &Terminator) -> Vec<ValueId> {
    match term {
        Terminator::JumpIf { cond, .. }
        | Terminator::Diamond { cond, .. }
        | Terminator::While { cond, .. } => vec![*cond],
        Terminator::Switch { tag, .. } => vec![*tag],
        Terminator::For {
            source,
            stages,
            exit,
            exit_trip,
            ..
        } => {
            let body = &cfg.blocks[cfg.label_to_block[&stages.body()].0].params;
            let exit = &cfg.blocks[cfg.label_to_block[exit].0].params;
            source
                .uses()
                .into_iter()
                .chain(body.iter().copied())
                .chain(exit_trip.trip_param(exit))
                .collect()
        }
        Terminator::Jump { .. }
        | Terminator::Return { .. }
        | Terminator::Diverge
        | Terminator::Fallthrough => Vec::new(),
    }
}

struct IncomingEdge {
    from: BlockIdx,
    args: Vec<ValueId>,
    first: usize,
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
            Terminator::For {
                source,
                stages,
                exit,
                exit_trip,
                exit_args,
            } => {
                push(stages.body(), &[], source.supplied_params());
                push(*exit, exit_args, exit_trip.supplied_params());
            }
            Terminator::While {
                stages,
                exit,
                exit_args,
                ..
            } => {
                push(stages.body(), &[], 0);
                push(*exit, exit_args, 0);
            }
            Terminator::Return { .. } | Terminator::Diverge | Terminator::Fallthrough => {}
        }
    }
    edges
}

/// RFC-0089 rule 1's chain written over the body.
struct Chain<'a> {
    shape: &'a Shape,
    owner: FxHashMap<Member, usize>,
    param_owner: FxHashMap<ValueId, usize>,
    /// By spine index, the label each spine block goes by in the stage that
    /// owns it: its own label.
    ///
    /// A spine block is owned by the stage whose branch it joins, whose loop
    /// it heads, or whose values its parameters take, and otherwise by the
    /// first stage that runs an instruction in it. Every other stage that runs
    /// one there gets a block of its own, which is the split rule 1 asks
    /// for where one block held two stages' instructions.
    label_owner: Vec<Option<usize>>,
    /// By spine index, the label a spine block goes by as the join of the
    /// branch before it: its own, or a fresh one where it also heads a loop
    /// another stage owns, since one block cannot open two stages.
    join_labels: Vec<Label>,
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

/// A stage's blocks as written, its entry, and its last block, which the
/// caller closes with the jump onward.
struct WrittenStage {
    entry: Label,
    blocks: Vec<Block>,
    last: Open,
    relabeled: Vec<Relabeled>,
}

/// A branch `cfg::demote_diamond` demoted, which now ends a block of its own
/// stage: `CfgBody::demoted_diamonds` names it by its block's label, and
/// `optimize::rejoin` restores it by that name.
struct Relabeled {
    from: Label,
    to: Label,
}

impl Chain<'_> {
    fn write(
        cfg: &mut CfgBody,
        labels: &mut LabelFactory,
        region: &Region,
        shape: &Shape,
        stages: Vec<Planned>,
    ) -> Option<()> {
        let owner: FxHashMap<Member, usize> = stages
            .iter()
            .enumerate()
            .flat_map(|(index, stage)| stage.members.iter().map(move |member| (*member, index)))
            .collect();
        let param_owner: FxHashMap<ValueId, usize> = stages
            .iter()
            .enumerate()
            .flat_map(|(index, stage)| stage.params.iter().map(move |param| (*param, index)))
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
        let supplied = cfg.blocks[region.body.0].params.clone();

        let mut written: Vec<WrittenStage> = Vec::with_capacity(stages.len());
        for index in 0..stages.len() {
            let first = (index == 0).then(|| Open {
                label: cfg.blocks[region.body.0].label,
                params: supplied.clone(),
                insts: Vec::new(),
            });
            written.push(chain.stage(cfg, labels, index, first)?);
        }
        let entries: Vec<Label> = written.iter().map(|stage| stage.entry).collect();
        let mut blocks_written: Vec<Block> = Vec::new();
        for (index, stage) in written.into_iter().enumerate() {
            for Relabeled { from, to } in stage.relabeled {
                cfg.demoted_diamonds.remove(&from);
                cfg.demoted_diamonds.insert(to);
            }
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
            blocks_written.extend(stage.blocks);
            blocks_written.push(stage.last.close(onward));
        }

        let stated = Stages::new(entries[0], entries[1..].to_vec());
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
        let (Terminator::For { stages, .. } | Terminator::While { stages, .. }) =
            &mut cfg.blocks[header.0].terminator
        else {
            return None;
        };
        *stages = stated;
        Some(())
    }

    fn owner_of(&self, member: Member) -> Option<usize> {
        self.owner.get(&member).copied()
    }

    /// The stage a spine block's parameters belong to, where they belong to
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

    /// Stage `index`'s blocks, with the body's block boundaries where the
    /// stage runs: a block per spine block that holds the stage's
    /// instructions, a join or a loop head, or parameters the stage takes,
    /// and each of its branches with their arms whole. `first` is the
    /// block it begins in, the body block for the first stage; a later stage
    /// begins at its first block, or at a block of its own where that one
    /// takes parameters or the stage runs nothing.
    fn stage(
        &self,
        cfg: &CfgBody,
        labels: &mut LabelFactory,
        index: usize,
        first: Option<Open>,
    ) -> Option<WrittenStage> {
        let spine = &self.shape.spine;
        let mine = |member: Member| self.owner_of(member) == Some(index);
        let mut writer = StageWriter {
            blocks: Vec::new(),
            entry: first.as_ref().map(|open| open.label),
            open: first,
            labels,
            relabeled: Vec::new(),
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
                    let reached = match heads {
                        true => Reached::AlsoByBackEdge,
                        false => Reached::OnlyFromBefore,
                    };
                    writer.begin(block.label, block.params.clone(), entering()?, reached);
                } else if spine_loop.is_none()
                    && !joins
                    && !owns_label
                    && runs_here
                    && writer.open.is_none()
                {
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
            if writer.open.is_none() {
                writer.begin_own();
            }
            let deciding = writer.open.take()?;
            if deciding.label != block.label && cfg.demoted_diamonds.contains(&block.label) {
                writer.relabeled.push(Relabeled {
                    from: block.label,
                    to: deciding.label,
                });
            }
            writer.close_deciding(deciding, term);
            for &arm in &self.shape.arms_of[&at] {
                let mut arm_block = cfg.blocks[arm.0].clone();
                retarget_edges(&mut arm_block.terminator, from, to);
                writer.blocks.push(arm_block);
            }
        }
        writer.finish()
    }
}

/// How control reaches a block a stage opens. A stage's entry is entered
/// only from the stage before it (RFC-0089 rule 1), so a stage that begins
/// at the header of a loop inside the body, which that loop's back edge
/// also enters, begins at a block of its own that jumps there.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Reached {
    OnlyFromBefore,
    AlsoByBackEdge,
}

/// A stage's blocks while they are written: those closed, the one open, and
/// the stage's entry once it has one.
struct StageWriter<'l> {
    blocks: Vec<Block>,
    entry: Option<Label>,
    open: Option<Open>,
    labels: &'l mut LabelFactory,
    relabeled: Vec<Relabeled>,
}

impl StageWriter<'_> {
    /// Opens `label`, closing the open block with a jump to it. A stage that
    /// has no block yet begins at `label`, or, where `label` takes
    /// parameters, at a block of its own that jumps there.
    fn begin(&mut self, label: Label, params: Vec<ValueId>, args: Vec<ValueId>, reached: Reached) {
        let jump = Terminator::Jump { label, args };
        match (self.open.take(), self.entry) {
            (Some(before), _) => self.blocks.push(before.close(jump)),
            (None, None) if params.is_empty() && reached == Reached::OnlyFromBefore => {
                self.entry = Some(label)
            }
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

    /// A `for` or pull `while` is a branch the spine passes when its body
    /// leaves on every path and no back edge makes it a loop. Its header
    /// holds no instruction: `acvus-interpreter`'s `prepare` reads the
    /// header as the label right above the terminator. So what the stage
    /// runs before it closes in a block of its own that jumps there.
    fn close_deciding(&mut self, deciding: Open, term: Terminator) {
        let traverses = matches!(term, Terminator::For { .. } | Terminator::While { .. });
        if !traverses || deciding.insts.is_empty() {
            self.blocks.push(deciding.close(term));
            return;
        }
        let header = self.labels.fresh();
        self.blocks.push(deciding.close(Terminator::Jump {
            label: header,
            args: Vec::new(),
        }));
        self.blocks.push(Open::empty(header).close(term));
    }

    fn begin_own(&mut self) {
        let own = self.labels.fresh();
        self.begin(own, Vec::new(), Vec::new(), Reached::OnlyFromBefore);
    }

    fn finish(mut self) -> Option<WrittenStage> {
        if self.open.is_none() && self.entry.is_none() {
            self.begin_own();
        }
        Some(WrittenStage {
            entry: self.entry?,
            blocks: self.blocks,
            last: self.open?,
            relabeled: self.relabeled,
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
        Terminator::For { exit, .. } | Terminator::While { exit, .. } => retarget(exit),
        Terminator::Return { .. } | Terminator::Diverge | Terminator::Fallthrough => {}
    }
}
