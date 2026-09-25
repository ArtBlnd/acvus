//! RFC-0094 rule 6: a back edge whose arguments decide the header's test
//! goes to the exit.
//!
//! The header's condition is the chain `&&` and `||` lower to (RFC-0063
//! rule 1): a `Diamond` per operator, whose arms send the join the test's
//! value or the constant that short-circuits, and a last `JumpIf` on the
//! chain's value, to the body or out of the loop. A back edge that sends a
//! header parameter a `bool` constant is read by walking that chain with
//! the parameter known and every other test unknown (Kleene's three
//! values). Where every path the walk may take leaves, the chain is false
//! whatever the other tests give, and the visit decides nothing the exit
//! could not be sent directly: that edge goes to the exit, carrying what
//! the header would have sent it on that visit. The walk also names every
//! instruction the visit may run, and each must be one an unused removal
//! may drop (`dce::unused_may_drop`, RFC-0048 rule 8), knowing no callee's
//! body, so a call qualifies only by its extern's `total` (RFC-0082 rule
//! 9). A skipped visit then computes nothing observable.
//!
//! A back edge here is a path: the edge a block sends the header through
//! blocks that hold no instruction and only pass their parameters on, as
//! the join of an `if` in the body does. Such a block computes nothing, so
//! the edge into it sends the header what the join passes on.
//!
//! The exit reads the header's parameters by dominance, so where it reads
//! one it gains a parameter for it: the header's own exit edge sends the
//! parameter, and the moved edge the value it sent the header.
//!
//! Once every edge sending the constant has left, a parameter every
//! entering edge sends the constant's opposite, and every remaining back
//! edge the same or itself, holds that value on every visit: it is folded
//! to the entering constant, and the chain is walked again to the one test
//! still undecided, which becomes the header's branch. Rules 1 to 5 read
//! that branch; where they do not convert the loop, it is left exactly as
//! it was.

use acvus_ast::Literal;
use rustc_hash::{FxHashMap, FxHashSet};
use smallvec::SmallVec;

use crate::analysis::domtree::DomTree;
use crate::analysis::inst_info;
use crate::analysis::interval::InstAt;
use crate::analysis::loans::Loans;
use crate::analysis::loops::{Invariants, Loop, LoopKind, LoopNest, edge_args};
use crate::analysis::raise::{FunctionSummary, Removal};
use crate::cfg::{BlockIdx, CfgBody, Terminator, demote_diamond, prune, reachable};
use crate::ir::{BinOp, Inst, InstKind, Label, UnaryOp, ValueId};
use crate::laws::LawTable;
use crate::optimize::dce;
use crate::optimize::forward::edges_mut;
use crate::optimize::ssa_pass::{apply_subst, apply_subst_terminator};
use crate::ty::Ty;

use super::{const_literals, converts, fresh};

pub(super) fn run(cfg: &mut CfgBody, laws: &LawTable) {
    let mut tried: FxHashSet<Label> = FxHashSet::default();
    while let Some(left) = next(cfg, laws, &mut tried) {
        *cfg = left;
    }
}

/// The body with the first loop not tried yet whose leaving edges make it
/// a loop rules 1 to 5 convert.
fn next(cfg: &CfgBody, laws: &LawTable, tried: &mut FxHashSet<Label>) -> Option<CfgBody> {
    let domtree = DomTree::build(cfg);
    let invariants = Invariants::of(cfg);
    let nest = LoopNest::of(cfg, &domtree, &invariants);
    let loans = Loans::build(cfg);
    let functions = FunctionSummary::unknown();
    let removal = Removal::of(cfg, laws, &functions);
    let preds = cfg.predecessors();
    let literals = const_literals(cfg);
    let defs = definitions(cfg);
    for (_, loop_) in nest.iter() {
        let header = cfg.blocks[loop_.natural.header.0].label;
        if !tried.insert(header) {
            continue;
        }
        let natural = &loop_.natural;
        let leaves = |label: Label| !natural.contains(cfg.label_to_block[&label]);
        let reading = Reading {
            cfg,
            loop_,
            domtree: &domtree,
            loans: &loans,
            removal: &removal,
            laws,
            preds: &preds,
            chain: Chain {
                cfg,
                defs: &defs,
                literals: &literals,
                leaves: &leaves,
            },
        };
        let Some(left) = reading.leaving() else {
            continue;
        };
        if converts(&left, laws, header) {
            return Some(left);
        }
    }
    None
}

fn definitions(cfg: &CfgBody) -> FxHashMap<ValueId, InstKind> {
    cfg.blocks
        .iter()
        .flat_map(|block| &block.insts)
        .flat_map(|inst| {
            inst_info::defs(&inst.kind)
                .into_iter()
                .map(|def| (def, inst.kind.clone()))
        })
        .collect()
}

/// A back edge as a path: `from`'s edge to `to`, which is the header or a
/// block that passes its parameters on to it, and what reaches each header
/// parameter along it.
struct BackPath {
    from: BlockIdx,
    to: Label,
    sends: Vec<ValueId>,
    /// The blocks between `to` and the header that only pass values on.
    through: Vec<Label>,
}

/// The chain of tests from a block to the branch that leaves the loop.
struct Chain<'a> {
    cfg: &'a CfgBody,
    defs: &'a FxHashMap<ValueId, InstKind>,
    literals: &'a FxHashMap<ValueId, Literal>,
    leaves: &'a dyn Fn(Label) -> bool,
}

/// A way through the chain, up to the branch that ends it.
#[derive(Clone, Default)]
struct Trace {
    visited: Vec<BlockIdx>,
    ran: Vec<InstAt>,
    bound: Vec<Bound>,
    known: FxHashMap<ValueId, bool>,
    decisions: Vec<Decision>,
}

/// A way through the chain and the edge it leaves the chain by.
struct Way {
    trace: Trace,
    end: End,
}

/// A join's parameter and what this way sent it.
#[derive(Clone, Copy)]
struct Bound {
    param: ValueId,
    arg: ValueId,
}

/// A test the way found undecided, the arm it took, and how much the way
/// had run and bound before it.
#[derive(Clone, Copy)]
struct Decision {
    at: BlockIdx,
    cond: ValueId,
    then: bool,
    ran_before: usize,
    bound_before: usize,
}

struct End {
    leaves: bool,
    label: Label,
    args: Vec<ValueId>,
}

/// One edge of a two-way branch.
struct Arm<'a> {
    then: bool,
    label: Label,
    args: &'a [ValueId],
}

fn arms<'a>(
    then_label: Label,
    then_args: &'a [ValueId],
    else_label: Label,
    else_args: &'a [ValueId],
) -> [Arm<'a>; 2] {
    [
        Arm {
            then: true,
            label: then_label,
            args: then_args,
        },
        Arm {
            then: false,
            label: else_label,
            args: else_args,
        },
    ]
}

impl Trace {
    fn argument(&self, value: ValueId) -> Option<ValueId> {
        self.bound
            .iter()
            .rev()
            .find(|bound| bound.param == value)
            .map(|bound| bound.arg)
    }

    fn decide(&mut self, at: BlockIdx, cond: ValueId, then: bool) {
        self.decisions.push(Decision {
            at,
            cond,
            then,
            ran_before: self.ran.len(),
            bound_before: self.bound.len(),
        });
    }

    /// `value` with each join parameter it names replaced by what this way
    /// sent it.
    fn resolve(&self, mut value: ValueId) -> ValueId {
        while let Some(arg) = self.argument(value) {
            value = arg;
        }
        value
    }
}

impl Chain<'_> {
    /// Every way from `from` through the chain, with `known` holding the
    /// values fixed on entry. `None` where the blocks are no chain: a jump
    /// that is no join, a branch that leaves from inside an arm, a last
    /// branch that is not one edge out of the loop and one edge on, or a
    /// block a way reaches twice.
    fn visit(&self, from: BlockIdx, known: FxHashMap<ValueId, bool>) -> Option<Vec<Way>> {
        let mut ways = Vec::new();
        let trace = Trace {
            known,
            ..Trace::default()
        };
        self.walk(from, trace, Vec::new(), &mut ways)?;
        Some(ways)
    }

    fn walk(
        &self,
        block: BlockIdx,
        mut trace: Trace,
        mut joins: Vec<Label>,
        ways: &mut Vec<Way>,
    ) -> Option<()> {
        if trace.visited.contains(&block) {
            return None;
        }
        let held = &self.cfg.blocks[block.0];
        trace.visited.push(block);
        trace
            .ran
            .extend((0..held.insts.len()).map(|at| InstAt { block, at }));
        match &held.terminator {
            Terminator::Jump { label, args } => {
                if joins.pop() != Some(*label) {
                    return None;
                }
                let to = self.cfg.label_to_block[label];
                self.bind(&mut trace, to, args)?;
                self.walk(to, trace, joins, ways)
            }
            Terminator::Diamond {
                cond,
                then_label,
                then_args,
                else_label,
                else_args,
                join,
            } => {
                joins.push(*join);
                let truth = self.truth(&trace, *cond);
                for arm in arms(*then_label, then_args, *else_label, else_args) {
                    if truth.is_some_and(|truth| truth != arm.then) {
                        continue;
                    }
                    let mut taken = trace.clone();
                    if truth.is_none() {
                        taken.decide(block, *cond, arm.then);
                    }
                    let to = self.cfg.label_to_block[&arm.label];
                    self.bind(&mut taken, to, arm.args)?;
                    self.walk(to, taken, joins.clone(), ways)?;
                }
                Some(())
            }
            Terminator::JumpIf {
                cond,
                then_label,
                then_args,
                else_label,
                else_args,
            } if joins.is_empty() && (self.leaves)(*then_label) != (self.leaves)(*else_label) => {
                let truth = self.truth(&trace, *cond);
                for arm in arms(*then_label, then_args, *else_label, else_args) {
                    if truth.is_some_and(|truth| truth != arm.then) {
                        continue;
                    }
                    let mut taken = trace.clone();
                    if truth.is_none() {
                        taken.decide(block, *cond, arm.then);
                    }
                    ways.push(Way {
                        trace: taken,
                        end: End {
                            leaves: (self.leaves)(arm.label),
                            label: arm.label,
                            args: arm.args.to_vec(),
                        },
                    });
                }
                Some(())
            }
            _ => None,
        }
    }

    fn bind(&self, trace: &mut Trace, to: BlockIdx, args: &[ValueId]) -> Option<()> {
        let params = &self.cfg.blocks[to.0].params;
        if params.len() != args.len() {
            return None;
        }
        for (param, arg) in params.iter().zip(args) {
            if let Some(truth) = self.truth(trace, *arg) {
                trace.known.insert(*param, truth);
            }
            trace.bound.push(Bound {
                param: *param,
                arg: *arg,
            });
        }
        Some(())
    }

    /// What a `bool` is on this way: a constant, a value the way fixed, or
    /// `!`, `==` or `!=` of such values; `None` where it is a test the way
    /// does not decide.
    fn truth(&self, trace: &Trace, value: ValueId) -> Option<bool> {
        if let Some(known) = trace.known.get(&value) {
            return Some(*known);
        }
        if let Some(Literal::Bool(literal)) = self.literals.get(&value) {
            return Some(*literal);
        }
        match self.defs.get(&value)? {
            InstKind::UnaryOp {
                op: UnaryOp::Not,
                operand,
                ..
            } => self.truth(trace, *operand).map(|truth| !truth),
            InstKind::BinOp {
                op: op @ (BinOp::Eq | BinOp::Neq),
                left,
                right,
                ..
            } => {
                let left = self.truth(trace, *left)?;
                let right = self.truth(trace, *right)?;
                Some((left == right) == matches!(op, BinOp::Eq))
            }
            _ => None,
        }
    }
}

struct Reading<'a> {
    cfg: &'a CfgBody,
    loop_: &'a Loop,
    domtree: &'a DomTree,
    loans: &'a Loans<'a>,
    removal: &'a Removal<'a>,
    laws: &'a LawTable,
    preds: &'a FxHashMap<BlockIdx, SmallVec<[BlockIdx; 2]>>,
    chain: Chain<'a>,
}

/// A header parameter the leaving edges fix, and the constant every entry
/// sends it.
struct Flag {
    index: usize,
    param: ValueId,
    entered: ValueId,
}

/// A back path that goes to the exit, and what it sends the exit.
struct Leaving<'a> {
    path: &'a BackPath,
    args: Vec<ValueId>,
}

/// The block the chain leaves the loop for, and the header parameters, by
/// index, that it or a block it dominates reads.
struct Exit {
    label: Label,
    read_past: Vec<usize>,
}

impl Reading<'_> {
    fn leaving(&self) -> Option<CfgBody> {
        let LoopKind::While = self.loop_.kind else {
            return None;
        };
        let natural = &self.loop_.natural;
        let [entering] = natural.entering[..] else {
            return None;
        };
        let header_label = self.cfg.blocks[natural.header.0].label;
        let entry_args = edge_args(&self.cfg.blocks[entering.0].terminator, header_label)?;
        let paths = self.back_paths()?;
        let params = &self.cfg.blocks[natural.header.0].params;
        params.iter().enumerate().find_map(|(index, &param)| {
            let flag = Flag {
                index,
                param,
                entered: entry_args[index],
            };
            self.leaving_on(&flag, &paths)
        })
    }

    fn boolean(&self, value: ValueId) -> Option<bool> {
        match self.chain.literals.get(&value)? {
            Literal::Bool(literal) => Some(*literal),
            _ => None,
        }
    }

    fn leaving_on(&self, flag: &Flag, paths: &[BackPath]) -> Option<CfgBody> {
        if self.cfg.val_types.get(&flag.param) != Some(&Ty::Bool) {
            return None;
        }
        let entered = self.boolean(flag.entered)?;
        let decides = !entered;
        let sends_decides = |path: &&BackPath| self.boolean(path.sends[flag.index]) == Some(decides);
        let leave: Vec<&BackPath> = paths.iter().filter(sends_decides).collect();
        if leave.is_empty() {
            return None;
        }
        let invariant = paths.iter().filter(|path| !sends_decides(path)).all(|path| {
            let sent = path.sends[flag.index];
            sent == flag.param || self.boolean(sent) == Some(entered)
        });
        if !invariant {
            return None;
        }

        let ways = self
            .chain
            .visit(self.loop_.natural.header, FxHashMap::from_iter([(flag.param, decides)]))?;
        let [first, ..] = &ways[..] else {
            return None;
        };
        let label = first.end.label;
        if !ways.iter().all(|way| way.end.leaves && way.end.label == label) {
            return None;
        }
        let removable = ways.iter().flat_map(|way| &way.trace.ran).all(|at| {
            let kind = &self.cfg.blocks[at.block.0].insts[at.at].kind;
            dce::unused_may_drop(*at, kind, self.loans, self.removal)
        });
        if !removable {
            return None;
        }
        let exit = Exit {
            label,
            read_past: self.read_past_exit(self.cfg.label_to_block[&label])?,
        };
        let leaving = leave
            .into_iter()
            .map(|path| {
                let mut args = self.exit_args_sent(&ways, path)?;
                args.extend(exit.read_past.iter().map(|&index| path.sends[index]));
                Some(Leaving { path, args })
            })
            .collect::<Option<Vec<_>>>()?;
        self.rewrite(flag, &leaving, &exit)
    }

    /// What the header would have sent the exit's arguments on the visit
    /// `path` starts, the same on every way the chain may take.
    fn exit_args_sent(&self, ways: &[Way], path: &BackPath) -> Option<Vec<ValueId>> {
        let mut sent: Vec<Vec<ValueId>> = ways
            .iter()
            .map(|way| {
                way.end
                    .args
                    .iter()
                    .map(|arg| self.sent(way.trace.resolve(*arg), path))
                    .collect::<Option<Vec<_>>>()
            })
            .collect::<Option<_>>()?;
        sent.dedup();
        match <[_; 1]>::try_from(sent) {
            Ok([args]) => Some(args),
            Err(_) => None,
        }
    }

    /// What the header would have sent for `value` on the visit `path`
    /// starts: a header parameter is what the path sends it, a value
    /// defined above the loop is itself.
    fn sent(&self, value: ValueId, path: &BackPath) -> Option<ValueId> {
        let natural = &self.loop_.natural;
        let params = &self.cfg.blocks[natural.header.0].params;
        if let Some(index) = params.iter().position(|param| *param == value) {
            return Some(path.sends[index]);
        }
        (!self.defined_in_loop().contains(&value)).then_some(value)
    }

    fn defined_in_loop(&self) -> FxHashSet<ValueId> {
        self.loop_
            .natural
            .blocks()
            .flat_map(|block| {
                let held = &self.cfg.blocks[block.0];
                held.params.iter().copied().chain(
                    held.insts
                        .iter()
                        .flat_map(|inst| inst_info::defs(&inst.kind)),
                )
            })
            .collect()
    }

    /// `None` where a block the exit dominates reads a value of the loop
    /// other than a header parameter, or a block past the exit it does not
    /// dominate reads one: that block would see the parameter as the visit
    /// before the skipped one left it.
    fn read_past_exit(&self, exit: BlockIdx) -> Option<Vec<usize>> {
        let natural = &self.loop_.natural;
        let params = &self.cfg.blocks[natural.header.0].params;
        let in_loop = self.defined_in_loop();
        let past_exit = reached_from(self.cfg, exit);
        let mut read: Vec<usize> = Vec::new();
        for (at, block) in self.cfg.blocks.iter().enumerate() {
            let at = BlockIdx(at);
            if natural.contains(at) {
                continue;
            }
            let reads = block
                .insts
                .iter()
                .flat_map(|inst| inst_info::uses(&inst.kind))
                .chain(inst_info::terminator_uses(&block.terminator))
                .filter(|value| in_loop.contains(value));
            for value in reads {
                if !self.domtree.dominates(exit, at) {
                    if past_exit.contains(&at) {
                        return None;
                    }
                    continue;
                }
                let index = params.iter().position(|param| *param == value)?;
                if !read.contains(&index) {
                    read.push(index);
                }
            }
        }
        Some(read)
    }

    /// The edges into the header from inside the loop, each followed back
    /// through the blocks that only pass their parameters on. `None` where
    /// a block reaches one of them by two edges.
    fn back_paths(&self) -> Option<Vec<BackPath>> {
        let natural = &self.loop_.natural;
        let header = natural.header;
        let header_label = self.cfg.blocks[header.0].label;
        let mut work: Vec<BackPath> = Vec::new();
        for &latch in self.preds.get(&header)?.iter() {
            if !natural.contains(latch) {
                continue;
            }
            let sends = edge_args(&self.cfg.blocks[latch.0].terminator, header_label)?;
            work.push(BackPath {
                from: latch,
                to: header_label,
                sends: sends.to_vec(),
                through: Vec::new(),
            });
        }
        let mut paths = Vec::new();
        while let Some(path) = work.pop() {
            let held = &self.cfg.blocks[path.from.0];
            let passes_on = path.from != header
                && held.insts.iter().all(|inst| matches!(inst.kind, InstKind::Nop))
                && matches!(held.terminator, Terminator::Jump { .. });
            let Some(preds) = self.preds.get(&path.from).filter(|_| passes_on) else {
                paths.push(path);
                continue;
            };
            for &pred in preds.iter() {
                let args = edge_args(&self.cfg.blocks[pred.0].terminator, held.label)?;
                let sends = path
                    .sends
                    .iter()
                    .map(|sent| match held.params.iter().position(|param| param == sent) {
                        Some(index) => args[index],
                        None => *sent,
                    })
                    .collect();
                let mut through = path.through.clone();
                through.push(held.label);
                work.push(BackPath {
                    from: pred,
                    to: held.label,
                    sends,
                    through,
                });
            }
        }
        Some(paths)
    }

    fn rewrite(&self, flag: &Flag, leaving: &[Leaving<'_>], exit: &Exit) -> Option<CfgBody> {
        let mut cfg = self.cfg.clone();
        let header = self.loop_.natural.header;
        let header_label = self.cfg.blocks[header.0].label;
        let exit_block = self.cfg.label_to_block[&exit.label];

        // The exit gains a parameter for each header parameter read past
        // it, and every edge already into it sends the parameter itself.
        let read: Vec<ValueId> = exit
            .read_past
            .iter()
            .map(|&index| self.cfg.blocks[header.0].params[index])
            .collect();
        let mut outs: FxHashMap<ValueId, ValueId> = FxHashMap::default();
        for &value in &read {
            let ty = self.cfg.val_types[&value].clone();
            outs.insert(value, fresh(&mut cfg, &ty));
        }
        for &pred in self.preds.get(&exit_block)?.iter() {
            for edge in edges_mut(&mut cfg.blocks[pred.0].terminator) {
                if *edge.to == exit.label {
                    edge.args.extend(read.iter().copied());
                }
            }
        }
        for at in 0..cfg.blocks.len() {
            if !self.domtree.dominates(exit_block, BlockIdx(at)) {
                continue;
            }
            let block = &mut cfg.blocks[at];
            for inst in &mut block.insts {
                apply_subst(&mut inst.kind, &outs);
            }
            apply_subst_terminator(&mut block.terminator, &outs);
        }
        cfg.blocks[exit_block.0]
            .params
            .extend(read.iter().map(|value| outs[value]));

        let mut through: FxHashSet<Label> = FxHashSet::default();
        for Leaving { path, args } in leaving {
            demote_diamond(&mut cfg, path.from);
            let mut moved = 0;
            for edge in edges_mut(&mut cfg.blocks[path.from.0].terminator) {
                if *edge.to == path.to {
                    *edge.to = exit.label;
                    *edge.args = args.clone();
                    moved += 1;
                }
            }
            if moved != 1 {
                return None;
            }
            through.extend(path.through.iter().copied());
        }

        for label in through {
            pass_on_its_one_jump(&mut cfg, label);
        }
        let alive = reachable(&cfg);
        prune(&mut cfg, &alive);
        demote_the_unjoined(&mut cfg);

        substitute(&mut cfg, &FxHashMap::from_iter([(flag.param, flag.entered)]));
        reduce(&mut cfg, self.laws, header_label, exit.label)?;
        Some(cfg)
    }
}

/// A block that passed values on and is now entered by one `Jump` holds
/// what that jump sends. A block no edge enters any more is left to the
/// pruning that follows.
fn pass_on_its_one_jump(cfg: &mut CfgBody, label: Label) {
    let block = cfg.label_to_block[&label];
    let preds = cfg.predecessors();
    let Some([pred]) = preds.get(&block).map(|preds| &preds[..]) else {
        return;
    };
    let Terminator::Jump { label: to, args } = &mut cfg.blocks[pred.0].terminator else {
        return;
    };
    if *to != label {
        return;
    }
    let passed = std::mem::take(args);
    let params = std::mem::take(&mut cfg.blocks[block.0].params);
    let subst: FxHashMap<ValueId, ValueId> = params.into_iter().zip(passed).collect();
    substitute(cfg, &subst);
}

/// The chain with its flag folded, as the one test left undecided: the
/// header runs every step up to that test and branches on it to the body
/// or the exit, and the chain's other blocks are gone. The steps after that
/// test compute tests the fold decided, and are dropped as an unused
/// removal drops them. `None` where no single test is left, where a step
/// after it is not one an unused removal may drop, or where a value of a
/// block that is gone is still read.
fn reduce(cfg: &mut CfgBody, laws: &LawTable, header: Label, exit: Label) -> Option<()> {
    let defs = definitions(cfg);
    let literals = const_literals(cfg);
    let at = cfg.label_to_block[&header];
    let leaves = |label: Label| label == exit;
    let chain = Chain {
        cfg,
        defs: &defs,
        literals: &literals,
        leaves: &leaves,
    };
    let ways = chain.visit(at, FxHashMap::default())?;
    let [one, other] = &ways[..] else {
        return None;
    };
    let ([decision], [opposite]) = (&one.trace.decisions[..], &other.trace.decisions[..]) else {
        return None;
    };
    let decided_alike =
        decision.at == opposite.at && decision.cond == opposite.cond && decision.then != opposite.then;
    if !decided_alike {
        return None;
    }
    let loans = Loans::build(cfg);
    let functions = FunctionSummary::unknown();
    let removal = Removal::of(cfg, laws, &functions);
    let dropped_unused = ways
        .iter()
        .flat_map(|way| &way.trace.ran[decision.ran_before..])
        .all(|at| {
            let kind = &cfg.blocks[at.block.0].insts[at.at].kind;
            dce::unused_may_drop(*at, kind, &loans, &removal)
        });
    if !dropped_unused {
        return None;
    }
    let body = ways.iter().find(|way| way.trace.decisions[0].then)?;
    let left = ways.iter().find(|way| !way.trace.decisions[0].then)?;
    if body.end.leaves || !left.end.leaves {
        return None;
    }
    let prefix: FxHashMap<ValueId, ValueId> = one.trace.bound[..decision.bound_before]
        .iter()
        .map(|bound| (bound.param, one.trace.resolve(bound.param)))
        .collect();
    let moved: Vec<Inst> = one.trace.ran[..decision.ran_before]
        .iter()
        .filter(|ran| ran.block != at)
        .map(|ran| cfg.blocks[ran.block.0].insts[ran.at].clone())
        .collect();
    let gone: FxHashSet<Label> = ways
        .iter()
        .flat_map(|way| &way.trace.visited)
        .filter(|block| **block != at)
        .map(|block| cfg.blocks[block.0].label)
        .collect();
    let kept: FxHashSet<ValueId> = moved
        .iter()
        .flat_map(|inst| inst_info::defs(&inst.kind))
        .collect();
    let lost: FxHashSet<ValueId> = gone
        .iter()
        .flat_map(|label| {
            let block = &cfg.blocks[cfg.label_to_block[label].0];
            block.params.iter().copied().chain(
                block
                    .insts
                    .iter()
                    .flat_map(|inst| inst_info::defs(&inst.kind)),
            )
        })
        .filter(|value| !kept.contains(value) && !prefix.contains_key(value))
        .collect();
    let branch = Terminator::JumpIf {
        cond: decision.cond,
        then_label: body.end.label,
        then_args: body.end.args.iter().map(|arg| body.trace.resolve(*arg)).collect(),
        else_label: left.end.label,
        else_args: left.end.args.iter().map(|arg| left.trace.resolve(*arg)).collect(),
    };
    let held = &mut cfg.blocks[at.0];
    held.insts.extend(moved);
    held.terminator = branch;
    cfg.demoted_diamonds.remove(&header);
    substitute(cfg, &prefix);
    let alive = reachable(cfg);
    prune(cfg, &alive);
    if gone.iter().any(|label| cfg.label_to_block.contains_key(label)) {
        return None;
    }
    let still_read = cfg.blocks.iter().any(|block| {
        block
            .insts
            .iter()
            .flat_map(|inst| inst_info::uses(&inst.kind))
            .chain(inst_info::terminator_uses(&block.terminator))
            .any(|value| lost.contains(&value))
    });
    (!still_read).then_some(())
}

/// A `Diamond` whose arm no longer reaches its join is a `JumpIf`
/// (RFC-0063 rule 5); `optimize::rejoin` restores one whose arms meet
/// again.
fn demote_the_unjoined(cfg: &mut CfgBody) {
    let unjoined: Vec<BlockIdx> = (0..cfg.blocks.len())
        .map(BlockIdx)
        .filter(|at| match &cfg.blocks[at.0].terminator {
            Terminator::Diamond {
                then_label,
                else_label,
                join,
                ..
            } => !rejoins(cfg, *at, *then_label, *join) || !rejoins(cfg, *at, *else_label, *join),
            _ => false,
        })
        .collect();
    for at in unjoined {
        demote_diamond(cfg, at);
    }
}

/// A join the pruning removed, every edge into it having left, is reached
/// by no arm.
fn rejoins(cfg: &CfgBody, at: BlockIdx, arm: Label, join: Label) -> bool {
    let Some(&join) = cfg.label_to_block.get(&join) else {
        return false;
    };
    let mut seen: FxHashSet<BlockIdx> = FxHashSet::default();
    let mut work = vec![cfg.label_to_block[&arm]];
    while let Some(block) = work.pop() {
        if block == join {
            return true;
        }
        if block == at || !seen.insert(block) {
            continue;
        }
        work.extend(cfg.successors(block));
    }
    false
}

fn reached_from(cfg: &CfgBody, from: BlockIdx) -> FxHashSet<BlockIdx> {
    let mut seen: FxHashSet<BlockIdx> = FxHashSet::default();
    let mut work = vec![from];
    while let Some(block) = work.pop() {
        if seen.insert(block) {
            work.extend(cfg.successors(block));
        }
    }
    seen
}

fn substitute(cfg: &mut CfgBody, subst: &FxHashMap<ValueId, ValueId>) {
    if subst.is_empty() {
        return;
    }
    for block in &mut cfg.blocks {
        for inst in &mut block.insts {
            apply_subst(&mut inst.kind, subst);
        }
        apply_subst_terminator(&mut block.terminator, subst);
    }
}
