//! matchck (patterns §3): the block lower marks where no arm of a `match`
//! answers (`Terminator::NoArmMatched`) is reached by no run. Within the one
//! body, the variants a scrutinee may hold are known where every value
//! reaching it was made in the body (`TagValues`); a value from outside it,
//! a parameter, a capture, a call's result, a context or an input, holds
//! any variant its type has. An `Option`, a `Result` and an enum a
//! declaration writes have their variants by their type (types §2, patterns
//! §2), so every one of them is seen; a program's enum from outside the body
//! is not, and its `match` needs `_`.
//!
//! A `match` stores its scrutinee in a slot of its own and tests parts of
//! it through references to them. From that store on, every run is followed
//! through the tests, one variant chosen at each part the first time a test
//! reads it, among those `TagValues` says the part may hold, so a run is one
//! choice per part. A test on a chosen part is known, and so is every value
//! it passes on; an edge whose condition is not known may run either way. A
//! run that reaches `NoArmMatched`, or a `Switch` with no arm for its choice
//! and no default, is one no arm answers. A walk ends where the body
//! returns, diverges, or stores the scrutinee of another `match`, which is
//! judged from its own store.
//!
//! Cost: per `match`, O(blocks · the runs it tells apart), a run being one
//! choice per part tested, each block visited once per run and set of known
//! conditions.

use std::collections::{BTreeMap, BTreeSet};

use acvus_ast::{Literal, Span};
use acvus_utils::Astr;
use rustc_hash::FxHashSet;

use super::{Refusal, RefusalKind};
use crate::decl::Declarations;
use crate::ir::cfg::{BlockId, Body, BodyId, Module, Op, Terminator};
use crate::ir::{PathSeg, RefTarget, SwitchKey, ValueId};
use crate::passes::analysis::{AnalysisManager, TagValues};
use crate::passes::domain::tags::Tags;
use crate::ty::{Concrete, ObjectTy, Ty, TyKind, Types};

/// The judgment over `body`, its refusals pushed to `out`.
pub(super) fn check(body: BodyId, module: &Module, am: &mut AnalysisManager<'_>, out: &mut Vec<Refusal>) {
    let tags = am.get::<TagValues>(module, body);
    let decls = am.decls();
    let body = module.body(body);
    let scrutinees = scrutinees(body, &tags);
    let mut refused: BTreeSet<Span> = BTreeSet::new();
    for slot in &scrutinees {
        for store in stores(body, *slot) {
            let judge = Judge { body, types: module.types(), decls, tags: &tags, slot: *slot, scrutinees: &scrutinees };
            let after = At { block: store.block, index: store.index + 1, run: Run::default() };
            for span in judge.unanswered(after) {
                if refused.insert(span) {
                    out.push(Refusal { kind: RefusalKind::NotExhaustive, span });
                }
            }
        }
    }
}

/// Every slot a `match` tests: the slot a `Switch`'s tag or a test's
/// operand is a reference into.
fn scrutinees(body: &Body, tags: &TagValues) -> BTreeSet<ValueId> {
    let mut out = BTreeSet::new();
    for (_, block) in body.blocks() {
        for inst in block.insts() {
            if let Op::TestVariant { src, .. } | Op::TestLiteral { src, .. } = inst.op()
                && let Some((slot, _)) = tags.lent(*src)
            {
                out.insert(slot);
            }
        }
        if let Terminator::Switch { tag, .. } = block.terminator()
            && let Some((slot, _)) = tags.lent(*tag)
        {
            out.insert(slot);
        }
    }
    out
}

/// An instruction's place: its block, and its index among the block's.
#[derive(Debug, Clone, Copy)]
struct Place {
    block: BlockId,
    index: usize,
}

/// Each whole store into `slot`.
fn stores(body: &Body, slot: ValueId) -> Vec<Place> {
    let mut out = Vec::new();
    for (block, record) in body.blocks() {
        for (index, inst) in record.insts().enumerate() {
            if matches!(inst.op(), Op::Assign { target: RefTarget::Var(target), path, .. } if *target == slot && path.is_empty()) {
                out.push(Place { block, index });
            }
        }
    }
    out
}

/// A run about to go on from the instruction at `index` of `block`, or
/// from its terminator where `index` is past the last.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct At {
    block: BlockId,
    index: usize,
    run: Run,
}

/// What a run chose at one part of the scrutinee.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
enum Choice {
    Tag(Astr),
    Bool(bool),
    /// A value no key and no literal names: a variant of a program's enum
    /// the body does not see, or a value of a type whose values no `match`
    /// lists all of.
    Unseen,
}

/// One run: its choice at each part of the scrutinee, and the conditions
/// known on it.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Default)]
struct Run {
    chosen: BTreeMap<Vec<PathSeg>, Choice>,
    known: BTreeMap<ValueId, bool>,
}

struct Judge<'j> {
    body: &'j Body,
    types: &'j Types<Concrete>,
    decls: &'j Declarations,
    tags: &'j TagValues,
    slot: ValueId,
    scrutinees: &'j BTreeSet<ValueId>,
}

impl Judge<'_> {
    /// The span of each place no arm answers that some run from `start`
    /// reaches.
    fn unanswered(&self, start: At) -> Vec<Span> {
        let mut spans = Vec::new();
        let mut seen: FxHashSet<At> = FxHashSet::default();
        let mut work = vec![start];
        while let Some(at) = work.pop() {
            if seen.insert(at.clone()) {
                self.step(at, &mut work, &mut spans);
            }
        }
        spans
    }

    /// Follows a run to the end of its block, making its choice at a part
    /// the first time it is tested, and queuing a run for each other
    /// choice from that test.
    fn step(&self, At { block, index, mut run }: At, work: &mut Vec<At>, spans: &mut Vec<Span>) {
        let record = self.body.block(block);
        for (at, inst) in record.insts().enumerate().skip(index) {
            match inst.op() {
                Op::Assign { target: RefTarget::Var(target), path, .. } if path.is_empty() && self.scrutinees.contains(target) => return,
                Op::TestVariant { src, tag } => {
                    let Some(path) = self.part_of(*src) else { continue };
                    let Some(chosen) = self.choose(&run, &path, Place { block, index: at }, work) else { return };
                    run = chosen;
                    let result = inst.result().expect("a test yields a value");
                    run.known.insert(result, run.chosen[&path] == Choice::Tag(*tag));
                }
                Op::TestLiteral { src, value } => {
                    let Some(path) = self.part_of(*src) else { continue };
                    let Some(chosen) = self.choose(&run, &path, Place { block, index: at }, work) else { return };
                    run = chosen;
                    let result = inst.result().expect("a test yields a value");
                    let holds = match (run.chosen[&path], value.desugared()) {
                        (Choice::Bool(chosen), Literal::Bool(tested)) => chosen == tested,
                        _ => false,
                    };
                    run.known.insert(result, holds);
                }
                Op::Const { value: Literal::Bool(value) } => {
                    run.known.insert(inst.result().expect("a constant yields a value"), *value);
                }
                _ => {}
            }
        }
        self.terminate(block, run, work, spans);
    }

    /// `run` with a choice at `path`: its own where it made one; else the
    /// first of the part's, a run with each other queued to go on from the
    /// test at `from`. `None` where the part holds no value, so that no run
    /// goes on.
    fn choose(&self, run: &Run, path: &[PathSeg], from: Place, work: &mut Vec<At>) -> Option<Run> {
        if run.chosen.contains_key(path) {
            return Some(run.clone());
        }
        let choices = self.choices(run, path)?;
        let (first, rest) = choices.split_first().expect("choices are some");
        for choice in rest {
            let mut other = run.clone();
            other.chosen.insert(path.to_vec(), *choice);
            work.push(At { block: from.block, index: from.index, run: other });
        }
        let mut taken = run.clone();
        taken.chosen.insert(path.to_vec(), *first);
        Some(taken)
    }

    /// Every value the part at `path` may hold, as far as a test tells
    /// them apart; `None` where it holds none.
    fn choices(&self, run: &Run, path: &[PathSeg]) -> Option<Vec<Choice>> {
        let ty = self.ty_at(run, path)?;
        let choices = match (self.tags.in_slot(self.slot, path), self.variants(ty)) {
            (Tags::None, _) => Vec::new(),
            (Tags::Of(held), Some((variants, _))) => {
                variants.into_iter().filter(|tag| held.contains(tag)).map(Choice::Tag).collect()
            }
            (Tags::Any, Some((variants, seen))) => {
                let mut choices: Vec<Choice> = variants.into_iter().map(Choice::Tag).collect();
                if !seen {
                    choices.push(Choice::Unseen);
                }
                choices
            }
            (_, None) => match self.types.kind(ty) {
                TyKind::Bool => vec![Choice::Bool(false), Choice::Bool(true)],
                _ => vec![Choice::Unseen],
            },
        };
        (!choices.is_empty()).then_some(choices)
    }

    /// The variants of `ty` where it is a sum, and whether every one of
    /// them is seen from here: an `Option`'s, a `Result`'s and an enum's a
    /// declaration writes are, by their type; a program's enum's are not,
    /// since a value from outside the body may hold one it does not see.
    fn variants(&self, ty: Ty) -> Option<(Vec<Astr>, bool)> {
        let tags = self.decls.core().tags;
        match self.types.kind(ty) {
            TyKind::Option(..) => Some((vec![tags.some, tags.none], true)),
            TyKind::Result(..) => Some((vec![tags.ok, tags.err], true)),
            TyKind::Enum { name, variants, .. } => {
                let declared = self.decls.types().kinds().any(|kind| matches!(kind, TyKind::Enum { name: written, .. } if written == name));
                Some((variants.iter().map(|(tag, _)| *tag).collect(), declared))
            }
            _ => None,
        }
    }

    /// The type of the scrutinee's part at `path`, a payload's by the
    /// variant `run` chose at its holder; `None` where the holder's choice
    /// has no payload.
    fn ty_at(&self, run: &Run, path: &[PathSeg]) -> Option<Ty> {
        let mut ty = self.body.ty_of(self.slot);
        for (depth, seg) in path.iter().enumerate() {
            ty = match (self.types.kind(ty), seg) {
                (TyKind::Tuple(items, _), PathSeg::Index(at)) => items[*at],
                (TyKind::Array(element, ..), PathSeg::Index(_)) => *element,
                (TyKind::Object(object), PathSeg::Field(name)) => field(object, *name),
                (TyKind::Option(payload, _), PathSeg::Payload) => *payload,
                (TyKind::Result(ok, err, _), PathSeg::Payload) => match run.chosen.get(&path[..depth]) {
                    Some(Choice::Tag(tag)) if *tag == self.decls.core().tags.ok => *ok,
                    Some(Choice::Tag(_)) => *err,
                    _ => return None,
                },
                (TyKind::Enum { variants, .. }, PathSeg::Payload) => match run.chosen.get(&path[..depth]) {
                    Some(Choice::Tag(tag)) => variants.iter().find(|(name, _)| name == tag).and_then(|(_, payload)| *payload)?,
                    _ => return None,
                },
                (kind, seg) => panic!("a scrutinee's part is reached by its type's own steps, not {seg:?} of {kind:?}"),
            };
        }
        Some(ty)
    }

    /// The path of the scrutinee's part `value` is a reference to.
    fn part_of(&self, value: ValueId) -> Option<Vec<PathSeg>> {
        match self.tags.lent(value) {
            Some((slot, path)) if slot == self.slot => Some(path.to_vec()),
            _ => None,
        }
    }

    /// The edges `run` may take out of `block`.
    fn terminate(&self, block: BlockId, run: Run, work: &mut Vec<At>, spans: &mut Vec<Span>) {
        let record = self.body.block(block);
        let terminator = record.terminator();
        let known = |run: &Run, value: &ValueId| run.known.get(value).copied();
        let go = |run: &Run, to: BlockId, args: &[ValueId], work: &mut Vec<At>| {
            let mut next = run.clone();
            for (param, arg) in self.body.block(to).params().iter().zip(args) {
                match known(run, arg) {
                    Some(value) => next.known.insert(*param, value),
                    None => next.known.remove(param),
                };
            }
            work.push(At { block: to, index: 0, run: next });
        };
        match terminator {
            Terminator::NoArmMatched { span } => spans.push(*span),
            Terminator::Jump { to, args } => go(&run, *to, args, work),
            Terminator::JumpIf { cond, then_to, then_args, else_to, else_args }
            | Terminator::Diamond { cond, then_to, then_args, else_to, else_args, .. } => match known(&run, cond) {
                Some(true) => go(&run, *then_to, then_args, work),
                Some(false) => go(&run, *else_to, else_args, work),
                None => {
                    go(&run, *then_to, then_args, work);
                    go(&run, *else_to, else_args, work);
                }
            },
            Terminator::Switch { tag, arms, default, .. } => {
                let Some(path) = self.part_of(*tag) else {
                    for edge in terminator.edges() {
                        go(&run, edge.to, edge.args, work);
                    }
                    return;
                };
                let runs: Vec<Run> = match run.chosen.get(&path) {
                    Some(_) => vec![run],
                    None => {
                        // A part that holds no value is reached by no run.
                        let Some(choices) = self.choices(&run, &path) else { return };
                        choices
                            .iter()
                            .map(|choice| {
                                let mut one = run.clone();
                                one.chosen.insert(path.clone(), *choice);
                                one
                            })
                            .collect()
                    }
                };
                for one in runs {
                    let key = match one.chosen[&path] {
                        Choice::Tag(tag) => Some(SwitchKey::Tag(tag)),
                        Choice::Bool(value) => Some(SwitchKey::Bool(value)),
                        Choice::Unseen => None,
                    };
                    match (arms.iter().find(|(at, _, _)| Some(*at) == key), default) {
                        (Some((_, to, args)), _) | (None, Some((to, args))) => go(&one, *to, args, work),
                        (None, None) => spans.push(record.terminator_span()),
                    }
                }
            }
            Terminator::Return { .. } | Terminator::Diverge => {}
            _ => {
                for edge in terminator.edges() {
                    go(&run, edge.to, edge.args, work);
                }
            }
        }
    }
}

/// The type of `object`'s field `name`.
fn field(object: &ObjectTy<Concrete>, name: Astr) -> Ty {
    object
        .fields()
        .iter()
        .find(|(field, _)| *field == name)
        .map(|(_, ty)| *ty)
        .unwrap_or_else(|| panic!("a scrutinee's field is one of its type"))
}
