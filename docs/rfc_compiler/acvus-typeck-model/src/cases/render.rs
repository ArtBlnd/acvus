//! What the model gives, written as a case writes it: decisions, coercions,
//! refusals, an outcome, and a trace's facts. An expectation compares its text with
//! these, whitespace left out.
//! Under `TRACE=1`, the facts a report is made of, what a group waits for and
//! its C, as a traced order prints them.

use std::collections::BTreeSet;
use std::fmt::Write;

use crate::check::{Outcome, Refusal};
use crate::class::{
    Apart, Cause, ClassId, Conflict, Conflicted, NamedConflict, NamedFact, Unified, Union,
};
use crate::coerce::{CoerceError, Coercion, Rank, Structural};
use crate::decl::DeclId;
use crate::dep::{Edge, Vertex};
use crate::node::{
    AsReference, DecidedBy, Decision, LendTarget, Narrowed, Narrowing, NodeKind, Reading, ValueRead,
};
use crate::record::Recording;
use crate::relation::{Named, Recorder};
use crate::solve::{State, Step};
use crate::split::Split;
use crate::trace::Report;
use crate::ty::{HasTerm, Mutability, Name, Term};
use crate::fallback::FellBack;
use crate::value::{Head, Sources, Took};

use super::parse::Phase;

/// A name, in backquotes where a case must write it so: unless it is words
/// joined by `::`.
pub fn name(text: &str) -> String {
    let word = |piece: &str| {
        !piece.is_empty()
            && piece
                .chars()
                .all(|c| c.is_alphanumeric() || c == '_' || c == '?')
    };
    if text.split("::").all(word) {
        text.to_string()
    } else {
        format!("`{text}`")
    }
}

pub fn decl(id: &DeclId) -> String {
    format!("{}#{}", name(id.name.as_str()), id.overload)
}

fn reference(mutability: Mutability) -> &'static str {
    match mutability {
        Mutability::Shared => "&",
        Mutability::Exclusive => "&mut",
    }
}

fn through(mutability: Option<Mutability>) -> &'static str {
    mutability.map_or("_", reference)
}

fn target(target: &LendTarget) -> String {
    match target {
        LendTarget::Place => "place".to_string(),
        LendTarget::Through(mutability) => format!("through {}", through(*mutability)),
    }
}

fn listed<I>(items: I) -> String
where
    I: IntoIterator<Item = String>,
{
    items.into_iter().collect::<Vec<_>>().join(", ")
}

pub fn decision(decision: &Decision) -> String {
    match decision {
        Decision::Use(id) => decl(id),
        Decision::Never => "never".to_string(),
        Decision::Lend {
            target: lent,
            mutability,
        } => format!("lend {} {}", reference(*mutability), target(lent)),
        Decision::Applied => "applied".to_string(),
        Decision::AppliedToTarget(mutability) => {
            format!("applied through {}", through(*mutability))
        }
        Decision::Read => "read".to_string(),
        Decision::BindLend {
            target: lent,
            mutability,
        } => format!("bind lend {} {}", reference(*mutability), target(lent)),
        Decision::Calls(lambdas) => format!(
            "calls {{{}}}",
            listed(lambdas.iter().map(|lambda| name(lambda.as_str())))
        ),
    }
}

pub fn coercion(coercion: &Result<Coercion, CoerceError>) -> String {
    match coercion {
        Ok(coercion) => ok_coercion(coercion),
        Err(CoerceError::None) => "none".to_string(),
        Err(CoerceError::Compositions(compositions)) => format!(
            "compositions [{}]",
            listed(compositions.iter().map(ok_coercion))
        ),
        Err(CoerceError::Conversions(ids)) => {
            format!("conversions [{}]", listed(ids.iter().map(decl)))
        }
    }
}

fn ok_coercion(coercion: &Coercion) -> String {
    match coercion {
        Coercion::Identity => "identity".to_string(),
        Coercion::Undecided => "undecided".to_string(),
        Coercion::Never => "never".to_string(),
        Coercion::Conversion(id) => format!("conversion {}", decl(id)),
        Coercion::Composition {
            reborrow,
            casts,
            structural,
        } => {
            let mut steps = Vec::new();
            if *reborrow {
                steps.push("reborrow".to_string());
            }
            steps.extend(casts.iter().map(|cast| format!("cast {}", decl(cast))));
            if let Some(structural) = structural {
                steps.push(match structural {
                    Structural::Shape { .. } => "shape".to_string(),
                });
            }
            steps.join(" + ")
        }
    }
}

pub fn refusal(refusal: &Refusal) -> String {
    match refusal {
        Refusal::Conflict { at, conflict } => format!("conflict at {at:?}: {conflict}"),
        Refusal::NoJoin(class) => format!("no_join {}", name(&class.to_string())),
        Refusal::Infinite(class) => format!("infinite {}", name(&class.to_string())),
        Refusal::NoValue(class) => format!("no_value {}", name(&class.to_string())),
        Refusal::Undeclared(what) => format!("undeclared: {what}"),
    }
}

/// A step, as an order writes it, its class named before it is taken.
pub fn step(state: &State, step: &Step) -> String {
    match step {
        Step::Solve(root) => format!("solve {}", name(&state.class_name(*root).to_string())),
        Step::Decide(node) => format!("decide {}", name(node.as_str())),
    }
}

/// Text compared with whitespace left out.
pub fn same(a: &str, b: &str) -> bool {
    let squeezed = |text: &str| {
        text.chars()
            .filter(|c| !c.is_whitespace())
            .collect::<String>()
    };
    squeezed(a) == squeezed(b)
}

/// A phase's report as a trace writes it:
/// its line, none where it is only what it recorded, and the relations it
/// recorded, each by its place in the log, named in `state`. A report that
/// recorded nothing and has no other line writes nothing.
pub struct Rendered {
    pub phase: Phase,
    pub line: Option<String>,
    pub recorded: Vec<(usize, String)>,
}

fn relation(state: &State, at: usize) -> String {
    Named {
        relation: &state.log[at].relation,
        arena: &state.arena,
    }
    .to_string()
}

/// A class named as the state names its equivalence class.
fn class_of(state: &State, id: ClassId) -> String {
    name(&state.class_name(id).to_string())
}

fn head(head: &Head) -> String {
    let members: Vec<String> = head.members.iter().map(|m| format!("{m:?}")).collect();
    if members.is_empty() {
        format!("{:?}", head.constructor)
    } else {
        format!("{:?} {{{}}}", head.constructor, listed(members))
    }
}

fn took(took: &Took) -> String {
    match took {
        Took::Held => "held".to_string(),
        Took::Joined { head: joined, .. } => head(joined),
        Took::Nothing => "nothing".to_string(),
        Took::NoJoin(_) => "no join".to_string(),
    }
}

pub fn report(state: &State, report: &Report) -> Rendered {
    let class = |id| class_of(state, id);
    let unless_empty = |recorded: &Recording, line: String| (!recorded.is_empty()).then_some(line);
    let (phase, line, recorded): (Phase, Option<String>, Option<&Recording>) = match report {
        Report::Recorded(recorded) => (Phase::Record, None, Some(recorded)),
        Report::Linked(linked) => (
            Phase::Unify,
            Some(format!(
                "link {}: {}",
                class(linked.container),
                class(linked.element)
            )),
            Some(&linked.recorded),
        ),
        Report::Merged(recorded) => (
            Phase::Join,
            unless_empty(recorded, "merge cycles".to_string()),
            Some(recorded),
        ),
        Report::Infinite { class: id, .. } => (
            Phase::Join,
            Some(format!("join {}: infinite", class(*id))),
            None,
        ),
        Report::Joined { class: id, joined } => (
            Phase::Join,
            Some(format!("join {}: {}", class(*id), took(&joined.took))),
            None,
        ),
        Report::Ranged { bound, recorded } => (
            Phase::Join,
            Some(format!("range #{bound} {}", relation(state, *bound))),
            Some(recorded),
        ),
        Report::Narrowed {
            row,
            class: id,
            recorded,
        } => (
            Phase::Join,
            Some(format!("row {} {}", row.name(), class(*id))),
            Some(recorded),
        ),
        Report::Split { at, split } => {
            let (what, recorded) = match split {
                Split::OneClass => ("one class", None),
                Split::Parts(recorded) => ("parts", Some(recorded)),
                Split::Unsolved => ("unsolved", None),
                Split::NoParts => ("no parts", None),
            };
            (
                Phase::Split,
                Some(format!("split #{at} {}: {what}", relation(state, *at))),
                recorded,
            )
        }
        Report::Common(recorded) => (
            Phase::Node,
            unless_empty(recorded, "common".to_string()),
            Some(recorded),
        ),
        Report::Decided(decided) => (
            Phase::Node,
            Some(format!(
                "decide {}: {}",
                name(decided.node.as_str()),
                decision(&decided.decision)
            )),
            Some(&decided.recorded),
        ),
        Report::FellBack(fell) => {
            let default = fell.default.clone().map(|_, part| state.class_name(part));
            (
                Phase::Fallback,
                Some(format!("default {}: {default}", class(fell.class))),
                Some(&fell.recorded),
            )
        }
    };
    Rendered {
        phase,
        line,
        recorded: recorded
            .into_iter()
            .flat_map(Recording::places)
            .map(|at| (at, relation(state, at)))
            .collect(),
    }
}

/// An outcome as the golden file keeps it: acceptance, every value and
/// decision, the classes unified with another, every check and refusal.
pub fn outcome(outcome: &Outcome) -> String {
    let mut text = String::new();
    let accepted = if outcome.accepted {
        "accepted"
    } else {
        "refused"
    };
    writeln!(text, "  {accepted}").expect("writing to a string");
    for (class, value) in &outcome.values {
        let value = value
            .as_ref()
            .map_or("none".to_string(), ToString::to_string);
        writeln!(text, "  value {}: {value}", name(&class.to_string()))
            .expect("writing to a string");
    }
    for (node, decided) in &outcome.decisions {
        let decided = decided.as_ref().map_or("undecided".to_string(), decision);
        writeln!(text, "  decision {}: {decided}", name(node.as_str()))
            .expect("writing to a string");
    }
    for set in &outcome.partition {
        if set.len() > 1 {
            let names = listed(set.iter().map(|class| name(&class.to_string())));
            writeln!(text, "  partition {{{names}}}").expect("writing to a string");
        }
    }
    for check in &outcome.checks {
        let unnamed = if check.unnamed.is_empty() {
            String::new()
        } else {
            let lambdas = listed(check.unnamed.iter().map(|lambda| name(lambda.as_str())));
            format!("  unnamed: {{{lambdas}}}")
        };
        writeln!(
            text,
            "  check {}: {:?}{unnamed}",
            name(&check.relation),
            check.coercion
        )
        .expect("writing to a string");
    }
    for refused in &outcome.refusals {
        writeln!(text, "  refusal {}", refusal(refused)).expect("writing to a string");
    }
    text
}

/// A node or a class named in a final state's I\* violations.
pub fn istar(kind: crate::istar::Kind, node: &Name) -> String {
    let kind = match kind {
        crate::istar::Kind::Completeness => "completeness",
        crate::istar::Kind::Soundness => "soundness",
    };
    format!("{kind} {}", name(node.as_str()))
}

/// The lines a traced order prints of one thing (`TRACE=1`), each with its
/// depth below the line it is a fact of.
#[derive(Default)]
pub struct Lines(pub Vec<(usize, String)>);

impl Lines {
    fn push(&mut self, depth: usize, text: String) {
        self.0.push((depth, text));
    }

    fn nest(&mut self, depth: usize, lines: Lines) {
        self.0
            .extend(lines.0.into_iter().map(|(at, text)| (depth + at, text)));
    }
}

/// A line's attributes, each `key: value` after two spaces; one with no value
/// is none and not written.
fn attributes<'a, I>(line: String, attributes: I) -> String
where
    I: IntoIterator<Item = (&'a str, Option<String>)>,
{
    attributes
        .into_iter()
        .filter_map(|(key, value)| value.map(|value| format!("  {key}: {value}")))
        .fold(line, |line, attribute| line + &attribute)
}

/// Items listed, none where there are none.
fn some_listed<I>(items: I) -> Option<String>
where
    I: IntoIterator<Item = String>,
{
    let listed = listed(items);
    (!listed.is_empty()).then_some(listed)
}

/// A class by its own name, not its equivalence class's: what a step's
/// unifying names stays as it was when a later union of the step merges it.
fn member(state: &State, id: ClassId) -> String {
    name(&state.arena.name(id).to_string())
}

/// The members of a class, each by its own name.
fn members(state: &State, members: &BTreeSet<ClassId>) -> String {
    let named: BTreeSet<String> = members.iter().map(|&id| member(state, id)).collect();
    format!("{{{}}}", listed(named))
}

/// A class as it was before a union, by the least of its members' names.
fn side(state: &State, members: &BTreeSet<ClassId>) -> String {
    let least = members
        .iter()
        .map(|&member| state.arena.name(member))
        .min()
        .expect("a class has a member");
    name(&least.to_string())
}

/// A kind, each part by its own name, as a relation writes it.
fn kind_of(state: &State, kind: &Term<ClassId>) -> String {
    kind.clone()
        .map(|_, part| state.arena.name(part).clone())
        .to_string()
}

/// Places in the log, a run of consecutive places written as its ends.
fn places<I>(places: I) -> Option<String>
where
    I: IntoIterator<Item = usize>,
{
    let mut runs: Vec<(usize, usize)> = Vec::new();
    for at in places {
        match runs.last_mut() {
            Some((_, end)) if *end + 1 == at => *end = at,
            _ => runs.push((at, at)),
        }
    }
    some_listed(runs.into_iter().map(|(start, end)| {
        if start == end {
            format!("#{start}")
        } else {
            format!("#{start}-#{end}")
        }
    }))
}

fn nodes<'a, I>(nodes: I) -> Option<String>
where
    I: IntoIterator<Item = &'a Name>,
{
    some_listed(nodes.into_iter().map(|node| name(node.as_str())))
}

/// Why two classes are unified, `at` the place of the relation the unifying
/// took, where it took one.
fn cause(state: &State, cause: &Cause, unions: &[Union], at: Option<usize>) -> String {
    match cause {
        Cause::Given => at.map_or("given".to_string(), |at| format!("#{at}")),
        Cause::Congruence { parent, part } => {
            let [a, b] = &unions[*parent].sides;
            format!(
                "congruence {part:?} of {} + {}",
                side(state, a),
                side(state, b)
            )
        }
        Cause::KindGiven { class, part } => format!("kind {} at {part:?}", member(state, *class)),
        Cause::Fact { class, fact } => {
            let fact = NamedFact {
                fact,
                arena: &state.arena,
            };
            format!("fact {}: {fact}", member(state, *class))
        }
    }
}

fn conflict(state: &State, conflict: &Conflict) -> String {
    NamedConflict {
        conflict,
        arena: &state.arena,
    }
    .to_string()
}

/// Each union unifying made, and the conflict that stopped it.
fn unified(state: &State, unified: &Unified, at: Option<usize>) -> Lines {
    let mut lines = Lines::default();
    for union in &unified.unions {
        let [a, b] = &union.sides;
        let merged: BTreeSet<ClassId> = a.union(b).copied().collect();
        let line = format!(
            "union {} + {}: {}",
            side(state, a),
            side(state, b),
            members(state, &merged)
        );
        let by = cause(state, &union.cause, &unified.unions, at);
        let kind = union.kind.as_ref().map(|kind| kind_of(state, kind));
        lines.push(0, attributes(line, [("by", Some(by)), ("kind", kind)]));
    }
    match &unified.conflict {
        Some(Conflicted::Apart(apart)) => {
            let Apart {
                cause: why,
                sides: [a, b],
                kinds,
                conflict: met,
            } = &**apart;
            let line = format!(
                "conflict {} + {}: {}",
                side(state, a),
                side(state, b),
                conflict(state, met)
            );
            let by = cause(state, why, &unified.unions, at);
            let kinds = kinds.iter().any(Option::is_some).then(|| {
                let [a, b] = kinds.each_ref().map(|kind| {
                    kind.as_ref()
                        .map_or("none".to_string(), |kind| kind_of(state, kind))
                });
                format!("{a} vs {b}")
            });
            lines.push(0, attributes(line, [("by", Some(by)), ("kinds", kinds)]));
        }
        Some(Conflicted::Holds {
            members: held,
            conflict: met,
        }) => lines.push(
            0,
            format!(
                "conflict {}: {}",
                members(state, held),
                conflict(state, met)
            ),
        ),
        None => {}
    }
    lines
}

/// What recorded a relation, a copy's by what recorded its original in the
/// group instantiated.
fn recorder(state: &State, recorder: &Recorder) -> String {
    match recorder {
        Recorder::Origin => "origin".to_string(),
        Recorder::Text(item) => format!("text {item}"),
        Recorder::Node(node) => format!("node {}", name(node.as_str())),
        Recorder::Common(node) => format!("common {}", name(node.as_str())),
        Recorder::Split(class) => format!("split {}", class_of(state, *class)),
        Recorder::Cycle => "cycle".to_string(),
        Recorder::FunctionType { from, to } => {
            format!(
                "function type {} ~> {}",
                class_of(state, *from),
                class_of(state, *to)
            )
        }
        Recorder::Check => "check".to_string(),
        Recorder::Row(row) => format!("row {}", row.name()),
        Recorder::Fallback => "fallback".to_string(),
        Recorder::Instance { at, of, .. } => {
            let made = state
                .instances
                .iter()
                .find(|made| made.at == *at)
                .expect("a copy's instance is the state's");
            let from = state
                .solved
                .get(&made.group)
                .expect("the group an instance copies is solved before it");
            format!(
                "instance {} of {}",
                name(at.as_str()),
                self::recorder(from, of)
            )
        }
    }
}

/// Each instance made, and each relation recorded with its recorder and what
/// unification did as the classes took it.
fn recording(state: &State, recorded: &Recording) -> Lines {
    let mut lines = Lines::default();
    for made in &recorded.instances {
        let line = format!(
            "instance {}: C of {}",
            name(made.at.as_str()),
            name(made.group.as_str())
        );
        let by = recorder(state, &made.by);
        let records = places(made.recorded.iter().copied());
        lines.push(
            0,
            attributes(line, [("by", Some(by)), ("records", records)]),
        );
    }
    for entry in &recorded.entries {
        let line = format!("record #{} {}", entry.at, relation(state, entry.at));
        let by = recorder(state, &state.log[entry.at].recorder);
        lines.push(0, attributes(line, [("by", Some(by))]));
        lines.nest(
            1,
            unified(state, &entry.unification.unified, Some(entry.at)),
        );
    }
    lines
}

/// What an equivalence class's value is taken from.
fn sources(state: &State, sources: &Sources) -> Lines {
    let mut lines = Lines::default();
    if let Some(equations) = places(sources.equations.iter().copied()) {
        lines.push(0, format!("equations: {equations}"));
    }
    for (lower, given) in &sources.lowers {
        lines.push(
            0,
            format!("lower {}: {}", class_of(state, *lower), head(given)),
        );
    }
    for (lower, given) in &sources.part_lowers {
        lines.push(
            0,
            format!("part lower {}: {}", class_of(state, *lower), head(given)),
        );
    }
    for given in &sources.facts {
        lines.push(0, format!("fact: {}", head(given)));
    }
    let left_out = sources.left_out.iter().map(|&lower| class_of(state, lower));
    if let Some(left_out) = some_listed(left_out) {
        lines.push(0, format!("left out: {left_out}"));
    }
    lines
}

/// A position a use reads: an argument by its index, the result after them.
fn position(position: usize, args: usize) -> String {
    if position < args {
        format!("arg{position}")
    } else {
        "result".to_string()
    }
}

/// The candidates a use's read narrows at each position in turn.
fn narrowed(stages: &[Narrowed], args: usize) -> Lines {
    let mut lines = Lines::default();
    for stage in stages {
        let ranks = stage.ranks.iter().map(|(id, rank)| match rank {
            Some(rank) if *rank == Rank::empty() => format!("{} empty", decl(id)),
            Some(rank) => format!("{} {rank:?}", decl(id)),
            None => format!("{} none", decl(id)),
        });
        lines.push(
            0,
            format!(
                "narrow {} {}: {} -> {}",
                position(stage.position, args),
                stage.value,
                listed(ranks),
                some_listed(stage.remaining.iter().map(decl)).unwrap_or("none".to_string())
            ),
        );
    }
    lines
}

fn narrowing(state: &State, narrowing: &Narrowing, args: usize) -> Lines {
    let mut lines = Lines::default();
    if let Some(candidates) = some_listed(narrowing.candidates.iter().map(decl)) {
        lines.push(0, format!("candidates: {candidates}"));
    }
    for (at, class, value) in &narrowing.values {
        lines.push(
            0,
            format!(
                "read {} {}: {value}",
                position(*at, args),
                class_of(state, *class)
            ),
        );
    }
    lines.nest(0, narrowed(&narrowing.stages, args));
    lines
}

/// The value read at a class of a lend, a projection or a binding, and what
/// it reads as.
fn as_reference(state: &State, at: &str, read: &AsReference) -> String {
    let reads_as = match read.reads_as {
        Some(Some(mutability)) => reference(mutability),
        Some(None) => "reference",
        None if read.value.decided().is_some() => "no reference",
        None => "none",
    };
    let line = format!("read {at} {}: {}", class_of(state, read.class), read.value);
    attributes(line, [("as", Some(reads_as.to_string()))])
}

/// The value a node read when it was decided.
fn decided_by(state: &State, node: &Name, by: &DecidedBy) -> Lines {
    let broke = [("broken by", nodes(&by.broken_on))];
    value_read(state, node, &by.value, broke)
}

/// The value a node reads: the nodes and relations it leaves out, with the
/// dependencies `broke` names, and what it reads at each position.
fn value_read(
    state: &State,
    node: &Name,
    value: &ValueRead,
    broke: [(&str, Option<String>); 1],
) -> Lines {
    let mut lines = Lines::default();
    let facts = [("exclude", nodes(&value.left_out))]
        .into_iter()
        .chain(broke)
        .chain([("left out", places(value.relations.iter().copied()))]);
    for (key, listed) in facts {
        if let Some(listed) = listed {
            lines.push(0, format!("{key}: {listed}"));
        }
    }
    let kind = &state.nodes[node].kind;
    match &value.reading {
        Reading::Nothing => {}
        Reading::Use(narrowed) => {
            let NodeKind::Use { args, .. } = kind else {
                unreachable!("a use's narrowing is a use's")
            };
            lines.nest(0, narrowing(state, narrowed, args.len()));
        }
        Reading::Place(read) => {
            let at = match kind {
                NodeKind::Lend { .. } => "place",
                NodeKind::Project { .. } => "operand",
                NodeKind::Use { .. } | NodeKind::CallValue { .. } | NodeKind::Bind { .. } => {
                    unreachable!("a place is read by a lend or a projection")
                }
            };
            lines.push(0, as_reference(state, at, read));
        }
        Reading::Binding { position, .. } => {
            lines.push(0, as_reference(state, "position", position))
        }
        Reading::Values {
            callee,
            value,
            lambdas,
        } => {
            let line = format!("read callee {}: {value}", class_of(state, *callee));
            lines.push(0, attributes(line, [("values", nodes(lambdas))]));
        }
    }
    lines
}

/// A phase's report as a traced order prints it: its line, then the facts it
/// is made of and what it recorded (`TRACE=1`).
pub fn traced(state: &State, report: &Report) -> Lines {
    let rendered = self::report(state, report);
    let mut detail = Lines::default();
    let recorded = match report {
        Report::Recorded(recorded)
        | Report::Merged(recorded)
        | Report::Common(recorded)
        | Report::Ranged { recorded, .. }
        | Report::Narrowed { recorded, .. }
        | Report::FellBack(FellBack { recorded, .. })
        | Report::Split {
            split: Split::Parts(recorded),
            ..
        } => Some(recorded),
        Report::Linked(linked) => Some(&linked.recorded),
        Report::Decided(decided) => {
            if let Some(by) = &decided.read {
                detail.nest(0, decided_by(state, &decided.node, by));
            }
            Some(&decided.recorded)
        }
        Report::Joined { joined, .. } => {
            detail.nest(0, sources(state, &joined.sources));
            if let Took::Joined { unified: made, .. } = &joined.took {
                detail.nest(0, self::unified(state, made, None));
            }
            None
        }
        Report::Infinite { .. } | Report::Split { .. } => None,
    };
    if let Some(recorded) = recorded {
        detail.nest(0, recording(state, recorded));
    }
    let mut lines = Lines::default();
    match rendered.line {
        Some(line) => {
            let passed = match report {
                Report::Decided(decided) => match &decided.read {
                    Some(DecidedBy {
                        value:
                            ValueRead {
                                reading: Reading::Binding { passed, .. },
                                ..
                            },
                        ..
                    }) => Some(through(*passed).to_string()),
                    _ => None,
                },
                _ => None,
            };
            lines.push(0, attributes(line, [("passed", passed)]));
            lines.nest(1, detail);
        }
        None => lines.nest(0, detail),
    }
    lines
}

/// The unbroken dependencies of `from` not yet met.
fn waits_for(state: &State, from: &Vertex) -> String {
    dependencies(state, state.waits_for(from))
}

fn dependencies(state: &State, waits: Vec<(Vertex, Edge)>) -> String {
    let waits = waits.into_iter().map(|(to, edge)| match (to, edge) {
        (Vertex::Class(lower), Edge::Lower) => format!("lower {}", class_of(state, lower)),
        (Vertex::Class(taking), Edge::Taking) => format!("taking {}", class_of(state, taking)),
        (Vertex::Class(owner), Edge::Part) => format!("part of {}", class_of(state, owner)),
        (Vertex::Node(node), Edge::Records) => format!("records {}", name(node.as_str())),
        (Vertex::Outside, Edge::Chosen) => "chosen".to_string(),
        (Vertex::Node(node), Edge::Reads) => format!("reads {}", name(node.as_str())),
        (Vertex::Outside, Edge::Reads) => "reads outside".to_string(),
        (to, edge) => unreachable!("the dependency graph has no edge {edge:?} to {to:?}"),
    });
    some_listed(waits).unwrap_or("none".to_string())
}

/// What a binding group waits for: each equivalence class not solved, and
/// each node not decided, with the unbroken dependencies of it not yet met,
/// and a node that waits for nothing with the value it reads, which decides
/// nothing.
pub fn waiting(state: &State) -> Lines {
    let mut lines = Lines::default();
    for root in state.classes.roots() {
        if state.is_solved(root) {
            continue;
        }
        let yes = |holds: bool| holds.then(|| "yes".to_string());
        let (no_value, enclosing) = (state.has_no_value(root), state.is_foreign(root));
        let wait = (!no_value && !enclosing).then(|| waits_for(state, &Vertex::Class(root)));
        let line = format!("unsolved {}", class_of(state, root));
        let facts = [
            ("universal", yes(state.is_universal(root))),
            ("no value", yes(no_value)),
            ("enclosing", yes(!no_value && enclosing)),
            ("wait", wait),
        ];
        lines.push(0, attributes(line, facts));
    }
    for (node, held) in &state.nodes {
        if held.decision.is_some() {
            continue;
        }
        let waits = state.waits_for(&Vertex::Node(node.clone()));
        let waits_for_nothing = waits.is_empty();
        let line = format!("undecided {}", name(node.as_str()));
        let wait = dependencies(state, waits);
        lines.push(0, attributes(line, [("wait", Some(wait))]));
        if waits_for_nothing {
            let none = [("broken by", None)];
            let read = state.read_now(node);
            lines.nest(1, value_read(state, node, &read, none));
        }
    }
    lines
}

/// What a binding group leaves in C at its fixpoint.
pub fn c(state: &State) -> Lines {
    let mut lines = Lines::default();
    lines.push(0, format!("C {}", name(state.group.as_str())));
    let c = state.c();
    if let Some(classes) = some_listed(c.iter().map(|&root| class_of(state, root))) {
        lines.push(1, format!("classes: {classes}"));
    }
    let undecided = state
        .nodes
        .iter()
        .filter(|(_, held)| held.decision.is_none())
        .map(|(node, _)| node);
    if let Some(undecided) = nodes(undecided) {
        lines.push(1, format!("nodes: {undecided}"));
    }
    for at in state.relations_on(&c) {
        let line = format!("relation #{at} {}", relation(state, at));
        let by = recorder(state, &state.log[at].recorder);
        lines.push(1, attributes(line, [("by", Some(by))]));
    }
    lines
}
