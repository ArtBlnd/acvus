//! What the model gives, written as a case writes it: decisions, coercions,
//! refusals, an outcome, and a trace's events. An expectation compares its text with
//! these, whitespace left out.
//! Under `TRACE=1`, the events a report is made of, what a group waits for and
//! its C, as a traced order prints them.

use std::collections::BTreeSet;
use std::fmt::Write;

use crate::check::{Outcome, Refusal};
use crate::class::{
    Apart, Cause, ClassId, Conflict, Conflicted, NamedConflict, NamedStructureConstraint, Unified,
    Union,
};
use crate::coerce::{CoerceError, Coercion, Rank, Structural};
use crate::decl::DeclId;
use crate::decomposition::Decomposition;
use crate::dep::{Edge, Vertex};
use crate::disjunction::{
    AsReference, DecidedBy, Decision, Disjunction, LendTarget, Narrowed, Narrowing, Reading,
    ValueRead,
};
use crate::record::Recording;
use crate::relation::{Named, Recorder};
use crate::solve::{State, Step};
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
        Step::Decide(disjunction) => format!("decide {}", name(disjunction.as_str())),
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
        Report::DomainGiven { bound, recorded } => (
            Phase::Join,
            Some(format!("domain #{bound} {}", relation(state, *bound))),
            Some(recorded),
        ),
        Report::Decomposed { at, decomposition } => {
            let (what, recorded) = match decomposition {
                Decomposition::OneClass => ("one class", None),
                Decomposition::Parts(recorded) => ("parts", Some(recorded)),
                Decomposition::Unsolved => ("unsolved", None),
                Decomposition::NoParts => ("no parts", None),
            };
            (
                Phase::Decomposition,
                Some(format!(
                    "decomposition #{at} {}: {what}",
                    relation(state, *at)
                )),
                recorded,
            )
        }
        Report::Generalized(recorded) => (
            Phase::Disjunction,
            unless_empty(recorded, "generalization".to_string()),
            Some(recorded),
        ),
        Report::Decided(decided) => (
            Phase::Disjunction,
            Some(format!(
                "decide {}: {}",
                name(decided.disjunction.as_str()),
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
    for (disjunction, decided) in &outcome.decisions {
        let decided = decided.as_ref().map_or("undecided".to_string(), decision);
        writeln!(text, "  decision {}: {decided}", name(disjunction.as_str()))
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

/// A disjunctive constraint or a class named in a final state's I\* violations.
pub fn istar(kind: crate::istar::Kind, disjunction: &Name) -> String {
    let kind = match kind {
        crate::istar::Kind::Completeness => "completeness",
        crate::istar::Kind::Soundness => "soundness",
    };
    format!("{kind} {}", name(disjunction.as_str()))
}

/// The lines a traced order prints of one thing (`TRACE=1`), each with its
/// depth below the line it is an event of.
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

/// A line with its attributes, each `key: value` after two spaces; one with no
/// value is none and not written.
fn with_attributes<'a, I>(line: String, attributes: I) -> String
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

/// A structure, each part by its own name, as a relation writes it.
fn structure_of(state: &State, structure: &Term<ClassId>) -> String {
    structure
        .clone()
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

fn names<'a, I>(names: I) -> Option<String>
where
    I: IntoIterator<Item = &'a Name>,
{
    some_listed(names.into_iter().map(|listed| name(listed.as_str())))
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
        Cause::StructureGiven { class, part } => {
            format!("structure {} at {part:?}", member(state, *class))
        }
        Cause::StructureConstraint {
            class,
            structure_constraint,
        } => {
            let structure_constraint = NamedStructureConstraint {
                structure_constraint,
                arena: &state.arena,
            };
            format!(
                "structure constraint {}: {structure_constraint}",
                member(state, *class)
            )
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
        let structure = union
            .structure
            .as_ref()
            .map(|structure| structure_of(state, structure));
        lines.push(
            0,
            with_attributes(line, [("by", Some(by)), ("structure", structure)]),
        );
    }
    match &unified.conflict {
        Some(Conflicted::Apart(apart)) => {
            let Apart {
                cause: why,
                sides: [a, b],
                structures,
                conflict: found,
            } = &**apart;
            let line = format!(
                "conflict {} + {}: {}",
                side(state, a),
                side(state, b),
                conflict(state, found)
            );
            let by = cause(state, why, &unified.unions, at);
            let structures = structures.iter().any(Option::is_some).then(|| {
                let [a, b] = structures.each_ref().map(|structure| {
                    structure.as_ref().map_or("none".to_string(), |structure| {
                        structure_of(state, structure)
                    })
                });
                format!("{a} vs {b}")
            });
            lines.push(
                0,
                with_attributes(line, [("by", Some(by)), ("structures", structures)]),
            );
        }
        Some(Conflicted::Holds {
            members: held,
            conflict: found,
        }) => lines.push(
            0,
            format!(
                "conflict {}: {}",
                members(state, held),
                conflict(state, found)
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
        Recorder::Disjunction(disjunction) => {
            format!("disjunctive constraint {}", name(disjunction.as_str()))
        }
        Recorder::Generalization(disjunction) => {
            format!("generalization {}", name(disjunction.as_str()))
        }
        Recorder::Decomposition(class) => format!("decomposition {}", class_of(state, *class)),
        Recorder::Cycle => "cycle".to_string(),
        Recorder::FunctionType { from, to } => {
            format!(
                "function type {} ~> {}",
                class_of(state, *from),
                class_of(state, *to)
            )
        }
        Recorder::Check => "check".to_string(),
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
            with_attributes(line, [("by", Some(by)), ("records", records)]),
        );
    }
    for entry in &recorded.entries {
        let line = format!("record #{} {}", entry.at, relation(state, entry.at));
        let by = recorder(state, &state.log[entry.at].recorder);
        lines.push(0, with_attributes(line, [("by", Some(by))]));
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
    for given in &sources.structure_constraints {
        lines.push(0, format!("structure constraint: {}", head(given)));
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
    with_attributes(line, [("as", Some(reads_as.to_string()))])
}

/// The value a disjunctive constraint read when it was decided.
fn decided_by(state: &State, disjunction: &Name, by: &DecidedBy) -> Lines {
    let broke = [("broken by", names(&by.broken_on))];
    value_read(state, disjunction, &by.value, broke)
}

/// The value a disjunctive constraint reads: the disjunctive constraints and relations it leaves out, with the
/// dependencies `broke` names, and what it reads at each position.
fn value_read(
    state: &State,
    disjunction: &Name,
    value: &ValueRead,
    broke: [(&str, Option<String>); 1],
) -> Lines {
    let mut lines = Lines::default();
    let attributes = [("exclude", names(&value.left_out))]
        .into_iter()
        .chain(broke)
        .chain([("left out", places(value.relations.iter().copied()))]);
    for (key, listed) in attributes {
        if let Some(listed) = listed {
            lines.push(0, format!("{key}: {listed}"));
        }
    }
    let kind = &state.disjunctions[disjunction].kind;
    match &value.reading {
        Reading::Nothing => {}
        Reading::Use(narrowed) => {
            let Disjunction::Use { args, .. } = kind else {
                unreachable!("a use's narrowing is a use's")
            };
            lines.nest(0, narrowing(state, narrowed, args.len()));
        }
        Reading::Place(read) => {
            let at = match kind {
                Disjunction::Lend { .. } => "place",
                Disjunction::Project { .. } => "operand",
                Disjunction::Use { .. }
                | Disjunction::CallValue { .. }
                | Disjunction::Bind { .. } => {
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
            lines.push(0, with_attributes(line, [("values", names(lambdas))]));
        }
    }
    lines
}

/// A phase's report as a traced order prints it: its line, then the events it
/// is made of and what it recorded (`TRACE=1`).
pub fn traced(state: &State, report: &Report) -> Lines {
    let rendered = self::report(state, report);
    let mut detail = Lines::default();
    let recorded = match report {
        Report::Recorded(recorded)
        | Report::Merged(recorded)
        | Report::Generalized(recorded)
        | Report::DomainGiven { recorded, .. }
        | Report::FellBack(FellBack { recorded, .. })
        | Report::Decomposed {
            decomposition: Decomposition::Parts(recorded),
            ..
        } => Some(recorded),
        Report::Linked(linked) => Some(&linked.recorded),
        Report::Decided(decided) => {
            if let Some(by) = &decided.read {
                detail.nest(0, decided_by(state, &decided.disjunction, by));
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
        Report::Infinite { .. } | Report::Decomposed { .. } => None,
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
            lines.push(0, with_attributes(line, [("passed", passed)]));
            lines.nest(1, detail);
        }
        None => lines.nest(0, detail),
    }
    lines
}

/// The unbroken dependencies of `from` not yet satisfied.
fn waits_for(state: &State, from: &Vertex) -> String {
    dependencies(state, state.waits_for(from))
}

fn dependencies(state: &State, waits: Vec<(Vertex, Edge)>) -> String {
    let waits = waits.into_iter().map(|(to, edge)| match (to, edge) {
        (Vertex::Class(lower), Edge::Lower) => format!("lower {}", class_of(state, lower)),
        (Vertex::Class(taking), Edge::Taking) => format!("taking {}", class_of(state, taking)),
        (Vertex::Class(owner), Edge::Part) => format!("part of {}", class_of(state, owner)),
        (Vertex::Disjunction(disjunction), Edge::Records) => {
            format!("records {}", name(disjunction.as_str()))
        }
        (Vertex::Outside, Edge::Chosen) => "chosen".to_string(),
        (Vertex::Disjunction(disjunction), Edge::Reads) => {
            format!("reads {}", name(disjunction.as_str()))
        }
        (Vertex::Outside, Edge::Reads) => "reads outside".to_string(),
        (to, edge) => unreachable!("the dependency graph has no edge {edge:?} to {to:?}"),
    });
    some_listed(waits).unwrap_or("none".to_string())
}

/// What a binding group waits for: each equivalence class not solved, and
/// each disjunctive constraint not decided, with the unbroken dependencies of it not yet satisfied,
/// and a disjunctive constraint that waits for nothing with the value it reads, which decides
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
        let attributes = [
            ("universal", yes(state.is_universal(root))),
            ("no value", yes(no_value)),
            ("enclosing", yes(!no_value && enclosing)),
            ("wait", wait),
        ];
        lines.push(0, with_attributes(line, attributes));
    }
    for (disjunction, held) in &state.disjunctions {
        if held.decision.is_some() {
            continue;
        }
        let waits = state.waits_for(&Vertex::Disjunction(disjunction.clone()));
        let waits_for_nothing = waits.is_empty();
        let line = format!("undecided {}", name(disjunction.as_str()));
        let wait = dependencies(state, waits);
        lines.push(0, with_attributes(line, [("wait", Some(wait))]));
        if waits_for_nothing {
            let none = [("broken by", None)];
            let read = state.read_now(disjunction);
            lines.nest(1, value_read(state, disjunction, &read, none));
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
        .disjunctions
        .iter()
        .filter(|(_, held)| held.decision.is_none())
        .map(|(disjunction, _)| disjunction);
    if let Some(undecided) = names(undecided) {
        lines.push(1, format!("disjunctive constraints: {undecided}"));
    }
    for at in state.relations_on(&c) {
        let line = format!("relation #{at} {}", relation(state, at));
        let by = recorder(state, &state.log[at].recorder);
        lines.push(1, with_attributes(line, [("by", Some(by))]));
    }
    lines
}
