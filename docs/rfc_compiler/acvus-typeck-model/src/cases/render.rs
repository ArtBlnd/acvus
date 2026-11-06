//! What the model gives, written as a case writes it: decisions, coercions,
//! refusals, an outcome, and a trace's events. An expectation compares its text with
//! these, whitespace left out.
//! Under `TRACE=1`, the events a report is made of, what a group leaves
//! undecided and its C, as a traced order prints them.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write;

use crate::check::{DecisionAtEnd, Outcome, Refusal};
use crate::choice::Chosen;
use crate::class::{
    Apart, Cause, ClassId, Conflict, Conflicted, NamedConflict, NamedStructureConstraint, Unified,
    Union,
};
use crate::coerce::{CoerceError, Coercion, Structural};
use crate::decl::DeclId;
use crate::decomposition::Decomposition;
use crate::disjunction::{Decision, LendTarget};
use crate::judgment::fallback::Fallen;
use crate::judgment::{FellBack, Judgment, Took};
use crate::record::Recording;
use crate::relation::{Item, Named, Recorder};
use crate::solve::State;
use crate::trace::Report;
use crate::ty::{Mutability, Name, Term};

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

fn target(target: &LendTarget) -> String {
    match target {
        LendTarget::Place => "place".to_string(),
        LendTarget::Through => "through".to_string(),
    }
}

fn listed<I>(items: I) -> String
where
    I: IntoIterator<Item = String>,
{
    items.into_iter().collect::<Vec<_>>().join(", ")
}

/// A disjunctive constraint as the outcome writes it once solving ends: its
/// decision; `undecided`; or `refused`, where a class it reads has no value
/// (D5).
pub fn decision_at_end(at_end: &DecisionAtEnd) -> String {
    match at_end {
        DecisionAtEnd::Decided(decided) => decision(decided),
        DecisionAtEnd::Undecided => "undecided".to_string(),
        DecisionAtEnd::RefusedWith => "refused".to_string(),
    }
}

pub fn decision(decision: &Decision) -> String {
    match decision {
        Decision::Lend {
            target: lent,
            mutability,
        } => format!("lend {} {}", reference(*mutability), target(lent)),
        Decision::Applied => "applied".to_string(),
        Decision::AppliedToTarget => "applied through".to_string(),
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

/// What a use became: `f#0`, or `never`.
pub fn chosen(chosen: &Chosen) -> String {
    match chosen {
        Chosen::Candidate(id) => decl(id),
        Chosen::Never => "never".to_string(),
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

/// A refusal: its cause, the type variables of its class, and the items on
/// it.
pub fn refusal(refusal: &Refusal) -> String {
    let members = listed(refusal.members.iter().map(|member| name(&member.to_string())));
    let items = listed(refusal.items.iter().map(|item| name(item)));
    format!("{} {{{members}}} by {{{items}}}", refusal.cause)
}

/// A judgment, as an order writes it, its class named before it is taken.
pub fn judgment(state: &State, judgment: &Judgment) -> String {
    match judgment {
        Judgment::Join(root) => format!("join {}", name(&state.class_name(*root).to_string())),
        Judgment::Decide(disjunction) => format!("decide {}", name(disjunction.as_str())),
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
/// recorded, each by its place in the record, named in `state`. A report that
/// recorded nothing and has no other line writes nothing.
pub struct Rendered {
    pub phase: Phase,
    pub line: Option<String>,
    pub recorded: Vec<(usize, String)>,
}

/// An item of the record: a relation with its classes' names, or a
/// disjunctive constraint by its name.
fn relation(state: &State, at: usize) -> String {
    match &state.classes.items()[at].item {
        Item::Relation(relation) => Named {
            relation,
            arena: &state.arena,
        }
        .to_string(),
        Item::Disjunction { name: disjunction, .. } => {
            format!("disjunctive constraint {}", name(disjunction.as_str()))
        }
        Item::Instance { at, group, .. } => {
            format!("instance {}: C of {}", name(at.as_str()), name(group.as_str()))
        }
        Item::Lambda { name: copy, of, .. } => {
            format!("lambda {} of {}", name(copy.as_str()), name(of.as_str()))
        }
        Item::Choice { name: chosen, .. } => format!("choice {}", name(chosen.as_str())),
    }
}

/// A class named as the state names its equivalence class.
fn class_of(state: &State, id: ClassId) -> String {
    name(&state.class_name(id).to_string())
}

pub fn report(state: &State, report: &Report) -> Rendered {
    let class = |id| class_of(state, id);
    let unless_empty = |recorded: &Recording, line: String| (!recorded.is_empty()).then_some(line);
    let (phase, line, recorded): (Phase, Option<String>, Option<&Recording>) = match report {
        Report::Recorded(recorded) => (Phase::Record, None, Some(recorded)),
        Report::Merged(recorded) => (
            Phase::Unify,
            unless_empty(recorded, "merge cycles".to_string()),
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
        Report::Joined { class: root, took } => {
            let (what, recorded) = match took {
                Took::Joined { head, recorded } => (format!("{:?}", head.constructor), Some(recorded)),
                Took::Contextual { head, recorded } => {
                    (format!("contextual {:?}", head.constructor), Some(recorded))
                }
                Took::NoJoin => ("no join".to_string(), None),
                Took::Contexts => ("contexts".to_string(), None),
                Took::Conflict => ("conflict".to_string(), None),
                Took::Infinite => ("infinite".to_string(), None),
            };
            (
                Phase::Join,
                Some(format!("join {}: {what}", class(*root))),
                recorded,
            )
        }
        Report::Decided {
            disjunction,
            decision: decided,
            recorded,
        } => (
            Phase::Disjunction,
            Some(format!(
                "decide {}: {}",
                name(disjunction.as_str()),
                decision(decided)
            )),
            Some(recorded),
        ),
        Report::Narrowed {
            choice,
            remaining,
            chosen: became,
            recorded,
        } => {
            let what = match became {
                Some(became) => self::chosen(became),
                None => format!("{{{}}}", listed(remaining.iter().map(decl))),
            };
            (
                Phase::Choice,
                Some(format!("narrow {}: {what}", name(choice.as_str()))),
                Some(recorded),
            )
        }
        Report::FellBack(fell) => {
            let line = match &fell.fallen {
                Fallen::Default(_) => {
                    let default = match &state.classes.held(fell.class).structure {
                        Some((structure, _)) => structure
                            .clone()
                            .map(|_, part| state.class_name(part))
                            .to_string(),
                        None => "none".to_string(),
                    };
                    format!("default {}: {default}", class(fell.class))
                }
                Fallen::NoType(_) => format!("fallback {}: conflict", class(fell.class)),
                Fallen::NoValue => format!("fallback {}: no_value", class(fell.class)),
            };
            (Phase::Fallback, Some(line), Some(&fell.recorded))
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
    // The disjunctive constraints and the uses, by their names, which are
    // apart.
    let decided: BTreeMap<&Name, String> = outcome
        .decisions
        .iter()
        .map(|(disjunction, at_end)| (disjunction, decision_at_end(at_end)))
        .chain(outcome.choices.iter().map(|(used, became)| {
            let shown = became
                .as_ref()
                .map_or_else(|| "undecided".to_string(), chosen);
            (used, shown)
        }))
        .collect();
    for (named, shown) in decided {
        writeln!(text, "  decision {}: {shown}", name(named.as_str()))
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

/// Places in the record, a run of consecutive places written as its ends.
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
/// took.
fn cause(state: &State, cause: &Cause, unions: &[Union], at: usize) -> String {
    match cause {
        Cause::Given => format!("#{at}"),
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
fn unified(state: &State, unified: &Unified, at: usize) -> Lines {
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
        Recorder::Choice(chosen) => format!("choice {}", name(chosen.as_str())),
        Recorder::Join(class) => format!("join {}", class_of(state, *class)),
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
                .instance(at)
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
        let by = recorder(state, &state.classes.items()[entry.at].recorder);
        lines.push(0, with_attributes(line, [("by", Some(by))]));
        lines.nest(1, unified(state, &entry.unification.unified, entry.at));
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
        | Report::FellBack(FellBack { recorded, .. })
        | Report::Decomposed {
            decomposition: Decomposition::Parts(recorded),
            ..
        } => Some(recorded),
        Report::Decided { recorded, .. } | Report::Narrowed { recorded, .. } => Some(recorded),
        Report::Joined {
            took: Took::Joined { recorded, .. } | Took::Contextual { recorded, .. },
            ..
        } => Some(recorded),
        Report::Joined { .. } | Report::Decomposed { .. } => None,
    };
    if let Some(recorded) = recorded {
        detail.nest(0, recording(state, recorded));
    }
    let mut lines = Lines::default();
    match rendered.line {
        Some(line) => {
            lines.push(0, line);
            lines.nest(1, detail);
        }
        None => lines.nest(0, detail),
    }
    lines
}

/// What a binding group leaves undecided: each equivalence class not solved,
/// and each disjunctive constraint not decided.
pub fn undecided(state: &State) -> Lines {
    let mut lines = Lines::default();
    for root in state.classes.roots() {
        if state.is_solved(root) {
            continue;
        }
        let yes = |holds: bool| holds.then(|| "yes".to_string());
        let (no_value, enclosing) = (state.has_no_value(root), state.is_foreign(root));
        let line = format!("unsolved {}", class_of(state, root));
        let attributes = [
            ("universal", yes(state.is_universal(root))),
            ("no value", yes(no_value)),
            ("enclosing", yes(!no_value && enclosing)),
        ];
        lines.push(0, with_attributes(line, attributes));
    }
    for disjunction in state.undecided() {
        lines.push(0, format!("undecided {}", name(disjunction.as_str())));
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
    let undecided = state.undecided();
    if let Some(undecided) = names(undecided.iter()) {
        lines.push(1, format!("disjunctive constraints: {undecided}"));
    }
    for at in state.relations_on(&c) {
        let line = format!("relation #{at} {}", relation(state, at));
        let by = recorder(state, &state.classes.items()[at].recorder);
        lines.push(1, with_attributes(line, [("by", Some(by))]));
    }
    lines
}
