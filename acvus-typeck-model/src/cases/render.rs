//! What the model gives, written as a case writes it: decisions, coercions,
//! refusals, a phase's reports and an outcome (`design/19-cases.md`). An
//! expectation compares its text with these, whitespace left out.

use std::fmt::Write;

use crate::check::{Outcome, Refusal};
use crate::coerce::{CoerceError, Coercion, Structural};
use crate::decl::DeclId;
use crate::node::{Decision, LendTarget};
use crate::relation::Named;
use crate::solve::{State, Step};
use crate::split::Split;
use crate::trace::Report;
use crate::ty::{Mutability, Name};
use crate::value::Took;

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
        Decision::Unreached(ids) => format!("unreached [{}]", listed(ids.iter().map(decl))),
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
        Decision::NoResult => "no result".to_string(),
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
                    Structural::Wrap(variant, inner) => {
                        format!("wrap {}({})", name(variant.as_str()), ok_coercion(inner))
                    }
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

/// A phase's report, and the relations it recorded, named in `state`.
pub struct Rendered {
    pub phase: Phase,
    pub line: String,
    pub recorded: Vec<String>,
}

pub fn report(state: &State, report: &Report) -> Rendered {
    let class = |id| name(&state.class_name(id).to_string());
    let relation = |at: usize| {
        Named {
            relation: &state.log[at].relation,
            arena: &state.arena,
        }
        .to_string()
    };
    let refused = |refused: &Option<Refusal>| match refused {
        Some(refused) => format!(", refused {}", self::refusal(refused)),
        None => String::new(),
    };
    let (phase, line, recorded) = match report {
        Report::Recorded(recorded) => (
            Phase::Record,
            "record the text".to_string(),
            recorded.clone(),
        ),
        Report::Linked(linked) => (
            Phase::Unify,
            format!(
                "unify linked {} to {}{}",
                class(linked.container),
                class(linked.element),
                refused(&linked.refused)
            ),
            linked.recorded.clone(),
        ),
        Report::Merged(recorded) => (
            Phase::Join,
            "join merged cycles".to_string(),
            recorded.clone(),
        ),
        Report::Infinite { class: id, refusal } => (
            Phase::Join,
            format!("join refused {}: {}", class(*id), self::refusal(refusal)),
            0..0,
        ),
        Report::LitLowers {
            class: id,
            recorded,
        } => (
            Phase::Join,
            format!("join unified {} with its LitVar lower bounds", class(*id)),
            recorded.clone(),
        ),
        Report::LitBoundaries {
            class: id,
            recorded,
        } => (
            Phase::Split,
            format!("split unified {} with what it passes into", class(*id)),
            recorded.clone(),
        ),
        Report::Joined { class: id, took } => {
            let took = match took {
                Took::Held => "held its kind".to_string(),
                Took::Joined { head, refused: r } => {
                    let members: Vec<String> =
                        head.members.iter().map(|m| format!("{m:?}")).collect();
                    if members.is_empty() {
                        format!("took {:?}{}", head.constructor, refused(r))
                    } else {
                        format!(
                            "took {:?} {{{}}}{}",
                            head.constructor,
                            listed(members),
                            refused(r)
                        )
                    }
                }
                Took::Nothing => "took nothing".to_string(),
                Took::NoJoin(r) => format!("has no join: {}", self::refusal(r)),
            };
            (Phase::Join, format!("join {} {took}", class(*id)), 0..0)
        }
        Report::LitJoined { bound, recorded } => (
            Phase::Join,
            format!("join unified the LitVar of {}", relation(*bound)),
            recorded.clone(),
        ),
        Report::Split { at, split } => {
            let (what, recorded) = match split {
                Split::OneClass => ("one class", 0..0),
                Split::Parts(recorded) => ("parts", recorded.clone()),
                Split::Unsolved => ("unsolved", 0..0),
                Split::NoParts => ("no parts", 0..0),
            };
            (
                Phase::Split,
                format!("split {}: {what}", relation(*at)),
                recorded,
            )
        }
        Report::Shared(recorded) => (
            Phase::Node,
            "node recorded what candidates share".to_string(),
            recorded.clone(),
        ),
        Report::Decided(decided) => (
            Phase::Node,
            format!(
                "node decided {} as {}",
                name(decided.node.as_str()),
                decision(&decided.decision)
            ),
            decided.recorded.clone(),
        ),
        Report::FellBack(fell) => {
            let default = fell.default.clone().map(|_, part| state.class_name(part));
            (
                Phase::Fallback,
                format!(
                    "fallback {} took {default}{}",
                    class(fell.class),
                    refused(&fell.refused)
                ),
                0..0,
            )
        }
    };
    Rendered {
        phase,
        line,
        recorded: recorded.map(relation).collect(),
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
        writeln!(
            text,
            "  check {}: {:?}",
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
