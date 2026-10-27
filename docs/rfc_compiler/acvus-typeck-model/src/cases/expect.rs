//! A case's expectations against its runs: each failure written as what was
//! expected and what the model gives.

use std::collections::BTreeSet;

use crate::check::{Cause, Outcome, Refusal};
use crate::coerce::{along_le, coerce};
use crate::decl::{DeclId, Decls};
use crate::disjunction::{Decision, LendTarget};
use crate::join::{join, le};
use crate::ty::{Mutability, Name};

use super::build::{closed, planted};
use super::parse::{Change, Expect, Line, Located, RefusalText};
use super::render;
use super::run::{Checked, Corruption, Explored, Run};

/// What failed, each with the line of its expectation where it has one.
pub type Failures = Vec<String>;

/// A certificate change's expectation: the `what`s its checker gives, over
/// every final state, each of which it fails.
pub struct ChangeFails {
    pub line: Line,
    pub fails: BTreeSet<String>,
}

/// The certificate changes a case writes, resolved, with what each expects.
pub struct Changes {
    pub corruptions: Vec<Corruption>,
    pub expected: Vec<ChangeFails>,
}

/// A decision as `render::decision` writes it.
pub fn decision(text: &str) -> Result<Decision, String> {
    let words: Vec<&str> = text.split_whitespace().collect();
    let reference = |word: &str| match word {
        "&" => Ok(Mutability::Shared),
        "&mut" => Ok(Mutability::Exclusive),
        other => Err(format!("a reference's mutability, not {other}")),
    };
    let target = |words: &[&str]| match words {
        ["place"] => Ok(LendTarget::Place),
        ["through"] => Ok(LendTarget::Through),
        _ => Err(format!("a lend's target, not {}", words.join(" "))),
    };
    let decl = |text: &str| -> Result<DeclId, String> {
        let (name, overload) = text
            .trim()
            .rsplit_once('#')
            .ok_or_else(|| format!("a declaration `name#n`, not {text}"))?;
        let overload = overload
            .parse()
            .map_err(|_| format!("an overload's number, not {overload}"))?;
        Ok(DeclId {
            name: Name::new(name.trim().trim_matches('`')),
            overload,
        })
    };
    let listed = |text: &str, open: char, close: char| -> Result<Vec<String>, String> {
        let inner = text
            .trim()
            .strip_prefix(open)
            .and_then(|inner| inner.strip_suffix(close))
            .ok_or_else(|| format!("a list in {open}{close}, not {text}"))?;
        Ok(inner
            .split(',')
            .map(str::trim)
            .filter(|item| !item.is_empty())
            .map(str::to_string)
            .collect())
    };
    Ok(match words.as_slice() {
        ["never"] => Decision::Never,
        ["applied"] => Decision::Applied,
        ["applied", "through"] => Decision::AppliedToTarget,
        ["read"] => Decision::Read,
        ["lend", mutability, rest @ ..] => Decision::Lend {
            target: target(rest)?,
            mutability: reference(mutability)?,
        },
        ["bind", "lend", mutability, rest @ ..] => Decision::BindLend {
            target: target(rest)?,
            mutability: reference(mutability)?,
        },
        ["calls", ..] => {
            let rest = text
                .trim()
                .strip_prefix("calls")
                .expect("the text begins so");
            let lambdas = listed(rest, '{', '}')?
                .iter()
                .map(|lambda| Name::new(lambda.trim_matches('`')))
                .collect();
            Decision::Calls(lambdas)
        }
        _ => Decision::Use(decl(text)?),
    })
}

/// A certificate change resolved against the declarations.
pub fn corruption(decls: &Decls, group: &str, change: &Change) -> Result<Corruption, String> {
    Ok(match change {
        Change::Value { class, ty } => Corruption::Value {
            group: group.to_string(),
            class: class.clone(),
            value: planted(decls.types(), ty)?,
        },
        Change::Decision {
            disjunction,
            decision: text,
        } => Corruption::Decision {
            group: group.to_string(),
            disjunction: disjunction.clone(),
            decision: decision(text)?,
        },
    })
}

fn value_of<'a>(outcome: &'a Outcome, class: &str) -> Option<&'a Option<crate::ty::Ty>> {
    outcome
        .values
        .iter()
        .find(|(name, _)| name.to_string() == class)
        .map(|(_, value)| value)
}

fn refusal_matches(text: &RefusalText, refusal: &Refusal) -> bool {
    let named = |wanted: &Option<String>| {
        wanted.as_ref().is_none_or(|wanted| {
            refusal
                .members
                .iter()
                .any(|member| member.to_string() == *wanted)
        })
    };
    match (text, &refusal.cause) {
        (RefusalText::Infinite(wanted), Cause::Infinite)
        | (RefusalText::NoJoin(wanted), Cause::NoJoin)
        | (RefusalText::NoMeet(wanted), Cause::NoMeet)
        | (RefusalText::NoValue(wanted), Cause::NoValue) => named(wanted),
        (RefusalText::Conflict(wanted), Cause::Conflict(_)) => wanted.as_ref().is_none_or(|wanted| {
            refusal
                .members
                .iter()
                .any(|member| member.to_string() == *wanted)
                || refusal.items.contains(wanted)
        }),
        (RefusalText::Undeclared, Cause::Undeclared(_)) => true,
        _ => false,
    }
}

/// Whether the outcome satisfies the expectation, which is one an outcome
/// satisfies; `Err` says how it does not.
fn outcome_satisfies(decls: &Decls, outcome: &Outcome, expect: &Expect) -> Result<(), String> {
    match expect {
        Expect::Accepted(accepted) => {
            if outcome.accepted == *accepted {
                Ok(())
            } else {
                Err(format!("accepted is {}", outcome.accepted))
            }
        }
        Expect::Value { class, ty } => {
            let Some(value) = value_of(outcome, class) else {
                return Err(format!("no class {class}"));
            };
            let expected = match ty {
                Some(ty) => Some(closed(decls.types(), ty)?),
                None => None,
            };
            if *value == expected {
                Ok(())
            } else {
                let shown = value
                    .as_ref()
                    .map_or("none".to_string(), ToString::to_string);
                Err(format!("{class} is {shown}"))
            }
        }
        Expect::Decision {
            disjunction,
            decision,
        } => {
            let Some(decided) = outcome.decisions.get(&Name::new(disjunction)) else {
                return Err(format!("no disjunctive constraint {disjunction}"));
            };
            let shown = decided
                .as_ref()
                .map_or("undecided".to_string(), render::decision);
            if render::same(decision, &shown) {
                Ok(())
            } else {
                Err(format!("{disjunction} is {shown}"))
            }
        }
        Expect::Refused(text) => {
            if outcome
                .refusals
                .iter()
                .any(|refusal| refusal_matches(text, refusal))
            {
                Ok(())
            } else {
                let refusals: Vec<String> = outcome.refusals.iter().map(render::refusal).collect();
                Err(format!("the refusals are [{}]", refusals.join("; ")))
            }
        }
        Expect::Failing(relations) => {
            let failing: Vec<&str> = outcome
                .checks
                .iter()
                .filter(|check| !check.holds())
                .map(|check| check.relation.as_str())
                .collect();
            if failing == *relations {
                Ok(())
            } else {
                Err(format!("the failing checks are {failing:?}"))
            }
        }
        Expect::Check { relation, coercion } => {
            let of: Vec<String> = outcome
                .checks
                .iter()
                .filter(|check| check.relation == *relation)
                .map(|check| render::coercion(&check.coercion))
                .collect();
            if of.iter().any(|shown| render::same(shown, coercion)) {
                Ok(())
            } else {
                Err(format!("the checks of {relation} are {of:?}"))
            }
        }
        Expect::Partition(names) => {
            let found = outcome
                .partition
                .iter()
                .any(|set| set.iter().map(ToString::to_string).collect::<BTreeSet<_>>() == *names);
            if found {
                Ok(())
            } else {
                Err("no equivalence class holds just these".to_string())
            }
        }
        _ => Ok(()),
    }
}

/// Whether the expectation is one an outcome satisfies.
fn of_outcome(expect: &Expect) -> bool {
    matches!(
        expect,
        Expect::Accepted(_)
            | Expect::Value { .. }
            | Expect::Decision { .. }
            | Expect::Refused(_)
            | Expect::Failing(_)
            | Expect::Check { .. }
            | Expect::Partition(_)
    )
}

fn istar_of(expects: &[Located<Expect>]) -> BTreeSet<String> {
    expects
        .iter()
        .filter_map(|Located { item, .. }| match item {
            Expect::Istar { kind, disjunction } => {
                Some(render::istar(*kind, &Name::new(disjunction)))
            }
            _ => None,
        })
        .collect()
}

/// How a block of expectations fails an outcome and the final states that
/// reach it, I\* expected to be exactly `istar`.
fn block_fails(
    decls: &Decls,
    run: &Run,
    outcome: &Outcome,
    block: &[Located<Expect>],
    istar: &BTreeSet<String>,
) -> Vec<String> {
    let mut fails: Vec<String> = block
        .iter()
        .filter(|located| of_outcome(&located.item))
        .filter_map(|Located { line, item }| {
            outcome_satisfies(decls, outcome, item)
                .err()
                .map(|error| format!("line {line}: {error}"))
        })
        .collect();
    let finals: Vec<&Checked> = run
        .finals
        .iter()
        .filter(|end| end.outcome.same_typing(outcome))
        .collect();
    for end in finals {
        if end.istar != *istar {
            fails.push(format!(
                "I* expected {:?}, a final state gives {:?}",
                istar, end.istar
            ));
        }
    }
    fails
}

/// The blocks matched to distinct outcomes among `outcomes`, each satisfying its
/// own; `None` where no such matching is.
fn matching(satisfies: &[Vec<bool>], outcomes: usize) -> Option<Vec<usize>> {
    fn assign(
        block: usize,
        satisfies: &[Vec<bool>],
        taken: &mut Vec<bool>,
        chosen: &mut Vec<usize>,
    ) -> bool {
        if block == satisfies.len() {
            return true;
        }
        for (outcome, &satisfies_it) in satisfies[block].iter().enumerate() {
            if satisfies_it && !taken[outcome] {
                taken[outcome] = true;
                chosen.push(outcome);
                if assign(block + 1, satisfies, taken, chosen) {
                    return true;
                }
                chosen.pop();
                taken[outcome] = false;
            }
        }
        false
    }
    let mut taken = vec![false; outcomes];
    let mut chosen = Vec::new();
    assign(0, satisfies, &mut taken, &mut chosen).then_some(chosen)
}

/// The failures of `expects` and `blocks` against one run.
fn run_fails(
    decls: &Decls,
    run: &Run,
    expects: &[Located<Expect>],
    blocks: &[Located<Vec<Located<Expect>>>],
    changes: &[ChangeFails],
) -> Failures {
    let mut fails = Failures::new();
    if let Explored::All {
        states,
        complete: false,
    } = run.explored
    {
        fails.push(format!("{states} states explored, past the budget"));
    }
    for end in &run.finals {
        if !end.unsound.is_empty() {
            fails.push(format!("the certificate fails: {:?}", end.unsound));
        }
    }
    let count = expects.iter().find_map(|Located { item, .. }| match item {
        Expect::Outcomes(n) => Some(*n),
        _ => None,
    });
    let wanted = match (count, blocks.len()) {
        (Some(n), _) => n,
        (None, 0) => 1,
        (None, blocks) => blocks,
    };
    if run.outcomes.len() != wanted {
        fails.push(format!(
            "{} outcomes, expected {wanted}",
            run.outcomes.len()
        ));
    }
    let top_istar = istar_of(expects);
    for outcome in &run.outcomes {
        for Located { line, item } in expects.iter().filter(|located| of_outcome(&located.item)) {
            if let Err(error) = outcome_satisfies(decls, outcome, item) {
                fails.push(format!("line {line}: {error}"));
            }
        }
    }
    if blocks.is_empty() {
        for end in &run.finals {
            if end.istar != top_istar {
                fails.push(format!(
                    "I* expected {:?}, a final state gives {:?}",
                    top_istar, end.istar
                ));
            }
        }
    } else {
        let istars: Vec<BTreeSet<String>> = blocks
            .iter()
            .map(|block| top_istar.union(&istar_of(&block.item)).cloned().collect())
            .collect();
        let satisfies: Vec<Vec<bool>> = blocks
            .iter()
            .zip(&istars)
            .map(|(block, istar)| {
                run.outcomes
                    .iter()
                    .map(|outcome| block_fails(decls, run, outcome, &block.item, istar).is_empty())
                    .collect()
            })
            .collect();
        if matching(&satisfies, run.outcomes.len()).is_none() {
            let mut unsatisfied = false;
            for ((block, istar), satisfies) in blocks.iter().zip(&istars).zip(&satisfies) {
                if satisfies.iter().any(|&satisfied| satisfied) {
                    continue;
                }
                unsatisfied = true;
                for (i, outcome) in run.outcomes.iter().enumerate() {
                    for fail in block_fails(decls, run, outcome, &block.item, istar) {
                        fails.push(format!(
                            "the outcome block of line {}, against outcome {}: {fail}",
                            block.line,
                            i + 1
                        ));
                    }
                }
            }
            if !unsatisfied {
                fails.push("no distinct outcomes satisfy the outcome blocks each".to_string());
            }
        }
    }
    for (
        ChangeFails {
            line,
            fails: expected,
        },
        corrupted,
    ) in changes.iter().zip(&run.corrupted)
    {
        if corrupted.fails != *expected {
            fails.push(format!(
                "line {line}: the changed certificate fails with {:?}",
                corrupted.fails
            ));
        }
        if corrupted.held > 0 {
            fails.push(format!(
                "line {line}: the changed certificate holds in {} final states",
                corrupted.held
            ));
        }
    }
    // The reports, and each step's line and the refusals it made.
    let (rendered, steps): (Vec<&super::render::Rendered>, Vec<String>) = match &run.explored {
        Explored::Written(taken) => (
            taken.iter().flat_map(|taken| &taken.reports).collect(),
            taken
                .iter()
                .flat_map(|taken| {
                    let refused = taken
                        .refused
                        .iter()
                        .map(|refusal| format!("refuse: {refusal}"));
                    std::iter::once(taken.step.clone()).chain(refused)
                })
                .collect(),
        ),
        Explored::All { .. } | Explored::Sampled(_) => (Vec::new(), Vec::new()),
    };
    for Located { line, item } in expects {
        match item {
            Expect::Trace(text) => {
                if !matches!(run.explored, Explored::Written(_)) {
                    fails.push(format!("line {line}: a trace needs a written order"));
                } else if !rendered
                    .iter()
                    .filter_map(|report| report.line.as_ref())
                    .any(|line| render::same(line, text))
                    && !steps.iter().any(|line| render::same(line, text))
                {
                    fails.push(format!("line {line}: no line of the trace is {text:?}"));
                }
            }
            Expect::Records {
                phase,
                relation,
                present,
            } => {
                if !matches!(run.explored, Explored::Written(_)) {
                    fails.push(format!(
                        "line {line}: what a phase records needs a written order"
                    ));
                    continue;
                }
                let found = rendered
                    .iter()
                    .filter(|report| report.phase == *phase)
                    .any(|report| {
                        report
                            .recorded
                            .iter()
                            .any(|(_, recorded)| render::same(recorded, relation))
                    });
                if found != *present {
                    let verb = if *present {
                        "records"
                    } else {
                        "does not record"
                    };
                    fails.push(format!(
                        "line {line}: {phase:?} {verb} {relation}, expected otherwise"
                    ));
                }
            }
            _ => {}
        }
    }
    fails
}

/// How a primitive's expectation fails on the declarations alone, where it
/// does; an expectation of no primitive is none.
fn primitive_fail(decls: &Decls, expect: &Expect) -> Result<Option<String>, String> {
    let types = decls.types();
    Ok(match expect {
        Expect::Join { a, b, joined } => {
            let (a, b) = (closed(types, a)?, closed(types, b)?);
            let expected = match joined {
                Some(ty) => Some(closed(types, ty)?),
                None => None,
            };
            let got = join(decls, &a, &b).ok();
            (got != expected).then(|| {
                let shown = got.map_or("none".to_string(), |ty| ty.to_string());
                format!("join({a}, {b}) is {shown}")
            })
        }
        Expect::Le { a, b, holds } => {
            let (a, b) = (closed(types, a)?, closed(types, b)?);
            let got = le(decls, &a, &b);
            (got != *holds).then(|| format!("{a} <= {b} is {got}"))
        }
        Expect::Coerce {
            along_le: along,
            from,
            to,
            coercion,
        } => {
            let (from, to) = (closed(types, from)?, closed(types, to)?);
            let got = if *along {
                along_le(decls, &from, &to)
            } else {
                coerce(decls, &from, &to)
            };
            let shown = render::coercion(&got);
            (!render::same(&shown, coercion)).then(|| format!("{from} ~> {to} is {shown}"))
        }
        _ => None,
    })
}

fn primitive_fails(decls: &Decls, expects: &[Located<Expect>]) -> Failures {
    let mut fails = Failures::new();
    for Located { line, item } in expects {
        match primitive_fail(decls, item) {
            Ok(None) => {}
            Ok(Some(fail)) | Err(fail) => fails.push(format!("line {line}: {fail}")),
        }
    }
    fails
}

/// Whether the expectation is a primitive's, satisfied by the declarations alone.
fn is_primitive(expect: &Expect) -> bool {
    matches!(
        expect,
        Expect::Join { .. } | Expect::Le { .. } | Expect::Coerce { .. }
    )
}

/// The certificate changes `expects` writes.
pub fn changes(decls: &Decls, expects: &[Located<Expect>]) -> Result<Changes, String> {
    let mut changes = Changes {
        corruptions: Vec::new(),
        expected: Vec::new(),
    };
    for Located { line, item } in expects {
        if let Expect::Certificate {
            group,
            change,
            fails,
        } = item
        {
            let corruption = corruption(decls, group, change)
                .map_err(|error| format!("line {line}: {error}"))?;
            changes.corruptions.push(corruption);
            changes.expected.push(ChangeFails {
                line: *line,
                fails: fails.clone(),
            });
        }
    }
    Ok(changes)
}

/// The failures of a set of expectations against every run.
pub fn fails(
    decls: &Decls,
    runs: &[Run],
    expects: &[Located<Expect>],
    blocks: &[Located<Vec<Located<Expect>>>],
    changes: &[ChangeFails],
) -> Failures {
    let (primitives, rest): (Vec<_>, Vec<_>) = expects
        .iter()
        .cloned()
        .partition(|located| is_primitive(&located.item));
    let mut fails = primitive_fails(decls, &primitives);
    for (i, run) in runs.iter().enumerate() {
        for fail in run_fails(decls, run, &rest, blocks, changes) {
            fails.push(if runs.len() > 1 {
                format!("run {}: {fail}", i + 1)
            } else {
                fail
            });
        }
    }
    fails
}
