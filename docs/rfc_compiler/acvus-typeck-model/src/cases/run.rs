//! A case's program solved as its orders say, each final state handed to the
//! checks of notes 14 and 15 and to the case's changed certificates
//! (`design/19-cases.md`, The runner), and under `TRACE=1` the trace of
//! solving printed (`design/19-cases.md`, `TRACE=1`).

use std::collections::{BTreeMap, BTreeSet};
use std::io::Write;
use std::rc::Rc;

use crate::certificate::Certificate;
use crate::check::{Outcome, Refusal};
use crate::class::{ClassId, ClassName};
use crate::explore::{Final, Order, explore_with, sample_with};
use crate::node::Decision;
use crate::program::{ENTRY, Program};
use crate::solve::{State, Step};
use crate::trace::Trace;
use crate::ty::{Name, Ty};

use super::parse::{self, Orders, StepText};
use super::render::{self, Lines, Rendered};

/// The budget of states for `orders: all` (`design/13-exploration.md`).
pub const BUDGET: usize = 100_000;

/// The steps a traced first order takes before it is stopped, a fallback
/// counted as one, as a written order writes it (`TRACE=1`).
pub const LIMIT: usize = 10_000;

/// Where a traced run prints (`TRACE=1`): stdout, each line prefixed with the
/// case's path, since the cases' threads print at once, and flushed as soon as
/// it is known, so that a step that does not end is named by the last line.
pub struct Tracer {
    path: String,
}

impl Tracer {
    pub fn new(path: &str) -> Tracer {
        Tracer {
            path: path.to_string(),
        }
    }

    pub fn line(&self, depth: usize, text: &str) {
        let mut out = std::io::stdout().lock();
        writeln!(out, "{}: {}{text}", self.path, "  ".repeat(depth))
            .and_then(|()| out.flush())
            .expect("writing the trace to stdout");
    }

    fn lines(&self, depth: usize, lines: Lines) {
        for (at, text) in lines.0 {
            self.line(depth + at, &text);
        }
    }
}

/// A certificate changed as a solver that erred would leave it.
pub enum Corruption {
    Value {
        group: String,
        class: ClassName,
        ty: Ty,
    },
    Decision {
        group: String,
        node: String,
        decision: Decision,
    },
}

/// What the checker found of one corruption, over every final state.
#[derive(Default)]
pub struct Corrupted {
    /// The violations' `what`.
    pub fails: BTreeSet<String>,
    /// How many final states' corrupted certificate holds.
    pub held: usize,
}

/// A final state's outcome, its I\* violations, and the violations of its
/// certificate unchanged.
pub struct Checked {
    pub outcome: Outcome,
    pub istar: BTreeSet<String>,
    pub unsound: Vec<String>,
}

/// One step a written order took, or the start of a group's state: its line
/// (`start g`, `step k g: s`, `fallback k g`), its phases' reports named in the
/// state it left, and the refusals it made.
pub struct Taken {
    pub step: String,
    pub reports: Vec<Rendered>,
    pub refused: Vec<String>,
}

pub enum Explored {
    /// Every order: the states visited, and whether every order was explored
    /// within the budget.
    All {
        states: usize,
        complete: bool,
    },
    Sampled(u64),
    Written(Vec<Taken>),
}

/// A run: its outcomes, distinct, in the order first reached, and its final
/// states.
pub struct Run {
    pub explored: Explored,
    pub outcomes: Vec<Outcome>,
    pub finals: Vec<Checked>,
    pub corrupted: Vec<Corrupted>,
}

/// The checks each final state is handed to, and under `TRACE=1` the witness
/// of each distinct outcome: the first order that reached it.
struct Visitor<'a> {
    program: &'a Program,
    corruptions: &'a [Corruption],
    finals: Vec<Checked>,
    corrupted: Vec<Corrupted>,
    witnesses: Option<Vec<(Outcome, Order)>>,
}

impl Visitor<'_> {
    fn visit(&mut self, end: &Final<'_>) {
        let unsound = crate::certificate::check(self.program, &Certificate::of(self.program, end))
            .iter()
            .map(|violation| format!("{violation:?}"))
            .collect();
        let istar = crate::istar::check(self.program, end)
            .into_iter()
            .map(|violation| render::istar(violation.kind, &violation.node))
            .collect();
        for (corruption, corrupted) in self.corruptions.iter().zip(&mut self.corrupted) {
            let mut certificate = Certificate::of(self.program, end);
            match corruption {
                Corruption::Value { group, class, ty } => {
                    certificate.set_value(group, class, ty.clone());
                }
                Corruption::Decision {
                    group,
                    node,
                    decision,
                } => certificate.set_decision(group, node, decision.clone()),
            }
            let violations = crate::certificate::check(self.program, &certificate);
            if violations.is_empty() {
                corrupted.held += 1;
            }
            corrupted
                .fails
                .extend(violations.into_iter().map(|violation| violation.what));
        }
        let outcome = end.entry.outcome_with(end.groups);
        if let Some(witnesses) = &mut self.witnesses
            && !witnesses
                .iter()
                .any(|(reached, _)| reached.same_typing(&outcome))
        {
            witnesses.push((outcome.clone(), end.order.clone()));
        }
        self.finals.push(Checked {
            outcome,
            istar,
            unsound,
        });
    }
}

/// Solves the program as `orders` say. Under `TRACE=1`, a written order's
/// trace is printed in full; for every order or sampled orders, the first
/// order is traced before them, and each distinct outcome's witness is
/// replayed traced after them.
pub fn run(
    program: &Program,
    orders: &Orders,
    corruptions: &[Corruption],
    tracer: Option<&Tracer>,
) -> Result<Run, String> {
    let mut visitor = Visitor {
        program,
        corruptions,
        finals: Vec::new(),
        corrupted: corruptions.iter().map(|_| Corrupted::default()).collect(),
        witnesses: tracer.map(|_| Vec::new()),
    };
    if let (Some(tracer), Orders::All | Orders::Sampled(_)) = (tracer, orders) {
        first(program, tracer);
    }
    let (explored, outcomes) = match orders {
        Orders::All => {
            let exploration = explore_with(program, BUDGET, |end| visitor.visit(end));
            if let Some(tracer) = tracer {
                let complete = if exploration.complete {
                    "complete"
                } else {
                    "incomplete"
                };
                tracer.line(
                    1,
                    &format!("explored: {} states, {complete}", exploration.states),
                );
            }
            let explored = Explored::All {
                states: exploration.states,
                complete: exploration.complete,
            };
            (explored, exploration.outcomes)
        }
        Orders::Sampled(orders) => {
            let outcomes = sample_with(program, *orders, |end| visitor.visit(end));
            if let Some(tracer) = tracer {
                tracer.line(1, &format!("sampled: {orders} orders"));
            }
            (Explored::Sampled(*orders), outcomes)
        }
        Orders::Written(steps) => {
            let (taken, outcome) = written(program, steps, tracer, &mut |end| visitor.visit(end))?;
            (Explored::Written(taken), vec![outcome])
        }
    };
    if let (Some(tracer), Some(witnesses)) = (tracer, &visitor.witnesses)
        && !matches!(orders, Orders::Written(_))
    {
        for (k, (outcome, order)) in witnesses.iter().enumerate() {
            witness(program, tracer, k + 1, outcome, order);
        }
    }
    Ok(Run {
        explored,
        outcomes,
        finals: visitor.finals,
        corrupted: visitor.corrupted,
    })
}

/// What one step, start or fallback did: its line (`start g`, `step k g: s`,
/// `fallback k g`), its phases' reports, and the refusals it made.
struct Did {
    line: String,
    trace: Trace,
    refused: Vec<Refusal>,
}

impl Did {
    /// As a written order's trace keeps it, named in the state it left.
    fn taken(self, state: &State) -> Taken {
        Taken {
            step: self.line,
            reports: self
                .trace
                .reports
                .iter()
                .map(|report| render::report(state, report))
                .collect(),
            refused: self.refused.iter().map(render::refusal).collect(),
        }
    }
}

/// An order of solving followed one step at a time, each step's line printed
/// before it is taken and what it did after, where a tracer is given.
struct Follow<'a> {
    tracer: Option<&'a Tracer>,
    /// The steps taken over the whole order, a fallback counted as one.
    taken: usize,
}

impl Follow<'_> {
    fn print(&self, depth: usize, text: &str) {
        if let Some(tracer) = self.tracer {
            tracer.line(depth, text);
        }
    }

    fn did(&self, state: &State, did: &Did) {
        let Some(tracer) = self.tracer else {
            return;
        };
        for report in &did.trace.reports {
            tracer.lines(3, render::traced(state, report));
        }
        for refused in &did.refused {
            tracer.line(3, &format!("refuse: {}", render::refusal(refused)));
        }
    }

    /// A group's state as solving starts it, with its start taken as a step.
    fn start(
        &self,
        program: &Program,
        group: &Name,
        solved: &BTreeMap<Name, Rc<State>>,
    ) -> (State, Did) {
        let line = format!("start {}", render::name(group.as_str()));
        self.print(2, &line);
        let started = State::start(program, group, solved);
        let did = Did {
            line,
            trace: started.trace,
            refused: started.state.refusals.iter().cloned().collect(),
        };
        self.did(&started.state, &did);
        if let Some(tracer) = self.tracer {
            tracer.lines(3, render::waiting(&started.state));
        }
        (started.state, did)
    }

    /// Takes the step, which `state` may take now.
    fn step(&mut self, state: &mut State, group: &str, step: &Step) -> Did {
        self.taken += 1;
        let line = format!(
            "step {} {}: {}",
            self.taken,
            render::name(group),
            render::step(state, step)
        );
        if self.tracer.is_some() {
            let steps: Vec<String> = state
                .steps()
                .iter()
                .map(|step| render::step(state, step))
                .collect();
            self.print(2, &format!("{line}  steps: {}", steps.join(", ")));
        }
        let before = state.refusals.clone();
        let trace = state.step(step);
        let did = Did {
            line,
            trace,
            refused: state.refusals.difference(&before).cloned().collect(),
        };
        self.did(state, &did);
        did
    }

    /// Falls back (l.288), at the fixpoint of the entry; the trace's
    /// `fell_back` says whether a class fell back.
    fn fall_back(&mut self, state: &mut State) -> Did {
        self.taken += 1;
        let line = format!("fallback {} {ENTRY}", self.taken);
        self.print(2, &line);
        let before = state.refusals.clone();
        let trace = state.fall_back();
        let did = Did {
            line,
            trace,
            refused: state.refusals.difference(&before).cloned().collect(),
        };
        self.did(state, &did);
        did
    }

    /// What a group waits for at its fixpoint, and what a group other than
    /// the entry leaves in C.
    fn fixpoint(&self, state: &State, group: &str) {
        let Some(tracer) = self.tracer else {
            return;
        };
        tracer.line(2, &format!("fixpoint {}", render::name(group)));
        tracer.lines(3, render::waiting(state));
        if group != ENTRY {
            tracer.lines(3, render::c(state));
        }
    }

    /// Whether the order has taken `LIMIT` steps, which a traced first order
    /// stops at.
    fn at_limit(&self) -> bool {
        if self.taken < LIMIT {
            return false;
        }
        self.print(2, &format!("limit: {LIMIT}"));
        true
    }
}

/// The root of the classes a written step names, which must be one.
fn root_named(state: &State, step: &str, class: &str) -> Result<ClassId, String> {
    let roots: BTreeSet<_> = state
        .arena
        .classes()
        .filter(|(_, name)| name.to_string() == *class)
        .map(|(id, _)| state.root(id))
        .collect();
    match roots.into_iter().collect::<Vec<_>>().as_slice() {
        [] => Err(format!("{step} {class}: no class of that name")),
        [root] => Ok(*root),
        _ => Err(format!(
            "{step} {class}: classes of two roots have that name"
        )),
    }
}

/// A step a case writes, as a step the state may take now.
fn step_of(state: &State, text: &StepText) -> Result<Step, String> {
    let steps = state.steps();
    let wanted = match text {
        StepText::Solve(class) => Step::Solve(root_named(state, "solve", class)?),
        StepText::Fill(class) => Step::Fill(root_named(state, "fill", class)?),
        StepText::Decide(node) => Step::Decide(Name::new(node)),
        StepText::Fallback => return Err("fallback is no step".to_string()),
    };
    if steps.contains(&wanted) {
        Ok(wanted)
    } else {
        let listed: Vec<String> = steps.iter().map(|step| render::step(state, step)).collect();
        Err(format!(
            "{} is not a step now; the steps are [{}]",
            render::step(state, &wanted),
            listed.join(", ")
        ))
    }
}

/// The written steps of one group, each with its phases' reports, the steps
/// taken pushed onto the group's last steps to a fixpoint in `rounds`, and a
/// fallback that fell a class back starting the next.
fn take(
    follow: &mut Follow<'_>,
    state: &mut State,
    group: &str,
    steps: &[StepText],
    rounds: &mut Vec<Vec<Step>>,
    taken: &mut Vec<Taken>,
) -> Result<(), String> {
    for text in steps {
        let did = match text {
            StepText::Fallback => {
                if group != ENTRY {
                    return Err(format!(
                        "fallback in {group}, inside which there is none (l.286)"
                    ));
                }
                if !state.steps().is_empty() {
                    return Err("fallback before the fixpoint (l.288)".to_string());
                }
                follow.fixpoint(state, group);
                let did = follow.fall_back(state);
                if did.trace.fell_back() {
                    rounds.push(Vec::new());
                }
                did
            }
            _ => {
                let step = step_of(state, text)?;
                let did = follow.step(state, group, &step);
                rounds
                    .last_mut()
                    .expect("a group's steps have a fixpoint to reach")
                    .push(step);
                did
            }
        };
        taken.push(did.taken(state));
    }
    Ok(())
}

/// One order: each group's written steps, then its first step until none is
/// left, the entry falling back wherever none is (l.288).
fn written<F>(
    program: &Program,
    steps: &BTreeMap<String, Vec<StepText>>,
    tracer: Option<&Tracer>,
    visit: &mut F,
) -> Result<(Vec<Taken>, Outcome), String>
where
    F: FnMut(&Final<'_>),
{
    let groups = program.group_order();
    for group in steps.keys() {
        if group != ENTRY && !groups.iter().any(|name| name.as_str() == group) {
            return Err(format!("the order of {group}, which is no group"));
        }
    }
    let mut follow = Follow { tracer, taken: 0 };
    follow.print(1, "written order");
    let mut order = Order::new();
    let mut taken = Vec::new();
    let mut solved: BTreeMap<Name, Rc<State>> = BTreeMap::new();
    for group in groups {
        let (mut state, did) = follow.start(program, &group, &solved);
        taken.push(did.taken(&state));
        let mut rounds = vec![Vec::new()];
        if let Some(steps) = steps.get(group.as_str()) {
            take(
                &mut follow,
                &mut state,
                group.as_str(),
                steps,
                &mut rounds,
                &mut taken,
            )?;
        }
        let [mut group_steps] = <[Vec<Step>; 1]>::try_from(rounds)
            .expect("a group other than the entry has no fallback");
        while let Some(step) = state.steps().first().cloned() {
            follow.step(&mut state, group.as_str(), &step);
            group_steps.push(step);
        }
        follow.fixpoint(&state, group.as_str());
        order.groups.push((group.clone(), group_steps));
        solved.insert(group, Rc::new(state));
    }
    let (mut state, did) = follow.start(program, &Name::new(ENTRY), &solved);
    taken.push(did.taken(&state));
    if let Some(steps) = steps.get(ENTRY) {
        take(
            &mut follow,
            &mut state,
            ENTRY,
            steps,
            &mut order.entry,
            &mut taken,
        )?;
    }
    loop {
        if let Some(step) = state.steps().first().cloned() {
            follow.step(&mut state, ENTRY, &step);
            order.push_step(step);
            continue;
        }
        follow.fixpoint(&state, ENTRY);
        if !follow.fall_back(&mut state).trace.fell_back() {
            break;
        }
        order.push_fallback();
    }
    visit(&Final {
        entry: &state,
        groups: &solved,
        order: &order,
    });
    Ok((taken, state.outcome_with(&solved)))
}

/// The first order, as sampled order 0 takes it: at every state its first
/// step, the entry falling back wherever none is (l.288), traced, and stopped
/// at `LIMIT` steps, so that a program whose exploration does not end still
/// prints what this order does.
fn first(program: &Program, tracer: &Tracer) {
    let mut follow = Follow {
        tracer: Some(tracer),
        taken: 0,
    };
    tracer.line(1, "first order");
    let mut solved: BTreeMap<Name, Rc<State>> = BTreeMap::new();
    for group in program.group_order() {
        let (mut state, _) = follow.start(program, &group, &solved);
        while let Some(step) = state.steps().first().cloned() {
            if follow.at_limit() {
                return;
            }
            follow.step(&mut state, group.as_str(), &step);
        }
        follow.fixpoint(&state, group.as_str());
        solved.insert(group, Rc::new(state));
    }
    let (mut state, _) = follow.start(program, &Name::new(ENTRY), &solved);
    loop {
        if follow.at_limit() {
            return;
        }
        if let Some(step) = state.steps().first().cloned() {
            follow.step(&mut state, ENTRY, &step);
            continue;
        }
        follow.fixpoint(&state, ENTRY);
        if !follow.fall_back(&mut state).trace.fell_back() {
            break;
        }
    }
    tracer.line(2, "end");
    for line in render::outcome(&state.outcome_with(&solved)).lines() {
        tracer.line(3, line.trim_start());
    }
}

/// An item of an order: a step, or a fallback that fell a class back.
enum Item<'a> {
    Step(&'a Step),
    Fallback,
}

/// Follows `order` from the program's start, taking its steps, captured as
/// the model's own values, as they were taken: each group's to its fixpoint,
/// then the entry's, a fallback between one fixpoint and the next steps, and
/// at the end a fallback that falls no class back. `at` sees each item in the
/// state it is taken in. The final state is the entry's before that last
/// fallback, as the exploration hands it on.
fn replay<F>(
    program: &Program,
    order: &Order,
    follow: &mut Follow<'_>,
    mut at: F,
) -> Result<(State, BTreeMap<Name, Rc<State>>), String>
where
    F: FnMut(&State, &str, Item<'_>) -> Result<(), String>,
{
    let mut solved: BTreeMap<Name, Rc<State>> = BTreeMap::new();
    let step = |follow: &mut Follow<'_>, state: &mut State, group: &str, step: &Step| {
        if !state.steps().contains(step) {
            return Err(format!(
                "{} in {group} is not a step now",
                render::step(state, step)
            ));
        }
        follow.step(state, group, step);
        Ok(())
    };
    for (group, steps) in &order.groups {
        let (mut state, _) = follow.start(program, group, &solved);
        for taken in steps {
            at(&state, group.as_str(), Item::Step(taken))?;
            step(follow, &mut state, group.as_str(), taken)?;
        }
        if !state.steps().is_empty() {
            return Err(format!(
                "{group} is not at its fixpoint where its steps end"
            ));
        }
        follow.fixpoint(&state, group.as_str());
        solved.insert(group.clone(), Rc::new(state));
    }
    let (mut state, _) = follow.start(program, &Name::new(ENTRY), &solved);
    for (round, steps) in order.entry.iter().enumerate() {
        if round > 0 {
            at(&state, ENTRY, Item::Fallback)?;
            follow.fixpoint(&state, ENTRY);
            if !follow.fall_back(&mut state).trace.fell_back() {
                return Err("a fallback of the order falls no class back".to_string());
            }
        }
        for taken in steps {
            at(&state, ENTRY, Item::Step(taken))?;
            step(follow, &mut state, ENTRY, taken)?;
        }
        if !state.steps().is_empty() {
            return Err(format!(
                "{ENTRY} is not at its fixpoint where its steps end"
            ));
        }
    }
    follow.fixpoint(&state, ENTRY);
    let mut after = state.clone();
    if follow.fall_back(&mut after).trace.fell_back() {
        return Err("a class falls back where the order ends".to_string());
    }
    Ok((state, solved))
}

/// A witness's order as a case writes it: a `//@ order g:` line for each
/// group, then the entry's `//@ order:` line, each step named in the state it
/// is taken in.
fn written_lines(program: &Program, order: &Order) -> Result<Vec<String>, String> {
    let mut texts: BTreeMap<String, Vec<String>> = BTreeMap::new();
    replay(
        program,
        order,
        &mut Follow {
            tracer: None,
            taken: 0,
        },
        |state, group, item| {
            let text = match item {
                Item::Step(step) => render::step(state, step),
                Item::Fallback => "fallback".to_string(),
            };
            texts.entry(group.to_string()).or_default().push(text);
            Ok(())
        },
    )?;
    let mut lines: Vec<String> = order
        .groups
        .iter()
        .map(|(group, _)| {
            let steps = texts.remove(group.as_str()).unwrap_or_default();
            format!("//@ order {group}: {}", steps.join(", "))
        })
        .collect();
    let steps = texts.remove(ENTRY).unwrap_or_default();
    lines.push(format!("//@ order: {}", steps.join(", ")));
    Ok(lines)
}

/// Whether the lines round-trip: whether parsing them and resolving their
/// steps with `step_of`, each in the state it is taken in, gives back the
/// order's steps; `Err` names where they do not.
fn round_trip(program: &Program, order: &Order, lines: &[String]) -> Result<(), String> {
    let source = lines.join("\n");
    let Orders::Written(written) = parse::parse(&source)?.orders else {
        return Err("the lines parse as no written order".to_string());
    };
    let mut items: BTreeMap<String, usize> = BTreeMap::new();
    replay(
        program,
        order,
        &mut Follow {
            tracer: None,
            taken: 0,
        },
        |state, group, item| {
            let at = items.entry(group.to_string()).or_default();
            let text = written.get(group).and_then(|steps| steps.get(*at));
            *at += 1;
            match (item, text) {
                (Item::Fallback, Some(StepText::Fallback)) => Ok(()),
                (Item::Step(step), Some(text)) => match step_of(state, text) {
                    Ok(resolved) if resolved == *step => Ok(()),
                    Ok(resolved) => Err(format!(
                        "{text:?} in {group} resolves to {}, and the order takes {}",
                        render::step(state, &resolved),
                        render::step(state, step)
                    )),
                    Err(error) => Err(format!("{text:?} in {group}: {error}")),
                },
                (Item::Fallback, text) => Err(format!("{group} writes {text:?} at a fallback")),
                (Item::Step(step), None) => Err(format!(
                    "{group} writes nothing at {}",
                    render::step(state, step)
                )),
            }
        },
    )?;
    for (group, steps) in &written {
        let followed = items.get(group).copied().unwrap_or_default();
        if steps.len() != followed {
            return Err(format!(
                "{group} writes {} items, and the order takes {followed}",
                steps.len()
            ));
        }
    }
    Ok(())
}

/// Prints the witness of the `k`th outcome: the outcome, its order as a case
/// writes it, whether those lines round-trip, then its replay's full trace and
/// what the replay reaches.
fn witness(program: &Program, tracer: &Tracer, k: usize, outcome: &Outcome, order: &Order) {
    tracer.line(1, &format!("witness {k}"));
    for line in render::outcome(outcome).lines() {
        tracer.line(2, line.trim_start());
    }
    let lines = match written_lines(program, order) {
        Ok(lines) => lines,
        Err(error) => {
            tracer.line(2, &format!("replay: fails: {error}"));
            return;
        }
    };
    for line in &lines {
        tracer.line(2, line);
    }
    match round_trip(program, order, &lines) {
        Ok(()) => tracer.line(2, "round-trip: yes"),
        Err(error) => tracer.line(2, &format!("round-trip: no: {error}")),
    }
    let mut follow = Follow {
        tracer: Some(tracer),
        taken: 0,
    };
    match replay(program, order, &mut follow, |_, _, _| Ok(())) {
        Ok((state, solved)) if state.outcome_with(&solved).same_typing(outcome) => {
            tracer.line(2, &format!("replay: outcome {k}"));
        }
        Ok(_) => tracer.line(2, "replay: another outcome"),
        Err(error) => tracer.line(2, &format!("replay: fails: {error}")),
    }
}
