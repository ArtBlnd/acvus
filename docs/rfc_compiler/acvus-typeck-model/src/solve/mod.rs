//! The state of a binding group's solving, and the loop on it: the record
//! (`record.rs`, `class.rs`) unified (`unify.rs`, `decomposition.rs`), what
//! a decided type names instantiated (`instance.rs`), the judgment, which
//! alone decides, reading the record (`judgment.rs`), and fallback at the
//! fixpoint (`fallback.rs`), each step returning the trace of what it
//! reported (`trace.rs`), and the checks of what is decided (`check.rs`).

use std::collections::{BTreeMap, BTreeSet};
use std::rc::Rc;

pub mod certificate;
pub mod cut;
pub mod decomposition;
pub mod explore;
pub mod instance;
pub mod narrow;
pub mod unify;

use crate::check::Outcome;
use crate::class::{Arena, ClassId, ClassName, Classes, StructureSource};
use crate::cut::Cut;
use crate::decl::Decls;
use crate::decomposition::Decomposition;
use crate::disjunction::{Decision, Disjunction};
use crate::judgment::Judgment;
use crate::program::{ENTRY, Function, Item, Part, Position, Program, TextClass};
use crate::record::{Entry, Recording};
use crate::relation::{Recorder, Relation, Standing};
use crate::trace::{Report, Trace};
use crate::ty::Name;

#[derive(Clone, Debug)]
pub struct State {
    pub(crate) decls: Rc<Decls>,
    pub(crate) arena: Arena,
    pub(crate) classes: Classes,
    /// The binding group being solved.
    pub(crate) group: Name,
    /// The groups enclosing it, innermost first.
    pub(crate) enclosing: Vec<Name>,
    pub(crate) functions: Rc<Vec<Function>>,
    /// The classes the lambdas' environments hold, which the groups
    /// enclosing them solve.
    pub(crate) environment: Rc<BTreeSet<Name>>,
    /// Each group solved before it, at its fixpoint.
    pub(crate) solved: BTreeMap<Name, Rc<State>>,
}

/// A binding group's state where its text's constraints are recorded and
/// nothing is solved, with the trace of the phases that made it.
pub struct Started {
    pub state: State,
    pub trace: Trace,
}

impl State {
    /// The state of the run's entry where the text's constraints are recorded
    /// and nothing is solved, the program having no other group.
    pub fn new(program: &Program) -> State {
        State::of_group(program, &Name::new(ENTRY), &BTreeMap::new())
    }

    /// The state of the binding group `group` where its text's constraints
    /// are recorded and nothing is solved. `solved` holds each
    /// group its uses make instances of, at its fixpoint.
    pub fn of_group(program: &Program, group: &Name, solved: &BTreeMap<Name, Rc<State>>) -> State {
        State::start(program, group, solved).state
    }

    /// As `of_group`, with the trace of what its phases reported: the text's
    /// relations recorded, and what follows at once.
    pub fn start(program: &Program, group: &Name, solved: &BTreeMap<Name, Rc<State>>) -> Started {
        // The groups solved so far cut the lambdas' environments up to this
        // group.
        let cut = Cut::of(program, solved, group);
        let mut state = State {
            decls: Rc::new(program.decls.clone()),
            arena: Arena::default(),
            classes: Classes::default(),
            group: group.clone(),
            enclosing: program.enclosing(group),
            functions: Rc::new(program.functions.clone()),
            environment: Rc::new(cut.environment()),
            solved: solved.clone(),
        };
        let classes: Vec<&TextClass> = program
            .classes
            .iter()
            .filter(|class| cut.owner(program, &class.name) == group)
            .collect();
        let items: Vec<(usize, &Item, Part)> = program
            .items
            .iter()
            .enumerate()
            .filter_map(|(i, (_, item))| Some((i, item, cut.part(program, i, group)?)))
            .collect();
        // Every class's standing is recorded before any relation that reads
        // it: the text's classes first, in the program's order.
        let mut standings = Recording::default();
        for class in &classes {
            let id = state.arena.intern(ClassName::Text(class.name.clone()));
            state.classes.grow(&state.arena);
            let standing = if program.universals.contains(&class.name) {
                Standing::Universal
            } else {
                Standing::Text
            };
            standings.push(state.record(Relation::Standing(id, standing), Recorder::Origin));
        }
        let mut named: Vec<&Name> = items
            .iter()
            .flat_map(|(_, item, part)| item.classes_in(part))
            .collect();
        for (_, item, part) in &items {
            if let Item::Use { callee, .. } = item
                && let Some(function) = program.function(callee)
                && state.enclosing.contains(&function.group)
            {
                let params = function.params.iter().enumerate();
                named.extend(
                    params
                        .filter(|(k, _)| part.holds(Position::Arg(*k)))
                        .map(|(_, param)| param),
                );
                if part.holds(Position::Result) {
                    named.push(&function.result);
                }
            }
        }
        for name in named {
            if state.arena.get(&ClassName::Text(name.clone())).is_none() {
                standings.push(state.share(name, cut.is_text_unknown(program, name)));
            }
        }
        for function in program.functions.iter().filter(|f| f.group == *group) {
            for param in function
                .params
                .iter()
                .filter(|param| cut.is_universal(program, param))
            {
                let id = state.text(param);
                if state.classes.standing(id) != Some(&Standing::Universal) {
                    let relation = Relation::Standing(id, Standing::Universal);
                    standings.push(state.record(relation, Recorder::Origin));
                }
            }
        }
        let mut trace = Trace::default();
        trace.push(Report::Recorded(standings));
        trace.push(Report::Recorded(state.record_origins(program, &classes)));
        trace.push(Report::Recorded(state.record_equations(&items)));
        trace.push(Report::Recorded(
            state.record_items(program, &items, solved),
        ));
        trace.append(state.settle());
        Started { state, trace }
    }

    pub(crate) fn text(&self, name: &Name) -> ClassId {
        self.arena
            .get(&ClassName::Text(name.clone()))
            .unwrap_or_else(|| panic!("the program names the class {name}"))
    }

    /// A class of an enclosing group, shared here, and whether it is
    /// an unknown of its own group: its standing is recorded as it is made.
    pub(crate) fn share(&mut self, name: &Name, unknown: bool) -> Entry {
        let id = self.arena.intern(ClassName::Text(name.clone()));
        self.classes.grow(&self.arena);
        self.record(
            Relation::Standing(id, Standing::Shared { unknown }),
            Recorder::Origin,
        )
    }

    /// The classes the text gives, in the program's order: those whose
    /// standing is the text's or a universal's.
    pub(crate) fn texts(&self) -> Vec<ClassId> {
        self.classes
            .items()
            .iter()
            .filter_map(|recorded| match recorded.relation() {
                Some(Relation::Standing(class, Standing::Text | Standing::Universal)) => {
                    Some(*class)
                }
                _ => None,
            })
            .collect()
    }

    /// An equivalence class with no written structure, not universal, and not
    /// of an enclosing group.
    pub(crate) fn is_unknown(&self, id: ClassId) -> bool {
        let written = matches!(
            self.classes.held(id).structure,
            Some((_, StructureSource::Written))
        );
        !written && !self.is_universal(id) && !self.is_foreign(id)
    }

    /// Whether the class is of an enclosing group, shared here.
    pub(crate) fn is_foreign(&self, id: ClassId) -> bool {
        self.standings(id)
            .any(|standing| matches!(standing, Standing::Shared { .. }))
    }

    /// Whether the class is an unknown of the enclosing group it is shared
    /// from.
    pub(crate) fn is_enclosing_unknown(&self, id: ClassId) -> bool {
        self.standings(id)
            .any(|standing| matches!(standing, Standing::Shared { unknown: true }))
    }

    pub(crate) fn is_universal(&self, id: ClassId) -> bool {
        self.standings(id)
            .any(|standing| matches!(standing, Standing::Universal))
    }

    /// Every standing the record gives a member of the class.
    fn standings(&self, id: ClassId) -> impl Iterator<Item = &Standing> + '_ {
        let root = self.root(id);
        self.classes
            .occurrences(root)
            .iter()
            .filter_map(|occurrence| self.classes.items()[occurrence.at].relation())
            .filter_map(|relation| match relation {
                Relation::Standing(_, standing) => Some(standing),
                _ => None,
            })
    }

    pub(crate) fn root(&self, id: ClassId) -> ClassId {
        self.classes.find(id)
    }

    /// The place in the record of the disjunctive constraint of this name.
    pub(crate) fn disjunction_at(&self, name: &Name) -> usize {
        self.classes
            .disjunction_at(name)
            .unwrap_or_else(|| panic!("the record has the disjunctive constraint {name}"))
    }

    /// The kind and the decision of the disjunctive constraint of this name.
    pub(crate) fn disjunction(&self, name: &Name) -> (&Disjunction, Option<&Decision>) {
        let at = self.disjunction_at(name);
        let (_, kind, decision) = self.classes.items()[at]
            .disjunction()
            .expect("the place names a disjunctive constraint");
        (kind, decision)
    }

    /// Every disjunctive constraint's name and decision, in the record's
    /// order.
    pub(crate) fn decisions(&self) -> BTreeMap<Name, Option<Decision>> {
        self.classes
            .disjunctions()
            .map(|(_, name, _, decision)| (name.clone(), decision.cloned()))
            .collect()
    }

    /// The names of the disjunctive constraints not yet decided, in the
    /// record's order.
    pub(crate) fn undecided(&self) -> Vec<Name> {
        self.classes
            .disjunctions()
            .filter(|(_, _, _, decision)| decision.is_none())
            .map(|(_, name, _, _)| name.clone())
            .collect()
    }

    /// The equivalence classes left with no structure, with a value still
    /// possible, of this group, and no universal: those whose type a judgment
    /// gives, and fallback at the fixpoint.
    pub(crate) fn unsolved(&self) -> Vec<ClassId> {
        self.classes
            .roots()
            .filter(|&root| {
                !self.is_solved(root)
                    && !self.has_no_value(root)
                    && !self.is_foreign(root)
                    && !self.is_universal(root)
            })
            .collect()
    }

    /// Whether the equivalence class took its constructor: it holds a
    /// structure.
    pub(crate) fn is_solved(&self, id: ClassId) -> bool {
        self.classes.held(id).structure.is_some()
    }

    pub(crate) fn has_no_value(&self, id: ClassId) -> bool {
        self.classes.held(id).no_value.is_some()
    }

    /// The name of an equivalence class: the least of its members' names.
    pub(crate) fn class_name(&self, id: ClassId) -> ClassName {
        self.classes
            .held(id)
            .members
            .iter()
            .map(|&member| self.arena.name(member).clone())
            .min()
            .expect("an equivalence class has a member")
    }

    /// Takes the judgment, and records what follows at once from it.
    pub fn step(&mut self, judgment: &Judgment) -> Trace {
        let mut trace = self.judge(judgment);
        trace.append(self.settle());
        trace
    }

    /// What follows at once from what is recorded: cycles of lower bounds and
    /// part lower bounds merged, each relation not yet decomposed
    /// decomposed where its sides are solved, with what that instantiates,
    /// and each use's choice narrowed (`narrow`). The trace holds what
    /// changed the state.
    pub(crate) fn settle(&mut self) -> Trace {
        let mut trace = Trace::default();
        loop {
            let before = self.progress();
            let merged = self.merge_cycles();
            if !merged.is_empty() {
                trace.push(Report::Merged(merged));
            }
            for i in 0..self.classes.items().len() {
                if self.classes.items()[i].decomposed {
                    continue;
                }
                match self.decompose_at(i) {
                    Decomposition::Unsolved => {}
                    decomposition => trace.push(Report::Decomposed {
                        at: i,
                        decomposition,
                    }),
                }
            }
            trace.append(self.narrow());
            if self.progress() == before {
                return trace;
            }
        }
    }

    /// The items recorded, those decomposed, and the candidates that remain
    /// to the uses' choices: what settling changes.
    fn progress(&self) -> (usize, usize, usize) {
        let items = self.classes.items();
        let decomposed = items.iter().filter(|recorded| recorded.decomposed).count();
        let remaining = self
            .classes
            .choices()
            .map(|(_, _, _, choice)| choice.remaining.len())
            .sum();
        (items.len(), decomposed, remaining)
    }

    /// Falls back and records what follows at once from it.
    pub fn fall_back(&mut self) -> Trace {
        let mut trace = Trace {
            reports: self
                .take_defaults()
                .into_iter()
                .map(Report::FellBack)
                .collect(),
        };
        trace.append(self.settle());
        trace
    }

    /// Takes the judgments to the fixpoint and falls back until no class is
    /// left to fall back, taking each time the judgment `choose` picks.
    pub fn run<F>(mut self, mut choose: F) -> Outcome
    where
        F: FnMut(&[Judgment]) -> usize,
    {
        loop {
            let judgments = self.judgments();
            if let Some(judgment) = judgments.get(choose(&judgments)) {
                let judgment = judgment.clone();
                self.step(&judgment);
                continue;
            }
            if !self.fall_back().fell_back() {
                return self.outcome();
            }
        }
    }
}
