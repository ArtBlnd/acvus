//! The state of a binding group's solving, and the composition of
//! the phases on it: recording (`record.rs`), unification (`unify.rs`), the
//! join that gives an equivalence class its value and the fill (`value.rs`),
//! splits (`split.rs`), the decisions of uses, lends, projections and
//! bindings (`node/`) in the order the dependencies allow (`dep.rs`),
//! fallback (`fallback.rs`) and the checks (`check.rs`): the steps to a
//! fixpoint and fallback after it, each returning the trace of
//! what its phases reported (`trace.rs`), and the state a node's value read
//! is taken from.

use std::collections::{BTreeMap, BTreeSet};
use std::rc::Rc;

use crate::check::{Outcome, Refusal};
use crate::class::{Arena, ClassId, ClassName, Classes};
use crate::cut::Cut;
use crate::decl::Decls;
use crate::dep::Eligible;
use crate::instance::Made;
use crate::node::{LitDefaults, NodeState};
use crate::program::{ENTRY, Function, Item, Part, Position, Program, TextClass};
use crate::read::ReadValue;
use crate::relation::{Recorded, Recorder, Relation};
use crate::split::Split;
use crate::trace::{Report, Trace};
use crate::ty::{HasTerm, Name};

#[derive(Clone, Debug)]
pub struct State {
    pub(crate) decls: Rc<Decls>,
    pub(crate) arena: Arena,
    pub(crate) classes: Classes,
    pub(crate) log: Vec<Recorded>,
    /// The classes the text gives, in the program's order.
    pub(crate) texts: Vec<ClassId>,
    /// The classes decided when solving starts, which are no unknowns.
    pub(crate) decided: BTreeSet<ClassId>,
    /// The lower bounds, part lower bounds and boundaries, by their place in
    /// the log, whose parts are related.
    pub(crate) split: BTreeSet<usize>,
    /// Those copied from a C where they were split, by their place in the log:
    /// what their splits recorded is copied with them, so that no state splits
    /// them again.
    pub(crate) copied_split: BTreeSet<usize>,
    /// The classes found to reach a part of their own, or whose sources have
    /// no join.
    pub(crate) no_value: BTreeSet<ClassId>,
    pub(crate) refusals: BTreeSet<Refusal>,
    /// The uses and lends, by name.
    pub(crate) nodes: BTreeMap<Name, NodeState>,
    /// In a state a value read is taken from, the classes found to give
    /// nothing, having no relation left.
    pub(crate) nothing: BTreeSet<ClassId>,
    /// The classes named by a kind "indexes `τ`" already linked along a view.
    pub(crate) viewed: BTreeSet<ClassId>,
    /// The decided nodes whose relations wait for the undecided nodes that
    /// broke their dependency on them.
    pub(crate) deferred: BTreeMap<Name, BTreeSet<Name>>,
    /// The binding group being solved.
    pub(crate) group: Name,
    /// The types of its functions' parameters, chosen by the uses that make
    /// instances of it.
    pub(crate) universals: BTreeSet<ClassId>,
    /// The classes of enclosing groups its text names, shared, not solved
    /// here, with those that are unknowns of their own group.
    pub(crate) foreign: BTreeSet<ClassId>,
    pub(crate) enclosing_unknowns: BTreeSet<ClassId>,
    /// The groups enclosing it, innermost first.
    pub(crate) enclosing: Vec<Name>,
    pub(crate) functions: Rc<Vec<Function>>,
    /// The classes the lambdas' environments hold, which the groups
    /// enclosing them solve.
    pub(crate) environment: Rc<BTreeSet<Name>>,
    /// Each group solved before it, at its fixpoint.
    pub(crate) solved: BTreeMap<Name, Rc<State>>,
    /// The instances made here, in the order made.
    pub(crate) instances: Vec<Made>,
    /// The lambdas an instance made anew, by name: each a copy of a lambda
    /// placed in the group it copied, whose environment is that instance's
    /// copies.
    pub(crate) lambdas: BTreeMap<Name, LambdaCopy>,
}

/// A lambda an instance made anew: the instance, the lambda of the
/// program it copies, and the classes here its environment holds, its
/// parameter types and then its result type, none where its own binding group
/// holds them.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct LambdaCopy {
    pub at: Name,
    pub of: Name,
    pub environment: Vec<Option<ClassId>>,
}

/// A binding group's state where its text's constraints are recorded and
/// nothing is solved, with the trace of the phases that made it.
pub struct Started {
    pub state: State,
    pub trace: Trace,
}

/// A step that may be taken.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub enum Step {
    /// The equivalence class takes its constructor and members.
    Solve(ClassId),
    /// The equivalence class nothing enters takes the one type the places it
    /// passes into give, or has no value where they give two.
    Fill(ClassId),
    /// The use or lend is decided by the value it reads.
    Decide(Name),
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
    /// relations recorded, the nodes decided as they are recorded, and what
    /// follows at once.
    pub fn start(program: &Program, group: &Name, solved: &BTreeMap<Name, Rc<State>>) -> Started {
        // The groups solved so far cut the lambdas' environments up to this
        // group.
        let cut = Cut::of(program, solved, group);
        let mut state = State {
            decls: Rc::new(program.decls.clone()),
            arena: Arena::default(),
            classes: Classes::default(),
            log: Vec::new(),
            texts: Vec::new(),
            decided: BTreeSet::new(),
            split: BTreeSet::new(),
            copied_split: BTreeSet::new(),
            no_value: BTreeSet::new(),
            refusals: BTreeSet::new(),
            nodes: BTreeMap::new(),
            nothing: BTreeSet::new(),
            viewed: BTreeSet::new(),
            deferred: BTreeMap::new(),
            group: group.clone(),
            universals: BTreeSet::new(),
            foreign: BTreeSet::new(),
            enclosing_unknowns: BTreeSet::new(),
            enclosing: program.enclosing(group),
            functions: Rc::new(program.functions.clone()),
            environment: Rc::new(cut.environment()),
            solved: solved.clone(),
            instances: Vec::new(),
            lambdas: BTreeMap::new(),
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
        for class in &classes {
            let id = state.arena.intern(ClassName::Text(class.name.clone()));
            state.texts.push(id);
        }
        state.classes.grow(&state.arena);
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
                state.share(name, cut.is_text_unknown(program, name));
            }
        }
        for function in program.functions.iter().filter(|f| f.group == *group) {
            for param in function
                .params
                .iter()
                .filter(|param| cut.is_universal(program, param))
            {
                let id = state.text(param);
                state.universals.insert(id);
            }
        }
        for class in &classes {
            if program.universals.contains(&class.name) {
                let id = state.text(&class.name);
                state.universals.insert(id);
            }
        }
        let mut trace = Trace::default();
        trace.push(Report::Recorded(state.record_origins(program, &classes)));
        trace.push(Report::Recorded(state.record_equations(&items)));
        state.decided = state
            .texts
            .iter()
            .copied()
            .filter(|&id| state.classes.held(id).kind.is_some())
            .collect();
        trace.push(Report::Recorded(
            state.record_items(program, &items, solved),
        ));
        for decided in state.decide_at_once() {
            trace.push(Report::Decided(decided));
        }
        trace.append(state.settle());
        Started { state, trace }
    }

    pub(crate) fn text(&self, name: &Name) -> ClassId {
        self.arena
            .get(&ClassName::Text(name.clone()))
            .unwrap_or_else(|| panic!("the program names the class {name}"))
    }

    /// A class of an enclosing group, shared here, and whether it is
    /// an unknown of its own group.
    pub(crate) fn share(&mut self, name: &Name, unknown: bool) -> ClassId {
        let id = self.arena.intern(ClassName::Text(name.clone()));
        self.classes.grow(&self.arena);
        self.foreign.insert(id);
        if unknown {
            self.enclosing_unknowns.insert(id);
        }
        id
    }

    /// An equivalence class not universal, not decided when solving starts,
    /// and not of an enclosing group: none of its members is.
    pub(crate) fn is_unknown(&self, id: ClassId) -> bool {
        let decided = self
            .classes
            .held(id)
            .members
            .iter()
            .any(|member| self.decided.contains(member));
        !decided && !self.is_universal(id) && !self.is_foreign(id)
    }

    pub(crate) fn is_foreign(&self, id: ClassId) -> bool {
        self.classes
            .held(id)
            .members
            .iter()
            .any(|member| self.foreign.contains(member))
    }

    pub(crate) fn is_universal(&self, id: ClassId) -> bool {
        self.classes
            .held(id)
            .members
            .iter()
            .any(|member| self.universals.contains(member))
    }

    pub(crate) fn root(&self, id: ClassId) -> ClassId {
        self.classes.find(id)
    }

    pub(crate) fn is_solved(&self, id: ClassId) -> bool {
        self.classes.held(id).solved
    }

    /// Holds a LitVar and no kind.
    pub(crate) fn is_lit(&self, id: ClassId) -> bool {
        let held = self.classes.held(id);
        held.lit.is_some() && held.kind.is_none()
    }

    pub(crate) fn has_no_value(&self, id: ClassId) -> bool {
        self.classes
            .held(id)
            .members
            .iter()
            .any(|member| self.no_value.contains(member))
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

    /// The steps that may be taken now: each eligible class solved or filled,
    /// and each eligible node decided where the value it reads now decides it,
    /// a node whose value decides nothing staying undecided.
    /// Where no such step is left, each eligible node its LitVars' defaults
    /// decide, as a class falls back only then.
    pub fn steps(&self) -> Vec<Step> {
        let Eligible {
            solve,
            fill,
            decide,
        } = self.eligible();
        let decided = |defaults: LitDefaults| -> Vec<Step> {
            decide
                .iter()
                .filter(|name| self.read_now(name, defaults).decision.is_some())
                .cloned()
                .map(Step::Decide)
                .collect()
        };
        let steps: Vec<Step> = solve
            .into_iter()
            .map(Step::Solve)
            .chain(fill.into_iter().map(Step::Fill))
            .chain(decided(LitDefaults::Withheld))
            .collect();
        if steps.is_empty() {
            return decided(LitDefaults::Taken);
        }
        steps
    }

    /// Whether a use reads its LitVars as their defaults now: where no step
    /// is left without them.
    #[cfg(test)]
    pub(crate) fn lit_defaults_now(&self) -> LitDefaults {
        let Eligible {
            solve,
            fill,
            decide,
        } = self.eligible();
        let without = !solve.is_empty()
            || !fill.is_empty()
            || decide.iter().any(|name| {
                self.read_now(name, LitDefaults::Withheld)
                    .decision
                    .is_some()
            });
        match without {
            true => LitDefaults::Withheld,
            false => LitDefaults::Taken,
        }
    }

    /// Takes the step, and records what follows at once from it.
    pub fn step(&mut self, step: &Step) -> Trace {
        let mut trace = match step {
            Step::Solve(root) => self.solve(*root),
            Step::Fill(root) => Trace {
                reports: vec![Report::Filled(self.fill(*root))],
            },
            Step::Decide(name) => Trace {
                reports: vec![Report::Decided(self.decide(name))],
            },
        };
        trace.append(self.settle());
        trace
    }

    /// The equivalence class is solved, unless it reaches a part of
    /// its own: its LitVar lower bounds joined, `?N ⇝ U` where it holds a
    /// LitVar alone, and then its constructor and members. Where it
    /// holds no kind and its LitVar lower bounds' join conflicts, its sources
    /// have no join. The trace holds what the class took, and the LitVars
    /// unified where any are.
    pub(crate) fn solve(&mut self, root: ClassId) -> Trace {
        if let Some(refusal) = self.refuse_infinite(root) {
            return Trace {
                reports: vec![Report::Infinite {
                    class: root,
                    refusal,
                }],
            };
        }
        let mut trace = Trace::default();
        let lowers = self.join_lit_lowers(root);
        let conflicted = lowers
            .entries
            .iter()
            .find_map(|entry| entry.unification.refused.clone());
        if !lowers.is_empty() {
            trace.push(Report::LitLowers {
                class: root,
                recorded: lowers,
            });
        }
        let class = self.root(root);
        if let Some(refusal) = conflicted.filter(|_| self.classes.held(class).kind.is_none()) {
            let joined = self.lit_no_join(class, refusal);
            trace.push(Report::Joined { class, joined });
            return trace;
        }
        let boundaries = self.unify_lit_boundaries(root);
        if !boundaries.is_empty() {
            trace.push(Report::LitBoundaries {
                class: root,
                recorded: boundaries,
            });
        }
        let class = self.root(root);
        let joined = self.join(root);
        trace.push(Report::Joined { class, joined });
        trace
    }

    /// What follows at once from what is recorded: cycles of lower bounds and
    /// part lower bounds merged, the parts of each lower bound, part
    /// lower bound and boundary whose two sides are solved related,
    /// and a LitVar lower bound of a solved class unified with it.
    /// The trace holds what changed the state.
    pub(crate) fn settle(&mut self) -> Trace {
        let mut trace = Trace::default();
        loop {
            let before = (self.log.len(), self.split.len());
            let merged = self.merge_cycles();
            if !merged.is_empty() {
                trace.push(Report::Merged(merged));
            }
            for linked in self.link_views() {
                trace.push(Report::Linked(linked));
            }
            let common = self.record_at_once();
            if !common.is_empty() {
                trace.push(Report::Common(common));
            }
            let settled = self.settled();
            for i in 0..self.log.len() {
                if self.split.contains(&i) {
                    continue;
                }
                match self.split_at(i) {
                    Split::Unsolved => {
                        if let Some(recorded) = self.join_lit_at(i, &settled) {
                            trace.push(Report::LitJoined { bound: i, recorded });
                        }
                    }
                    split => trace.push(Report::Split { at: i, split }),
                }
            }
            if (self.log.len(), self.split.len()) == before {
                return trace;
            }
        }
    }

    /// The lower bounds and part lower bounds of the equivalence class, other
    /// than itself.
    pub(crate) fn lower_bounds(&self, id: ClassId) -> Vec<ClassId> {
        let root = self.root(id);
        self.log
            .iter()
            .filter_map(|recorded| match recorded.relation {
                Relation::LowerBound { lower, upper }
                | Relation::PartLowerBound { lower, upper }
                    if self.root(upper) == root && self.root(lower) != root =>
                {
                    Some(lower)
                }
                _ => None,
            })
            .collect()
    }

    /// The places the equivalence class passes into: the `α` of each
    /// lower bound and part lower bound whose `τ` it is, and the `U` of each
    /// boundary whose `T` it is, other than itself.
    pub(crate) fn places(&self, id: ClassId) -> Vec<ClassId> {
        let root = self.root(id);
        self.log
            .iter()
            .filter_map(|recorded| match recorded.relation {
                Relation::LowerBound { lower, upper }
                | Relation::PartLowerBound { lower, upper }
                    if self.root(lower) == root && self.root(upper) != root =>
                {
                    Some(upper)
                }
                Relation::Boundary { from, to }
                    if self.root(from) == root && self.root(to) != root =>
                {
                    Some(to)
                }
                _ => None,
            })
            .collect()
    }

    /// The types the equivalence class passes into by a boundary.
    pub(crate) fn boundaries_from(&self, id: ClassId) -> Vec<ClassId> {
        let root = self.root(id);
        self.log
            .iter()
            .filter_map(|recorded| match recorded.relation {
                Relation::Boundary { from, to }
                    if self.root(from) == root && self.root(to) != root =>
                {
                    Some(to)
                }
                _ => None,
            })
            .collect()
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

    /// Solves to the fixpoint and falls back until no class is left to fall
    /// back, taking at each step the first step `choose` picks.
    pub fn run<F>(mut self, mut choose: F) -> Outcome
    where
        F: FnMut(&[Step]) -> usize,
    {
        loop {
            let steps = self.steps();
            if let Some(step) = steps.get(choose(&steps)) {
                let step = step.clone();
                self.step(&step);
                continue;
            }
            if !self.fall_back().fell_back() {
                return self.outcome();
            }
        }
    }

    /// The state a node's value read is taken from: the relations
    /// recorded by neither the nodes `excluded` nor solving, solved and filled
    /// where each class's value is decided by the table of values, splits and
    /// fills whatever its other dependencies, no node decided, nothing fallen
    /// back, and a class with no relation left giving nothing. Classes are
    /// solved and filled in one order here.
    pub(crate) fn read_state(&self, excluded: &BTreeSet<Name>) -> State {
        let mut read = State {
            decls: Rc::clone(&self.decls),
            arena: self.arena.clone(),
            classes: Classes::default(),
            log: Vec::new(),
            texts: self.texts.clone(),
            decided: self.decided.clone(),
            split: BTreeSet::new(),
            copied_split: BTreeSet::new(),
            no_value: BTreeSet::new(),
            refusals: BTreeSet::new(),
            nodes: BTreeMap::new(),
            nothing: BTreeSet::new(),
            viewed: BTreeSet::new(),
            deferred: BTreeMap::new(),
            group: self.group.clone(),
            universals: self.universals.clone(),
            foreign: self.foreign.clone(),
            enclosing_unknowns: self.enclosing_unknowns.clone(),
            enclosing: self.enclosing.clone(),
            functions: Rc::clone(&self.functions),
            environment: Rc::clone(&self.environment),
            solved: self.solved.clone(),
            instances: Vec::new(),
            lambdas: self.lambdas.clone(),
        };
        read.classes.grow(&read.arena);
        for (at, recorded) in self.log.iter().enumerate() {
            if recorded.recorder.is_derived() || left_out_by(&recorded.recorder, excluded) {
                continue;
            }
            let entry = read.record(recorded.relation.clone(), recorded.recorder.clone());
            if self.copied_split.contains(&at) {
                read.split.insert(entry.at);
                read.copied_split.insert(entry.at);
            }
        }
        read.settle();
        loop {
            let eligible = read.eligible();
            if let Some(&root) = eligible.solve.first() {
                read.solve(root);
                read.settle();
                continue;
            }
            if let Some(&root) = eligible.fill.first() {
                read.fill(root);
                read.settle();
                continue;
            }
            let more: Vec<ClassId> = read
                .fallback_targets()
                .into_iter()
                .filter(|&root| !read.gives_nothing(root))
                .collect();
            if more.is_empty() {
                return read;
            }
            read.nothing.extend(more);
        }
    }

    /// The value the class has here: a position with no kind is undecided,
    /// and one holding only a LitVar is that LitVar.
    pub(crate) fn read_value(&self, id: ClassId) -> ReadValue {
        self.read_value_in(id, &mut BTreeSet::new())
    }

    fn read_value_in(&self, id: ClassId, visiting: &mut BTreeSet<ClassId>) -> ReadValue {
        let root = self.root(id);
        let held = self.classes.held(root);
        if self.has_no_value(root) || !visiting.insert(root) {
            return ReadValue::Undecided;
        }
        let value = match (&held.kind, &held.lit) {
            (Some((kind, _)), _) => {
                let term = kind
                    .clone()
                    .map(|_, part| self.read_value_in(part, visiting));
                ReadValue::from_term(term).unwrap_or(ReadValue::Undecided)
            }
            (None, Some(lit)) => ReadValue::Lit(lit.clone()),
            (None, None) => ReadValue::Undecided,
        };
        visiting.remove(&root);
        value
    }
}

/// Whether a value read leaving out the relations of the nodes `excluded`
/// leaves out a relation `recorder` recorded: one of those nodes
/// recorded it.
pub(crate) fn left_out_by(recorder: &Recorder, excluded: &BTreeSet<Name>) -> bool {
    recorder.nodes().iter().any(|node| excluded.contains(node))
}
