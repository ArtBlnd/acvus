//! The state of a binding group's solving (l.208–222), and the composition of
//! the phases on it: recording (`record.rs`), unification (`unify.rs`), the
//! join that gives an equivalence class its value (`value.rs`), splits
//! (`split.rs`), the decisions of uses, lends, projections and bindings
//! (`node/`) in the order the dependencies allow (`dep.rs`), fallback
//! (`fallback.rs`) and the checks (`check.rs`): the steps to a fixpoint and
//! fallback after it (l.295–299), each returning the trace of what its phases
//! reported (`trace.rs`), and the state a node's value read is taken from
//! (l.267).

use std::collections::{BTreeMap, BTreeSet};
use std::rc::Rc;

use crate::check::{Outcome, Refusal};
use crate::class::{Arena, ClassId, ClassName, Classes};
use crate::decl::Decls;
use crate::instance::Made;
use crate::node::NodeState;
use crate::program::{ENTRY, Function, Item, Program, Readings, TextClass};
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
    /// The classes decided when solving starts, which are no unknowns (l.218).
    pub(crate) decided: BTreeSet<ClassId>,
    /// The lower bounds and boundaries, by their place in the log, whose
    /// parts are related.
    pub(crate) split: BTreeSet<usize>,
    /// The classes found to reach a part of their own, or whose sources have
    /// no join.
    pub(crate) no_value: BTreeSet<ClassId>,
    pub(crate) refusals: BTreeSet<Refusal>,
    /// The uses and lends, by name.
    pub(crate) nodes: BTreeMap<Name, NodeState>,
    /// In a state a value read is taken from, the classes found to give
    /// nothing, having no relation left (l.267).
    pub(crate) nothing: BTreeSet<ClassId>,
    pub(crate) readings: Readings,
    /// The classes of the types the host gives (l.282).
    pub(crate) hosts: BTreeSet<ClassId>,
    /// The classes named by a kind "indexes `τ`" already linked along a view
    /// (l.161).
    pub(crate) viewed: BTreeSet<ClassId>,
    /// The decided nodes whose relations wait for the undecided nodes that
    /// broke their dependency on them (l.269).
    pub(crate) deferred: BTreeMap<Name, BTreeSet<Name>>,
    /// For each undecided node, the decided nodes it broke its dependency on,
    /// whose relations its read leaves out (`Deferral::Reading`).
    pub(crate) broken: BTreeMap<Name, BTreeSet<Name>>,
    /// The binding group being solved.
    pub(crate) group: Name,
    /// The types of its functions' parameters, chosen by the uses that make
    /// instances of it (l.217, l.246).
    pub(crate) universals: BTreeSet<ClassId>,
    /// The classes of enclosing groups its text names, shared, not solved
    /// here (l.218, l.222), with those that are unknowns of their own group.
    pub(crate) foreign: BTreeSet<ClassId>,
    pub(crate) enclosing_unknowns: BTreeSet<ClassId>,
    /// The groups enclosing it, innermost first.
    pub(crate) enclosing: Vec<Name>,
    pub(crate) functions: Rc<Vec<Function>>,
    /// The classes the lambdas' environments hold, which the groups
    /// enclosing them solve (`design/17-environment.md`).
    pub(crate) environment: Rc<BTreeSet<Name>>,
    /// Each group solved before it, at its fixpoint (l.220).
    pub(crate) solved: BTreeMap<Name, Rc<State>>,
    /// The instances made here, in the order made (l.222).
    pub(crate) instances: Vec<Made>,
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
    /// The equivalence class takes its constructor and members (l.275).
    Solve(ClassId),
    /// The use or lend is decided by the value it reads (l.267).
    Decide(Name),
}

impl State {
    /// The state of the run's entry where the text's constraints are recorded
    /// and nothing is solved (l.230), the program having no other group.
    pub fn new(program: &Program) -> State {
        State::of_group(program, &Name::new(ENTRY), &BTreeMap::new())
    }

    /// The state of the binding group `group` where its text's constraints
    /// are recorded and nothing is solved (l.216, l.230). `solved` holds each
    /// group its uses make instances of, at its fixpoint.
    pub fn of_group(program: &Program, group: &Name, solved: &BTreeMap<Name, Rc<State>>) -> State {
        State::start(program, group, solved).state
    }

    /// As `of_group`, with the trace of what its phases reported: the text's
    /// relations recorded, the nodes decided as they are recorded, and what
    /// follows at once.
    pub fn start(program: &Program, group: &Name, solved: &BTreeMap<Name, Rc<State>>) -> Started {
        let mut state = State {
            decls: Rc::new(program.decls.clone()),
            arena: Arena::default(),
            classes: Classes::default(),
            log: Vec::new(),
            texts: Vec::new(),
            decided: BTreeSet::new(),
            split: BTreeSet::new(),
            no_value: BTreeSet::new(),
            refusals: BTreeSet::new(),
            nodes: BTreeMap::new(),
            nothing: BTreeSet::new(),
            readings: program.readings,
            hosts: BTreeSet::new(),
            viewed: BTreeSet::new(),
            deferred: BTreeMap::new(),
            broken: BTreeMap::new(),
            group: group.clone(),
            universals: BTreeSet::new(),
            foreign: BTreeSet::new(),
            enclosing_unknowns: BTreeSet::new(),
            enclosing: program.enclosing(group),
            functions: Rc::new(program.functions.clone()),
            environment: Rc::new(program.environment()),
            solved: solved.clone(),
            instances: Vec::new(),
        };
        let classes: Vec<&TextClass> = program
            .classes
            .iter()
            .filter(|class| program.owner(&class.name) == group)
            .collect();
        let items: Vec<(usize, &Item)> = program
            .items
            .iter()
            .enumerate()
            .filter(|&(i, _)| program.item_group(i) == group)
            .map(|(i, (_, item))| (i, item))
            .collect();
        for class in &classes {
            let id = state.arena.intern(ClassName::Text(class.name.clone()));
            state.texts.push(id);
        }
        state.classes.grow(&state.arena);
        let mut named: Vec<&Name> = items.iter().flat_map(|(_, item)| item.classes()).collect();
        for (_, item) in &items {
            if let Item::Use { callee, .. } = item
                && let Some(function) = program.function(callee)
                && state.enclosing.contains(&function.group)
            {
                named.extend(function.params.iter().chain([&function.result]));
            }
        }
        for name in named {
            if state.arena.get(&ClassName::Text(name.clone())).is_none() {
                state.share(name, program.is_text_unknown(name));
            }
        }
        for function in program.functions.iter().filter(|f| f.group == *group) {
            for param in function
                .params
                .iter()
                .filter(|param| program.is_universal(param))
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

    /// A class of an enclosing group, shared here (l.222), and whether it is
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

    /// A class not universal, not decided when solving starts, and not of an
    /// enclosing group (l.218).
    pub(crate) fn is_unknown(&self, id: ClassId) -> bool {
        !self.decided.contains(&id) && !self.is_universal(id) && !self.is_foreign(id)
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

    /// The steps that may be taken now.
    pub fn steps(&self) -> Vec<Step> {
        let (classes, nodes) = self.eligible();
        classes
            .into_iter()
            .map(Step::Solve)
            .chain(nodes.into_iter().map(Step::Decide))
            .collect()
    }

    /// Takes the step, and records what follows at once from it.
    pub fn step(&mut self, step: &Step) -> Trace {
        let mut trace = match step {
            Step::Solve(root) => self.solve(*root),
            Step::Decide(name) => Trace {
                reports: vec![Report::Decided(self.decide(name))],
            },
        };
        trace.append(self.settle());
        trace
    }

    /// The equivalence class is solved (l.275), unless it reaches a part of
    /// its own: its LitVar lower bounds joined, `?N ⇝ U` where it holds a
    /// LitVar alone (l.291), and then its constructor and members. The trace
    /// holds what the class took, and the LitVars unified where any are.
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
        if !lowers.is_empty() {
            trace.push(Report::LitLowers {
                class: root,
                recorded: lowers,
            });
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

    /// What follows at once from what is recorded: lower-bound cycles merged
    /// (l.275), the parts of each lower bound and boundary whose two sides are
    /// solved related (l.284), and a LitVar lower bound of a solved class
    /// unified with it (l.176). The trace holds what changed the state.
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
            let shared = self.record_at_once();
            if !shared.is_empty() {
                trace.push(Report::Shared(shared));
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

    /// The lower bounds of the equivalence class, other than itself.
    pub(crate) fn lower_bounds(&self, id: ClassId) -> Vec<ClassId> {
        let root = self.root(id);
        self.log
            .iter()
            .filter_map(|recorded| match recorded.relation {
                Relation::LowerBound { lower, upper }
                    if self.root(upper) == root && self.root(lower) != root =>
                {
                    Some(lower)
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

    /// Falls back (l.299) and records what follows at once from it.
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

    /// The state a node's value read is taken from (l.267): the relations
    /// recorded by neither the nodes `excluded` nor solving, solved where
    /// each class's value is decided by its sources whatever its other
    /// dependencies, no node decided, nothing fallen back, and a class with
    /// no relation left giving nothing. Classes are solved in one order here.
    pub(crate) fn read_state(&self, excluded: &BTreeSet<Name>) -> State {
        let mut read = State {
            decls: Rc::clone(&self.decls),
            arena: self.arena.clone(),
            classes: Classes::default(),
            log: Vec::new(),
            texts: self.texts.clone(),
            decided: self.decided.clone(),
            split: BTreeSet::new(),
            no_value: BTreeSet::new(),
            refusals: BTreeSet::new(),
            nodes: BTreeMap::new(),
            nothing: BTreeSet::new(),
            readings: self.readings,
            hosts: self.hosts.clone(),
            viewed: BTreeSet::new(),
            deferred: BTreeMap::new(),
            broken: BTreeMap::new(),
            group: self.group.clone(),
            universals: self.universals.clone(),
            foreign: self.foreign.clone(),
            enclosing_unknowns: self.enclosing_unknowns.clone(),
            enclosing: self.enclosing.clone(),
            functions: Rc::clone(&self.functions),
            environment: Rc::clone(&self.environment),
            solved: self.solved.clone(),
            instances: Vec::new(),
        };
        read.classes.grow(&read.arena);
        for recorded in &self.log {
            if recorded.recorder.is_derived() || left_out_by(&recorded.recorder, excluded) {
                continue;
            }
            let _ = read.record(recorded.relation.clone(), recorded.recorder.clone());
        }
        read.settle();
        loop {
            if let Some(&root) = read.eligible().0.first() {
                read.solve(root);
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
/// leaves out a relation `recorder` recorded (l.267): one of those nodes
/// recorded it.
pub(crate) fn left_out_by(recorder: &Recorder, excluded: &BTreeSet<Name>) -> bool {
    recorder.nodes().iter().any(|node| excluded.contains(node))
}
