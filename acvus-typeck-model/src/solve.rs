//! Steps, state, fixpoint, fallback and acceptance (l.295–301), with the
//! splits a class taking a constructor records (l.284–291).

use std::collections::{BTreeMap, BTreeSet};
use std::rc::Rc;

use crate::class::{Arena, ClassId, ClassName, Classes, Conflict, Fact, KindSource, LinkAt};
use crate::coerce::{CoerceError, Coercion, along_le, coerce};
use crate::decl::{Decls, Node};
use crate::instance::Instance;
use crate::node::{Decision, NodeKind, NodeState};
use crate::program::{ENTRY, Function, Item, Origin, Place, Program, Readings, TextClass};
use crate::relation::{Named, Recorded, Recorder, Relation};
use crate::sig::SigTy;
use crate::ty::{Mutability, Name, Term, Ty, Variance};
use crate::value::{Head, head};

/// Where a definition gives no result (l.301).
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub enum Refusal {
    /// No type holds what a relation made one equivalence class hold.
    Conflict { at: String, conflict: String },
    /// The types an equivalence class's sources give have no join.
    NoJoin(ClassName),
    /// The equivalence class reaches a part of its own (l.275).
    Infinite(ClassName),
    /// The equivalence class has no value once solving ends.
    NoValue(ClassName),
    /// A type the text writes disagrees with its declaration.
    Undeclared(String),
}

/// A relation an entering value is checked by once everything is solved
/// (l.291, l.293), with the coercion that changes it.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Check {
    pub relation: String,
    pub coercion: Result<Coercion, CoerceError>,
}

/// What solving decides (I3).
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Outcome {
    pub values: BTreeMap<ClassName, Option<Ty>>,
    pub decisions: BTreeMap<Name, Option<Decision>>,
    pub partition: BTreeSet<BTreeSet<ClassName>>,
    pub checks: Vec<Check>,
    pub refusals: BTreeSet<Refusal>,
    pub accepted: bool,
}

/// How a relation's two sides are related part by part (l.286).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Side {
    Lower,
    Boundary,
}

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
    /// Each group solved before it, at its fixpoint (l.220).
    pub(crate) solved: BTreeMap<Name, Rc<State>>,
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
            solved: solved.clone(),
        };
        let classes: Vec<&TextClass> = program
            .classes
            .iter()
            .filter(|class| class.group == *group)
            .collect();
        let items: Vec<(usize, &Item)> = program
            .items
            .iter()
            .enumerate()
            .filter(|(_, (at, _))| at == group)
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
            for param in &function.params {
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
        for class in &classes {
            let id = state.text(&class.name);
            match &class.origin {
                Origin::Var => {}
                Origin::Lit(lit) => state.record(Relation::Lit(id, lit.clone()), Recorder::Origin),
                Origin::Written(term) => {
                    let term = term.clone().map(|_, part| state.text(&part));
                    let relation = Relation::Kind(id, term, KindSource::Written);
                    state.record(relation, Recorder::Origin);
                }
                Origin::Host(term) => {
                    state.hosts.insert(id);
                    let term = term.clone().map(|_, part| state.text(&part));
                    let relation = Relation::Kind(id, term, KindSource::Written);
                    state.record(relation, Recorder::Origin);
                }
                Origin::Lambda(lambda) => {
                    let function = program.function(lambda).expect("the lambda is a function");
                    let arity = function.params.len();
                    let term = Term::FunctionValue { arity };
                    let relation = Relation::Kind(id, term, KindSource::Written);
                    state.record(relation, Recorder::Origin);
                    state.record(Relation::Values(id, lambda.clone()), Recorder::Origin);
                }
            }
        }
        for &(i, item) in &items {
            if let Item::Equation(a, b) = item {
                let relation = Relation::Equation(state.text(a), state.text(b));
                state.record(relation, Recorder::Text(i));
            }
        }
        state.decided = state
            .texts
            .iter()
            .copied()
            .filter(|&id| state.classes.held(id).kind.is_some())
            .collect();
        for &(i, item) in &items {
            match item {
                Item::Pass { value, into, place } => {
                    let (value, into) = (state.text(value), state.text(into));
                    let meeting = match place {
                        Place::ThroughRef => false,
                        Place::ByName => {
                            state.is_unknown(into) || state.enclosing_unknowns.contains(&into)
                        }
                        Place::Slot => state.is_unknown(into),
                    };
                    let relation = if meeting {
                        Relation::LowerBound {
                            lower: value,
                            upper: into,
                        }
                    } else {
                        Relation::Boundary {
                            from: value,
                            to: into,
                        }
                    };
                    state.record(relation, Recorder::Text(i));
                }
                Item::CallValue {
                    name,
                    callee,
                    args,
                    result,
                } => {
                    let kind = NodeKind::CallValue {
                        callee: state.text(callee),
                        args: args.iter().map(|arg| state.text(arg)).collect(),
                        result: state.text(result),
                    };
                    state.add_node(name, kind);
                }
                Item::Use {
                    callee,
                    args,
                    result,
                    ..
                } if program.function(callee).is_some() => {
                    let function = program.function(callee).expect("the function is declared");
                    let args: Vec<ClassId> = args.iter().map(|arg| state.text(arg)).collect();
                    let result = state.text(result);
                    state.call(i, function, &args, result, solved);
                }
                Item::Use {
                    name,
                    callee,
                    args,
                    result,
                } => {
                    let kind = NodeKind::Use {
                        callee: callee.clone(),
                        args: args.iter().map(|arg| state.text(arg)).collect(),
                        result: state.text(result),
                    };
                    state.add_node(name, kind);
                }
                Item::Lend {
                    name,
                    mutability,
                    place,
                    result,
                } => {
                    let kind = NodeKind::Lend {
                        mutability: *mutability,
                        place: state.text(place),
                        result: state.text(result),
                    };
                    state.add_node(name, kind);
                }
                Item::Project {
                    name,
                    projection,
                    operand,
                    result,
                } => {
                    let kind = NodeKind::Project {
                        projection: projection.clone(),
                        operand: state.text(operand),
                        result: state.text(result),
                    };
                    state.add_node(name, kind);
                }
                Item::Bind {
                    name,
                    path,
                    position,
                    binding,
                } => {
                    let kind = NodeKind::Bind {
                        path: path.clone(),
                        position: state.text(position),
                        binding: state.text(binding),
                    };
                    state.add_node(name, kind);
                }
                Item::Equation(..) => {}
            }
        }
        state.decide_at_once();
        state.settle();
        state
    }

    pub(crate) fn add_node(&mut self, name: &Name, kind: NodeKind) {
        let node = NodeState {
            kind,
            decision: None,
            shared: BTreeSet::new(),
            references: BTreeSet::new(),
        };
        if self.nodes.insert(name.clone(), node).is_some() {
            panic!("the program names the node {name} once");
        }
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

    /// A use that calls a function (l.199–202). Within the function's group it
    /// takes that body's own types: each argument passes into its parameter's
    /// type, universal and so a boundary (Example 10), and the body's result
    /// type enters the use's result. From another group it makes an instance
    /// of the group's C (l.222).
    fn call(
        &mut self,
        item: usize,
        function: &Function,
        args: &[ClassId],
        result: ClassId,
        solved: &BTreeMap<Name, Rc<State>>,
    ) {
        let recorder = Recorder::Text(item);
        if function.group == self.group || self.enclosing.contains(&function.group) {
            let params: Vec<ClassId> = function.params.iter().map(|p| self.text(p)).collect();
            let returned = self.text(&function.result);
            self.record_call(args, &params, returned, result, &recorder);
            return;
        }
        let from = Rc::clone(
            solved
                .get(&function.group)
                .unwrap_or_else(|| panic!("{} is solved first (l.220)", function.group)),
        );
        let at = Name::new(&format!("{}@{item}", function.name));
        let mut copy = Instance::new(&at, &from);
        let params: Vec<ClassId> = function
            .params
            .iter()
            .map(|param| copy.class(self, from.text(param)))
            .collect();
        let returned = copy.class(self, from.text(&function.result));
        copy.record_rest(self);
        self.record_call(args, &params, returned, result, &recorder);
    }

    pub(crate) fn record_call(
        &mut self,
        args: &[ClassId],
        params: &[ClassId],
        returned: ClassId,
        result: ClassId,
        recorder: &Recorder,
    ) {
        for (&arg, &param) in args.iter().zip(params) {
            let relation = if self.is_unknown(param) {
                Relation::LowerBound {
                    lower: arg,
                    upper: param,
                }
            } else {
                Relation::Boundary {
                    from: arg,
                    to: param,
                }
            };
            self.record(relation, recorder.clone());
        }
        self.record(
            Relation::LowerBound {
                lower: returned,
                upper: result,
            },
            recorder.clone(),
        );
    }

    pub(crate) fn record(&mut self, relation: Relation, recorder: Recorder) {
        let decls = Rc::clone(&self.decls);
        let types = decls.types();
        let given = match &relation {
            Relation::Equation(a, b) => self.classes.unify(*a, *b, types),
            Relation::Fact(class, fact) => self.classes.give_fact(*class, fact.clone(), types),
            Relation::Kind(class, term, source) => match types.check(term) {
                Ok(()) => {
                    let given = self.classes.give_kind(*class, term.clone(), *source, types);
                    self.classes.mark_solved(*class);
                    given
                }
                Err(error) => {
                    self.refusals
                        .insert(Refusal::Undeclared(format!("{error:?}")));
                    Ok(())
                }
            },
            Relation::Lit(class, lit) => self.classes.give_lit(*class, lit.clone(), types),
            Relation::Reference(class, reference) => {
                self.classes.give_reference(*class, *reference, types)
            }
            Relation::Values(class, lambda) => {
                self.classes.give_value(*class, lambda.clone());
                Ok(())
            }
            Relation::LowerBound { .. } | Relation::Boundary { .. } => Ok(()),
        };
        if let Err(conflict) = given {
            let at = Named {
                relation: &relation,
                arena: &self.arena,
            }
            .to_string();
            self.refuse_conflict(at, conflict);
        }
        self.log.push(Recorded { relation, recorder });
    }

    fn refuse_conflict(&mut self, at: String, conflict: Conflict) {
        self.refusals.insert(Refusal::Conflict {
            at,
            conflict: format!("{conflict:?}"),
        });
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
    pub fn step(&mut self, step: &Step) {
        match step {
            Step::Solve(root) => self.solve(*root),
            Step::Decide(name) => self.decide(name),
        }
        self.settle();
    }

    pub(crate) fn solve(&mut self, root: ClassId) {
        if self.reaches_own_part(root) {
            let name = self.class_name(root);
            let members = self.classes.held(root).members.clone();
            self.no_value.extend(members);
            self.refusals.insert(Refusal::Infinite(name));
            return;
        }
        let settled = self.settled();
        for lower in self.lower_bounds(root) {
            if self.is_lit(lower) && settled.contains(&self.root(lower)) {
                self.record(Relation::Equation(lower, root), Recorder::Split(root));
            }
        }
        if self.is_lit(root) {
            for to in self.boundaries_from(root) {
                self.record(Relation::Equation(root, to), Recorder::Split(root));
            }
        }
        let root = self.root(root);
        if self.classes.held(root).kind.is_none() {
            let mut sources: Vec<Head> = self
                .lower_bounds(root)
                .into_iter()
                .filter(|&lower| self.is_solved(lower))
                .filter_map(|lower| self.classes.held(lower).kind.as_ref().map(|(k, _)| head(k)))
                .collect();
            sources.extend(self.fact_heads(root));
            let Some(first) = sources.first().cloned() else {
                return;
            };
            let joined = sources
                .iter()
                .skip(1)
                .try_fold(first, |joined, source| self.head_join(&joined, source));
            let Some(joined) = joined else {
                let name = self.class_name(root);
                let members = self.classes.held(root).members.clone();
                self.no_value.extend(members);
                self.refusals.insert(Refusal::NoJoin(name));
                return;
            };
            let owner = self.class_name(root);
            let kind = self.skeleton(&joined, &owner);
            let decls = Rc::clone(&self.decls);
            if let Err(conflict) =
                self.classes
                    .give_kind(root, kind, KindSource::Solved, decls.types())
            {
                self.refuse_conflict(format!("the value of {owner}"), conflict);
            }
        }
        self.classes.mark_solved(root);
    }

    /// What follows at once from what is recorded: lower-bound cycles merged
    /// (l.275), the parts of each lower bound and boundary whose two sides are
    /// solved related (l.284), and a LitVar lower bound of a solved class
    /// unified with it (l.176).
    pub(crate) fn settle(&mut self) {
        loop {
            let before = (self.log.len(), self.split.len());
            self.merge_cycles();
            self.link_views();
            self.record_at_once();
            let settled = self.settled();
            for i in 0..self.log.len() {
                if self.split.contains(&i) {
                    continue;
                }
                match self.log[i].relation.clone() {
                    Relation::LowerBound { lower, upper } => {
                        if self.root(lower) == self.root(upper) {
                            self.split.insert(i);
                        } else if self.is_solved(lower) && self.is_solved(upper) {
                            self.split.insert(i);
                            self.split_parts(lower, upper, Side::Lower, upper);
                        } else if self.is_lit(lower)
                            && self.is_solved(upper)
                            && settled.contains(&self.root(lower))
                        {
                            self.split.insert(i);
                            self.record(Relation::Equation(lower, upper), Recorder::Split(upper));
                        }
                    }
                    Relation::Boundary { from, to } => {
                        if self.root(from) == self.root(to) {
                            self.split.insert(i);
                        } else if self.is_solved(from) && self.is_solved(to) {
                            self.split.insert(i);
                            self.split_parts(from, to, Side::Boundary, from);
                        }
                    }
                    Relation::Equation(..)
                    | Relation::Fact(..)
                    | Relation::Kind(..)
                    | Relation::Lit(..)
                    | Relation::Reference(..)
                    | Relation::Values(..) => {
                        self.split.insert(i);
                    }
                }
            }
            if (self.log.len(), self.split.len()) == before {
                return;
            }
        }
    }

    /// Links each kind "indexes `τ`" an extern-defined container's class holds
    /// to the slice's element along the signature of its view (l.161). A
    /// container with no view cannot have the kind.
    fn link_views(&mut self) {
        let pending: Vec<(ClassId, ClassId)> = self
            .classes
            .roots()
            .flat_map(|root| {
                let held = self.classes.held(root);
                let container = matches!(held.kind, Some((Term::Extern { .. }, _)));
                held.facts.iter().filter_map(move |fact| match fact {
                    Fact::Indexes(element) if container => Some((root, *element)),
                    _ => None,
                })
            })
            .filter(|(_, element)| !self.viewed.contains(element))
            .collect();
        for (root, element) in pending {
            self.viewed.insert(element);
            self.link_view(root, element);
        }
    }

    fn link_view(&mut self, root: ClassId, element: ClassId) {
        let recorder = Recorder::Split(root);
        let Some((kind, _)) = self.classes.held(root).kind.clone() else {
            return;
        };
        let constructor = kind.constructor();
        let view = self
            .decls
            .tree()
            .up(&Node::Ref(Mutability::Shared, constructor.clone()))
            .cloned()
            .filter(|cast| {
                matches!(self.decls.decl(cast).sig.result(), SigTy::Term(term)
                    if matches!(&**term, Term::Ref(_, SigTy::Term(target))
                        if matches!(&**target, Term::Slice(_))))
            });
        let Some(view) = view else {
            let at = format!(
                "{}: indexes {}",
                self.class_name(root),
                self.class_name(element)
            );
            self.refuse_conflict(at, Conflict::Fact(Fact::Indexes(element), constructor));
            return;
        };
        let sig = self.decls.decl(&view).sig.clone();
        let (from, to) = (
            Rc::new(self.class_name(root)),
            Rc::new(self.class_name(element)),
        );
        let link = |at: LinkAt| ClassName::Link {
            from: from.clone(),
            to: to.clone(),
            step: 0,
            at,
        };
        let mut vars = BTreeMap::new();
        let var_name = |var| link(LinkAt::Var(var));
        let param = self.instantiate(
            &sig.params()[0],
            &mut vars,
            link(LinkAt::Param),
            &var_name,
            &recorder,
        );
        let result = self.instantiate(
            sig.result(),
            &mut vars,
            link(LinkAt::Result),
            &var_name,
            &recorder,
        );
        let target = |state: &State, id: ClassId| match &state.classes.held(id).kind {
            Some((Term::Ref(_, target), _)) => Some(*target),
            _ => None,
        };
        let (Some(container), Some(slice)) = (target(self, param), target(self, result)) else {
            return;
        };
        let (Some((container_kind, _)), Some((Term::Slice(slice_element), _))) = (
            self.classes.held(container).kind.clone(),
            self.classes.held(slice).kind.clone(),
        ) else {
            return;
        };
        for ((_, part), (_, param_part)) in kind.parts().into_iter().zip(container_kind.parts()) {
            self.record(Relation::Equation(*part, *param_part), recorder.clone());
        }
        self.record(Relation::Equation(element, slice_element), recorder);
    }

    /// Merges each cycle of lower bounds alone into one equivalence class.
    fn merge_cycles(&mut self) {
        let mut edges: BTreeMap<ClassId, BTreeSet<ClassId>> = BTreeMap::new();
        for recorded in &self.log {
            if let Relation::LowerBound { lower, upper } = recorded.relation {
                let (lower, upper) = (self.root(lower), self.root(upper));
                if lower != upper {
                    edges.entry(upper).or_default().insert(lower);
                }
            }
        }
        let reach = |from: ClassId| {
            let mut seen = BTreeSet::new();
            let mut work = vec![from];
            while let Some(at) = work.pop() {
                for &next in edges.get(&at).into_iter().flatten() {
                    if seen.insert(next) {
                        work.push(next);
                    }
                }
            }
            seen
        };
        let mut merges = Vec::new();
        for &from in edges.keys() {
            for to in reach(from) {
                if to != from && reach(to).contains(&from) && from < to {
                    merges.push((from, to));
                }
            }
        }
        for (a, b) in merges {
            if self.root(a) != self.root(b) {
                self.record(Relation::Equation(a, b), Recorder::Cycle);
            }
        }
    }

    /// Records the relations between the parts of a lower bound's or a
    /// boundary's two sides (l.286), linking the parts through the declared
    /// casts where their constructors differ (l.291).
    pub(crate) fn split_parts(&mut self, from: ClassId, to: ClassId, side: Side, by: ClassId) {
        let recorder = Recorder::Split(by);
        let (Some((from_kind, _)), Some((to_kind, _))) = (
            self.classes.held(from).kind.clone(),
            self.classes.held(to).kind.clone(),
        ) else {
            return;
        };
        let (from_c, to_c) = (from_kind.constructor(), to_kind.constructor());
        if from_c == to_c {
            let variance = from_c.variance();
            for (key, from_part) in from_kind.parts() {
                let Some(&to_part) = to_kind.part(&key) else {
                    continue;
                };
                let relation = match (variance, side) {
                    (Variance::Invariant, _) => Relation::Equation(*from_part, to_part),
                    (Variance::Covariant, Side::Lower) => Relation::LowerBound {
                        lower: *from_part,
                        upper: to_part,
                    },
                    (Variance::Covariant, Side::Boundary) if self.is_unknown(to_part) => {
                        Relation::LowerBound {
                            lower: *from_part,
                            upper: to_part,
                        }
                    }
                    (Variance::Covariant, Side::Boundary) => Relation::Boundary {
                        from: *from_part,
                        to: to_part,
                    },
                };
                self.record(relation, recorder.clone());
            }
            return;
        }
        if let (
            Term::Ref(Mutability::Exclusive, from_target),
            Term::Ref(Mutability::Shared, to_target),
        ) = (&from_kind, &to_kind)
        {
            self.record(Relation::Equation(*from_target, *to_target), recorder);
            return;
        }
        let path = self
            .decls
            .tree()
            .path(&Node::Owned(from_c.clone()), &Node::Owned(to_c.clone()));
        let Some(path) = path else {
            return;
        };
        let (from_name, to_name) = (Rc::new(self.class_name(from)), Rc::new(self.class_name(to)));
        let mut current = from;
        for (step, cast) in path.iter().enumerate() {
            let sig = self.decls.decl(cast).sig.clone();
            let link = |at: LinkAt| ClassName::Link {
                from: from_name.clone(),
                to: to_name.clone(),
                step,
                at,
            };
            let mut vars = BTreeMap::new();
            let var_name = |var| link(LinkAt::Var(var));
            let param = self.instantiate(
                &sig.params()[0],
                &mut vars,
                link(LinkAt::Param),
                &var_name,
                &recorder,
            );
            let result = self.instantiate(
                sig.result(),
                &mut vars,
                link(LinkAt::Result),
                &var_name,
                &recorder,
            );
            let (Some((current_kind, _)), Some((param_kind, _))) = (
                self.classes.held(current).kind.clone(),
                self.classes.held(param).kind.clone(),
            ) else {
                return;
            };
            for ((_, current_part), (_, param_part)) in
                current_kind.parts().into_iter().zip(param_kind.parts())
            {
                self.record(
                    Relation::Equation(*current_part, *param_part),
                    recorder.clone(),
                );
            }
            current = result;
        }
        self.split_parts(current, to, side, by);
    }

    /// Whether the equivalence class reaches a part of its own along recorded
    /// lower bounds and parts (l.275).
    fn reaches_own_part(&self, root: ClassId) -> bool {
        let root = self.root(root);
        let mut seen: BTreeSet<(ClassId, bool)> = BTreeSet::new();
        let mut work = vec![(root, false)];
        while let Some((at, through_part)) = work.pop() {
            let mut next: Vec<(ClassId, bool)> = self
                .lower_bounds(at)
                .into_iter()
                .map(|lower| (self.root(lower), through_part))
                .collect();
            if let Some((kind, _)) = &self.classes.held(at).kind {
                next.extend(
                    kind.parts()
                        .into_iter()
                        .map(|(_, part)| (self.root(*part), true)),
                );
            }
            for (class, through_part) in next {
                if class == root && through_part {
                    return true;
                }
                if seen.insert((class, through_part)) {
                    work.push((class, through_part));
                }
            }
        }
        false
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

    /// The equivalence classes left to fall back to a default take it
    /// (l.299): those that are not solved and have no unbroken dependency.
    /// Returns whether any did.
    pub fn fall_back(&mut self) -> bool {
        let targets = self.fallback_targets();
        let decls = Rc::clone(&self.decls);
        let types = decls.types();
        for &root in &targets {
            let term = match &self.classes.held(root).lit {
                Some(lit) => Term::Scalar(lit.default_scalar()),
                None => Term::Never,
            };
            if let Err(conflict) = self
                .classes
                .give_kind(root, term, KindSource::Solved, types)
            {
                let name = self.class_name(root);
                self.refuse_conflict(format!("the default of {name}"), conflict);
            }
            self.classes.mark_solved(root);
        }
        self.settle();
        !targets.is_empty()
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
            if !self.fall_back() {
                return self.outcome();
            }
        }
    }

    /// The closed type the equivalence class holds, where it has one.
    pub fn value(&self, id: ClassId) -> Option<Ty> {
        self.value_in(id, &mut BTreeSet::new())
    }

    fn value_in(&self, id: ClassId, visiting: &mut BTreeSet<ClassId>) -> Option<Ty> {
        let root = self.root(id);
        if self.has_no_value(root) || !visiting.insert(root) {
            return None;
        }
        let (kind, _) = self.classes.held(root).kind.clone()?;
        let term = kind
            .try_map(|_, part| self.value_in(part, visiting).ok_or(()))
            .ok()?;
        visiting.remove(&root);
        Ty::new(term).ok()
    }

    /// Checks each value the text makes enter a type (l.293), in the order of
    /// their relations' names, which no order of solving changes (I3).
    fn checks(&self) -> Vec<Check> {
        let mut checks = Vec::new();
        for recorded in &self.log {
            if !recorded.recorder.is_entering() {
                continue;
            }
            let (from, to, lower) = match recorded.relation {
                Relation::LowerBound { lower, upper } => (lower, upper, true),
                Relation::Boundary { from, to } => (from, to, false),
                Relation::Equation(..)
                | Relation::Fact(..)
                | Relation::Kind(..)
                | Relation::Lit(..)
                | Relation::Reference(..)
                | Relation::Values(..) => continue,
            };
            let (Some(from_value), Some(to_value)) = (self.value(from), self.value(to)) else {
                continue;
            };
            let written = matches!(self.classes.held(to).kind, Some((_, KindSource::Written)));
            let coercion = if lower {
                match along_le(&self.decls, &from_value, &to_value) {
                    Ok(coercion) => Ok(coercion),
                    Err(_) if written => coerce(&self.decls, &from_value, &to_value),
                    Err(error) => Err(error),
                }
            } else {
                coerce(&self.decls, &from_value, &to_value)
            };
            let relation = Named {
                relation: &recorded.relation,
                arena: &self.arena,
            }
            .to_string();
            checks.push(Check { relation, coercion });
        }
        checks.sort_by(|a, b| {
            (&a.relation, format!("{:?}", a.coercion))
                .cmp(&(&b.relation, format!("{:?}", b.coercion)))
        });
        checks
    }

    pub fn outcome(&self) -> Outcome {
        let values = self
            .texts
            .iter()
            .map(|&id| (self.arena.name(id).clone(), self.value(id)))
            .collect();
        let mut partition: BTreeMap<ClassId, BTreeSet<ClassName>> = BTreeMap::new();
        for &id in &self.texts {
            partition
                .entry(self.root(id))
                .or_default()
                .insert(self.arena.name(id).clone());
        }
        let checks = self.checks();
        let mut refusals = self.refusals.clone();
        for root in self.classes.roots() {
            if self.value(root).is_none() && !self.has_no_value(root) {
                refusals.insert(Refusal::NoValue(self.class_name(root)));
            }
        }
        let decisions: BTreeMap<Name, Option<Decision>> = self
            .nodes
            .iter()
            .map(|(name, node)| (name.clone(), node.decision.clone()))
            .collect();
        let accepted = refusals.is_empty()
            && checks.iter().all(|check| check.coercion.is_ok())
            && decisions
                .values()
                .all(|decision| matches!(decision, Some(decision) if decision.is_reached()));
        Outcome {
            values,
            decisions,
            partition: partition.into_values().collect(),
            checks,
            refusals,
            accepted,
        }
    }

    /// The outcome of the entry's state, with what each other binding group
    /// decided at its fixpoint: the values of its text's classes, its decided
    /// nodes, its checks of values entering classes it solved, and its
    /// refusals. Its classes and nodes left in C need no value and no decision
    /// there (l.301).
    pub fn outcome_with(&self, groups: &BTreeMap<Name, Rc<State>>) -> Outcome {
        let mut outcome = self.outcome();
        for group in groups.values() {
            outcome.values.extend(
                group
                    .texts
                    .iter()
                    .map(|&id| (group.arena.name(id).clone(), group.value(id))),
            );
            let decided: Vec<(Name, Decision)> = group
                .nodes
                .iter()
                .filter_map(|(name, node)| Some((name.clone(), node.decision.clone()?)))
                .collect();
            let reached = decided.iter().all(|(_, decision)| decision.is_reached());
            outcome.decisions.extend(
                decided
                    .into_iter()
                    .map(|(name, decision)| (name, Some(decision))),
            );
            let checks = group.checks();
            let checked = checks.iter().all(|check| check.coercion.is_ok());
            outcome.checks.extend(checks);
            outcome.refusals.extend(group.refusals.iter().cloned());
            outcome.accepted &= reached && checked && group.refusals.is_empty();
        }
        outcome
    }

    /// A text of the state that names classes as the outcome does, so that
    /// two states reached apart compare by it.
    pub fn fingerprint(&self) -> String {
        let mut roots: Vec<String> = self
            .classes
            .roots()
            .map(|root| {
                let held = self.classes.held(root);
                let kind = held.kind.as_ref().map(|(kind, source)| {
                    let kind = kind.clone().map(|_, part| self.class_name(part));
                    format!("{kind:?}/{source:?}")
                });
                format!(
                    "{}:{kind:?}:{:?}:{}:{}",
                    self.class_name(root),
                    held.lit,
                    held.solved,
                    self.has_no_value(root),
                )
            })
            .collect();
        roots.sort();
        let decisions: Vec<_> = self.nodes.values().map(|node| &node.decision).collect();
        format!(
            "{roots:?}|{}|{:?}|{:?}|{decisions:?}|{:?}|{:?}",
            self.log.len(),
            self.split,
            self.refusals,
            self.deferred,
            self.broken,
        )
    }
}
