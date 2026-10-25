//! The dependency graph: its disjunctive constraints are the equivalence classes
//! of the moment and the uses and lends, and its edges the dependencies. A
//! dependency within a strongly connected component is broken, so the unbroken
//! ones have no cycle.

use std::collections::{BTreeMap, BTreeSet};

use crate::class::{ClassId, StructureConstraint};
use crate::disjunction::Reads;
use crate::relation::Relation;
use crate::solve::State;
use crate::ty::{Constructor, NONE, Name, PartKey, SOME, Term};

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub(crate) enum Vertex {
    Class(ClassId),
    Disjunction(Name),
    /// The use that chooses a universal, which lies outside the binding group
    /// and is never decided in it.
    Outside,
}

/// A dependency.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub(crate) enum Edge {
    /// An equivalence class on an unsolved lower bound or part lower bound.
    Lower,
    /// An equivalence class on one whose taking a constructor records
    /// relations into it: it lies within the `τ` of a lower bound or a part
    /// lower bound `τ ≤ α`, or within the `U` of a coercion constraint `α ⇝ U`, `α`
    /// unsolved, or it is the class `e?`'s wrapping passes into, its error
    /// part unsolved.
    Taking,
    /// An equivalence class on an unsolved one that has it as a part by a structure constraint,
    /// taking a constructor. A class holding a structure is
    /// solved.
    Part,
    /// An equivalence class on a disjunctive constraint at whose position it lies, or within the
    /// depth its candidates reach, that has not recorded its relations: one
    /// undecided, or one decided whose relations wait for the disjunctive constraints that broke
    /// their dependency on it.
    Records,
    /// A disjunctive constraint on an undecided disjunctive constraint that records relations into the classes at
    /// the positions it reads.
    Reads,
    /// A universal on the use that chooses it.
    Chosen,
}

/// What meets an unbroken dependency on an unsolved equivalence class
/// besides its being solved.
pub(crate) struct Met {
    /// The classes holding no structure and a domain that a value reaching a class
    /// they bound stays within, that wait for nothing: such a lower bound
    /// gives its domain at once (`State::domain_up`) and is not waited for.
    pub(crate) giving_domains: BTreeSet<ClassId>,
    /// The classes nothing enters.
    pub(crate) nothing_enters: BTreeSet<ClassId>,
}

/// What may be taken now.
pub(crate) struct Eligible {
    /// The equivalence classes that may be solved.
    pub(crate) solve: Vec<ClassId>,
    /// The disjunctive constraints whose unbroken dependencies are all decided, which the value
    /// each reads may decide.
    pub(crate) decide: Vec<Name>,
}

pub(crate) struct Graph {
    edges: BTreeMap<Vertex, BTreeSet<(Vertex, Edge)>>,
    /// Each vertex's strongly connected component, by its least vertex.
    component: BTreeMap<Vertex, Vertex>,
}

impl Graph {
    fn unbroken(&self, from: &Vertex) -> impl Iterator<Item = &(Vertex, Edge)> + '_ {
        let own = self.component.get(from);
        self.edges
            .get(from)
            .into_iter()
            .flatten()
            .filter(move |(to, _)| self.component.get(to) != own)
    }

    fn broken(&self, from: &Vertex, to: &Vertex) -> bool {
        let edge = self
            .edges
            .get(from)
            .is_some_and(|edges| edges.iter().any(|(target, _)| target == to));
        edge && self.component.get(from) == self.component.get(to)
    }

    /// The vertices that reach `to`.
    fn reaching(&self, to: &Vertex) -> BTreeSet<Vertex> {
        let mut reaching = BTreeSet::new();
        let mut work = vec![to.clone()];
        while let Some(at) = work.pop() {
            for (from, edges) in &self.edges {
                if edges.iter().any(|(target, _)| *target == at) && reaching.insert(from.clone()) {
                    work.push(from.clone());
                }
            }
        }
        reaching
    }
}

impl State {
    pub(crate) fn graph(&self) -> Graph {
        let mut edges: BTreeMap<Vertex, BTreeSet<(Vertex, Edge)>> = BTreeMap::new();
        let mut add = |from: Vertex, to: Vertex, edge: Edge| {
            edges.entry(from).or_default().insert((to, edge));
        };
        for recorded in &self.log {
            match recorded.relation {
                Relation::LowerBound { lower, upper }
                | Relation::PartLowerBound { lower, upper } => {
                    let (lower, upper) = (self.root(lower), self.root(upper));
                    if lower == upper || self.is_solved(upper) {
                        continue;
                    }
                    if !self.is_solved(lower) {
                        add(Vertex::Class(upper), Vertex::Class(lower), Edge::Lower);
                    }
                    for within in self.within(lower) {
                        if within != upper && !self.is_solved(within) {
                            add(Vertex::Class(within), Vertex::Class(upper), Edge::Taking);
                        }
                    }
                }
                Relation::CoercionConstraint { from, to } => {
                    let (from, to) = (self.root(from), self.root(to));
                    if from == to || self.is_solved(from) {
                        continue;
                    }
                    for within in self.within(to) {
                        if within != from && !self.is_solved(within) {
                            add(Vertex::Class(within), Vertex::Class(from), Edge::Taking);
                        }
                    }
                    // A function value passing into a function type records,
                    // at its decomposition, into the parameters in the environment of
                    // each lambda whose value reaches it.
                    if matches!(
                        self.classes.held(to).structure,
                        Some((Term::Function { .. }, _))
                    ) {
                        for lambda in self.values_of(from) {
                            let mut held = self.environment_of(&lambda);
                            held.pop();
                            for param in held.into_iter().flatten() {
                                let param = self.root(param);
                                if param != from && !self.is_solved(param) {
                                    add(Vertex::Class(param), Vertex::Class(from), Edge::Taking);
                                }
                            }
                        }
                    }
                }
                Relation::Wrap { from, to } => {
                    let (from, to) = (self.root(from), self.root(to));
                    if from != to && !self.is_solved(from) && !self.is_solved(to) {
                        add(Vertex::Class(to), Vertex::Class(from), Edge::Taking);
                    }
                }
                _ => {}
            }
        }
        for root in self.classes.roots() {
            if self.is_universal(root) {
                add(Vertex::Class(root), Vertex::Outside, Edge::Chosen);
            }
            if !self.is_solved(root) {
                for structure_constraint in &self.classes.held(root).structure_constraints {
                    let part = self.root(structure_constraint.class());
                    if part != root && !self.is_solved(part) {
                        add(Vertex::Class(part), Vertex::Class(root), Edge::Part);
                    }
                }
            }
        }
        let undecided: Vec<(&Name, &crate::disjunction::DisjunctionState)> = self
            .disjunctions
            .iter()
            .filter(|(_, disjunction)| disjunction.decision.is_none())
            .collect();
        let deferred = self
            .disjunctions
            .iter()
            .filter(|(name, _)| self.deferred.contains_key(*name));
        for (name, _) in undecided.iter().copied().chain(deferred) {
            for class in self.recorded_into_by(name) {
                if self.is_solved(class) {
                    continue;
                }
                add(
                    Vertex::Class(class),
                    Vertex::Disjunction((*name).clone()),
                    Edge::Records,
                );
            }
        }
        // The disjunctive constraints that record into a class, each with the class at
        // whose position it lies, reached along the dependencies between
        // classes and the links a relation not yet decomposed will make between
        // the parts of its two sides.
        let links = self.decomposition_links();
        let recording = |class: ClassId, edges: &BTreeMap<Vertex, BTreeSet<(Vertex, Edge)>>| {
            let mut classes = BTreeSet::from([self.root(class)]);
            let mut work = vec![self.root(class)];
            let mut disjunctions = BTreeSet::new();
            while let Some(at) = work.pop() {
                for &next in links.get(&at).into_iter().flatten() {
                    if classes.insert(next) {
                        work.push(next);
                    }
                }
                for (to, edge) in edges.get(&Vertex::Class(at)).into_iter().flatten() {
                    match (to, edge) {
                        (Vertex::Class(next), Edge::Lower | Edge::Taking | Edge::Part) => {
                            if classes.insert(*next) {
                                work.push(*next);
                            }
                        }
                        (Vertex::Disjunction(_), Edge::Records)
                        | (Vertex::Outside, Edge::Chosen) => {
                            disjunctions.insert((to.clone(), at));
                        }
                        _ => {}
                    }
                }
            }
            disjunctions
        };
        let mut reads = Vec::new();
        for (name, _) in &undecided {
            let Reads { read, classes } = self.reads(name);
            for position in classes {
                for (recorder, at) in recording(position, &edges) {
                    let left_out = match &recorder {
                        Vertex::Disjunction(recorder) => {
                            recorder == *name || !self.records_into(recorder, position, at, read)
                        }
                        Vertex::Class(_) | Vertex::Outside => false,
                    };
                    if !left_out {
                        reads.push(((*name).clone(), recorder, Edge::Reads));
                    }
                }
            }
        }
        for (from, to, edge) in reads {
            edges
                .entry(Vertex::Disjunction(from))
                .or_default()
                .insert((to, edge));
        }
        let component = components(&edges);
        Graph { edges, component }
    }

    /// For each relation not yet decomposed, the parts its decomposition will relate,
    /// among those its two sides hold now, by a structure, a structure constraint or
    /// a function value's environment: each part that will
    /// take a relation, with the part of the other side it will take it from.
    /// A lower bound's or a part lower bound's upper side takes from the lower,
    /// a coercion constraint's type passed into from the value, a function type's
    /// parameter's environment parameter from the type's parameter, a function
    /// type's result from the environment's result, and the class `e?`'s
    /// wrapping passes into from its error part.
    pub(crate) fn decomposition_links(&self) -> BTreeMap<ClassId, BTreeSet<ClassId>> {
        let mut links: BTreeMap<ClassId, BTreeSet<ClassId>> = BTreeMap::new();
        let mut link = |taking: ClassId, from: ClassId| {
            let (taking, from) = (self.root(taking), self.root(from));
            if taking != from {
                links.entry(taking).or_default().insert(from);
            }
        };
        for (i, recorded) in self.log.iter().enumerate() {
            if self.decomposed.contains(&i) {
                continue;
            }
            let (from, to) = match recorded.relation {
                Relation::LowerBound { lower, upper }
                | Relation::PartLowerBound { lower, upper } => (lower, upper),
                Relation::CoercionConstraint { from, to } => (from, to),
                Relation::Wrap { from, to } => {
                    link(to, from);
                    continue;
                }
                _ => continue,
            };
            let (from_parts, to_parts) = (self.parts_held(from), self.parts_held(to));
            for (key, &taking) in &to_parts {
                if let Some(&given) = from_parts.get(key) {
                    match key {
                        PartKey::Param(_) => link(given, taking),
                        _ => link(taking, given),
                    }
                }
            }
        }
        links
    }

    /// The parts a class holds now, by name: those of its structure, those its
    /// structure constraints name, and, for a function value, the parameter and
    /// result types in its lambdas' environments at a function type's
    /// parameters and result.
    fn parts_held(&self, id: ClassId) -> BTreeMap<PartKey, ClassId> {
        let root = self.root(id);
        let held = self.classes.held(root);
        let mut parts = BTreeMap::new();
        match &held.structure {
            Some((Term::FunctionValue { .. }, _)) => {
                for lambda in self.values_of(root) {
                    let mut environment = self.environment_of(&lambda);
                    if let Some(Some(result)) = environment.pop() {
                        parts.insert(PartKey::Return, result);
                    }
                    for (i, param) in environment.into_iter().enumerate() {
                        if let Some(param) = param {
                            parts.insert(PartKey::Param(i), param);
                        }
                    }
                }
            }
            Some((structure, _)) => {
                for (key, &part) in structure.parts() {
                    parts.insert(key, part);
                }
            }
            None => {
                for structure_constraint in &held.structure_constraints {
                    let key = match structure_constraint {
                        StructureConstraint::Target(_) => PartKey::Target,
                        StructureConstraint::Field(name, _) => PartKey::Field(name.clone()),
                        StructureConstraint::Component(i, _) => PartKey::Component(*i),
                        StructureConstraint::Element(_) => PartKey::Element,
                        StructureConstraint::Payload { variant, .. } => match variant.as_str() {
                            SOME | NONE => PartKey::Payload(Name::new(SOME)),
                            _ => PartKey::Payload(variant.clone()),
                        },
                    };
                    parts.insert(key, structure_constraint.class());
                }
            }
        }
        parts
    }

    /// The roots of the classes at any depth within the structure of `id`'s
    /// equivalence class.
    pub(crate) fn within(&self, id: ClassId) -> BTreeSet<ClassId> {
        let mut seen = BTreeSet::new();
        let mut work = vec![self.root(id)];
        while let Some(at) = work.pop() {
            let Some((structure, _)) = &self.classes.held(at).structure else {
                continue;
            };
            for (_, part) in structure.parts() {
                let part = self.root(*part);
                if seen.insert(part) {
                    work.push(part);
                }
            }
        }
        seen
    }

    /// The undecided disjunctive constraints that broke their dependency on the disjunctive constraint.
    pub(crate) fn broken_on(&self, name: &Name) -> BTreeSet<Name> {
        let graph = self.graph();
        let disjunction = Vertex::Disjunction(name.clone());
        self.disjunctions
            .iter()
            .filter(|(other, state)| *other != name && state.decision.is_none())
            .map(|(other, _)| other.clone())
            .filter(|other| graph.broken(&Vertex::Disjunction(other.clone()), &disjunction))
            .collect()
    }

    /// The disjunctive constraints whose relations the value a disjunctive constraint reads leaves out: itself
    /// and those that transitively depend on it.
    pub(crate) fn dependents(&self, name: &Name) -> BTreeSet<Name> {
        let disjunction = Vertex::Disjunction(name.clone());
        let mut excluded: BTreeSet<Name> = self
            .graph()
            .reaching(&disjunction)
            .into_iter()
            .filter_map(|vertex| match vertex {
                Vertex::Disjunction(name) => Some(name),
                Vertex::Class(_) | Vertex::Outside => None,
            })
            .collect();
        excluded.insert(name.clone());
        excluded
    }

    pub(crate) fn gives_nothing(&self, id: ClassId) -> bool {
        self.classes
            .held(id)
            .members
            .iter()
            .any(|member| self.nothing.contains(member))
    }

    /// Whether an unbroken dependency on `to` is met without its being solved:
    /// a lower bound or part lower bound that gives a domain or that nothing
    /// enters, or, in a state a value read
    /// is taken from, a class that gives nothing; and a disjunctive constraint's on a
    /// disjunctive constraint once that is decided, whose relations a class depends on until they
    /// are recorded.
    fn satisfied(&self, to: &Vertex, edge: Edge, met: &Met) -> bool {
        match (to, edge) {
            (Vertex::Disjunction(disjunction), Edge::Reads) => {
                self.disjunctions[disjunction].decision.is_some()
            }
            (Vertex::Class(class), Edge::Lower) => {
                met.giving_domains.contains(class)
                    || met.nothing_enters.contains(class)
                    || self.gives_nothing(*class)
            }
            (Vertex::Class(class), Edge::Taking | Edge::Part) => self.gives_nothing(*class),
            _ => false,
        }
    }

    /// What meets an unbroken dependency on an unsolved class in `graph`: the
    /// classes nothing enters, and then the ones giving a domain.
    pub(crate) fn met_in(&self, graph: &Graph) -> Met {
        let mut met = Met {
            giving_domains: BTreeSet::new(),
            nothing_enters: self.nothing_enters_in(graph),
        };
        loop {
            let more: Vec<ClassId> = self
                .classes
                .roots()
                .filter(|&root| {
                    self.gives_its_domain_up(root)
                        && !met.giving_domains.contains(&root)
                        && !self.is_foreign(root)
                })
                .filter(|&root| {
                    graph
                        .unbroken(&Vertex::Class(root))
                        .all(|(to, edge)| self.satisfied(to, *edge, &met))
                })
                .collect();
            if more.is_empty() {
                return met;
            }
            met.giving_domains.extend(more);
        }
    }

    /// Whether the class holds no structure and a domain that a value reaching a
    /// class it bounds stays within (`Domain::upward`).
    pub(crate) fn gives_its_domain_up(&self, root: ClassId) -> bool {
        let held = self.classes.held(root);
        held.structure.is_none()
            && held
                .domain
                .as_ref()
                .is_some_and(|domain| domain.upward().is_some())
    }

    /// The equivalence classes nothing enters: unsolved, holding no
    /// structure, no domain that admits no `!`, which only a literal, a value, gives,
    /// and no structure constraint, each of their lower bounds
    /// and part lower bounds one nothing enters, and each of their unbroken
    /// dependencies met, as one on a lower bound or a part lower bound nothing
    /// enters is. In a state a value read is taken from, a class that
    /// gives nothing is not one, but is met as one is.
    fn nothing_enters_in(&self, graph: &Graph) -> BTreeSet<ClassId> {
        let mut found = BTreeSet::new();
        loop {
            let enters_nothing =
                |class: &ClassId| found.contains(class) || self.gives_nothing(*class);
            let more: Vec<ClassId> = self
                .classes
                .roots()
                .filter(|root| !enters_nothing(root))
                .filter(|&root| {
                    let held = self.classes.held(root);
                    !self.is_solved(root)
                        && !self.has_no_value(root)
                        && !self.is_foreign(root)
                        && held.structure.is_none()
                        && held
                            .domain
                            .as_ref()
                            .is_none_or(|domain| domain.admits(&Constructor::Never))
                        && held.structure_constraints.is_empty()
                })
                .filter(|&root| {
                    self.lower_bounds(root)
                        .into_iter()
                        .all(|lower| enters_nothing(&self.root(lower)))
                })
                .filter(|&root| {
                    graph
                        .unbroken(&Vertex::Class(root))
                        .all(|(to, edge)| match (to, edge) {
                            (Vertex::Class(class), Edge::Lower) => enters_nothing(class),
                            (Vertex::Class(class), Edge::Taking | Edge::Part) => {
                                self.gives_nothing(*class)
                            }
                            _ => false,
                        })
                })
                .collect();
            if more.is_empty() {
                return found;
            }
            found.extend(more);
        }
    }

    /// The unsolved equivalence classes whose unbroken dependencies are all
    /// solved or met.
    pub(crate) fn unblocked(&self, graph: &Graph, met: &Met) -> Vec<ClassId> {
        self.classes
            .roots()
            .filter(|&root| {
                !self.is_solved(root) && !self.has_no_value(root) && !self.is_foreign(root)
            })
            .filter(|&root| {
                graph
                    .unbroken(&Vertex::Class(root))
                    .all(|(to, edge)| self.satisfied(to, *edge, met))
            })
            .collect()
    }

    /// What may be taken now: the equivalence classes that may be solved,
    /// being unblocked and given something to take by their sources, and the
    /// disjunctive constraints whose unbroken dependencies are all decided.
    pub(crate) fn eligible(&self) -> Eligible {
        let graph = self.graph();
        let met = self.met_in(&graph);
        let solve = self
            .unblocked(&graph, &met)
            .into_iter()
            .filter(|&root| self.has_work(root, &graph))
            .collect();
        let decide = self
            .disjunctions
            .iter()
            .filter(|(_, disjunction)| disjunction.decision.is_none())
            .map(|(name, _)| name)
            .filter(|name| {
                graph
                    .unbroken(&Vertex::Disjunction((*name).clone()))
                    .all(|(to, edge)| self.satisfied(to, *edge, &met))
            })
            .cloned()
            .collect();
        Eligible { solve, decide }
    }

    /// The unbroken dependencies of `from` not yet met:
    /// what an equivalence class or a disjunctive constraint still waits for, as a traced order
    /// prints it (`TRACE=1`). A class that waits for
    /// nothing is unblocked, and a disjunctive constraint that waits for nothing may be decided.
    #[cfg(test)]
    pub(crate) fn waits_for(&self, from: &Vertex) -> Vec<(Vertex, Edge)> {
        let graph = self.graph();
        let met = self.met_in(&graph);
        graph
            .unbroken(from)
            .filter(|(to, edge)| !self.satisfied(to, *edge, &met))
            .cloned()
            .collect()
    }

    fn has_work(&self, root: ClassId, graph: &Graph) -> bool {
        let held = self.classes.held(root);
        if held.structure.is_some() {
            return true;
        }
        if !self.structure_constraint_heads(root).is_empty() || self.domain_head(root).is_some() {
            return true;
        }
        self.lower_bounds(root).into_iter().any(|lower| {
            let lower = self.root(lower);
            !graph.broken(&Vertex::Class(root), &Vertex::Class(lower)) && self.is_solved(lower)
        })
    }
}

/// The strongly connected components of a graph, each named by its least
/// vertex.
fn components(edges: &BTreeMap<Vertex, BTreeSet<(Vertex, Edge)>>) -> BTreeMap<Vertex, Vertex> {
    let reach = |from: &Vertex| {
        let mut seen = BTreeSet::from([from.clone()]);
        let mut work = vec![from.clone()];
        while let Some(at) = work.pop() {
            for (next, _) in edges.get(&at).into_iter().flatten() {
                if seen.insert(next.clone()) {
                    work.push(next.clone());
                }
            }
        }
        seen
    };
    let vertices: BTreeSet<Vertex> = edges
        .iter()
        .flat_map(|(from, to)| {
            std::iter::once(from.clone()).chain(to.iter().map(|(to, _)| to.clone()))
        })
        .collect();
    let reaches: BTreeMap<Vertex, BTreeSet<Vertex>> = vertices
        .iter()
        .map(|vertex| (vertex.clone(), reach(vertex)))
        .collect();
    vertices
        .iter()
        .map(|vertex| {
            let least = reaches[vertex]
                .iter()
                .filter(|other| reaches[*other].contains(vertex))
                .min()
                .cloned()
                .expect("a vertex reaches itself");
            (vertex.clone(), least)
        })
        .collect()
}
