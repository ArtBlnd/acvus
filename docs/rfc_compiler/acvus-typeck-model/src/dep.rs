//! The dependency graph (l.234–265): its nodes are the equivalence classes
//! of the moment and the uses and lends, and its edges the dependencies the
//! rows of l.236–242 give. A dependency within a strongly connected component
//! is broken (l.263), so the unbroken ones have no cycle (l.265).

use std::collections::{BTreeMap, BTreeSet};

use crate::class::ClassId;
use crate::node::Reads;
use crate::relation::Relation;
use crate::solve::State;
use crate::ty::{Name, Term};

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub(crate) enum Vertex {
    Class(ClassId),
    Node(Name),
    /// The use that chooses a universal, which lies outside the binding group
    /// and is never decided in it (l.240).
    Outside,
}

/// A dependency (l.238–242).
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub(crate) enum Edge {
    /// An equivalence class on an unsolved lower bound or part lower bound.
    Lower,
    /// An equivalence class on one whose taking a constructor records
    /// relations into it: it lies within the `τ` of a lower bound or a part
    /// lower bound `τ ≤ α`, or within the `U` of a boundary `α ⇝ U`, `α`
    /// unsolved, or it is the class `e?`'s wrapping passes into, its error
    /// part unsolved (l.238 (a)).
    Taking,
    /// An equivalence class on an unsolved one that has it as a part by a kind
    /// known in part, taking a constructor (l.239). A class holding a kind is
    /// solved (note 06, reading 4).
    Part,
    /// An equivalence class on a node at whose position it lies, or within the
    /// depth its candidates reach, that has not recorded its relations: one
    /// undecided, or one decided whose relations wait for the nodes that broke
    /// their dependency on it (l.238 (b), l.263).
    Records,
    /// A node on an undecided node that records relations into the classes at
    /// the positions it reads (l.241–242).
    Reads,
    /// A binding on an undecided projection on its path (`holes.md`, O1).
    Path,
    /// A universal on the use that chooses it (l.240).
    Chosen,
}

/// What meets an unbroken dependency on an unsolved equivalence class
/// besides its being solved (l.238).
pub(crate) struct Met {
    /// The classes holding only a LitVar that wait for nothing.
    pub(crate) settled: BTreeSet<ClassId>,
    /// The classes nothing enters (l.282).
    pub(crate) nothing_enters: BTreeSet<ClassId>,
}

/// What may be taken now.
pub(crate) struct Eligible {
    /// The equivalence classes that may be solved (l.269).
    pub(crate) solve: Vec<ClassId>,
    /// The equivalence classes that may be filled (l.282).
    pub(crate) fill: Vec<ClassId>,
    /// The nodes whose unbroken dependencies are all decided, which the value
    /// each reads may decide (l.261).
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

    /// The vertices that reach `to` (l.263).
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
                Relation::Boundary { from, to } => {
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
                    // at its split, into the parameters in the environment of
                    // each lambda whose value reaches it (l.197, l.238 (a)).
                    if matches!(self.classes.held(to).kind, Some((Term::Function { .. }, _))) {
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
                for fact in &self.classes.held(root).facts {
                    let part = self.root(fact.class());
                    if part != root && !self.is_solved(part) {
                        add(Vertex::Class(part), Vertex::Class(root), Edge::Part);
                    }
                }
            }
        }
        let undecided: Vec<(&Name, &crate::node::NodeState)> = self
            .nodes
            .iter()
            .filter(|(_, node)| node.decision.is_none())
            .collect();
        let deferred = self
            .nodes
            .iter()
            .filter(|(name, _)| self.deferred.contains_key(*name));
        let mut passed_out = Vec::new();
        for (name, node) in undecided.iter().copied().chain(deferred) {
            let only_out = self.passed_out_only(&node.kind);
            for position in self.positions(&node.kind) {
                let position = self.root(position);
                if self.is_solved(position) {
                    continue;
                }
                if only_out.contains(&position) {
                    passed_out.push((position, (*name).clone()));
                } else {
                    add(
                        Vertex::Class(position),
                        Vertex::Node((*name).clone()),
                        Edge::Records,
                    );
                }
            }
        }
        // A node that records into a class only a boundary out of it changes
        // it only by the fill of a class nothing enters or LitVars alone
        // enter (l.278, l.282), so the class waits for it only while it is
        // one, judged without those dependencies (l.238 (b)).
        let provisional = Graph {
            component: components(&edges),
            edges: edges.clone(),
        };
        let filled = self.filled_by_places_in(&provisional);
        for (position, name) in passed_out {
            if filled.contains(&position) {
                edges
                    .entry(Vertex::Class(position))
                    .or_default()
                    .insert((Vertex::Node(name), Edge::Records));
            }
        }
        // The nodes that record into a class (l.244), each with the class at
        // whose position it lies.
        let recording = |class: ClassId, edges: &BTreeMap<Vertex, BTreeSet<(Vertex, Edge)>>| {
            let mut classes = BTreeSet::from([self.root(class)]);
            let mut work = vec![self.root(class)];
            let mut nodes = BTreeSet::new();
            while let Some(at) = work.pop() {
                for (to, edge) in edges.get(&Vertex::Class(at)).into_iter().flatten() {
                    match (to, edge) {
                        (Vertex::Class(next), Edge::Lower | Edge::Taking | Edge::Part) => {
                            if classes.insert(*next) {
                                work.push(*next);
                            }
                        }
                        (Vertex::Node(_), Edge::Records) | (Vertex::Outside, Edge::Chosen) => {
                            nodes.insert((to.clone(), at));
                        }
                        _ => {}
                    }
                }
            }
            nodes
        };
        let mut reads = Vec::new();
        for (name, _) in &undecided {
            let Reads {
                read,
                classes,
                path,
            } = self.reads(name);
            for position in classes {
                for (recorder, at) in recording(position, &edges) {
                    let left_out = match &recorder {
                        Vertex::Node(recorder) => {
                            recorder == *name || !self.records_into(recorder, at, read)
                        }
                        Vertex::Class(_) | Vertex::Outside => false,
                    };
                    if !left_out {
                        reads.push(((*name).clone(), recorder, Edge::Reads));
                    }
                }
            }
            for step in path {
                if self.nodes[&step].decision.is_none() {
                    reads.push(((*name).clone(), Vertex::Node(step), Edge::Path));
                }
            }
        }
        for (from, to, edge) in reads {
            edges
                .entry(Vertex::Node(from))
                .or_default()
                .insert((to, edge));
        }
        let component = components(&edges);
        Graph { edges, component }
    }

    /// The roots of the classes at any depth within the kind of `id`'s
    /// equivalence class.
    pub(crate) fn within(&self, id: ClassId) -> BTreeSet<ClassId> {
        let mut seen = BTreeSet::new();
        let mut work = vec![self.root(id)];
        while let Some(at) = work.pop() {
            let Some((kind, _)) = &self.classes.held(at).kind else {
                continue;
            };
            for (_, part) in kind.parts() {
                let part = self.root(*part);
                if seen.insert(part) {
                    work.push(part);
                }
            }
        }
        seen
    }

    /// The undecided nodes that broke their dependency on the node (l.263).
    pub(crate) fn broken_on(&self, name: &Name) -> BTreeSet<Name> {
        let graph = self.graph();
        let node = Vertex::Node(name.clone());
        self.nodes
            .iter()
            .filter(|(other, state)| *other != name && state.decision.is_none())
            .map(|(other, _)| other.clone())
            .filter(|other| graph.broken(&Vertex::Node(other.clone()), &node))
            .collect()
    }

    /// The nodes whose relations the value a node reads leaves out: itself
    /// and those that transitively depend on it (l.261).
    pub(crate) fn dependents(&self, name: &Name) -> BTreeSet<Name> {
        let node = Vertex::Node(name.clone());
        let mut excluded: BTreeSet<Name> = self
            .graph()
            .reaching(&node)
            .into_iter()
            .filter_map(|vertex| match vertex {
                Vertex::Node(name) => Some(name),
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
    /// a class holding only a LitVar that waits for nothing, or a lower bound
    /// or part lower bound nothing enters (l.238), or, in a state a value read
    /// is taken from, a class that gives nothing (l.261); and a node's on a
    /// node once that is decided, whose relations a class depends on until they
    /// are recorded (l.238 (b), l.241).
    fn satisfied(&self, to: &Vertex, edge: Edge, met: &Met) -> bool {
        match (to, edge) {
            (Vertex::Node(node), Edge::Reads | Edge::Path) => self.nodes[node].decision.is_some(),
            (Vertex::Class(class), Edge::Lower) => {
                met.settled.contains(class)
                    || met.nothing_enters.contains(class)
                    || self.gives_nothing(*class)
            }
            (Vertex::Class(class), Edge::Taking | Edge::Part) => self.gives_nothing(*class),
            _ => false,
        }
    }

    /// The unsolved equivalence classes holding only a LitVar that wait for
    /// nothing. Joined, a LitVar is unified (l.169), so such a lower bound
    /// gives its LitVar and is not waited for.
    pub(crate) fn settled(&self) -> BTreeSet<ClassId> {
        self.met_in(&self.graph()).settled
    }

    /// What meets an unbroken dependency on an unsolved class in `graph`: the
    /// classes nothing enters, and then the settled ones.
    pub(crate) fn met_in(&self, graph: &Graph) -> Met {
        let mut met = Met {
            settled: BTreeSet::new(),
            nothing_enters: self.nothing_enters_in(graph),
        };
        loop {
            let more: Vec<ClassId> = self
                .classes
                .roots()
                .filter(|&root| {
                    self.is_lit(root) && !met.settled.contains(&root) && !self.is_foreign(root)
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
            met.settled.extend(more);
        }
    }

    /// The equivalence classes that take their type from the places they pass
    /// into (l.282): those nothing enters, and those LitVars alone enter,
    /// judged alike, each of their lower bounds and part lower bounds one of
    /// these and each of their unbroken dependencies met as such a lower
    /// bound's is, one holding a LitVar or with a lower bound at least.
    fn filled_by_places_in(&self, graph: &Graph) -> BTreeSet<ClassId> {
        let nothing = self.nothing_enters_in(graph);
        let mut found: BTreeSet<ClassId> = BTreeSet::new();
        loop {
            let alone = |class: &ClassId| found.contains(class) || nothing.contains(class);
            let more: Vec<ClassId> = self
                .classes
                .roots()
                .filter(|root| !alone(root))
                .filter(|&root| {
                    let held = self.classes.held(root);
                    !self.is_solved(root)
                        && !self.has_no_value(root)
                        && !self.is_foreign(root)
                        && held.kind.is_none()
                        && held.facts.is_empty()
                })
                .filter(|&root| {
                    let lowers = self.lower_bounds(root);
                    (self.is_lit(root) || !lowers.is_empty())
                        && lowers.into_iter().all(|lower| alone(&self.root(lower)))
                })
                .filter(|&root| {
                    graph
                        .unbroken(&Vertex::Class(root))
                        .all(|(to, edge)| match (to, edge) {
                            (Vertex::Class(class), Edge::Lower) => alone(class),
                            (Vertex::Class(class), Edge::Taking | Edge::Part) => {
                                self.gives_nothing(*class)
                            }
                            _ => false,
                        })
                })
                .collect();
            if more.is_empty() {
                found.extend(nothing);
                return found;
            }
            found.extend(more);
        }
    }

    /// The equivalence classes nothing enters (l.282): unsolved, holding no
    /// kind, no LitVar and no kind known in part, each of their lower bounds
    /// and part lower bounds one nothing enters, and each of their unbroken
    /// dependencies met, as one on a lower bound or a part lower bound nothing
    /// enters is (l.238). In a state a value read is taken from, a class that
    /// gives nothing is not one, but is met as one is (l.261).
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
                        && held.kind.is_none()
                        && held.lit.is_none()
                        && held.facts.is_empty()
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
    /// solved or met (l.269).
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
    /// being unblocked and given something to take or a LitVar to unify by
    /// their sources; those that may be filled, nothing entering them and each
    /// place they pass into having a value (l.282); and the nodes whose
    /// unbroken dependencies are all decided.
    pub(crate) fn eligible(&self) -> Eligible {
        let graph = self.graph();
        let met = self.met_in(&graph);
        let solve = self
            .unblocked(&graph, &met)
            .into_iter()
            .filter(|&root| self.has_work(root, &graph, &met.settled))
            .collect();
        let fill = met
            .nothing_enters
            .iter()
            .copied()
            .filter(|&root| self.fills_now(root))
            .collect();
        let decide = self
            .nodes
            .iter()
            .filter(|(_, node)| node.decision.is_none())
            .map(|(name, _)| name)
            .filter(|name| {
                graph
                    .unbroken(&Vertex::Node((*name).clone()))
                    .all(|(to, edge)| self.satisfied(to, *edge, &met))
            })
            .cloned()
            .collect();
        Eligible {
            solve,
            fill,
            decide,
        }
    }

    /// The unbroken dependencies of `from` not yet met (l.238–244, l.263):
    /// what an equivalence class or a node still waits for, as a traced order
    /// prints it (`design/19-cases.md`, `TRACE=1`). A class that waits for
    /// nothing is unblocked, and a node that waits for nothing may be decided.
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

    fn has_work(&self, root: ClassId, graph: &Graph, settled: &BTreeSet<ClassId>) -> bool {
        let held = self.classes.held(root);
        if held.kind.is_some() {
            return true;
        }
        if !self.fact_heads(root).is_empty() {
            return true;
        }
        if held.lit.is_some() && !self.boundaries_from(root).is_empty() {
            return true;
        }
        self.lower_bounds(root).into_iter().any(|lower| {
            let lower = self.root(lower);
            !graph.broken(&Vertex::Class(root), &Vertex::Class(lower))
                && (self.is_solved(lower) || settled.contains(&lower))
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
