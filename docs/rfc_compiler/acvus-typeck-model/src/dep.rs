//! The dependency graph (l.236–267): its nodes are the equivalence classes
//! of the moment and the uses and lends, and its edges the dependencies the
//! rows of l.238–244 give. A dependency within a strongly connected component
//! is broken (l.265), so the unbroken ones have no cycle (l.267).

use std::collections::{BTreeMap, BTreeSet};

use crate::class::ClassId;
use crate::node::Reads;
use crate::relation::Relation;
use crate::solve::State;
use crate::ty::Name;

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub(crate) enum Vertex {
    Class(ClassId),
    Node(Name),
    /// The use that chooses a universal, which lies outside the binding group
    /// and is never decided in it (l.242).
    Outside,
}

/// A dependency (l.240–244).
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub(crate) enum Edge {
    /// An equivalence class on an unsolved lower bound.
    Lower,
    /// An equivalence class on one whose taking a constructor records
    /// relations into it: it lies within the `τ` of a lower bound `τ ≤ α`, or
    /// within the `U` of a boundary `α ⇝ U`, `α` unsolved (l.240 (a)).
    Taking,
    /// An equivalence class on an undecided node at whose position it lies,
    /// or within the depth its candidates reach (l.240 (b)).
    Records,
    /// A node on an undecided node that records relations into the classes at
    /// the positions it reads (l.243–244).
    Reads,
    /// A binding on an undecided projection on its path (`holes.md`, O1).
    Path,
    /// A universal on the use that chooses it (l.242).
    Chosen,
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

    /// The vertices that reach `to` (l.265).
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
                Relation::LowerBound { lower, upper } => {
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
                }
                _ => {}
            }
        }
        for root in self.classes.roots() {
            if self.is_universal(root) {
                add(Vertex::Class(root), Vertex::Outside, Edge::Chosen);
            }
        }
        let undecided: Vec<(&Name, &crate::node::NodeState)> = self
            .nodes
            .iter()
            .filter(|(_, node)| node.decision.is_none())
            .collect();
        for (name, node) in &undecided {
            for position in self.positions(&node.kind) {
                let position = self.root(position);
                if !self.is_solved(position) {
                    add(
                        Vertex::Class(position),
                        Vertex::Node((*name).clone()),
                        Edge::Records,
                    );
                }
            }
        }
        // The nodes that record into a class (l.246), each with the class at
        // whose position it lies.
        let recording = |class: ClassId, edges: &BTreeMap<Vertex, BTreeSet<(Vertex, Edge)>>| {
            let mut classes = BTreeSet::from([self.root(class)]);
            let mut work = vec![self.root(class)];
            let mut nodes = BTreeSet::new();
            while let Some(at) = work.pop() {
                for (to, edge) in edges.get(&Vertex::Class(at)).into_iter().flatten() {
                    match (to, edge) {
                        (Vertex::Class(next), Edge::Lower | Edge::Taking) => {
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

    /// The undecided nodes that broke their dependency on the node (l.265).
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
    /// and those that transitively depend on it (l.263).
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

    fn satisfied(&self, to: &Vertex, edge: Edge, settled: &BTreeSet<ClassId>) -> bool {
        match (to, edge) {
            (Vertex::Class(class), Edge::Lower) => {
                settled.contains(class) || self.gives_nothing(*class)
            }
            (Vertex::Class(class), Edge::Taking) => self.gives_nothing(*class),
            _ => false,
        }
    }

    /// The unsolved equivalence classes holding only a LitVar that wait for
    /// nothing. Joined, a LitVar is unified (l.171), so such a lower bound
    /// gives its LitVar and is not waited for.
    pub(crate) fn settled(&self) -> BTreeSet<ClassId> {
        self.settled_in(&self.graph())
    }

    pub(crate) fn settled_in(&self, graph: &Graph) -> BTreeSet<ClassId> {
        let mut settled = BTreeSet::new();
        loop {
            let more: Vec<ClassId> = self
                .classes
                .roots()
                .filter(|&root| {
                    self.is_lit(root) && !settled.contains(&root) && !self.is_foreign(root)
                })
                .filter(|&root| {
                    graph
                        .unbroken(&Vertex::Class(root))
                        .all(|(to, edge)| self.satisfied(to, *edge, &settled))
                })
                .collect();
            if more.is_empty() {
                return settled;
            }
            settled.extend(more);
        }
    }

    /// The unsolved equivalence classes whose unbroken dependencies are all
    /// solved, settled, or give nothing (l.271).
    pub(crate) fn unblocked(&self, graph: &Graph, settled: &BTreeSet<ClassId>) -> Vec<ClassId> {
        self.classes
            .roots()
            .filter(|&root| {
                !self.is_solved(root) && !self.has_no_value(root) && !self.is_foreign(root)
            })
            .filter(|&root| {
                graph
                    .unbroken(&Vertex::Class(root))
                    .all(|(to, edge)| self.satisfied(to, *edge, settled))
            })
            .collect()
    }

    /// The equivalence classes that may be solved now, being unblocked and
    /// given something to take or a LitVar to unify by their sources, and the
    /// nodes that may be decided now, whose unbroken dependencies are all
    /// decided.
    pub(crate) fn eligible(&self) -> (Vec<ClassId>, Vec<Name>) {
        let graph = self.graph();
        let settled = self.settled_in(&graph);
        let classes = self
            .unblocked(&graph, &settled)
            .into_iter()
            .filter(|&root| self.has_work(root, &graph, &settled))
            .collect();
        let nodes = self
            .nodes
            .iter()
            .filter(|(_, node)| node.decision.is_none())
            .map(|(name, _)| name)
            .filter(|name| {
                graph
                    .unbroken(&Vertex::Node((*name).clone()))
                    .all(|(to, edge)| self.satisfied(to, *edge, &settled))
            })
            .cloned()
            .collect();
        (classes, nodes)
    }

    /// The unbroken dependencies of `from` not yet met (l.240–246, l.265):
    /// what an equivalence class or a node still waits for, as a traced order
    /// prints it (`design/19-cases.md`, `TRACE=1`). A class that waits for
    /// nothing is unblocked, and a node that waits for nothing may be decided.
    #[cfg(test)]
    pub(crate) fn waits_for(&self, from: &Vertex) -> Vec<(Vertex, Edge)> {
        let graph = self.graph();
        let settled = self.settled_in(&graph);
        graph
            .unbroken(from)
            .filter(|(to, edge)| !self.satisfied(to, *edge, &settled))
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
