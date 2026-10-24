//! I\*: each decision against the one its node makes reading what is recorded
//! once solving ends.
//!
//! The dependency here is computed on its own, from what each node records at
//! the end, never through `State::graph`, which is the means under test.

use std::collections::{BTreeMap, BTreeSet};
use std::rc::Rc;

use crate::class::ClassId;
use crate::explore::Final;
use crate::instance::{Instance, Made};
use crate::node::{Common, Decision, NodeKind, NodeState, Read};
use crate::program::{ENTRY, Program, Projection};
use crate::relation::{Recorder, Relation, copy_name};
use crate::solve::State;
use crate::ty::Name;

/// Which way a decision fails I\*, by the outcome it is part of.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub enum Kind {
    /// The program is refused.
    Completeness,
    /// The program is accepted, with a typing other than the one I\* gives.
    Soundness,
}

/// A decision that is not the one its node makes reading what is recorded at
/// the end.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub struct Violation {
    pub kind: Kind,
    /// The binding group of the state the node is decided again in.
    pub group: Name,
    pub node: Name,
    pub decided: Decision,
    /// What it becomes decided again, none where that decides nothing.
    pub again: Option<Decision>,
    /// The nodes whose relations the decision again leaves out.
    pub left_out: BTreeSet<Name>,
}

/// Every decided node of a final state, decided again in each context its
/// decision holds for.
pub fn check(program: &Program, end: &Final<'_>) -> Vec<Violation> {
    let accepted = end.entry.outcome_with(end.groups).accepted;
    let mut violations = Vec::new();
    let states = end.groups.values().map(|state| &**state).chain([end.entry]);
    for state in states {
        violations.extend(check_view(&view(state, end.groups), accepted, |_| true));
    }
    for (group, parent) in &program.groups {
        let Some(parent) = parent else {
            continue;
        };
        let enclosing = if parent.as_str() == ENTRY {
            end.entry
        } else {
            &*end.groups[parent]
        };
        let lambda = view(&end.groups[group], end.groups);
        let at = Name::new(&format!("{group}#check"));
        let mut here = enclosing.clone();
        let mut copy = Instance::new(&at, Recorder::Check, &lambda);
        copy.record_rest(&mut here);
        let made = here
            .instances
            .last()
            .expect("an instance is kept where it is recorded")
            .clone();
        add_decided(&mut here, &made, &lambda);
        violations.extend(check_view(&here, accepted, |name| {
            name.as_str().starts_with(&format!("{at}/"))
        }));
    }
    violations
}

/// The state with, for each instance it made, a node for each node the
/// instantiated group decided, under the name of its copy, so that the
/// relations copied from it are that node's (`Recorder::nodes`).
fn view(state: &State, groups: &BTreeMap<Name, Rc<State>>) -> State {
    let mut view = state.clone();
    for made in &state.instances {
        let from = self::view(&groups[&made.group], groups);
        add_decided(&mut view, made, &from);
    }
    view
}

fn add_decided(here: &mut State, made: &Made, from: &State) {
    let mut copy = Instance::resume(made, from);
    // An instance of parts of C has the decided nodes in those parts.
    let components = made.parts.as_ref().map(|_| from.components());
    for (name, node) in &from.nodes {
        let Some(decision) = &node.decision else {
            continue;
        };
        if let (Some(parts), Some(components)) = (&made.parts, &components)
            && !components
                .of_node(from, node)
                .is_some_and(|part| parts.contains(&part))
        {
            continue;
        }
        let kind = copy.kind(here, &node.kind);
        let references = node
            .references
            .iter()
            .map(|&class| copy.class(here, class))
            .collect();
        let common = node.common.clone().map(|common| match common {
            Common::Use { part, classes } => Common::Use {
                part,
                classes: classes
                    .into_iter()
                    .map(|class| copy.class(here, class))
                    .collect(),
            },
            Common::Lend => Common::Lend,
        });
        here.nodes.insert(
            copy_name(&made.at, name),
            NodeState {
                kind,
                decision: Some(decision.clone()),
                common,
                references,
            },
        );
    }
}

/// Decides again each decided node `subject` admits, in a final state whose
/// program is `accepted` or not.
fn check_view<F>(view: &State, accepted: bool, subject: F) -> Vec<Violation>
where
    F: Fn(&Name) -> bool,
{
    let graph = EndGraph::of(view);
    let mut violations = Vec::new();
    for (name, node) in &view.nodes {
        let Some(decided) = node.decision.clone() else {
            continue;
        };
        if !subject(name) {
            continue;
        }
        let left_out = graph.exclusion(name);
        let again = match &node.kind {
            NodeKind::Bind { path, .. } if path.iter().any(|step| left_out.contains(step)) => {
                let mut cut = view.clone();
                for step in path.iter().filter(|step| left_out.contains(*step)) {
                    cut.nodes
                        .get_mut(step)
                        .expect("a binding's path is of the view's nodes")
                        .decision = None;
                }
                cut.decision_under(name, &left_out)
            }
            _ => view.decision_under(name, &left_out),
        };
        if again.as_ref() == Some(&decided) {
            continue;
        }
        let kind = if accepted {
            Kind::Soundness
        } else {
            Kind::Completeness
        };
        violations.push(Violation {
            kind,
            group: view.group.clone(),
            node: name.clone(),
            decided,
            again,
            left_out,
        });
    }
    violations
}

/// What each node records into at the end, as each of the two reads takes it.
struct Records {
    value: BTreeSet<ClassId>,
    reference: BTreeSet<ClassId>,
}

impl Records {
    fn of(&self, read: Read) -> &BTreeSet<ClassId> {
        match read {
            Read::Value => &self.value,
            Read::Reference => &self.reference,
        }
    }
}

/// The dependency at the end: for each
/// node, the nodes it depends on.
struct EndGraph {
    on: BTreeMap<Name, BTreeSet<Name>>,
}

impl EndGraph {
    fn of(view: &State) -> EndGraph {
        let reach = Reach::of(view);
        let records: BTreeMap<Name, Records> = view
            .nodes
            .keys()
            .map(|name| {
                let records = Records {
                    value: records(view, name, Read::Value),
                    reference: records(view, name, Read::Reference),
                };
                (name.clone(), records)
            })
            .collect();
        let on = view
            .nodes
            .keys()
            .map(|name| (name.clone(), depends_on(view, &reach, &records, name)))
            .collect();
        EndGraph { on }
    }

    /// `n` and every node that transitively depends on it.
    fn exclusion(&self, n: &Name) -> BTreeSet<Name> {
        let mut left_out = BTreeSet::from([n.clone()]);
        let mut work = vec![n.clone()];
        while let Some(at) = work.pop() {
            for (node, on) in &self.on {
                if on.contains(&at) && left_out.insert(node.clone()) {
                    work.push(node.clone());
                }
            }
        }
        left_out
    }
}

/// The nodes `name` depends on at the end: those that record into a class
/// reached from the classes at the positions it reads, as
/// what it reads takes them, and a binding's path.
fn depends_on(
    view: &State,
    reach: &Reach,
    records: &BTreeMap<Name, Records>,
    name: &Name,
) -> BTreeSet<Name> {
    let kind = &view.nodes[name].kind;
    let read = kind.read();
    let reached = reach.from(view, view.read_positions(kind));
    let mut on: BTreeSet<Name> = records
        .iter()
        .filter(|(other, records)| *other != name && !records.of(read).is_disjoint(&reached))
        .map(|(other, _)| other.clone())
        .collect();
    if let NodeKind::Bind { path, .. } = kind {
        for step in path {
            assert!(
                view.nodes.contains_key(step),
                "a binding's path is of the view's nodes"
            );
            on.insert(step.clone());
        }
    }
    on
}

/// The roots of the classes the node records into at the end, as a read of
/// `read` takes them. A use records into
/// what its candidate would record into, its
/// candidates' where it is not decided: the classes it makes anew are its own,
/// and no other node reads them. Another decided node
/// records into what its own relations record into, those of the instances it
/// made among them; one not decided, into the classes at its positions. A
/// read of whether a class is a reference leaves out what cannot decide it.
fn records(view: &State, name: &Name, read: Read) -> BTreeSet<ClassId> {
    let node = &view.nodes[name];
    let classes: BTreeSet<ClassId> = match (&node.kind, &node.decision) {
        (
            NodeKind::Use {
                callee,
                args,
                result,
                ..
            },
            decision,
        ) => {
            let candidates: Vec<usize> = match decision {
                Some(Decision::Use(id)) => vec![view.candidate_index(callee, args.len(), id)],
                Some(Decision::Never) => {
                    return result.iter().map(|&result| view.root(result)).collect();
                }
                Some(other) => unreachable!("a use is decided as a candidate or never: {other:?}"),
                None => (0..view.candidates(callee, args.len()).len()).collect(),
            };
            candidates
                .into_iter()
                .flat_map(|candidate| view.would_record_by(name, candidate).into_keys())
                .collect()
        }
        (_, Some(_)) => view
            .log
            .iter()
            .filter(|recorded| recorded.recorder.nodes().contains(name))
            .flat_map(|recorded| into(&recorded.relation))
            .collect(),
        (_, None) => view.positions(node),
    };
    let mut roots: BTreeSet<ClassId> = classes.into_iter().map(|class| view.root(class)).collect();
    if read == Read::Reference {
        let mut cannot_decide: BTreeSet<ClassId> = node.references.iter().copied().collect();
        match &node.kind {
            NodeKind::Use { .. } => {}
            NodeKind::Lend { place, .. } => {
                cannot_decide.insert(*place);
            }
            NodeKind::Bind { position, .. } => {
                cannot_decide.insert(*position);
            }
            NodeKind::Project { projection, .. } if *projection != Projection::Deref => {
                return BTreeSet::new();
            }
            NodeKind::Project { .. } | NodeKind::CallValue { .. } => {}
        }
        for class in cannot_decide {
            roots.remove(&view.root(class));
        }
    }
    roots
}

/// The classes a relation records into: the upper side of a lower bound or a
/// part lower bound, the type passed into at a boundary, both sides of an
/// equation, and the class a kind, a fact, a LitVar, a common part's record
/// of whether it is a reference or a lambda is given to.
fn into(relation: &Relation) -> Vec<ClassId> {
    match relation {
        Relation::LowerBound { upper, .. } | Relation::PartLowerBound { upper, .. } => {
            vec![*upper]
        }
        Relation::Boundary { to, .. } | Relation::Wrap { to, .. } => vec![*to],
        Relation::Equation(a, b) => vec![*a, *b],
        Relation::Fact(class, _)
        | Relation::Kind(class, ..)
        | Relation::Lit(class, _)
        | Relation::Reference(class, _)
        | Relation::Values(class, _) => vec![*class],
    }
}

/// Reach at the end: from a class to each
/// class it has as a lower bound or a part lower bound, along every one
/// recorded, a split's among them. Every class has taken its constructor by
/// the end.
struct Reach {
    edges: BTreeMap<ClassId, BTreeSet<ClassId>>,
}

impl Reach {
    fn of(view: &State) -> Reach {
        let mut edges: BTreeMap<ClassId, BTreeSet<ClassId>> = BTreeMap::new();
        for recorded in &view.log {
            if let Relation::LowerBound { lower, upper }
            | Relation::PartLowerBound { lower, upper } = recorded.relation
            {
                let (lower, upper) = (view.root(lower), view.root(upper));
                if lower != upper {
                    edges.entry(upper).or_default().insert(lower);
                }
            }
        }
        Reach { edges }
    }

    fn from<I>(&self, view: &State, classes: I) -> BTreeSet<ClassId>
    where
        I: IntoIterator<Item = ClassId>,
    {
        let mut reached: BTreeSet<ClassId> =
            classes.into_iter().map(|class| view.root(class)).collect();
        let mut work: Vec<ClassId> = reached.iter().copied().collect();
        while let Some(at) = work.pop() {
            for &next in self.edges.get(&at).into_iter().flatten() {
                if reached.insert(next) {
                    work.push(next);
                }
            }
        }
        reached
    }
}
