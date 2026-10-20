//! Instances: a binding group's C made anew at a use, and recorded in the use's
//! binding group (l.195–222).

use std::collections::{BTreeMap, BTreeSet};
use std::rc::Rc;

use crate::class::{ClassId, ClassName, KindSource};
use crate::node::{NodeKind, NodeState};
use crate::relation::{Recorder, Relation};
use crate::solve::State;
use crate::ty::{Name, Ty};

/// C of the binding group `from` reached its fixpoint in, being made anew at
/// the use `at` (l.219, l.222).
pub(crate) struct Instance<'a> {
    at: Rc<Name>,
    from: &'a State,
    /// The roots of C's classes in `from`: those left unsolved, or whose value
    /// is not closed, and the universals.
    c: BTreeSet<ClassId>,
    /// Each root of `from` met so far, with its class here.
    made: BTreeMap<ClassId, ClassId>,
}

impl<'a> Instance<'a> {
    pub(crate) fn new(at: &Name, from: &'a State) -> Instance<'a> {
        let c = from
            .classes
            .roots()
            .filter(|&root| from.is_universal(root) || from.value(root).is_none())
            .collect();
        Instance {
            at: Rc::new(at.clone()),
            from,
            c,
            made: BTreeMap::new(),
        }
    }

    /// What records a copied relation here: the copy of an undecided node, or
    /// the use that made the instance.
    fn recorder(&self, of: &Recorder) -> Recorder {
        match of {
            Recorder::Node(node) if self.from.nodes[node].decision.is_none() => {
                Recorder::Node(self.node(node))
            }
            _ => Recorder::Instance {
                at: (*self.at).clone(),
                of: Box::new(of.clone()),
            },
        }
    }

    fn node(&self, name: &Name) -> Name {
        Name::new(&format!("{}/{name}", self.at))
    }

    /// The class here of the class `id` of `from`: for a class of C, one made
    /// anew, holding the kind `from` solved it to; for another, its closed
    /// type, written (`design/09-group.md`, reading 3).
    pub(crate) fn class(&mut self, here: &mut State, id: ClassId) -> ClassId {
        let root = self.from.root(id);
        if let Some(&made) = self.made.get(&root) {
            return made;
        }
        let name = ClassName::Copy {
            at: Rc::clone(&self.at),
            of: Rc::new(self.from.class_name(root)),
        };
        if !self.c.contains(&root) {
            let ty: Ty = self
                .from
                .value(root)
                .expect("a class outside C has a closed value");
            let recorder = self.recorder(&Recorder::Origin);
            let made = here.closed(name, &ty, &recorder);
            self.made.insert(root, made);
            return made;
        }
        let made = here.arena.intern(name);
        here.classes.grow(&here.arena);
        self.made.insert(root, made);
        let held = self.from.classes.held(root).clone();
        if held
            .members
            .iter()
            .any(|member| self.from.decided.contains(member))
        {
            here.decided.insert(made);
        }
        if let Some((kind, KindSource::Solved)) = held.kind {
            let term = kind.map(|_, part| self.class(here, part));
            let recorder = self.recorder(&Recorder::Split(root));
            here.record(Relation::Kind(made, term, KindSource::Solved), recorder);
        }
        made
    }

    /// Records here every relation of `from` on a class of C, and makes anew
    /// each node `from` left undecided (l.219).
    pub(crate) fn record_rest(&mut self, here: &mut State) {
        let from = self.from;
        for recorded in &from.log {
            let mentions_c = recorded
                .relation
                .classes()
                .into_iter()
                .any(|class| self.c.contains(&from.root(class)));
            if !mentions_c {
                continue;
            }
            let relation = match recorded.relation.clone() {
                Relation::Equation(..) => continue,
                relation => relation.map(|class| self.class(here, class)),
            };
            let recorder = self.recorder(&recorded.recorder);
            here.record(relation, recorder);
        }
        let undecided: Vec<(&Name, &NodeState)> = from
            .nodes
            .iter()
            .filter(|(_, node)| node.decision.is_none())
            .collect();
        for (name, node) in undecided {
            let kind = match node.kind.clone() {
                NodeKind::Use {
                    callee,
                    args,
                    result,
                } => NodeKind::Use {
                    callee,
                    args: args.into_iter().map(|arg| self.class(here, arg)).collect(),
                    result: self.class(here, result),
                },
                NodeKind::Lend {
                    mutability,
                    place,
                    result,
                } => NodeKind::Lend {
                    mutability,
                    place: self.class(here, place),
                    result: self.class(here, result),
                },
                NodeKind::Project {
                    projection,
                    operand,
                    result,
                } => NodeKind::Project {
                    projection,
                    operand: self.class(here, operand),
                    result: self.class(here, result),
                },
                NodeKind::Bind {
                    path,
                    position,
                    binding,
                } => NodeKind::Bind {
                    path: path.iter().map(|step| self.node(step)).collect(),
                    position: self.class(here, position),
                    binding: self.class(here, binding),
                },
            };
            let copy = self.node(name);
            here.add_node(&copy, kind);
            let shared = node.shared.clone();
            let references: BTreeSet<ClassId> = node
                .references
                .iter()
                .map(|&class| self.class(here, class))
                .collect();
            let state = here.nodes.get_mut(&copy).expect("the node was added");
            state.shared = shared;
            state.references = references;
        }
    }
}

impl State {
    /// A class made anew that holds the closed type `ty`, written.
    pub(crate) fn closed(&mut self, name: ClassName, ty: &Ty, recorder: &Recorder) -> ClassId {
        let shared = Rc::new(name.clone());
        let term = ty
            .term()
            .clone()
            .map(|key, part| self.closed(ClassName::Part(shared.clone(), key), &part, recorder));
        self.written(name, term, recorder)
    }
}
