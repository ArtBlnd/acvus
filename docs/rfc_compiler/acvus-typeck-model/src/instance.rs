//! Instances: a binding group's C made anew at a use, and recorded in the use's
//! binding group (l.190–217).

use std::collections::{BTreeMap, BTreeSet};
use std::rc::Rc;

use crate::class::{ClassId, ClassName, KindSource};
use crate::node::{NodeKind, NodeState};
use crate::record::{Instantiated, Recording};
use crate::relation::{Recorder, Relation, copy_name};
use crate::solve::State;
use crate::ty::{Name, Ty};

/// An instance a state made: of the C of the binding group `group`, named `at`
/// and made by `by`, with each root of that group's state it met and that
/// root's class in the state that made it (l.217).
#[derive(Clone, Debug)]
pub struct Made {
    pub at: Name,
    pub by: Recorder,
    pub group: Name,
    pub made: BTreeMap<ClassId, ClassId>,
}

/// C of the binding group `from` reached its fixpoint in, being made anew at
/// the use `at` (l.214, l.217).
pub(crate) struct Instance<'a> {
    at: Rc<Name>,
    /// What made the instance, which every relation it records is recorded by
    /// (l.263).
    by: Recorder,
    from: &'a State,
    /// The roots of C's classes in `from`: those left unsolved, or whose value
    /// is not closed, and the universals.
    c: BTreeSet<ClassId>,
    /// Each root of `from` met so far, with its class here.
    made: BTreeMap<ClassId, ClassId>,
    /// The relations it has recorded here so far.
    recorded: Recording,
}

impl State {
    /// The roots of the classes C holds at this group's fixpoint (l.214):
    /// those left unsolved, or whose value is not closed, and the universals.
    pub(crate) fn c(&self) -> BTreeSet<ClassId> {
        self.classes
            .roots()
            .filter(|&root| self.is_universal(root) || self.value(root).is_none())
            .collect()
    }

    /// The relations of the log on a class of `c` that an instance copies, by
    /// their places: an equation is kept by the classes it unified.
    pub(crate) fn relations_on(&self, c: &BTreeSet<ClassId>) -> Vec<usize> {
        self.log
            .iter()
            .enumerate()
            .filter(|(_, recorded)| {
                !matches!(recorded.relation, Relation::Equation(..))
                    && recorded
                        .relation
                        .classes()
                        .into_iter()
                        .any(|class| c.contains(&self.root(class)))
            })
            .map(|(at, _)| at)
            .collect()
    }
}

impl<'a> Instance<'a> {
    pub(crate) fn new(at: &Name, by: Recorder, from: &'a State) -> Instance<'a> {
        Instance {
            at: Rc::new(at.clone()),
            by,
            from,
            c: from.c(),
            made: BTreeMap::new(),
            recorded: Recording::default(),
        }
    }

    /// The instance `made` by `by` resumed, to map more of `from` the way it
    /// did.
    pub(crate) fn resume(made: &Made, from: &'a State) -> Instance<'a> {
        Instance {
            made: made.made.clone(),
            ..Instance::new(&made.at, made.by.clone(), from)
        }
    }

    /// A node's kind here, its classes those here and a binding's path the
    /// copies of its projections.
    pub(crate) fn kind(&mut self, here: &mut State, kind: &NodeKind) -> NodeKind {
        match kind.clone() {
            NodeKind::Use {
                callee,
                args,
                result,
                exit,
            } => NodeKind::Use {
                callee,
                args: args.into_iter().map(|arg| self.class(here, arg)).collect(),
                result: self.class(here, result),
                exit: exit.map(|exit| self.class(here, exit)),
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
            NodeKind::CallValue {
                callee,
                args,
                result,
            } => NodeKind::CallValue {
                callee: self.class(here, callee),
                args: args.into_iter().map(|arg| self.class(here, arg)).collect(),
                result: self.class(here, result),
            },
        }
    }

    /// What records a copied relation here: the copy of an undecided node, or
    /// the instance, made by what made it.
    fn recorder(&self, of: &Recorder) -> Recorder {
        match of {
            Recorder::Node(node) if self.from.nodes[node].decision.is_none() => {
                Recorder::Node(self.node(node))
            }
            _ => Recorder::Instance {
                at: (*self.at).clone(),
                by: Box::new(self.by.clone()),
                of: Box::new(of.clone()),
            },
        }
    }

    pub(crate) fn node(&self, name: &Name) -> Name {
        copy_name(&self.at, name)
    }

    /// The class here of the class `id` of `from`: for a class of C, one made
    /// anew, holding the kind `from` solved it to; for another, its closed
    /// type, written (`design/09-group.md`, reading 3).
    pub(crate) fn class(&mut self, here: &mut State, id: ClassId) -> ClassId {
        let root = self.from.root(id);
        if let Some(&made) = self.made.get(&root) {
            return made;
        }
        let shared = self
            .from
            .classes
            .held(root)
            .members
            .iter()
            .find(|member| self.from.foreign.contains(member))
            .copied();
        if let Some(shared) = shared {
            let ClassName::Text(name) = self.from.arena.name(shared).clone() else {
                unreachable!("a class of an enclosing group is the text's");
            };
            let made = match here.arena.get(&ClassName::Text(name.clone())) {
                Some(made) => made,
                None => {
                    let unknown = self.from.enclosing_unknowns.contains(&shared);
                    here.share(&name, unknown)
                }
            };
            self.made.insert(root, made);
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
            let made = here.closed(name, &ty, &recorder, &mut self.recorded);
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
            let relation = Relation::Kind(made, term, KindSource::Solved);
            self.recorded.push(here.record(relation, recorder));
        }
        made
    }

    /// Records here every relation of `from` on a class of C, makes anew each
    /// node `from` left undecided (l.214), and keeps the instance here.
    /// A relation `from` split is split here too: what its split recorded is
    /// among the relations copied, and no class of the copy takes a
    /// constructor here that it did not hold in `from` (l.217, l.280).
    /// Returns what the instance recorded here, itself the instance made.
    pub(crate) fn record_rest(&mut self, here: &mut State) -> Recording {
        let from = self.from;
        for at in from.relations_on(&self.c) {
            let recorded = &from.log[at];
            let relation = recorded
                .relation
                .clone()
                .map(|class| self.class(here, class));
            let recorder = self.recorder(&recorded.recorder);
            let entry = here.record(relation, recorder);
            if from.split.contains(&at) {
                here.split.insert(entry.at);
                here.copied_split.insert(entry.at);
            }
            self.recorded.push(entry);
        }
        let undecided: Vec<(&Name, &NodeState)> = from
            .nodes
            .iter()
            .filter(|(_, node)| node.decision.is_none())
            .collect();
        for (name, node) in undecided {
            let kind = self.kind(here, &node.kind);
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
        here.instances.push(Made {
            at: (*self.at).clone(),
            by: self.by.clone(),
            group: from.group.clone(),
            made: self.made.clone(),
        });
        let mut recorded = std::mem::take(&mut self.recorded);
        recorded.instances.push(Instantiated {
            at: (*self.at).clone(),
            by: self.by.clone(),
            group: from.group.clone(),
            recorded: recorded.places().collect(),
        });
        recorded
    }
}
