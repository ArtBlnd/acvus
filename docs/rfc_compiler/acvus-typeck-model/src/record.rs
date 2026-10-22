//! Recording (l.217–231): the constraints a body's text gives, recorded in the
//! equivalence classes as they are given, whatever is solved (l.223) — a
//! meeting as a lower bound and a boundary as `T ⇝ U` (l.219), an invariant
//! place as an equation, and a use, a lend, a projection and a binding as a
//! node — and the classes of a type a declaration writes (l.120, l.211).

use std::collections::{BTreeMap, BTreeSet};
use std::rc::Rc;

use crate::class::{ClassId, ClassName, KindSource};
use crate::instance::Instance;
use crate::node::{NodeKind, NodeState};
use crate::program::{Function, Item, Origin, Place, Program, TextClass};
use crate::relation::{Recorded, Recorder, Relation};
use crate::sig::{SigTy, SigVar};
use crate::solve::State;
use crate::ty::{Name, PartKey, Term, Ty};
use crate::unify::Unification;

/// A relation's place in the log, and what unification did as the
/// equivalence classes took it.
#[must_use]
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Entry {
    pub at: usize,
    pub unification: Unification,
}

/// An instance made while recording (l.215): its name, what made it, the
/// binding group whose C it copies, and the relations it recorded, by their
/// places in the log.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Instantiated {
    pub at: Name,
    pub by: Recorder,
    pub group: Name,
    pub recorded: Vec<usize>,
}

/// What recording did (l.223): each relation recorded, in the order
/// recorded, and each instance made (l.215).
#[derive(Clone, Default, PartialEq, Eq, Debug)]
pub struct Recording {
    pub entries: Vec<Entry>,
    pub instances: Vec<Instantiated>,
}

impl Recording {
    pub fn push(&mut self, entry: Entry) {
        self.entries.push(entry);
    }

    pub fn append(&mut self, mut other: Recording) {
        self.entries.append(&mut other.entries);
        self.instances.append(&mut other.instances);
    }

    /// The relations recorded, by their places in the log.
    pub fn places(&self) -> impl Iterator<Item = usize> + '_ {
        self.entries.iter().map(|entry| entry.at)
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

impl State {
    /// Records the relation, which the equivalence classes take at once
    /// (l.223).
    pub(crate) fn record(&mut self, relation: Relation, recorder: Recorder) -> Entry {
        let unification = self.hold(&relation);
        let at = self.log.len();
        self.log.push(Recorded { relation, recorder });
        Entry { at, unification }
    }

    /// The types the text gives its classes: a literal's LitVar, a type the
    /// text or the host writes, and a lambda's function value (l.223).
    pub(crate) fn record_origins(
        &mut self,
        program: &Program,
        classes: &[&TextClass],
    ) -> Recording {
        let mut recorded = Recording::default();
        for class in classes {
            let id = self.text(&class.name);
            match &class.origin {
                Origin::Var => {}
                Origin::Lit(lit) => {
                    recorded.push(self.record(Relation::Lit(id, lit.clone()), Recorder::Origin));
                }
                Origin::Written(term) => {
                    let term = term.clone().map(|_, part| self.text(&part));
                    let relation = Relation::Kind(id, term, KindSource::Written);
                    recorded.push(self.record(relation, Recorder::Origin));
                }
                Origin::Host(term) => {
                    self.hosts.insert(id);
                    let term = term.clone().map(|_, part| self.text(&part));
                    let relation = Relation::Kind(id, term, KindSource::Written);
                    recorded.push(self.record(relation, Recorder::Origin));
                }
                Origin::Lambda(lambda) => {
                    let function = program.function(lambda).expect("the lambda is a function");
                    let arity = function.params.len();
                    let term = Term::FunctionValue { arity };
                    let relation = Relation::Kind(id, term, KindSource::Written);
                    recorded.push(self.record(relation, Recorder::Origin));
                    let relation = Relation::Values(id, lambda.clone());
                    recorded.push(self.record(relation, Recorder::Origin));
                }
            }
        }
        recorded
    }

    /// The equations the text's invariant places give (l.229).
    pub(crate) fn record_equations(&mut self, items: &[(usize, &Item)]) -> Recording {
        let mut recorded = Recording::default();
        for &(i, item) in items {
            if let Item::Equation(a, b) = item {
                let relation = Relation::Equation(self.text(a), self.text(b));
                recorded.push(self.record(relation, Recorder::Text(i)));
            }
        }
        recorded
    }

    /// The rest of the text: each value passing into a type, as a meeting or
    /// a boundary (l.219, l.227–228), each call of a function (l.192–195), and
    /// each other use, lend, projection and binding, as a node (l.231).
    pub(crate) fn record_items(
        &mut self,
        program: &Program,
        items: &[(usize, &Item)],
        solved: &BTreeMap<Name, Rc<State>>,
    ) -> Recording {
        let mut recorded = Recording::default();
        for &(i, item) in items {
            match item {
                Item::Pass { value, into, place } => {
                    let (value, into) = (self.text(value), self.text(into));
                    let meeting = match place {
                        Place::ThroughRef => false,
                        Place::ByName => {
                            self.is_unknown(into) || self.enclosing_unknowns.contains(&into)
                        }
                        Place::Slot => self.is_unknown(into),
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
                    recorded.push(self.record(relation, Recorder::Text(i)));
                }
                Item::CallValue {
                    name,
                    callee,
                    args,
                    result,
                } => {
                    let kind = NodeKind::CallValue {
                        callee: self.text(callee),
                        args: args.iter().map(|arg| self.text(arg)).collect(),
                        result: self.text(result),
                    };
                    self.add_node(name, kind);
                }
                Item::Use {
                    callee,
                    args,
                    result,
                    ..
                } if program.function(callee).is_some() => {
                    let function = program.function(callee).expect("the function is declared");
                    let args: Vec<ClassId> = args.iter().map(|arg| self.text(arg)).collect();
                    let result = self.text(result);
                    recorded.append(self.call(i, function, &args, result, solved));
                }
                Item::Use {
                    name,
                    callee,
                    args,
                    result,
                    exit,
                } => {
                    let kind = NodeKind::Use {
                        callee: callee.clone(),
                        args: args.iter().map(|arg| self.text(arg)).collect(),
                        result: self.text(result),
                        exit: exit.as_ref().map(|exit| self.text(exit)),
                    };
                    self.add_node(name, kind);
                }
                Item::Lend {
                    name,
                    mutability,
                    place,
                    result,
                } => {
                    let kind = NodeKind::Lend {
                        mutability: *mutability,
                        place: self.text(place),
                        result: self.text(result),
                    };
                    self.add_node(name, kind);
                }
                Item::Project {
                    name,
                    projection,
                    operand,
                    result,
                } => {
                    let kind = NodeKind::Project {
                        projection: projection.clone(),
                        operand: self.text(operand),
                        result: self.text(result),
                    };
                    self.add_node(name, kind);
                }
                Item::Bind {
                    name,
                    path,
                    position,
                    binding,
                } => {
                    let kind = NodeKind::Bind {
                        path: path.clone(),
                        position: self.text(position),
                        binding: self.text(binding),
                    };
                    self.add_node(name, kind);
                }
                Item::Equation(..) => {}
            }
        }
        recorded
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

    /// A use that calls a function (l.192–195). Within the function's group it
    /// takes that body's own types: each argument passes into its parameter's
    /// type, universal and so a boundary (Example 10), and the body's result
    /// type enters the use's result. From another group it makes an instance
    /// of the group's C (l.215).
    pub(crate) fn call(
        &mut self,
        item: usize,
        function: &Function,
        args: &[ClassId],
        result: ClassId,
        solved: &BTreeMap<Name, Rc<State>>,
    ) -> Recording {
        let recorder = Recorder::Text(item);
        if function.group == self.group || self.enclosing.contains(&function.group) {
            let params: Vec<ClassId> = function.params.iter().map(|p| self.text(p)).collect();
            let returned = self.text(&function.result);
            return self.record_call(args, &params, returned, result, &recorder);
        }
        let from = Rc::clone(
            solved
                .get(&function.group)
                .unwrap_or_else(|| panic!("{} is solved first (l.213)", function.group)),
        );
        let at = Name::new(&format!("{}@{item}", function.name));
        let mut copy = Instance::new(&at, recorder.clone(), &from);
        let params: Vec<ClassId> = function
            .params
            .iter()
            .map(|param| copy.class(self, from.text(param)))
            .collect();
        let returned = copy.class(self, from.text(&function.result));
        let mut recorded = copy.record_rest(self);
        recorded.append(self.record_call(args, &params, returned, result, &recorder));
        recorded
    }

    /// Each argument passing into its parameter's type, and the result type
    /// passing into the result (l.192–197), each a meeting where the type
    /// passed into is an unknown and a boundary otherwise (l.219).
    pub(crate) fn record_call(
        &mut self,
        args: &[ClassId],
        params: &[ClassId],
        returned: ClassId,
        result: ClassId,
        recorder: &Recorder,
    ) -> Recording {
        let mut recorded = Recording::default();
        for (&value, &into) in args.iter().zip(params).chain([(&returned, &result)]) {
            let relation = if self.is_unknown(into) {
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
            recorded.push(self.record(relation, recorder.clone()));
        }
        recorded
    }

    /// A class made anew that holds nothing, an unknown of the group (l.211).
    pub(crate) fn fresh(&mut self, name: ClassName) -> ClassId {
        let class = self.arena.intern(name);
        self.classes.grow(&self.arena);
        class
    }

    /// A class made anew that holds `term` as a declaration writes it, the
    /// relation it records kept in `recorded`.
    pub(crate) fn written(
        &mut self,
        name: ClassName,
        term: Term<ClassId>,
        recorder: &Recorder,
        recorded: &mut Recording,
    ) -> ClassId {
        let class = self.arena.intern(name);
        self.classes.grow(&self.arena);
        self.decided.insert(class);
        recorded.push(self.record(
            Relation::Kind(class, term, KindSource::Written),
            recorder.clone(),
        ));
        class
    }

    /// The classes of a type a declaration writes: the type named `name`, its
    /// parts named after it, and its variables those of `vars`, each made
    /// where first met and named by `var_name`. A variable is an unknown, and
    /// a written term is decided (l.211). The relations it records are kept in
    /// `recorded`.
    pub(crate) fn instantiate<F>(
        &mut self,
        ty: &SigTy,
        vars: &mut BTreeMap<SigVar, ClassId>,
        name: ClassName,
        var_name: &F,
        recorder: &Recorder,
        recorded: &mut Recording,
    ) -> ClassId
    where
        F: Fn(SigVar) -> ClassName,
    {
        match ty {
            SigTy::Var(var) => {
                if let Some(&class) = vars.get(var) {
                    return class;
                }
                let class = self.arena.intern(var_name(*var));
                self.classes.grow(&self.arena);
                vars.insert(*var, class);
                class
            }
            SigTy::Term(term) => {
                let shared = Rc::new(name.clone());
                let term = (**term).clone().map(|key: PartKey, part| {
                    let part_name = ClassName::Part(shared.clone(), key);
                    self.instantiate(&part, vars, part_name, var_name, recorder, recorded)
                });
                self.written(name, term, recorder, recorded)
            }
        }
    }

    /// A class made anew that holds the closed type `ty`, written, the
    /// relations it records kept in `recorded`.
    pub(crate) fn closed(
        &mut self,
        name: ClassName,
        ty: &Ty,
        recorder: &Recorder,
        recorded: &mut Recording,
    ) -> ClassId {
        let shared = Rc::new(name.clone());
        let term = ty.term().clone().map(|key, part| {
            let part_name = ClassName::Part(shared.clone(), key);
            self.closed(part_name, &part, recorder, recorded)
        });
        self.written(name, term, recorder, recorded)
    }
}
