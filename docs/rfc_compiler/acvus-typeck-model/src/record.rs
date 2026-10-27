//! Recording: the constraints a body's text gives, recorded in the
//! equivalence classes as they are given, whatever is solved — a
//! lower bound as `τ ≤ α` and a coercion constraint as `T ⇝ U`, an invariant
//! place as an equation, and a use, a lend, a projection and a binding as a
//! disjunctive constraint — and the classes of a type a declaration writes.

use std::collections::{BTreeMap, BTreeSet};
use std::rc::Rc;

use crate::class::{ClassId, ClassName, StructureSource};
use crate::disjunction::{Disjunction, DisjunctionAt};
use crate::domain::Domain;
use crate::instance::Instance;
use crate::program::{Function, Item, Origin, Part, Place, Position, Program, TextClass};
use crate::relation::{Item as RecordItem, Recorder, Relation, Role};
use crate::sig::{SigTy, SigVar};
use crate::solve::State;
use crate::ty::{Name, PartKey, Term, Ty};
use crate::unify::Unification;

/// A relation's place in the record, and what unification did as the
/// equivalence classes took it.
#[must_use]
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Entry {
    pub at: usize,
    pub unification: Unification,
}

/// An instance made while recording: its name, what made it, the
/// binding group whose C it copies, and the relations it recorded, by their
/// places in the record.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Instantiated {
    pub at: Name,
    pub by: Recorder,
    pub group: Name,
    pub recorded: Vec<usize>,
}

/// What recording did: each relation recorded, in the order
/// recorded, and each instance made.
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

    /// The relations recorded, by their places in the record.
    pub fn places(&self) -> impl Iterator<Item = usize> + '_ {
        self.entries.iter().map(|entry| entry.at)
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

impl State {
    /// Records the relation in the record, which the equivalence classes take
    /// at once, having no value where they cannot.
    pub(crate) fn record(&mut self, relation: Relation, recorder: Recorder) -> Entry {
        let at = self
            .classes
            .record(RecordItem::Relation(relation.clone()), recorder);
        let unification = self.hold(&relation);
        Entry { at, unification }
    }

    /// Records the disjunctive constraint `name` in the record, undecided.
    /// The equivalence classes take nothing from it until it is decided.
    pub(crate) fn add_disjunction(
        &mut self,
        name: &Name,
        kind: Disjunction,
        recorder: Recorder,
    ) -> Entry {
        let item = RecordItem::Disjunction {
            name: name.clone(),
            kind,
            decision: None,
        };
        let at = self.classes.record(item, recorder);
        Entry {
            at,
            unification: Unification::default(),
        }
    }

    /// The types the text gives its classes: a literal's LitVar, a type the
    /// text or the host writes, and a lambda's function value.
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
                    let relation = Relation::Domain(id, Domain::of_literal(lit));
                    recorded.push(self.record(relation, Recorder::Origin));
                }
                Origin::Written(term) | Origin::Host(term) | Origin::Outward(term) => {
                    let term = term.clone().map(|_, part| self.text(&part));
                    let relation = Relation::Structure(id, term, StructureSource::Written);
                    recorded.push(self.record(relation, Recorder::Origin));
                }
                Origin::Lambda(lambda) => {
                    let function = program.function(lambda).expect("the lambda is a function");
                    let arity = function.params.len();
                    let term = Term::FunctionValue { arity };
                    let relation = Relation::Structure(id, term, StructureSource::Written);
                    recorded.push(self.record(relation, Recorder::Origin));
                    let relation = Relation::Values(id, lambda.clone());
                    recorded.push(self.record(relation, Recorder::Origin));
                }
            }
        }
        recorded
    }

    /// The equations the text's invariant places give.
    pub(crate) fn record_equations(&mut self, items: &[(usize, &Item, Part)]) -> Recording {
        let mut recorded = Recording::default();
        for &(i, item, _) in items {
            if let Item::Equation(a, b) = item {
                let relation = Relation::Equation(self.text(a), self.text(b));
                recorded.push(self.record(relation, Recorder::Text(i)));
            }
        }
        recorded
    }

    /// The rest of the text: each value passing into a type, as a lower bound or
    /// a coercion constraint, each call of a function, and
    /// each other use, lend, projection and binding, as a disjunctive constraint; of a
    /// use, the constraints at the positions its part holds.
    pub(crate) fn record_items(
        &mut self,
        program: &Program,
        items: &[(usize, &Item, Part)],
        solved: &BTreeMap<Name, Rc<State>>,
    ) -> Recording {
        let mut recorded = Recording::default();
        for (i, item, part) in items {
            let i = *i;
            let held = |state: &State, position: Position| {
                part.holds(position)
                    .then(|| state.text(item.class_at(position)))
            };
            match item {
                Item::Pass { value, into, place } => {
                    let (value, into) = (self.text(value), self.text(into));
                    let lower_bound = match place {
                        Place::ThroughRef => false,
                        Place::ByName => {
                            self.is_unknown(into) || self.is_enclosing_unknown(into)
                        }
                        Place::Slot => self.is_unknown(into),
                    };
                    let relation = if lower_bound {
                        Relation::LowerBound {
                            lower: value,
                            upper: into,
                        }
                    } else {
                        Relation::CoercionConstraint {
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
                    let kind = Disjunction::CallValue {
                        callee: self.text(callee),
                        args: args.iter().map(|arg| self.text(arg)).collect(),
                        result: self.text(result),
                    };
                    recorded.push(self.add_disjunction(name, kind, Recorder::Text(i)));
                }
                Item::Use { callee, args, .. } if program.function(callee).is_some() => {
                    let function = program.function(callee).expect("the function is declared");
                    let args: Vec<Option<ClassId>> = (0..args.len())
                        .map(|k| held(self, Position::Arg(k)))
                        .collect();
                    let result = held(self, Position::Result);
                    recorded.append(self.call(i, function, &args, result, solved));
                }
                Item::Use {
                    name, callee, args, ..
                } if self.candidates(callee, args.len()).len() == 1 => {
                    let held: Vec<(Position, ClassId)> = item
                        .positions()
                        .into_iter()
                        .filter_map(|position| Some((position, held(self, position)?)))
                        .collect();
                    let kind = Disjunction::Use {
                        callee: callee.clone(),
                        args: (0..args.len())
                            .map(|k| held.iter().find(|(p, _)| *p == Position::Arg(k)).map(|(_, c)| *c))
                            .collect(),
                        result: held.iter().find(|(p, _)| *p == Position::Result).map(|(_, c)| *c),
                        exit: held.iter().find(|(p, _)| *p == Position::Exit).map(|(_, c)| *c),
                    };
                    recorded.append(self.record_one(name, callee, kind, Recorder::Text(i)));
                }
                Item::Use {
                    name, callee, args, ..
                } => {
                    // Of several candidates, the positions where every
                    // candidate writes one closed type are recorded apart, by
                    // the generalization; the disjunctive constraint holds the rest.
                    let several = self.candidates(callee, args.len()).len() >= 2;
                    let closed: BTreeSet<Position> = if several {
                        let sigs: Vec<crate::sig::Signature> = self
                            .candidates(callee, args.len())
                            .into_iter()
                            .map(|decl| decl.sig.clone())
                            .collect();
                        let generalization = crate::disjunction::generalization::Generalization::of(
                            &sigs.iter().collect::<Vec<_>>(),
                        );
                        item.positions()
                            .into_iter()
                            .filter(|&position| generalization.is_closed_at(position))
                            .collect()
                    } else {
                        BTreeSet::new()
                    };
                    let apart: Vec<(Position, ClassId)> = closed
                        .iter()
                        .filter_map(|&position| Some((position, held(self, position)?)))
                        .collect();
                    let disjunction = |position: Position| {
                        held(self, position).filter(|_| !closed.contains(&position))
                    };
                    let kind = Disjunction::Use {
                        callee: callee.clone(),
                        args: (0..args.len())
                            .map(|k| disjunction(Position::Arg(k)))
                            .collect(),
                        result: disjunction(Position::Result),
                        exit: item
                            .positions()
                            .contains(&Position::Exit)
                            .then(|| disjunction(Position::Exit))
                            .flatten(),
                    };
                    // The use is an item of every group that records a
                    // position of it; one holding no position where the
                    // candidates differ has nothing to decide here.
                    recorded.push(self.add_disjunction(name, kind, Recorder::Text(i)));
                    recorded.append(self.record_shared(name));
                    if !apart.is_empty() {
                        recorded.append(self.record_closed(name, callee, args.len(), &apart));
                    }
                }
                Item::Lend {
                    name,
                    mutability,
                    place,
                    result,
                } => {
                    let kind = Disjunction::Lend {
                        mutability: *mutability,
                        place: self.text(place),
                        result: self.text(result),
                    };
                    recorded.push(self.add_disjunction(name, kind, Recorder::Text(i)));
                    recorded.append(self.record_shared(name));
                }
                Item::Project {
                    name,
                    projection,
                    operand,
                    result,
                } => {
                    let kind = Disjunction::Project {
                        projection: projection.clone(),
                        operand: self.text(operand),
                        result: self.text(result),
                    };
                    recorded.push(self.add_disjunction(name, kind, Recorder::Text(i)));
                }
                Item::Bind {
                    name,
                    path,
                    position,
                    binding,
                } => {
                    let kind = Disjunction::Bind {
                        path: path.clone(),
                        position: self.text(position),
                        binding: self.text(binding),
                    };
                    recorded.push(self.add_disjunction(name, kind, Recorder::Text(i)));
                }
                Item::Equation(..) => {}
            }
        }
        recorded
    }

    /// A use that calls a function, at the positions of its
    /// constraints this group records, none at the others. Within the
    /// function's group it takes that body's own types: each argument passes
    /// into its parameter's type, universal and so a coercion constraint,
    /// and the body's result type enters the use's result. From another group
    /// it makes an instance of the group's C: of all of it where this
    /// group records every position, and otherwise of the parts of C the
    /// positions it records lie in (`Instance::part`).
    pub(crate) fn call(
        &mut self,
        item: usize,
        function: &Function,
        args: &[Option<ClassId>],
        result: Option<ClassId>,
        solved: &BTreeMap<Name, Rc<State>>,
    ) -> Recording {
        let recorder = Recorder::Text(item);
        let pairs = |params: Vec<Option<ClassId>>, returned: Option<ClassId>| {
            let pairs: Vec<(ClassId, ClassId)> = args
                .iter()
                .zip(params)
                .filter_map(|(arg, param)| Some(((*arg)?, param?)))
                .chain(returned.zip(result))
                .collect();
            pairs
        };
        if function.group == self.group || self.enclosing.contains(&function.group) {
            let params: Vec<Option<ClassId>> = function
                .params
                .iter()
                .zip(args)
                .map(|(param, arg)| arg.map(|_| self.text(param)))
                .collect();
            let returned = result.map(|_| self.text(&function.result));
            return self.record_call(&pairs(params, returned), &recorder);
        }
        let from = Rc::clone(
            solved
                .get(&function.group)
                .unwrap_or_else(|| panic!("{} is solved first", function.group)),
        );
        let at = Name::new(&format!("{}@{item}", function.name));
        let positions: Vec<(ClassId, bool)> = function
            .params
            .iter()
            .zip(args)
            .map(|(param, arg)| (from.text(param), arg.is_some()))
            .chain([(from.text(&function.result), result.is_some())])
            .collect();
        let mut copy = if positions.iter().all(|&(_, held)| held) {
            Instance::new(&at, recorder.clone(), &from)
        } else {
            Instance::part(&at, recorder.clone(), &from, &positions)
        };
        let params: Vec<Option<ClassId>> = function
            .params
            .iter()
            .zip(args)
            .map(|(param, arg)| arg.map(|_| copy.class(self, from.text(param))))
            .collect();
        let returned = result.map(|_| copy.class(self, from.text(&function.result)));
        let mut recorded = copy.record_rest(self);
        recorded.append(self.record_call(&pairs(params, returned), &recorder));
        recorded
    }

    /// Each value passing into a type, as `pairs` gives them: an argument
    /// into its parameter's type, and the result type into the result,
    /// each a lower bound where the type passed into is an unknown
    /// and a coercion constraint otherwise.
    pub(crate) fn record_call(
        &mut self,
        pairs: &[(ClassId, ClassId)],
        recorder: &Recorder,
    ) -> Recording {
        let mut recorded = Recording::default();
        for &(value, into) in pairs {
            let relation = if self.is_unknown(into) {
                Relation::LowerBound {
                    lower: value,
                    upper: into,
                }
            } else {
                Relation::CoercionConstraint {
                    from: value,
                    to: into,
                }
            };
            recorded.push(self.record(relation, recorder.clone()));
        }
        recorded
    }

    /// A class made anew that holds nothing, labelled `label`: the parts a
    /// structure writes, which the structure's relation attaches, and the
    /// text's classes, which their standing attaches.
    pub(crate) fn fresh(&mut self, label: ClassName) -> ClassId {
        let class = self.arena.intern(label);
        self.classes.grow(&self.arena);
        class
    }

    /// A class made anew by an item, in a role of it: made, labelled and
    /// attached to the item at once.
    pub(crate) fn make(&mut self, making: Making) -> ClassId {
        let class = self.fresh(making.label);
        self.classes.attach(making.item, class, making.role);
        class
    }

    /// A class made anew that holds `term` as a declaration writes it, the
    /// relation it records kept in `recorded`.
    pub(crate) fn written(
        &mut self,
        making: Making,
        term: Term<ClassId>,
        recorder: &Recorder,
        recorded: &mut Recording,
    ) -> ClassId {
        let class = self.make(making);
        recorded.push(self.record(
            Relation::Structure(class, term, StructureSource::Written),
            recorder.clone(),
        ));
        class
    }

    /// The classes of a type a declaration writes: the type made by `making`,
    /// its parts labelled after it, and its variables those of `vars`, each
    /// made where first found by `var`. A variable is an unknown, and a term
    /// is written. The relations it records are kept in `recorded`.
    pub(crate) fn instantiate<F>(
        &mut self,
        ty: &SigTy,
        vars: &mut BTreeMap<SigVar, ClassId>,
        making: Making,
        var: &F,
        recorder: &Recorder,
        recorded: &mut Recording,
    ) -> ClassId
    where
        F: Fn(SigVar) -> Making,
    {
        match ty {
            SigTy::Var(v) => {
                if let Some(&class) = vars.get(v) {
                    return class;
                }
                let class = self.make(var(*v));
                vars.insert(*v, class);
                class
            }
            SigTy::Term(term) => {
                let shared = Rc::new(making.label.clone());
                let term = (**term).clone().map(|key: PartKey, part| {
                    let part = self.instantiate_part(&part, vars, shared.clone(), key, var, recorder, recorded);
                    part
                });
                self.written(making, term, recorder, recorded)
            }
        }
    }

    /// A part of a type a declaration writes, labelled after the class
    /// `owner` whose part of key `key` it is: a variable as `instantiate`
    /// makes it, and a term written, whose structure attaches its parts.
    #[allow(clippy::too_many_arguments)]
    fn instantiate_part<F>(
        &mut self,
        ty: &SigTy,
        vars: &mut BTreeMap<SigVar, ClassId>,
        owner: Rc<ClassName>,
        key: PartKey,
        var: &F,
        recorder: &Recorder,
        recorded: &mut Recording,
    ) -> ClassId
    where
        F: Fn(SigVar) -> Making,
    {
        match ty {
            SigTy::Var(v) => {
                if let Some(&class) = vars.get(v) {
                    return class;
                }
                let class = self.make(var(*v));
                vars.insert(*v, class);
                class
            }
            SigTy::Term(term) => {
                let label = ClassName::Part(owner, key);
                let shared = Rc::new(label.clone());
                let term = (**term).clone().map(|key: PartKey, part| {
                    self.instantiate_part(&part, vars, shared.clone(), key, var, recorder, recorded)
                });
                let class = self.fresh(label);
                recorded.push(self.record(
                    Relation::Structure(class, term, StructureSource::Written),
                    recorder.clone(),
                ));
                class
            }
        }
    }

    /// A class made anew by `making` that holds the closed type `ty`,
    /// written, its parts labelled after it, the relations it records kept
    /// in `recorded`.
    pub(crate) fn closed(
        &mut self,
        making: Making,
        ty: &Ty,
        recorder: &Recorder,
        recorded: &mut Recording,
    ) -> ClassId {
        let shared = Rc::new(making.label.clone());
        let term = ty.term().clone().map(|key, part| {
            self.closed_part(shared.clone(), key, &part, recorder, recorded)
        });
        self.written(making, term, recorder, recorded)
    }

    fn closed_part(
        &mut self,
        owner: Rc<ClassName>,
        key: PartKey,
        ty: &Ty,
        recorder: &Recorder,
        recorded: &mut Recording,
    ) -> ClassId {
        let label = ClassName::Part(owner, key);
        let shared = Rc::new(label.clone());
        let term = ty.term().clone().map(|key, part| {
            self.closed_part(shared.clone(), key, &part, recorder, recorded)
        });
        let class = self.fresh(label);
        recorded.push(self.record(
            Relation::Structure(class, term, StructureSource::Written),
            recorder.clone(),
        ));
        class
    }
}

/// A class an item is to make anew: the item, the class's role in it, and
/// its label.
#[derive(Clone, Debug)]
pub struct Making {
    pub item: usize,
    pub role: Role,
    pub label: ClassName,
}

/// A disjunctive constraint as the maker of classes: each class it makes
/// is labelled after it and takes the role of what it is made as.
#[derive(Clone, Debug)]
pub(crate) struct Owner {
    pub(crate) item: usize,
    pub(crate) name: Rc<Name>,
}

impl Owner {
    /// The making of the class the disjunctive constraint makes as `at`.
    pub(crate) fn made(&self, at: DisjunctionAt) -> Making {
        Making {
            item: self.item,
            role: Role::Made(at.clone()),
            label: ClassName::Disjunction(Rc::clone(&self.name), at),
        }
    }
}

impl State {
    /// The disjunctive constraint of this name as a maker of classes.
    pub(crate) fn owner(&self, name: &Name) -> Owner {
        Owner {
            item: self.disjunction_at(name),
            name: Rc::new(name.clone()),
        }
    }
}
