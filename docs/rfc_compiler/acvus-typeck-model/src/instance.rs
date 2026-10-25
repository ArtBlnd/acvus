//! Instances: a binding group's C made anew at a use, and recorded in the use's
//! binding group.

use std::collections::{BTreeMap, BTreeSet};
use std::rc::Rc;

use crate::class::{ClassId, ClassName, StructureSource};
use crate::disjunction::{Disjunction, DisjunctionState, Generalized};
use crate::record::{Instantiated, Recording};
use crate::relation::{Recorder, Relation, copy_name};
use crate::solve::State;
use crate::ty::{Name, Term, Ty};

/// An instance a state made: of the C of the binding group `group`, named `at`
/// and made by `by`, with each root of that group's state it found and that
/// root's class in the state that made it.
#[derive(Clone, Debug)]
pub struct Made {
    pub at: Name,
    pub by: Recorder,
    pub group: Name,
    pub made: BTreeMap<ClassId, ClassId>,
    /// The parts of that C it made anew (`State::components`), none naming
    /// all of it.
    pub parts: Option<BTreeSet<usize>>,
}

/// C's classes, each in its part: two of C's classes lie in one part
/// where one of C's relations is on both, a disjunctive constraint left undecided names both, a
/// binding's disjunctive constraint and the projections on its path name them, one is a part of
/// the other's structure, or one holds a function value whose Values name a lambda
/// placed in the group and the other lies in that lambda's environment, which
/// an instance makes anew with it. A part is what an instance makes anew
/// together, and a use of the group's function ties the positions whose types
/// lie in one part.
pub(crate) struct Components {
    /// Each root of C, with its part.
    parts: BTreeMap<ClassId, usize>,
}

impl Components {
    /// The part of C the class lies in, none where it lies outside C.
    pub(crate) fn of(&self, state: &State, class: ClassId) -> Option<usize> {
        self.parts.get(&state.root(class)).copied()
    }

    /// The part of C a disjunctive constraint left undecided lies in, none where it names no
    /// class of C.
    pub(crate) fn of_disjunction(
        &self,
        state: &State,
        disjunction: &DisjunctionState,
    ) -> Option<usize> {
        named_by(disjunction)
            .into_iter()
            .find_map(|class| self.of(state, class))
    }
}

/// C of the binding group `from` reached its fixpoint in, being made anew at
/// the use `at`.
pub(crate) struct Instance<'a> {
    at: Rc<Name>,
    /// What made the instance, which every relation it records is recorded by.
    by: Recorder,
    from: &'a State,
    /// The roots of C's classes in `from` it makes anew: those left unsolved,
    /// or whose value is not closed, and the universals, all of them or those
    /// in the parts of C at the positions the use records (`Instance::part`).
    c: BTreeSet<ClassId>,
    /// The parts of C it makes anew, none naming all of C.
    parts: Option<BTreeSet<usize>>,
    /// The disjunctive constraints `from` left undecided it makes anew, none naming them all.
    disjunctions: Option<BTreeSet<Name>>,
    /// Each root of `from` found so far, with its class here.
    made: BTreeMap<ClassId, ClassId>,
    /// The relations it has recorded here so far.
    recorded: Recording,
}

impl State {
    /// The roots of the classes C holds at this group's fixpoint:
    /// those left unsolved, or whose value is not closed, the universals,
    /// those whose value carries an identity this group's body made, and
    /// those whose function value names a lambda whose environment is here,
    /// each of which each instance makes anew.
    pub(crate) fn c(&self) -> BTreeSet<ClassId> {
        let made = self.made_identities();
        self.classes
            .roots()
            .filter(|&root| {
                self.is_universal(root)
                    || self
                        .value(root)
                        .is_none_or(|value| carries_identity(&value, &made))
                    || self
                        .values_of(root)
                        .iter()
                        .any(|lambda| self.environment_here(lambda))
            })
            .collect()
    }

    /// C's classes, each in its part (`Components`).
    pub(crate) fn components(&self) -> Components {
        let c = self.c();
        let mut edges: BTreeMap<ClassId, BTreeSet<ClassId>> = BTreeMap::new();
        let mut tie = |classes: Vec<ClassId>| {
            let roots: Vec<ClassId> = classes
                .into_iter()
                .map(|class| self.root(class))
                .filter(|root| c.contains(root))
                .collect();
            for &a in &roots {
                for &b in &roots {
                    edges.entry(a).or_default().insert(b);
                }
            }
        };
        for at in self.relations_on(&c) {
            let relation = &self.log[at].relation;
            let mut classes = relation.classes();
            if let Relation::Values(_, lambda) = relation
                && self.environment_here(lambda)
            {
                classes.extend(self.environment_of(lambda).into_iter().flatten());
            }
            tie(classes);
        }
        for disjunction in self
            .disjunctions
            .values()
            .filter(|disjunction| disjunction.decision.is_none())
        {
            let mut classes = named_by(disjunction);
            if let Disjunction::Bind { path, .. } = &disjunction.kind {
                for step in path {
                    classes.extend(
                        self.disjunctions
                            .get(step)
                            .map(named_by)
                            .into_iter()
                            .flatten(),
                    );
                }
            }
            tie(classes);
        }
        for &root in &c {
            if let Some((structure, _)) = &self.classes.held(root).structure {
                let parts = structure.parts().into_iter().map(|(_, part)| *part);
                tie([root].into_iter().chain(parts).collect());
            }
        }
        let mut parts: BTreeMap<ClassId, usize> = BTreeMap::new();
        let mut count = 0;
        for &start in &c {
            if parts.contains_key(&start) {
                continue;
            }
            let part = count;
            count += 1;
            let mut work = vec![start];
            parts.insert(start, part);
            while let Some(at) = work.pop() {
                for &next in edges.get(&at).into_iter().flatten() {
                    if let std::collections::btree_map::Entry::Vacant(entry) = parts.entry(next) {
                        entry.insert(part);
                        work.push(next);
                    }
                }
            }
        }
        Components { parts }
    }

    /// Whether the lambda's environment holds a class of this group, which
    /// an instance of it copies with the lambda.
    pub(crate) fn environment_here(&self, lambda: &Name) -> bool {
        self.environment_in(lambda).is_some_and(|environment| {
            environment
                .into_iter()
                .flatten()
                .any(|class| !self.is_foreign(class))
        })
    }

    /// The identities this state's body made, at calls of a declaration whose
    /// result carries one no parameter does, or anew at the instances it made.
    pub(crate) fn made_identities(&self) -> BTreeSet<Name> {
        self.log
            .iter()
            .filter(|recorded| {
                matches!(
                    recorded.recorder,
                    Recorder::Disjunction(_)
                        | Recorder::Generalization(_)
                        | Recorder::Instance { .. }
                )
            })
            .filter_map(|recorded| match &recorded.relation {
                Relation::Structure(_, Term::Identity(name), _) => Some(name.clone()),
                _ => None,
            })
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
            parts: None,
            disjunctions: None,
            made: BTreeMap::new(),
            recorded: Recording::default(),
        }
    }

    /// An instance of the parts of C the positions a use records here lie in,
    /// the use's other constraints recorded by another group.
    /// `positions` are the classes of `from` at the function's parameters and
    /// result, each with whether the use records it here. The cut groups the
    /// use's positions by these parts, so no part holds positions of both.
    /// Panics where a part of C, or a disjunctive constraint left in it, lies at no position,
    /// which no rule places.
    pub(crate) fn part(
        at: &Name,
        by: Recorder,
        from: &'a State,
        positions: &[(ClassId, bool)],
    ) -> Instance<'a> {
        let components = from.components();
        let parts_at = |held: bool| -> BTreeSet<usize> {
            positions
                .iter()
                .filter(|&&(_, at)| at == held)
                .filter_map(|&(class, _)| components.of(from, class))
                .collect()
        };
        let (here, elsewhere) = (parts_at(true), parts_at(false));
        assert!(
            here.is_disjoint(&elsewhere),
            "{at}: the cut groups a use's positions by the parts of C"
        );
        let placed = |part: Option<usize>| {
            part.is_some_and(|part| here.contains(&part) || elsewhere.contains(&part))
        };
        let c = from.c();
        if let Some(root) = c.iter().find(|&&root| !placed(components.of(from, root))) {
            panic!(
                "{at}: {} lies in a part of C at no position of a use whose constraints two \
                 groups record, which no rule places",
                from.class_name(*root)
            );
        }
        let mut disjunctions = BTreeSet::new();
        for (name, disjunction) in &from.disjunctions {
            if disjunction.decision.is_some() {
                continue;
            }
            let part = components.of_disjunction(from, disjunction);
            if !placed(part) {
                panic!(
                    "{at}: the disjunctive constraint {name} left in C lies at no position of a use whose \
                     constraints two groups record, which no rule places"
                );
            }
            if part.is_some_and(|part| here.contains(&part)) {
                disjunctions.insert(name.clone());
            }
        }
        Instance {
            c: c.into_iter()
                .filter(|&root| {
                    components
                        .of(from, root)
                        .is_some_and(|part| here.contains(&part))
                })
                .collect(),
            parts: Some(here),
            disjunctions: Some(disjunctions),
            ..Instance::new(at, by, from)
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

    /// A disjunctive constraint's kind here, its classes those here and a binding's path the
    /// copies of its projections.
    pub(crate) fn kind(&mut self, here: &mut State, kind: &Disjunction) -> Disjunction {
        match kind.clone() {
            Disjunction::Use {
                callee,
                args,
                result,
                exit,
            } => Disjunction::Use {
                callee,
                args: args
                    .into_iter()
                    .map(|arg| arg.map(|arg| self.class(here, arg)))
                    .collect(),
                result: result.map(|result| self.class(here, result)),
                exit: exit.map(|exit| self.class(here, exit)),
            },
            Disjunction::Lend {
                mutability,
                place,
                result,
            } => Disjunction::Lend {
                mutability,
                place: self.class(here, place),
                result: self.class(here, result),
            },
            Disjunction::Project {
                projection,
                operand,
                result,
            } => Disjunction::Project {
                projection,
                operand: self.class(here, operand),
                result: self.class(here, result),
            },
            Disjunction::Bind {
                path,
                position,
                binding,
            } => Disjunction::Bind {
                path: path.iter().map(|step| self.disjunction(step)).collect(),
                position: self.class(here, position),
                binding: self.class(here, binding),
            },
            Disjunction::CallValue {
                callee,
                args,
                result,
            } => Disjunction::CallValue {
                callee: self.class(here, callee),
                args: args.into_iter().map(|arg| self.class(here, arg)).collect(),
                result: self.class(here, result),
            },
        }
    }

    /// What records a copied relation here: the copy of an undecided disjunctive constraint, or
    /// the instance, made by what made it.
    fn recorder(&self, of: &Recorder) -> Recorder {
        match of {
            Recorder::Disjunction(disjunction)
                if self.from.disjunctions[disjunction].decision.is_none() =>
            {
                Recorder::Disjunction(self.disjunction(disjunction))
            }
            // A generalization recorded where the disjunctive constraint is not, at positions
            // where every candidate writes one closed type, is as
            // that of a disjunctive constraint left undecided.
            Recorder::Generalization(disjunction)
                if self
                    .from
                    .disjunctions
                    .get(disjunction)
                    .is_none_or(|disjunction| disjunction.decision.is_none()) =>
            {
                Recorder::Generalization(self.disjunction(disjunction))
            }
            _ => Recorder::Instance {
                at: (*self.at).clone(),
                by: Box::new(self.by.clone()),
                of: Box::new(of.clone()),
            },
        }
    }

    pub(crate) fn disjunction(&self, name: &Name) -> Name {
        copy_name(&self.at, name)
    }

    /// The copy here of a lambda placed in the group instantiated, its
    /// environment this instance's copies.
    fn lambda(&mut self, here: &mut State, lambda: &Name) -> Name {
        let copy = copy_name(&self.at, lambda);
        if !here.lambdas.contains_key(&copy) {
            let of = self
                .from
                .lambdas
                .get(lambda)
                .map_or(lambda.clone(), |copied| copied.of.clone());
            let environment = self
                .from
                .environment_of(lambda)
                .into_iter()
                .map(|class| class.map(|class| self.class(here, class)))
                .collect();
            let made = crate::solve::LambdaCopy {
                at: (*self.at).clone(),
                of,
                environment,
            };
            here.lambdas.insert(copy.clone(), made);
        }
        copy
    }

    /// The class here of the class `id` of `from`: for a class of C, one made
    /// anew, holding the structure `from` solved it to; for another, its closed
    /// type, written.
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
        if let Some((structure, StructureSource::Solved)) = held.structure {
            let term = structure.map(|_, part| self.class(here, part));
            let recorder = self.recorder(&Recorder::Decomposition(root));
            let relation = Relation::Structure(made, term, StructureSource::Solved);
            self.recorded.push(here.record(relation, recorder));
        }
        made
    }

    /// Records here every relation of `from` on a class of C, makes anew each
    /// disjunctive constraint `from` left undecided, and keeps the instance here.
    /// A relation `from` decomposed is decomposed here too: what its decomposition recorded is
    /// among the relations copied, and no class of the copy takes a
    /// constructor here that it did not hold in `from`.
    /// Returns what the instance recorded here, itself the instance made.
    pub(crate) fn record_rest(&mut self, here: &mut State) -> Recording {
        let from = self.from;
        for at in from.relations_on(&self.c) {
            let recorded = &from.log[at];
            let relation = match recorded.relation.clone() {
                Relation::Structure(class, Term::Identity(name), source) => {
                    Relation::Structure(class, Term::Identity(copy_name(&self.at, &name)), source)
                }
                Relation::Values(class, lambda) if from.environment_here(&lambda) => {
                    Relation::Values(class, self.lambda(here, &lambda))
                }
                relation => relation,
            }
            .map(|class| self.class(here, class));
            let recorder = self.recorder(&recorded.recorder);
            let entry = here.record(relation, recorder);
            if from.decomposed.contains(&at) {
                here.decomposed.insert(entry.at);
                here.copied_decomposed.insert(entry.at);
            }
            self.recorded.push(entry);
        }
        let undecided: Vec<(&Name, &DisjunctionState)> = from
            .disjunctions
            .iter()
            .filter(|(name, disjunction)| {
                disjunction.decision.is_none()
                    && self
                        .disjunctions
                        .as_ref()
                        .is_none_or(|disjunctions| disjunctions.contains(*name))
            })
            .collect();
        for (name, disjunction) in undecided {
            let kind = self.kind(here, &disjunction.kind);
            let copy = self.disjunction(name);
            here.add_disjunction(&copy, kind);
            let generalized =
                disjunction
                    .generalized
                    .clone()
                    .map(|generalized| match generalized {
                        Generalized::Use { part, classes } => Generalized::Use {
                            part,
                            classes: classes
                                .into_iter()
                                .map(|(k, class)| (k, self.class(here, class)))
                                .collect(),
                        },
                        Generalized::Lend => Generalized::Lend,
                    });
            let references: BTreeSet<ClassId> = disjunction
                .references
                .iter()
                .map(|&class| self.class(here, class))
                .collect();
            let state = here
                .disjunctions
                .get_mut(&copy)
                .expect("the disjunctive constraint was added");
            state.generalized = generalized;
            state.references = references;
        }
        here.instances.push(Made {
            at: (*self.at).clone(),
            by: self.by.clone(),
            group: from.group.clone(),
            made: self.made.clone(),
            parts: self.parts.clone(),
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

/// The classes a disjunctive constraint names: those of its structure, and those its generalization
/// made and recorded whether they are references.
fn named_by(disjunction: &DisjunctionState) -> Vec<ClassId> {
    let mut classes: Vec<ClassId> = match &disjunction.kind {
        Disjunction::Use {
            args, result, exit, ..
        } => args
            .iter()
            .flatten()
            .chain(result)
            .chain(exit)
            .copied()
            .collect(),
        Disjunction::Lend { place, result, .. } => vec![*place, *result],
        Disjunction::Project {
            operand, result, ..
        } => vec![*operand, *result],
        Disjunction::Bind {
            position, binding, ..
        } => vec![*position, *binding],
        Disjunction::CallValue {
            callee,
            args,
            result,
        } => [*callee]
            .into_iter()
            .chain(args.iter().copied())
            .chain([*result])
            .collect(),
    };
    if let Some(Generalized::Use { classes: made, .. }) = &disjunction.generalized {
        classes.extend(made.values());
    }
    classes.extend(&disjunction.references);
    classes
}

/// Whether the value carries one of the identities, at any depth.
fn carries_identity(value: &Ty, identities: &BTreeSet<Name>) -> bool {
    match value.term() {
        Term::Identity(name) => identities.contains(name),
        term => term
            .parts()
            .into_iter()
            .any(|(_, part)| carries_identity(part, identities)),
    }
}
