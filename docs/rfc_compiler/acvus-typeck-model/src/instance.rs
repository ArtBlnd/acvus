//! Instances: a binding group's C made anew at a use, and recorded in the use's
//! binding group, and a lambda's C made anew where a function value is called
//! or passes into a function type.

use std::collections::{BTreeMap, BTreeSet};
use std::rc::Rc;

use crate::class::{ClassId, ClassName, Side};
use crate::relation::{Role, Standing};
use crate::disjunction::Disjunction;
use crate::record::{Instantiated, Making, Recording};
use crate::relation::{Item, Recorder, Relation, copy_name};
use crate::solve::State;
use crate::ty::{Name, Term, Ty};

/// An instance a state made, as the record gives it: of the C of the
/// binding group `group`, named `at` and made by `by`, with each root of
/// that group's state it copied and that root's class in the state that
/// made it, read from the instance item's copies.
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

/// A lambda an instance made anew, as the record gives it: the instance,
/// the lambda of the program it copies, and the classes here its
/// environment holds, its parameter types and then its result type, none
/// where its own binding group holds them.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct LambdaCopy {
    pub at: Name,
    pub of: Name,
    pub environment: Vec<Option<ClassId>>,
}

impl State {
    /// Every instance this state made, in the order made.
    pub(crate) fn instances(&self) -> Vec<Made> {
        self.classes
            .items()
            .iter()
            .enumerate()
            .filter_map(|(at, recorded)| match &recorded.item {
                Item::Instance {
                    at: name,
                    by,
                    group,
                    parts,
                } => Some(Made {
                    at: name.clone(),
                    by: by.clone(),
                    group: group.clone(),
                    made: self.copies(at, group),
                    parts: parts.clone(),
                }),
                _ => None,
            })
            .collect()
    }

    /// The instance named `at` this state made.
    pub(crate) fn instance(&self, at: &Name) -> Option<Made> {
        self.instances().into_iter().find(|made| made.at == *at)
    }

    /// Each root of the group `group`'s state the instance item at `item`
    /// copied, with its class here.
    fn copies(&self, item: usize, group: &Name) -> BTreeMap<ClassId, ClassId> {
        let from = self
            .solved
            .get(group)
            .expect("the group an instance copies is solved before it");
        self.classes
            .roles_of(item)
            .into_iter()
            .filter_map(|(class, role)| match role {
                Role::Copy(name) => {
                    let original = from
                        .arena
                        .get(&name)
                        .expect("a copy names a class of the group instantiated");
                    Some((from.root(original), class))
                }
                _ => None,
            })
            .collect()
    }

    /// The lambda copies this state holds, by the copy's name.
    pub(crate) fn lambda_copies(&self) -> BTreeMap<Name, LambdaCopy> {
        self.classes
            .items()
            .iter()
            .enumerate()
            .filter_map(|(item, recorded)| match &recorded.item {
                Item::Lambda { name, of, at } => {
                    let roles = self.classes.roles_of(item);
                    let places = roles
                        .iter()
                        .filter_map(|(_, role)| match role {
                            Role::Environment(i) => Some(*i),
                            _ => None,
                        })
                        .max()
                        .map_or(0, |i| i + 1);
                    let environment = (0..places)
                        .map(|i| {
                            roles.iter().find_map(|(class, role)| {
                                (*role == Role::Environment(i)).then_some(*class)
                            })
                        })
                        .collect();
                    Some((
                        name.clone(),
                        LambdaCopy {
                            at: at.clone(),
                            of: of.clone(),
                            environment,
                        },
                    ))
                }
                _ => None,
            })
            .collect()
    }

    /// The lambda copy of this name, where the record holds one.
    pub(crate) fn lambda_copy(&self, name: &Name) -> Option<LambdaCopy> {
        self.lambda_copies().remove(name)
    }
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
    pub(crate) fn of_disjunction(&self, state: &State, kind: &Disjunction) -> Option<usize> {
        kind.classes()
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
    /// The instance's item in the record it records in, once recorded.
    item: Option<usize>,
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
            let item = &self.classes.items()[at].item;
            let mut classes: Vec<ClassId> = self
                .classes
                .roles_of(at)
                .into_iter()
                .map(|(class, _)| class)
                .collect();
            match item {
                Item::Relation(Relation::Values(_, lambda)) if self.environment_here(lambda) => {
                    classes.extend(self.environment_of(lambda).into_iter().flatten());
                }
                Item::Disjunction {
                    kind: Disjunction::Bind { path, .. },
                    ..
                } => {
                    for step in path {
                        let (kind, _) = self.disjunction(step);
                        classes.extend(kind.classes());
                    }
                }
                _ => {}
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
        self.classes
            .items()
            .iter()
            .filter(|recorded| {
                matches!(
                    recorded.recorder,
                    Recorder::Disjunction(_)
                        | Recorder::Generalization(_)
                        | Recorder::Instance { .. }
                )
            })
            .filter_map(|recorded| match recorded.relation() {
                Some(Relation::Structure(_, Term::Identity(name), _)) => Some(name.clone()),
                _ => None,
            })
            .collect()
    }

    /// The items of the record on a class of `c` that an instance copies, by
    /// their places: every relation but an equation, which is kept by the
    /// classes it unified, and every disjunctive constraint left undecided,
    /// a decided one's relations being its own.
    pub(crate) fn relations_on(&self, c: &BTreeSet<ClassId>) -> Vec<usize> {
        self.classes
            .items()
            .iter()
            .enumerate()
            .filter(|(at, recorded)| match &recorded.item {
                Item::Relation(Relation::Equation(..) | Relation::Standing(..)) => false,
                Item::Disjunction { decision, .. } if decision.is_some() => false,
                Item::Instance { .. } | Item::Lambda { .. } => false,
                _ => self
                    .classes
                    .roles_of(*at)
                    .into_iter()
                    .any(|(class, _)| c.contains(&self.root(class))),
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
            item: None,
            recorded: Recording::default(),
        }
    }

    /// The instance's item in `here`, recorded where it is not yet.
    fn item(&mut self, here: &mut State) -> usize {
        if let Some(item) = self.item {
            return item;
        }
        let item = Item::Instance {
            at: (*self.at).clone(),
            by: self.by.clone(),
            group: self.from.group.clone(),
            parts: self.parts.clone(),
        };
        let at = here.classes.record(item, self.by.clone());
        self.item = Some(at);
        at
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
        for (_, name, kind, decision) in from.classes.disjunctions() {
            if decision.is_some() {
                continue;
            }
            let part = components.of_disjunction(from, kind);
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

    /// The instance `made`, recorded in `here`, resumed, to map more of
    /// `from` the way it did.
    pub(crate) fn resume(here: &State, made: &Made, from: &'a State) -> Instance<'a> {
        let item = here
            .classes
            .items()
            .iter()
            .position(|recorded| matches!(&recorded.item, Item::Instance { at, .. } if *at == made.at))
            .expect("an instance resumed is recorded");
        Instance {
            item: Some(item),
            parts: made.parts.clone(),
            ..Instance::new(&made.at, made.by.clone(), from)
        }
    }

    /// A disjunctive constraint's kind here, its classes those here and a binding's path the
    /// copies of its projections.
    pub(crate) fn kind(&mut self, here: &mut State, kind: &Disjunction) -> Disjunction {
        let at = Rc::clone(&self.at);
        kind.clone()
            .map(|class| self.class(here, class), |step| copy_name(&at, step))
    }

    /// What records a copied relation here: the copy of an undecided disjunctive constraint's
    /// generalization, or the instance, made by what made it.
    fn recorder(&self, of: &Recorder) -> Recorder {
        match of {
            // A generalization recorded where the disjunctive constraint is not, at positions
            // where every candidate writes one closed type, is as
            // that of a disjunctive constraint left undecided.
            Recorder::Generalization(disjunction)
                if self
                    .from
                    .classes
                    .disjunction_at(disjunction)
                    .is_none_or(|at| {
                        self.from.classes.items()[at]
                            .disjunction()
                            .is_some_and(|(_, _, decision)| decision.is_none())
                    }) =>
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
        if here.lambda_copy(&copy).is_none() {
            let of = self
                .from
                .lambda_copy(lambda)
                .map_or(lambda.clone(), |copied| copied.of);
            let environment: Vec<Option<ClassId>> = self
                .from
                .environment_of(lambda)
                .into_iter()
                .map(|class| class.map(|class| self.class(here, class)))
                .collect();
            let item = Item::Lambda {
                name: copy.clone(),
                of,
                at: (*self.at).clone(),
            };
            let at = here.classes.record(item, self.by.clone());
            for (i, class) in environment.into_iter().enumerate() {
                if let Some(class) = class {
                    here.classes.attach(at, class, Role::Environment(i));
                }
            }
        }
        copy
    }

    /// The class here of the class `id` of `from`: for a class of C, one made
    /// anew, which the relations copied with C give what `from` gave it; for
    /// another, its closed type, written.
    pub(crate) fn class(&mut self, here: &mut State, id: ClassId) -> ClassId {
        let root = self.from.root(id);
        let of = self.from.class_name(root);
        if let Some(item) = self.item
            && let Some(made) = here.classes.in_role(item, &Role::Copy(of.clone()))
        {
            return made;
        }
        let shared = self
            .from
            .classes
            .held(root)
            .members
            .iter()
            .find(|&&member| self.from.classes.standing(member).is_some_and(|standing| {
                matches!(standing, Standing::Shared { .. })
            }))
            .copied();
        if let Some(shared) = shared {
            let ClassName::Text(name) = self.from.arena.name(shared).clone() else {
                unreachable!("a class of an enclosing group is the text's");
            };
            return match here.arena.get(&ClassName::Text(name.clone())) {
                Some(made) => made,
                None => {
                    let unknown = self.from.is_enclosing_unknown(shared);
                    let entry = here.share(&name, unknown);
                    self.recorded.push(entry);
                    here.text(&name)
                }
            };
        }
        let item = self.item(here);
        let making = Making {
            item,
            role: Role::Copy(of.clone()),
            label: ClassName::Copy {
                at: Rc::clone(&self.at),
                of: Rc::new(of),
            },
        };
        let made = if self.c.contains(&root) {
            here.make(making)
        } else {
            let ty: Ty = self
                .from
                .value(root)
                .expect("a class outside C has a closed value");
            let recorder = self.recorder(&Recorder::Origin);
            here.closed(making, &ty, &recorder, &mut self.recorded)
        };
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
            let recorded = &from.classes.items()[at];
            let recorder = self.recorder(&recorded.recorder);
            let entry = match &recorded.item {
                Item::Relation(relation) => {
                    let relation = match relation.clone() {
                        Relation::Structure(class, Term::Identity(name), source) => {
                            Relation::Structure(
                                class,
                                Term::Identity(copy_name(&self.at, &name)),
                                source,
                            )
                        }
                        Relation::Values(class, lambda) if from.environment_here(&lambda) => {
                            Relation::Values(class, self.lambda(here, &lambda))
                        }
                        relation => relation,
                    }
                    .map(|class| self.class(here, class));
                    here.record(relation, recorder)
                }
                Item::Disjunction { name, kind, .. } => {
                    if self
                        .disjunctions
                        .as_ref()
                        .is_some_and(|disjunctions| !disjunctions.contains(name))
                    {
                        continue;
                    }
                    let kind = self.kind(here, kind);
                    let copy = self.disjunction(name);
                    let entry = here.add_disjunction(&copy, kind, recorder);
                    // The classes the original made anew in each role are
                    // the copy's, in the same roles.
                    for (class, role) in from.classes.roles_of(at) {
                        if let Role::Made(made) = role {
                            let class = self.class(here, class);
                            here.classes.attach(entry.at, class, Role::Made(made));
                        }
                    }
                    entry
                }
                Item::Instance { .. } | Item::Lambda { .. } => continue,
            };
            if recorded.decomposed {
                here.classes.mark_decomposed(entry.at);
            }
            self.recorded.push(entry);
        }
        self.item(here);
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

impl State {
    /// An instance of the lambda's C made at `at` by `by`, each value
    /// of `args` passing into the copy of the lambda's parameter in its generic
    /// part, or into its parameter in the environment, and the copy of its
    /// result type in its generic part, or the environment's, passing into
    /// `result`.
    pub(crate) fn instantiate_lambda(
        &mut self,
        at: &Name,
        by: &Recorder,
        lambda: &Name,
        args: &[ClassId],
        result: ClassId,
    ) -> Recording {
        let function = self.lambda_function(lambda).clone();
        let from = Rc::clone(
            self.solved
                .get(&function.group)
                .expect("a lambda's group is solved first"),
        );
        let mut copy = Instance::new(at, by.clone(), &from);
        let mut found = Vec::new();
        let held = self.environment_of(lambda);
        for (name, held) in function.params.iter().chain([&function.result]).zip(held) {
            let class = match held {
                Some(class) => class,
                None => copy.class(self, from.text(name)),
            };
            found.push(class);
        }
        let returned = found.pop().expect("the result type follows the parameters");
        let mut recorded = copy.record_rest(self);
        let pairs: Vec<(ClassId, ClassId)> = args
            .iter()
            .copied()
            .zip(found)
            .chain([(returned, result)])
            .collect();
        recorded.append(self.record_call(&pairs, by));
        recorded
    }

    /// The parameter types of the lambda, and then its result type, each the
    /// class here where its environment holds it, none where its binding group does:
    /// for a lambda an instance
    /// made anew, that instance's copies.
    pub(crate) fn environment_of(&self, lambda: &Name) -> Vec<Option<ClassId>> {
        self.environment_in(lambda).unwrap_or_else(|| {
            panic!(
                "{lambda}'s value reaches a group its environment is not in, \
                 which is not yet encoded"
            )
        })
    }

    /// As `environment_of`, none where its environment is not here.
    pub(crate) fn environment_in(&self, lambda: &Name) -> Option<Vec<Option<ClassId>>> {
        if let Some(copy) = self.lambda_copy(lambda) {
            return Some(copy.environment);
        }
        let function = self.lambda_function(lambda);
        function
            .params
            .iter()
            .chain([&function.result])
            .map(|name| {
                if self.environment.contains(name) {
                    self.arena.get(&ClassName::Text(name.clone())).map(Some)
                } else {
                    Some(None)
                }
            })
            .collect()
    }

    /// The lambda of the program a lambda here is, or copies.
    pub(crate) fn lambda_function(&self, lambda: &Name) -> &crate::program::Function {
        let of = self.lambda_copy(lambda).map_or(lambda.clone(), |copy| copy.of);
        self.functions
            .iter()
            .find(|function| function.name == of)
            .expect("a lambda is a function")
    }

    /// The lambdas the Values of the class's function value names: those of
    /// the function values that reach it, by equations or lower bounds.
    pub(crate) fn values_of(&self, id: ClassId) -> BTreeSet<Name> {
        let mut seen = BTreeSet::from([self.root(id)]);
        let mut work = vec![self.root(id)];
        let mut values = BTreeSet::new();
        while let Some(at) = work.pop() {
            values.extend(self.classes.values(at));
            for occurrence in self.classes.occurrences(at) {
                if occurrence.side() != Some(Side::Upper) {
                    continue;
                }
                let Some(
                    Relation::LowerBound { lower, .. } | Relation::PartLowerBound { lower, .. },
                ) = self.classes.items()[occurrence.at].relation()
                else {
                    continue;
                };
                let lower = *lower;
                let lower = self.root(lower);
                if seen.insert(lower) {
                    work.push(lower);
                }
            }
        }
        values
    }
}
