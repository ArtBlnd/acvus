//! What all a disjunctive constraint not yet decided may become shares, recorded before it is
//! decided: a use's
//! candidates' generalization, and a lend's result being a reference.

use std::collections::{BTreeMap, BTreeSet};
use std::rc::Rc;

use crate::class::{ClassId, ClassName};
use crate::decl::{DeclId, Node};
use crate::domain::Domain;
use crate::program::Position;
use crate::record::Recording;
use crate::relation::{Recorder, Relation};
use crate::sig::{SigTy, SigVar, Signature};
use crate::solve::State;
use crate::ty::{Constructor, Mutability, Name, Term};

use super::{Disjunction, DisjunctionAt, Generalized};

/// Whether the type a signature writes is a reference, none where it is a
/// variable.
pub(super) fn is_reference(ty: &SigTy) -> Option<bool> {
    match ty {
        SigTy::Var(_) => None,
        SigTy::Term(term) => Some(matches!(**term, Term::Ref(..))),
    }
}

/// The generalization of a use's candidates: the anti-unification of
/// their signatures. Where they all write one constructor and members it is
/// kept, and its parts are generalized in turn; elsewhere a variable
/// `SigVar::Type(k)` stands, one for each tuple of the types the candidates
/// write there. A reference is not decomposed where a whole value passes, at the
/// top of a parameter or of the result, since a cast goes between two
/// references of one constructor.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Generalization {
    pub params: Vec<SigTy>,
    pub result: SigTy,
    /// For each variable, the type each candidate writes there, in the
    /// candidates' order.
    pub vars: Vec<Vec<SigTy>>,
}

impl Generalization {
    pub fn of(sigs: &[&Signature]) -> Generalization {
        let mut vars = Vec::new();
        let arity = sigs.first().map_or(0, |sig| sig.params().len());
        let params = (0..arity)
            .map(|i| {
                let tys: Vec<&SigTy> = sigs.iter().map(|sig| &sig.params()[i]).collect();
                generalize(&tys, true, &mut vars)
            })
            .collect();
        let results: Vec<&SigTy> = sigs.iter().map(|sig| sig.result()).collect();
        let result = generalize(&results, true, &mut vars);
        Generalization {
            params,
            result,
            vars,
        }
    }

    /// Whether the generalization writes one closed type at the position, every
    /// candidate writing it there, so that the use's constraint there holds
    /// whatever the use becomes. An exit is not.
    pub fn is_closed_at(&self, position: Position) -> bool {
        match position {
            Position::Arg(k) => self.params[k].is_closed(),
            Position::Result => self.result.is_closed(),
            Position::Exit => false,
        }
    }

    /// The variables a type of the generalization writes, each once, in the
    /// order of its parts.
    pub fn vars_in(ty: &SigTy) -> Vec<usize> {
        let mut found = Vec::new();
        collect_vars(ty, &mut found);
        found
    }

    /// Whether every candidate writes a reference at the variable, or none
    /// does; none where some write a variable or they differ.
    pub fn is_reference(&self, k: usize) -> Option<bool> {
        let references = self.vars[k]
            .iter()
            .map(is_reference)
            .collect::<Option<Vec<bool>>>()?;
        if references.iter().all(|&reference| reference) {
            Some(true)
        } else if references.iter().all(|&reference| !reference) {
            Some(false)
        } else {
            None
        }
    }
}

fn collect_vars(ty: &SigTy, found: &mut Vec<usize>) {
    match ty {
        SigTy::Var(SigVar::Type(k)) => {
            if !found.contains(k) {
                found.push(*k);
            }
        }
        SigTy::Var(SigVar::Identity) => {}
        SigTy::Term(term) => {
            for (_, part) in term.parts() {
                collect_vars(part, found);
            }
        }
    }
}

/// The generalization of the types the candidates write at one place, `whole`
/// where a whole value passes there.
fn generalize(tys: &[&SigTy], whole: bool, vars: &mut Vec<Vec<SigTy>>) -> SigTy {
    let first = tys[0];
    if first.is_closed() && tys.iter().all(|ty| *ty == first) {
        return first.clone();
    }
    let terms: Option<Vec<&Term<SigTy>>> = tys
        .iter()
        .map(|ty| match ty {
            SigTy::Term(term) => Some(&**term),
            SigTy::Var(_) => None,
        })
        .collect();
    if let Some(terms) = terms {
        let head = terms[0];
        let alike = terms.iter().all(|term| {
            term.constructor() == head.constructor() && term.members() == head.members()
        });
        let decompose = !(whole && matches!(head, Term::Ref(..)));
        if alike && decompose {
            let term = head.clone().map(|key, _| {
                let parts: Vec<&SigTy> = terms
                    .iter()
                    .map(|term| {
                        term.part(&key).expect(
                            "terms of one constructor and members have one part of each name",
                        )
                    })
                    .collect();
                generalize(&parts, false, vars)
            });
            return SigTy::term(term);
        }
    }
    let tuple: Vec<SigTy> = tys.iter().map(|&ty| ty.clone()).collect();
    let k = match vars.iter().position(|held| *held == tuple) {
        Some(k) => k,
        None => {
            vars.push(tuple);
            vars.len() - 1
        }
    };
    SigTy::Var(SigVar::Type(k))
}

impl State {
    /// Records, for each disjunctive constraint not yet decided, the generalization of all it may
    /// become, once: for a use of several candidates, their generalization's
    /// result entering the use's, each variable it writes made anew as
    /// an unknown, whose domain is a reference or none where every candidate
    /// agrees there; for a lend, that its result is a reference. An argument
    /// passes into no part of it: every candidate takes it by a coercion of
    /// its own, which a type common to them does not imply. Returns the
    /// relations recorded.
    pub(crate) fn record_generalizations(&mut self) -> Recording {
        let mut recorded = Recording::default();
        let pending: Vec<Name> = self
            .disjunctions
            .iter()
            .filter(|(_, disjunction)| {
                disjunction.decision.is_none() && disjunction.generalized.is_none()
            })
            .map(|(name, _)| name.clone())
            .collect();
        for name in pending {
            recorded.append(self.record_generalization(&name));
        }
        recorded
    }

    /// Records the constraints of the use `name`, of several candidates, at
    /// the positions `closed` where its generalization writes one closed type:
    /// each argument there passing into that type, and that
    /// type entering the result, recorded by the generalization whatever the use
    /// becomes. Returns the relations recorded.
    pub(crate) fn record_closed(
        &mut self,
        name: &Name,
        callee: &Name,
        arity: usize,
        closed: &[(Position, ClassId)],
    ) -> Recording {
        let mut recorded = Recording::default();
        let sigs: Vec<Signature> = self
            .candidates(callee, arity)
            .into_iter()
            .map(|decl| decl.sig.clone())
            .collect();
        let part = Generalization::of(&sigs.iter().collect::<Vec<_>>());
        let recorder = Recorder::Generalization(name.clone());
        let owner = Rc::new(name.clone());
        let at = |at: DisjunctionAt| ClassName::Disjunction(owner.clone(), at);
        let mut vars: BTreeMap<SigVar, ClassId> = BTreeMap::new();
        let var_name = |var: SigVar| -> ClassName {
            panic!("a closed type of the generalization writes no variable, as {var:?}")
        };
        for &(position, class) in closed {
            assert!(
                part.is_closed_at(position),
                "{name}: the generalization writes no closed type at {position:?}"
            );
            recorded.append(match position {
                Position::Arg(k) => self.record_argument(
                    class,
                    &part.params[k],
                    &mut vars,
                    at(DisjunctionAt::GeneralizedParam(k)),
                    &var_name,
                    &recorder,
                ),
                Position::Result => self.record_result(
                    class,
                    &part.result,
                    &mut vars,
                    at(DisjunctionAt::GeneralizedResult),
                    &var_name,
                    &recorder,
                ),
                Position::Exit => unreachable!("an exit is not closed"),
            });
        }
        recorded
    }

    fn record_generalization(&mut self, name: &Name) -> Recording {
        let mut recorded = Recording::default();
        let disjunction = self.disjunctions[name].clone();
        let recorder = Recorder::Generalization(name.clone());
        let (generalized, references) = match &disjunction.kind {
            Disjunction::Use {
                callee, args, result, ..
            } => {
                let sigs: Vec<Signature> = self
                    .candidates(callee, args.len())
                    .into_iter()
                    .map(|decl| decl.sig.clone())
                    .collect();
                if sigs.len() < 2 {
                    return recorded;
                }
                let part = Generalization::of(&sigs.iter().collect::<Vec<_>>());
                let owner = Rc::new(name.clone());
                let at = |at: DisjunctionAt| ClassName::Disjunction(owner.clone(), at);
                let mut vars: BTreeMap<SigVar, ClassId> = BTreeMap::new();
                let mut classes = BTreeMap::new();
                let mut references = BTreeSet::new();
                for k in Generalization::vars_in(&part.result) {
                    let class = self.fresh(at(DisjunctionAt::Generalized(k)));
                    vars.insert(SigVar::Type(k), class);
                    classes.insert(k, class);
                    let domain = match part.is_reference(k) {
                        Some(true) => Domain::references(),
                        Some(false) => Domain::no_references(),
                        None => continue,
                    };
                    recorded.push(self.record(Relation::Domain(class, domain), recorder.clone()));
                    references.insert(class);
                }
                let var_name = |var: SigVar| -> ClassName {
                    panic!("the generalization's variable {var:?} is made before its types")
                };
                if let Some(result) = result {
                    recorded.append(self.record_result(
                        *result,
                        &part.result,
                        &mut vars,
                        at(DisjunctionAt::GeneralizedResult),
                        &var_name,
                        &recorder,
                    ));
                }
                let generalized = Generalized::Use {
                    part: Rc::new(part),
                    classes,
                };
                (generalized, references)
            }
            Disjunction::Lend { result, .. } => {
                // Its mutability is still the lend's to record, which a read
                // of whether the result is a reference reads too, so
                // the lend still records into what such a read reads.
                let relation = Relation::Domain(*result, Domain::references());
                recorded.push(self.record(relation, recorder));
                (Generalized::Lend, BTreeSet::new())
            }
            Disjunction::Project { .. }
            | Disjunction::Bind { .. }
            | Disjunction::CallValue { .. } => {
                return recorded;
            }
        };
        let state = self
            .disjunctions
            .get_mut(name)
            .expect("the disjunctive constraint is the state's");
        state.generalized = Some(generalized);
        state.references = references;
        recorded
    }

    /// The index among a use's candidates of the declaration `id`.
    pub(crate) fn candidate_index(&self, callee: &Name, arity: usize, id: &DeclId) -> usize {
        self.candidates(callee, arity)
            .iter()
            .position(|decl| decl.id == *id)
            .expect("a use is decided as one of its candidates")
    }

    /// For each candidate of a use, the classes it would record a lower bound
    /// or an equation into, each with whether that type is a reference, none
    /// where it is an unknown: its result, and the classes inside its
    /// arguments' structures that its parameters reach.
    pub(super) fn would_record_per_candidate(
        &self,
        name: &Name,
    ) -> Vec<BTreeMap<ClassId, Option<bool>>> {
        let Disjunction::Use { callee, args, .. } = &self.disjunctions[name].kind else {
            return Vec::new();
        };
        (0..self.candidates(callee, args.len()).len())
            .map(|candidate| self.would_record_by(name, candidate))
            .collect()
    }

    /// The classes the use's candidate at index `candidate` would record a
    /// lower bound or an equation into once decided, as
    /// `would_record_per_candidate` gives them: inside each argument, by the
    /// decompositions of its passing; the result, or, once the generalization is
    /// recorded, each of its variables the result writes, into which the
    /// candidate's type there enters, and inside it by the decomposition; and for
    /// `e?`, the class its exit's value enters, an `Option` or a `Result`.
    pub(crate) fn would_record_by(
        &self,
        name: &Name,
        candidate: usize,
    ) -> BTreeMap<ClassId, Option<bool>> {
        let disjunction = &self.disjunctions[name];
        let Disjunction::Use {
            callee,
            args,
            result,
            exit,
        } = &disjunction.kind
        else {
            return BTreeMap::new();
        };
        let sig = &self.candidates(callee, args.len())[candidate].sig;
        let generalized = match &disjunction.generalized {
            Some(Generalized::Use { part, classes }) => Some((part, classes)),
            _ => None,
        };
        let mut found = BTreeMap::new();
        for (arg, param) in args.iter().zip(sig.params()) {
            let Some(arg) = arg else {
                continue;
            };
            self.would_record(*arg, param, &mut found);
        }
        match generalized {
            None => {
                if let Some(result) = result {
                    found.insert(self.root(*result), is_reference(sig.result()));
                    self.would_record_into_result(*result, sig.result(), &mut found);
                }
            }
            Some((part, classes)) => {
                for k in Generalization::vars_in(&part.result) {
                    let chosen = &part.vars[k][candidate];
                    found.insert(self.root(classes[&k]), is_reference(chosen));
                    self.would_record_into_result(classes[&k], chosen, &mut found);
                }
            }
        }
        if let Some(exit) = exit {
            found.insert(self.root(*exit), Some(false));
        }
        found
    }

    /// The classes inside the structure of a class a use's result type `ty` enters
    /// that the decomposition of that lower bound would record into: each part where
    /// `ty` writes a part, a type or a variable, which a part lower bound or
    /// an equation relates, and inside it in turn where `ty` writes a type.
    /// Where the two constructors differ, the casts' results relate every part
    /// of the class, each a type parameter of the last cast's result.
    fn would_record_into_result(
        &self,
        class: ClassId,
        ty: &SigTy,
        found: &mut BTreeMap<ClassId, Option<bool>>,
    ) {
        let SigTy::Term(term) = ty else {
            return;
        };
        let Some((structure, _)) = &self.classes.held(self.root(class)).structure else {
            return;
        };
        if structure.constructor() != term.constructor() {
            for (_, &part) in structure.parts() {
                found.insert(self.root(part), None);
            }
            return;
        }
        for (key, written) in term.parts() {
            if let Some(&part) = structure.part(&key) {
                found.insert(self.root(part), is_reference(written));
                self.would_record_into_result(part, written, found);
            }
        }
    }

    /// The classes inside an argument's structure that a candidate's parameter
    /// would record a lower bound or an equation into, by the decompositions. The
    /// argument passes into the type the parameter writes, which
    /// records into that type alone, and each of its parts passes into that
    /// type's part: a lower bound where the parameter writes a variable, an
    /// unknown, recording an equation into the part, unless it is a
    /// structural type's, whose part lower bound records into the variable
    /// alone; and where the parameter writes a type, the part passes into it
    /// in turn. Two constructors are linked as a decomposition links them: the targets
    /// of a reborrow, and the parts of the first declared cast's parameter,
    /// into which the argument passes; the later casts' parameters and the
    /// parameter's type relate the casts' results, which are not the
    /// argument's. Where the parameter writes a function type and the argument
    /// holds a function value, the decomposition makes an instance of each lambda its
    /// Values names, each parameter of the function type entering that
    /// lambda's parameter in its environment.
    pub(super) fn would_record(
        &self,
        class: ClassId,
        ty: &SigTy,
        found: &mut BTreeMap<ClassId, Option<bool>>,
    ) {
        let SigTy::Term(term) = ty else {
            return;
        };
        let Some((structure, _)) = &self.classes.held(class).structure else {
            return;
        };
        if let (Term::Function { params, .. }, Term::FunctionValue { .. }) = (&**term, structure) {
            for lambda in self.values_of(class) {
                let mut held = self.environment_of(&lambda);
                held.pop();
                for (param, held) in params.iter().zip(held) {
                    if let Some(env) = held {
                        found.insert(self.root(env), is_reference(param));
                    }
                }
            }
            return;
        }
        let (from, to) = (structure.constructor(), term.constructor());
        if from == to {
            for (key, written) in term.parts() {
                if let Some(&part) = structure.part(&key) {
                    self.would_record_part(&to, part, written, found);
                }
            }
            return;
        }
        if let (Term::Ref(Mutability::Exclusive, target), Term::Ref(Mutability::Shared, written)) =
            (structure, &**term)
        {
            self.would_record_part(&to, *target, written, found);
            return;
        }
        let path = self.decls.tree().path(&Node::Owned(from), &Node::Owned(to));
        if let Some(cast) = path.as_deref().and_then(<[_]>::first) {
            let param = &self.decls.decl(cast).sig.params()[0];
            self.would_record(class, param, found);
        }
    }

    /// As `would_record`, at the part `part` of an argument passing into the
    /// part `written` of a type of constructor `of`.
    fn would_record_part(
        &self,
        of: &Constructor,
        part: ClassId,
        written: &SigTy,
        found: &mut BTreeMap<ClassId, Option<bool>>,
    ) {
        match written {
            SigTy::Var(_) if !of.is_structural() => {
                found.insert(self.root(part), is_reference(written));
            }
            SigTy::Var(_) => {}
            SigTy::Term(_) => self.would_record(part, written, found),
        }
    }
}
