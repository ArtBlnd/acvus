//! The generalization of a use's candidates: the anti-unification of their
//! signatures, by which the positions where every candidate writes one
//! closed type are recorded apart from the use's disjunctive constraint, and
//! which the judgment reads at the positions where they differ.

use std::collections::BTreeMap;

use crate::class::ClassId;
use crate::program::Position;
use crate::record::{Making, Recording};
use crate::relation::{Recorder, Relation};
use crate::sig::{SigTy, SigVar, Signature};
use crate::solve::State;
use crate::ty::{Name, Term};

use super::{Disjunction, DisjunctionAt};
use crate::domain::Domain;

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
            .map(|ty| match ty {
                SigTy::Var(_) => None,
                SigTy::Term(term) => Some(matches!(**term, Term::Ref(..))),
            })
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
    /// Records, as the disjunctive constraint `name` is recorded, what every
    /// alternative of it shares, which holds whatever it becomes: for a use
    /// of several candidates, their generalization's result entering the
    /// use's result, each variable it writes a class made anew, with a domain
    /// of references or of none where every candidate agrees there; for a
    /// lend, that its result is a reference. An argument passes into no part
    /// of it: every candidate takes it by a coercion of its own, which a type
    /// common to them does not imply. Returns the relations recorded.
    pub(crate) fn record_shared(&mut self, name: &Name) -> Recording {
        let mut recorded = Recording::default();
        let (kind, _) = self.disjunction(name);
        let kind = kind.clone();
        let recorder = Recorder::Generalization(name.clone());
        match &kind {
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
                let owner = self.owner(name);
                let mut vars: BTreeMap<SigVar, ClassId> = BTreeMap::new();
                for k in Generalization::vars_in(&part.result) {
                    let class = self.make(owner.made(DisjunctionAt::Generalized(k)));
                    vars.insert(SigVar::Type(k), class);
                    let domain = match part.is_reference(k) {
                        Some(true) => Domain::references(),
                        Some(false) => Domain::no_references(),
                        None => continue,
                    };
                    recorded.push(self.record(Relation::Domain(class, domain), recorder.clone()));
                }
                let var = |var: SigVar| -> Making {
                    panic!("the generalization's variable {var:?} is made before its types")
                };
                if let Some(result) = result {
                    recorded.append(self.record_result(
                        *result,
                        &part.result,
                        &mut vars,
                        owner.made(DisjunctionAt::GeneralizedResult),
                        &var,
                        &recorder,
                    ));
                }
            }
            Disjunction::Lend { result, .. } => {
                let relation = Relation::Domain(*result, Domain::references());
                recorded.push(self.record(relation, recorder));
            }
            Disjunction::Project { .. }
            | Disjunction::Bind { .. }
            | Disjunction::CallValue { .. } => {}
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
        let owner = self.owner(name);
        let mut vars: BTreeMap<SigVar, ClassId> = BTreeMap::new();
        let var = |var: SigVar| -> Making {
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
                    owner.made(DisjunctionAt::GeneralizedParam(k)),
                    &var,
                    &recorder,
                ),
                Position::Result => self.record_result(
                    class,
                    &part.result,
                    &mut vars,
                    owner.made(DisjunctionAt::GeneralizedResult),
                    &var,
                    &recorder,
                ),
                Position::Exit => unreachable!("an exit is not closed"),
            });
        }
        recorded
    }
}
