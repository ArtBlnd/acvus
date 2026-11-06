//! The generalization of a use's candidates: the anti-unification of their
//! signatures, recorded at every position of the use as the use is
//! recorded, whatever candidate it comes to; and what every alternative of
//! a lend shares, recorded as the lend is, whatever it is decided as.

use std::collections::BTreeMap;

use crate::class::ClassId;
use crate::record::{Making, Recording};
use crate::relation::{Recorder, Relation};
use crate::sig::{SigTy, SigVar, Signature};
use crate::solve::State;
use crate::ty::{Name, Term};

use super::{Disjunction, DisjunctionAt};
use crate::domain::{Domain, GeneralizedDomain};

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

    /// Whether a parameter of the generalization writes the variable `k`.
    pub fn writes_in_params(&self, k: usize) -> bool {
        self.params
            .iter()
            .any(|param| Generalization::vars_in(param).contains(&k))
    }

    /// Whether the result of the generalization writes the variable `k`.
    pub fn writes_in_result(&self, k: usize) -> bool {
        Generalization::vars_in(&self.result).contains(&k)
    }

    /// The variables a type of the generalization writes, each once, in the
    /// order of its parts.
    pub fn vars_in(ty: &SigTy) -> Vec<usize> {
        let mut found = Vec::new();
        collect_vars(ty, &mut found);
        found
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
    /// Records, as the lend `name` is recorded, what every alternative of it
    /// shares, which holds whatever it is decided as (D1): its result is a
    /// reference, `Domain::references()`. Returns the relations recorded;
    /// none for a disjunctive constraint other than a lend. It decides
    /// nothing and gives the result no structure.
    pub(crate) fn record_shared(&mut self, name: &Name) -> Recording {
        let mut recorded = Recording::default();
        let (kind, _) = self.disjunction(name);
        let kind = kind.clone();
        let recorder = Recorder::Generalization(name.clone());
        match &kind {
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

    /// Records, as the use `name` of several candidates is recorded, their
    /// generalization at every position the use holds, whatever candidate
    /// it comes to: each variable of the generalization a class made anew,
    /// with the domain the use's choice gives it (`GeneralizedDomain`), each
    /// argument passing into the generalization's parameter type, and its
    /// result type entering the use's result. Returns the relations
    /// recorded.
    pub(crate) fn record_generalization(&mut self, name: &Name) -> Recording {
        let mut recorded = Recording::default();
        let (of, choice) = self.choice(name);
        let of = of.clone();
        let sigs: Vec<Signature> = self.signatures(choice);
        let part = Generalization::of(&sigs.iter().collect::<Vec<_>>());
        let recorder = Recorder::Generalization(name.clone());
        let owner = self.use_owner(name);
        let mut vars: BTreeMap<SigVar, ClassId> = BTreeMap::new();
        for (k, written) in part.vars.iter().enumerate() {
            let class = self.make(owner.made(DisjunctionAt::Generalized(k)));
            vars.insert(SigVar::Type(k), class);
            let domain = GeneralizedDomain {
                choice: name.clone(),
                var: k,
                constructors: written
                    .iter()
                    .map(|ty| match ty {
                        SigTy::Term(term) => Some(term.constructor()),
                        SigTy::Var(_) => None,
                    })
                    .collect(),
            };
            let relation = Relation::GeneralizedDomain(class, domain);
            recorded.push(self.record(relation, recorder.clone()));
        }
        let var = |var: SigVar| -> Making {
            panic!("the generalization's variable {var:?} is made before its types")
        };
        for (k, (arg, param)) in of.args.iter().zip(&part.params).enumerate() {
            let Some(arg) = arg else {
                continue;
            };
            recorded.append(self.record_argument(
                *arg,
                param,
                &mut vars,
                owner.made(DisjunctionAt::GeneralizedParam(k)),
                &var,
                &recorder,
            ));
        }
        if let Some(result) = of.result {
            recorded.append(self.record_result(
                result,
                &part.result,
                &mut vars,
                owner.made(DisjunctionAt::GeneralizedResult),
                &var,
                &recorder,
            ));
        }
        recorded
    }
}
