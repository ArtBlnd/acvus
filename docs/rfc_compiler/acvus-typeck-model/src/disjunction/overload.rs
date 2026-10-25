//! Overload resolution: a use's candidates, the positions where
//! they differ, the narrowing, and what a decided candidate's signature
//! records.

use std::collections::{BTreeMap, BTreeSet};

use crate::class::{ClassId, ClassName};
use crate::coerce::{Rank, coerce_read};
use crate::decl::{Decl, DeclId};
use crate::read::ReadValue;
use crate::record::Recording;
use crate::relation::{Recorder, Relation};
use crate::sig::{SigTy, Signature};
use crate::solve::State;
use crate::ty::{HasTerm, Name, Term};

use super::Decision;

/// The candidates a use's read leaves at one position where they differ.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Narrowed {
    /// The argument's index.
    pub position: usize,
    pub value: ReadValue,
    /// Each candidate left before it, with the rank of the coercion its
    /// passing goes by there, none where no coercion reaches it.
    pub ranks: Vec<(DeclId, Option<Rank>)>,
    /// Those of the lowest rank, which remain.
    pub remaining: Vec<DeclId>,
}

/// How a use's candidates are narrowed by the value it reads.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Narrowing {
    pub candidates: Vec<DeclId>,
    /// The class at each argument where the candidates differ, by index, and
    /// the value read there.
    pub values: Vec<(usize, ClassId, ReadValue)>,
    /// The narrowing at each of those arguments in turn, none where an
    /// argument reads as `!`.
    pub stages: Vec<Narrowed>,
    /// The candidate it is decided as, or `!` where an argument reads as `!`;
    /// none where not exactly one remains.
    pub decision: Option<Decision>,
}

impl State {
    /// The candidates of a use: the declarations its name reaches whose
    /// parameters are as many as its arguments.
    pub(crate) fn candidates(&self, callee: &Name, arity: usize) -> Vec<&Decl> {
        self.decls
            .candidates(callee)
            .iter()
            .filter(|decl| decl.sig.params().len() == arity)
            .collect()
    }

    /// The class a use's disjunctive constraint holds at the argument `i` where its candidates
    /// differ: an argument where they differ is no closed type of their
    /// generalization, so the disjunctive constraint holds it.
    pub(crate) fn differing_class(args: &[Option<ClassId>], i: usize) -> ClassId {
        args[i].expect(
            "a use's disjunctive constraint holds the arguments where its candidates differ",
        )
    }

    /// The arguments where a use's candidates differ, which it reads, each
    /// candidate's type there taken as the narrowing takes it: a type
    /// parameter is an unknown made anew at the use, which decides
    /// nothing, so two candidates that differ there only in their type
    /// parameters reach any value read there alike. None where it has
    /// no candidate.
    pub(crate) fn differing(candidates: &[&Signature], args: usize) -> Vec<usize> {
        let Some(first) = candidates.first() else {
            return Vec::new();
        };
        (0..args)
            .filter(|&i| {
                let at = ReadValue::of_sig(&first.params()[i]);
                candidates
                    .iter()
                    .any(|sig| ReadValue::of_sig(&sig.params()[i]) != at)
            })
            .collect()
    }

    /// An argument passing into its parameter: a lower bound where the parameter
    /// is an unknown, a coercion constraint where a declaration writes it.
    /// Returns the relations recorded.
    pub(super) fn record_argument<F>(
        &mut self,
        arg: ClassId,
        param: &SigTy,
        vars: &mut BTreeMap<crate::sig::SigVar, ClassId>,
        name: ClassName,
        var_name: &F,
        recorder: &Recorder,
    ) -> Recording
    where
        F: Fn(crate::sig::SigVar) -> ClassName,
    {
        let mut recorded = Recording::default();
        let param = self.instantiate(param, vars, name, var_name, recorder, &mut recorded);
        let relation = if self.is_unknown(param) {
            Relation::LowerBound {
                lower: arg,
                upper: param,
            }
        } else {
            Relation::CoercionConstraint {
                from: arg,
                to: param,
            }
        };
        recorded.push(self.record(relation, recorder.clone()));
        recorded
    }

    /// The signature's result type entering the use's result. Returns the relations recorded.
    pub(super) fn record_result<F>(
        &mut self,
        result: ClassId,
        ty: &SigTy,
        vars: &mut BTreeMap<crate::sig::SigVar, ClassId>,
        name: ClassName,
        var_name: &F,
        recorder: &Recorder,
    ) -> Recording
    where
        F: Fn(crate::sig::SigVar) -> ClassName,
    {
        let mut recorded = Recording::default();
        let returned = self.instantiate(ty, vars, name, var_name, recorder, &mut recorded);
        let relation = Relation::LowerBound {
            lower: returned,
            upper: result,
        };
        recorded.push(self.record(relation, recorder.clone()));
        recorded
    }

    /// Narrows the candidates by the value the use reads at each argument
    /// where they differ, in order.
    pub(super) fn narrow(
        &self,
        callee: &Name,
        args: &[Option<ClassId>],
        excluded: &BTreeSet<Name>,
    ) -> Narrowing {
        let candidates = self.candidates(callee, args.len());
        let ids: Vec<DeclId> = candidates.iter().map(|decl| decl.id.clone()).collect();
        let sigs: Vec<&Signature> = candidates.iter().map(|decl| &decl.sig).collect();
        let differing = Self::differing(&sigs, args.len());
        let read = self.read_state(excluded);
        let read_at: Vec<(usize, ClassId, ReadValue)> = differing
            .iter()
            .map(|&i| {
                let class = Self::differing_class(args, i);
                (i, class, read.read_value(class))
            })
            .collect();
        let narrowing = |stages, decision| Narrowing {
            candidates: ids.clone(),
            values: read_at.clone(),
            stages,
            decision,
        };
        let values: Vec<(usize, ReadValue)> = read_at
            .iter()
            .map(|(i, _, value)| (*i, value.clone()))
            .collect();
        let never = values
            .iter()
            .any(|(_, value)| matches!(value.decided(), Some(Term::Never)));
        if never {
            return narrowing(Vec::new(), Some(Decision::Never));
        }
        let narrowed = |values: &[(usize, ReadValue)]| {
            let mut remaining: Vec<usize> = (0..sigs.len()).collect();
            let mut stages = Vec::new();
            for (i, value) in values {
                let ranks: Vec<(usize, Option<Rank>)> = remaining
                    .iter()
                    .map(|&c| {
                        let written = ReadValue::of_sig(&sigs[c].params()[*i]);
                        let rank = coerce_read(&self.decls, value, &written);
                        (c, rank.ok().and_then(|coercion| coercion.rank()))
                    })
                    .collect();
                let lowest = ranks.iter().filter_map(|(_, rank)| rank.clone()).min();
                remaining = ranks
                    .iter()
                    .filter(|(_, rank)| rank.is_some() && *rank == lowest)
                    .map(|(c, _)| *c)
                    .collect();
                stages.push(Narrowed {
                    position: *i,
                    value: value.clone(),
                    ranks: ranks
                        .into_iter()
                        .map(|(c, rank)| (ids[c].clone(), rank))
                        .collect(),
                    remaining: remaining.iter().map(|&c| ids[c].clone()).collect(),
                });
            }
            (remaining, stages)
        };
        let (remaining, stages) = narrowed(&values);
        let decision = match remaining.as_slice() {
            [one] => Some(Decision::Use(ids[*one].clone())),
            _ => None,
        };
        narrowing(stages, decision)
    }
}
