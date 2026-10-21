//! Overload resolution (l.329–344): a use's candidates, the positions where
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

/// The candidates a use's read leaves at one position where they differ
/// (l.337–344).
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Narrowed {
    /// The argument's index, the result's being the number of arguments.
    pub position: usize,
    pub value: ReadValue,
    /// Each candidate left before it, with the rank of the coercion its
    /// passing goes by there, none where no coercion reaches it.
    pub ranks: Vec<(DeclId, Option<Rank>)>,
    /// Those of the lowest rank, which remain.
    pub remaining: Vec<DeclId>,
}

/// How a use's candidates are narrowed by the value it reads (l.335–344).
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Narrowing {
    pub candidates: Vec<DeclId>,
    /// The class at each position where the candidates differ, by argument
    /// index with the result last, and the value read there.
    pub values: Vec<(usize, ClassId, ReadValue)>,
    /// The narrowing at each of those positions in turn, none where an
    /// argument reads as `!`.
    pub stages: Vec<Narrowed>,
    /// The narrowing again with each LitVar taken as its family's default,
    /// where several candidates remained and a value read holds a LitVar.
    pub retry: Option<Vec<Narrowed>>,
    /// The candidate it is decided as, or `!` where an argument reads as `!`;
    /// none where not exactly one remains (l.344).
    pub decision: Option<Decision>,
}

impl State {
    /// The candidates of a use: the declarations its name reaches whose
    /// parameters are as many as its arguments (l.331).
    pub(crate) fn candidates(&self, callee: &Name, arity: usize) -> Vec<&Decl> {
        self.decls
            .candidates(callee)
            .iter()
            .filter(|decl| decl.sig.params().len() == arity)
            .collect()
    }

    /// A position of a candidate's signature, by argument index with the
    /// result last.
    pub(super) fn at_position(sig: &Signature, i: usize) -> &SigTy {
        sig.params().get(i).unwrap_or(sig.result())
    }

    /// The positions where a use's candidates differ, which it reads (l.267);
    /// none where it has no candidate.
    pub(crate) fn differing(candidates: &[&Signature], args: usize) -> Vec<usize> {
        let Some(first) = candidates.first() else {
            return Vec::new();
        };
        (0..=args)
            .filter(|&i| {
                let at = Self::at_position(first, i);
                candidates.iter().any(|sig| Self::at_position(sig, i) != at)
            })
            .collect()
    }

    /// An argument passing into its parameter: a meeting where the parameter
    /// is an unknown, a boundary where a declaration writes it (reading 3).
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
            Relation::Boundary {
                from: arg,
                to: param,
            }
        };
        recorded.push(self.record(relation, recorder.clone()));
        recorded
    }

    /// The signature's result type entering the use's result (note 07,
    /// reading 2). Returns the relations recorded.
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

    /// Narrows the candidates by the value the use reads at each position
    /// where they differ, then by the result (l.337–344).
    pub(super) fn narrow(
        &self,
        callee: &Name,
        args: &[ClassId],
        result: ClassId,
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
                let class = args.get(i).copied().unwrap_or(result);
                (i, class, read.read_value(class))
            })
            .collect();
        let narrowing = |stages, retry, decision| Narrowing {
            candidates: ids.clone(),
            values: read_at.clone(),
            stages,
            retry,
            decision,
        };
        let values: Vec<(usize, ReadValue)> = read_at
            .iter()
            .map(|(i, _, value)| (*i, value.clone()))
            .collect();
        let never = values
            .iter()
            .any(|(i, value)| *i < args.len() && matches!(value.decided(), Some(Term::Never)));
        if never {
            return narrowing(Vec::new(), None, Some(Decision::Never));
        }
        let narrowed = |values: &[(usize, ReadValue)]| {
            let mut remaining: Vec<usize> = (0..sigs.len()).collect();
            let mut stages = Vec::new();
            for (i, value) in values {
                let ranks: Vec<(usize, Option<Rank>)> = remaining
                    .iter()
                    .map(|&c| {
                        let written = ReadValue::of_sig(Self::at_position(sigs[c], *i));
                        let rank = if *i < args.len() {
                            coerce_read(&self.decls, value, &written)
                        } else {
                            coerce_read(&self.decls, &written, value)
                        };
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
        let (mut remaining, stages) = narrowed(&values);
        let mut retry = None;
        if remaining.len() > 1 && values.iter().any(|(_, value)| value.has_lit()) {
            let defaulted: Vec<(usize, ReadValue)> = values
                .iter()
                .map(|(i, value)| (*i, value.with_defaults()))
                .collect();
            let (again, stages) = narrowed(&defaulted);
            remaining = again;
            retry = Some(stages);
        }
        let decision = match remaining.as_slice() {
            [one] => Some(Decision::Use(ids[*one].clone())),
            _ => None,
        };
        narrowing(stages, retry, decision)
    }
}
