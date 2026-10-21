//! Overload resolution (l.329–344): a use's candidates, the positions where
//! they differ, the narrowing, and what a decided candidate's signature
//! records.

use std::collections::{BTreeMap, BTreeSet};

use crate::class::{ClassId, ClassName};
use crate::coerce::coerce_read;
use crate::decl::{Decl, DeclId};
use crate::read::ReadValue;
use crate::relation::{Recorder, Relation};
use crate::sig::{SigTy, Signature};
use crate::solve::State;
use crate::ty::{HasTerm, Name, Term};

use super::Decision;

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

    /// The positions where a use's candidates differ, which it reads (l.267).
    pub(crate) fn differing(candidates: &[&Signature], args: usize) -> Vec<usize> {
        (0..=args)
            .filter(|&i| {
                let first = Self::at_position(candidates[0], i);
                candidates
                    .iter()
                    .any(|sig| Self::at_position(sig, i) != first)
            })
            .collect()
    }

    /// An argument passing into its parameter: a meeting where the parameter
    /// is an unknown, a boundary where a declaration writes it (reading 3).
    pub(super) fn record_argument<F>(
        &mut self,
        arg: ClassId,
        param: &SigTy,
        vars: &mut BTreeMap<crate::sig::SigVar, ClassId>,
        name: ClassName,
        var_name: &F,
        recorder: &Recorder,
    ) where
        F: Fn(crate::sig::SigVar) -> ClassName,
    {
        let param = self.instantiate(param, vars, name, var_name, recorder);
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
        self.record(relation, recorder.clone());
    }

    /// The signature's result type entering the use's result (note 07,
    /// reading 2).
    pub(super) fn record_result<F>(
        &mut self,
        result: ClassId,
        ty: &SigTy,
        vars: &mut BTreeMap<crate::sig::SigVar, ClassId>,
        name: ClassName,
        var_name: &F,
        recorder: &Recorder,
    ) where
        F: Fn(crate::sig::SigVar) -> ClassName,
    {
        let returned = self.instantiate(ty, vars, name, var_name, recorder);
        self.record(
            Relation::LowerBound {
                lower: returned,
                upper: result,
            },
            recorder.clone(),
        );
    }

    /// Narrows the candidates by the value the use reads at each position
    /// where they differ, then by the result (l.337–344).
    pub(super) fn narrow(
        &self,
        callee: &Name,
        args: &[ClassId],
        result: ClassId,
        excluded: &BTreeSet<Name>,
    ) -> Decision {
        let candidates = self.candidates(callee, args.len());
        let ids: Vec<DeclId> = candidates.iter().map(|decl| decl.id.clone()).collect();
        let sigs: Vec<&Signature> = candidates.iter().map(|decl| &decl.sig).collect();
        let differing = Self::differing(&sigs, args.len());
        let read = self.read_state(excluded);
        let values: Vec<(usize, ReadValue)> = differing
            .iter()
            .map(|&i| (i, read.read_value(args.get(i).copied().unwrap_or(result))))
            .collect();
        let never = values
            .iter()
            .any(|(i, value)| *i < args.len() && matches!(value.decided(), Some(Term::Never)));
        if never {
            return Decision::Never;
        }
        let narrowed = |values: &[(usize, ReadValue)]| {
            let mut remaining: Vec<usize> = (0..sigs.len()).collect();
            for (i, value) in values {
                let ranks: Vec<(usize, Option<crate::coerce::Rank>)> = remaining
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
                    .into_iter()
                    .filter(|(_, rank)| rank.is_some() && *rank == lowest)
                    .map(|(c, _)| c)
                    .collect();
            }
            remaining
        };
        let mut remaining = narrowed(&values);
        if remaining.len() > 1 && values.iter().any(|(_, value)| value.has_lit()) {
            let defaulted: Vec<(usize, ReadValue)> = values
                .iter()
                .map(|(i, value)| (*i, value.with_defaults()))
                .collect();
            remaining = narrowed(&defaulted);
        }
        match remaining.as_slice() {
            [one] => Decision::Use(ids[*one].clone()),
            _ => Decision::Unreached(remaining.into_iter().map(|c| ids[c].clone()).collect()),
        }
    }
}
