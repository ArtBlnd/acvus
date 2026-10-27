//! A use's candidates, the arguments where they differ, and what a decided
//! candidate's signature records.

use std::collections::BTreeMap;

use crate::class::ClassId;
use crate::decl::Decl;
use crate::record::{Making, Recording};
use crate::relation::{Recorder, Relation};
use crate::sig::{SigTy, Signature};
use crate::solve::State;
use crate::ty::Name;

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

    /// The arguments where a use's candidates differ: a type parameter is an
    /// unknown made anew at the use, so two candidates that differ there only
    /// in their type parameters are alike there. None where it has no
    /// candidate.
    pub(crate) fn differing(candidates: &[&Signature], args: usize) -> Vec<usize> {
        let Some(first) = candidates.first() else {
            return Vec::new();
        };
        (0..args)
            .filter(|&i| {
                candidates
                    .iter()
                    .any(|sig| !alike(&sig.params()[i], &first.params()[i]))
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
        making: Making,
        var: &F,
        recorder: &Recorder,
    ) -> Recording
    where
        F: Fn(crate::sig::SigVar) -> Making,
    {
        let mut recorded = Recording::default();
        let param = self.instantiate(param, vars, making, var, recorder, &mut recorded);
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
        making: Making,
        var: &F,
        recorder: &Recorder,
    ) -> Recording
    where
        F: Fn(crate::sig::SigVar) -> Making,
    {
        let mut recorded = Recording::default();
        let returned = self.instantiate(ty, vars, making, var, recorder, &mut recorded);
        let relation = Relation::LowerBound {
            lower: returned,
            upper: result,
        };
        recorded.push(self.record(relation, recorder.clone()));
        recorded
    }
}

/// Whether two candidates' types write one term at every position, a
/// variable alike any other.
fn alike(a: &SigTy, b: &SigTy) -> bool {
    match (a, b) {
        (SigTy::Var(_), SigTy::Var(_)) => true,
        (SigTy::Term(s), SigTy::Term(t)) => {
            s.constructor() == t.constructor()
                && s.members() == t.members()
                && s.parts()
                    .into_iter()
                    .zip(t.parts())
                    .all(|((_, x), (_, y))| alike(x, y))
        }
        (SigTy::Var(_), SigTy::Term(_)) | (SigTy::Term(_), SigTy::Var(_)) => false,
    }
}
