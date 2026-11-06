//! A use's candidates, and what a signature records at a use: each argument
//! passing into its parameter's type, and the result type entering the
//! result.

use std::collections::BTreeMap;

use crate::class::ClassId;
use crate::decl::Decl;
use crate::record::{Making, Recording};
use crate::relation::{Recorder, Relation};
use crate::sig::SigTy;
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

    /// An argument passing into its parameter: a lower bound where the parameter
    /// is an unknown, a coercion constraint where a declaration writes it.
    /// Returns the relations recorded.
    pub(crate) fn record_argument<F>(
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
    pub(crate) fn record_result<F>(
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
